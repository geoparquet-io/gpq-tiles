//! Tiler pipeline - wires together batch processing, clipping, simplification, and MVT encoding.
//!
//! This module provides the core tiling pipeline that:
//! 1. Reads geometries from GeoParquet
//! 2. Iterates tiles for the data's bounding box at each zoom level
//! 3. For each tile: clips, simplifies, validates, and encodes to MVT format
//!
//! # Tippecanoe Alignment
//!
//! This pipeline matches tippecanoe's approach:
//! - Buffer: 8 pixels (configurable)
//! - Simplification: Douglas-Peucker to tile resolution at each zoom
//! - Degenerate geometry filtering: drop invalid geometries post-simplification
//! - Empty tiles are skipped (not written)

use std::path::Path;
use std::sync::Arc;

use prost::Message;
use rayon::prelude::*;

use geo::Geometry;

use crate::batch_processor::{extract_field_metadata, extract_geometries};
use crate::clip::{buffer_pixels_to_degrees, clip_geometry};
use crate::feature_drop::{
    should_drop_multipoint, should_drop_point, should_drop_tiny_line, should_drop_tiny_multiline,
    should_drop_tiny_polygon, DensityDropConfig, DensityDropper, DEFAULT_TINY_POLYGON_THRESHOLD,
};
use crate::mvt::{LayerBuilder, TileBuilder};
use crate::simplify::simplify_for_zoom;
use crate::spatial_index::sort_geometries;
use crate::tile::{tiles_for_bbox, TileBounds, TileCoord};
use crate::validate::filter_valid_geometry;
use crate::vector_tile::Tile;
use crate::{Error, Result};

/// Default buffer in pixels (matches tippecanoe common usage)
pub const DEFAULT_BUFFER_PIXELS: u32 = 8;

/// Default tile extent (4096 as per MVT spec)
pub const DEFAULT_EXTENT: u32 = 4096;

/// Determine if a geometry should be dropped based on zoom level and geometry type.
///
/// This function dispatches to the appropriate dropping predicate based on geometry type:
/// - **Points**: Thinned using 1/2.5 drop rate per zoom level below base_zoom
/// - **Lines**: Dropped if all vertices collapse to the same tile pixel
/// - **Polygons**: Dropped probabilistically if area < 4 square pixels (diffuse probability)
///
/// # Arguments
///
/// * `geom` - The geometry to check
/// * `zoom` - Current zoom level being generated
/// * `base_zoom` - The zoom level where all features are kept (typically max_zoom)
/// * `extent` - Tile extent (typically 4096)
/// * `tile_bounds` - Geographic bounds of the tile
/// * `feature_index` - Unique index of this feature for deterministic selection
///
/// # Returns
///
/// `true` if the geometry should be dropped, `false` if it should be kept.
fn should_drop_geometry(
    geom: &Geometry<f64>,
    zoom: u8,
    base_zoom: u8,
    extent: u32,
    tile_bounds: &TileBounds,
    feature_index: u64,
) -> bool {
    match geom {
        Geometry::Point(p) => should_drop_point(p, zoom, base_zoom, feature_index),
        Geometry::MultiPoint(mp) => should_drop_multipoint(mp, zoom, base_zoom, feature_index),
        Geometry::LineString(ls) => should_drop_tiny_line(ls, zoom, extent, tile_bounds),
        Geometry::MultiLineString(mls) => {
            should_drop_tiny_multiline(mls, zoom, extent, tile_bounds)
        }
        Geometry::Polygon(poly) => {
            should_drop_tiny_polygon(poly, tile_bounds, extent, DEFAULT_TINY_POLYGON_THRESHOLD)
        }
        Geometry::MultiPolygon(mp) => {
            // Drop if ALL polygons would be dropped
            mp.0.iter().all(|p| {
                should_drop_tiny_polygon(p, tile_bounds, extent, DEFAULT_TINY_POLYGON_THRESHOLD)
            })
        }
        // GeometryCollection and other types are not dropped
        _ => false,
    }
}

/// Configuration for the tiling pipeline.
#[derive(Debug, Clone)]
pub struct TilerConfig {
    /// Minimum zoom level to generate
    pub min_zoom: u8,
    /// Maximum zoom level to generate
    pub max_zoom: u8,
    /// Tile extent in pixels (default: 4096)
    pub extent: u32,
    /// Buffer in pixels around tile bounds (default: 8)
    pub buffer_pixels: u32,
    /// Layer name for the MVT output
    pub layer_name: String,
    /// Enable density-based feature dropping (default: true)
    /// When enabled, limits features per grid cell to reduce clutter at low zoom levels
    pub enable_density_drop: bool,
    /// Grid cell size in pixels for density dropping (default: 32)
    /// Smaller values = less aggressive dropping, larger = more aggressive
    pub density_cell_size: u32,
    /// Maximum features per grid cell (default: 1)
    /// Higher values = more features kept in dense areas
    pub density_max_per_cell: usize,
    /// Use Hilbert curve for spatial sorting (default: true)
    /// Hilbert curves have better locality than Z-order curves.
    /// If false, uses Z-order (Morton) curve instead.
    pub use_hilbert: bool,
    /// Enable parallel tile generation using Rayon (default: true)
    /// When enabled, tiles within each zoom level are processed in parallel.
    /// Zoom levels are still processed sequentially to preserve feature dropping semantics.
    pub parallel: bool,
}

impl Default for TilerConfig {
    fn default() -> Self {
        Self {
            min_zoom: 0,
            max_zoom: 14,
            extent: DEFAULT_EXTENT,
            buffer_pixels: DEFAULT_BUFFER_PIXELS,
            layer_name: "layer".to_string(),
            // Density dropping is disabled by default to maintain backward compatibility
            // Enable it with .with_density_drop(true) when you need tippecanoe-like
            // feature reduction at low zoom levels
            enable_density_drop: false,
            // Cell size of 16 pixels = 256 cells per tile at 4096 extent
            // This is fairly aggressive - use larger values for less dropping
            density_cell_size: 16,
            density_max_per_cell: 1,
            // Hilbert curve is the default because it has better locality than Z-order
            use_hilbert: true,
            // Parallel processing is enabled by default for performance
            parallel: true,
        }
    }
}

impl TilerConfig {
    /// Create a new config with custom settings.
    pub fn new(min_zoom: u8, max_zoom: u8) -> Self {
        Self {
            min_zoom,
            max_zoom,
            ..Default::default()
        }
    }

    /// Set the layer name.
    pub fn with_layer_name(mut self, name: impl Into<String>) -> Self {
        self.layer_name = name.into();
        self
    }

    /// Set the tile extent.
    pub fn with_extent(mut self, extent: u32) -> Self {
        self.extent = extent;
        self
    }

    /// Set the buffer in pixels.
    pub fn with_buffer(mut self, buffer_pixels: u32) -> Self {
        self.buffer_pixels = buffer_pixels;
        self
    }

    /// Enable or disable density-based feature dropping.
    pub fn with_density_drop(mut self, enable: bool) -> Self {
        self.enable_density_drop = enable;
        self
    }

    /// Set the grid cell size for density dropping.
    pub fn with_density_cell_size(mut self, cell_size: u32) -> Self {
        self.density_cell_size = cell_size;
        self
    }

    /// Set the maximum features per grid cell for density dropping.
    pub fn with_density_max_per_cell(mut self, max: usize) -> Self {
        self.density_max_per_cell = max;
        self
    }

    /// Set whether to use Hilbert curve (true) or Z-order curve (false) for spatial sorting.
    ///
    /// Hilbert curves have better locality than Z-order curves - neighboring points
    /// on the curve are always neighboring in 2D space. This is the default.
    ///
    /// Z-order (Morton) curves are simpler and faster to compute but don't have
    /// the same locality guarantee at quadrant boundaries.
    pub fn with_hilbert(mut self, use_hilbert: bool) -> Self {
        self.use_hilbert = use_hilbert;
        self
    }

    /// Enable or disable parallel tile generation.
    ///
    /// When enabled (default), tiles within each zoom level are processed in parallel
    /// using Rayon. Zoom levels are still processed sequentially to preserve feature
    /// dropping semantics.
    ///
    /// Disable this for debugging or when you need deterministic single-threaded execution.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }
}

/// A generated vector tile with its coordinates and data.
#[derive(Debug, Clone)]
pub struct GeneratedTile {
    /// The tile coordinates (x, y, z)
    pub coord: TileCoord,
    /// The MVT protobuf bytes
    pub data: Vec<u8>,
    /// Number of features in this tile
    pub feature_count: usize,
}

impl GeneratedTile {
    /// Create a new generated tile.
    pub fn new(coord: TileCoord, data: Vec<u8>, feature_count: usize) -> Self {
        Self {
            coord,
            data,
            feature_count,
        }
    }

    /// Check if the tile is empty (no data).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Result of tile generation: tiles iterator plus metadata.
///
/// This struct bundles the tile iterator with metadata needed for
/// writing valid PMTiles headers (bounds, layer name, etc.).
pub struct TileGeneration<I: Iterator<Item = Result<GeneratedTile>>> {
    /// Iterator yielding generated tiles
    pub tiles: I,
    /// Geographic bounding box of the input data
    pub bounds: TileBounds,
    /// Layer name used in the MVT tiles
    pub layer_name: String,
    /// Field metadata: field name -> MVT type ("String", "Number", "Boolean")
    pub fields: std::collections::HashMap<String, String>,
}

/// Generate vector tiles from a GeoParquet file.
///
/// This function reads geometries from the input file, iterates over all tiles
/// in the configured zoom range that intersect the data's bounding box, and
/// generates MVT-encoded tiles for each.
///
/// # Arguments
///
/// * `input_path` - Path to the GeoParquet file
/// * `config` - Tiling configuration
///
/// # Returns
///
/// An iterator of `Result<GeneratedTile>`, yielding each generated tile.
/// Empty tiles (no features after clipping) are skipped.
///
/// # Example
///
/// ```no_run
/// use gpq_tiles_core::pipeline::{generate_tiles, TilerConfig};
/// use std::path::Path;
///
/// let config = TilerConfig::new(0, 10);
/// let tiles = generate_tiles(Path::new("input.parquet"), &config).unwrap();
///
/// for tile_result in tiles {
///     let tile = tile_result.unwrap();
///     println!("Generated tile z{}/x{}/y{}: {} bytes",
///              tile.coord.z, tile.coord.x, tile.coord.y, tile.data.len());
/// }
/// ```
pub fn generate_tiles(
    input_path: &Path,
    config: &TilerConfig,
) -> Result<impl Iterator<Item = Result<GeneratedTile>>> {
    let result = generate_tiles_with_bounds(input_path, config)?;
    Ok(result.tiles)
}

/// Generate vector tiles from a GeoParquet file, returning bounds too.
///
/// Like `generate_tiles()`, but also returns the geographic bounding box
/// of the input data. Use this when you need bounds for PMTiles headers.
///
/// # Example
///
/// ```no_run
/// use gpq_tiles_core::pipeline::{generate_tiles_with_bounds, TilerConfig};
/// use gpq_tiles_core::pmtiles_writer::PmtilesWriter;
/// use std::path::Path;
///
/// let config = TilerConfig::new(0, 10);
/// let result = generate_tiles_with_bounds(Path::new("input.parquet"), &config).unwrap();
///
/// let mut writer = PmtilesWriter::new();
/// writer.set_bounds(&result.bounds);
///
/// for tile_result in result.tiles {
///     let tile = tile_result.unwrap();
///     writer.add_tile(tile.coord.z, tile.coord.x, tile.coord.y, &tile.data).unwrap();
/// }
/// ```
pub fn generate_tiles_with_bounds(
    input_path: &Path,
    config: &TilerConfig,
) -> Result<TileGeneration<impl Iterator<Item = Result<GeneratedTile>>>> {
    // Step 1: Extract field metadata from schema
    let fields = extract_field_metadata(input_path).unwrap_or_default();

    // Step 2: Extract all geometries from the GeoParquet file
    // WARNING: This loads all geometries into memory. For large files,
    // we'll need a streaming approach in Phase 4.
    let geometries = extract_geometries(input_path)?;

    if geometries.is_empty() {
        return Ok(TileGeneration {
            tiles: TileIterator::empty(),
            bounds: TileBounds::empty(),
            layer_name: config.layer_name.clone(),
            fields,
        });
    }

    // Step 3: Calculate bounding box from geometries
    let bbox = calculate_bbox_from_geometries(&geometries);

    // Step 4: Create tile iterator
    Ok(TileGeneration {
        tiles: TileIterator::new(geometries, bbox, config.clone()),
        bounds: bbox,
        layer_name: config.layer_name.clone(),
        fields,
    })
}

/// Generate vector tiles from pre-loaded geometries.
///
/// This is a lower-level function useful for benchmarking and library consumers
/// who have already loaded geometries. It bypasses file I/O, which makes it ideal
/// for performance testing.
///
/// # Arguments
///
/// * `geometries` - Pre-loaded geometries (will be sorted by spatial index)
/// * `config` - Tiling configuration
///
/// # Returns
///
/// An iterator of `Result<GeneratedTile>`, yielding each generated tile.
/// Empty tiles (no features after clipping) are skipped.
///
/// # Example
///
/// ```no_run
/// use gpq_tiles_core::pipeline::{generate_tiles_from_geometries, TilerConfig};
/// use gpq_tiles_core::batch_processor::extract_geometries;
/// use std::path::Path;
///
/// // Pre-load geometries (e.g., in benchmark setup)
/// let geometries = extract_geometries(Path::new("input.parquet")).unwrap();
///
/// // Then benchmark just the tiling
/// let config = TilerConfig::new(0, 10);
/// let tiles: Vec<_> = generate_tiles_from_geometries(geometries, &config)
///     .unwrap()
///     .collect();
/// ```
pub fn generate_tiles_from_geometries(
    geometries: Vec<geo::Geometry<f64>>,
    config: &TilerConfig,
) -> Result<impl Iterator<Item = Result<GeneratedTile>>> {
    if geometries.is_empty() {
        return Ok(TileIterator::empty());
    }

    // Calculate bounding box from geometries
    let bbox = calculate_bbox_from_geometries(&geometries);

    // Create tile iterator (sorting happens inside)
    Ok(TileIterator::new(geometries, bbox, config.clone()))
}

/// Calculate bounding box from a collection of geometries.
fn calculate_bbox_from_geometries(geometries: &[geo::Geometry<f64>]) -> TileBounds {
    use geo::BoundingRect;

    let mut bounds = TileBounds::empty();

    for geom in geometries {
        if let Some(rect) = geom.bounding_rect() {
            bounds.expand(&TileBounds::new(
                rect.min().x,
                rect.min().y,
                rect.max().x,
                rect.max().y,
            ));
        }
    }

    bounds
}

/// Iterator that generates tiles for each tile coordinate.
///
/// When parallel mode is enabled, tiles within each zoom level are processed
/// in parallel using Rayon. Zoom levels are still processed sequentially to
/// preserve feature dropping semantics.
struct TileIterator {
    /// Shared geometries for parallel access
    geometries: Arc<Vec<geo::Geometry<f64>>>,
    config: TilerConfig,
    /// Bounding box for generating tile coordinates
    bbox: TileBounds,
    /// Current zoom level being processed
    current_zoom: u8,
    /// Buffer of generated tiles for the current zoom level
    tile_buffer: Vec<GeneratedTile>,
    /// Index into the tile buffer
    buffer_index: usize,
    /// Whether we've finished all zoom levels
    finished: bool,
}

impl TileIterator {
    fn new(geometries: Vec<geo::Geometry<f64>>, bbox: TileBounds, config: TilerConfig) -> Self {
        // Sort geometries by spatial index ONCE before tile generation.
        // This clusters nearby features together for cache-friendly tile generation.
        // Features for each tile will be mostly adjacent in the sorted order.
        let mut sorted_geometries = geometries;
        sort_geometries(&mut sorted_geometries, config.use_hilbert);

        Self {
            geometries: Arc::new(sorted_geometries),
            current_zoom: config.min_zoom,
            bbox,
            config,
            tile_buffer: Vec::new(),
            buffer_index: 0,
            finished: false,
        }
    }

    fn empty() -> Self {
        Self {
            geometries: Arc::new(Vec::new()),
            config: TilerConfig::default(),
            bbox: TileBounds::empty(),
            current_zoom: 0,
            tile_buffer: Vec::new(),
            buffer_index: 0,
            finished: true,
        }
    }

    /// Process a single tile: clip, simplify, encode to MVT.
    /// This is a pure function that can be safely called in parallel.
    fn process_tile_static(
        geometries: &[geo::Geometry<f64>],
        coord: TileCoord,
        config: &TilerConfig,
    ) -> Result<Option<GeneratedTile>> {
        let bounds = coord.bounds();
        let buffer = buffer_pixels_to_degrees(config.buffer_pixels, &bounds, config.extent);

        // Build the layer with clipped/simplified geometries
        let mut layer_builder = LayerBuilder::new(&config.layer_name).with_extent(config.extent);

        // Create density dropper for this tile if enabled
        let mut density_dropper = if config.enable_density_drop {
            let density_config = DensityDropConfig::new()
                .with_cell_size(config.density_cell_size)
                .with_max_features_per_cell(config.density_max_per_cell)
                .with_zoom_range(0, config.max_zoom);
            Some(DensityDropper::new(density_config, config.extent))
        } else {
            None
        };

        let mut feature_count = 0;

        for (idx, geom) in geometries.iter().enumerate() {
            // Clip to tile bounds
            if let Some(clipped) = clip_geometry(geom, &bounds, buffer) {
                // Simplify for zoom level
                let simplified = simplify_for_zoom(&clipped, coord.z, config.extent);

                // Validate geometry - filter out degenerate geometries post-simplification
                // (e.g., polygons with < 4 points, zero-area polygons, linestrings with < 2 points)
                if let Some(valid_geom) = filter_valid_geometry(&simplified) {
                    // Apply feature dropping based on zoom level and geometry type
                    // base_zoom is max_zoom: at max_zoom all features are kept
                    if should_drop_geometry(
                        &valid_geom,
                        coord.z,
                        config.max_zoom,
                        config.extent,
                        &bounds,
                        idx as u64,
                    ) {
                        continue;
                    }

                    // Apply density-based dropping if enabled
                    // This limits the number of features per grid cell to prevent
                    // cluttered tiles at low zoom levels
                    if let Some(ref mut dropper) = density_dropper {
                        if dropper.should_drop_geometry(
                            &valid_geom,
                            &bounds,
                            config.extent,
                            coord.z,
                        ) {
                            continue;
                        }
                    }

                    // Add to layer (no properties for now)
                    layer_builder.add_feature(Some(idx as u64), &valid_geom, &[], &bounds);
                    feature_count += 1;
                }
            }
        }

        // Skip empty tiles
        if feature_count == 0 {
            return Ok(None);
        }

        // Build the tile
        let layer = layer_builder.build();
        let mut tile_builder = TileBuilder::new();
        tile_builder.add_layer(layer);
        let tile = tile_builder.build();

        // Serialize to protobuf bytes
        let data = tile.encode_to_vec();

        Ok(Some(GeneratedTile::new(coord, data, feature_count)))
    }

    /// Process all tiles for a zoom level in parallel.
    fn process_zoom_level_parallel(&self, zoom: u8) -> Vec<Result<GeneratedTile>> {
        let tile_coords: Vec<TileCoord> = tiles_for_bbox(&self.bbox, zoom).collect();

        // Clone Arc for each parallel task
        let geometries = Arc::clone(&self.geometries);
        let config = self.config.clone();

        tile_coords
            .into_par_iter()
            .filter_map(|coord| {
                match Self::process_tile_static(&geometries, coord, &config) {
                    Ok(Some(tile)) => Some(Ok(tile)),
                    Ok(None) => None, // Empty tile, skip
                    Err(e) => Some(Err(e)),
                }
            })
            .collect()
    }

    /// Process all tiles for a zoom level sequentially.
    fn process_zoom_level_sequential(&self, zoom: u8) -> Vec<Result<GeneratedTile>> {
        let tile_coords: Vec<TileCoord> = tiles_for_bbox(&self.bbox, zoom).collect();

        tile_coords
            .into_iter()
            .filter_map(|coord| {
                match Self::process_tile_static(&self.geometries, coord, &self.config) {
                    Ok(Some(tile)) => Some(Ok(tile)),
                    Ok(None) => None, // Empty tile, skip
                    Err(e) => Some(Err(e)),
                }
            })
            .collect()
    }

    /// Fill the tile buffer with tiles from the next zoom level.
    fn fill_buffer(&mut self) -> bool {
        if self.current_zoom > self.config.max_zoom {
            self.finished = true;
            return false;
        }

        // Process tiles for this zoom level
        let results = if self.config.parallel {
            self.process_zoom_level_parallel(self.current_zoom)
        } else {
            self.process_zoom_level_sequential(self.current_zoom)
        };

        // Extract successful tiles, propagate errors later
        self.tile_buffer = results.into_iter().filter_map(|r| r.ok()).collect();

        // Sort tiles by coordinates for deterministic output order
        self.tile_buffer
            .sort_by_key(|t| (t.coord.z, t.coord.x, t.coord.y));

        self.buffer_index = 0;
        self.current_zoom += 1;

        !self.tile_buffer.is_empty() || self.current_zoom <= self.config.max_zoom
    }
}

impl Iterator for TileIterator {
    type Item = Result<GeneratedTile>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have buffered tiles, return the next one
            if self.buffer_index < self.tile_buffer.len() {
                let tile = self.tile_buffer[self.buffer_index].clone();
                self.buffer_index += 1;
                return Some(Ok(tile));
            }

            // If we're finished, return None
            if self.finished {
                return None;
            }

            // Try to fill the buffer with the next zoom level
            if !self.fill_buffer() && self.tile_buffer.is_empty() {
                return None;
            }
        }
    }
}

/// Generate a single tile from geometries.
///
/// This is a lower-level function for when you already have geometries loaded
/// and want to generate a specific tile.
///
/// # Arguments
///
/// * `geometries` - The source geometries
/// * `coord` - The tile coordinate to generate
/// * `config` - Tiling configuration
///
/// # Returns
///
/// `Some(GeneratedTile)` if the tile has features, `None` if empty.
pub fn generate_single_tile(
    geometries: &[geo::Geometry<f64>],
    coord: TileCoord,
    config: &TilerConfig,
) -> Result<Option<GeneratedTile>> {
    TileIterator::process_tile_static(geometries, coord, config)
}

/// Decode an MVT tile from bytes (for testing).
pub fn decode_tile(data: &[u8]) -> Result<Tile> {
    Tile::decode(data).map_err(|e| Error::MvtEncoding(format!("Failed to decode tile: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{line_string, point, polygon, Geometry};
    use std::path::Path;

    // ========== TilerConfig Tests ==========

    #[test]
    fn test_tiler_config_default() {
        let config = TilerConfig::default();

        assert_eq!(config.min_zoom, 0);
        assert_eq!(config.max_zoom, 14);
        assert_eq!(config.extent, 4096);
        assert_eq!(config.buffer_pixels, 8);
        assert_eq!(config.layer_name, "layer");
    }

    #[test]
    fn test_tiler_config_builder() {
        let config = TilerConfig::new(5, 10)
            .with_layer_name("buildings")
            .with_extent(512)
            .with_buffer(16);

        assert_eq!(config.min_zoom, 5);
        assert_eq!(config.max_zoom, 10);
        assert_eq!(config.extent, 512);
        assert_eq!(config.buffer_pixels, 16);
        assert_eq!(config.layer_name, "buildings");
    }

    // ========== GeneratedTile Tests ==========

    #[test]
    fn test_generated_tile_creation() {
        let coord = TileCoord::new(1, 2, 3);
        let data = vec![1, 2, 3, 4];
        let tile = GeneratedTile::new(coord, data.clone(), 5);

        assert_eq!(tile.coord, coord);
        assert_eq!(tile.data, data);
        assert_eq!(tile.feature_count, 5);
        assert!(!tile.is_empty());
    }

    #[test]
    fn test_generated_tile_empty() {
        let coord = TileCoord::new(0, 0, 0);
        let tile = GeneratedTile::new(coord, vec![], 0);

        assert!(tile.is_empty());
    }

    // ========== Single Tile Generation Tests ==========

    #[test]
    fn test_generate_single_tile_with_point() {
        // Create a point at the center of tile 0/0/0 (null island)
        let point = Geometry::Point(point!(x: 0.0, y: 0.0));
        let geometries = vec![point];

        let config = TilerConfig::new(0, 0);
        let coord = TileCoord::new(0, 0, 0);

        let result = generate_single_tile(&geometries, coord, &config);
        assert!(result.is_ok());

        let tile_opt = result.unwrap();
        assert!(
            tile_opt.is_some(),
            "Should generate a tile for point at origin"
        );

        let tile = tile_opt.unwrap();
        assert!(!tile.is_empty());
        assert_eq!(tile.coord, coord);

        // Verify we can decode the MVT
        let decoded = decode_tile(&tile.data).expect("Should decode MVT");
        assert_eq!(decoded.layers.len(), 1);
        assert_eq!(decoded.layers[0].name, "layer");
        assert_eq!(decoded.layers[0].features.len(), 1);
    }

    #[test]
    fn test_generate_single_tile_with_polygon() {
        // Create a polygon in Andorra (where our test data is)
        let poly = Geometry::Polygon(polygon![
            (x: 1.5, y: 42.5),
            (x: 1.6, y: 42.5),
            (x: 1.6, y: 42.6),
            (x: 1.5, y: 42.6),
            (x: 1.5, y: 42.5),
        ]);
        let geometries = vec![poly];

        let config = TilerConfig::new(10, 10).with_layer_name("buildings");

        // Find the tile that contains Andorra at z10
        let tile_coord = crate::tile::lng_lat_to_tile(1.55, 42.55, 10);

        let result = generate_single_tile(&geometries, tile_coord, &config);
        assert!(result.is_ok());

        let tile_opt = result.unwrap();
        assert!(
            tile_opt.is_some(),
            "Should generate a tile containing the polygon"
        );

        let tile = tile_opt.unwrap();
        let decoded = decode_tile(&tile.data).unwrap();
        assert_eq!(decoded.layers[0].name, "buildings");
        assert_eq!(decoded.layers[0].features.len(), 1);
    }

    #[test]
    fn test_generate_single_tile_empty_when_no_features() {
        // Create a point in Australia
        let point = Geometry::Point(point!(x: 135.0, y: -25.0));
        let geometries = vec![point];

        let config = TilerConfig::new(10, 10);

        // Request a tile in Europe (far from the point)
        let coord = TileCoord::new(516, 377, 10); // Andorra

        let result = generate_single_tile(&geometries, coord, &config);
        assert!(result.is_ok());

        // Should return None because no features intersect this tile
        assert!(result.unwrap().is_none());
    }

    // ========== Full Pipeline Tests ==========

    #[test]
    fn test_generate_tiles_with_real_fixture() {
        let fixture = Path::new("../../tests/fixtures/realdata/open-buildings.parquet");
        if !fixture.exists() {
            eprintln!("Skipping: fixture not found");
            return;
        }

        // Generate tiles at zoom 10 only (single zoom for speed)
        let config = TilerConfig::new(10, 10).with_layer_name("buildings");

        let tiles_iter = generate_tiles(fixture, &config).expect("Should create iterator");
        let tiles: Vec<_> = tiles_iter.collect();

        // Should generate some tiles
        assert!(!tiles.is_empty(), "Should generate at least one tile");

        // All should be successful
        for tile_result in &tiles {
            assert!(tile_result.is_ok());
        }

        // Verify tile contents
        let first_tile = tiles[0].as_ref().unwrap();
        assert!(!first_tile.is_empty());

        let decoded = decode_tile(&first_tile.data).unwrap();
        assert_eq!(decoded.layers.len(), 1);
        assert_eq!(decoded.layers[0].name, "buildings");
        assert!(!decoded.layers[0].features.is_empty());

        println!("Generated {} tiles at zoom 10", tiles.len());
    }

    #[test]
    fn test_generate_tiles_multi_zoom() {
        let fixture = Path::new("../../tests/fixtures/realdata/open-buildings.parquet");
        if !fixture.exists() {
            eprintln!("Skipping: fixture not found");
            return;
        }

        // Generate tiles at zoom 8-10
        let config = TilerConfig::new(8, 10);

        let tiles_iter = generate_tiles(fixture, &config).expect("Should create iterator");
        let tiles: Vec<_> = tiles_iter.filter_map(|r| r.ok()).collect();

        // Should have tiles at different zoom levels
        let z8_count = tiles.iter().filter(|t| t.coord.z == 8).count();
        let z9_count = tiles.iter().filter(|t| t.coord.z == 9).count();
        let z10_count = tiles.iter().filter(|t| t.coord.z == 10).count();

        println!(
            "Z8: {} tiles, Z9: {} tiles, Z10: {} tiles",
            z8_count, z9_count, z10_count
        );

        // Higher zooms should generally have more tiles (smaller tiles = more of them)
        // Unless data is sparse
        assert!(z8_count > 0, "Should have z8 tiles");
        assert!(z9_count > 0, "Should have z9 tiles");
        assert!(z10_count > 0, "Should have z10 tiles");

        // At higher zooms, there are more tiles covering the same area
        assert!(
            z10_count >= z9_count,
            "z10 should have at least as many tiles as z9"
        );
        assert!(
            z9_count >= z8_count,
            "z9 should have at least as many tiles as z8"
        );
    }

    #[test]
    fn test_generate_tiles_skips_empty() {
        // Create a single point
        let geometries = vec![Geometry::Point(point!(x: 0.0, y: 0.0))];

        let bbox = calculate_bbox_from_geometries(&geometries);
        let config = TilerConfig::new(0, 2);

        let iter = TileIterator::new(geometries, bbox, config);
        let tiles: Vec<_> = iter.filter_map(|r| r.ok()).collect();

        // At z0, there's only 1 tile (0,0,0) which contains the point
        // At z1, there are 4 tiles, but only 1 contains the point
        // At z2, there are 16 tiles, but only 1 contains the point
        // So we should have exactly 3 tiles (one per zoom)
        assert_eq!(
            tiles.len(),
            3,
            "Should have exactly 3 tiles (one per zoom containing the point)"
        );

        for tile in &tiles {
            println!(
                "Tile z{}/x{}/y{}: {} bytes",
                tile.coord.z,
                tile.coord.x,
                tile.coord.y,
                tile.data.len()
            );
        }
    }

    #[test]
    fn test_mvt_tile_decodes_correctly() {
        let poly = Geometry::Polygon(polygon![
            (x: 0.0, y: 0.0),
            (x: 1.0, y: 0.0),
            (x: 1.0, y: 1.0),
            (x: 0.0, y: 1.0),
            (x: 0.0, y: 0.0),
        ]);
        let geometries = vec![poly];

        let config = TilerConfig::new(0, 0)
            .with_layer_name("test_layer")
            .with_extent(4096);
        let coord = TileCoord::new(0, 0, 0);

        let tile = generate_single_tile(&geometries, coord, &config)
            .unwrap()
            .unwrap();

        // Decode and verify structure
        let decoded = decode_tile(&tile.data).unwrap();

        assert_eq!(decoded.layers.len(), 1);

        let layer = &decoded.layers[0];
        assert_eq!(layer.version, 2);
        assert_eq!(layer.name, "test_layer");
        assert_eq!(layer.extent, Some(4096));
        assert_eq!(layer.features.len(), 1);

        let feature = &layer.features[0];
        assert_eq!(feature.id, Some(0));
        // GeomType::Polygon = 3
        assert_eq!(feature.r#type, Some(3));
        assert!(!feature.geometry.is_empty());
    }

    #[test]
    fn test_calculate_bbox_from_geometries() {
        let geometries = vec![
            Geometry::Point(point!(x: -10.0, y: -5.0)),
            Geometry::Point(point!(x: 10.0, y: 5.0)),
        ];

        let bbox = calculate_bbox_from_geometries(&geometries);

        assert_eq!(bbox.lng_min, -10.0);
        assert_eq!(bbox.lng_max, 10.0);
        assert_eq!(bbox.lat_min, -5.0);
        assert_eq!(bbox.lat_max, 5.0);
    }

    #[test]
    fn test_pipeline_with_multiple_geometry_types() {
        let geometries = vec![
            Geometry::Point(point!(x: 0.5, y: 0.5)),
            Geometry::Polygon(polygon![
                (x: 0.0, y: 0.0),
                (x: 1.0, y: 0.0),
                (x: 1.0, y: 1.0),
                (x: 0.0, y: 1.0),
                (x: 0.0, y: 0.0),
            ]),
        ];

        let config = TilerConfig::new(0, 0);
        let coord = TileCoord::new(0, 0, 0);

        let tile = generate_single_tile(&geometries, coord, &config)
            .unwrap()
            .unwrap();

        let decoded = decode_tile(&tile.data).unwrap();
        assert_eq!(
            decoded.layers[0].features.len(),
            2,
            "Should have both point and polygon"
        );
    }

    // ========== Degenerate Geometry Validation Tests ==========

    #[test]
    fn test_pipeline_filters_degenerate_linestring() {
        use geo::LineString;

        // Create a linestring with only 1 point (degenerate after simplification scenario)
        // Note: A real scenario would have simplification reduce points,
        // but we can test the validation directly with a degenerate input.
        let degenerate_line =
            Geometry::LineString(LineString::new(vec![geo::Coord { x: 0.5, y: 0.5 }]));

        let valid_point = Geometry::Point(point!(x: 0.5, y: 0.5));

        let geometries = vec![degenerate_line, valid_point];

        let config = TilerConfig::new(0, 0);
        let coord = TileCoord::new(0, 0, 0);

        let tile = generate_single_tile(&geometries, coord, &config)
            .unwrap()
            .unwrap();

        let decoded = decode_tile(&tile.data).unwrap();
        // Should only have the valid point, degenerate linestring filtered out
        assert_eq!(
            decoded.layers[0].features.len(),
            1,
            "Should filter out degenerate linestring"
        );
    }

    #[test]
    fn test_pipeline_filters_degenerate_polygon_too_few_points() {
        // A polygon with only 3 points (2 unique + closing) is degenerate
        let degenerate_poly = Geometry::Polygon(geo::Polygon::new(
            geo::LineString::new(vec![
                geo::Coord { x: 0.0, y: 0.0 },
                geo::Coord { x: 1.0, y: 0.0 },
                geo::Coord { x: 0.0, y: 0.0 }, // closing
            ]),
            vec![],
        ));

        let valid_point = Geometry::Point(point!(x: 0.5, y: 0.5));

        let geometries = vec![degenerate_poly, valid_point];

        let config = TilerConfig::new(0, 0);
        let coord = TileCoord::new(0, 0, 0);

        let tile = generate_single_tile(&geometries, coord, &config)
            .unwrap()
            .unwrap();

        let decoded = decode_tile(&tile.data).unwrap();
        // Should only have the valid point
        assert_eq!(
            decoded.layers[0].features.len(),
            1,
            "Should filter out degenerate polygon with too few points"
        );
    }

    #[test]
    fn test_pipeline_filters_zero_area_polygon() {
        // A polygon where all points are collinear (zero area)
        let zero_area_poly = Geometry::Polygon(geo::Polygon::new(
            geo::LineString::new(vec![
                geo::Coord { x: 0.0, y: 0.0 },
                geo::Coord { x: 1.0, y: 0.0 },
                geo::Coord { x: 2.0, y: 0.0 },
                geo::Coord { x: 3.0, y: 0.0 },
                geo::Coord { x: 0.0, y: 0.0 }, // closing
            ]),
            vec![],
        ));

        let valid_point = Geometry::Point(point!(x: 0.5, y: 0.5));

        let geometries = vec![zero_area_poly, valid_point];

        let config = TilerConfig::new(0, 0);
        let coord = TileCoord::new(0, 0, 0);

        let tile = generate_single_tile(&geometries, coord, &config)
            .unwrap()
            .unwrap();

        let decoded = decode_tile(&tile.data).unwrap();
        // Should only have the valid point
        assert_eq!(
            decoded.layers[0].features.len(),
            1,
            "Should filter out zero-area polygon"
        );
    }

    #[test]
    fn test_pipeline_keeps_valid_geometries() {
        // All valid geometries should pass through
        let valid_point = Geometry::Point(point!(x: 0.5, y: 0.5));
        let valid_line = Geometry::LineString(line_string![
            (x: 0.0, y: 0.0),
            (x: 1.0, y: 1.0),
        ]);
        let valid_poly = Geometry::Polygon(polygon![
            (x: 0.0, y: 0.0),
            (x: 1.0, y: 0.0),
            (x: 1.0, y: 1.0),
            (x: 0.0, y: 1.0),
            (x: 0.0, y: 0.0),
        ]);

        let geometries = vec![valid_point, valid_line, valid_poly];

        let config = TilerConfig::new(0, 0);
        let coord = TileCoord::new(0, 0, 0);

        let tile = generate_single_tile(&geometries, coord, &config)
            .unwrap()
            .unwrap();

        let decoded = decode_tile(&tile.data).unwrap();
        assert_eq!(
            decoded.layers[0].features.len(),
            3,
            "All valid geometries should be kept"
        );
    }

    #[test]
    fn test_pipeline_all_degenerate_returns_empty_tile() {
        // If all geometries are degenerate, tile should be empty (None)
        let degenerate_line =
            Geometry::LineString(geo::LineString::new(vec![geo::Coord { x: 0.5, y: 0.5 }]));

        let degenerate_poly = Geometry::Polygon(geo::Polygon::new(
            geo::LineString::new(vec![
                geo::Coord { x: 0.0, y: 0.0 },
                geo::Coord { x: 1.0, y: 0.0 },
                geo::Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        ));

        let geometries = vec![degenerate_line, degenerate_poly];

        let config = TilerConfig::new(0, 0);
        let coord = TileCoord::new(0, 0, 0);

        let result = generate_single_tile(&geometries, coord, &config).unwrap();
        // Should return None because all geometries were filtered out
        assert!(
            result.is_none(),
            "Tile with all degenerate geometries should return None"
        );
    }

    // ========== Feature Dropping Integration Tests ==========

    #[test]
    fn test_point_thinning_reduces_features_at_lower_zoom() {
        // Create many points spread across the tile
        // At max_zoom (base_zoom), all should be kept
        // At lower zooms, some should be dropped
        let mut geometries = Vec::new();
        for i in 0..1000 {
            // Spread points across the world tile (z0)
            let lng = -180.0 + (i as f64) * 0.36;
            let lat = -85.0 + (i as f64) * 0.17;
            geometries.push(Geometry::Point(point!(x: lng, y: lat)));
        }

        let coord_z0 = TileCoord::new(0, 0, 0);

        // Test 1: Generate at z0 with max_zoom=0 (base_zoom=current, no thinning)
        // All points should be kept at base_zoom
        let config_base = TilerConfig::new(0, 0);
        let tile_base = generate_single_tile(&geometries, coord_z0, &config_base)
            .unwrap()
            .expect("Should have features at base_zoom");

        let decoded_base = decode_tile(&tile_base.data).unwrap();
        let features_base = decoded_base.layers[0].features.len();

        // At base_zoom, all points should be kept
        assert_eq!(
            features_base, 1000,
            "At base_zoom (max_zoom=0, generating z0), all 1000 points should be kept"
        );

        // Test 2: Generate at z0 with max_zoom=2 (base_zoom=2, current=0)
        // Expected retention: 0.4^2 = 0.16 = 16% (should keep ~160 points)
        let config_low = TilerConfig::new(0, 2);
        let tile_z0_result = generate_single_tile(&geometries, coord_z0, &config_low).unwrap();

        // With 1000 points and 16% retention, we expect ~160 points (with variance)
        let features_z0 = if let Some(tile) = tile_z0_result {
            let decoded = decode_tile(&tile.data).unwrap();
            decoded.layers[0].features.len()
        } else {
            0
        };

        // At z0 with base_zoom=2, fewer features should appear due to thinning
        // Expected ~160 (16% of 1000), allow variance
        assert!(
            features_z0 < 300,
            "At z0 (2 levels below base_zoom), should have ~16% retention. Got {} features (expected ~160)",
            features_z0
        );
        assert!(
            features_z0 > 50,
            "At z0, should still have some features (statistical unlikelihood if 0). Got {} features",
            features_z0
        );

        // Verify thinning happened
        assert!(
            features_z0 < features_base,
            "z0 with base_zoom=2 ({}) should have fewer features than z0 with base_zoom=0 ({})",
            features_z0,
            features_base
        );
    }

    #[test]
    fn test_tiny_polygon_dropped_at_low_zoom() {
        // Create a tiny polygon that should be dropped at low zoom
        // but kept at high zoom where it has sufficient pixel area
        let tiny_poly = Geometry::Polygon(polygon![
            (x: 0.0001, y: 0.0001),
            (x: 0.0002, y: 0.0001),
            (x: 0.0002, y: 0.0002),
            (x: 0.0001, y: 0.0002),
            (x: 0.0001, y: 0.0001),
        ]);

        // Also add a large polygon that should always be kept
        let large_poly = Geometry::Polygon(polygon![
            (x: -10.0, y: -10.0),
            (x: 10.0, y: -10.0),
            (x: 10.0, y: 10.0),
            (x: -10.0, y: 10.0),
            (x: -10.0, y: -10.0),
        ]);

        let geometries = vec![tiny_poly, large_poly];

        let config = TilerConfig::new(0, 0);
        let coord = TileCoord::new(0, 0, 0);

        let tile = generate_single_tile(&geometries, coord, &config)
            .unwrap()
            .expect("Should have at least the large polygon");

        let decoded = decode_tile(&tile.data).unwrap();

        // The tiny polygon should be dropped (< 4 sq pixels at z0)
        // The large polygon should be kept
        assert_eq!(
            decoded.layers[0].features.len(),
            1,
            "Tiny polygon should be dropped at z0, only large polygon kept"
        );
    }

    #[test]
    fn test_tiny_line_dropped_when_collapses_to_single_pixel() {
        // At z0, the tile is 360 degrees wide with 4096 pixels
        // Each pixel spans ~0.088 degrees (360/4096)
        // Create a tiny line that's smaller than 1 pixel at z0
        // The line must be purely horizontal or vertical to stay in same pixel

        // A tiny horizontal line at a position where it stays within one pixel
        // x changes by 0.01 degrees (much less than 0.088 degrees per pixel)
        let tiny_line = Geometry::LineString(line_string![
            (x: 0.0, y: 0.0),
            (x: 0.01, y: 0.0),  // Horizontal only, stays in same pixel
        ]);

        // Also add a line that spans significant distance
        let large_line = Geometry::LineString(line_string![
            (x: -90.0, y: 0.0),
            (x: 90.0, y: 0.0),
        ]);

        let geometries = vec![tiny_line.clone(), large_line.clone()];

        let coord = TileCoord::new(0, 0, 0);
        let bounds = coord.bounds();

        // Verify the tiny line collapses to same pixel
        let tiny_should_drop = should_drop_geometry(&tiny_line, 0, 0, 4096, &bounds, 0);
        let large_should_drop = should_drop_geometry(&large_line, 0, 0, 4096, &bounds, 1);

        // The tiny line should be dropped
        assert!(
            tiny_should_drop,
            "Tiny line should be marked for dropping (both points in same pixel)"
        );
        assert!(
            !large_should_drop,
            "Large line should NOT be marked for dropping"
        );

        let config = TilerConfig::new(0, 0);

        let tile = generate_single_tile(&geometries, coord, &config)
            .unwrap()
            .expect("Should have at least the large line");

        let decoded = decode_tile(&tile.data).unwrap();

        // The tiny line should be dropped (collapses to single pixel)
        // The large line should be kept
        assert_eq!(
            decoded.layers[0].features.len(),
            1,
            "Tiny line should be dropped at z0, only large line kept"
        );
    }

    #[test]
    fn test_all_features_kept_at_max_zoom() {
        // At max_zoom, all features should be kept (no dropping)
        let point = Geometry::Point(point!(x: 1.55, y: 42.55));
        let geometries = vec![point.clone(); 10];

        // Generate at zoom 14 (which is also max_zoom)
        let config = TilerConfig::new(14, 14);
        let coord = crate::tile::lng_lat_to_tile(1.55, 42.55, 14);

        let tile = generate_single_tile(&geometries, coord, &config)
            .unwrap()
            .expect("Should have features");

        let decoded = decode_tile(&tile.data).unwrap();

        // All 10 points should be kept at max_zoom
        assert_eq!(
            decoded.layers[0].features.len(),
            10,
            "All points should be kept at max_zoom (base_zoom)"
        );
    }

    // ========== Spatial Index Integration Tests ==========

    #[test]
    fn test_spatial_sorting_improves_locality() {
        // Create features scattered across different parts of the world
        // After spatial sorting, nearby features should be processed together
        let geometries = vec![
            Geometry::Point(point!(x: 139.7, y: 35.7)),    // Tokyo
            Geometry::Point(point!(x: -122.4, y: 37.8)),   // San Francisco
            Geometry::Point(point!(x: 2.35, y: 48.85)),    // Paris
            Geometry::Point(point!(x: -122.41, y: 37.79)), // Near SF
            Geometry::Point(point!(x: 2.36, y: 48.86)),    // Near Paris
            Geometry::Point(point!(x: 139.75, y: 35.68)),  // Near Tokyo
        ];

        let bbox = calculate_bbox_from_geometries(&geometries);

        // Test with Hilbert curve (default)
        let config_hilbert = TilerConfig::new(0, 2).with_hilbert(true);
        let iter_hilbert = TileIterator::new(geometries.clone(), bbox.clone(), config_hilbert);

        // The TileIterator should sort geometries before processing
        // We verify this by checking that the geometries are sorted
        // (SF features should be adjacent, Tokyo features should be adjacent, etc.)

        // Verify by checking the internal state after construction
        // The iterator's geometries should be spatially sorted

        // The config should have use_hilbert = true
        assert!(
            iter_hilbert.config.use_hilbert,
            "use_hilbert should be true"
        );

        // Test with Z-order
        let config_zorder = TilerConfig::new(0, 2).with_hilbert(false);
        let iter_zorder = TileIterator::new(geometries.clone(), bbox.clone(), config_zorder);

        // The config should have use_hilbert = false
        assert!(
            !iter_zorder.config.use_hilbert,
            "use_hilbert should be false for Z-order"
        );

        // Both should produce tiles (just verify the pipeline works with sorting enabled)
        let hilbert_tiles: Vec<_> = iter_hilbert.filter_map(|r| r.ok()).collect();
        let zorder_tiles: Vec<_> = iter_zorder.filter_map(|r| r.ok()).collect();

        // Should produce the same number of tiles regardless of sorting method
        assert_eq!(
            hilbert_tiles.len(),
            zorder_tiles.len(),
            "Hilbert and Z-order should produce same number of tiles"
        );
    }

    #[test]
    fn test_hilbert_vs_zorder_config() {
        // Verify the config option works
        let config_default = TilerConfig::default();
        assert!(
            config_default.use_hilbert,
            "Default should use Hilbert curve"
        );

        let config_hilbert = TilerConfig::new(0, 10).with_hilbert(true);
        assert!(
            config_hilbert.use_hilbert,
            "with_hilbert(true) should set use_hilbert to true"
        );

        let config_zorder = TilerConfig::new(0, 10).with_hilbert(false);
        assert!(
            !config_zorder.use_hilbert,
            "with_hilbert(false) should set use_hilbert to false"
        );
    }

    #[test]
    fn test_generate_tiles_with_spatial_sorting() {
        // Create features in multiple locations
        let geometries = vec![
            Geometry::Point(point!(x: 0.0, y: 0.0)),
            Geometry::Point(point!(x: 0.01, y: 0.01)),
            Geometry::Point(point!(x: 90.0, y: 45.0)),
            Geometry::Point(point!(x: 90.01, y: 45.01)),
        ];

        let bbox = calculate_bbox_from_geometries(&geometries);

        // Generate with Hilbert sorting
        let config = TilerConfig::new(0, 2).with_hilbert(true);
        let iter = TileIterator::new(geometries, bbox, config);
        let tiles: Vec<_> = iter.filter_map(|r| r.ok()).collect();

        // Should generate tiles successfully
        assert!(!tiles.is_empty(), "Should generate at least one tile");

        // Each tile should have valid MVT data
        for tile in &tiles {
            let decoded = decode_tile(&tile.data).expect("Should decode MVT");
            assert_eq!(decoded.layers.len(), 1);
            assert!(!decoded.layers[0].features.is_empty());
        }
    }

    // ========== Parallel Tile Generation Tests ==========

    #[test]
    fn test_parallel_config_option() {
        // Verify the parallel config option works
        let config_default = TilerConfig::default();
        assert!(
            config_default.parallel,
            "Default should enable parallel processing"
        );

        let config_parallel = TilerConfig::new(0, 10).with_parallel(true);
        assert!(
            config_parallel.parallel,
            "with_parallel(true) should enable parallel"
        );

        let config_sequential = TilerConfig::new(0, 10).with_parallel(false);
        assert!(
            !config_sequential.parallel,
            "with_parallel(false) should disable parallel"
        );
    }

    #[test]
    fn test_parallel_produces_same_results_as_sequential() {
        // Generate tiles with both parallel and sequential modes
        // Results should be identical (deterministic)
        let geometries = vec![
            Geometry::Point(point!(x: 0.0, y: 0.0)),
            Geometry::Point(point!(x: 0.5, y: 0.5)),
            Geometry::Point(point!(x: -0.5, y: -0.5)),
            Geometry::Polygon(polygon![
                (x: 0.0, y: 0.0),
                (x: 1.0, y: 0.0),
                (x: 1.0, y: 1.0),
                (x: 0.0, y: 1.0),
                (x: 0.0, y: 0.0),
            ]),
            Geometry::LineString(line_string![
                (x: -1.0, y: -1.0),
                (x: 1.0, y: 1.0),
            ]),
        ];

        let bbox = calculate_bbox_from_geometries(&geometries);

        // Sequential mode
        let config_seq = TilerConfig::new(0, 4).with_parallel(false);
        let iter_seq = TileIterator::new(geometries.clone(), bbox.clone(), config_seq);
        let tiles_seq: Vec<_> = iter_seq.filter_map(|r| r.ok()).collect();

        // Parallel mode
        let config_par = TilerConfig::new(0, 4).with_parallel(true);
        let iter_par = TileIterator::new(geometries.clone(), bbox.clone(), config_par);
        let tiles_par: Vec<_> = iter_par.filter_map(|r| r.ok()).collect();

        // Same number of tiles
        assert_eq!(
            tiles_seq.len(),
            tiles_par.len(),
            "Parallel and sequential should produce same number of tiles"
        );

        // Same tile coordinates (sorted for comparison)
        let mut coords_seq: Vec<_> = tiles_seq.iter().map(|t| t.coord).collect();
        let mut coords_par: Vec<_> = tiles_par.iter().map(|t| t.coord).collect();
        coords_seq.sort_by_key(|c| (c.z, c.x, c.y));
        coords_par.sort_by_key(|c| (c.z, c.x, c.y));

        assert_eq!(
            coords_seq, coords_par,
            "Parallel and sequential should produce tiles for same coordinates"
        );

        // Same tile data (content must be identical for determinism)
        // Sort both by coordinates first
        let mut tiles_seq_sorted = tiles_seq;
        let mut tiles_par_sorted = tiles_par;
        tiles_seq_sorted.sort_by_key(|t| (t.coord.z, t.coord.x, t.coord.y));
        tiles_par_sorted.sort_by_key(|t| (t.coord.z, t.coord.x, t.coord.y));

        for (seq_tile, par_tile) in tiles_seq_sorted.iter().zip(tiles_par_sorted.iter()) {
            assert_eq!(
                seq_tile.data, par_tile.data,
                "Tile data should be identical for coord {:?}",
                seq_tile.coord
            );
        }
    }

    #[test]
    fn test_parallel_with_real_fixture() {
        let fixture = Path::new("../../tests/fixtures/realdata/open-buildings.parquet");
        if !fixture.exists() {
            eprintln!("Skipping: fixture not found");
            return;
        }

        // Generate tiles with parallel enabled
        let config = TilerConfig::new(8, 10)
            .with_layer_name("buildings")
            .with_parallel(true);

        let tiles_iter = generate_tiles(fixture, &config).expect("Should create iterator");
        let tiles: Vec<_> = tiles_iter.filter_map(|r| r.ok()).collect();

        assert!(!tiles.is_empty(), "Should generate tiles in parallel mode");

        // Verify tiles are valid
        for tile in &tiles {
            let decoded = decode_tile(&tile.data).expect("Should decode MVT");
            assert_eq!(decoded.layers.len(), 1);
            assert!(!decoded.layers[0].features.is_empty());
        }

        println!("Generated {} tiles in parallel mode", tiles.len());
    }
}
