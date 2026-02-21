//! Tiler pipeline - wires together batch processing, clipping, simplification, and MVT encoding.
//!
//! This module provides the core tiling pipeline that:
//! 1. Reads geometries from GeoParquet
//! 2. Iterates tiles for the data's bounding box at each zoom level
//! 3. For each tile: clips, simplifies, and encodes to MVT format
//!
//! # Tippecanoe Alignment
//!
//! This pipeline matches tippecanoe's approach:
//! - Buffer: 8 pixels (configurable)
//! - Simplification: Douglas-Peucker to tile resolution at each zoom
//! - Empty tiles are skipped (not written)

use std::path::Path;

use prost::Message;

use crate::batch_processor::extract_geometries;
use crate::clip::{buffer_pixels_to_degrees, clip_geometry};
use crate::mvt::{LayerBuilder, TileBuilder};
use crate::simplify::simplify_for_zoom;
use crate::tile::{tiles_for_bbox, TileBounds, TileCoord};
use crate::vector_tile::Tile;
use crate::{Error, Result};

/// Default buffer in pixels (matches tippecanoe common usage)
pub const DEFAULT_BUFFER_PIXELS: u32 = 8;

/// Default tile extent (4096 as per MVT spec)
pub const DEFAULT_EXTENT: u32 = 4096;

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
}

impl Default for TilerConfig {
    fn default() -> Self {
        Self {
            min_zoom: 0,
            max_zoom: 14,
            extent: DEFAULT_EXTENT,
            buffer_pixels: DEFAULT_BUFFER_PIXELS,
            layer_name: "layer".to_string(),
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
}

/// A generated vector tile with its coordinates and data.
#[derive(Debug, Clone)]
pub struct GeneratedTile {
    /// The tile coordinates (x, y, z)
    pub coord: TileCoord,
    /// The MVT protobuf bytes
    pub data: Vec<u8>,
}

impl GeneratedTile {
    /// Create a new generated tile.
    pub fn new(coord: TileCoord, data: Vec<u8>) -> Self {
        Self { coord, data }
    }

    /// Check if the tile is empty (no data).
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
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
    // Step 1: Extract all geometries from the GeoParquet file
    // WARNING: This loads all geometries into memory. For large files,
    // we'll need a streaming approach in Phase 4.
    let geometries = extract_geometries(input_path)?;

    if geometries.is_empty() {
        return Ok(TileIterator::empty());
    }

    // Step 2: Calculate bounding box from geometries
    let bbox = calculate_bbox_from_geometries(&geometries);

    // Step 3: Create tile iterator
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
struct TileIterator {
    geometries: Vec<geo::Geometry<f64>>,
    config: TilerConfig,
    tile_coords: Vec<TileCoord>,
    current_index: usize,
}

impl TileIterator {
    fn new(geometries: Vec<geo::Geometry<f64>>, bbox: TileBounds, config: TilerConfig) -> Self {
        // Collect all tile coordinates for all zoom levels
        let mut tile_coords = Vec::new();
        for zoom in config.min_zoom..=config.max_zoom {
            tile_coords.extend(tiles_for_bbox(&bbox, zoom));
        }

        Self {
            geometries,
            config,
            tile_coords,
            current_index: 0,
        }
    }

    fn empty() -> Self {
        Self {
            geometries: Vec::new(),
            config: TilerConfig::default(),
            tile_coords: Vec::new(),
            current_index: 0,
        }
    }

    /// Process a single tile: clip, simplify, encode to MVT.
    fn process_tile(&self, coord: TileCoord) -> Result<Option<GeneratedTile>> {
        let bounds = coord.bounds();
        let buffer =
            buffer_pixels_to_degrees(self.config.buffer_pixels, &bounds, self.config.extent);

        // Build the layer with clipped/simplified geometries
        let mut layer_builder =
            LayerBuilder::new(&self.config.layer_name).with_extent(self.config.extent);

        let mut feature_count = 0;

        for (idx, geom) in self.geometries.iter().enumerate() {
            // Clip to tile bounds
            if let Some(clipped) = clip_geometry(geom, &bounds, buffer) {
                // Simplify for zoom level
                let simplified = simplify_for_zoom(&clipped, coord.z, self.config.extent);

                // Add to layer (no properties for now)
                layer_builder.add_feature(Some(idx as u64), &simplified, &[], &bounds);
                feature_count += 1;
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

        Ok(Some(GeneratedTile::new(coord, data)))
    }
}

impl Iterator for TileIterator {
    type Item = Result<GeneratedTile>;

    fn next(&mut self) -> Option<Self::Item> {
        // Find the next non-empty tile
        while self.current_index < self.tile_coords.len() {
            let coord = self.tile_coords[self.current_index];
            self.current_index += 1;

            match self.process_tile(coord) {
                Ok(Some(tile)) => return Some(Ok(tile)),
                Ok(None) => continue, // Empty tile, skip
                Err(e) => return Some(Err(e)),
            }
        }

        None
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
    let bounds = coord.bounds();
    let buffer = buffer_pixels_to_degrees(config.buffer_pixels, &bounds, config.extent);

    let mut layer_builder = LayerBuilder::new(&config.layer_name).with_extent(config.extent);

    let mut feature_count = 0;

    for (idx, geom) in geometries.iter().enumerate() {
        if let Some(clipped) = clip_geometry(geom, &bounds, buffer) {
            let simplified = simplify_for_zoom(&clipped, coord.z, config.extent);
            layer_builder.add_feature(Some(idx as u64), &simplified, &[], &bounds);
            feature_count += 1;
        }
    }

    if feature_count == 0 {
        return Ok(None);
    }

    let layer = layer_builder.build();
    let mut tile_builder = TileBuilder::new();
    tile_builder.add_layer(layer);
    let tile = tile_builder.build();
    let data = tile.encode_to_vec();

    Ok(Some(GeneratedTile::new(coord, data)))
}

/// Decode an MVT tile from bytes (for testing).
pub fn decode_tile(data: &[u8]) -> Result<Tile> {
    Tile::decode(data).map_err(|e| Error::MvtEncoding(format!("Failed to decode tile: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{point, polygon, Geometry};
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
        let tile = GeneratedTile::new(coord, data.clone());

        assert_eq!(tile.coord, coord);
        assert_eq!(tile.data, data);
        assert!(!tile.is_empty());
    }

    #[test]
    fn test_generated_tile_empty() {
        let coord = TileCoord::new(0, 0, 0);
        let tile = GeneratedTile::new(coord, vec![]);

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
}
