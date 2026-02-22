#![recursion_limit = "256"]
//! Core library for converting GeoParquet to PMTiles vector tiles.
//!
//! This library provides the foundational functionality for reading GeoParquet files
//! and converting them into PMTiles vector tile archives with MVT encoding.
//!
//! # Examples
//!
//! ```no_run
//! use gpq_tiles_core::{Converter, Config};
//!
//! let config = Config {
//!     min_zoom: 0,
//!     max_zoom: 14,
//!     ..Default::default()
//! };
//!
//! let converter = Converter::new(config);
//! converter.convert("input.parquet", "output.pmtiles").unwrap();
//! ```

use arrow_schema::SchemaRef;
use geoparquet::reader::GeoParquetReaderBuilder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashMap;
use std::path::Path;
use thiserror::Error;

use crate::tile::{tiles_for_bbox, TileBounds, TileCoord};

// Include the protobuf-generated code
pub mod vector_tile {
    include!(concat!(env!("OUT_DIR"), "/vector_tile.rs"));
}

pub mod batch_processor;
pub mod clip;
pub mod feature_drop;
#[cfg(test)]
mod golden;
#[cfg(test)]
mod integration_tests;
pub mod mvt;
pub mod pipeline;
pub mod pmtiles_writer;
pub mod simplify;
pub mod tile;
pub mod validate;

/// Errors that can occur during GeoParquet to PMTiles conversion
#[derive(Error, Debug)]
pub enum Error {
    #[error("Failed to read GeoParquet file: {0}")]
    GeoParquetRead(String),

    #[error("Failed to write PMTiles: {0}")]
    PMTilesWrite(String),

    #[error("Invalid geometry at feature {feature_id}: {reason}")]
    InvalidGeometry { feature_id: usize, reason: String },

    #[error("MVT encoding failed: {0}")]
    MvtEncoding(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Configuration for the GeoParquet to PMTiles conversion
#[derive(Debug, Clone)]
pub struct Config {
    /// Minimum zoom level to generate
    pub min_zoom: u8,
    /// Maximum zoom level to generate
    pub max_zoom: u8,
    /// Tile extent (default: 4096 as per MVT spec)
    pub extent: u32,
    /// Feature dropping density threshold
    pub drop_density: DropDensity,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            min_zoom: 0,
            max_zoom: 14,
            extent: 4096,
            drop_density: DropDensity::Medium,
        }
    }
}

/// Feature dropping density levels
#[derive(Debug, Clone, Copy)]
pub enum DropDensity {
    Low,
    Medium,
    High,
}

/// Main converter struct
pub struct Converter {
    #[allow(dead_code)] // Used in future phases
    config: Config,
}

impl Converter {
    /// Create a new converter with the given configuration
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Convert a GeoParquet file to PMTiles
    ///
    /// # Phase 2 Progress
    ///
    /// Currently:
    /// - ✓ Reads GeoParquet with geoarrow
    /// - ✓ Iterates features via RecordBatch
    /// - ✓ Calculates tile grid (placeholder bbox for now)
    /// - TODO: Extract geometries from batches (needed for real bbox)
    /// - TODO: Calculate actual dataset bounds from geometries
    /// - TODO: Clip to tile bounds
    /// - TODO: Simplify geometries
    /// - TODO: Encode as MVT
    /// - TODO: Write to PMTiles
    pub fn convert<P: AsRef<Path>, Q: AsRef<Path>>(&self, input: P, output: Q) -> Result<()> {
        let input_path = input.as_ref();
        let output_path = output.as_ref();

        log::info!(
            "Converting {} to {}",
            input_path.display(),
            output_path.display()
        );

        // Step 1: Read GeoParquet file
        let (schema, num_rows) = self.read_geoparquet(input_path)?;

        log::info!(
            "Read GeoParquet: {} rows, {} columns",
            num_rows,
            schema.fields().len()
        );

        // Step 2: Iterate features in batches
        let batch_count = self.iterate_features(input_path)?;

        log::info!("Processed {} batches of features", batch_count);

        // Step 3: Calculate dataset bounding box
        let bbox = self.calculate_bounds(input_path)?;

        log::info!(
            "Dataset bounds: ({}, {}) to ({}, {})",
            bbox.lng_min,
            bbox.lat_min,
            bbox.lng_max,
            bbox.lat_max
        );
        log::info!(
            "  Width: {:.4}°, Height: {:.4}°",
            bbox.width(),
            bbox.height()
        );

        // Step 4: Generate tile grid for all zoom levels
        let tile_grid = self.generate_tile_grid(&bbox)?;

        for (zoom, tiles) in &tile_grid {
            log::info!("  Zoom {}: {} tiles", zoom, tiles.len());
        }

        // TODO: Extract geometries and generate tiles

        // Create empty output file as proof of concept
        std::fs::File::create(output_path)?;

        log::info!(
            "Phase 2 in progress: Read {} features in {} batches from {}",
            num_rows,
            batch_count,
            input_path.display()
        );

        Ok(())
    }

    /// Read a GeoParquet file and return schema + row count
    fn read_geoparquet<P: AsRef<Path>>(&self, path: P) -> Result<(SchemaRef, usize)> {
        let path = path.as_ref();

        if !path.exists() {
            return Err(Error::GeoParquetRead(format!(
                "Input file does not exist: {}",
                path.display()
            )));
        }

        let file = std::fs::File::open(path)
            .map_err(|e| Error::GeoParquetRead(format!("Failed to open file: {}", e)))?;

        // Create parquet reader builder
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| Error::GeoParquetRead(format!("Failed to create reader: {}", e)))?;

        // Get GeoParquet metadata
        let metadata_result = builder
            .geoparquet_metadata()
            .ok_or_else(|| Error::GeoParquetRead("No GeoParquet metadata found".to_string()))?;

        let metadata = metadata_result.map_err(|e| {
            Error::GeoParquetRead(format!("Failed to parse GeoParquet metadata: {}", e))
        })?;

        // Get schema with GeoArrow types (parse WKB to native GeoArrow)
        let schema = builder
            .geoarrow_schema(
                &metadata,
                true,               // parse_wkb: convert WKB to native GeoArrow types
                Default::default(), // coord_type: use default coordinate type
            )
            .map_err(|e| Error::GeoParquetRead(format!("Failed to infer schema: {}", e)))?;

        // Get row count from metadata
        let num_rows = builder.metadata().file_metadata().num_rows() as usize;

        if num_rows == 0 {
            log::warn!("GeoParquet file is empty");
        }

        Ok((schema, num_rows))
    }

    /// Iterate through features in a GeoParquet file
    /// Returns the number of batches processed
    fn iterate_features<P: AsRef<Path>>(&self, path: P) -> Result<usize> {
        let path = path.as_ref();

        let file = std::fs::File::open(path)
            .map_err(|e| Error::GeoParquetRead(format!("Failed to open file: {}", e)))?;

        // Create parquet reader builder
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .map_err(|e| Error::GeoParquetRead(format!("Failed to create reader: {}", e)))?;

        // Build the reader
        let reader = builder
            .build()
            .map_err(|e| Error::GeoParquetRead(format!("Failed to build reader: {}", e)))?;

        let mut batch_count = 0;
        let mut total_rows = 0;

        // Iterate through batches
        for batch_result in reader {
            let batch = batch_result
                .map_err(|e| Error::GeoParquetRead(format!("Failed to read batch: {}", e)))?;

            batch_count += 1;
            total_rows += batch.num_rows();

            log::debug!("Batch {}: {} rows", batch_count, batch.num_rows());

            // TODO: Extract geometries from batch
            // TODO: Process features for tiling
        }

        log::info!(
            "Iterated {} total rows in {} batches",
            total_rows,
            batch_count
        );

        Ok(batch_count)
    }

    /// Calculate the bounding box of all features in the GeoParquet file
    fn calculate_bounds<P: AsRef<Path>>(&self, path: P) -> Result<TileBounds> {
        batch_processor::calculate_bbox(path.as_ref())
    }

    /// Generate tile grid for all zoom levels
    fn generate_tile_grid(&self, bbox: &TileBounds) -> Result<HashMap<u8, Vec<TileCoord>>> {
        let mut tile_grid = HashMap::new();

        for zoom in self.config.min_zoom..=self.config.max_zoom {
            let tiles: Vec<TileCoord> = tiles_for_bbox(bbox, zoom).collect();
            tile_grid.insert(zoom, tiles);
        }

        Ok(tile_grid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert_eq!(config.min_zoom, 0);
        assert_eq!(config.max_zoom, 14);
        assert_eq!(config.extent, 4096);
    }

    #[test]
    fn test_converter_creation() {
        let config = Config::default();
        let _converter = Converter::new(config);
        // If we get here without panicking, the test passes
    }

    #[test]
    fn test_convert_nonexistent_file() {
        let config = Config::default();
        let converter = Converter::new(config);

        let result = converter.convert("/nonexistent/file.parquet", "/tmp/output.pmtiles");

        assert!(result.is_err());
        match result {
            Err(Error::GeoParquetRead(_)) => {} // Expected error type
            _ => panic!("Expected GeoParquetRead error"),
        }
    }

    #[test]
    fn test_read_geoparquet() {
        let config = Config::default();
        let converter = Converter::new(config);

        let fixture = "../../tests/fixtures/realdata/open-buildings.parquet";

        if Path::new(fixture).exists() {
            let result = converter.read_geoparquet(fixture);
            assert!(result.is_ok());

            let (schema, num_rows) = result.unwrap();
            assert!(num_rows > 0, "Should have rows");
            assert!(!schema.fields().is_empty(), "Should have columns");
        }
    }

    #[test]
    fn test_convert_with_real_fixture() {
        let config = Config::default();
        let converter = Converter::new(config);

        // Use one of our real fixtures
        let fixture = "../../tests/fixtures/realdata/open-buildings.parquet";
        let output = "/tmp/test-output.pmtiles";

        // Only run if fixture exists
        if Path::new(fixture).exists() {
            let result = converter.convert(fixture, output);
            assert!(result.is_ok());

            // Verify output was created
            assert!(Path::new(output).exists());

            // Clean up
            let _ = std::fs::remove_file(output);
        }
    }
}
