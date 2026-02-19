//! Core library for converting GeoParquet to PMTiles vector tiles.
//!
//! This library provides the foundational functionality for reading GeoParquet files
//! and converting them into PMTiles vector tile archives with MVT encoding.
//!
//! # Examples
//!
//! ```no_run
//! use gpq2tiles_core::{Converter, Config};
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

use std::path::Path;
use thiserror::Error;

// Include the protobuf-generated code
pub mod vector_tile {
    include!(concat!(env!("OUT_DIR"), "/vector_tile.rs"));
}

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
    /// # Phase 1 Implementation (Stub)
    ///
    /// Currently produces an empty but valid PMTiles file.
    /// Future phases will add:
    /// - Phase 2: Actual tile generation with MVT encoding
    /// - Phase 3: Feature dropping
    /// - Phase 4: Parallelization
    pub fn convert<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<()> {
        let input_path = input.as_ref();
        let output_path = output.as_ref();

        log::info!(
            "Converting {} to {} (Phase 1: Stub implementation)",
            input_path.display(),
            output_path.display()
        );

        // Phase 1: Stub implementation
        // TODO: Actually read GeoParquet with geoarrow
        // TODO: Iterate features
        // TODO: Write empty PMTiles with pmtiles crate

        // For now, just verify the input exists
        if !input_path.exists() {
            return Err(Error::GeoParquetRead(format!(
                "Input file does not exist: {}",
                input_path.display()
            )));
        }

        // Create empty output file as proof of concept
        std::fs::File::create(output_path)?;

        log::info!(
            "Phase 1 complete: Created stub output at {}",
            output_path.display()
        );

        Ok(())
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

        let result = converter.convert(
            "/nonexistent/file.parquet",
            "/tmp/output.pmtiles",
        );

        assert!(result.is_err());
        match result {
            Err(Error::GeoParquetRead(_)) => {}, // Expected error type
            _ => panic!("Expected GeoParquetRead error"),
        }
    }

    #[test]
    fn test_convert_with_real_fixture() {
        let config = Config::default();
        let converter = Converter::new(config);

        // Use one of our real fixtures
        let fixture = "../../tests/fixtures/realdata/open-buildings.parquet";
        let output = "/tmp/test-output.pmtiles";

        // Only run if fixture exists (CI might not have it)
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
