//! Python bindings for gpq-tiles
//!
//! This module exposes the gpq-tiles-core functionality to Python via pyo3.

use gpq_tiles_core::{Compression, Config, Converter, DropDensity};
use pyo3::prelude::*;

/// Convert GeoParquet to PMTiles
///
/// Args:
///     input (str): Path to input GeoParquet file
///     output (str): Path to output PMTiles file
///     min_zoom (int, optional): Minimum zoom level. Defaults to 0.
///     max_zoom (int, optional): Maximum zoom level. Defaults to 14.
///     drop_density (str, optional): Feature dropping density ("low", "medium", "high"). Defaults to "medium".
///     compression (str, optional): Compression algorithm ("gzip", "brotli", "zstd", "none"). Defaults to "gzip".
///
/// Returns:
///     None
///
/// Raises:
///     Exception: If conversion fails
///
/// Example:
///     >>> from gpq_tiles import convert
///     >>> convert("buildings.parquet", "buildings.pmtiles", min_zoom=0, max_zoom=14)
///     >>> convert("buildings.parquet", "buildings.pmtiles", compression="zstd")
#[pyfunction]
#[pyo3(signature = (input, output, min_zoom=0, max_zoom=14, drop_density="medium", compression="gzip"))]
fn convert(
    input: &str,
    output: &str,
    min_zoom: u8,
    max_zoom: u8,
    drop_density: &str,
    compression: &str,
) -> PyResult<()> {
    // Parse drop density
    let drop_density = match drop_density.to_lowercase().as_str() {
        "low" => DropDensity::Low,
        "medium" => DropDensity::Medium,
        "high" => DropDensity::High,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid drop density: {}",
                drop_density
            )))
        }
    };

    // Parse compression
    let compression = Compression::from_str(compression).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Invalid compression: '{}'. Valid options: none, gzip, brotli, zstd",
            compression
        ))
    })?;

    // Build config
    let config = Config {
        min_zoom,
        max_zoom,
        extent: 4096,
        drop_density,
        layer_name: None,
        compression,
    };

    // Create converter and run
    let converter = Converter::new(config);

    converter
        .convert(input, output)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(())
}

/// gpq_tiles: Fast GeoParquet to PMTiles converter
///
/// This module provides Python bindings for the gpq-tiles Rust library.
#[pymodule]
fn gpq_tiles(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(convert, m)?)?;
    Ok(())
}
