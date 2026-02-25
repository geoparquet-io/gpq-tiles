//! Python bindings for gpq-tiles
//!
//! This module exposes the gpq-tiles-core functionality to Python via pyo3.

use gpq_tiles_core::pipeline::{generate_tiles_to_writer, StreamingMode, TilerConfig};
use gpq_tiles_core::{Compression, DropDensity, PropertyFilter, StreamingPmtilesWriter};
use pyo3::prelude::*;
use std::path::Path;

/// Convert GeoParquet to PMTiles
///
/// Args:
///     input (str): Path to input GeoParquet file
///     output (str): Path to output PMTiles file
///     min_zoom (int, optional): Minimum zoom level. Defaults to 0.
///     max_zoom (int, optional): Maximum zoom level. Defaults to 14.
///     drop_density (str, optional): Feature dropping density ("low", "medium", "high"). Defaults to "medium".
///     compression (str, optional): Compression algorithm ("gzip", "brotli", "zstd", "none"). Defaults to "gzip".
///     include (list[str], optional): Whitelist of property names to include. Cannot be used with exclude or exclude_all.
///     exclude (list[str], optional): Blacklist of property names to exclude. Cannot be used with include or exclude_all.
///     exclude_all (bool, optional): Exclude all properties, output geometry only. Defaults to False.
///     layer_name (str, optional): Override the layer name (defaults to input filename stem).
///     streaming_mode (str, optional): Memory/speed tradeoff ("fast", "low-memory"). Defaults to "fast".
///         - "fast": Single file pass, ~1-2GB memory for large files. Best for files up to ~10GB.
///         - "low-memory": Process one zoom at a time, ~100-200MB peak. Best for 10GB+ files.
///     parallel_tiles (bool, optional): Enable parallel tile generation. Defaults to True.
///     parallel_geoms (bool, optional): Enable parallel geometry processing. Defaults to True.
///
/// Returns:
///     None
///
/// Raises:
///     ValueError: If invalid parameters or conflicting filter options
///     RuntimeError: If conversion fails
///
/// Example:
///     >>> from gpq_tiles import convert
///     >>> convert("buildings.parquet", "buildings.pmtiles", min_zoom=0, max_zoom=14)
///     >>> convert("buildings.parquet", "buildings.pmtiles", compression="zstd")
///     >>> convert("buildings.parquet", "buildings.pmtiles", include=["name", "height"])
///     >>> convert("buildings.parquet", "buildings.pmtiles", exclude=["internal_id"])
///     >>> convert("buildings.parquet", "buildings.pmtiles", exclude_all=True)
///     >>> convert("buildings.parquet", "buildings.pmtiles", layer_name="my_layer")
///     >>> # Low memory mode for very large files
///     >>> convert("huge.parquet", "huge.pmtiles", streaming_mode="low-memory")
///     >>> # Disable parallelism for debugging
///     >>> convert("data.parquet", "data.pmtiles", parallel_tiles=False, parallel_geoms=False)
#[pyfunction]
#[pyo3(signature = (input, output, min_zoom=0, max_zoom=14, drop_density="medium", compression="gzip", include=None, exclude=None, exclude_all=false, layer_name=None, streaming_mode="fast", parallel_tiles=true, parallel_geoms=true))]
fn convert(
    input: &str,
    output: &str,
    min_zoom: u8,
    max_zoom: u8,
    drop_density: &str,
    compression: &str,
    include: Option<Vec<String>>,
    exclude: Option<Vec<String>>,
    exclude_all: bool,
    layer_name: Option<String>,
    streaming_mode: &str,
    parallel_tiles: bool,
    parallel_geoms: bool,
) -> PyResult<()> {
    // Validate property filter options are mutually exclusive
    let filter_count = include.is_some() as u8 + exclude.is_some() as u8 + exclude_all as u8;
    if filter_count > 1 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Cannot combine include, exclude, and exclude_all. Use only one property filter option.",
        ));
    }

    // Build property filter
    let property_filter = if exclude_all {
        PropertyFilter::ExcludeAll
    } else if let Some(fields) = include {
        PropertyFilter::include(fields)
    } else if let Some(fields) = exclude {
        PropertyFilter::exclude(fields)
    } else {
        PropertyFilter::None
    };

    // Parse drop density
    let drop_density_config = match drop_density.to_lowercase().as_str() {
        "low" => DropDensity::Low,
        "medium" => DropDensity::Medium,
        "high" => DropDensity::High,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid drop density: '{}'. Valid options: low, medium, high",
                drop_density
            )))
        }
    };

    // Parse compression
    let compression_config = Compression::from_str(compression).ok_or_else(|| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Invalid compression: '{}'. Valid options: none, gzip, brotli, zstd",
            compression
        ))
    })?;

    // Parse streaming mode
    let streaming_mode_config = match streaming_mode.to_lowercase().as_str() {
        "fast" => StreamingMode::Fast,
        "low-memory" => StreamingMode::LowMemory,
        _ => {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid streaming mode: '{}'. Valid options: fast, low-memory",
                streaming_mode
            )))
        }
    };

    // Derive layer name from input filename if not provided
    let input_path = Path::new(input);
    let layer_name_str = layer_name.unwrap_or_else(|| {
        input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("layer")
            .to_string()
    });

    // Build TilerConfig
    let config = TilerConfig::new(min_zoom, max_zoom)
        .with_layer_name(&layer_name_str)
        .with_density_drop(matches!(
            drop_density_config,
            DropDensity::Medium | DropDensity::High
        ))
        .with_density_max_per_cell(match drop_density_config {
            DropDensity::Low => 10,
            DropDensity::Medium => 3,
            DropDensity::High => 1,
        })
        .with_property_filter(property_filter)
        .with_streaming_mode(streaming_mode_config)
        .with_parallel(parallel_tiles)
        .with_parallel_geoms(parallel_geoms)
        .with_quiet(true); // Suppress progress output in Python

    // Create streaming writer and generate tiles
    let mut writer = StreamingPmtilesWriter::new(compression_config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    generate_tiles_to_writer(input_path, &config, &mut writer)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Finalize and write to output file
    let output_path = Path::new(output);
    writer
        .finalize(output_path)
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
