//! GeoParquet file quality assessment for streaming optimization.
//!
//! Detects whether a GeoParquet file is well-optimized for streaming processing:
//! - Has geo metadata extension
//! - Has row group bounding boxes
//! - Is spatially sorted (Hilbert)
//!
//! Emits warnings when files could benefit from optimization with geoparquet-io tools.

use std::path::Path;

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::FileReader;
use parquet::file::serialized_reader::SerializedFileReader;

use crate::{Error, Result};

/// Quality assessment of a GeoParquet file for streaming processing.
#[derive(Debug, Clone)]
pub struct GeoParquetQuality {
    /// Whether the file has GeoParquet `geo` metadata extension
    pub has_geo_metadata: bool,
    /// Whether row groups have bounding box metadata
    pub has_row_group_bboxes: bool,
    /// Number of row groups in the file
    pub row_group_count: usize,
    /// File size in bytes
    pub file_size_bytes: u64,
    /// Percentage of row groups with overlapping bboxes (None if not checked)
    pub row_group_overlap_pct: Option<f32>,
    /// Whether features appear to be Hilbert-sorted (None if not checked for small files)
    pub is_hilbert_sorted: Option<bool>,
}

impl GeoParquetQuality {
    /// Returns true if the file is well-optimized for streaming
    pub fn is_optimized(&self) -> bool {
        self.has_geo_metadata
            && (self.row_group_count <= 1 || self.has_row_group_bboxes)
            && self.is_hilbert_sorted.unwrap_or(true)
    }

    /// Returns a list of optimization suggestions
    pub fn suggestions(&self) -> Vec<String> {
        let mut suggestions = Vec::new();

        if !self.has_geo_metadata {
            suggestions.push("File missing GeoParquet metadata extension".to_string());
        }

        if self.row_group_count > 1 && !self.has_row_group_bboxes {
            suggestions
                .push("Row groups lack bounding box metadata - cannot skip spatially".to_string());
        }

        // Large file with few row groups
        let size_mb = self.file_size_bytes / (1024 * 1024);
        if size_mb > 500 && self.row_group_count < 5 {
            suggestions.push(format!(
                "Large file ({}MB) with only {} row groups limits streaming efficiency",
                size_mb, self.row_group_count
            ));
        }

        if let Some(overlap) = self.row_group_overlap_pct {
            if overlap > 20.0 {
                suggestions.push(format!(
                    "Row group bboxes overlap significantly ({:.1}%)",
                    overlap
                ));
            }
        }

        if let Some(false) = self.is_hilbert_sorted {
            suggestions.push("File does not appear to be spatially sorted".to_string());
        }

        suggestions
    }
}

/// Assess the quality of a GeoParquet file for streaming processing.
///
/// Performs cheap checks (O(1) metadata reads) first, then more expensive
/// checks (sampling) only for large files.
pub fn assess_quality(path: &Path) -> Result<GeoParquetQuality> {
    let file = std::fs::File::open(path)
        .map_err(|e| Error::GeoParquetRead(format!("Failed to open file: {}", e)))?;

    let file_size_bytes = file
        .metadata()
        .map_err(|e| Error::GeoParquetRead(format!("Failed to get file metadata: {}", e)))?
        .len();

    let reader = SerializedFileReader::new(file)
        .map_err(|e| Error::GeoParquetRead(format!("Failed to create parquet reader: {}", e)))?;

    let parquet_metadata = reader.metadata();
    let file_metadata = parquet_metadata.file_metadata();

    // Check for geo metadata in key-value pairs
    let has_geo_metadata = file_metadata
        .key_value_metadata()
        .map(|kv| {
            kv.iter()
                .any(|pair| pair.key.to_lowercase().contains("geo"))
        })
        .unwrap_or(false);

    let row_group_count = parquet_metadata.num_row_groups();

    // Check for row group bboxes (would be in column statistics or custom metadata)
    // For now, we check if row groups have statistics on geometry-like columns
    let has_row_group_bboxes = check_row_group_bboxes(&reader);

    // For large files (>1GB), sample to check Hilbert sorting
    let is_hilbert_sorted = if file_size_bytes > 1024 * 1024 * 1024 {
        Some(check_hilbert_sorted(path)?)
    } else {
        None // Don't check for small files
    };

    Ok(GeoParquetQuality {
        has_geo_metadata,
        has_row_group_bboxes,
        row_group_count,
        file_size_bytes,
        row_group_overlap_pct: None, // TODO: implement bbox overlap check
        is_hilbert_sorted,
    })
}

/// Check if row groups have bounding box information.
fn check_row_group_bboxes(reader: &SerializedFileReader<std::fs::File>) -> bool {
    let metadata = reader.metadata();

    // Check if any row group has statistics that could serve as bbox
    // In well-formed GeoParquet, the geo metadata should contain bbox per row group
    // For now, just check if there's more than one row group with statistics
    if metadata.num_row_groups() <= 1 {
        return true; // Single row group doesn't need bbox filtering
    }

    // Check file-level geo metadata for covering information
    if let Some(kv) = metadata.file_metadata().key_value_metadata() {
        for pair in kv {
            if pair.key.to_lowercase() == "geo" {
                if let Some(value) = &pair.value {
                    // Check if geo metadata contains "covering" which indicates bbox support
                    if value.contains("covering") || value.contains("bbox") {
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// Sample the first N features to check if they're Hilbert-sorted.
fn check_hilbert_sorted(path: &Path) -> Result<bool> {
    use crate::spatial_index::{encode_hilbert, lng_lat_to_world_coords};
    use geo::{BoundingRect, Centroid};

    let file = std::fs::File::open(path)
        .map_err(|e| Error::GeoParquetRead(format!("Failed to open file: {}", e)))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| Error::GeoParquetRead(format!("Failed to create reader: {}", e)))?;

    // Only read first batch for sampling
    let reader = builder
        .with_batch_size(1000)
        .build()
        .map_err(|e| Error::GeoParquetRead(format!("Failed to build reader: {}", e)))?;

    let mut hilbert_indices: Vec<u64> = Vec::new();

    // Process first batch only
    if let Some(batch_result) = reader.into_iter().next() {
        let batch = batch_result
            .map_err(|e| Error::GeoParquetRead(format!("Failed to read batch: {}", e)))?;

        // Use our existing batch processor to extract geometries
        let schema = batch.schema();
        let geom_idx = schema
            .fields()
            .iter()
            .position(|f| f.name() == "geometry" || f.name().contains("geom"));

        if let Some(idx) = geom_idx {
            let geom_col = batch.column(idx);
            let geom_field = schema.field(idx);

            // Convert to GeoArrow and extract centroids
            let geom_array = geoarrow::array::from_arrow_array(geom_col.as_ref(), geom_field)
                .map_err(|e| {
                    Error::GeoParquetRead(format!("Failed to parse geometry array: {}", e))
                })?;

            // Get centroids and compute Hilbert indices
            use geoarrow::datatypes::GeoArrowType;
            use geoarrow_array::cast::AsGeoArrowArray;
            use geoarrow_array::GeoArrowArrayAccessor;

            match geom_array.data_type() {
                GeoArrowType::Polygon(_) => {
                    let arr = geom_array.as_polygon();
                    for item in arr.iter().take(1000) {
                        if let Some(Ok(poly)) = item {
                            use geo_traits::to_geo::ToGeoGeometry;
                            if let Some(geom) = poly.try_to_geometry() {
                                if let Some(centroid) = geom.centroid() {
                                    let (wx, wy) =
                                        lng_lat_to_world_coords(centroid.x(), centroid.y());
                                    hilbert_indices.push(encode_hilbert(wx, wy));
                                }
                            }
                        }
                    }
                }
                GeoArrowType::Point(_) => {
                    let arr = geom_array.as_point();
                    for item in arr.iter().take(1000) {
                        if let Some(Ok(pt)) = item {
                            use geo_traits::to_geo::ToGeoGeometry;
                            if let Some(geom) = pt.try_to_geometry() {
                                if let Some(rect) = geom.bounding_rect() {
                                    let center = rect.center();
                                    let (wx, wy) = lng_lat_to_world_coords(center.x, center.y);
                                    hilbert_indices.push(encode_hilbert(wx, wy));
                                }
                            }
                        }
                    }
                }
                _ => {
                    // For other geometry types, skip the check
                    return Ok(true);
                }
            }
        }
    }

    // Check if indices are mostly sorted (allow 5% out of order)
    if hilbert_indices.len() < 10 {
        return Ok(true); // Too few samples to determine
    }

    let mut inversions = 0;
    for i in 1..hilbert_indices.len() {
        if hilbert_indices[i] < hilbert_indices[i - 1] {
            inversions += 1;
        }
    }

    let inversion_rate = inversions as f64 / hilbert_indices.len() as f64;
    Ok(inversion_rate < 0.05) // Less than 5% inversions = considered sorted
}

/// Emit quality warnings to stderr.
///
/// If `quiet` is true, no warnings are emitted.
pub fn emit_quality_warnings(quality: &GeoParquetQuality, quiet: bool) {
    if quiet {
        return;
    }

    let suggestions = quality.suggestions();
    if suggestions.is_empty() {
        return;
    }

    eprintln!("\n⚠ Input file not optimized for streaming:");
    for suggestion in &suggestions {
        eprintln!("  • {}", suggestion);
    }
    eprintln!();
    eprintln!("  For best performance, optimize with geoparquet-io:");
    eprintln!("    gpq optimize input.parquet -o optimized.parquet --hilbert");
    eprintln!();
    eprintln!("  Proceeding anyway (may use more memory)...\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assess_quality_with_geo_metadata() {
        let fixture = Path::new("../../tests/fixtures/realdata/open-buildings.parquet");
        if !fixture.exists() {
            eprintln!("Skipping: fixture not found");
            return;
        }

        let quality = assess_quality(fixture).expect("Should assess quality");
        assert!(quality.has_geo_metadata, "Should have geo metadata");
        assert!(
            quality.row_group_count >= 1,
            "Should have at least 1 row group"
        );
    }

    #[test]
    fn test_detects_missing_geo_metadata() {
        let fixture = Path::new("../../tests/fixtures/streaming/no-geo-metadata.parquet");
        if !fixture.exists() {
            eprintln!("Skipping: fixture not found");
            return;
        }

        let quality = assess_quality(fixture).expect("Should assess quality");
        assert!(
            !quality.has_geo_metadata,
            "Should detect missing geo metadata"
        );

        let suggestions = quality.suggestions();
        assert!(
            suggestions
                .iter()
                .any(|s| s.contains("missing GeoParquet metadata")),
            "Should suggest adding geo metadata"
        );
    }

    #[test]
    fn test_detects_multiple_row_groups() {
        let fixture = Path::new("../../tests/fixtures/streaming/multi-rowgroup-small.parquet");
        if !fixture.exists() {
            eprintln!("Skipping: fixture not found");
            return;
        }

        let quality = assess_quality(fixture).expect("Should assess quality");
        assert!(
            quality.row_group_count > 1,
            "Should have multiple row groups, got {}",
            quality.row_group_count
        );
    }

    #[test]
    fn test_suggestions_empty_for_good_file() {
        let fixture = Path::new("../../tests/fixtures/realdata/open-buildings.parquet");
        if !fixture.exists() {
            eprintln!("Skipping: fixture not found");
            return;
        }

        let quality = assess_quality(fixture).expect("Should assess quality");
        // Good files should have few or no suggestions
        // (single row group files don't need bbox filtering)
        let suggestions = quality.suggestions();
        assert!(
            suggestions.len() <= 1,
            "Good file should have minimal suggestions, got: {:?}",
            suggestions
        );
    }
}
