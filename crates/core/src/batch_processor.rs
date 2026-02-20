//! Arrow-native geometry batch processing.
//!
//! Processes geometries within Arrow RecordBatch lifetime to preserve zero-copy benefits.
//! DO NOT extract geometries to Vec<Geometry> - that defeats Arrow's purpose.

use std::path::Path;
use std::sync::Arc;

use geo::Geometry;
use geo_traits::to_geo::ToGeoGeometry;
use geoarrow::array::from_arrow_array;
use geoarrow_array::{GeoArrowArray, GeoArrowArrayAccessor, downcast_geoarrow_array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::{Error, Result};
use crate::tile::TileBounds;

/// Process geometries in a GeoParquet file batch-by-batch.
///
/// The callback receives each geometry converted to geo::Geometry for processing.
/// Conversion happens within batch scope to minimize memory usage - we process
/// one geometry at a time rather than bulk-extracting to Vec<Geometry>.
///
/// # Arguments
///
/// * `path` - Path to the GeoParquet file
/// * `callback` - Function called for each geometry with the geometry and its row index
///
/// # Returns
///
/// Total number of geometries processed
pub fn process_geometries<F>(
    path: &Path,
    mut callback: F,
) -> Result<usize>
where
    F: FnMut(Geometry<f64>, usize) -> Result<()>,
{
    let file = std::fs::File::open(path)
        .map_err(|e| Error::GeoParquetRead(format!("Failed to open: {}", e)))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| Error::GeoParquetRead(format!("Failed to create reader: {}", e)))?;

    let reader = builder.build()
        .map_err(|e| Error::GeoParquetRead(format!("Failed to build reader: {}", e)))?;

    let mut total_processed = 0;
    let mut row_offset = 0;

    for batch_result in reader {
        let batch = batch_result
            .map_err(|e| Error::GeoParquetRead(format!("Failed to read batch: {}", e)))?;

        // Find geometry column by name
        let schema = batch.schema();
        let geom_idx = schema.fields().iter()
            .position(|f| f.name() == "geometry" || f.name().contains("geom"))
            .ok_or_else(|| Error::GeoParquetRead("No geometry column found".to_string()))?;

        let geom_col = batch.column(geom_idx);
        let geom_field = schema.field(geom_idx);

        // Convert Arrow array to GeoArrow geometry array
        let geom_array: Arc<dyn GeoArrowArray> = from_arrow_array(geom_col.as_ref(), geom_field)
            .map_err(|e| Error::GeoParquetRead(format!("Failed to parse geometry array: {}", e)))?;

        // Process each geometry within batch scope using the downcast macro
        // This avoids bulk extraction while leveraging GeoArrow's type system
        let batch_count = process_geoarrow_array(
            geom_array.as_ref(),
            row_offset,
            &mut callback,
        )?;

        total_processed += batch_count;
        row_offset += batch.num_rows();
    }

    Ok(total_processed)
}

/// Process geometries from a GeoArrow array, calling the callback for each valid geometry.
///
/// Uses the downcast_geoarrow_array macro to handle all geometry types uniformly.
fn process_geoarrow_array<F>(
    array: &dyn GeoArrowArray,
    row_offset: usize,
    callback: &mut F,
) -> Result<usize>
where
    F: FnMut(Geometry<f64>, usize) -> Result<()>,
{
    // Helper function that works with any GeoArrowArrayAccessor
    fn process_accessor<'a, A, F>(
        accessor: &'a A,
        row_offset: usize,
        callback: &mut F,
    ) -> Result<usize>
    where
        A: GeoArrowArrayAccessor<'a>,
        F: FnMut(Geometry<f64>, usize) -> Result<()>,
    {
        let mut count = 0;
        for (i, item) in accessor.iter().enumerate() {
            if let Some(geom_result) = item {
                // Convert GeoArrow scalar to geo::Geometry
                // This happens one-at-a-time within batch scope
                let geom_trait = geom_result
                    .map_err(|e| Error::GeoParquetRead(format!("Invalid geometry at index {}: {}", i, e)))?;

                // Use ToGeoGeometry trait to convert to geo::Geometry
                if let Some(geo_geom) = geom_trait.try_to_geometry() {
                    callback(geo_geom, row_offset + i)?;
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    // Use the downcast macro to dispatch to the correct array type
    downcast_geoarrow_array!(array, |typed_array| {
        process_accessor(typed_array, row_offset, callback)
    })
}

/// Calculate bounding box by streaming through batches.
/// Does NOT load all geometries into memory.
pub fn calculate_bbox(path: &Path) -> Result<TileBounds> {
    let mut bounds = TileBounds::empty();

    process_geometries(path, |geom, _idx| {
        use geo::BoundingRect;
        if let Some(rect) = geom.bounding_rect() {
            bounds.expand(&TileBounds::new(
                rect.min().x,
                rect.min().y,
                rect.max().x,
                rect.max().y,
            ));
        }
        Ok(())
    })?;

    if !bounds.is_valid() {
        return Err(Error::GeoParquetRead("No valid geometries found".to_string()));
    }

    Ok(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_geometries_iterates_all() {
        let fixture = Path::new("../../tests/fixtures/realdata/open-buildings.parquet");
        if !fixture.exists() {
            eprintln!("Skipping: fixture not found");
            return;
        }

        let mut count = 0;
        let result = process_geometries(fixture, |_geom, _idx| {
            count += 1;
            Ok(())
        });

        assert!(result.is_ok());
        assert!(count > 100, "Should have many features, got {}", count);
    }

    #[test]
    fn test_calculate_bbox_returns_valid_bounds() {
        let fixture = Path::new("../../tests/fixtures/realdata/open-buildings.parquet");
        if !fixture.exists() {
            eprintln!("Skipping: fixture not found");
            return;
        }

        let bbox = calculate_bbox(fixture).expect("Should calculate bbox");

        // Andorra bounds: ~1.4-1.8 lon, ~42.4-42.7 lat
        assert!(bbox.lng_min > 1.0 && bbox.lng_min < 2.0, "lng_min={}", bbox.lng_min);
        assert!(bbox.lng_max > 1.0 && bbox.lng_max < 2.0, "lng_max={}", bbox.lng_max);
        assert!(bbox.lat_min > 42.0 && bbox.lat_min < 43.0, "lat_min={}", bbox.lat_min);
        assert!(bbox.lat_max > 42.0 && bbox.lat_max < 43.0, "lat_max={}", bbox.lat_max);
    }
}
