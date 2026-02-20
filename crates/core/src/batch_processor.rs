//! Arrow-native geometry batch processing.
//!
//! Processes geometries within Arrow RecordBatch lifetime to preserve zero-copy benefits.
//! DO NOT extract geometries to Vec<Geometry> - that defeats Arrow's purpose.

use std::path::Path;
use crate::{Error, Result};
use crate::tile::TileBounds;

/// Process geometries in a GeoParquet file batch-by-batch.
///
/// The callback receives each geometry and its row index, processed within
/// the Arrow batch scope (no allocations per feature).
pub fn process_geometries<F>(
    _path: &Path,
    _callback: F,
) -> Result<usize>
where
    F: FnMut(geo::Geometry<f64>, usize) -> Result<()>,
{
    todo!("Implement batch processing")
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
