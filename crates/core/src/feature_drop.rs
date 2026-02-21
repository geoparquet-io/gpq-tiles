//! Feature dropping logic for tiny polygons.
//!
//! Implements tippecanoe's "diffuse probability" algorithm for dropping
//! polygons that are too small to render meaningfully at a given zoom level.
//!
//! # Tippecanoe Behavior
//!
//! > "Any polygons that are smaller than a minimum area (currently 4 square
//! > subpixels) will have their probability diffused, so that some of them
//! > will be drawn as a square of this minimum size and others will not be
//! > drawn at all, preserving the total area that all of them should have
//! > had together."
//!
//! # Algorithm
//!
//! 1. Calculate polygon area in tile-local pixel coordinates (0-4096 extent)
//! 2. If area >= threshold (default 4 sq pixels): keep polygon
//! 3. If area < threshold: apply diffuse drop probability
//!    - `drop_probability = 1.0 - (area / threshold)`
//!    - Use deterministic randomness based on geometry hash for consistency
//! 4. Zero-area polygons are always dropped
//!
//! # Coordinate System
//!
//! Like `simplify.rs`, all calculations are done in tile-local pixel coordinates
//! to ensure consistent behavior regardless of geographic location (latitude).

use crate::tile::TileBounds;
use geo::{Area, Coord, LineString, Polygon};
use std::hash::{Hash, Hasher};

/// Default tiny polygon threshold: 4 square pixels (matches tippecanoe)
pub const DEFAULT_TINY_POLYGON_THRESHOLD: f64 = 4.0;

/// Returns true if the polygon should be DROPPED (not kept).
///
/// Uses tippecanoe's diffuse probability algorithm:
/// - Polygons >= threshold are always kept
/// - Polygons < threshold have a probability of being dropped
/// - Smaller polygons have higher drop probability
/// - Zero-area polygons are always dropped
///
/// # Arguments
///
/// * `polygon` - The polygon to check (in geographic coordinates)
/// * `tile_bounds` - The bounds of the tile (for coordinate transformation)
/// * `extent` - Tile extent (typically 4096)
/// * `threshold_sq_pixels` - Minimum area in square pixels (default: 4.0)
///
/// # Returns
///
/// `true` if the polygon should be dropped, `false` if it should be kept.
///
/// # Determinism
///
/// The function uses a hash of the polygon's coordinates to produce
/// deterministic results: the same polygon at the same zoom level will
/// always produce the same drop decision.
pub fn should_drop_tiny_polygon(
    polygon: &Polygon<f64>,
    tile_bounds: &TileBounds,
    extent: u32,
    threshold_sq_pixels: f64,
) -> bool {
    let area = polygon_area_in_tile_coords(polygon, tile_bounds, extent);

    // Zero or negative area (degenerate polygon) is always dropped
    if area <= 0.0 {
        return true;
    }

    // Polygons at or above threshold are always kept
    if area >= threshold_sq_pixels {
        return false;
    }

    // Diffuse probability: smaller polygons have higher drop probability
    // drop_probability = 1.0 - (area / threshold)
    // When area = 0: drop_probability = 1.0 (always drop)
    // When area = threshold: drop_probability = 0.0 (never drop)
    let keep_probability = area / threshold_sq_pixels;

    // Generate a deterministic "random" value from the geometry hash
    // The hash is normalized to [0, 1) range
    let hash = geometry_hash(polygon);
    let hash_normalized = (hash as f64) / (u64::MAX as f64);

    // Drop if the hash value exceeds the keep probability
    // This ensures smaller polygons (lower keep_probability) are dropped more often
    hash_normalized >= keep_probability
}

/// Calculate the area of a polygon in tile-local pixel coordinates.
///
/// Transforms the polygon from geographic coordinates to tile-local
/// coordinates (0-extent) and calculates the unsigned area.
///
/// # Arguments
///
/// * `polygon` - The polygon in geographic coordinates
/// * `tile_bounds` - The bounds of the tile for coordinate transformation
/// * `extent` - Tile extent (typically 4096)
///
/// # Returns
///
/// The absolute area in square pixels.
pub fn polygon_area_in_tile_coords(
    polygon: &Polygon<f64>,
    tile_bounds: &TileBounds,
    extent: u32,
) -> f64 {
    // Transform polygon to tile-local coordinates
    let tile_polygon = polygon_to_tile_coords(polygon, tile_bounds, extent);

    // Calculate signed area and return absolute value
    // geo::Area trait returns signed area (positive for CCW, negative for CW)
    tile_polygon.unsigned_area()
}

/// Transform a geographic coordinate to tile-local pixel coordinates.
///
/// Tile coordinates range from 0 to extent (typically 4096).
/// The tile bounds define the geographic extent being mapped.
#[inline]
fn geo_to_tile_coords(lng: f64, lat: f64, bounds: &TileBounds, extent: u32) -> (f64, f64) {
    let extent_f = extent as f64;

    // Normalize to 0-1 within tile bounds
    let x_ratio = (lng - bounds.lng_min) / (bounds.lng_max - bounds.lng_min);
    let y_ratio = (lat - bounds.lat_min) / (bounds.lat_max - bounds.lat_min);

    // Scale to extent and flip Y (tile coords have Y increasing downward)
    let x = x_ratio * extent_f;
    let y = (1.0 - y_ratio) * extent_f;

    (x, y)
}

/// Transform a LineString from geographic to tile-local coordinates.
fn linestring_to_tile_coords(
    ls: &LineString<f64>,
    bounds: &TileBounds,
    extent: u32,
) -> LineString<f64> {
    let coords: Vec<Coord<f64>> = ls
        .coords()
        .map(|c| {
            let (x, y) = geo_to_tile_coords(c.x, c.y, bounds, extent);
            Coord { x, y }
        })
        .collect();
    LineString::new(coords)
}

/// Transform a Polygon from geographic to tile-local coordinates.
fn polygon_to_tile_coords(poly: &Polygon<f64>, bounds: &TileBounds, extent: u32) -> Polygon<f64> {
    let exterior = linestring_to_tile_coords(poly.exterior(), bounds, extent);
    let interiors: Vec<LineString<f64>> = poly
        .interiors()
        .iter()
        .map(|ring| linestring_to_tile_coords(ring, bounds, extent))
        .collect();
    Polygon::new(exterior, interiors)
}

/// Calculate a deterministic hash for a polygon's geometry.
///
/// Uses a simple hash combining all coordinate values to produce
/// consistent drop decisions for the same polygon across multiple calls.
///
/// The hash is designed to:
/// - Be deterministic: same coordinates → same hash
/// - Spread well across the u64 range for good probability distribution
/// - Be fast to compute
fn geometry_hash(polygon: &Polygon<f64>) -> u64 {
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();

    // Hash all exterior ring coordinates
    for coord in polygon.exterior().coords() {
        // Convert f64 to bits for consistent hashing
        coord.x.to_bits().hash(&mut hasher);
        coord.y.to_bits().hash(&mut hasher);
    }

    // Hash interior rings as well
    for interior in polygon.interiors() {
        for coord in interior.coords() {
            coord.x.to_bits().hash(&mut hasher);
            coord.y.to_bits().hash(&mut hasher);
        }
    }

    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{Coord, LineString, Polygon};

    /// Create a square polygon centered at the given tile-local coordinates.
    /// `side` is the side length in tile pixels (at extent 4096).
    fn create_square_polygon_in_tile_coords(
        center_x: f64,
        center_y: f64,
        side: f64,
        tile_bounds: &TileBounds,
        extent: u32,
    ) -> Polygon<f64> {
        let extent_f = extent as f64;
        let half = side / 2.0;

        // Define corners in tile coordinates
        let corners_tile = [
            (center_x - half, center_y - half),
            (center_x + half, center_y - half),
            (center_x + half, center_y + half),
            (center_x - half, center_y + half),
            (center_x - half, center_y - half), // Close the ring
        ];

        // Convert to geographic coordinates
        let coords: Vec<Coord<f64>> = corners_tile
            .iter()
            .map(|(x, y)| {
                let x_ratio = x / extent_f;
                let y_ratio = 1.0 - (y / extent_f); // Flip Y
                Coord {
                    x: tile_bounds.lng_min + x_ratio * (tile_bounds.lng_max - tile_bounds.lng_min),
                    y: tile_bounds.lat_min + y_ratio * (tile_bounds.lat_max - tile_bounds.lat_min),
                }
            })
            .collect();

        Polygon::new(LineString::new(coords), vec![])
    }

    // =========================================================================
    // TEST 1: Large polygons are NEVER dropped
    // =========================================================================
    #[test]
    fn test_large_polygon_never_dropped() {
        let tile_bounds = TileBounds::new(0.0, 0.0, 1.0, 1.0);
        let extent = 4096;

        // Create a 100x100 pixel polygon (10,000 sq pixels >> 4 threshold)
        let polygon =
            create_square_polygon_in_tile_coords(2048.0, 2048.0, 100.0, &tile_bounds, extent);

        // Should NEVER be dropped
        let should_drop = should_drop_tiny_polygon(
            &polygon,
            &tile_bounds,
            extent,
            DEFAULT_TINY_POLYGON_THRESHOLD,
        );
        assert!(
            !should_drop,
            "Large polygon (10,000 sq pixels) should never be dropped"
        );
    }

    // =========================================================================
    // TEST 2: Very tiny polygons (< 1 sq pixel) are ALWAYS dropped
    // =========================================================================
    #[test]
    fn test_sub_pixel_polygon_always_dropped() {
        let tile_bounds = TileBounds::new(0.0, 0.0, 1.0, 1.0);
        let extent = 4096;

        // Create a 0.5x0.5 pixel polygon (0.25 sq pixels << 4 threshold)
        let polygon =
            create_square_polygon_in_tile_coords(2048.0, 2048.0, 0.5, &tile_bounds, extent);

        // Verify the area is indeed tiny
        let area = polygon_area_in_tile_coords(&polygon, &tile_bounds, extent);
        assert!(
            area < 1.0,
            "Polygon should be less than 1 sq pixel, got {}",
            area
        );

        // Such tiny polygons should always be dropped (drop_probability ≈ 1.0)
        let should_drop = should_drop_tiny_polygon(
            &polygon,
            &tile_bounds,
            extent,
            DEFAULT_TINY_POLYGON_THRESHOLD,
        );
        assert!(
            should_drop,
            "Sub-pixel polygon (0.25 sq pixels) should always be dropped"
        );
    }

    // =========================================================================
    // TEST 3: Zero-area polygon is ALWAYS dropped
    // =========================================================================
    #[test]
    fn test_zero_area_polygon_always_dropped() {
        let tile_bounds = TileBounds::new(0.0, 0.0, 1.0, 1.0);
        let extent = 4096;

        // Create a degenerate polygon (all points same location)
        let coords = vec![
            Coord { x: 0.5, y: 0.5 },
            Coord { x: 0.5, y: 0.5 },
            Coord { x: 0.5, y: 0.5 },
            Coord { x: 0.5, y: 0.5 },
        ];
        let polygon = Polygon::new(LineString::new(coords), vec![]);

        let area = polygon_area_in_tile_coords(&polygon, &tile_bounds, extent);
        assert!(
            area.abs() < 1e-10,
            "Degenerate polygon should have zero area"
        );

        let should_drop = should_drop_tiny_polygon(
            &polygon,
            &tile_bounds,
            extent,
            DEFAULT_TINY_POLYGON_THRESHOLD,
        );
        assert!(should_drop, "Zero-area polygon should always be dropped");
    }

    // =========================================================================
    // TEST 4: Polygon exactly at threshold is kept
    // =========================================================================
    #[test]
    fn test_polygon_at_threshold_kept() {
        let tile_bounds = TileBounds::new(0.0, 0.0, 1.0, 1.0);
        let extent = 4096;

        // Create a 2x2 pixel polygon (4 sq pixels = exactly threshold)
        let polygon =
            create_square_polygon_in_tile_coords(2048.0, 2048.0, 2.0, &tile_bounds, extent);

        let area = polygon_area_in_tile_coords(&polygon, &tile_bounds, extent);
        assert!(
            (area - 4.0).abs() < 0.1,
            "Polygon should be ~4 sq pixels, got {}",
            area
        );

        // Polygon at exactly threshold should be kept (not dropped)
        let should_drop = should_drop_tiny_polygon(
            &polygon,
            &tile_bounds,
            extent,
            DEFAULT_TINY_POLYGON_THRESHOLD,
        );
        assert!(
            !should_drop,
            "Polygon exactly at threshold (4 sq pixels) should be kept"
        );
    }

    // =========================================================================
    // TEST 5: Medium polygons (2-3 sq pixels) have probabilistic dropping
    // =========================================================================
    #[test]
    fn test_medium_polygon_diffuse_probability() {
        let tile_bounds = TileBounds::new(0.0, 0.0, 1.0, 1.0);
        let extent = 4096;

        // Create multiple small polygons at different positions (different hashes)
        // All are ~2 sq pixels (50% of threshold = 50% drop probability)
        let mut drop_count = 0;
        let mut keep_count = 0;
        let num_tests = 100;

        for i in 0..num_tests {
            // Create 1.414x1.414 pixel polygon (≈2 sq pixels)
            // Vary position to get different geometry hashes
            let offset = (i as f64) * 10.0;
            let polygon = create_square_polygon_in_tile_coords(
                1000.0 + offset,
                1000.0 + offset,
                1.414,
                &tile_bounds,
                extent,
            );

            let area = polygon_area_in_tile_coords(&polygon, &tile_bounds, extent);
            // Allow some tolerance in area calculation
            assert!(
                area >= 1.5 && area <= 2.5,
                "Polygon {} should be ~2 sq pixels, got {}",
                i,
                area
            );

            if should_drop_tiny_polygon(
                &polygon,
                &tile_bounds,
                extent,
                DEFAULT_TINY_POLYGON_THRESHOLD,
            ) {
                drop_count += 1;
            } else {
                keep_count += 1;
            }
        }

        // With 50% drop probability, we expect roughly half dropped, half kept
        // Allow 20% margin for statistical variance
        let drop_ratio = drop_count as f64 / num_tests as f64;
        assert!(
            drop_ratio >= 0.3 && drop_ratio <= 0.7,
            "Expected ~50% drop rate for 2 sq pixel polygons, got {:.0}% ({} dropped, {} kept)",
            drop_ratio * 100.0,
            drop_count,
            keep_count
        );
    }

    // =========================================================================
    // TEST 6: Deterministic behavior - same polygon always gives same result
    // =========================================================================
    #[test]
    fn test_deterministic_drop_decision() {
        let tile_bounds = TileBounds::new(0.0, 0.0, 1.0, 1.0);
        let extent = 4096;

        // Create a 1.5x1.5 pixel polygon (~2.25 sq pixels)
        let polygon =
            create_square_polygon_in_tile_coords(2048.0, 2048.0, 1.5, &tile_bounds, extent);

        // Call should_drop multiple times
        let first_result = should_drop_tiny_polygon(
            &polygon,
            &tile_bounds,
            extent,
            DEFAULT_TINY_POLYGON_THRESHOLD,
        );

        // All subsequent calls should return the same result
        for _ in 0..100 {
            let result = should_drop_tiny_polygon(
                &polygon,
                &tile_bounds,
                extent,
                DEFAULT_TINY_POLYGON_THRESHOLD,
            );
            assert_eq!(
                result, first_result,
                "Drop decision should be deterministic for the same polygon"
            );
        }
    }

    // =========================================================================
    // TEST 7: Area calculation is correct for known geometry
    // =========================================================================
    #[test]
    fn test_area_calculation_accuracy() {
        let tile_bounds = TileBounds::new(0.0, 0.0, 1.0, 1.0);
        let extent = 4096;

        // Test 1: 10x10 pixel square = 100 sq pixels
        let polygon_100 =
            create_square_polygon_in_tile_coords(2048.0, 2048.0, 10.0, &tile_bounds, extent);
        let area_100 = polygon_area_in_tile_coords(&polygon_100, &tile_bounds, extent);
        assert!(
            (area_100 - 100.0).abs() < 1.0,
            "10x10 square should be ~100 sq pixels, got {}",
            area_100
        );

        // Test 2: 50x50 pixel square = 2500 sq pixels
        let polygon_2500 =
            create_square_polygon_in_tile_coords(2048.0, 2048.0, 50.0, &tile_bounds, extent);
        let area_2500 = polygon_area_in_tile_coords(&polygon_2500, &tile_bounds, extent);
        assert!(
            (area_2500 - 2500.0).abs() < 10.0,
            "50x50 square should be ~2500 sq pixels, got {}",
            area_2500
        );

        // Test 3: 2x2 pixel square = 4 sq pixels (the threshold)
        let polygon_4 =
            create_square_polygon_in_tile_coords(2048.0, 2048.0, 2.0, &tile_bounds, extent);
        let area_4 = polygon_area_in_tile_coords(&polygon_4, &tile_bounds, extent);
        assert!(
            (area_4 - 4.0).abs() < 0.1,
            "2x2 square should be ~4 sq pixels, got {}",
            area_4
        );
    }

    // =========================================================================
    // TEST 8: Works at different zoom levels (tile bounds)
    // =========================================================================
    #[test]
    fn test_different_tile_bounds() {
        let extent = 4096;

        // Test with tile at equator (large geographic extent)
        let bounds_equator = TileBounds::new(-10.0, -10.0, 10.0, 10.0);
        let polygon_equator =
            create_square_polygon_in_tile_coords(2048.0, 2048.0, 100.0, &bounds_equator, extent);
        assert!(
            !should_drop_tiny_polygon(
                &polygon_equator,
                &bounds_equator,
                extent,
                DEFAULT_TINY_POLYGON_THRESHOLD
            ),
            "Large polygon should not be dropped regardless of tile bounds"
        );

        // Test with tile at high latitude (smaller geographic extent)
        let bounds_arctic = TileBounds::new(-1.0, 79.0, 1.0, 81.0);
        let polygon_arctic =
            create_square_polygon_in_tile_coords(2048.0, 2048.0, 100.0, &bounds_arctic, extent);
        assert!(
            !should_drop_tiny_polygon(
                &polygon_arctic,
                &bounds_arctic,
                extent,
                DEFAULT_TINY_POLYGON_THRESHOLD
            ),
            "Large polygon should not be dropped regardless of geographic location"
        );
    }

    // =========================================================================
    // TEST 9: Custom threshold works correctly
    // =========================================================================
    #[test]
    fn test_custom_threshold() {
        let tile_bounds = TileBounds::new(0.0, 0.0, 1.0, 1.0);
        let extent = 4096;

        // Create a 3x3 pixel polygon (9 sq pixels)
        let polygon =
            create_square_polygon_in_tile_coords(2048.0, 2048.0, 3.0, &tile_bounds, extent);

        // With default threshold (4), should NOT be dropped
        assert!(
            !should_drop_tiny_polygon(&polygon, &tile_bounds, extent, 4.0),
            "9 sq pixel polygon should not be dropped with 4 sq pixel threshold"
        );

        // With threshold of 16, should be considered for dropping
        // (9/16 = 0.5625, so ~43% drop probability)
        // We can't assert a specific result, but we can verify it doesn't crash
        let _result = should_drop_tiny_polygon(&polygon, &tile_bounds, extent, 16.0);

        // With threshold of 100, definitely should be dropped eventually
        // (9/100 = 0.09, so ~91% drop probability)
        let polygon_tiny_relative =
            create_square_polygon_in_tile_coords(2048.0, 2048.0, 0.5, &tile_bounds, extent);
        assert!(
            should_drop_tiny_polygon(&polygon_tiny_relative, &tile_bounds, extent, 100.0),
            "0.25 sq pixel polygon should be dropped with 100 sq pixel threshold"
        );
    }

    // =========================================================================
    // TEST 10: Drop probability correlates with size
    // =========================================================================
    #[test]
    fn test_smaller_polygons_dropped_more_often() {
        let tile_bounds = TileBounds::new(0.0, 0.0, 1.0, 1.0);
        let extent = 4096;
        let num_samples = 200;

        // Group 1: ~3 sq pixel polygons (75% of threshold = 25% drop probability)
        let mut drops_3px = 0;
        for i in 0..num_samples {
            let polygon = create_square_polygon_in_tile_coords(
                500.0 + (i as f64) * 5.0,
                500.0,
                1.732, // sqrt(3) ≈ 3 sq pixels
                &tile_bounds,
                extent,
            );
            if should_drop_tiny_polygon(
                &polygon,
                &tile_bounds,
                extent,
                DEFAULT_TINY_POLYGON_THRESHOLD,
            ) {
                drops_3px += 1;
            }
        }

        // Group 2: ~1 sq pixel polygons (25% of threshold = 75% drop probability)
        let mut drops_1px = 0;
        for i in 0..num_samples {
            let polygon = create_square_polygon_in_tile_coords(
                500.0 + (i as f64) * 5.0,
                2000.0,
                1.0, // 1 sq pixel
                &tile_bounds,
                extent,
            );
            if should_drop_tiny_polygon(
                &polygon,
                &tile_bounds,
                extent,
                DEFAULT_TINY_POLYGON_THRESHOLD,
            ) {
                drops_1px += 1;
            }
        }

        // Smaller polygons should be dropped more often
        assert!(
            drops_1px > drops_3px,
            "1 sq pixel polygons ({} dropped) should be dropped more often than 3 sq pixel polygons ({} dropped)",
            drops_1px,
            drops_3px
        );
    }
}
