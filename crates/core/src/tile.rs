//! Tile coordinate math and utilities
//!
//! This module provides functions for converting between geographic coordinates (lat/lng)
//! and tile coordinates (x/y/z) using Web Mercator projection.

use std::f64::consts::PI;

/// Tile coordinates: x, y, and zoom level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord {
    pub x: u32,
    pub y: u32,
    pub z: u8,
}

impl TileCoord {
    /// Create a new tile coordinate
    pub fn new(x: u32, y: u32, z: u8) -> Self {
        Self { x, y, z }
    }

    /// Get the bounding box of this tile in geographic coordinates (lng/lat)
    pub fn bounds(&self) -> TileBounds {
        let n = 2_f64.powi(self.z as i32);
        let lng_min = (self.x as f64) / n * 360.0 - 180.0;
        let lng_max = (self.x as f64 + 1.0) / n * 360.0 - 180.0;

        let lat_rad = |y: f64| {
            let y_rad = PI * (1.0 - 2.0 * y / n);
            y_rad.sinh().atan().to_degrees()
        };

        let lat_max = lat_rad(self.y as f64);
        let lat_min = lat_rad(self.y as f64 + 1.0);

        TileBounds {
            lng_min,
            lat_min,
            lng_max,
            lat_max,
        }
    }
}

/// Geographic bounding box
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileBounds {
    pub lng_min: f64,
    pub lat_min: f64,
    pub lng_max: f64,
    pub lat_max: f64,
}

impl TileBounds {
    /// Create a new bounding box
    pub fn new(lng_min: f64, lat_min: f64, lng_max: f64, lat_max: f64) -> Self {
        Self {
            lng_min,
            lat_min,
            lng_max,
            lat_max,
        }
    }

    /// Create an empty/invalid bounding box
    pub fn empty() -> Self {
        Self {
            lng_min: f64::INFINITY,
            lat_min: f64::INFINITY,
            lng_max: f64::NEG_INFINITY,
            lat_max: f64::NEG_INFINITY,
        }
    }

    /// Check if this is a valid bounding box
    pub fn is_valid(&self) -> bool {
        self.lng_min <= self.lng_max && self.lat_min <= self.lat_max
    }

    /// Expand this bounding box to include another
    pub fn expand(&mut self, other: &Self) {
        self.lng_min = self.lng_min.min(other.lng_min);
        self.lat_min = self.lat_min.min(other.lat_min);
        self.lng_max = self.lng_max.max(other.lng_max);
        self.lat_max = self.lat_max.max(other.lat_max);
    }

    /// Get the width in degrees
    pub fn width(&self) -> f64 {
        self.lng_max - self.lng_min
    }

    /// Get the height in degrees
    pub fn height(&self) -> f64 {
        self.lat_max - self.lat_min
    }
}

/// Convert longitude/latitude to tile coordinates at a given zoom level
///
/// Uses Web Mercator projection (EPSG:3857)
///
/// # Arguments
///
/// * `lng` - Longitude in degrees (-180 to 180)
/// * `lat` - Latitude in degrees (-85.0511 to 85.0511, Web Mercator bounds)
/// * `zoom` - Zoom level (0-30)
///
/// # Returns
///
/// TileCoord with x, y, and zoom
pub fn lng_lat_to_tile(lng: f64, lat: f64, zoom: u8) -> TileCoord {
    let n = 2_f64.powi(zoom as i32);

    // Convert longitude to tile x
    let x = ((lng + 180.0) / 360.0 * n).floor() as u32;

    // Convert latitude to tile y (Web Mercator)
    let lat_rad = lat.to_radians();
    let y = ((1.0 - lat_rad.tan().asinh() / PI) / 2.0 * n).floor() as u32;

    TileCoord::new(x, y, zoom)
}

/// Get all tiles that intersect a geographic bounding box at a given zoom level
///
/// # Arguments
///
/// * `bbox` - Geographic bounding box
/// * `zoom` - Zoom level
///
/// # Returns
///
/// Iterator of TileCoord that intersect the bbox
pub fn tiles_for_bbox(bbox: &TileBounds, zoom: u8) -> impl Iterator<Item = TileCoord> {
    // Get corner tiles
    let min_tile = lng_lat_to_tile(bbox.lng_min, bbox.lat_max, zoom); // Note: lat_max for min_y
    let max_tile = lng_lat_to_tile(bbox.lng_max, bbox.lat_min, zoom); // Note: lat_min for max_y

    // Iterate over all tiles in the range
    (min_tile.y..=max_tile.y).flat_map(move |y| {
        (min_tile.x..=max_tile.x).map(move |x| TileCoord::new(x, y, zoom))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lng_lat_to_tile_origin() {
        // Origin (null island: 0, 0) at zoom 0
        let tile = lng_lat_to_tile(0.0, 0.0, 0);
        assert_eq!(tile, TileCoord::new(0, 0, 0));
    }

    #[test]
    fn test_lng_lat_to_tile_zoom_1() {
        // Test various points at zoom 1
        let tile = lng_lat_to_tile(0.0, 0.0, 1);
        assert_eq!(tile.x, 1);
        assert_eq!(tile.y, 1);
        assert_eq!(tile.z, 1);

        // Top-left quadrant
        let tile = lng_lat_to_tile(-90.0, 45.0, 1);
        assert_eq!(tile.x, 0);

        // Top-right quadrant
        let tile = lng_lat_to_tile(90.0, 45.0, 1);
        assert_eq!(tile.x, 1);
    }

    #[test]
    fn test_tile_bounds() {
        // Tile 0,0,0 should cover the whole world
        let tile = TileCoord::new(0, 0, 0);
        let bounds = tile.bounds();

        assert!((bounds.lng_min - (-180.0)).abs() < 0.0001);
        assert!((bounds.lng_max - 180.0).abs() < 0.0001);
        // Lat bounds are Web Mercator limits (~85.05 degrees)
        assert!(bounds.lat_min < -85.0);
        assert!(bounds.lat_max > 85.0);
    }

    #[test]
    fn test_tiles_for_bbox_single_tile() {
        // Small bbox that fits in one tile
        let bbox = TileBounds::new(-1.0, -1.0, 1.0, 1.0);
        let tiles: Vec<_> = tiles_for_bbox(&bbox, 10).collect();

        // Should be at least 1 tile
        assert!(!tiles.is_empty());

        // All tiles should be at zoom 10
        for tile in &tiles {
            assert_eq!(tile.z, 10);
        }
    }

    #[test]
    fn test_tiles_for_bbox_multiple_tiles() {
        // Larger bbox spanning multiple tiles at zoom 5
        let bbox = TileBounds::new(-10.0, -10.0, 10.0, 10.0);
        let tiles: Vec<_> = tiles_for_bbox(&bbox, 5).collect();

        // Should cover multiple tiles
        assert!(tiles.len() > 1);

        // Check bounds are reasonable
        let first = tiles.first().unwrap();
        let last = tiles.last().unwrap();
        assert!(first.x <= last.x);
        assert!(first.y <= last.y);
    }

    #[test]
    fn test_bbox_expand() {
        let mut bbox1 = TileBounds::new(-10.0, -10.0, 10.0, 10.0);
        let bbox2 = TileBounds::new(-20.0, -5.0, 5.0, 15.0);

        bbox1.expand(&bbox2);

        assert_eq!(bbox1.lng_min, -20.0);
        assert_eq!(bbox1.lat_min, -10.0);
        assert_eq!(bbox1.lng_max, 10.0);
        assert_eq!(bbox1.lat_max, 15.0);
    }

    #[test]
    fn test_bbox_empty() {
        let bbox = TileBounds::empty();
        assert!(!bbox.is_valid());

        let mut bbox = TileBounds::empty();
        bbox.expand(&TileBounds::new(-10.0, -10.0, 10.0, 10.0));
        assert!(bbox.is_valid());
        assert_eq!(bbox.lng_min, -10.0);
    }

    #[test]
    fn test_tile_coord_round_trip() {
        // For various zooms, check that a tile's center converts back to the same tile
        for zoom in 0..=14 {
            // Use valid tile coordinates for each zoom (max tile = 2^zoom - 1)
            let max_coord = 2_u32.pow(zoom as u32) - 1;
            let x = max_coord.min(100);
            let y = max_coord.min(200);

            let tile = TileCoord::new(x, y, zoom);
            let bounds = tile.bounds();

            let center_lng = (bounds.lng_min + bounds.lng_max) / 2.0;
            let center_lat = (bounds.lat_min + bounds.lat_max) / 2.0;

            let tile_back = lng_lat_to_tile(center_lng, center_lat, zoom);

            assert_eq!(tile, tile_back, "Round-trip failed at zoom {}", zoom);
        }
    }
}
