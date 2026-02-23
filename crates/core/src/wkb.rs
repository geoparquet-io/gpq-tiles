//! WKB serialization utilities for temporary file storage.
//!
//! This module provides functions to serialize `geo::Geometry` to Well-Known Binary (WKB)
//! format and back. This is used for streaming pipelines that need to spill features
//! to disk when memory pressure is high.
//!
//! # Note on Usage
//!
//! This module is intended for **temp file storage only**, not for bulk geometry extraction
//! from GeoParquet files. For reading geometries from Parquet, use GeoArrow's columnar
//! decoding which provides better performance (see `batch_processor.rs`).
//!
//! # Examples
//!
//! ```
//! use geo::{Geometry, Point, point};
//! use gpq_tiles_core::wkb::{geometry_to_wkb, wkb_to_geometry};
//!
//! let point = Geometry::Point(point!(x: 1.5, y: 2.5));
//! let wkb_bytes = geometry_to_wkb(&point).unwrap();
//! let restored = wkb_to_geometry(&wkb_bytes).unwrap();
//!
//! // Round-trip preserves geometry
//! assert!(matches!(restored, Geometry::Point(_)));
//! ```

use geo::Geometry;
use geozero::wkb::Wkb;
use geozero::{CoordDimensions, ToGeo, ToWkb};
use std::collections::HashMap;

/// Errors that can occur during WKB serialization/deserialization.
#[derive(Debug, thiserror::Error)]
pub enum WkbError {
    #[error("WKB encode error: {0}")]
    EncodeError(String),

    #[error("WKB decode error: {0}")]
    DecodeError(String),

    #[error("Property serialization error: {0}")]
    PropertySerializeError(String),

    #[error("Property deserialization error: {0}")]
    PropertyDeserializeError(String),
}

pub type Result<T> = std::result::Result<T, WkbError>;

/// Serialize a geometry to WKB bytes.
///
/// Uses standard OGC WKB format with XY coordinates (no Z or M dimensions).
///
/// # Arguments
/// * `geom` - The geometry to serialize
///
/// # Returns
/// WKB-encoded bytes on success, or an error if encoding fails.
///
/// # Example
/// ```
/// use geo::{Geometry, LineString, line_string};
/// use gpq_tiles_core::wkb::geometry_to_wkb;
///
/// let line = Geometry::LineString(line_string![(x: 0.0, y: 0.0), (x: 1.0, y: 1.0)]);
/// let wkb = geometry_to_wkb(&line).unwrap();
/// assert!(!wkb.is_empty());
/// ```
pub fn geometry_to_wkb(geom: &Geometry) -> Result<Vec<u8>> {
    geom.to_wkb(CoordDimensions::xy())
        .map_err(|e| WkbError::EncodeError(e.to_string()))
}

/// Deserialize WKB bytes back to a geometry.
///
/// Handles standard OGC WKB format.
///
/// # Arguments
/// * `wkb` - The WKB-encoded bytes
///
/// # Returns
/// The deserialized geometry on success, or an error if decoding fails.
///
/// # Example
/// ```
/// use geo::{Geometry, Point, point};
/// use gpq_tiles_core::wkb::{geometry_to_wkb, wkb_to_geometry};
///
/// let original = Geometry::Point(point!(x: 42.0, y: -73.5));
/// let wkb = geometry_to_wkb(&original).unwrap();
/// let restored = wkb_to_geometry(&wkb).unwrap();
/// ```
pub fn wkb_to_geometry(wkb: &[u8]) -> Result<Geometry> {
    Wkb(wkb.to_vec())
        .to_geo()
        .map_err(|e| WkbError::DecodeError(e.to_string()))
}

/// Property value type for feature attributes.
///
/// Matches the MVT property types supported by the tile format.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum PropertyValue {
    String(String),
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
}

/// Serialize feature properties to MessagePack bytes.
///
/// MessagePack is more compact than JSON (~30% smaller) and faster to
/// serialize/deserialize, making it ideal for temp file storage.
///
/// # Arguments
/// * `props` - Map of property names to values
///
/// # Returns
/// MessagePack-encoded bytes on success, or an error if serialization fails.
pub fn serialize_properties(props: &HashMap<String, PropertyValue>) -> Result<Vec<u8>> {
    rmp_serde::to_vec(props).map_err(|e| WkbError::PropertySerializeError(e.to_string()))
}

/// Deserialize MessagePack bytes back to feature properties.
///
/// # Arguments
/// * `data` - The MessagePack-encoded bytes
///
/// # Returns
/// The property map on success, or an error if deserialization fails.
pub fn deserialize_properties(data: &[u8]) -> Result<HashMap<String, PropertyValue>> {
    rmp_serde::from_slice(data).map_err(|e| WkbError::PropertyDeserializeError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{line_string, point, polygon, Coord, LineString, MultiPolygon, Point, Polygon};

    // ========================================================================
    // Geometry Round-Trip Tests
    // ========================================================================

    #[test]
    fn test_point_round_trip() {
        let original = Geometry::Point(point!(x: 1.5, y: 2.5));
        let wkb = geometry_to_wkb(&original).expect("encode should succeed");
        let restored = wkb_to_geometry(&wkb).expect("decode should succeed");

        match restored {
            Geometry::Point(p) => {
                assert!((p.x() - 1.5).abs() < 1e-10);
                assert!((p.y() - 2.5).abs() < 1e-10);
            }
            _ => panic!("Expected Point, got {:?}", restored),
        }
    }

    #[test]
    fn test_linestring_round_trip() {
        let original = Geometry::LineString(line_string![
            (x: 0.0, y: 0.0),
            (x: 1.0, y: 1.0),
            (x: 2.0, y: 0.0)
        ]);
        let wkb = geometry_to_wkb(&original).expect("encode should succeed");
        let restored = wkb_to_geometry(&wkb).expect("decode should succeed");

        match restored {
            Geometry::LineString(ls) => {
                assert_eq!(ls.0.len(), 3);
                assert!((ls.0[0].x - 0.0).abs() < 1e-10);
                assert!((ls.0[1].x - 1.0).abs() < 1e-10);
                assert!((ls.0[2].x - 2.0).abs() < 1e-10);
            }
            _ => panic!("Expected LineString, got {:?}", restored),
        }
    }

    #[test]
    fn test_polygon_round_trip() {
        let original = Geometry::Polygon(polygon![
            (x: 0.0, y: 0.0),
            (x: 4.0, y: 0.0),
            (x: 4.0, y: 4.0),
            (x: 0.0, y: 4.0),
            (x: 0.0, y: 0.0)
        ]);
        let wkb = geometry_to_wkb(&original).expect("encode should succeed");
        let restored = wkb_to_geometry(&wkb).expect("decode should succeed");

        match restored {
            Geometry::Polygon(poly) => {
                assert_eq!(poly.exterior().0.len(), 5);
                assert!(poly.interiors().is_empty());
            }
            _ => panic!("Expected Polygon, got {:?}", restored),
        }
    }

    #[test]
    fn test_polygon_with_hole_round_trip() {
        // Exterior ring
        let exterior = LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 10.0, y: 0.0 },
            Coord { x: 10.0, y: 10.0 },
            Coord { x: 0.0, y: 10.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);
        // Interior hole
        let hole = LineString::from(vec![
            Coord { x: 2.0, y: 2.0 },
            Coord { x: 8.0, y: 2.0 },
            Coord { x: 8.0, y: 8.0 },
            Coord { x: 2.0, y: 8.0 },
            Coord { x: 2.0, y: 2.0 },
        ]);
        let original = Geometry::Polygon(Polygon::new(exterior, vec![hole]));

        let wkb = geometry_to_wkb(&original).expect("encode should succeed");
        let restored = wkb_to_geometry(&wkb).expect("decode should succeed");

        match restored {
            Geometry::Polygon(poly) => {
                assert_eq!(poly.exterior().0.len(), 5);
                assert_eq!(poly.interiors().len(), 1);
                assert_eq!(poly.interiors()[0].0.len(), 5);
            }
            _ => panic!("Expected Polygon, got {:?}", restored),
        }
    }

    #[test]
    fn test_multipolygon_round_trip() {
        let poly1 = polygon![
            (x: 0.0, y: 0.0),
            (x: 1.0, y: 0.0),
            (x: 1.0, y: 1.0),
            (x: 0.0, y: 1.0),
            (x: 0.0, y: 0.0)
        ];
        let poly2 = polygon![
            (x: 5.0, y: 5.0),
            (x: 6.0, y: 5.0),
            (x: 6.0, y: 6.0),
            (x: 5.0, y: 6.0),
            (x: 5.0, y: 5.0)
        ];
        let original = Geometry::MultiPolygon(MultiPolygon::new(vec![poly1, poly2]));

        let wkb = geometry_to_wkb(&original).expect("encode should succeed");
        let restored = wkb_to_geometry(&wkb).expect("decode should succeed");

        match restored {
            Geometry::MultiPolygon(mp) => {
                assert_eq!(mp.0.len(), 2);
            }
            _ => panic!("Expected MultiPolygon, got {:?}", restored),
        }
    }

    #[test]
    fn test_multipoint_round_trip() {
        use geo::MultiPoint;

        let points = vec![
            Point::new(1.0, 2.0),
            Point::new(3.0, 4.0),
            Point::new(5.0, 6.0),
        ];
        let original = Geometry::MultiPoint(MultiPoint::new(points));

        let wkb = geometry_to_wkb(&original).expect("encode should succeed");
        let restored = wkb_to_geometry(&wkb).expect("decode should succeed");

        match restored {
            Geometry::MultiPoint(mp) => {
                assert_eq!(mp.0.len(), 3);
            }
            _ => panic!("Expected MultiPoint, got {:?}", restored),
        }
    }

    #[test]
    fn test_multilinestring_round_trip() {
        use geo::MultiLineString;

        let lines = vec![
            line_string![(x: 0.0, y: 0.0), (x: 1.0, y: 1.0)],
            line_string![(x: 2.0, y: 2.0), (x: 3.0, y: 3.0)],
        ];
        let original = Geometry::MultiLineString(MultiLineString::new(lines));

        let wkb = geometry_to_wkb(&original).expect("encode should succeed");
        let restored = wkb_to_geometry(&wkb).expect("decode should succeed");

        match restored {
            Geometry::MultiLineString(mls) => {
                assert_eq!(mls.0.len(), 2);
            }
            _ => panic!("Expected MultiLineString, got {:?}", restored),
        }
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_point_at_origin() {
        let original = Geometry::Point(point!(x: 0.0, y: 0.0));
        let wkb = geometry_to_wkb(&original).expect("encode should succeed");
        let restored = wkb_to_geometry(&wkb).expect("decode should succeed");

        match restored {
            Geometry::Point(p) => {
                assert!((p.x() - 0.0).abs() < 1e-10);
                assert!((p.y() - 0.0).abs() < 1e-10);
            }
            _ => panic!("Expected Point"),
        }
    }

    #[test]
    fn test_point_extreme_coordinates() {
        // Test with coordinates at the edge of typical geographic bounds
        let original = Geometry::Point(point!(x: 180.0, y: -90.0));
        let wkb = geometry_to_wkb(&original).expect("encode should succeed");
        let restored = wkb_to_geometry(&wkb).expect("decode should succeed");

        match restored {
            Geometry::Point(p) => {
                assert!((p.x() - 180.0).abs() < 1e-10);
                assert!((p.y() - (-90.0)).abs() < 1e-10);
            }
            _ => panic!("Expected Point"),
        }
    }

    #[test]
    fn test_decode_invalid_wkb() {
        let invalid_bytes = vec![0x00, 0x01, 0x02, 0x03];
        let result = wkb_to_geometry(&invalid_bytes);
        assert!(result.is_err());
        assert!(matches!(result, Err(WkbError::DecodeError(_))));
    }

    #[test]
    fn test_decode_empty_bytes() {
        let result = wkb_to_geometry(&[]);
        assert!(result.is_err());
    }

    // ========================================================================
    // Property Serialization Tests
    // ========================================================================

    #[test]
    fn test_properties_round_trip_string() {
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Test".to_string()),
        );
        props.insert(
            "description".to_string(),
            PropertyValue::String("A test feature".to_string()),
        );

        let bytes = serialize_properties(&props).expect("serialize should succeed");
        let restored = deserialize_properties(&bytes).expect("deserialize should succeed");

        assert_eq!(props, restored);
    }

    #[test]
    fn test_properties_round_trip_numbers() {
        let mut props = HashMap::new();
        props.insert("int_val".to_string(), PropertyValue::Int(-42));
        props.insert("uint_val".to_string(), PropertyValue::UInt(42));
        props.insert("float_val".to_string(), PropertyValue::Float(3.14159));

        let bytes = serialize_properties(&props).expect("serialize should succeed");
        let restored = deserialize_properties(&bytes).expect("deserialize should succeed");

        assert_eq!(props, restored);
    }

    #[test]
    fn test_properties_round_trip_bool() {
        let mut props = HashMap::new();
        props.insert("active".to_string(), PropertyValue::Bool(true));
        props.insert("deleted".to_string(), PropertyValue::Bool(false));

        let bytes = serialize_properties(&props).expect("serialize should succeed");
        let restored = deserialize_properties(&bytes).expect("deserialize should succeed");

        assert_eq!(props, restored);
    }

    #[test]
    fn test_properties_round_trip_mixed() {
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("Building".to_string()),
        );
        props.insert("floors".to_string(), PropertyValue::Int(5));
        props.insert("area".to_string(), PropertyValue::Float(1234.56));
        props.insert("commercial".to_string(), PropertyValue::Bool(true));
        props.insert("id".to_string(), PropertyValue::UInt(999));

        let bytes = serialize_properties(&props).expect("serialize should succeed");
        let restored = deserialize_properties(&bytes).expect("deserialize should succeed");

        assert_eq!(props, restored);
    }

    #[test]
    fn test_properties_empty() {
        let props: HashMap<String, PropertyValue> = HashMap::new();

        let bytes = serialize_properties(&props).expect("serialize should succeed");
        let restored = deserialize_properties(&bytes).expect("deserialize should succeed");

        assert!(restored.is_empty());
    }

    #[test]
    fn test_properties_unicode() {
        let mut props = HashMap::new();
        props.insert(
            "name".to_string(),
            PropertyValue::String("東京タワー".to_string()),
        );
        props.insert(
            "emoji".to_string(),
            PropertyValue::String("🗼🌏".to_string()),
        );

        let bytes = serialize_properties(&props).expect("serialize should succeed");
        let restored = deserialize_properties(&bytes).expect("deserialize should succeed");

        assert_eq!(props, restored);
    }

    #[test]
    fn test_deserialize_invalid_msgpack() {
        let invalid_bytes = vec![0xFF, 0xFF, 0xFF];
        let result = deserialize_properties(&invalid_bytes);
        assert!(result.is_err());
        assert!(matches!(result, Err(WkbError::PropertyDeserializeError(_))));
    }
}
