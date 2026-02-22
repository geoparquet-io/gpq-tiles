//! Geometry validation for post-simplification degenerate geometry detection.
//!
//! After simplification, geometries can become degenerate (invalid for MVT encoding):
//! - Polygons with fewer than 4 points (need 3 unique + closing point)
//! - LineStrings with fewer than 2 points
//! - Zero-area polygons (all points collinear or coincident)
//! - Empty geometries
//!
//! # Tippecanoe Behavior
//!
//! Tippecanoe silently drops degenerate geometries rather than attempting repair.
//! This is the approach we follow - simple filtering is preferred over complex repair logic.
//!
//! # Usage
//!
//! ```
//! use gpq_tiles_core::validate::is_valid_geometry;
//! use geo::{Geometry, LineString, Coord};
//!
//! let line = LineString::new(vec![
//!     Coord { x: 0.0, y: 0.0 },
//!     Coord { x: 1.0, y: 1.0 },
//! ]);
//! assert!(is_valid_geometry(&Geometry::LineString(line)));
//! ```

use geo::{Area, Geometry, LineString, MultiLineString, MultiPolygon, Polygon};

/// Minimum number of points for a valid polygon ring (3 unique + closing = 4)
pub const MIN_POLYGON_RING_POINTS: usize = 4;

/// Minimum number of points for a valid linestring
pub const MIN_LINESTRING_POINTS: usize = 2;

/// Minimum area threshold for polygons (in tile coordinates squared)
/// Polygons with area smaller than this are considered degenerate.
/// This is separate from the "tiny polygon" dropping which uses diffuse probability.
pub const MIN_POLYGON_AREA: f64 = 1e-10;

/// Result of geometry validation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationResult {
    /// Geometry is valid and can be encoded
    Valid,
    /// Geometry is invalid and should be dropped
    Invalid(InvalidReason),
}

/// Reason why a geometry is invalid
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidReason {
    /// Polygon ring has fewer than 4 points
    PolygonTooFewPoints {
        ring_index: usize,
        point_count: usize,
    },
    /// LineString has fewer than 2 points
    LineStringTooFewPoints { point_count: usize },
    /// Polygon has zero or near-zero area (degenerate)
    ZeroAreaPolygon,
    /// Geometry is empty (no coordinates)
    EmptyGeometry,
    /// MultiPolygon has no valid polygons after filtering
    NoValidPolygons,
    /// MultiLineString has no valid linestrings after filtering
    NoValidLineStrings,
}

impl ValidationResult {
    /// Returns true if the geometry is valid
    pub fn is_valid(&self) -> bool {
        matches!(self, ValidationResult::Valid)
    }

    /// Returns true if the geometry is invalid
    pub fn is_invalid(&self) -> bool {
        matches!(self, ValidationResult::Invalid(_))
    }
}

/// Check if a geometry is valid for MVT encoding.
///
/// Returns `true` if the geometry is valid, `false` if it should be dropped.
///
/// # Arguments
/// * `geom` - The geometry to validate
///
/// # Returns
/// `true` if valid, `false` if degenerate/invalid
pub fn is_valid_geometry(geom: &Geometry<f64>) -> bool {
    validate_geometry(geom).is_valid()
}

/// Validate a geometry and return detailed validation result.
///
/// # Arguments
/// * `geom` - The geometry to validate
///
/// # Returns
/// `ValidationResult::Valid` if the geometry can be encoded,
/// `ValidationResult::Invalid(reason)` if it should be dropped
pub fn validate_geometry(geom: &Geometry<f64>) -> ValidationResult {
    match geom {
        Geometry::Point(_) => ValidationResult::Valid, // Points are always valid
        Geometry::MultiPoint(mp) => {
            if mp.0.is_empty() {
                ValidationResult::Invalid(InvalidReason::EmptyGeometry)
            } else {
                ValidationResult::Valid
            }
        }
        Geometry::LineString(ls) => validate_linestring(ls),
        Geometry::MultiLineString(mls) => validate_multi_linestring(mls),
        Geometry::Polygon(poly) => validate_polygon(poly),
        Geometry::MultiPolygon(mp) => validate_multi_polygon(mp),
        // GeometryCollection and other types - pass through as valid
        // (they may contain valid sub-geometries)
        _ => ValidationResult::Valid,
    }
}

/// Validate a LineString geometry.
pub fn validate_linestring(ls: &LineString<f64>) -> ValidationResult {
    let point_count = ls.0.len();
    if point_count < MIN_LINESTRING_POINTS {
        ValidationResult::Invalid(InvalidReason::LineStringTooFewPoints { point_count })
    } else {
        ValidationResult::Valid
    }
}

/// Validate a MultiLineString geometry.
pub fn validate_multi_linestring(mls: &MultiLineString<f64>) -> ValidationResult {
    if mls.0.is_empty() {
        return ValidationResult::Invalid(InvalidReason::EmptyGeometry);
    }

    // Check if at least one linestring is valid
    let has_valid = mls.0.iter().any(|ls| validate_linestring(ls).is_valid());

    if has_valid {
        ValidationResult::Valid
    } else {
        ValidationResult::Invalid(InvalidReason::NoValidLineStrings)
    }
}

/// Validate a Polygon geometry.
pub fn validate_polygon(poly: &Polygon<f64>) -> ValidationResult {
    // Check exterior ring has enough points
    let exterior_count = poly.exterior().0.len();
    if exterior_count < MIN_POLYGON_RING_POINTS {
        return ValidationResult::Invalid(InvalidReason::PolygonTooFewPoints {
            ring_index: 0,
            point_count: exterior_count,
        });
    }

    // Check interior rings have enough points
    for (idx, interior) in poly.interiors().iter().enumerate() {
        let interior_count = interior.0.len();
        if interior_count < MIN_POLYGON_RING_POINTS {
            return ValidationResult::Invalid(InvalidReason::PolygonTooFewPoints {
                ring_index: idx + 1, // +1 because exterior is ring 0
                point_count: interior_count,
            });
        }
    }

    // Check for zero-area polygon
    let area = poly.unsigned_area();
    if area < MIN_POLYGON_AREA {
        return ValidationResult::Invalid(InvalidReason::ZeroAreaPolygon);
    }

    ValidationResult::Valid
}

/// Validate a MultiPolygon geometry.
pub fn validate_multi_polygon(mp: &MultiPolygon<f64>) -> ValidationResult {
    if mp.0.is_empty() {
        return ValidationResult::Invalid(InvalidReason::EmptyGeometry);
    }

    // Check if at least one polygon is valid
    let has_valid = mp.0.iter().any(|poly| validate_polygon(poly).is_valid());

    if has_valid {
        ValidationResult::Valid
    } else {
        ValidationResult::Invalid(InvalidReason::NoValidPolygons)
    }
}

/// Filter a geometry, returning `Some(geometry)` if valid, `None` if invalid.
///
/// For multi-geometries, this filters out invalid components and returns
/// a new geometry with only the valid parts.
///
/// # Arguments
/// * `geom` - The geometry to filter
///
/// # Returns
/// `Some(geometry)` if the geometry (or part of it) is valid, `None` if completely invalid
pub fn filter_valid_geometry(geom: &Geometry<f64>) -> Option<Geometry<f64>> {
    match geom {
        Geometry::Point(_) => Some(geom.clone()),
        Geometry::MultiPoint(mp) => {
            if mp.0.is_empty() {
                None
            } else {
                Some(geom.clone())
            }
        }
        Geometry::LineString(ls) => {
            if validate_linestring(ls).is_valid() {
                Some(geom.clone())
            } else {
                None
            }
        }
        Geometry::MultiLineString(mls) => filter_multi_linestring(mls),
        Geometry::Polygon(poly) => {
            if validate_polygon(poly).is_valid() {
                Some(geom.clone())
            } else {
                None
            }
        }
        Geometry::MultiPolygon(mp) => filter_multi_polygon(mp),
        // Other types pass through unchanged
        other => Some(other.clone()),
    }
}

/// Filter a MultiLineString, keeping only valid linestrings.
fn filter_multi_linestring(mls: &MultiLineString<f64>) -> Option<Geometry<f64>> {
    let valid_lines: Vec<LineString<f64>> = mls
        .0
        .iter()
        .filter(|ls| validate_linestring(ls).is_valid())
        .cloned()
        .collect();

    if valid_lines.is_empty() {
        None
    } else if valid_lines.len() == 1 {
        // Downgrade to single LineString if only one remains
        Some(Geometry::LineString(
            valid_lines.into_iter().next().unwrap(),
        ))
    } else {
        Some(Geometry::MultiLineString(MultiLineString::new(valid_lines)))
    }
}

/// Filter a MultiPolygon, keeping only valid polygons.
fn filter_multi_polygon(mp: &MultiPolygon<f64>) -> Option<Geometry<f64>> {
    let valid_polygons: Vec<Polygon<f64>> =
        mp.0.iter()
            .filter(|poly| validate_polygon(poly).is_valid())
            .cloned()
            .collect();

    if valid_polygons.is_empty() {
        None
    } else if valid_polygons.len() == 1 {
        // Downgrade to single Polygon if only one remains
        Some(Geometry::Polygon(
            valid_polygons.into_iter().next().unwrap(),
        ))
    } else {
        Some(Geometry::MultiPolygon(MultiPolygon::new(valid_polygons)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{Coord, MultiPoint, Point};

    // =========================================================================
    // HELPER FUNCTIONS
    // =========================================================================

    fn make_linestring(coords: &[(f64, f64)]) -> LineString<f64> {
        LineString::new(coords.iter().map(|&(x, y)| Coord { x, y }).collect())
    }

    fn make_polygon(exterior: &[(f64, f64)]) -> Polygon<f64> {
        Polygon::new(make_linestring(exterior), vec![])
    }

    fn make_polygon_with_hole(exterior: &[(f64, f64)], hole: &[(f64, f64)]) -> Polygon<f64> {
        Polygon::new(make_linestring(exterior), vec![make_linestring(hole)])
    }

    // =========================================================================
    // POINT TESTS
    // =========================================================================

    #[test]
    fn test_point_always_valid() {
        let point = Geometry::Point(Point::new(0.0, 0.0));
        assert!(is_valid_geometry(&point));
        assert_eq!(validate_geometry(&point), ValidationResult::Valid);
    }

    #[test]
    fn test_multipoint_valid() {
        let mp = Geometry::MultiPoint(geo::MultiPoint::new(vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
        ]));
        assert!(is_valid_geometry(&mp));
    }

    #[test]
    fn test_multipoint_empty_invalid() {
        let mp = Geometry::MultiPoint(MultiPoint::new(vec![]));
        assert!(!is_valid_geometry(&mp));
        assert_eq!(
            validate_geometry(&mp),
            ValidationResult::Invalid(InvalidReason::EmptyGeometry)
        );
    }

    // =========================================================================
    // LINESTRING TESTS
    // =========================================================================

    #[test]
    fn test_linestring_valid_two_points() {
        let ls = Geometry::LineString(make_linestring(&[(0.0, 0.0), (1.0, 1.0)]));
        assert!(is_valid_geometry(&ls));
    }

    #[test]
    fn test_linestring_valid_many_points() {
        let ls = Geometry::LineString(make_linestring(&[
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]));
        assert!(is_valid_geometry(&ls));
    }

    #[test]
    fn test_linestring_invalid_one_point() {
        let ls = Geometry::LineString(make_linestring(&[(0.0, 0.0)]));
        assert!(!is_valid_geometry(&ls));
        assert_eq!(
            validate_geometry(&ls),
            ValidationResult::Invalid(InvalidReason::LineStringTooFewPoints { point_count: 1 })
        );
    }

    #[test]
    fn test_linestring_invalid_empty() {
        let ls = Geometry::LineString(make_linestring(&[]));
        assert!(!is_valid_geometry(&ls));
        assert_eq!(
            validate_geometry(&ls),
            ValidationResult::Invalid(InvalidReason::LineStringTooFewPoints { point_count: 0 })
        );
    }

    // =========================================================================
    // MULTILINESTRING TESTS
    // =========================================================================

    #[test]
    fn test_multilinestring_valid() {
        let mls = Geometry::MultiLineString(MultiLineString::new(vec![
            make_linestring(&[(0.0, 0.0), (1.0, 1.0)]),
            make_linestring(&[(2.0, 2.0), (3.0, 3.0)]),
        ]));
        assert!(is_valid_geometry(&mls));
    }

    #[test]
    fn test_multilinestring_empty_invalid() {
        let mls = Geometry::MultiLineString(MultiLineString::new(vec![]));
        assert!(!is_valid_geometry(&mls));
        assert_eq!(
            validate_geometry(&mls),
            ValidationResult::Invalid(InvalidReason::EmptyGeometry)
        );
    }

    #[test]
    fn test_multilinestring_with_one_valid_line() {
        // One valid, one invalid - should be valid overall
        let mls = Geometry::MultiLineString(MultiLineString::new(vec![
            make_linestring(&[(0.0, 0.0), (1.0, 1.0)]), // valid
            make_linestring(&[(2.0, 2.0)]),             // invalid (1 point)
        ]));
        assert!(is_valid_geometry(&mls));
    }

    #[test]
    fn test_multilinestring_all_invalid_lines() {
        let mls = Geometry::MultiLineString(MultiLineString::new(vec![
            make_linestring(&[(0.0, 0.0)]), // invalid
            make_linestring(&[(1.0, 1.0)]), // invalid
        ]));
        assert!(!is_valid_geometry(&mls));
        assert_eq!(
            validate_geometry(&mls),
            ValidationResult::Invalid(InvalidReason::NoValidLineStrings)
        );
    }

    // =========================================================================
    // POLYGON TESTS
    // =========================================================================

    #[test]
    fn test_polygon_valid_triangle() {
        // Triangle: 3 unique points + closing = 4 points
        let poly = Geometry::Polygon(make_polygon(&[
            (0.0, 0.0),
            (1.0, 0.0),
            (0.5, 1.0),
            (0.0, 0.0), // closing point
        ]));
        assert!(is_valid_geometry(&poly));
    }

    #[test]
    fn test_polygon_valid_square() {
        let poly = Geometry::Polygon(make_polygon(&[
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.0, 0.0), // closing point
        ]));
        assert!(is_valid_geometry(&poly));
    }

    #[test]
    fn test_polygon_invalid_too_few_points() {
        // Only 3 points (2 unique + closing) - not enough for a polygon
        let poly = Geometry::Polygon(make_polygon(&[
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 0.0), // closing point
        ]));
        assert!(!is_valid_geometry(&poly));
        assert_eq!(
            validate_geometry(&poly),
            ValidationResult::Invalid(InvalidReason::PolygonTooFewPoints {
                ring_index: 0,
                point_count: 3
            })
        );
    }

    #[test]
    fn test_polygon_invalid_two_points() {
        // Note: geo's LineString automatically closes the ring, so 2 input points
        // become 3 points (2 unique + closing). Still invalid for a polygon.
        let poly = Geometry::Polygon(make_polygon(&[(0.0, 0.0), (1.0, 0.0)]));
        assert!(!is_valid_geometry(&poly));
        // The ring will have 3 points after closing, but still invalid
        if let ValidationResult::Invalid(InvalidReason::PolygonTooFewPoints {
            ring_index,
            point_count,
        }) = validate_geometry(&poly)
        {
            assert_eq!(ring_index, 0);
            assert!(
                point_count < MIN_POLYGON_RING_POINTS,
                "Expected fewer than {} points, got {}",
                MIN_POLYGON_RING_POINTS,
                point_count
            );
        } else {
            panic!("Expected PolygonTooFewPoints error");
        }
    }

    #[test]
    fn test_polygon_invalid_empty() {
        let poly = Geometry::Polygon(make_polygon(&[]));
        assert!(!is_valid_geometry(&poly));
    }

    #[test]
    fn test_polygon_invalid_zero_area_collinear() {
        // All points on a line - zero area
        let poly = Geometry::Polygon(make_polygon(&[
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (0.0, 0.0),
        ]));
        assert!(!is_valid_geometry(&poly));
        assert_eq!(
            validate_geometry(&poly),
            ValidationResult::Invalid(InvalidReason::ZeroAreaPolygon)
        );
    }

    #[test]
    fn test_polygon_invalid_zero_area_coincident() {
        // All points at same location - zero area
        let poly = Geometry::Polygon(make_polygon(&[
            (0.5, 0.5),
            (0.5, 0.5),
            (0.5, 0.5),
            (0.5, 0.5),
        ]));
        assert!(!is_valid_geometry(&poly));
        assert_eq!(
            validate_geometry(&poly),
            ValidationResult::Invalid(InvalidReason::ZeroAreaPolygon)
        );
    }

    #[test]
    fn test_polygon_with_valid_hole() {
        let poly = Geometry::Polygon(make_polygon_with_hole(
            &[
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (0.0, 0.0),
            ],
            &[(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0), (2.0, 2.0)],
        ));
        assert!(is_valid_geometry(&poly));
    }

    #[test]
    fn test_polygon_with_invalid_hole() {
        // Exterior is valid, but hole has too few points
        let poly = Geometry::Polygon(make_polygon_with_hole(
            &[
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 10.0),
                (0.0, 10.0),
                (0.0, 0.0),
            ],
            &[
                (2.0, 2.0),
                (8.0, 2.0),
                (2.0, 2.0), // Only 3 points
            ],
        ));
        assert!(!is_valid_geometry(&poly));
        assert_eq!(
            validate_geometry(&poly),
            ValidationResult::Invalid(InvalidReason::PolygonTooFewPoints {
                ring_index: 1, // Interior ring index
                point_count: 3
            })
        );
    }

    // =========================================================================
    // MULTIPOLYGON TESTS
    // =========================================================================

    #[test]
    fn test_multipolygon_valid() {
        let mp = Geometry::MultiPolygon(MultiPolygon::new(vec![
            make_polygon(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]),
            make_polygon(&[(2.0, 2.0), (3.0, 2.0), (3.0, 3.0), (2.0, 3.0), (2.0, 2.0)]),
        ]));
        assert!(is_valid_geometry(&mp));
    }

    #[test]
    fn test_multipolygon_empty_invalid() {
        let mp = Geometry::MultiPolygon(MultiPolygon::new(vec![]));
        assert!(!is_valid_geometry(&mp));
        assert_eq!(
            validate_geometry(&mp),
            ValidationResult::Invalid(InvalidReason::EmptyGeometry)
        );
    }

    #[test]
    fn test_multipolygon_with_one_valid_polygon() {
        // One valid, one invalid
        let mp = Geometry::MultiPolygon(MultiPolygon::new(vec![
            make_polygon(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]), // valid
            make_polygon(&[(2.0, 2.0), (3.0, 2.0), (2.0, 2.0)]), // invalid (too few points)
        ]));
        assert!(is_valid_geometry(&mp));
    }

    #[test]
    fn test_multipolygon_all_invalid() {
        let mp = Geometry::MultiPolygon(MultiPolygon::new(vec![
            make_polygon(&[(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]), // invalid
            make_polygon(&[(2.0, 2.0), (3.0, 2.0), (2.0, 2.0)]), // invalid
        ]));
        assert!(!is_valid_geometry(&mp));
        assert_eq!(
            validate_geometry(&mp),
            ValidationResult::Invalid(InvalidReason::NoValidPolygons)
        );
    }

    // =========================================================================
    // FILTER TESTS
    // =========================================================================

    #[test]
    fn test_filter_valid_geometry_passes_through() {
        let ls = Geometry::LineString(make_linestring(&[(0.0, 0.0), (1.0, 1.0)]));
        let filtered = filter_valid_geometry(&ls);
        assert!(filtered.is_some());
        assert_eq!(filtered.unwrap(), ls);
    }

    #[test]
    fn test_filter_invalid_geometry_returns_none() {
        let ls = Geometry::LineString(make_linestring(&[(0.0, 0.0)]));
        let filtered = filter_valid_geometry(&ls);
        assert!(filtered.is_none());
    }

    #[test]
    fn test_filter_multilinestring_removes_invalid() {
        let mls = Geometry::MultiLineString(MultiLineString::new(vec![
            make_linestring(&[(0.0, 0.0), (1.0, 1.0)]), // valid
            make_linestring(&[(2.0, 2.0)]),             // invalid
            make_linestring(&[(3.0, 3.0), (4.0, 4.0)]), // valid
        ]));

        let filtered = filter_valid_geometry(&mls);
        assert!(filtered.is_some());

        let result = filtered.unwrap();
        if let Geometry::MultiLineString(result_mls) = result {
            assert_eq!(result_mls.0.len(), 2);
        } else {
            panic!("Expected MultiLineString");
        }
    }

    #[test]
    fn test_filter_multilinestring_downgrades_to_single() {
        let mls = Geometry::MultiLineString(MultiLineString::new(vec![
            make_linestring(&[(0.0, 0.0), (1.0, 1.0)]), // valid
            make_linestring(&[(2.0, 2.0)]),             // invalid
        ]));

        let filtered = filter_valid_geometry(&mls);
        assert!(filtered.is_some());

        // Should downgrade to single LineString
        let result = filtered.unwrap();
        assert!(
            matches!(result, Geometry::LineString(_)),
            "Should downgrade to LineString when only one valid line remains"
        );
    }

    #[test]
    fn test_filter_multipolygon_removes_invalid() {
        let mp = Geometry::MultiPolygon(MultiPolygon::new(vec![
            make_polygon(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]), // valid
            make_polygon(&[(2.0, 2.0), (3.0, 2.0), (2.0, 2.0)]),                         // invalid
            make_polygon(&[(4.0, 4.0), (5.0, 4.0), (5.0, 5.0), (4.0, 5.0), (4.0, 4.0)]), // valid
        ]));

        let filtered = filter_valid_geometry(&mp);
        assert!(filtered.is_some());

        let result = filtered.unwrap();
        if let Geometry::MultiPolygon(result_mp) = result {
            assert_eq!(result_mp.0.len(), 2);
        } else {
            panic!("Expected MultiPolygon");
        }
    }

    #[test]
    fn test_filter_multipolygon_downgrades_to_single() {
        let mp = Geometry::MultiPolygon(MultiPolygon::new(vec![
            make_polygon(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]), // valid
            make_polygon(&[(2.0, 2.0), (3.0, 2.0), (2.0, 2.0)]),                         // invalid
        ]));

        let filtered = filter_valid_geometry(&mp);
        assert!(filtered.is_some());

        // Should downgrade to single Polygon
        let result = filtered.unwrap();
        assert!(
            matches!(result, Geometry::Polygon(_)),
            "Should downgrade to Polygon when only one valid polygon remains"
        );
    }

    #[test]
    fn test_filter_all_invalid_returns_none() {
        let mp = Geometry::MultiPolygon(MultiPolygon::new(vec![
            make_polygon(&[(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]), // invalid
            make_polygon(&[(2.0, 2.0), (3.0, 2.0), (2.0, 2.0)]), // invalid
        ]));

        let filtered = filter_valid_geometry(&mp);
        assert!(filtered.is_none());
    }

    // =========================================================================
    // EDGE CASE TESTS
    // =========================================================================

    #[test]
    fn test_polygon_near_zero_area_but_valid() {
        // Very small but valid triangle
        let poly = Geometry::Polygon(make_polygon(&[
            (0.0, 0.0),
            (0.001, 0.0),
            (0.0005, 0.001),
            (0.0, 0.0),
        ]));
        // This has non-zero area (0.0000005), should be valid
        assert!(is_valid_geometry(&poly));
    }

    #[test]
    fn test_validation_result_methods() {
        let valid = ValidationResult::Valid;
        let invalid = ValidationResult::Invalid(InvalidReason::EmptyGeometry);

        assert!(valid.is_valid());
        assert!(!valid.is_invalid());
        assert!(!invalid.is_valid());
        assert!(invalid.is_invalid());
    }
}
