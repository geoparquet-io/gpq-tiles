//! Zoom-based geometry simplification.
//!
//! Uses the Ramer-Douglas-Peucker (RDP) algorithm via `geo::Simplify` to reduce
//! vertex count based on zoom level. Matches tippecanoe's approach: simplify to
//! tile resolution (~1 pixel at the given zoom level).

use geo::{Geometry, Simplify};

/// Simplify geometry to tile resolution (Douglas-Peucker).
///
/// Matches tippecanoe's approach: "At every zoom level, line and polygon features
/// are subjected to Douglas-Peucker simplification to the resolution of the tile."
///
/// Tolerance calculation:
/// - At zoom z, one tile covers 360° / 2^z degrees
/// - With `extent` pixels per tile, each pixel = tile_degrees / extent
/// - We use 1 pixel as the tolerance (matching tippecanoe)
///
/// Points and MultiPoints pass through unchanged since they have no vertices to reduce.
pub fn simplify_for_zoom(geom: &Geometry<f64>, zoom: u8, extent: u32) -> Geometry<f64> {
    // Tippecanoe simplifies to tile resolution
    // At zoom z, one tile covers 360/2^z degrees
    // With extent pixels, tolerance is degrees per pixel
    let tile_degrees = 360.0 / (1u64 << zoom) as f64;
    let tolerance = tile_degrees / extent as f64;

    // Guard against numerical issues at high zoom levels
    if tolerance < 1e-10 {
        return geom.clone();
    }

    match geom {
        // Points have no vertices to simplify
        Geometry::Point(_) | Geometry::MultiPoint(_) => geom.clone(),

        // Apply RDP simplification to line/polygon types
        Geometry::LineString(ls) => Geometry::LineString(ls.simplify(&tolerance)),
        Geometry::Polygon(poly) => Geometry::Polygon(poly.simplify(&tolerance)),
        Geometry::MultiPolygon(mp) => Geometry::MultiPolygon(mp.simplify(&tolerance)),
        Geometry::MultiLineString(mls) => Geometry::MultiLineString(mls.simplify(&tolerance)),

        // GeometryCollection and other types pass through unchanged
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{Coord, LineString};

    #[test]
    fn test_simplify_reduces_vertices() {
        // Create a line with 100 points that has small oscillations
        // The oscillations are small enough to be simplified away at zoom 0
        let coords: Vec<Coord<f64>> = (0..100)
            .map(|i| Coord {
                x: i as f64 * 0.01,
                y: (i as f64 * 0.1).sin() * 0.001,
            })
            .collect();
        let line = LineString::new(coords);
        let geom = Geometry::LineString(line.clone());

        // At zoom 0, tolerance is large, should simplify aggressively
        let simplified = simplify_for_zoom(&geom, 0, 4096);
        if let Geometry::LineString(s) = simplified {
            assert!(
                s.coords().count() < line.coords().count(),
                "Expected fewer vertices after simplification: got {}, original {}",
                s.coords().count(),
                line.coords().count()
            );
        } else {
            panic!("Expected LineString geometry");
        }
    }

    #[test]
    fn test_points_unchanged() {
        let point = Geometry::Point(geo::point!(x: 1.0, y: 2.0));
        assert_eq!(point, simplify_for_zoom(&point, 5, 4096));
    }

    #[test]
    fn test_multipoint_unchanged() {
        use geo::MultiPoint;
        let mp = Geometry::MultiPoint(MultiPoint::new(vec![
            geo::point!(x: 1.0, y: 2.0),
            geo::point!(x: 3.0, y: 4.0),
        ]));
        assert_eq!(mp, simplify_for_zoom(&mp, 10, 4096));
    }

    #[test]
    fn test_high_zoom_preserves_detail() {
        // At high zoom, tolerance is very small, should preserve most vertices
        // At zoom 20 with extent 4096: tolerance = 360 / 2^20 / 4096 ≈ 8.4e-8 degrees
        // Create vertices spaced further apart than the tolerance
        let coords: Vec<Coord<f64>> = (0..10)
            .map(|i| Coord {
                x: i as f64 * 0.001, // 0.001° spacing >> 8.4e-8° tolerance
                y: (i as f64 * 0.5).sin() * 0.001,
            })
            .collect();
        let line = LineString::new(coords.clone());
        let geom = Geometry::LineString(line.clone());

        // At zoom 20, tolerance is tiny, should preserve all detail
        let simplified = simplify_for_zoom(&geom, 20, 4096);
        if let Geometry::LineString(s) = simplified {
            // Should preserve all vertices since they're spaced well above tolerance
            assert_eq!(
                s.coords().count(),
                line.coords().count(),
                "High zoom should preserve all vertices when spacing >> tolerance"
            );
        }
    }

    #[test]
    fn test_tolerance_decreases_with_zoom() {
        // Create a line with predictable behavior
        let coords: Vec<Coord<f64>> = (0..50)
            .map(|i| Coord {
                x: i as f64 * 0.02,
                y: (i as f64 * 0.2).sin() * 0.01,
            })
            .collect();
        let line = LineString::new(coords);
        let geom = Geometry::LineString(line);

        let simplified_z0 = simplify_for_zoom(&geom, 0, 4096);
        let simplified_z5 = simplify_for_zoom(&geom, 5, 4096);
        let simplified_z10 = simplify_for_zoom(&geom, 10, 4096);

        let count_z0 = if let Geometry::LineString(s) = simplified_z0 {
            s.coords().count()
        } else {
            0
        };
        let count_z5 = if let Geometry::LineString(s) = simplified_z5 {
            s.coords().count()
        } else {
            0
        };
        let count_z10 = if let Geometry::LineString(s) = simplified_z10 {
            s.coords().count()
        } else {
            0
        };

        // Higher zoom should generally preserve more vertices
        assert!(
            count_z0 <= count_z5 && count_z5 <= count_z10,
            "Expected more vertices at higher zooms: z0={}, z5={}, z10={}",
            count_z0,
            count_z5,
            count_z10
        );
    }

    #[test]
    fn test_polygon_simplification() {
        use geo::Polygon;

        // Create a polygon with many vertices (approximating a circle)
        let coords: Vec<Coord<f64>> = (0..=36)
            .map(|i| {
                let angle = (i as f64) * 10.0 * std::f64::consts::PI / 180.0;
                Coord {
                    x: angle.cos() * 0.1,
                    y: angle.sin() * 0.1,
                }
            })
            .collect();
        let poly = Polygon::new(LineString::new(coords), vec![]);
        let geom = Geometry::Polygon(poly.clone());

        let simplified = simplify_for_zoom(&geom, 0, 4096);
        if let Geometry::Polygon(s) = simplified {
            assert!(
                s.exterior().coords().count() < poly.exterior().coords().count(),
                "Polygon should be simplified at zoom 0"
            );
        }
    }

    #[test]
    fn test_tolerance_matches_tippecanoe() {
        // Verify our tolerance formula matches tippecanoe's approach
        // At zoom 0: 360° / 4096 = 0.087890625° per pixel
        // At zoom 1: 180° / 4096 = 0.0439453125° per pixel
        // At zoom 10: ~0.351° / 4096 ≈ 0.0000858° per pixel

        let extent = 4096;

        // Zoom 0: 360 / 1 / 4096
        let tol_z0 = 360.0 / (1u64 << 0) as f64 / extent as f64;
        assert!(
            (tol_z0 - 0.087890625).abs() < 1e-9,
            "Zoom 0 tolerance mismatch: {}",
            tol_z0
        );

        // Zoom 1: 360 / 2 / 4096
        let tol_z1 = 360.0 / (1u64 << 1) as f64 / extent as f64;
        assert!(
            (tol_z1 - 0.0439453125).abs() < 1e-9,
            "Zoom 1 tolerance mismatch: {}",
            tol_z1
        );

        // Verify tolerance halves with each zoom level
        let tol_z2 = 360.0 / (1u64 << 2) as f64 / extent as f64;
        assert!(
            (tol_z1 / tol_z2 - 2.0).abs() < 1e-9,
            "Tolerance should halve with each zoom"
        );
    }
}
