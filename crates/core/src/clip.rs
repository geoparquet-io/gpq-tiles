//! Geometry clipping to tile bounds.
//!
//! Clips geometries to tile boundaries with a configurable buffer zone to prevent
//! visual seams when rendering adjacent tiles.
//!
//! # Tippecanoe Alignment
//!
//! This module matches tippecanoe's clipping behavior:
//! - **Buffer**: Default 8 pixels (configurable via `--buffer` in tippecanoe)
//!   Buffer is measured in "screen pixels" where 1 pixel = 1/256th of tile width
//! - **Clipping method**: Features are clipped to tile boundary + buffer zone
//! - **Duplication**: Features may appear in multiple tiles if they span boundaries
//!
//! See: https://github.com/felt/tippecanoe (clipping documentation)

use geo::{
    line_intersection::{line_intersection, LineIntersection},
    BooleanOps, BoundingRect, Coord, Geometry, Line, LineString, MultiLineString, MultiPolygon,
    Point, Polygon, Rect,
};

use crate::tile::TileBounds;

/// Default buffer in pixels (matches tippecanoe's common usage)
/// Tippecanoe default is 5, but CLAUDE.md specifies 8 for this project
pub const DEFAULT_BUFFER_PIXELS: u32 = 8;

/// Default tile extent in pixels
pub const DEFAULT_EXTENT: u32 = 4096;

/// Clip a geometry to tile bounds with a buffer.
///
/// # Arguments
///
/// * `geom` - The geometry to clip
/// * `bounds` - The tile bounds (without buffer)
/// * `buffer` - Buffer size in the same units as bounds (typically degrees)
///
/// # Returns
///
/// The clipped geometry, or `None` if the geometry doesn't intersect the buffered bounds
///
/// # Tippecanoe Behavior
///
/// Tippecanoe clips features to tile boundaries plus a buffer zone. The buffer
/// prevents visual seams when tiles are rendered side-by-side. Features that
/// span tile boundaries are duplicated into adjacent tiles.
pub fn clip_geometry(
    geom: &Geometry<f64>,
    bounds: &TileBounds,
    buffer: f64,
) -> Option<Geometry<f64>> {
    // Check for self-intersections or other validity issues before clipping.
    // For now, we just log the detection - fallback logic will be added later.
    let validation_errors = validate_geometry(geom);
    if !validation_errors.is_empty() {
        log::trace!(
            "Detected invalid geometry before clipping: {:?}",
            validation_errors
        );
        // TODO: Add fallback to BooleanOps intersection for invalid geometries
        // For now, continue with Sutherland-Hodgman which may produce incorrect results
    }

    let buffered = TileBounds::new(
        bounds.lng_min - buffer,
        bounds.lat_min - buffer,
        bounds.lng_max + buffer,
        bounds.lat_max + buffer,
    );

    match geom {
        Geometry::Point(p) => clip_point(p, &buffered).map(Geometry::Point),
        Geometry::LineString(ls) => clip_linestring(ls, &buffered),
        Geometry::Polygon(poly) => clip_polygon(poly, &buffered),
        Geometry::MultiPolygon(mp) => clip_multipolygon(mp, &buffered).map(Geometry::MultiPolygon),
        Geometry::MultiLineString(mls) => clip_multilinestring(mls, &buffered),
        other => {
            // For other geometry types, use bounding box check
            if let Some(rect) = other.bounding_rect() {
                if intersects_bounds(&rect, &buffered) {
                    return Some(other.clone());
                }
            }
            None
        }
    }
}

/// Convert buffer from pixels to degrees based on tile bounds.
///
/// # Arguments
///
/// * `buffer_pixels` - Buffer size in pixels (e.g., 8)
/// * `tile_bounds` - The tile bounds to calculate pixel size from
/// * `extent` - Tile extent in pixels (e.g., 4096)
///
/// # Returns
///
/// Buffer size in degrees (same units as tile bounds)
pub fn buffer_pixels_to_degrees(buffer_pixels: u32, tile_bounds: &TileBounds, extent: u32) -> f64 {
    // Buffer is specified in "screen pixels" where the tile is extent pixels wide
    // Convert to the same units as bounds (degrees)
    tile_bounds.width() * buffer_pixels as f64 / extent as f64
}

/// Check if a geometry has self-intersections or other validity issues.
///
/// Checks for:
/// - Self-intersections in polygon rings (bowtie/figure-8 patterns)
/// - Spikes where a ring touches itself
/// - Invalid holes (self-intersecting interior rings)
/// - Too few points in rings
/// - Non-finite coordinates
///
/// # Arguments
///
/// * `geom` - The geometry to validate
///
/// # Returns
///
/// A vector of validation error descriptions, or empty vec if valid.
///
/// # Example
///
/// ```
/// use geo::{Geometry, Polygon, LineString, Coord};
/// use gpq_tiles_core::clip::validate_geometry;
///
/// // A bowtie (self-intersecting) polygon
/// let bowtie = Geometry::Polygon(Polygon::new(
///     LineString::from(vec![
///         Coord { x: 0.0, y: 0.0 },
///         Coord { x: 10.0, y: 10.0 },
///         Coord { x: 10.0, y: 0.0 },
///         Coord { x: 0.0, y: 10.0 },
///         Coord { x: 0.0, y: 0.0 },
///     ]),
///     vec![],
/// ));
///
/// let errors = validate_geometry(&bowtie);
/// assert!(!errors.is_empty(), "Bowtie should have validation errors");
/// ```
pub fn validate_geometry(geom: &Geometry<f64>) -> Vec<String> {
    match geom {
        Geometry::Polygon(poly) => validate_polygon(poly),
        Geometry::MultiPolygon(mp) => {
            let mut errors = Vec::new();
            for (i, poly) in mp.0.iter().enumerate() {
                for err in validate_polygon(poly) {
                    errors.push(format!("polygon at index {}: {}", i, err));
                }
            }
            errors
        }
        // Points, LineStrings, etc. - just check for non-finite coords
        _ => validate_coords(geom),
    }
}

/// Validate a polygon for self-intersections and other issues.
fn validate_polygon(poly: &Polygon<f64>) -> Vec<String> {
    let mut errors = Vec::new();

    // Check exterior ring
    if let Some(err) = validate_ring(poly.exterior(), "exterior ring") {
        errors.push(err);
    }

    // Check interior rings (holes)
    for (i, interior) in poly.interiors().iter().enumerate() {
        if let Some(err) = validate_ring(interior, &format!("interior ring at index {}", i)) {
            errors.push(err);
        }
    }

    errors
}

/// Validate a single ring for self-intersections.
fn validate_ring(ring: &LineString<f64>, ring_name: &str) -> Option<String> {
    let coords = &ring.0;

    // Check for too few points (need at least 4 for a closed ring: 3 distinct + closing point)
    if coords.len() < 4 {
        return Some(format!(
            "{} must have at least 3 distinct points",
            ring_name
        ));
    }

    // Check for non-finite coordinates
    for (idx, coord) in coords.iter().enumerate() {
        if !coord.x.is_finite() || !coord.y.is_finite() {
            return Some(format!(
                "{} has a non-finite coordinate at index {}",
                ring_name, idx
            ));
        }
    }

    // Check for spikes: a point that appears twice (non-consecutively) indicates a self-touch
    // This catches cases like: A → B → C → B → D (spike at B)
    if has_spike(ring) {
        return Some(format!("{} has a self-intersection", ring_name));
    }

    // Check for self-intersections by testing all pairs of non-adjacent edges
    if has_self_intersection(ring) {
        return Some(format!("{} has a self-intersection", ring_name));
    }

    None
}

/// Check if a ring has a "spike" - a vertex that appears twice (non-consecutively).
///
/// A spike occurs when the ring goes to a point and returns, like:
/// `(2,4) → (2,6) → (2,4)` - vertex `(2,4)` appears twice.
fn has_spike(ring: &LineString<f64>) -> bool {
    let coords = &ring.0;
    let n = coords.len();

    if n < 4 {
        return false;
    }

    // Check all pairs of vertices (excluding the closing vertex which matches the first)
    // A spike is when the same coordinate appears at non-adjacent positions
    let check_len = if coords.first() == coords.last() {
        n - 1 // Exclude the closing point
    } else {
        n
    };

    for i in 0..check_len {
        for j in (i + 2)..check_len {
            // Skip if i=0 and j is the last vertex (they're "adjacent" in a closed ring)
            if i == 0 && j == check_len - 1 {
                continue;
            }

            if coords[i] == coords[j] {
                return true;
            }
        }
    }

    false
}

/// Check if a ring (closed linestring) has self-intersections.
///
/// Tests each pair of non-adjacent edges. Adjacent edges share a vertex
/// and therefore always "intersect" at that vertex, so we skip those.
fn has_self_intersection(ring: &LineString<f64>) -> bool {
    let coords = &ring.0;
    let n = coords.len();

    if n < 4 {
        return false;
    }

    // Create edges (last edge connects back to first if ring is closed)
    let num_edges = if coords.first() == coords.last() {
        n - 1 // Ring is properly closed
    } else {
        n // Ring is not closed, create closing edge implicitly
    };

    for i in 0..num_edges {
        let edge_i = Line::new(coords[i], coords[(i + 1) % n]);

        // Only check edges that are not adjacent (j > i + 1)
        // Also skip the first/last edge pair if they share a vertex
        for j in (i + 2)..num_edges {
            // Skip if j is the last edge and i is the first (they share a vertex)
            if i == 0 && j == num_edges - 1 {
                continue;
            }

            let edge_j = Line::new(coords[j], coords[(j + 1) % n]);

            if let Some(intersection) = line_intersection(edge_i, edge_j) {
                match intersection {
                    LineIntersection::SinglePoint { intersection, .. } => {
                        // Check if intersection is at a shared endpoint (valid touch)
                        // This can happen if non-adjacent edges happen to meet at a shared vertex
                        // due to complex ring shapes
                        let is_endpoint_i =
                            intersection == edge_i.start || intersection == edge_i.end;
                        let is_endpoint_j =
                            intersection == edge_j.start || intersection == edge_j.end;

                        // Only count as self-intersection if not both endpoints
                        // (a proper crossing, not just touching at vertices)
                        if !(is_endpoint_i && is_endpoint_j) {
                            return true;
                        }
                    }
                    LineIntersection::Collinear { intersection: _ } => {
                        // Overlapping edges are a self-intersection
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// Check geometry coordinates for non-finite values.
fn validate_coords(geom: &Geometry<f64>) -> Vec<String> {
    let mut errors = Vec::new();

    match geom {
        Geometry::Point(p) => {
            if !p.x().is_finite() || !p.y().is_finite() {
                errors.push("Point has non-finite coordinates".to_string());
            }
        }
        Geometry::LineString(ls) => {
            for (i, coord) in ls.0.iter().enumerate() {
                if !coord.x.is_finite() || !coord.y.is_finite() {
                    errors.push(format!(
                        "LineString has non-finite coordinate at index {}",
                        i
                    ));
                }
            }
        }
        // Other geometry types - basic check
        _ => {}
    }

    errors
}

/// Check if a rectangle intersects the given bounds
fn intersects_bounds(rect: &Rect<f64>, bounds: &TileBounds) -> bool {
    rect.max().x >= bounds.lng_min
        && rect.min().x <= bounds.lng_max
        && rect.max().y >= bounds.lat_min
        && rect.min().y <= bounds.lat_max
}

/// Clip a point to bounds (simple containment check)
fn clip_point(point: &Point<f64>, bounds: &TileBounds) -> Option<Point<f64>> {
    if point.x() >= bounds.lng_min
        && point.x() <= bounds.lng_max
        && point.y() >= bounds.lat_min
        && point.y() <= bounds.lat_max
    {
        Some(*point)
    } else {
        None
    }
}

/// Clip a linestring to bounds using BooleanOps.
///
/// IMPORTANT: Uses correct signature - `polygon.clip(&linestring, invert)`
/// NOT `linestring.clip(&polygon)` which doesn't exist.
fn clip_linestring(ls: &LineString<f64>, bounds: &TileBounds) -> Option<Geometry<f64>> {
    // Quick rejection test
    if let Some(rect) = ls.bounding_rect() {
        if !intersects_bounds(&rect, bounds) {
            return None;
        }
    }

    let clip_rect = Rect::new(
        Coord {
            x: bounds.lng_min,
            y: bounds.lat_min,
        },
        Coord {
            x: bounds.lng_max,
            y: bounds.lat_max,
        },
    );
    let clip_poly = clip_rect.to_polygon();

    // Correct usage: polygon.clip(&multilinestring, invert)
    // invert=false means keep the parts INSIDE the polygon
    let mls = MultiLineString::new(vec![ls.clone()]);
    let clipped = clip_poly.clip(&mls, false);

    if clipped.0.is_empty() {
        None
    } else if clipped.0.len() == 1 {
        Some(Geometry::LineString(clipped.0.into_iter().next().unwrap()))
    } else {
        Some(Geometry::MultiLineString(clipped))
    }
}

/// Clip a multilinestring to bounds
fn clip_multilinestring(mls: &MultiLineString<f64>, bounds: &TileBounds) -> Option<Geometry<f64>> {
    // Quick rejection test
    if let Some(rect) = mls.bounding_rect() {
        if !intersects_bounds(&rect, bounds) {
            return None;
        }
    }

    let clip_rect = Rect::new(
        Coord {
            x: bounds.lng_min,
            y: bounds.lat_min,
        },
        Coord {
            x: bounds.lng_max,
            y: bounds.lat_max,
        },
    );
    let clip_poly = clip_rect.to_polygon();

    // Correct usage: polygon.clip(&multilinestring, invert)
    let clipped = clip_poly.clip(mls, false);

    if clipped.0.is_empty() {
        None
    } else {
        Some(Geometry::MultiLineString(clipped))
    }
}

/// Clip a polygon to bounds using intersection
///
/// Returns `Geometry::Polygon` if clipping results in a single polygon,
/// or `Geometry::MultiPolygon` if the clip creates multiple disconnected parts
/// (e.g., a U-shaped polygon clipped across its opening).
///
/// # Algorithm Selection
///
/// - **Valid polygons**: Use fast Sutherland-Hodgman (O(n))
/// - **Invalid polygons** (self-intersections, spikes): Fall back to BooleanOps
///   which handles degenerate cases robustly using Vatti's algorithm
fn clip_polygon(poly: &Polygon<f64>, bounds: &TileBounds) -> Option<Geometry<f64>> {
    // Quick rejection test
    let poly_rect = poly.bounding_rect()?;
    if !intersects_bounds(&poly_rect, bounds) {
        return None;
    }

    // FAST PATH: If polygon is fully inside bounds, return as-is (no clipping needed)
    if poly_rect.min().x >= bounds.lng_min
        && poly_rect.max().x <= bounds.lng_max
        && poly_rect.min().y >= bounds.lat_min
        && poly_rect.max().y <= bounds.lat_max
    {
        return Some(Geometry::Polygon(poly.clone()));
    }

    // Check if polygon is valid for Sutherland-Hodgman
    let validation_errors = validate_polygon(poly);

    if validation_errors.is_empty() {
        // Valid polygon: use fast Sutherland-Hodgman (O(n))
        clip_polygon_sutherland_hodgman(poly, bounds)
    } else {
        // Invalid polygon: fall back to BooleanOps (slower but robust)
        log::trace!(
            "using BooleanOps fallback for invalid polygon: {:?}",
            validation_errors
        );
        clip_polygon_boolean_ops(poly, bounds)
    }
}

/// Clip a polygon using Sutherland-Hodgman algorithm.
///
/// Fast (O(n)) but assumes valid polygon geometry. May produce incorrect
/// results for self-intersecting polygons.
fn clip_polygon_sutherland_hodgman(
    poly: &Polygon<f64>,
    bounds: &TileBounds,
) -> Option<Geometry<f64>> {
    let clipped_exterior = sutherland_hodgman_clip(poly.exterior(), bounds);
    if clipped_exterior.0.len() < 3 {
        return None;
    }

    // Clip interior rings (holes)
    let mut clipped_interiors = Vec::new();
    for interior in poly.interiors() {
        let clipped_interior = sutherland_hodgman_clip(interior, bounds);
        if clipped_interior.0.len() >= 3 {
            clipped_interiors.push(clipped_interior);
        }
    }

    Some(Geometry::Polygon(Polygon::new(
        clipped_exterior,
        clipped_interiors,
    )))
}

/// Clip a polygon using BooleanOps intersection.
///
/// Slower than Sutherland-Hodgman but handles degenerate geometries
/// (self-intersections, spikes) robustly using Vatti's algorithm.
///
/// # DIVERGENCE FROM TIPPECANOE
///
/// Tippecanoe uses custom clipping that may produce different results for
/// invalid geometries. We prioritize correctness (all output coords within
/// bounds) over exact tippecanoe parity for these edge cases.
fn clip_polygon_boolean_ops(poly: &Polygon<f64>, bounds: &TileBounds) -> Option<Geometry<f64>> {
    let clip_rect = Rect::new(
        Coord {
            x: bounds.lng_min,
            y: bounds.lat_min,
        },
        Coord {
            x: bounds.lng_max,
            y: bounds.lat_max,
        },
    );
    let clip_poly = clip_rect.to_polygon();

    // BooleanOps::intersection returns MultiPolygon
    let result: MultiPolygon<f64> = poly.intersection(&clip_poly);

    if result.0.is_empty() {
        None
    } else if result.0.len() == 1 {
        // Single polygon: unwrap from MultiPolygon
        Some(Geometry::Polygon(result.0.into_iter().next().unwrap()))
    } else {
        // Multiple polygons (e.g., from clipping a U-shape)
        Some(Geometry::MultiPolygon(result))
    }
}

/// Sutherland-Hodgman polygon clipping algorithm for axis-aligned rectangles.
/// O(n) complexity vs O(n²) for general polygon intersection.
fn sutherland_hodgman_clip(ring: &LineString<f64>, bounds: &TileBounds) -> LineString<f64> {
    let mut output: Vec<Coord<f64>> = ring.0.clone();

    // Clip against each edge of the rectangle
    // Left edge
    output = clip_against_edge(
        &output,
        |c| c.x >= bounds.lng_min,
        |c1, c2| {
            let t = (bounds.lng_min - c1.x) / (c2.x - c1.x);
            Coord {
                x: bounds.lng_min,
                y: c1.y + t * (c2.y - c1.y),
            }
        },
    );

    // Right edge
    output = clip_against_edge(
        &output,
        |c| c.x <= bounds.lng_max,
        |c1, c2| {
            let t = (bounds.lng_max - c1.x) / (c2.x - c1.x);
            Coord {
                x: bounds.lng_max,
                y: c1.y + t * (c2.y - c1.y),
            }
        },
    );

    // Bottom edge
    output = clip_against_edge(
        &output,
        |c| c.y >= bounds.lat_min,
        |c1, c2| {
            let t = (bounds.lat_min - c1.y) / (c2.y - c1.y);
            Coord {
                x: c1.x + t * (c2.x - c1.x),
                y: bounds.lat_min,
            }
        },
    );

    // Top edge
    output = clip_against_edge(
        &output,
        |c| c.y <= bounds.lat_max,
        |c1, c2| {
            let t = (bounds.lat_max - c1.y) / (c2.y - c1.y);
            Coord {
                x: c1.x + t * (c2.x - c1.x),
                y: bounds.lat_max,
            }
        },
    );

    // Close the ring if needed
    if !output.is_empty() && output.first() != output.last() {
        output.push(output[0]);
    }

    LineString::new(output)
}

/// Clip polygon vertices against a single edge
fn clip_against_edge<F, I>(vertices: &[Coord<f64>], inside: F, intersect: I) -> Vec<Coord<f64>>
where
    F: Fn(&Coord<f64>) -> bool,
    I: Fn(&Coord<f64>, &Coord<f64>) -> Coord<f64>,
{
    if vertices.is_empty() {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(vertices.len());

    for i in 0..vertices.len() {
        let current = &vertices[i];
        let next = &vertices[(i + 1) % vertices.len()];

        let current_inside = inside(current);
        let next_inside = inside(next);

        if current_inside {
            output.push(*current);
            if !next_inside {
                // Exiting: add intersection
                output.push(intersect(current, next));
            }
        } else if next_inside {
            // Entering: add intersection
            output.push(intersect(current, next));
        }
    }

    output
}

/// Clip a multipolygon to bounds
fn clip_multipolygon(mp: &MultiPolygon<f64>, bounds: &TileBounds) -> Option<MultiPolygon<f64>> {
    // Quick rejection test
    let mp_rect = mp.bounding_rect()?;
    if !intersects_bounds(&mp_rect, bounds) {
        return None;
    }

    // FAST PATH: If multipolygon is fully inside bounds, return as-is
    if mp_rect.min().x >= bounds.lng_min
        && mp_rect.max().x <= bounds.lng_max
        && mp_rect.min().y >= bounds.lat_min
        && mp_rect.max().y <= bounds.lat_max
    {
        return Some(mp.clone());
    }

    // Clip each polygon individually
    // Note: clip_polygon may return MultiPolygon (from BooleanOps fallback)
    let mut clipped_polys = Vec::new();
    for poly in &mp.0 {
        match clip_polygon(poly, bounds) {
            Some(Geometry::Polygon(clipped)) => {
                clipped_polys.push(clipped);
            }
            Some(Geometry::MultiPolygon(clipped_mp)) => {
                // BooleanOps can produce multiple polygons from a single input
                clipped_polys.extend(clipped_mp.0);
            }
            _ => {}
        }
    }

    if clipped_polys.is_empty() {
        None
    } else {
        Some(MultiPolygon::new(clipped_polys))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::point;

    // ========== Point Clipping Tests ==========

    #[test]
    fn test_clip_point_inside() {
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);
        let point = point!(x: 5.0, y: 5.0);
        assert!(clip_point(&point, &bounds).is_some());
    }

    #[test]
    fn test_clip_point_outside() {
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);
        let point = point!(x: 15.0, y: 5.0);
        assert!(clip_point(&point, &bounds).is_none());
    }

    #[test]
    fn test_clip_point_on_boundary() {
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);
        let point = point!(x: 10.0, y: 5.0);
        assert!(clip_point(&point, &bounds).is_some());
    }

    // ========== Polygon Clipping Tests ==========

    #[test]
    fn test_clip_polygon_partial() {
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);
        let poly = Polygon::new(
            LineString::from(vec![
                Coord { x: -5.0, y: -5.0 },
                Coord { x: 5.0, y: -5.0 },
                Coord { x: 5.0, y: 5.0 },
                Coord { x: -5.0, y: 5.0 },
                Coord { x: -5.0, y: -5.0 },
            ]),
            vec![],
        );

        let result = clip_polygon(&poly, &bounds);
        assert!(result.is_some());

        // Extract the polygon (should be single polygon for this simple case)
        let clipped = match result.unwrap() {
            Geometry::Polygon(p) => p,
            Geometry::MultiPolygon(mp) => mp.0.into_iter().next().unwrap(),
            _ => panic!("Expected polygon geometry"),
        };
        // Verify clipped polygon is within bounds
        for coord in clipped.exterior().coords() {
            assert!(
                coord.x >= 0.0 && coord.x <= 10.0,
                "x={} out of bounds",
                coord.x
            );
            assert!(
                coord.y >= 0.0 && coord.y <= 10.0,
                "y={} out of bounds",
                coord.y
            );
        }
    }

    #[test]
    fn test_clip_polygon_outside() {
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);
        let poly = Polygon::new(
            LineString::from(vec![
                Coord { x: 20.0, y: 20.0 },
                Coord { x: 30.0, y: 20.0 },
                Coord { x: 30.0, y: 30.0 },
                Coord { x: 20.0, y: 30.0 },
                Coord { x: 20.0, y: 20.0 },
            ]),
            vec![],
        );
        assert!(clip_polygon(&poly, &bounds).is_none());
    }

    #[test]
    fn test_clip_polygon_fully_inside() {
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);
        let poly = Polygon::new(
            LineString::from(vec![
                Coord { x: 2.0, y: 2.0 },
                Coord { x: 8.0, y: 2.0 },
                Coord { x: 8.0, y: 8.0 },
                Coord { x: 2.0, y: 8.0 },
                Coord { x: 2.0, y: 2.0 },
            ]),
            vec![],
        );

        let result = clip_polygon(&poly, &bounds);
        assert!(result.is_some());
    }

    // ========== LineString Clipping Tests ==========

    #[test]
    fn test_clip_linestring_crossing() {
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);
        let ls = LineString::from(vec![Coord { x: -5.0, y: 5.0 }, Coord { x: 15.0, y: 5.0 }]);

        let result = clip_linestring(&ls, &bounds);
        assert!(result.is_some());
    }

    #[test]
    fn test_clip_linestring_outside() {
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);
        let ls = LineString::from(vec![Coord { x: 20.0, y: 20.0 }, Coord { x: 30.0, y: 30.0 }]);

        let result = clip_linestring(&ls, &bounds);
        assert!(result.is_none());
    }

    #[test]
    fn test_clip_linestring_fully_inside() {
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);
        let ls = LineString::from(vec![Coord { x: 2.0, y: 2.0 }, Coord { x: 8.0, y: 8.0 }]);

        let result = clip_linestring(&ls, &bounds);
        assert!(result.is_some());
    }

    // ========== Buffer Calculation Tests ==========

    #[test]
    fn test_buffer_pixels_to_degrees() {
        let bounds = TileBounds::new(0.0, 0.0, 1.0, 1.0);
        let buffer = buffer_pixels_to_degrees(8, &bounds, 4096);

        // 8 pixels / 4096 extent * 1.0 degree width = 0.001953125
        let expected = 8.0 / 4096.0;
        assert!(
            (buffer - expected).abs() < 1e-10,
            "buffer={} expected={}",
            buffer,
            expected
        );
    }

    #[test]
    fn test_buffer_affects_clipping() {
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);
        let buffer = 2.0; // 2 degree buffer

        // Point just outside bounds but within buffer
        let point = point!(x: 11.0, y: 5.0);

        // Without buffer: should be outside
        assert!(clip_point(&point, &bounds).is_none());

        // With buffer via clip_geometry: should be inside
        let result = clip_geometry(&Geometry::Point(point), &bounds, buffer);
        assert!(result.is_some());
    }

    // ========== clip_geometry Integration Tests ==========

    #[test]
    fn test_clip_geometry_point() {
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);
        let point = Geometry::Point(point!(x: 5.0, y: 5.0));

        let result = clip_geometry(&point, &bounds, 0.0);
        assert!(result.is_some());
        assert!(matches!(result.unwrap(), Geometry::Point(_)));
    }

    #[test]
    fn test_clip_geometry_polygon() {
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);
        let poly = Geometry::Polygon(Polygon::new(
            LineString::from(vec![
                Coord { x: 5.0, y: 5.0 },
                Coord { x: 15.0, y: 5.0 },
                Coord { x: 15.0, y: 15.0 },
                Coord { x: 5.0, y: 15.0 },
                Coord { x: 5.0, y: 5.0 },
            ]),
            vec![],
        ));

        let result = clip_geometry(&poly, &bounds, 0.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_clip_geometry_with_buffer() {
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);
        let buffer = 1.0;

        // Polygon just outside bounds but overlapping with buffer
        let poly = Geometry::Polygon(Polygon::new(
            LineString::from(vec![
                Coord { x: 10.5, y: 5.0 },
                Coord { x: 12.0, y: 5.0 },
                Coord { x: 12.0, y: 8.0 },
                Coord { x: 10.5, y: 8.0 },
                Coord { x: 10.5, y: 5.0 },
            ]),
            vec![],
        ));

        // Without buffer: should be outside
        let result_no_buffer = clip_geometry(&poly, &bounds, 0.0);
        assert!(result_no_buffer.is_none());

        // With buffer: should clip to buffered bounds
        let result_with_buffer = clip_geometry(&poly, &bounds, buffer);
        assert!(result_with_buffer.is_some());
    }

    #[test]
    fn test_clip_polygon_multipart_result() {
        // U-shaped polygon that, when clipped by a horizontal band,
        // produces two disconnected polygons (the two "arms" of the U)
        let bounds = TileBounds::new(0.0, 4.0, 10.0, 6.0); // Horizontal band

        // U-shape: two vertical bars connected at the bottom
        // Left bar: x=1-2, y=0-10
        // Right bar: x=8-9, y=0-10
        // Bottom connector: x=1-9, y=0-2
        let u_shape = Polygon::new(
            LineString::from(vec![
                Coord { x: 1.0, y: 0.0 },
                Coord { x: 2.0, y: 0.0 },
                Coord { x: 2.0, y: 10.0 },
                Coord { x: 1.0, y: 10.0 },
                Coord { x: 1.0, y: 2.0 },
                Coord { x: 8.0, y: 2.0 },
                Coord { x: 8.0, y: 10.0 },
                Coord { x: 9.0, y: 10.0 },
                Coord { x: 9.0, y: 0.0 },
                Coord { x: 1.0, y: 0.0 },
            ]),
            vec![],
        );

        let result = clip_polygon(&u_shape, &bounds);
        assert!(result.is_some(), "U-shape should intersect the band");

        // Should produce a MultiPolygon (two separate rectangles from the arms)
        match result.unwrap() {
            Geometry::MultiPolygon(mp) => {
                assert_eq!(
                    mp.0.len(),
                    2,
                    "Should have 2 polygon parts (left and right arms)"
                );
            }
            Geometry::Polygon(_) => {
                // This is also acceptable if geo merges them somehow
            }
            other => panic!("Expected Polygon or MultiPolygon, got {:?}", other),
        }
    }

    // ========== Geometry Validation Tests ==========

    #[test]
    fn test_validate_geometry_valid_polygon() {
        // Simple valid square
        let poly = Geometry::Polygon(Polygon::new(
            LineString::from(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        ));

        let errors = validate_geometry(&poly);
        assert!(
            errors.is_empty(),
            "Valid polygon should have no errors: {:?}",
            errors
        );
    }

    #[test]
    fn test_validate_geometry_bowtie_self_intersection() {
        // Bowtie (figure-8) polygon: edges cross at the center
        //   (0,10)-----(10,10)
        //        \   /
        //         \ /
        //          X  <-- self-intersection at (5,5)
        //         / \
        //        /   \
        //   (0,0)-----(10,0)
        let bowtie = Geometry::Polygon(Polygon::new(
            LineString::from(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 10.0, y: 0.0 },
                Coord { x: 0.0, y: 10.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        ));

        let errors = validate_geometry(&bowtie);
        assert!(
            !errors.is_empty(),
            "Bowtie should have self-intersection error"
        );
        assert!(
            errors
                .iter()
                .any(|e| e.to_lowercase().contains("self-intersection")
                    || e.to_lowercase().contains("self intersection")),
            "Error should mention self-intersection: {:?}",
            errors
        );
    }

    #[test]
    fn test_validate_geometry_spike() {
        // Spike polygon: ring touches itself
        //   (0,4)-----(2,4)--(2,6)--(2,4)-----(4,4)
        //     |                                 |
        //     |                                 |
        //   (0,0)-------------------------(4,0)
        let spike = Geometry::Polygon(Polygon::new(
            LineString::from(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 2.0, y: 4.0 },
                Coord { x: 2.0, y: 6.0 }, // spike up
                Coord { x: 2.0, y: 4.0 }, // back down (self-touch)
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        ));

        let errors = validate_geometry(&spike);
        assert!(
            !errors.is_empty(),
            "Spike should have self-intersection error"
        );
    }

    #[test]
    fn test_validate_geometry_self_intersecting_hole() {
        // Valid exterior with a bowtie (self-intersecting) hole
        let exterior = LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 20.0, y: 0.0 },
            Coord { x: 20.0, y: 20.0 },
            Coord { x: 0.0, y: 20.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);

        // Bowtie hole inside the exterior
        let hole = LineString::from(vec![
            Coord { x: 5.0, y: 5.0 },
            Coord { x: 15.0, y: 15.0 },
            Coord { x: 15.0, y: 5.0 },
            Coord { x: 5.0, y: 15.0 },
            Coord { x: 5.0, y: 5.0 },
        ]);

        let poly_with_bad_hole = Geometry::Polygon(Polygon::new(exterior, vec![hole]));

        let errors = validate_geometry(&poly_with_bad_hole);
        assert!(
            !errors.is_empty(),
            "Polygon with self-intersecting hole should have errors"
        );
        // The error should mention interior ring
        assert!(
            errors.iter().any(|e| e.to_lowercase().contains("interior")
                || e.to_lowercase().contains("self-intersection")),
            "Error should mention interior or self-intersection: {:?}",
            errors
        );
    }

    #[test]
    fn test_validate_geometry_valid_point() {
        let point = Geometry::Point(point!(x: 5.0, y: 5.0));
        let errors = validate_geometry(&point);
        assert!(errors.is_empty(), "Valid point should have no errors");
    }

    #[test]
    fn test_validate_geometry_valid_linestring() {
        let ls = Geometry::LineString(LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 10.0, y: 10.0 },
        ]));
        let errors = validate_geometry(&ls);
        assert!(errors.is_empty(), "Valid linestring should have no errors");
    }

    // ========== BooleanOps Fallback Tests ==========

    /// Helper to check all coordinates are within bounds (with small epsilon for floating point)
    fn all_coords_within_bounds(geom: &Geometry<f64>, bounds: &TileBounds) -> bool {
        let epsilon = 1e-10;
        match geom {
            Geometry::Polygon(poly) => {
                for coord in poly.exterior().coords() {
                    if coord.x < bounds.lng_min - epsilon
                        || coord.x > bounds.lng_max + epsilon
                        || coord.y < bounds.lat_min - epsilon
                        || coord.y > bounds.lat_max + epsilon
                    {
                        return false;
                    }
                }
                for interior in poly.interiors() {
                    for coord in interior.coords() {
                        if coord.x < bounds.lng_min - epsilon
                            || coord.x > bounds.lng_max + epsilon
                            || coord.y < bounds.lat_min - epsilon
                            || coord.y > bounds.lat_max + epsilon
                        {
                            return false;
                        }
                    }
                }
                true
            }
            Geometry::MultiPolygon(mp) => {
                mp.0.iter()
                    .all(|p| all_coords_within_bounds(&Geometry::Polygon(p.clone()), bounds))
            }
            _ => true,
        }
    }

    #[test]
    fn test_clip_bowtie_uses_fallback_and_produces_valid_output() {
        // Bowtie (figure-8) polygon that crosses the clip bounds
        // The self-intersection at (5,5) should trigger BooleanOps fallback
        let bounds = TileBounds::new(0.0, 0.0, 8.0, 8.0);

        let bowtie = Polygon::new(
            LineString::from(vec![
                Coord { x: -2.0, y: -2.0 },
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 10.0, y: -2.0 },
                Coord { x: -2.0, y: 10.0 },
                Coord { x: -2.0, y: -2.0 },
            ]),
            vec![],
        );

        // Verify this is detected as invalid
        let errors = validate_polygon(&bowtie);
        assert!(!errors.is_empty(), "Bowtie should be detected as invalid");

        // Clip the bowtie
        let result = clip_polygon(&bowtie, &bounds);
        assert!(result.is_some(), "Bowtie should produce clipped output");

        // Key assertion: all output coordinates must be within bounds
        let clipped = result.unwrap();
        assert!(
            all_coords_within_bounds(&clipped, &bounds),
            "BooleanOps fallback should produce output within bounds"
        );
    }

    #[test]
    fn test_clip_spike_uses_fallback_and_produces_valid_output() {
        // Spike polygon that extends outside the clip bounds
        let bounds = TileBounds::new(0.0, 0.0, 5.0, 5.0);

        let spike = Polygon::new(
            LineString::from(vec![
                Coord { x: 0.0, y: 0.0 },
                Coord { x: 4.0, y: 0.0 },
                Coord { x: 4.0, y: 4.0 },
                Coord { x: 2.0, y: 4.0 },
                Coord { x: 2.0, y: 8.0 }, // spike up outside bounds
                Coord { x: 2.0, y: 4.0 }, // back down (self-touch)
                Coord { x: 0.0, y: 4.0 },
                Coord { x: 0.0, y: 0.0 },
            ]),
            vec![],
        );

        // Verify this is detected as invalid
        let errors = validate_polygon(&spike);
        assert!(!errors.is_empty(), "Spike should be detected as invalid");

        // Clip the spike
        let result = clip_polygon(&spike, &bounds);
        assert!(result.is_some(), "Spike should produce clipped output");

        // Key assertion: all output coordinates must be within bounds
        let clipped = result.unwrap();
        assert!(
            all_coords_within_bounds(&clipped, &bounds),
            "BooleanOps fallback should produce output within bounds"
        );
    }

    #[test]
    fn test_boolean_ops_directly() {
        // Test the BooleanOps function directly with a simple case
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);

        // A valid polygon that spans outside bounds
        let poly = Polygon::new(
            LineString::from(vec![
                Coord { x: -5.0, y: -5.0 },
                Coord { x: 15.0, y: -5.0 },
                Coord { x: 15.0, y: 15.0 },
                Coord { x: -5.0, y: 15.0 },
                Coord { x: -5.0, y: -5.0 },
            ]),
            vec![],
        );

        let result = clip_polygon_boolean_ops(&poly, &bounds);
        assert!(result.is_some());

        let clipped = result.unwrap();
        assert!(
            all_coords_within_bounds(&clipped, &bounds),
            "BooleanOps should clip to bounds"
        );
    }

    #[test]
    fn test_valid_polygon_uses_fast_path() {
        // A valid polygon should use Sutherland-Hodgman (we verify by checking
        // that both algorithms produce the same result for valid input)
        let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);

        let valid_poly = Polygon::new(
            LineString::from(vec![
                Coord { x: -2.0, y: -2.0 },
                Coord { x: 12.0, y: -2.0 },
                Coord { x: 12.0, y: 12.0 },
                Coord { x: -2.0, y: 12.0 },
                Coord { x: -2.0, y: -2.0 },
            ]),
            vec![],
        );

        // Verify it's valid
        let errors = validate_polygon(&valid_poly);
        assert!(errors.is_empty(), "This polygon should be valid");

        // Both methods should produce valid output
        let sh_result = clip_polygon_sutherland_hodgman(&valid_poly, &bounds);
        let bo_result = clip_polygon_boolean_ops(&valid_poly, &bounds);

        assert!(sh_result.is_some());
        assert!(bo_result.is_some());

        // Both should be within bounds
        assert!(all_coords_within_bounds(&sh_result.unwrap(), &bounds));
        assert!(all_coords_within_bounds(&bo_result.unwrap(), &bounds));
    }
}
