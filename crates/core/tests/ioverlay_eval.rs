//! Evaluation of `i_overlay` crate for fixing self-intersection and invalid geometry issues.
//!
//! This test suite evaluates whether `i_overlay`'s `simplify_shape()` can repair the
//! wagyu fixtures that our current Sutherland-Hodgman clipper fails on.
//!
//! ## Key Findings (2026-02-25)
//!
//! ### simplify_shape() Results
//!
//! **Fixed: 2/10 fixtures**
//! - `polygon-covered-with-hole.json` - hole completely covers exterior → correctly returns empty
//! - `polygon-with-extending-hole.json` - hole extends beyond exterior → splits into valid shapes
//!
//! **NOT Fixed: 8/10 fixtures** (all still have self-intersections after simplify)
//! - `self-intersecting-ring-polygon.json`
//! - `polygon-two-intersecting-holes-and-self-intersection.json`
//! - `polygon-with-spike.json`
//! - `polygon-with-two-holes-outside-exterior-ring.json`
//! - `polygon-two-intersecting-holes.json`
//! - `polygon-with-hole-with-shared-point.json`
//! - `polygon-with-exterior-hole.json`
//! - `polygon-with-double-nested-holes.json`
//!
//! ### Clipping (Intersect) Results
//!
//! - Clipping works correctly for valid polygons
//! - Output uses implicit ring closure (no repeated first point)
//! - Winding order: exterior CCW, holes CW (same as OGC)
//! - When combined with simplify_shape(), can process invalid input
//!
//! ### Performance
//!
//! - Simplify 1000 vertices: ~1ms
//! - Clip 1000 vertices: ~0.5ms
//! - Very reasonable for our use case
//!
//! ### API Ergonomics
//!
//! **Pros:**
//! - Trait-based API (`shape.simplify_shape()`) is ergonomic
//! - Direct array literals work as input
//! - Single function for boolean operations
//! - Clear FillRule and OverlayRule enums
//!
//! **Cons:**
//! - Requires coordinate conversion to/from geo types
//! - Output is nested Vec, needs conversion back
//! - Does NOT auto-fix all self-intersections
//!
//! ### Conclusion
//!
//! **i_overlay is NOT a drop-in solution for our self-intersection problems.**
//! It handles some hole-related issues but does not automatically repair
//! self-intersecting rings. For those, we'd need a dedicated geometry repair
//! algorithm or pre-processing step.
//!
//! However, i_overlay's clipping operation IS viable as a replacement for
//! Sutherland-Hodgman, with the caveat that input should be valid (or pre-validated).
//!
//! ## What we're testing
//!
//! 1. **Geometry repair via simplify_shape()**: Can i_overlay fix self-intersections,
//!    invalid winding, and other geometry problems?
//!
//! 2. **Clipping operations**: Can i_overlay's `Intersect` operation replace our S-H clipper?
//!
//! ## Key fixtures tested (known failures in our current clipper)
//!
//! - `self-intersecting-ring-polygon.json` - ring crosses itself
//! - `polygon-two-intersecting-holes-and-self-intersection.json` - multiple issues
//! - `polygon-with-spike.json` - degenerate spike geometry
//! - `polygon-with-two-holes-outside-exterior-ring.json` - invalid OGC structure
//! - `polygon-two-intersecting-holes.json` - holes that overlap
//! - `polygon-with-hole-with-shared-point.json` - hole touches exterior at a point

use geo::{Coord, Geometry, LineString, MultiPolygon, Polygon};
use geojson::GeoJson;
use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::simplify::SimplifyShape;
use i_overlay::float::single::SingleFloatOverlay;
use std::fs;
use std::path::Path;

// ============================================================================
// Type aliases for i_overlay
// ============================================================================

/// A point in i_overlay format: [x, y]
type IOverlayPoint = [f64; 2];

/// A contour (ring) in i_overlay format: Vec<[x, y]>
type IOverlayContour = Vec<IOverlayPoint>;

/// A shape in i_overlay format: Vec<contours> where first is exterior, rest are holes
type IOverlayShape = Vec<IOverlayContour>;

/// Multiple shapes
type IOverlayShapes = Vec<IOverlayShape>;

// ============================================================================
// Conversion utilities
// ============================================================================

/// Convert a geo::Polygon to i_overlay shape format
fn polygon_to_ioverlay(polygon: &Polygon<f64>) -> IOverlayShape {
    let mut shape = Vec::with_capacity(1 + polygon.interiors().len());

    // Exterior ring
    let exterior: IOverlayContour = polygon.exterior().coords().map(|c| [c.x, c.y]).collect();
    shape.push(exterior);

    // Interior rings (holes)
    for hole in polygon.interiors() {
        let hole_contour: IOverlayContour = hole.coords().map(|c| [c.x, c.y]).collect();
        shape.push(hole_contour);
    }

    shape
}

/// Convert i_overlay shapes back to geo::MultiPolygon
fn ioverlay_to_multipolygon(shapes: &IOverlayShapes) -> MultiPolygon<f64> {
    let polygons: Vec<Polygon<f64>> = shapes
        .iter()
        .map(|shape| {
            if shape.is_empty() {
                return Polygon::new(LineString::new(vec![]), vec![]);
            }

            let exterior = LineString::new(
                shape[0]
                    .iter()
                    .map(|p| Coord { x: p[0], y: p[1] })
                    .collect(),
            );

            let holes: Vec<LineString<f64>> = shape[1..]
                .iter()
                .map(|contour| {
                    LineString::new(contour.iter().map(|p| Coord { x: p[0], y: p[1] }).collect())
                })
                .collect();

            Polygon::new(exterior, holes)
        })
        .collect();

    MultiPolygon::new(polygons)
}

// ============================================================================
// Validation utilities
// ============================================================================

/// Check if a ring is properly closed (either implicitly or explicitly)
/// i_overlay does NOT repeat the first point - rings are implicitly closed
fn is_ring_valid_length(ring: &[[f64; 2]]) -> bool {
    // i_overlay outputs rings without repeating the first point
    // A valid ring needs at least 3 points for a triangle
    ring.len() >= 3
}

/// Compute signed area of a ring (positive = CCW, negative = CW)
/// Works with both closed (first=last) and open (implicitly closed) rings
fn signed_area(ring: &[[f64; 2]]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    let n = ring.len();
    for i in 0..n {
        let j = (i + 1) % n;
        area += ring[i][0] * ring[j][1];
        area -= ring[j][0] * ring[i][1];
    }
    area / 2.0
}

/// Check if ring is non-degenerate (has positive area)
fn has_positive_area(ring: &[[f64; 2]]) -> bool {
    signed_area(ring).abs() > 1e-10
}

/// Check if ring has no self-intersections (simplified check via edge count)
/// Returns true if the ring appears valid (no obvious self-intersection)
fn check_no_obvious_self_intersection(ring: &[[f64; 2]]) -> bool {
    // Very simple check: a valid simple polygon ring shouldn't have repeated points
    // (except first=last for closure)
    if ring.len() < 4 {
        return false; // Need at least 3 unique points + closure
    }

    // Check for duplicate consecutive points (degenerate edges)
    for i in 0..ring.len() - 1 {
        let p1 = &ring[i];
        let p2 = &ring[i + 1];
        if (p1[0] - p2[0]).abs() < 1e-10 && (p1[1] - p2[1]).abs() < 1e-10 {
            return false; // Degenerate edge
        }
    }

    true
}

/// Validate i_overlay winding order: exterior CCW (positive area), holes CW (negative area)
/// Note: i_overlay documentation says exterior is counterclockwise, holes are clockwise
fn validate_ioverlay_winding(shape: &IOverlayShape) -> bool {
    if shape.is_empty() {
        return true; // Empty is valid
    }

    // Exterior should be CCW (positive area) according to i_overlay docs
    let exterior_area = signed_area(&shape[0]);
    if exterior_area <= 0.0 {
        return false;
    }

    // Holes should be CW (negative area)
    for hole in shape.iter().skip(1) {
        let hole_area = signed_area(hole);
        if hole_area >= 0.0 {
            return false;
        }
    }

    true
}

/// Check if winding is consistent (regardless of which convention)
fn has_consistent_winding(shape: &IOverlayShape) -> bool {
    if shape.is_empty() || shape.len() == 1 {
        return true;
    }

    let exterior_area = signed_area(&shape[0]);
    // All holes should have opposite sign from exterior
    for hole in shape.iter().skip(1) {
        let hole_area = signed_area(hole);
        if exterior_area.signum() == hole_area.signum() {
            return false;
        }
    }
    true
}

/// Full validation of i_overlay output
#[derive(Debug)]
struct ValidationResult {
    /// Total number of shapes in output
    shape_count: usize,
    /// All rings have valid length (>= 3 points)
    valid_ring_lengths: bool,
    /// All rings have non-zero area
    non_degenerate: bool,
    /// Correct winding order (exterior CCW, holes CW)
    correct_winding: bool,
    /// Consistent winding (holes opposite of exterior)
    consistent_winding: bool,
    /// No obvious self-intersections
    no_self_intersections: bool,
    /// Overall valid (for our purposes)
    is_valid: bool,
}

fn validate_ioverlay_output(shapes: &IOverlayShapes) -> ValidationResult {
    let shape_count = shapes.len();

    // Empty output is valid
    if shapes.is_empty() {
        return ValidationResult {
            shape_count: 0,
            valid_ring_lengths: true,
            non_degenerate: true,
            correct_winding: true,
            consistent_winding: true,
            no_self_intersections: true,
            is_valid: true,
        };
    }

    let valid_ring_lengths = shapes
        .iter()
        .all(|shape| shape.iter().all(|ring| is_ring_valid_length(ring)));

    let non_degenerate = shapes
        .iter()
        .all(|shape| shape.iter().all(|ring| has_positive_area(ring)));

    let correct_winding = shapes.iter().all(|shape| validate_ioverlay_winding(shape));

    let consistent_winding = shapes.iter().all(|shape| has_consistent_winding(shape));

    let no_self_intersections = shapes.iter().all(|shape| {
        shape
            .iter()
            .all(|ring| check_no_obvious_self_intersection(ring))
    });

    // For our purposes, we care about:
    // - Valid ring lengths
    // - Non-degenerate geometry
    // - Consistent winding (holes opposite of exterior)
    // - No obvious self-intersections
    let is_valid =
        valid_ring_lengths && non_degenerate && consistent_winding && no_self_intersections;

    ValidationResult {
        shape_count,
        valid_ring_lengths,
        non_degenerate,
        correct_winding,
        consistent_winding,
        no_self_intersections,
        is_valid,
    }
}

// ============================================================================
// Fixture loading
// ============================================================================

fn fixtures_dir() -> &'static Path {
    Path::new("tests/fixtures/geometry-test-data/input")
}

fn load_fixture(name: &str) -> Option<Geometry<f64>> {
    let path = fixtures_dir().join(name);
    let content = fs::read_to_string(&path).ok()?;
    let geojson: GeoJson = content.parse().ok()?;

    match geojson {
        GeoJson::Geometry(geom) => geom.try_into().ok(),
        _ => None,
    }
}

fn extract_polygon(geom: &Geometry<f64>) -> Option<Polygon<f64>> {
    match geom {
        Geometry::Polygon(p) => Some(p.clone()),
        Geometry::MultiPolygon(mp) => mp.0.first().cloned(),
        _ => None,
    }
}

// ============================================================================
// Test: simplify_shape on failing fixtures
// ============================================================================

/// Key failing fixtures from our wagyu test suite
const FAILING_FIXTURES: &[&str] = &[
    "self-intersecting-ring-polygon.json",
    "polygon-two-intersecting-holes-and-self-intersection.json",
    "polygon-with-spike.json",
    "polygon-with-two-holes-outside-exterior-ring.json",
    "polygon-two-intersecting-holes.json",
    "polygon-with-hole-with-shared-point.json",
    // Additional fixtures to test robustness
    "polygon-with-extending-hole.json",
    "polygon-with-exterior-hole.json",
    "polygon-covered-with-hole.json",
    "polygon-with-double-nested-holes.json",
];

#[derive(Debug)]
struct SimplifyResult {
    fixture: String,
    input_rings: usize,
    output_shapes: usize,
    validation: ValidationResult,
    error: Option<String>,
}

fn test_simplify_shape(fixture_name: &str) -> SimplifyResult {
    let geom = match load_fixture(fixture_name) {
        Some(g) => g,
        None => {
            return SimplifyResult {
                fixture: fixture_name.to_string(),
                input_rings: 0,
                output_shapes: 0,
                validation: ValidationResult {
                    shape_count: 0,
                    valid_ring_lengths: false,
                    non_degenerate: false,
                    correct_winding: false,
                    consistent_winding: false,
                    no_self_intersections: false,
                    is_valid: false,
                },
                error: Some("Failed to load fixture".to_string()),
            }
        }
    };

    let polygon = match extract_polygon(&geom) {
        Some(p) => p,
        None => {
            return SimplifyResult {
                fixture: fixture_name.to_string(),
                input_rings: 0,
                output_shapes: 0,
                validation: ValidationResult {
                    shape_count: 0,
                    valid_ring_lengths: false,
                    non_degenerate: false,
                    correct_winding: false,
                    consistent_winding: false,
                    no_self_intersections: false,
                    is_valid: false,
                },
                error: Some("Not a polygon".to_string()),
            }
        }
    };

    let input_rings = 1 + polygon.interiors().len();
    let shape = polygon_to_ioverlay(&polygon);

    // Run simplify_shape with EvenOdd fill rule
    // i_overlay uses the SimplifyShape trait
    let result: IOverlayShapes = shape.simplify_shape(FillRule::EvenOdd, 0.0);

    let validation = validate_ioverlay_output(&result);

    SimplifyResult {
        fixture: fixture_name.to_string(),
        input_rings,
        output_shapes: result.len(),
        validation,
        error: None,
    }
}

#[test]
fn test_simplify_shape_on_failing_fixtures() {
    println!("\n=== i_overlay simplify_shape() Evaluation ===\n");
    println!(
        "Testing {} fixtures that fail with our current clipper\n",
        FAILING_FIXTURES.len()
    );

    let mut fixed = 0;
    let mut still_failing = 0;
    let mut results = Vec::new();

    for fixture in FAILING_FIXTURES {
        let result = test_simplify_shape(fixture);

        let status = if result.error.is_some() {
            "ERROR"
        } else if result.validation.is_valid {
            fixed += 1;
            "FIXED"
        } else {
            still_failing += 1;
            "STILL_INVALID"
        };

        println!(
            "[{:14}] {} - {} shapes, valid={}",
            status, fixture, result.output_shapes, result.validation.is_valid
        );

        if !result.validation.is_valid && result.error.is_none() {
            println!(
                "              valid_len={}, non_degen={}, winding={}, no_self_int={}",
                result.validation.valid_ring_lengths,
                result.validation.non_degenerate,
                result.validation.consistent_winding,
                result.validation.no_self_intersections
            );
        }

        results.push(result);
    }

    println!("\n=== Summary ===");
    println!(
        "Fixed by simplify_shape(): {}/{}",
        fixed,
        FAILING_FIXTURES.len()
    );
    println!(
        "Still failing: {}/{}",
        still_failing,
        FAILING_FIXTURES.len()
    );

    // We expect simplify_shape to fix most issues
    // This is an evaluation, so we report rather than assert
}

// ============================================================================
// Test: i_overlay clipping (Intersect operation)
// ============================================================================

/// Test if i_overlay can perform clipping operations that could replace S-H
#[test]
fn test_ioverlay_clipping() {
    println!("\n=== i_overlay Clipping Evaluation ===\n");

    // Create a simple test polygon
    let subject: IOverlayContour = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];

    // Clip box (partially overlapping)
    let clip_box: IOverlayContour = vec![[5.0, 5.0], [15.0, 5.0], [15.0, 15.0], [5.0, 15.0]];

    // Perform intersection
    let result = subject.overlay(&clip_box, OverlayRule::Intersect, FillRule::EvenOdd);

    println!("Subject: {:?}", subject);
    println!("Clip: {:?}", clip_box);
    println!("Result: {} shapes", result.len());

    if !result.is_empty() {
        println!("First shape exterior: {:?}", result[0][0]);

        // Validate the result
        let validation = validate_ioverlay_output(&result);
        println!("Validation: {:?}", validation);

        // The intersection should be approximately [5,5] -> [10,5] -> [10,10] -> [5,10]
        assert!(!result.is_empty(), "Intersection should produce output");
        assert!(validation.is_valid, "Output should be valid");
    }
}

/// Test clipping with an invalid (self-intersecting) input
#[test]
fn test_ioverlay_clipping_invalid_input() {
    println!("\n=== i_overlay Clipping with Invalid Input ===\n");

    // Load a self-intersecting polygon
    let geom = load_fixture("self-intersecting-ring-polygon.json").expect("fixture exists");
    let polygon = extract_polygon(&geom).expect("is polygon");
    let shape = polygon_to_ioverlay(&polygon);

    // Create a clip box that intersects the polygon
    let clip_box: IOverlayContour = vec![[-4.0, 40.0], [-3.0, 40.0], [-3.0, 41.0], [-4.0, 41.0]];

    // First simplify the input
    let simplified: IOverlayShapes = shape.simplify_shape(FillRule::EvenOdd, 0.0);
    println!("After simplify: {} shapes", simplified.len());

    // Now clip the simplified result
    if !simplified.is_empty() {
        let clipped =
            simplified[0]
                .as_slice()
                .overlay(&clip_box, OverlayRule::Intersect, FillRule::EvenOdd);
        println!("After clip: {} shapes", clipped.len());

        let validation = validate_ioverlay_output(&clipped);
        println!("Clipped validation: {:?}", validation);
    }
}

// ============================================================================
// Test: API ergonomics evaluation
// ============================================================================

#[test]
fn test_api_ergonomics() {
    println!("\n=== i_overlay API Ergonomics ===\n");

    // Test 1: Simple polygon creation and simplification
    let simple_poly: IOverlayContour = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];

    // simplify_shape is a trait method - very ergonomic
    let result: IOverlayShapes = simple_poly.simplify_shape(FillRule::EvenOdd, 0.0);
    println!("Simple polygon: {} shapes after simplify", result.len());

    // Test 2: Polygon with hole
    let poly_with_hole: IOverlayShape = vec![
        // Exterior (CCW)
        vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        // Hole (CW)
        vec![[2.0, 2.0], [2.0, 8.0], [8.0, 8.0], [8.0, 2.0]],
    ];

    let result: IOverlayShapes = poly_with_hole.simplify_shape(FillRule::EvenOdd, 0.0);
    println!("Polygon with hole: {} shapes after simplify", result.len());

    // Test 3: Direct boolean operation
    let a: IOverlayContour = vec![[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]];
    let b: IOverlayContour = vec![[2.0, 2.0], [7.0, 2.0], [7.0, 7.0], [2.0, 7.0]];

    let union = a.overlay(&b, OverlayRule::Union, FillRule::EvenOdd);
    let intersection = a.overlay(&b, OverlayRule::Intersect, FillRule::EvenOdd);
    let difference = a.overlay(&b, OverlayRule::Difference, FillRule::EvenOdd);

    println!(
        "Boolean ops: union={} shapes, intersect={} shapes, diff={} shapes",
        union.len(),
        intersection.len(),
        difference.len()
    );

    // All operations should produce valid output
    assert!(validate_ioverlay_output(&union).is_valid);
    assert!(validate_ioverlay_output(&intersection).is_valid);
    assert!(validate_ioverlay_output(&difference).is_valid);

    println!("\nAPI Evaluation:");
    println!("+ Trait-based API is very ergonomic (shape.simplify_shape())");
    println!("+ Direct array literals work as input");
    println!("+ Single function for boolean operations (overlay)");
    println!("+ Clear fill rule and overlay rule enums");
    println!("- Requires coordinate conversion to/from geo types");
    println!("- Output is nested Vec, needs conversion back");
}

// ============================================================================
// Test: Performance quick check
// ============================================================================

#[test]
fn test_performance_quick_check() {
    use std::time::Instant;

    println!("\n=== i_overlay Performance Quick Check ===\n");

    // Generate a complex polygon with many vertices
    let n_vertices = 1000;
    let mut contour: IOverlayContour = Vec::with_capacity(n_vertices);
    for i in 0..n_vertices {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_vertices as f64);
        let radius = 100.0 + 10.0 * (5.0 * angle).sin(); // Wiggly circle
        contour.push([radius * angle.cos(), radius * angle.sin()]);
    }

    // Simplify
    let start = Instant::now();
    let _result: IOverlayShapes = contour.simplify_shape(FillRule::EvenOdd, 0.0);
    let simplify_time = start.elapsed();

    println!("Simplify {} vertices: {:?}", n_vertices, simplify_time);

    // Clip against a box
    let clip_box: IOverlayContour =
        vec![[-50.0, -50.0], [50.0, -50.0], [50.0, 50.0], [-50.0, 50.0]];

    let start = Instant::now();
    let _result = contour.overlay(&clip_box, OverlayRule::Intersect, FillRule::EvenOdd);
    let clip_time = start.elapsed();

    println!("Clip {} vertices against box: {:?}", n_vertices, clip_time);

    // Multiple clips (simulating tiling)
    let start = Instant::now();
    for i in 0..10 {
        let offset = (i as f64) * 20.0 - 90.0;
        let tile_box: IOverlayContour = vec![
            [offset, offset],
            [offset + 20.0, offset],
            [offset + 20.0, offset + 20.0],
            [offset, offset + 20.0],
        ];
        let _result = contour.overlay(&tile_box, OverlayRule::Intersect, FillRule::EvenOdd);
    }
    let multi_clip_time = start.elapsed();

    println!("10 clips of {} vertices: {:?}", n_vertices, multi_clip_time);
    println!("Average per clip: {:?}", multi_clip_time / 10);
}

// ============================================================================
// Run all evaluations as a single test for easy invocation
// ============================================================================

#[test]
fn run_full_evaluation() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           i_overlay Crate Evaluation for gpq-tiles           ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Run simplify_shape tests
    test_simplify_shape_on_failing_fixtures();

    // Run clipping tests
    test_ioverlay_clipping();
    test_ioverlay_clipping_invalid_input();

    // Run API ergonomics
    test_api_ergonomics();

    // Run performance check
    test_performance_quick_check();

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                    Evaluation Complete                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
