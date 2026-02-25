//! Robustness and correctness tests for geometry clipping using wagyu/geometry-test-data fixtures.
//!
//! These tests verify that `clip_geometry` handles edge cases and degenerate geometries
//! without panicking. The fixtures are from the Mapnik/wagyu project and include:
//!
//! - **Curated fixtures** (input/): Hand-crafted edge cases like self-intersecting rings,
//!   spikes, and intersecting holes
//! - **Fuzzer fixtures** (input-polyjson/): Crash cases discovered by fuzzing wagyu
//!
//! # What we're testing
//!
//! ## Robustness tests (original)
//! 1. The clipper doesn't panic on malformed input
//! 2. The clipper returns Some or None (not crash)
//! 3. We can identify which fixtures our current implementation fails on
//!
//! ## Correctness tests (added)
//! 1. All output coordinates are within clip bounds (plus buffer)
//! 2. Polygon rings are properly closed (first coord == last coord)
//! 3. Winding order follows OGC convention (exterior CCW, holes CW)
//! 4. Specific test cases with known expected outputs
//!
//! # Reference
//!
//! - https://github.com/mapnik/geometry-test-data
//! - Fixtures originally from https://github.com/nickolasclarke/geo-clipper/issues/5

use std::fs;
use std::path::Path;

use geo::{Coord, Geometry, LineString, MultiPolygon, Polygon};
use geojson::GeoJson;
use gpq_tiles_core::clip::clip_geometry;
use gpq_tiles_core::tile::TileBounds;

/// Test result for a single fixture
#[derive(Debug)]
#[allow(dead_code)]
enum FixtureResult {
    /// Clipping succeeded (returned Some or None)
    Pass,
    /// Clipping panicked
    Panic(String),
    /// Failed to parse the fixture
    ParseError(String),
}

/// Standard tile bounds for testing (roughly world bounds normalized)
/// Using a tile that's likely to intersect most test geometries
fn test_bounds() -> TileBounds {
    // A reasonable tile bounds that will intersect most of the test fixtures
    // The fixtures use various coordinate systems (some geographic, some pixel-space)
    TileBounds::new(-180.0, -90.0, 180.0, 90.0)
}

/// Alternative bounds for pixel-space coordinates (used by polyjson fixtures)
fn pixel_bounds() -> TileBounds {
    // Many polyjson fixtures use pixel coordinates in the range 0-4096+
    TileBounds::new(0.0, 0.0, 4096.0, 4096.0)
}

/// Parse a GeoJSON file into a geo::Geometry
fn parse_geojson_file(path: &Path) -> Result<Geometry<f64>, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("Read error: {}", e))?;

    let geojson: GeoJson = content
        .parse()
        .map_err(|e| format!("GeoJSON parse error: {}", e))?;

    match geojson {
        GeoJson::Geometry(geom) => {
            use std::convert::TryInto;
            geom.try_into()
                .map_err(|e| format!("Geometry conversion error: {:?}", e))
        }
        GeoJson::Feature(f) => {
            if let Some(geom) = f.geometry {
                use std::convert::TryInto;
                geom.try_into()
                    .map_err(|e| format!("Geometry conversion error: {:?}", e))
            } else {
                Err("Feature has no geometry".to_string())
            }
        }
        GeoJson::FeatureCollection(_) => Err("FeatureCollection not supported".to_string()),
    }
}

/// Parse a polyjson file (raw coordinate arrays) into a geo::Geometry
///
/// Polyjson format is just the coordinates array without the GeoJSON wrapper:
/// - Polygon: `[[[x,y], ...]]` - array of rings, each ring is array of [x,y] coords
/// - MultiPolygon: `[[[[x,y], ...]], ...]` - array of polygons
fn parse_polyjson_file(path: &Path) -> Result<Geometry<f64>, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("Read error: {}", e))?;

    let value: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("JSON parse error: {}", e))?;

    parse_polyjson_value(&value)
}

/// Determine the nesting depth to reach a number
fn get_nesting_depth(value: &serde_json::Value) -> usize {
    let mut depth = 0;
    let mut current = value;

    while let Some(arr) = current.as_array() {
        if arr.is_empty() {
            break;
        }
        depth += 1;
        current = &arr[0];
    }

    depth
}

fn parse_polyjson_value(value: &serde_json::Value) -> Result<Geometry<f64>, String> {
    // Determine structure by counting nesting depth to reach a number:
    // - Depth 2: [[x, y], ...] - single ring (rare)
    // - Depth 3: [[[x, y], ...]] - Polygon (array of rings)
    // - Depth 4: [[[[x, y], ...], ...], ...] - MultiPolygon (array of polygons)

    let depth = get_nesting_depth(value);

    match depth {
        2 => {
            // Single ring: [[x, y], ...]
            let ring_arr = value.as_array().ok_or("Expected array")?;
            let ring = parse_ring(ring_arr)?;
            Ok(Geometry::Polygon(Polygon::new(ring, vec![])))
        }
        3 => {
            // Polygon: [[[x, y], ...], [[x, y], ...]]
            // First element is exterior, rest are holes
            let polygon_arr = value.as_array().ok_or("Expected array")?;
            if polygon_arr.is_empty() {
                return Err("Empty polygon".to_string());
            }

            let exterior_arr = polygon_arr[0]
                .as_array()
                .ok_or("Expected exterior ring array")?;
            let exterior = parse_ring(exterior_arr)?;

            let interiors: Vec<LineString<f64>> = polygon_arr
                .iter()
                .skip(1)
                .filter_map(|v| v.as_array())
                .filter_map(|arr| parse_ring(arr).ok())
                .collect();

            Ok(Geometry::Polygon(Polygon::new(exterior, interiors)))
        }
        4 => {
            // MultiPolygon: [[[[x, y], ...]], [[[x, y], ...]]]
            let multi_arr = value.as_array().ok_or("Expected array")?;

            let polygons: Vec<Polygon<f64>> = multi_arr
                .iter()
                .filter_map(|poly_val| {
                    let poly_arr = poly_val.as_array()?;
                    if poly_arr.is_empty() {
                        return None;
                    }

                    let exterior_arr = poly_arr[0].as_array()?;
                    let exterior = parse_ring(exterior_arr).ok()?;

                    let interiors: Vec<LineString<f64>> = poly_arr
                        .iter()
                        .skip(1)
                        .filter_map(|v| v.as_array())
                        .filter_map(|arr| parse_ring(arr).ok())
                        .collect();

                    Some(Polygon::new(exterior, interiors))
                })
                .collect();

            if polygons.is_empty() {
                Err("Empty multipolygon".to_string())
            } else {
                Ok(Geometry::MultiPolygon(MultiPolygon::new(polygons)))
            }
        }
        _ => Err(format!("Unsupported nesting depth: {}", depth)),
    }
}

fn parse_ring(arr: &[serde_json::Value]) -> Result<LineString<f64>, String> {
    let coords: Result<Vec<Coord<f64>>, String> = arr
        .iter()
        .map(|v| {
            // Handle both [x, y] format and {x, y} as numbers directly
            if let Some(coord_arr) = v.as_array() {
                if coord_arr.len() < 2 {
                    return Err("Coordinate array too short".to_string());
                }
                let x = coord_arr[0]
                    .as_f64()
                    .or_else(|| coord_arr[0].as_i64().map(|i| i as f64))
                    .ok_or("Expected number for x coordinate")?;
                let y = coord_arr[1]
                    .as_f64()
                    .or_else(|| coord_arr[1].as_i64().map(|i| i as f64))
                    .ok_or("Expected number for y coordinate")?;
                Ok(Coord { x, y })
            } else {
                Err("Expected coordinate array".to_string())
            }
        })
        .collect();

    Ok(LineString::new(coords?))
}

/// Run clip_geometry and catch panics
fn test_clip(geom: &Geometry<f64>, bounds: &TileBounds) -> FixtureResult {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        clip_geometry(geom, bounds, 0.0)
    }));

    match result {
        Ok(_) => FixtureResult::Pass,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            FixtureResult::Panic(msg)
        }
    }
}

/// Run tests on a directory of fixtures
fn run_fixture_tests(
    dir: &Path,
    parse_fn: fn(&Path) -> Result<Geometry<f64>, String>,
    bounds: &TileBounds,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let mut passed = Vec::new();
    let mut failed = Vec::new();
    let mut parse_errors = Vec::new();

    if !dir.exists() {
        eprintln!("Fixture directory does not exist: {:?}", dir);
        return (passed, failed, parse_errors);
    }

    let entries: Vec<_> = fs::read_dir(dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
        })
        .collect();

    for entry in entries {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();

        match parse_fn(&path) {
            Ok(geom) => match test_clip(&geom, bounds) {
                FixtureResult::Pass => passed.push(name),
                FixtureResult::Panic(msg) => {
                    failed.push(format!("{}: PANIC - {}", name, msg));
                }
                FixtureResult::ParseError(msg) => {
                    parse_errors.push(format!("{}: {}", name, msg));
                }
            },
            Err(e) => {
                parse_errors.push(format!("{}: {}", name, e));
            }
        }
    }

    (passed, failed, parse_errors)
}

/// Get the path to the geometry-test-data submodule
fn fixture_base_path() -> std::path::PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    std::path::PathBuf::from(manifest_dir).join("tests/fixtures/geometry-test-data")
}

#[test]
fn test_curated_fixtures() {
    let base = fixture_base_path();
    let input_dir = base.join("input");

    println!("\n=== Testing Curated Fixtures (GeoJSON) ===\n");

    let (passed, failed, parse_errors) =
        run_fixture_tests(&input_dir, parse_geojson_file, &test_bounds());

    println!("Passed: {}", passed.len());
    println!("Failed (panic): {}", failed.len());
    println!("Parse errors: {}", parse_errors.len());

    if !failed.is_empty() {
        println!("\nFailed fixtures:");
        for f in &failed {
            println!("  - {}", f);
        }
    }

    if !parse_errors.is_empty() {
        println!("\nParse errors:");
        for e in &parse_errors {
            println!("  - {}", e);
        }
    }

    // Report summary but don't fail the test - we're documenting current behavior
    let total = passed.len() + failed.len();
    println!(
        "\n=== Curated fixtures: {}/{} passed ===\n",
        passed.len(),
        total
    );
}

#[test]
fn test_polyjson_fixtures() {
    let base = fixture_base_path();
    let polyjson_dir = base.join("input-polyjson");

    println!("\n=== Testing Polyjson Fixtures (Fuzzer Cases) ===\n");

    let (passed, failed, parse_errors) =
        run_fixture_tests(&polyjson_dir, parse_polyjson_file, &pixel_bounds());

    println!("Passed: {}", passed.len());
    println!("Failed (panic): {}", failed.len());
    println!("Parse errors: {}", parse_errors.len());

    if !failed.is_empty() {
        println!("\nFailed fixtures:");
        for f in &failed {
            println!("  - {}", f);
        }
    }

    if !parse_errors.is_empty() && parse_errors.len() <= 10 {
        println!("\nParse errors:");
        for e in &parse_errors {
            println!("  - {}", e);
        }
    } else if !parse_errors.is_empty() {
        println!(
            "\n(Showing first 10 of {} parse errors)",
            parse_errors.len()
        );
        for e in parse_errors.iter().take(10) {
            println!("  - {}", e);
        }
    }

    // Report summary but don't fail the test
    let total = passed.len() + failed.len();
    println!(
        "\n=== Polyjson fixtures: {}/{} passed ===\n",
        passed.len(),
        total
    );
}

#[test]
fn test_specific_edge_cases() {
    let base = fixture_base_path();
    let input_dir = base.join("input");

    // Test specific high-priority edge cases
    let priority_fixtures = [
        "self-intersecting-ring-polygon.json",
        "polygon-two-intersecting-holes-and-self-intersection.json",
        "polygon-with-spike.json",
        "multi-polygon-with-spikes.json",
    ];

    println!("\n=== Testing Priority Edge Cases ===\n");

    for fixture_name in priority_fixtures {
        let path = input_dir.join(fixture_name);
        if !path.exists() {
            println!("{}: SKIPPED (file not found)", fixture_name);
            continue;
        }

        match parse_geojson_file(&path) {
            Ok(geom) => {
                // Test with multiple bounds configurations
                let bounds_configs = [
                    ("world", TileBounds::new(-180.0, -90.0, 180.0, 90.0)),
                    ("tight", TileBounds::new(-10.0, 39.0, 0.0, 42.0)),
                    ("partial", TileBounds::new(-5.0, 40.0, -3.0, 41.0)),
                ];

                for (config_name, bounds) in bounds_configs {
                    match test_clip(&geom, &bounds) {
                        FixtureResult::Pass => {
                            println!("{} [{}]: PASS", fixture_name, config_name);
                        }
                        FixtureResult::Panic(msg) => {
                            println!("{} [{}]: PANIC - {}", fixture_name, config_name, msg);
                        }
                        FixtureResult::ParseError(msg) => {
                            println!("{} [{}]: ERROR - {}", fixture_name, config_name, msg);
                        }
                    }
                }
            }
            Err(e) => {
                println!("{}: PARSE ERROR - {}", fixture_name, e);
            }
        }
    }

    println!();
}

/// Test clipping with bounds that ACTUALLY intersect geometries (forcing real clipping)
///
/// Many fixtures might pass with world bounds because the geometry is fully contained
/// and the fast path returns early. This test uses tighter bounds to force actual clipping.
#[test]
fn test_curated_with_tight_bounds() {
    let base = fixture_base_path();
    let input_dir = base.join("input");

    println!("\n=== Testing Curated Fixtures with Tight Bounds ===\n");

    // Bounds that will intersect most fixtures (many are in Spain region)
    let tight_bounds = TileBounds::new(-5.0, 39.5, -2.5, 41.5);

    let (passed, failed, parse_errors) =
        run_fixture_tests(&input_dir, parse_geojson_file, &tight_bounds);

    println!("Passed: {}", passed.len());
    println!("Failed (panic): {}", failed.len());
    println!("Parse errors: {}", parse_errors.len());

    if !failed.is_empty() {
        println!("\nFailed fixtures:");
        for f in &failed {
            println!("  - {}", f);
        }
    }

    let total = passed.len() + failed.len();
    println!(
        "\n=== Curated fixtures (tight bounds): {}/{} passed ===\n",
        passed.len(),
        total
    );
}

/// Test polyjson fixtures with bounds that force actual clipping
#[test]
fn test_polyjson_with_intersecting_bounds() {
    let base = fixture_base_path();
    let polyjson_dir = base.join("input-polyjson");

    println!("\n=== Testing Polyjson with Intersecting Bounds ===\n");

    // Bounds that will partially clip many fixtures (pixel-space)
    let intersecting_bounds = TileBounds::new(10.0, 10.0, 35.0, 35.0);

    let (passed, failed, parse_errors) =
        run_fixture_tests(&polyjson_dir, parse_polyjson_file, &intersecting_bounds);

    println!("Passed: {}", passed.len());
    println!("Failed (panic): {}", failed.len());
    println!("Parse errors: {}", parse_errors.len());

    if !failed.is_empty() {
        println!("\nFailed fixtures:");
        for f in &failed {
            println!("  - {}", f);
        }
    }

    let total = passed.len() + failed.len();
    println!(
        "\n=== Polyjson fixtures (intersecting bounds): {}/{} passed ===\n",
        passed.len(),
        total
    );
}

// ============================================================================
// CORRECTNESS VALIDATION HELPERS
// ============================================================================

/// Epsilon for coordinate comparisons (accounts for floating point errors)
const COORD_EPSILON: f64 = 1e-10;

/// Correctness issue found in clipped output
#[derive(Debug, Clone)]
enum CorrectnessIssue {
    /// Coordinate is outside the clip bounds (plus buffer)
    CoordinateOutsideBounds {
        coord: Coord<f64>,
        bounds: TileBounds,
        buffer: f64,
    },
    /// Ring is not properly closed (first != last)
    RingNotClosed { first: Coord<f64>, last: Coord<f64> },
    /// Winding order is incorrect
    IncorrectWinding {
        ring_type: &'static str,
        expected: &'static str,
        actual: &'static str,
    },
    /// Ring has too few points (< 4 for closed polygon ring)
    InsufficientPoints { count: usize },
    /// Signed area is zero (degenerate polygon)
    DegeneratePolygon,
}

impl std::fmt::Display for CorrectnessIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorrectnessIssue::CoordinateOutsideBounds {
                coord,
                bounds,
                buffer,
            } => {
                write!(
                    f,
                    "Coord ({:.6}, {:.6}) outside bounds x:[{:.6}, {:.6}] y:[{:.6}, {:.6}] (buffer: {:.6})",
                    coord.x, coord.y,
                    bounds.lng_min - buffer, bounds.lng_max + buffer,
                    bounds.lat_min - buffer, bounds.lat_max + buffer,
                    buffer,
                )
            }
            CorrectnessIssue::RingNotClosed { first, last } => {
                write!(
                    f,
                    "Ring not closed: first ({:.6}, {:.6}) != last ({:.6}, {:.6})",
                    first.x, first.y, last.x, last.y
                )
            }
            CorrectnessIssue::IncorrectWinding {
                ring_type,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "{} has incorrect winding: expected {}, got {}",
                    ring_type, expected, actual
                )
            }
            CorrectnessIssue::InsufficientPoints { count } => {
                write!(
                    f,
                    "Ring has only {} points (need >= 4 for closed ring)",
                    count
                )
            }
            CorrectnessIssue::DegeneratePolygon => {
                write!(f, "Polygon has zero area (degenerate)")
            }
        }
    }
}

/// Check if all coordinates in a geometry are within the given bounds (plus buffer).
fn validate_within_bounds(
    geom: &Geometry<f64>,
    bounds: &TileBounds,
    buffer: f64,
) -> Vec<CorrectnessIssue> {
    let mut issues = Vec::new();

    let buffered_min_x = bounds.lng_min - buffer - COORD_EPSILON;
    let buffered_max_x = bounds.lng_max + buffer + COORD_EPSILON;
    let buffered_min_y = bounds.lat_min - buffer - COORD_EPSILON;
    let buffered_max_y = bounds.lat_max + buffer + COORD_EPSILON;

    let check_coord = |coord: &Coord<f64>| -> Option<CorrectnessIssue> {
        if coord.x < buffered_min_x
            || coord.x > buffered_max_x
            || coord.y < buffered_min_y
            || coord.y > buffered_max_y
        {
            Some(CorrectnessIssue::CoordinateOutsideBounds {
                coord: *coord,
                bounds: *bounds,
                buffer,
            })
        } else {
            None
        }
    };

    match geom {
        Geometry::Point(p) => {
            if let Some(issue) = check_coord(&Coord { x: p.x(), y: p.y() }) {
                issues.push(issue);
            }
        }
        Geometry::LineString(ls) => {
            for coord in ls.coords() {
                if let Some(issue) = check_coord(coord) {
                    issues.push(issue);
                }
            }
        }
        Geometry::Polygon(poly) => {
            for coord in poly.exterior().coords() {
                if let Some(issue) = check_coord(coord) {
                    issues.push(issue);
                }
            }
            for interior in poly.interiors() {
                for coord in interior.coords() {
                    if let Some(issue) = check_coord(coord) {
                        issues.push(issue);
                    }
                }
            }
        }
        Geometry::MultiPolygon(mp) => {
            for poly in &mp.0 {
                issues.extend(validate_within_bounds(
                    &Geometry::Polygon(poly.clone()),
                    bounds,
                    buffer,
                ));
            }
        }
        Geometry::MultiLineString(mls) => {
            for ls in &mls.0 {
                issues.extend(validate_within_bounds(
                    &Geometry::LineString(ls.clone()),
                    bounds,
                    buffer,
                ));
            }
        }
        _ => {} // Other geometry types not checked
    }

    issues
}

/// Check if polygon rings are properly closed (first coord == last coord).
fn validate_rings_closed(poly: &Polygon<f64>) -> Vec<CorrectnessIssue> {
    let mut issues = Vec::new();

    // Check exterior ring
    let exterior = poly.exterior();
    if exterior.0.len() < 4 {
        issues.push(CorrectnessIssue::InsufficientPoints {
            count: exterior.0.len(),
        });
    } else if let (Some(first), Some(last)) = (exterior.0.first(), exterior.0.last()) {
        if (first.x - last.x).abs() > COORD_EPSILON || (first.y - last.y).abs() > COORD_EPSILON {
            issues.push(CorrectnessIssue::RingNotClosed {
                first: *first,
                last: *last,
            });
        }
    }

    // Check interior rings (holes)
    for interior in poly.interiors() {
        if interior.0.len() < 4 {
            issues.push(CorrectnessIssue::InsufficientPoints {
                count: interior.0.len(),
            });
        } else if let (Some(first), Some(last)) = (interior.0.first(), interior.0.last()) {
            if (first.x - last.x).abs() > COORD_EPSILON || (first.y - last.y).abs() > COORD_EPSILON
            {
                issues.push(CorrectnessIssue::RingNotClosed {
                    first: *first,
                    last: *last,
                });
            }
        }
    }

    issues
}

/// Check if polygon has correct winding order (OGC convention).
/// Exterior rings should be counter-clockwise (positive signed area).
/// Interior rings (holes) should be clockwise (negative signed area).
fn validate_winding_order(poly: &Polygon<f64>) -> Vec<CorrectnessIssue> {
    let mut issues = Vec::new();

    // Use signed area to determine winding order
    // geo uses the shoelace formula: positive = CCW, negative = CW
    let exterior_area = signed_ring_area(poly.exterior());

    // Exterior should be CCW (positive area in geo's convention)
    if exterior_area < 0.0 {
        issues.push(CorrectnessIssue::IncorrectWinding {
            ring_type: "Exterior ring",
            expected: "counter-clockwise (CCW)",
            actual: "clockwise (CW)",
        });
    } else if exterior_area.abs() < COORD_EPSILON {
        issues.push(CorrectnessIssue::DegeneratePolygon);
    }

    // Interior rings (holes) should be CW (negative area)
    for (i, interior) in poly.interiors().iter().enumerate() {
        let interior_area = signed_ring_area(interior);
        if interior_area > 0.0 {
            issues.push(CorrectnessIssue::IncorrectWinding {
                ring_type: "Interior ring",
                expected: "clockwise (CW)",
                actual: "counter-clockwise (CCW)",
            });
        }
        let _ = i; // Suppress unused warning
    }

    issues
}

/// Calculate signed area of a ring using the shoelace formula.
/// Positive = CCW, Negative = CW (in standard math coordinates where Y increases upward).
fn signed_ring_area(ring: &LineString<f64>) -> f64 {
    if ring.0.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    let n = ring.0.len();
    for i in 0..n {
        let j = (i + 1) % n;
        area += ring.0[i].x * ring.0[j].y;
        area -= ring.0[j].x * ring.0[i].y;
    }
    area / 2.0
}

/// Validate all correctness properties of a clipped geometry.
fn validate_clipped_geometry(
    geom: &Geometry<f64>,
    bounds: &TileBounds,
    buffer: f64,
) -> Vec<CorrectnessIssue> {
    let mut issues = Vec::new();

    // Check coordinates are within bounds
    issues.extend(validate_within_bounds(geom, bounds, buffer));

    // Check polygon-specific properties
    match geom {
        Geometry::Polygon(poly) => {
            issues.extend(validate_rings_closed(poly));
            issues.extend(validate_winding_order(poly));
        }
        Geometry::MultiPolygon(mp) => {
            for poly in &mp.0 {
                issues.extend(validate_rings_closed(poly));
                issues.extend(validate_winding_order(poly));
            }
        }
        _ => {}
    }

    issues
}

/// Extended test result including correctness information
#[derive(Debug)]
struct CorrectnessResult {
    /// Did clipping complete without panic?
    clipped_ok: bool,
    /// Was there output (Some) or was it rejected (None)?
    has_output: bool,
    /// Correctness issues found in the output
    issues: Vec<CorrectnessIssue>,
}

/// Test clipping and validate correctness of output
fn test_clip_correctness(
    geom: &Geometry<f64>,
    bounds: &TileBounds,
    buffer: f64,
) -> CorrectnessResult {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        clip_geometry(geom, bounds, buffer)
    }));

    match result {
        Ok(Some(clipped)) => {
            let issues = validate_clipped_geometry(&clipped, bounds, buffer);
            CorrectnessResult {
                clipped_ok: true,
                has_output: true,
                issues,
            }
        }
        Ok(None) => CorrectnessResult {
            clipped_ok: true,
            has_output: false,
            issues: vec![],
        },
        Err(_) => CorrectnessResult {
            clipped_ok: false,
            has_output: false,
            issues: vec![],
        },
    }
}

/// Test that documents fuzzer cases separately for detailed analysis
#[test]
fn test_fuzzer_fixtures() {
    let base = fixture_base_path();
    let polyjson_dir = base.join("input-polyjson");

    println!("\n=== Testing Fuzzer-Discovered Cases ===\n");

    if !polyjson_dir.exists() {
        println!("Fixture directory does not exist: {:?}", polyjson_dir);
        println!("\n=== Fuzzer fixtures: skipped ===\n");
        return;
    }

    // Focus on fuzzer-* files specifically
    let entries: Vec<_> = fs::read_dir(&polyjson_dir)
        .expect("Failed to read polyjson directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .file_name()
                .map(|n| n.to_string_lossy().starts_with("fuzzer-"))
                .unwrap_or(false)
        })
        .collect();

    let mut passed = 0;
    let mut failed = Vec::new();
    let mut parse_errors = 0;

    for entry in &entries {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();

        match parse_polyjson_file(&path) {
            Ok(geom) => match test_clip(&geom, &pixel_bounds()) {
                FixtureResult::Pass => passed += 1,
                FixtureResult::Panic(msg) => {
                    failed.push(format!("{}: {}", name, msg));
                }
                FixtureResult::ParseError(_) => parse_errors += 1,
            },
            Err(_) => parse_errors += 1,
        }
    }

    println!("Fuzzer fixtures found: {}", entries.len());
    println!("Passed: {}", passed);
    println!("Failed (panic): {}", failed.len());
    println!("Parse errors: {}", parse_errors);

    if !failed.is_empty() {
        println!("\nFailed fuzzer fixtures:");
        for f in &failed {
            println!("  - {}", f);
        }
    }

    println!(
        "\n=== Fuzzer fixtures: {}/{} passed ===\n",
        passed,
        passed + failed.len()
    );
}

// ============================================================================
// CORRECTNESS TESTS
// ============================================================================

/// Test clipping a simple square that gets cut in half.
/// Expected: The result should be a rectangle covering the intersection.
#[test]
fn test_correctness_square_clipped_in_half() {
    println!("\n=== Correctness: Square Clipped in Half ===\n");

    // Square from (0,0) to (10,10)
    let square = Geometry::Polygon(Polygon::new(
        LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 10.0, y: 0.0 },
            Coord { x: 10.0, y: 10.0 },
            Coord { x: 0.0, y: 10.0 },
            Coord { x: 0.0, y: 0.0 },
        ]),
        vec![],
    ));

    // Clip bounds: right half only (5,0) to (15,10)
    // Expected result: rectangle (5,0) to (10,10)
    let bounds = TileBounds::new(5.0, 0.0, 15.0, 10.0);

    let result = test_clip_correctness(&square, &bounds, 0.0);

    println!("Clipped OK: {}", result.clipped_ok);
    println!("Has output: {}", result.has_output);
    println!("Issues: {}", result.issues.len());

    for issue in &result.issues {
        println!("  - {}", issue);
    }

    assert!(result.clipped_ok, "Clipping should not panic");
    assert!(result.has_output, "Square should intersect bounds");

    // All output coordinates should be within bounds
    let bound_issues: Vec<_> = result
        .issues
        .iter()
        .filter(|i| matches!(i, CorrectnessIssue::CoordinateOutsideBounds { .. }))
        .collect();
    assert!(
        bound_issues.is_empty(),
        "All coordinates should be within bounds: {:?}",
        bound_issues
    );

    println!("\n✓ Square clipped correctly\n");
}

/// Test clipping a triangle with one vertex outside bounds.
/// The result should be a quadrilateral (the triangle with corner cut off).
#[test]
fn test_correctness_triangle_one_vertex_outside() {
    println!("\n=== Correctness: Triangle with One Vertex Outside ===\n");

    // Triangle with vertices at (0,0), (10,0), (5,10)
    // The top vertex (5,10) is outside if bounds.lat_max = 5
    let triangle = Geometry::Polygon(Polygon::new(
        LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 10.0, y: 0.0 },
            Coord { x: 5.0, y: 10.0 },
            Coord { x: 0.0, y: 0.0 },
        ]),
        vec![],
    ));

    // Bounds that cut off the top of the triangle
    let bounds = TileBounds::new(-1.0, -1.0, 11.0, 5.0);

    let result = test_clip_correctness(&triangle, &bounds, 0.0);

    println!("Clipped OK: {}", result.clipped_ok);
    println!("Has output: {}", result.has_output);
    println!("Issues: {}", result.issues.len());

    for issue in &result.issues {
        println!("  - {}", issue);
    }

    assert!(result.clipped_ok, "Clipping should not panic");
    assert!(result.has_output, "Triangle should intersect bounds");

    // Check bounds
    let bound_issues: Vec<_> = result
        .issues
        .iter()
        .filter(|i| matches!(i, CorrectnessIssue::CoordinateOutsideBounds { .. }))
        .collect();
    assert!(
        bound_issues.is_empty(),
        "All coordinates should be within bounds: {:?}",
        bound_issues
    );

    println!("\n✓ Triangle clipped correctly\n");
}

/// Test clipping a self-intersecting bowtie polygon.
/// Document what output we actually get (correctness may be poor).
#[test]
fn test_correctness_bowtie_self_intersecting() {
    println!("\n=== Correctness: Self-Intersecting Bowtie ===\n");

    // Bowtie (figure-8) polygon - edges cross at center
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

    // Bounds that will force clipping (only right half)
    let bounds = TileBounds::new(5.0, 0.0, 15.0, 10.0);

    let result = test_clip_correctness(&bowtie, &bounds, 0.0);

    println!("Clipped OK: {}", result.clipped_ok);
    println!("Has output: {}", result.has_output);
    println!("Issues: {}", result.issues.len());

    for issue in &result.issues {
        println!("  - {}", issue);
    }

    if result.has_output {
        println!("\nNOTE: Self-intersecting input produced output.");
        println!("Sutherland-Hodgman may produce incorrect results for this case.");
    }

    // We don't assert correctness here - we're documenting behavior
    println!("\n⚠ Bowtie test completed (documenting behavior, not asserting correctness)\n");
}

/// Test clipping a polygon with a spike (self-touching ring).
#[test]
fn test_correctness_spike_polygon() {
    println!("\n=== Correctness: Polygon with Spike ===\n");

    // Polygon with a spike: goes up and back down to same point
    let spike = Geometry::Polygon(Polygon::new(
        LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 10.0, y: 0.0 },
            Coord { x: 10.0, y: 5.0 },
            Coord { x: 5.0, y: 5.0 },
            Coord { x: 5.0, y: 10.0 }, // spike up
            Coord { x: 5.0, y: 5.0 },  // back to same point
            Coord { x: 0.0, y: 5.0 },
            Coord { x: 0.0, y: 0.0 },
        ]),
        vec![],
    ));

    // Bounds that clip off the spike
    let bounds = TileBounds::new(-1.0, -1.0, 11.0, 7.0);

    let result = test_clip_correctness(&spike, &bounds, 0.0);

    println!("Clipped OK: {}", result.clipped_ok);
    println!("Has output: {}", result.has_output);
    println!("Issues: {}", result.issues.len());

    for issue in &result.issues {
        println!("  - {}", issue);
    }

    println!("\n⚠ Spike polygon test completed (documenting behavior)\n");
}

/// Comprehensive correctness summary for all curated fixtures.
/// Counts fixtures by category: correct, incorrect output, panic.
#[test]
fn test_correctness_summary_curated() {
    let base = fixture_base_path();
    let input_dir = base.join("input");

    println!("\n=== CORRECTNESS SUMMARY: Curated Fixtures ===\n");

    let bounds = TileBounds::new(-5.0, 39.5, -2.5, 41.5); // Tight bounds to force clipping

    let mut total_fixtures = 0;
    let mut panicked = 0;
    let mut no_output = 0;
    let mut correct_output = 0;
    let mut incorrect_output = 0;
    let mut parse_errors = 0;

    let mut issues_by_type: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    let mut incorrect_fixtures: Vec<(String, Vec<CorrectnessIssue>)> = Vec::new();

    if !input_dir.exists() {
        println!("Fixture directory does not exist: {:?}", input_dir);
        return;
    }

    let entries: Vec<_> = fs::read_dir(&input_dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
        })
        .collect();

    for entry in &entries {
        let path = entry.path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();

        match parse_geojson_file(&path) {
            Ok(geom) => {
                total_fixtures += 1;
                let result = test_clip_correctness(&geom, &bounds, 0.0);

                if !result.clipped_ok {
                    panicked += 1;
                } else if !result.has_output {
                    no_output += 1;
                } else if result.issues.is_empty() {
                    correct_output += 1;
                } else {
                    incorrect_output += 1;
                    incorrect_fixtures.push((name.clone(), result.issues.clone()));

                    // Categorize issues
                    for issue in &result.issues {
                        let issue_type = match issue {
                            CorrectnessIssue::CoordinateOutsideBounds { .. } => "coords_outside",
                            CorrectnessIssue::RingNotClosed { .. } => "ring_not_closed",
                            CorrectnessIssue::IncorrectWinding { .. } => "bad_winding",
                            CorrectnessIssue::InsufficientPoints { .. } => "too_few_points",
                            CorrectnessIssue::DegeneratePolygon => "degenerate",
                        };
                        *issues_by_type.entry(issue_type.to_string()).or_insert(0) += 1;
                    }
                }
            }
            Err(_) => {
                parse_errors += 1;
            }
        }
    }

    println!("Total fixtures tested: {}", total_fixtures);
    println!("Parse errors (skipped): {}", parse_errors);
    println!();
    println!("Results:");
    println!("  ✓ Correct output:   {}", correct_output);
    println!("  ✗ Incorrect output: {}", incorrect_output);
    println!("  ◌ No output (None): {}", no_output);
    println!("  ✗ Panicked:         {}", panicked);
    println!();

    if !issues_by_type.is_empty() {
        println!("Issue breakdown:");
        for (issue_type, count) in &issues_by_type {
            println!("  - {}: {}", issue_type, count);
        }
        println!();
    }

    if !incorrect_fixtures.is_empty() {
        println!("Fixtures with incorrect output:");
        for (name, issues) in &incorrect_fixtures {
            println!("  {}:", name);
            for issue in issues.iter().take(3) {
                println!("    - {}", issue);
            }
            if issues.len() > 3 {
                println!("    ... and {} more issues", issues.len() - 3);
            }
        }
    }

    println!("\n=== End Correctness Summary ===\n");
}

/// Comprehensive correctness summary for polyjson (fuzzer) fixtures.
#[test]
fn test_correctness_summary_polyjson() {
    let base = fixture_base_path();
    let polyjson_dir = base.join("input-polyjson");

    println!("\n=== CORRECTNESS SUMMARY: Polyjson Fixtures ===\n");

    // Bounds that will clip many fixtures
    let bounds = TileBounds::new(10.0, 10.0, 35.0, 35.0);

    let mut total_fixtures = 0;
    let mut panicked = 0;
    let mut no_output = 0;
    let mut correct_output = 0;
    let mut incorrect_output = 0;
    let mut parse_errors = 0;

    let mut issues_by_type: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();

    if !polyjson_dir.exists() {
        println!("Fixture directory does not exist: {:?}", polyjson_dir);
        return;
    }

    let entries: Vec<_> = fs::read_dir(&polyjson_dir)
        .expect("Failed to read directory")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "json")
                .unwrap_or(false)
        })
        .collect();

    for entry in &entries {
        let path = entry.path();

        match parse_polyjson_file(&path) {
            Ok(geom) => {
                total_fixtures += 1;
                let result = test_clip_correctness(&geom, &bounds, 0.0);

                if !result.clipped_ok {
                    panicked += 1;
                } else if !result.has_output {
                    no_output += 1;
                } else if result.issues.is_empty() {
                    correct_output += 1;
                } else {
                    incorrect_output += 1;

                    // Categorize issues
                    for issue in &result.issues {
                        let issue_type = match issue {
                            CorrectnessIssue::CoordinateOutsideBounds { .. } => "coords_outside",
                            CorrectnessIssue::RingNotClosed { .. } => "ring_not_closed",
                            CorrectnessIssue::IncorrectWinding { .. } => "bad_winding",
                            CorrectnessIssue::InsufficientPoints { .. } => "too_few_points",
                            CorrectnessIssue::DegeneratePolygon => "degenerate",
                        };
                        *issues_by_type.entry(issue_type.to_string()).or_insert(0) += 1;
                    }
                }
            }
            Err(_) => {
                parse_errors += 1;
            }
        }
    }

    println!("Total fixtures tested: {}", total_fixtures);
    println!("Parse errors (skipped): {}", parse_errors);
    println!();
    println!("Results:");
    println!("  ✓ Correct output:   {}", correct_output);
    println!("  ✗ Incorrect output: {}", incorrect_output);
    println!("  ◌ No output (None): {}", no_output);
    println!("  ✗ Panicked:         {}", panicked);
    println!();

    if !issues_by_type.is_empty() {
        println!("Issue breakdown:");
        for (issue_type, count) in &issues_by_type {
            println!("  - {}: {}", issue_type, count);
        }
    }

    println!("\n=== End Correctness Summary ===\n");
}

/// Test that validates simple, expected clipping behavior works correctly.
/// This serves as a sanity check that the validation helpers themselves are working.
#[test]
fn test_correctness_validation_sanity_check() {
    println!("\n=== Sanity Check: Validation Helpers ===\n");

    // A simple square fully inside bounds should have zero issues
    let square = Polygon::new(
        LineString::from(vec![
            Coord { x: 2.0, y: 2.0 },
            Coord { x: 8.0, y: 2.0 },
            Coord { x: 8.0, y: 8.0 },
            Coord { x: 2.0, y: 8.0 },
            Coord { x: 2.0, y: 2.0 },
        ]),
        vec![],
    );

    let bounds = TileBounds::new(0.0, 0.0, 10.0, 10.0);

    // Test bounds validation
    let bounds_issues = validate_within_bounds(&Geometry::Polygon(square.clone()), &bounds, 0.0);
    assert!(
        bounds_issues.is_empty(),
        "Square inside bounds should have no bounds issues: {:?}",
        bounds_issues
    );

    // Test ring closure validation
    let closure_issues = validate_rings_closed(&square);
    assert!(
        closure_issues.is_empty(),
        "Properly closed ring should have no closure issues: {:?}",
        closure_issues
    );

    // Test winding order validation
    let winding_issues = validate_winding_order(&square);
    assert!(
        winding_issues.is_empty(),
        "CCW exterior should have no winding issues: {:?}",
        winding_issues
    );

    println!("✓ All validation helpers working correctly\n");
}

/// Test winding order detection with intentionally wrong winding.
#[test]
fn test_correctness_winding_detection() {
    println!("\n=== Winding Order Detection ===\n");

    // CW exterior (wrong for OGC)
    let cw_exterior = Polygon::new(
        LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 0.0, y: 10.0 },
            Coord { x: 10.0, y: 10.0 },
            Coord { x: 10.0, y: 0.0 },
            Coord { x: 0.0, y: 0.0 },
        ]),
        vec![],
    );

    let issues = validate_winding_order(&cw_exterior);
    println!("CW exterior issues: {:?}", issues);

    let has_winding_issue = issues
        .iter()
        .any(|i| matches!(i, CorrectnessIssue::IncorrectWinding { .. }));

    assert!(
        has_winding_issue,
        "CW exterior should be detected as incorrect winding"
    );

    // CCW exterior (correct for OGC)
    let ccw_exterior = Polygon::new(
        LineString::from(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 10.0, y: 0.0 },
            Coord { x: 10.0, y: 10.0 },
            Coord { x: 0.0, y: 10.0 },
            Coord { x: 0.0, y: 0.0 },
        ]),
        vec![],
    );

    let issues = validate_winding_order(&ccw_exterior);
    println!("CCW exterior issues: {:?}", issues);

    assert!(
        issues.is_empty(),
        "CCW exterior should have no winding issues"
    );

    println!("\n✓ Winding detection working correctly\n");
}
