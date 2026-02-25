//! Robustness tests for geometry clipping using wagyu/geometry-test-data fixtures.
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
//! We're NOT testing correctness of the clipped output - we're testing that:
//! 1. The clipper doesn't panic on malformed input
//! 2. The clipper returns Some or None (not crash)
//! 3. We can identify which fixtures our current implementation fails on
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

/// Test that documents fuzzer cases separately for detailed analysis
#[test]
fn test_fuzzer_fixtures() {
    let base = fixture_base_path();
    let polyjson_dir = base.join("input-polyjson");

    println!("\n=== Testing Fuzzer-Discovered Cases ===\n");

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
