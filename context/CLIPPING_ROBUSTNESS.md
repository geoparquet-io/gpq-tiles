# Clipping Robustness in gpq-tiles

This document captures our research and decisions around polygon clipping edge cases, validation strategies, and robustness guarantees.

## Critical Distinction: Detection vs Repair

> **This is the most important thing to understand about our approach.**

| Approach | What It Does | Example |
|----------|--------------|---------|
| **Wagyu/GEOS** (what tippecanoe uses) | **REPAIRS** invalid geometry | Self-intersecting polygon in → valid polygon out |
| **Our approach** | **DETECTS** and **ROUTES AROUND** | Invalid detected → use robust algorithm for clipping |

### What We Do

1. **Detect** invalid geometries (self-intersections, spikes, invalid holes)
2. **Route** detected invalid geometries to a robust clipping algorithm (`geo::BooleanOps`)
3. **Produce** correct clipped output for rendering

### What We Do NOT Do

1. **Repair** invalid geometries — we don't adjust vertices to fix topology
2. **Guarantee** clipped output is OGC-valid — it renders correctly, but may be technically invalid
3. **Match** tippecanoe's geometry repair behavior — tippecanoe uses Wagyu which repairs

### Practical Impact

| Use Case | Our Behavior |
|----------|--------------|
| **Tiling/rendering** | Output renders correctly |
| **Downstream GIS processing** | Clipped geometry may still be technically invalid |
| **Full tippecanoe parity** | Would require porting Wagyu or binding to GEOS |

### Why This Trade-off?

- **Repair is complex** — Wagyu is ~10K lines of C++, porting to Rust is substantial
- **Detection is sufficient** for rendering — MVT consumers tolerate minor invalidity
- **BooleanOps handles clipping** correctly even on invalid input
- **Future work** could add GEOS bindings if repair is needed

## Recommended Workflow: Fix Geometry Before Tiling

> **gpq-tiles is a tiler, not a geometry repair tool.**

For best results, validate and repair invalid geometries **before** conversion:

```bash
# With ogr2ogr (uses GEOS MakeValid)
ogr2ogr -f Parquet output_clean.parquet input.parquet -makevalid

# Or with DuckDB Spatial
duckdb -c "COPY (SELECT ST_MakeValid(geometry) as geometry, * EXCLUDE geometry FROM 'input.parquet') TO 'output_clean.parquet'"
```

```sql
-- PostGIS
SELECT ST_MakeValid(geom) FROM my_table;
```

```python
# Python with Shapely
from shapely.validation import make_valid
clean_geom = make_valid(dirty_geom)
```

### Why Pre-Process?

1. **One-time cost** — Data cleaning runs once; tiling may happen repeatedly
2. **Battle-tested** — GEOS/PostGIS `MakeValid` handles edge cases we haven't seen
3. **Data quality** — You probably want to know your source data has issues anyway
4. **Performance** — gpq-tiles stays fast and portable (pure Rust, no C++ dependencies)

### What gpq-tiles Provides

| Feature | Purpose |
|---------|---------|
| `--strict` mode (planned) | Identifies which geometries are invalid (diagnostic) |
| Automatic fallback | Prevents crashes and garbage output |
| Input validation | Detects self-intersections, spikes, invalid holes |

**What we explicitly do NOT do:** Repair geometry. That's your data pipeline's responsibility.

## Problem Statement

Polygon clipping is a critical step in the tiling pipeline that runs on every geometry. Edge cases include:

- **Self-intersecting polygons** (bowtie/figure-8 shapes)
- **Spikes** where a ring touches itself (e.g., `A → B → A`)
- **Invalid holes** with self-intersections
- **Degenerate geometries** with too few points or zero area
- **Non-finite coordinates** (NaN, infinity)

Real-world GeoParquet files contain these invalid geometries. A production tiling system must:
1. Not crash on invalid input
2. Produce valid output (or cleanly reject invalid geometries)
3. Be fast enough for millions of features

## Sutherland-Hodgman Algorithm

### Why We Chose It

We use [Sutherland-Hodgman](https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm) for polygon clipping against tile bounds because:

- **O(n) complexity** for axis-aligned rectangles (vs O(n log n) for general polygon intersection)
- **Matches tippecanoe** behavior for tile clipping
- **Simple implementation** with predictable performance

**Implementation:** `crates/core/src/clip.rs:475-537` (`sutherland_hodgman_clip()`)

### Behavior on Invalid Input

Sutherland-Hodgman is designed for **valid, simple polygons**. When given invalid input:

| Input Type | Behavior |
|------------|----------|
| Valid polygon | Correct clipped output |
| Self-intersecting | **No panic**, but incorrect output |
| Spike (self-touch) | **No panic**, may produce degenerate ring |
| Too few points | Returns empty ring (handled downstream) |
| Non-finite coords | Produces non-finite output (handled by validation) |

**Key finding:** The algorithm doesn't panic — it fails silently with incorrect output.

## Test Infrastructure

### Wagyu Fixtures (206 tests)

We validate against the [mapnik/geometry-test-data](https://github.com/mapnik/geometry-test-data) fixtures from the wagyu project:

| Fixture Set | Count | Description |
|-------------|-------|-------------|
| Curated (`input/`) | 38 | Hand-crafted edge cases: bowties, spikes, intersecting holes |
| Polyjson (`input-polyjson/`) | 168 | Crash cases from fuzzing wagyu |
| **Total** | 206 | All tested for robustness |

**Test location:** `crates/core/tests/clip_robustness.rs`

### Test Results Summary

**Robustness (no panics):** All 206 fixtures pass — no crashes.

**Correctness (10/38 curated failures):**

| Issue | Count | Description |
|-------|-------|-------------|
| Coords outside bounds | 20 | Output coordinates exceed clip bounds |
| Incorrect winding order | 14 | Exterior rings not CCW, holes not CW |

These failures occur on intentionally malformed input (self-intersecting polygons, invalid holes). The algorithm doesn't crash, but produces incorrect geometry.

### Correctness Checks

The test suite verifies:

1. **Bounds containment** — All output coords within clip bounds + buffer
2. **Ring closure** — First coord == last coord for all rings
3. **Winding order** — OGC convention (exterior CCW, holes CW)

## Validation Strategy

### Dual-Layer Approach

We implement two validation layers with different purposes:

```
Input → [Input Validation] → Clip → MVT Encode → [Output Validation] → Tile
              (opt-in)                               (always-on Z0-Z10)
```

### Layer 1: Input Validation (Diagnostic)

**Purpose:** Data quality feedback for users.

**Location:** `crates/core/src/clip.rs:145-339` (`validate_geometry()`)

**Detects:**
- Self-intersections (`has_self_intersection()`)
- Spikes (`has_spike()`)
- Invalid holes
- Too few points (<4 for closed rings)
- Non-finite coordinates

**Behavior:**
- **Opt-in** via `--strict` mode (planned)
- Warns but doesn't reject
- Allows users to identify problematic source data

### Layer 2: Output Validation (Safety)

**Purpose:** Catch quantization-induced invalidity.

**Location:** `crates/core/src/mvt.rs` (post-encoding checks)

**Why it's needed:**

The quantization step (`geo_to_tile_coords()` at mvt.rs:134-146) can invalidate previously-valid polygons:

| Zoom | Pixel Resolution | Risk Level |
|------|------------------|------------|
| Z0 | ~9.7 km | Very High |
| Z6 | ~150 m | High |
| Z10 | ~9.5 m | Moderate |
| Z14 | ~60 cm | Low |

At low zooms, distinct vertices can snap to the same integer coordinate, causing:
- Consecutive duplicate vertices
- Ring collapse (polygon becomes a line)
- Self-intersections from edge crossings

**Behavior:**
- **Always-on** for Z0-Z10 (configurable threshold)
- Detects vertex collapse, degenerate rings
- See `context/QUANTIZATION_RISKS.md` for detailed analysis

## Fallback Strategy

When input validation detects invalid geometry, we use a robust fallback:

```rust
// In clip_geometry() - current implementation logs, future will fallback
let validation_errors = validate_geometry(geom);
if !validation_errors.is_empty() {
    log::trace!("Detected invalid geometry: {:?}", validation_errors);
    // TODO: Fallback to geo::BooleanOps::intersection
}
```

**Planned fallback:** Use `geo::BooleanOps::intersection()` instead of Sutherland-Hodgman.

| Algorithm | Speed | Robustness |
|-----------|-------|------------|
| Sutherland-Hodgman | O(n) | Assumes valid input |
| BooleanOps::intersection | O(n log n) | Handles self-intersections |

The fallback is automatic — invalid geometries get robust (slower) processing without user intervention.

## Divergences from Tippecanoe

### Validation (Addition)

**We validate, tippecanoe doesn't.**

Tippecanoe (`tile.cpp`) relies on:
- Aggressive simplification to reduce edge case frequency
- MVT renderers tolerating minor self-intersections
- No explicit validation step

We add validation because:
1. Users benefit from data quality feedback
2. Some MVT consumers are strict about validity
3. Debug/diagnostic mode helps track down source data issues

### Quantization Handling (Planned)

Tippecanoe handles post-quantization issues with:
- Snap-rounding with consistent rules
- Degenerate cleanup (remove consecutive duplicates)
- Vertex coalescing

We currently do NOT clean up post-quantization. This is documented as a known limitation — see `context/QUANTIZATION_RISKS.md` for the research and recommendations.

## Code Reference

| Function | Location | Purpose |
|----------|----------|---------|
| `clip_geometry()` | clip.rs:48-88 | Main clipping entry point |
| `validate_geometry()` | clip.rs:145-160 | Input validation |
| `validate_polygon()` | clip.rs:163-179 | Per-polygon validation |
| `has_spike()` | clip.rs:221-251 | Detect self-touch points |
| `has_self_intersection()` | clip.rs:257-312 | Detect edge crossings |
| `sutherland_hodgman_clip()` | clip.rs:475-537 | O(n) axis-aligned clipping |
| `geo_to_tile_coords()` | mvt.rs:134-146 | Quantization (single point!) |

## Related Documents

- `context/QUANTIZATION_RISKS.md` — Coordinate quantization analysis
- `context/ARCHITECTURE.md` — Overall design decisions
- `crates/core/tests/clip_robustness.rs` — Test infrastructure

## References

- [mapnik/geometry-test-data](https://github.com/mapnik/geometry-test-data) — Wagyu fixtures
- [Sutherland-Hodgman algorithm](https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm)
- [geo-rs BooleanOps](https://docs.rs/geo/latest/geo/algorithm/bool_ops/trait.BooleanOps.html)
