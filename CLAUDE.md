# gpq-tiles - Claude Code Instructions

## Project Overview

GeoParquet → PMTiles converter in Rust. Library-first design with CLI and Python bindings.

**Goal:** Faster than tippecanoe for typical GeoParquet workflows, with native Arrow integration.

## Critical Constraints

### 1. Test-Driven Development (TDD) is MANDATORY

Every feature follows: **failing test → implementation → refactor**

```bash
# Workflow
cargo test --package gpq-tiles-core <module> -- --nocapture  # Verify red
# Implement
cargo test --package gpq-tiles-core <module> -- --nocapture  # Verify green
git commit -m "feat: implement X (TDD green)"
```

Do NOT write implementation code without a failing test first.

### 2. Arrow/GeoArrow is the Data Layer

**The entire point of this library is zero-copy Arrow integration.**

DO:
```rust
// Process geometries within Arrow batch lifetime
for batch in reader {
    let geom_col = batch.column(geom_idx);
    // Use geoarrow accessors - no heap allocation per feature
    let polygon_array = PolygonArray::try_from(geom_col)?;
    for i in 0..polygon_array.len() {
        let poly_ref = polygon_array.value(i);  // Borrows from batch
        // Process immediately, write to tile
    }
}
```

DO NOT:
```rust
// WRONG: Deserializing every geometry defeats Arrow's purpose
let geom: geo::Geometry = geozero::wkb::Wkb(wkb.to_vec()).to_geo();
geometries.push((geom, row_offset + i));  // Heap allocation per feature!
```

**Exception:** Complex operations (boolean clipping) may require temporary `geo::Geometry` conversion, but this should be minimized and documented.

### 3. Reference Implementations

Study these for algorithm decisions:

- **tippecanoe** (https://github.com/felt/tippecanoe) - Feature dropping, simplification strategies
- **planetiler** (https://github.com/onthegomap/planetiler) - Java, but excellent architecture for streaming tile generation

When in doubt about how to handle edge cases (degenerate geometries, antimeridian, etc.), check how tippecanoe handles it.

## Architecture

```
crates/
├── core/     # ALL tiling logic lives here
├── cli/      # Thin wrapper: args → core::convert()
└── python/   # pyo3 bindings → core
```

**Library-first:** CLI and Python are thin consumers. Never put logic in CLI/Python that belongs in core.

## Key Types

```rust
// Tile coordinates (Web Mercator)
TileCoord { x: u32, y: u32, z: u8 }

// Geographic bounds
TileBounds { lng_min, lat_min, lng_max, lat_max }

// Converter configuration
Config { min_zoom, max_zoom, extent, drop_density }
```

## Testing

```bash
# Fast iteration
cargo watch -x "test --lib"

# Full suite
cargo test

# Specific module
cargo test --package gpq-tiles-core <module> -- --nocapture

# With coverage
cargo tarpaulin --out html
```

### Test Fixtures

- `tests/fixtures/realdata/` - Real GeoParquet files for integration tests
- `tests/fixtures/golden/` - tippecanoe-generated PMTiles for comparison

### Property-Based Tests

Use `proptest` for geometry operations:
- MVT round-trip (encode → decode ≈ original)
- Tile coordinate invariants
- Simplification validity

## Dependencies

| Crate | Purpose |
|-------|---------|
| `geoarrow` | Zero-copy geometry access from Arrow |
| `geoparquet` | GeoParquet metadata parsing |
| `geo` | Geometry algorithms (clipping, simplification) |
| `prost` | MVT protobuf encoding |
| `rayon` | Parallel tile generation (Phase 4) |
| `rstar` | R-tree spatial indexing |

**Note:** The `pmtiles` crate is READ-ONLY. We implement our own PMTiles v3 writer.

## Common Pitfalls

1. **geozero vs geoarrow**: Don't use geozero for bulk geometry extraction. It defeats Arrow's zero-copy benefits.

2. **BooleanOps signature**: `polygon.clip(&linestring)`, not `linestring.clip(&polygon)`

3. **PMTiles crate**: Read-only - must write custom PMTiles v3 writer from spec

4. **CI workflow**: Use `dtolnay/rust-toolchain`, not `rust-action`

## Current Status

See `ROADMAP.md` for phase details. Currently in Phase 2 (Naive Tiling).

## Setup

After cloning, configure git to use the project's hooks:

```bash
git config core.hooksPath .githooks
```

This enables the pre-commit hook that runs `cargo fmt --check` before each commit.

## Commands

```bash
# Build
cargo build

# Test
cargo test

# Run CLI
cargo run --package gpq-tiles-cli -- input.parquet output.pmtiles

# Benchmarks
cargo bench

# Format code (required before commit)
cargo fmt --all
```
