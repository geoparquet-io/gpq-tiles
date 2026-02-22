# gpq-tiles - Claude Code Instructions

## Project Overview

GeoParquet → PMTiles converter in Rust. Library-first design with CLI and Python bindings.

**Goal:** Faster than tippecanoe for typical GeoParquet workflows, with native Arrow integration.

**Status:** Phase 4 complete (262 tests). See `ROADMAP.md` for details.

## Documentation Philosophy

**Prefer concise, DRY documentation. One doc, one purpose.**

| Document | Purpose | When to Update |
|----------|---------|----------------|
| `README.md` | Quick start, links to other docs | Major user-facing changes |
| `ROADMAP.md` | Implementation phases, progress tracking | Phase milestones, test counts |
| `docs/ARCHITECTURE.md` | Design decisions, tippecanoe divergences | Algorithm changes, new divergences |
| `CLAUDE.md` | AI assistant instructions | Process changes, new pitfalls |

**Rules:**
- Keep README lean — link out, don't duplicate
- Never duplicate content across docs
- Delete docs that serve no unique purpose
- Update test counts in ROADMAP.md, not elsewhere

## Critical Constraints

### 1. Test-Driven Development (TDD) is MANDATORY

Every feature follows: **failing test → implementation → refactor**

```bash
cargo test --package gpq-tiles-core <module> -- --nocapture  # Verify red
# Implement
cargo test --package gpq-tiles-core <module> -- --nocapture  # Verify green
git commit -m "feat: implement X (TDD green)"
```

### 2. Arrow/GeoArrow is the Data Layer

**The entire point of this library is zero-copy Arrow integration.**

DO:
```rust
// Process geometries within Arrow batch lifetime
for batch in reader {
    let geom_col = batch.column(geom_idx);
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

**Exception:** Complex operations (boolean clipping) may require temporary `geo::Geometry` conversion.

### 3. Reference Implementations (CRITICAL)

**All algorithms MUST match tippecanoe behavior as closely as possible.**

- **tippecanoe** (https://github.com/felt/tippecanoe) - PRIMARY reference
- **planetiler** (https://github.com/onthegomap/planetiler) - Secondary reference

**When deviating from tippecanoe:**
```rust
// DIVERGENCE FROM TIPPECANOE: [reason]
// Tippecanoe does X (see tile.cpp:L312)
// We do Y because [Rust limitation / performance / etc.]
```

Document all divergences in `docs/ARCHITECTURE.md`.

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
TileCoord { x: u32, y: u32, z: u8 }           // Tile coordinates (Web Mercator)
TileBounds { lng_min, lat_min, lng_max, lat_max }  // Geographic bounds
TilerConfig { min_zoom, max_zoom, ... }       // Pipeline configuration
```

## Commands

```bash
cargo build                   # Build
cargo test                    # Test (262 tests)
cargo bench                   # Benchmarks
cargo fmt --all               # Format (required before commit)
cargo run --package gpq-tiles -- input.parquet output.pmtiles
```

## Common Pitfalls

1. **geozero vs geoarrow**: Don't use geozero for bulk geometry extraction
2. **BooleanOps signature**: `polygon.clip(&linestring)`, not reverse
3. **PMTiles crate**: Read-only — we implement our own v3 writer
4. **CI workflow**: Use `dtolnay/rust-toolchain`, not `rust-action`
5. **rstar**: Listed in deps but we use space-filling curves for spatial indexing instead

## Setup

```bash
git config core.hooksPath .githooks  # Enable pre-commit hooks
```
