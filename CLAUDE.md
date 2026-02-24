# gpq-tiles - Claude Code Instructions

## Project Overview

GeoParquet → PMTiles converter in Rust. Library-first design with CLI and Python bindings.

**Goal:** Faster than tippecanoe for typical GeoParquet workflows, with native Arrow integration.

**Status:** Phase 5 complete (329 tests). See `ROADMAP.md` for details.

## Documentation Philosophy

**Prefer concise, DRY documentation. One doc, one purpose.**

### Folder Structure

| Folder | Purpose |
|--------|---------|
| `docs/` | Human-facing documentation only (mkdocs site content) |
| `context/` | AI/developer context: architecture, plans, design decisions |

### Key Documents

| Document | Purpose | When to Update |
|----------|---------|----------------|
| `README.md` | Quick start, links to other docs | Major user-facing changes |
| `ROADMAP.md` | Implementation phases, progress tracking | Phase milestones, test counts |
| `context/ARCHITECTURE.md` | Design decisions, tippecanoe divergences | Algorithm changes, new divergences |
| `context/plans/` | Implementation plans and design docs | New features, major changes |
| `DEVELOPMENT.md` | Day-to-day dev workflow, Python setup | New tools, workflow changes |
| `CONTRIBUTING.md` | How to contribute, commit conventions | Process changes |
| `CLAUDE.md` | AI assistant instructions | Process changes, new pitfalls |

**Rules:**
- Keep README lean — link out, don't duplicate
- Never duplicate content across docs
- Delete docs that serve no unique purpose
- Update test counts in ROADMAP.md, not elsewhere
- `docs/` is ONLY for mkdocs human-facing content
- `context/` is for architecture, plans, and AI context

## Critical Constraints

### 1. Test-Driven Development (TDD) is MANDATORY

Every feature follows: **failing test → implementation → refactor**

```bash
cargo test --package gpq-tiles-core <module> -- --nocapture  # Verify red
# Implement
cargo test --package gpq-tiles-core <module> -- --nocapture  # Verify green
git commit -m "feat: implement X (TDD green)"
```

### 2. Arrow/GeoArrow: Columnar I/O, Not Zero-Copy Geometry Processing

**Arrow gives us efficient columnar I/O and streaming, but geometry operations require `geo::Geometry` conversion.**

What we GET from Arrow/GeoArrow:
- Columnar decoding (only geometry column parsed, properties lazy-loaded)
- Row-group streaming (memory = O(row_group), not O(file))
- No double-copy (Arrow → geo directly, not Arrow → WKB → geo)

What we DON'T get (yet):
- Zero-copy clipping (geo::BooleanOps requires owned `geo::Polygon`)
- Zero-copy simplification (geo::Simplify requires owned `geo::LineString`)

DO:
```rust
// Iterate geometries within Arrow batch, convert only what's needed
for batch in reader {
    let geom_col = batch.column(geom_idx);
    let geom_array = geoarrow::array::from_arrow_array(geom_col, geom_field)?;
    for geom in geom_array.iter() {
        let geo_geom: geo::Geometry = geom.try_to_geometry()?;  // Conversion needed for clipping
        let clipped = clip_geometry(&geo_geom, &tile_bounds)?;
        // Process immediately, don't accumulate all features
    }
}
```

DO NOT:
```rust
// WRONG: Deserializing to WKB first defeats Arrow's columnar benefits
let geom: geo::Geometry = geozero::wkb::Wkb(wkb.to_vec()).to_geo();
```

**See also:** `context/plans/2026-02-23-streaming-design.md` for streaming architecture details.

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

Document all divergences in `context/ARCHITECTURE.md`.

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
cargo test                    # Test (262 Rust tests)
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
6. **Python tooling**: Always use `uv` for Python work (not pip/poetry). See `DEVELOPMENT.md` for setup
7. **Streaming vs non-streaming**: `generate_tiles_streaming()` exists but `generate_tiles()` is the default. Streaming is for files larger than memory

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/). See `CONTRIBUTING.md` for details.

```bash
# Examples
feat: add WKT geometry encoding support
fix: guard against degenerate linestrings in simplify
perf(core): parallelize geometry processing
docs: update ROADMAP with Phase 9

# With scope
feat(cli): add --streaming-mode flag
fix(core): prevent OOM on large files
```

## Setup

```bash
git config core.hooksPath .githooks  # Enable pre-commit hooks
```
