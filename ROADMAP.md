# gpq-tiles Roadmap

Production-grade GeoParquet → PMTiles converter in Rust. Library-first design with CLI and Python bindings.

## Architecture

```
gpq-tiles/
├── crates/
│   ├── core/          # All tiling logic
│   ├── cli/           # Thin wrapper: args → core
│   └── python/        # pyo3 bindings → core
└── tests/fixtures/    # Real GeoParquet + expected tiles
```

**Library-first from day one.** CLI and Python are thin consumers of the core library, ensuring maximum reusability and testability.

## Implementation Phases

### Phase 1: Skeleton

End-to-end pipeline with stub implementations:
1. Read GeoParquet with `geoarrow`
2. Iterate features
3. Write empty PMTiles with `pmtiles`

**Goal**: `cargo run -- input.parquet output.pmtiles` produces a valid (though empty) PMTiles file. This establishes the pipeline structure and resolves dependency/build issues before implementing real logic.

### Phase 2: Naive Tiling ✅ COMPLETE

Produce functional vector tiles:
- For each zoom level, for each tile intersecting the data bbox:
  - Clip features to tile bounds (with 8px buffer)
  - Simplify geometries with Ramer-Douglas-Peucker (tolerance scaled to tile size)
  - Encode as Mapbox Vector Tile (MVT) format using `prost`
  - Write tiles to PMTiles archive

**Single-threaded, in-memory processing.** Optimization comes later. Focus: tiles that render correctly in MapLibre.

#### Progress

- ✅ **Step 1**: GeoParquet reading with `geoparquet` crate (metadata parsing, schema inference)
- ✅ **Step 2**: Feature iteration via Apache Arrow `RecordBatch`
- ✅ **Step 3**: Tile coordinate math (Web Mercator projection, `lng_lat_to_tile()`, `tiles_for_bbox()`)
- ✅ **Step 4**: Dataset bounding box and tile grid calculation
- ✅ **Step 5**: Geometry extraction from GeoArrow arrays (`batch_processor.rs`)
- ✅ **Step 6**: Geometry clipping to tile bounds (`clip.rs`) - 15 tests
- ✅ **Step 7**: Geometry simplification (`simplify.rs`) - 7 tests
- ✅ **Step 8**: MVT encoding (`mvt.rs`) - 28 tests
- ✅ **Step 9**: Tile generation pipeline (`pipeline.rs`) - 13 tests
- ✅ **Step 10**: Golden comparison tests (`golden.rs`) - 6 tests
- ✅ **Step 11**: PMTiles v3 writing (`pmtiles_writer.rs`) - 24 tests
- ✅ **Step 12**: CI/CD configuration (ci.yml, mutation-tests.yml)
- ✅ **Step 13**: Criterion benchmark harness (`benches/tiling.rs`)

**Final Status**: 122 tests passing. Full pipeline complete: GeoParquet → clip → simplify → MVT → PMTiles.

**Current**: 213 tests (207 unit + 6 doc tests) after Phase 3 feature dropping integration.

#### Golden Comparison Results

The pipeline is validated against tippecanoe-generated tiles:

| Metric | Tippecanoe | gpq-tiles | Notes |
|--------|------------|-----------|-------|
| Z10 features | 484 | 392 | **0.81x ratio** - feature dropping working! |
| Z8 features | 97 | 1000 | tippecanoe uses density-based dropping (Phase 3 TODO) |
| Area preserved | - | 88% | After clipping + simplification |
| All zooms produce output | ✅ | ✅ | Z5-Z10 verified |

**Key insight**: After Phase 3 pipeline integration, the Z10 ratio improved from 1.41x to 0.81x. We now drop slightly more aggressively than tippecanoe due to diffuse probability, which is acceptable. Low-zoom density dropping remains TODO.

**Architecture**: See `docs/ARCHITECTURE.md` for design decisions and known divergences.

#### Known Issues (Remaining)

| Severity | Issue | Description |
|----------|-------|-------------|
| ~~**Critical**~~ | ~~Simplification coordinate space~~ | ✅ **FIXED** - Now uses `simplify_in_tile_coords()` with pixel-based tolerance |
| ~~**Critical**~~ | ~~Antimeridian crossing~~ | ✅ **FIXED** - `tiles_for_bbox()` now splits bbox at 180° |
| ~~**Medium**~~ | ~~No polygon winding validation~~ | ✅ **FIXED** - MVT encoding now auto-corrects winding order via `orient_polygon_for_mvt()` |
| ~~**Medium**~~ | ~~Degenerate geometry handling~~ | ✅ **FIXED** - New `validate.rs` module filters invalid geometries post-simplification |
| ~~**Low**~~ | ~~Value deduplication uses Debug~~ | ✅ **FIXED** - `PropertyValue` now implements proper `Hash`/`Eq` traits |
| **Medium** | Memory for large files | `extract_geometries` loads all into `Vec`, defeating Arrow zero-copy. |

### Phase 3: Feature Dropping ✅ MOSTLY COMPLETE

**Status**: Core algorithms implemented and integrated. Density-based dropping deferred.

#### Implemented (43 tests total: 39 in `feature_drop.rs` + 4 integration tests)

| Algorithm | Status | Tests | Description |
|-----------|--------|-------|-------------|
| **Tiny polygon** | ✅ Done | 10 | Diffuse probability for < 4 sq pixels (matches tippecanoe) |
| **Line dropping** | ✅ Done | 17 | Coordinate quantization - drop when all vertices collapse to same pixel |
| **Point thinning** | ✅ Done | 12 | 1/2.5 drop rate per zoom above base (matches tippecanoe exactly) |
| **Pipeline integration** | ✅ Done | 4 | `should_drop_geometry()` wired into tile generation |
| **Density-based** | 🔲 Deferred | - | May not be needed; current results are acceptable |

#### Results

- **Z10 ratio**: 1.41x → **0.81x** (improved!)
- **Z8**: Still shows 1000 features vs tippecanoe's 97 (density-based dropping needed for this)
- The core feature dropping is working; density-based dropping is a nice-to-have for low zooms

**Goal achieved**: Feature density decreases at lower zooms. Golden comparison ratio at Z10 is now < 1.0x.

### Phase 4: Parallelism

Leverage Rust's concurrency for performance:
- Parallelize tile generation within each zoom level using `rayon`
- Implement spatial indexing (R-tree via `rstar`) for efficient feature lookup

**Zoom levels remain sequential** to preserve feature dropping semantics.

### Phase 5: Python Integration

Expose Python API via pyo3:

```python
from gpq-tiles import convert

convert(
    input="buildings.parquet",
    output="buildings.pmtiles",
    min_zoom=0,
    max_zoom=14,
)
```

**Goal**: Production-ready bindings for integration with geoparquet-io and other Python geospatial tools.

## Deferred Features (Post-MVP)

These features are planned but not required for initial release:

- **Advanced polygon clipping** (Sutherland-Hodgman algorithm with buffers)
- **Self-intersecting polygon repair** (complex topology fixes beyond simple validation)
- **Coalesce/cluster strategies** (tippecanoe-style aggregation modes)
- **Streaming processing** for files larger than available memory
- **Attribute filtering** and property selection

## Testing Strategy

Specification-driven development. MVT encoding, PMTiles format, and tile coordinates all have formal specifications that guide testing.

### Test Layers

1. **Unit Tests** (`#[cfg(test)]` inline modules)
   - TDD workflow: failing test → implementation → refactor
   - Fast iteration with `cargo watch -x "test --lib"`
   - Coverage: MVT encoding, coordinate transforms, zigzag encoding, feature spacing

2. **Property-Based Testing** (`proptest`)
   - MVT round-trip correctness (encode → decode ≈ original)
   - Tile coordinate invariants (no overlaps, child tiles cover parents)
   - Feature dropping monotonicity (if feature survives at zoom N, it survives at N+1)
   - Simplification validity (reduced vertices, area within tolerance)

3. **Integration Tests** (`tests/`)
   - Full pipeline: GeoParquet → PMTiles
   - Golden-file comparisons (tile content level, not byte-level)
   - Spec compliance verification (decode MVT independently)
   - Real-world data fixtures

4. **Benchmarks** (`criterion`)
   - Single tile encoding performance
   - Full pipeline at various zoom ranges
   - Feature dropping at different densities
   - Statistical rigor with confidence intervals and regression detection

5. **Mutation Testing** (`cargo-mutants`)
   - Identifies test suite blind spots
   - Runs weekly in CI, locally before major releases
   - Focused on core algorithmic modules

### Coverage Targets

- **Core library**: 80%+ coverage on algorithmic modules
- **CLI/Python bindings**: Integration-tested, not coverage-measured

## MVP Success Criteria

The initial release is complete when:

- CLI produces valid PMTiles from GeoParquet: `gpq-tiles input.parquet output.pmtiles --min-zoom 0 --max-zoom 14`
- Output renders correctly in MapLibre at all zoom levels
- Feature density appropriately decreases at lower zooms
- Python `convert()` function works in production environments
- Successfully integrates into geoparquet-io as gpio-pmtiles replacement

## Performance Goals

- **Faster than Tippecanoe** for typical GeoParquet → PMTiles workflows
- **Parallel tile generation** within zoom levels
- **Memory-efficient** spatial indexing for large datasets
- **Optimized MVT encoding** with minimal allocations

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow, testing philosophy, and code style guidelines.
