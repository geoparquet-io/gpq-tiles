# gpq-tiles Roadmap

Production-grade GeoParquet → PMTiles converter in Rust.

**Current:** 262 tests (252 unit + 10 doc). 1.4x faster than tippecanoe.

## Phase Summary

| Phase | Status | Tests | Description |
|-------|--------|-------|-------------|
| 1. Skeleton | ✅ | - | Read GeoParquet → write empty PMTiles |
| 2. Naive Tiling | ✅ | 122 | Full pipeline: clip → simplify → MVT → PMTiles |
| 3. Feature Dropping | ✅ | 226 | Tiny polygon, line, point, density dropping |
| 4. Parallelism | ✅ | 262 | Space-filling curves, Rayon, benchmarks |
| 5. Python | 🔲 | - | pyo3 bindings |

## Phase 4: Parallelism ✅ COMPLETE

### Spatial Indexing

Space-filling curve sorting (not R-tree) following tippecanoe's approach:

```rust
let config = TilerConfig::new(0, 14)
    .with_hilbert(true)      // Hilbert curve (default) or Z-order
    .with_parallel(true);    // Rayon parallelization (default)
```

**Why space-filling curves?**
- Sort once, scan sequentially — no random spatial queries
- Cache-friendly — features for same tile are adjacent in memory
- Parallel-friendly — sorted stream partitions cleanly

### Rayon Parallelization

- Tiles within each zoom level processed in parallel
- Zoom levels sequential to preserve feature dropping semantics
- Results sorted by `(z, x, y)` for deterministic output

### Benchmarks

```bash
cargo bench --package gpq-tiles-core
```

| Benchmark | Description |
|-----------|-------------|
| `single_tile` | Z8/Z10 tile generation |
| `full_pipeline` | Z0-8, Z0-10 zoom ranges |
| `parallel_vs_sequential` | Rayon vs single-threaded |
| `density_dropping` | With/without density drop |
| `hilbert_vs_zorder` | Curve comparison |

**Performance** (1000 features, Z0-10): gpq-tiles ~134ms, tippecanoe ~194ms

## Phase 3: Feature Dropping ✅ COMPLETE

| Algorithm | Description |
|-----------|-------------|
| Tiny polygon | Diffuse probability for < 4 sq pixels |
| Line dropping | Coordinate quantization collapse |
| Point thinning | 1/2.5 drop rate per zoom above base |
| Density-based | Grid-cell limiting (configurable) |

```rust
let config = TilerConfig::new(0, 14)
    .with_density_drop(true)
    .with_density_cell_size(32)
    .with_density_max_per_cell(1);
```

**Golden comparison** (vs tippecanoe v2.49.0):
- Z10: 0.81x ratio (392 vs 484 features)
- Z8: 0.78x ratio (76 vs 97 features)

## Phase 2: Naive Tiling ✅ COMPLETE

Full pipeline implemented:
1. GeoParquet reading (`geoparquet` crate)
2. Feature extraction from GeoArrow arrays
3. Tile coordinate math (Web Mercator)
4. Geometry clipping with buffer
5. RDP simplification per zoom
6. Geometry validation (winding order, degenerates)
7. MVT encoding (zigzag delta coordinates)
8. PMTiles v3 writing (Hilbert ordering)

## Phase 5: Python Integration 🔲 TODO

```python
from gpq_tiles import convert

convert(
    input="buildings.parquet",
    output="buildings.pmtiles",
    min_zoom=0,
    max_zoom=14,
)
```

## Known Issues

| Severity | Issue | Status |
|----------|-------|--------|
| Medium | Memory for large files | `extract_geometries` loads all into Vec |

See `docs/ARCHITECTURE.md` for design decisions and tippecanoe divergences.
