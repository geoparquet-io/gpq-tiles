## v0.2.0 (2026-02-25)

## v0.1.0 (2026-02-24)

### Feat

- set up commitizen for automated versioning and releases
- **quality**: warn about pathologically small row groups
- default to zstd compression, expose parallel options in CLI
- add progress bars for cleaner output
- parallelize geometry processing within row groups (#37)
- parallelize tile processing for large geometries (#33)
- **cli**: add --streaming-mode flag with progress reporting
- **pipeline**: implement ExternalSort streaming mode
- **pipeline**: add StreamingMode::ExternalSort variant
- **core**: add external sort and WKB serialization modules
- **streaming**: add StreamingPmtilesWriter with LowMemory mode
- **streaming**: add memory budget configuration and tracking
- **streaming**: add row-group-based streaming tile generation
- **quality**: add GeoParquet file quality detection for streaming
- add tile deduplication with XXH3 hashing and run_length encoding
- add compression options (gzip, brotli, zstd, none) for PMTiles output
- add property filtering with --include/-y, --exclude/-x, --exclude-all/-X flags
- add 17K feature fixture for parallelization benchmarks
- add tilestats metadata to PMTiles output
- auto-extract field metadata from GeoParquet schema
- add field metadata support to PMTiles writer
- derive layer name from input filename, add --layer-name CLI flag
- complete Phase 5 Python bindings with uv/ruff tooling
- add benchmark suite with generate_tiles_from_geometries API
- **pipeline**: add Rayon parallel tile generation
- **pipeline**: wire spatial indexing into tile generation
- **spatial-index**: add space-filling curve sorting for efficient tile generation
- complete Phase 3 with density-based dropping
- integrate feature dropping into pipeline (Phase 3)
- add point thinning (1/2.5 drop rate per zoom)
- add line dropping (coordinate quantization algorithm)
- implement tiny polygon dropping with diffuse probability
- implement PMTiles v3 writer (Tasks 7-9)
- implement tiler pipeline wiring clip → simplify → MVT
- implement MVT encoding for vector tiles
- add golden comparison tests against tippecanoe output
- implement geometry clipping with correct BooleanOps
- implement zoom-based simplification
- implement Arrow-native geometry batch processing (TDD green)

### Fix

- **release**: add version to core dep, add READMEs, fix benchmark filter
- **release**: install protoc inside manylinux container
- **python**: copy README into crate for sdist builds
- **ci**: use tag triggers for release, skip slow benchmarks
- consolidate release workflows and fix version bump detection
- guard against degenerate linestrings in simplify and fix flaky test
- PMTiles now compatible with pmtiles.io and standard viewers
- **golden**: update stale Z8 test to use full pipeline
- resolve three medium/low priority issues
- simplify geometry in tile-local pixel coordinates
- handle antimeridian crossing in tiles_for_bbox
- **clip**: preserve all polygon parts when clipping produces MultiPolygon
- use real bbox calculation instead of world bounds
- resolve CI timeouts and coverage linker errors
- upgrade pyo3 0.24 → 0.28 for Python 3.14 support
- resolve CI failures for benchmark, check, test, and security audit

### Perf

- **streaming**: add memory benchmarks for streaming pipeline
