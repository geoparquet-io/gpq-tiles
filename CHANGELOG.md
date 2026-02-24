# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-24

### Added

**Core Functionality**
- GeoParquet to PMTiles conversion with MVT encoding
- Property filtering with `--include`, `--exclude`, `--exclude-all` flags (tippecanoe `-y/-x/-X` compatibility)
- Compression options: gzip, brotli, zstd (default), none
- Streaming modes: fast (row-group based) and low-memory (external sort)
- Tile deduplication via XXH3 hashing and run-length encoding
- Progress bars with `--verbose` flag showing phase progress
- Quality detection and warnings for unoptimized GeoParquet files

**Parallelization**
- Parallel tile processing for large geometries spanning many tiles
- Parallel geometry processing within row groups
- CLI flags `--no-parallel` and `--no-parallel-geoms` for debugging

**Metadata & Schema**
- Auto-extraction of field metadata from GeoParquet schema
- Tilestats generation in PMTiles metadata
- Vector layers metadata with field types
- Layer name derivation from input filename or `--layer-name` flag

**APIs**
- **CLI**: Full-featured command-line interface (`gpq-tiles`)
- **Rust**: Library API with `Converter` (high-level) and `generate_tiles` (low-level)
- **Python**: Basic bindings via pyo3 (`gpq_tiles.convert()`)

**Testing & Benchmarks**
- 329 tests (Rust unit, integration, golden tests)
- Streaming benchmarks measuring memory usage
- External sort integration tests
- Golden tests against tippecanoe v2.49.0 output

**Documentation**
- Comprehensive API reference (CLI, Python, Rust)
- Advanced usage guide (performance tuning, troubleshooting, CI/CD integration)
- Getting started guide with examples
- Architecture documentation with tippecanoe alignment notes
- Streaming design specification

**CI/CD**
- Release workflow for crates.io publishing
- Python wheel builds (Linux, macOS, Windows)
- Documentation deployment to GitHub Pages
- Codecov integration for coverage reporting
- Dependabot for dependency updates
- CodeRabbit for PR reviews

### Fixed
- Degenerate linestring handling in simplification (<2 points)
- Flaky streaming memory tracking test

### Changed
- Default compression changed from gzip to zstd (5s faster encoding on 3.3GB test file)
- Documentation organization: `docs/` for mkdocs site, `context/` for architecture
- All documentation verified against actual benchmarks (removed speculative claims)

### Performance
- Zstd compression: 2:59 encoding time, 254MB output (3.3GB input, zoom 0-8)
- Gzip compression: 3:04 encoding time, 175MB output (same test)
- Row-group streaming: memory bounded by largest row group (~100-200MB typical)

### Notes
- **CLI and Rust API**: Production-ready with full feature set
- **Python API**: Basic conversion only; property filters, streaming modes, and progress callbacks tracked in [#45](https://github.com/geoparquet-io/gpq-tiles/issues/45)
- Tested against tippecanoe v2.49.0 for MVT output compliance

[0.1.0]: https://github.com/geoparquet-io/gpq-tiles/releases/tag/v0.1.0
