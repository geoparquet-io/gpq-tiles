# gpq2tiles

[![CI](https://github.com/geoparquet-io/gpq-tiles/actions/workflows/ci.yml/badge.svg)](https://github.com/geoparquet-io/gpq-tiles/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/geoparquet-io/gpq-tiles/branch/main/graph/badge.svg)](https://codecov.io/gh/geoparquet-io/gpq-tiles)
[![Crates.io](https://img.shields.io/crates/v/gpq2tiles.svg)](https://crates.io/crates/gpq2tiles)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A Rust library and CLI tool for converting GeoParquet files into PMTiles vector tile archives. Designed as a faster, more correct replacement for `gpio-pmtiles`, with native Python bindings for integration into the [geoparquet-io](https://github.com/geoparquet-io/geoparquet-io) ecosystem.

## Features

- **Fast**: Parallelized tile generation using Rayon
- **Correct**: MVT encoding follows the Mapbox Vector Tile specification exactly
- **Smart**: Density-based feature dropping prevents cluttered maps at low zoom levels
- **Flexible**: Use as a Rust library, CLI tool, or Python package

## Installation

### Rust CLI

```bash
cargo install gpq2tiles
```

### Python

```bash
pip install gpq2tiles
```

### From Source

```bash
git clone git@github.com:geoparquet-io/gpq-tiles.git
cd gpq-tiles
cargo build --release
```

The compiled binary will be in `target/release/gpq2tiles`.

## Usage

### CLI

```bash
# Basic usage
gpq2tiles input.parquet output.pmtiles

# With zoom levels
gpq2tiles input.parquet output.pmtiles --min-zoom 0 --max-zoom 14

# With feature dropping control
gpq2tiles input.parquet output.pmtiles --min-zoom 0 --max-zoom 14 --drop-density low
```

### Python

```python
from gpq2tiles import convert

convert(
    input="buildings.parquet",
    output="buildings.pmtiles",
    min_zoom=0,
    max_zoom=14,
)
```

### Rust Library

```rust
use gpq2tiles_core::{Converter, Config};

let config = Config {
    min_zoom: 0,
    max_zoom: 14,
    ..Default::default()
};

let converter = Converter::new(config);
converter.convert("input.parquet", "output.pmtiles")?;
```

## How It Works

1. **Read**: Loads GeoParquet using `geoarrow`
2. **Tile**: For each zoom level, clips features to tile boundaries and simplifies geometry
3. **Drop**: Applies density-based feature dropping to prevent overcrowding at low zooms
4. **Encode**: Converts geometries to Mapbox Vector Tile (MVT) format using protobuf
5. **Write**: Packages tiles into a PMTiles archive

The library prioritizes correctness and performance through:
- Exact MVT command encoding (zigzag-encoded delta coordinates)
- Parallel tile generation with Rayon
- Spatial indexing with R-trees for efficient feature lookup
- RDP simplification tuned per zoom level

## Development

### Prerequisites

- Rust 1.75+ (`rustup install stable`)
- protoc (Protocol Buffers compiler)
  - macOS: `brew install protobuf`
  - Ubuntu: `apt-get install protobuf-compiler`
  - Other: https://grpc.io/docs/protoc-installation/

### Building

```bash
# Full workspace
cargo build

# Just the core library
cargo build -p gpq2tiles-core

# Release build with optimizations
cargo build --release
```

### Testing

```bash
# Run all tests
cargo test

# Run tests with coverage
cargo tarpaulin --out html

# Run benchmarks
cargo bench

# Run mutation tests (slow!)
cargo mutants
```

### Useful Commands

```bash
# Auto-run tests on file changes
cargo install cargo-watch
cargo watch -x "test --lib"

# Format code
cargo fmt

# Lint
cargo clippy -- -D warnings

# Check documentation
cargo doc --open
```

## Project Structure

```
gpq-tiles/
├── crates/
│   ├── core/          # Core library (gpq2tiles-core)
│   ├── cli/           # CLI binary (gpq2tiles)
│   └── python/        # Python bindings (pyo3)
├── tests/
│   └── fixtures/      # Test GeoParquet files and expected outputs
├── benches/           # Criterion benchmarks
└── Cargo.toml         # Workspace configuration
```

## Comparison to Tippecanoe

`gpq2tiles` is inspired by [Tippecanoe](https://github.com/felt/tippecanoe) but optimized for the GeoParquet → PMTiles workflow:

| Feature | gpq2tiles | Tippecanoe |
|---------|-----------|------------|
| Input format | GeoParquet | GeoJSON, CSV |
| Output format | PMTiles | PMTiles, MBTiles |
| Language | Rust | C++ |
| Feature dropping | Density-based (MVP) | Multiple strategies |
| Parallelism | Per-tile (Rayon) | Per-zoom |
| Python bindings | Native (pyo3) | CLI wrapper |

## Roadmap

- [x] Phase 1: Skeleton (read GeoParquet → write empty PMTiles)
- [ ] Phase 2: Naive tiling (zoom-by-zoom tile generation)
- [ ] Phase 3: Feature dropping (density-based)
- [ ] Phase 4: Parallelism (Rayon + spatial index)
- [ ] Phase 5: Python bindings (pyo3)
- [ ] Future: Attribute filtering, coalesce strategies, streaming for huge files

See the full [development roadmap](ROADMAP.md) for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`cargo test`)
5. Format and lint (`cargo fmt && cargo clippy`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Testing Philosophy

This project follows a layered testing approach:

1. **Unit tests**: Fast, focused tests for algorithmic correctness (MVT encoding, coordinate transforms)
2. **Property-based tests**: `proptest` for edge cases (geometry round-trips, tile coordinate invariants)
3. **Integration tests**: GeoParquet → PMTiles pipelines with golden file comparison
4. **Benchmarks**: `criterion` for performance regression detection
5. **Mutation tests**: `cargo-mutants` to find test suite gaps (run before releases)

See the [testing documentation](TESTING.md) for details.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by [Tippecanoe](https://github.com/felt/tippecanoe) by Mapbox/Felt
- Built on [geoarrow](https://github.com/geoarrow/geoarrow-rs), [pmtiles](https://github.com/stadiamaps/pmtiles-rs), and the Rust geospatial ecosystem
- Part of the [geoparquet-io](https://github.com/geoparquet-io/geoparquet-io) project
