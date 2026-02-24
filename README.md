# gpq-tiles

[![CI](https://github.com/geoparquet-io/gpq-tiles/actions/workflows/ci.yml/badge.svg)](https://github.com/geoparquet-io/gpq-tiles/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/geoparquet-io/gpq-tiles/branch/main/graph/badge.svg)](https://codecov.io/gh/geoparquet-io/gpq-tiles)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Fast GeoParquet → PMTiles converter in Rust. 1.4x faster than tippecanoe on typical workflows.

## Quick Start

```bash
# Install
cargo install gpq-tiles

# Basic conversion
gpq-tiles input.parquet output.pmtiles --min-zoom 0 --max-zoom 14

# With property filtering (matches tippecanoe -y/-x/-X flags)
gpq-tiles input.parquet output.pmtiles --include name --include population
gpq-tiles input.parquet output.pmtiles --exclude internal_id
gpq-tiles input.parquet output.pmtiles --exclude-all  # geometry only

# With compression options
gpq-tiles input.parquet output.pmtiles --compression zstd  # fastest decompression
```

**Python:**
```python
from gpq_tiles import convert
convert("input.parquet", "output.pmtiles", min_zoom=0, max_zoom=14, compression="zstd")
```

**Suppress warnings:**
```bash
gpq-tiles input.parquet output.pmtiles --quiet  # No optimization warnings
```

**Rust:**
```rust
use gpq_tiles_core::pipeline::{generate_tiles, TilerConfig};
let config = TilerConfig::new(0, 14);
let tiles = generate_tiles(Path::new("input.parquet"), &config)?;
```

## Features

- **Fast** — Parallel tile generation with Rayon, space-filling curve sorting
- **Correct** — MVT spec compliance, golden tests against tippecanoe v2.49.0
- **Smart** — Density-based feature dropping, tiny polygon removal, point thinning
- **Flexible** — Property filtering (`--include`/`--exclude`), compression options (gzip/brotli/zstd)
- **Efficient** — Tile deduplication via XXH3 hashing and run_length encoding
- **Streaming** — Process files larger than memory via row-group streaming

## Best Practices

For optimal performance with large files, optimize your GeoParquet input:

```bash
# Hilbert-sort and add row group bboxes with geoparquet-io
gpq optimize input.parquet -o optimized.parquet --hilbert
```

gpq-tiles will warn if input files aren't optimized. See [geoparquet-io](https://github.com/geoparquet-io/geoparquet-io) for file optimization tools.

## Project Structure

```
crates/
├── core/     # All tiling logic (gpq-tiles-core)
├── cli/      # Thin wrapper (gpq-tiles)
└── python/   # pyo3 bindings
```

## Development

```bash
git clone git@github.com:geoparquet-io/gpq-tiles.git && cd gpq-tiles
git config core.hooksPath .githooks  # Enable pre-commit hooks
cargo build && cargo test
```

**Prerequisites:** Rust 1.75+, protoc (`brew install protobuf` / `apt install protobuf-compiler`)

**Key commands:**
```bash
cargo test                    # Run all tests (333 total)
cargo bench                   # Run benchmarks
cargo fmt && cargo clippy     # Format and lint
cargo tarpaulin --out html    # Coverage report
```

## Documentation

| Document | Purpose |
|----------|---------|
| [ROADMAP.md](ROADMAP.md) | Implementation phases and progress |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Development workflow, Python setup |
| [context/ARCHITECTURE.md](context/ARCHITECTURE.md) | Design decisions, tippecanoe divergences |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [CLAUDE.md](CLAUDE.md) | AI assistant instructions |

## Contributing

1. Fork → branch → make changes with tests → `cargo test && cargo fmt && cargo clippy`
2. Push → open PR with clear description
3. All logic goes in `crates/core`, not CLI/Python

**Commit format:** `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `chore:`

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Acknowledgments

Built on [tippecanoe](https://github.com/felt/tippecanoe) algorithms, [geoarrow-rs](https://github.com/geoarrow/geoarrow-rs), and the Rust geospatial ecosystem. Part of [geoparquet-io](https://github.com/geoparquet-io/geoparquet-io).
