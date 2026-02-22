# gpq-tiles

[![CI](https://github.com/geoparquet-io/gpq-tiles/actions/workflows/ci.yml/badge.svg)](https://github.com/geoparquet-io/gpq-tiles/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/geoparquet-io/gpq-tiles/branch/main/graph/badge.svg)](https://codecov.io/gh/geoparquet-io/gpq-tiles)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Fast GeoParquet → PMTiles converter in Rust. 1.4x faster than tippecanoe on typical workflows.

## Quick Start

```bash
# Install
cargo install gpq-tiles

# Convert
gpq-tiles input.parquet output.pmtiles --min-zoom 0 --max-zoom 14
```

**Python:**
```python
from gpq_tiles import convert
convert("input.parquet", "output.pmtiles", min_zoom=0, max_zoom=14)
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
cargo test                    # Run all tests (262 Rust + 10 Python)
cargo bench                   # Run benchmarks
cargo fmt && cargo clippy     # Format and lint
cargo tarpaulin --out html    # Coverage report
```

## Documentation

| Document | Purpose |
|----------|---------|
| [ROADMAP.md](ROADMAP.md) | Implementation phases and progress |
| [DEVELOPMENT.md](DEVELOPMENT.md) | Development workflow, Python setup |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Design decisions, tippecanoe divergences |
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
