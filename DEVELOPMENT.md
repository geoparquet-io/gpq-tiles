# Development Guide

Quick reference for working on gpq-tiles.

## Initial Setup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install protoc (choose your platform)
# macOS
brew install protobuf

# Ubuntu/Debian
sudo apt-get install protobuf-compiler

# Verify installation
protoc --version  # Should be 3.x or higher
```

## Development Tools

```bash
# Install recommended tools
cargo install cargo-watch    # Auto-run tests on file changes
cargo install cargo-tarpaulin # Coverage reporting
cargo install cargo-mutants   # Mutation testing

# Optional but useful
cargo install cargo-edit      # Add/remove dependencies with `cargo add`
cargo install cargo-outdated  # Check for outdated dependencies
```

## Day-to-Day Workflow

### Fast Feedback Loop (TDD)

```bash
# Auto-run unit tests on save
cargo watch -x "test --lib"

# Auto-run specific test
cargo watch -x "test --lib mvt_encoding"
```

### Full Test Suite

```bash
# All tests
cargo test

# Just unit tests (fast)
cargo test --lib

# Just integration tests
cargo test --test '*'

# Specific crate
cargo test -p gpq-tiles-core

# With output (show println!)
cargo test -- --nocapture
```

### Coverage

```bash
# Generate HTML coverage report
cargo tarpaulin --out html --all-features

# Open report
open tarpaulin-report.html  # macOS
xdg-open tarpaulin-report.html  # Linux
```

### Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench mvt_encoding

# Compare with baseline
cargo bench -- --save-baseline before
# ... make changes ...
cargo bench -- --baseline before
```

### Large File Benchmarks (ADM4)

**CRITICAL:** For accurate benchmarks, you MUST use `--streaming-mode external-sort`.

The default `fast` mode processes row groups sequentially. Only `external-sort` mode
enables full parallelization (both `parallel` and `parallel_geoms`), which is 4x faster.

```bash
# CORRECT: Full parallelization (~3 min on 16-core machine)
cargo run --release -- input.parquet output.pmtiles \
    --streaming-mode external-sort \
    --min-zoom 0 --max-zoom 8

# WRONG: Sequential processing (~12 min) - DO NOT USE FOR BENCHMARKS
cargo run --release -- input.parquet output.pmtiles \
    --min-zoom 0 --max-zoom 8
```

The ADM4 test file (3.3GB, ~364k features) can be downloaded from:
https://data.fieldmaps.io/edge-matched/humanitarian/intl/adm4_polygons.parquet

Place it in `tests/fixtures/large/` (this directory is gitignored).

**Expected benchmark output with external-sort:**
```
⠋ Reading GeoParquet [████████████████████████████████████████] 364/364 row groups | ✓ 363,783 records
⠋ Sorting by tile ID... ✓ Sorted
⠋ Encoding tiles [████████████████████████████████████████] 530033/530033 (100%)
✓ Converted adm4_polygons.parquet → output.pmtiles
       530,033 tiles in 3 minutes (2889 tiles/sec)
  5.17 GiB peak memory
```

### Code Quality

```bash
# Format code
cargo fmt

# Check formatting without changing files
cargo fmt --check

# Lint
cargo clippy

# Lint in CI mode (deny warnings)
cargo clippy -- -D warnings

# Fix auto-fixable clippy warnings
cargo clippy --fix
```

### Documentation

```bash
# Build and open docs
cargo doc --open

# Build docs for all crates
cargo doc --workspace --no-deps

# Check for broken links
cargo doc --workspace --no-deps 2>&1 | grep warning
```

### Building

```bash
# Debug build (fast, unoptimized)
cargo build

# Release build (slow, optimized)
cargo build --release

# Build specific crate
cargo build -p gpq-tiles-core

# Check compilation without building
cargo check
```

## Python Development

Uses **uv** for fast dependency management and **ruff** for linting.

### Quick Start

```bash
cd crates/python

# Create venv and install dev dependencies (uv auto-detects pyproject.toml)
uv venv
uv pip install maturin
uv sync --group dev

# Build Rust extension and install in development mode
uv run maturin develop

# Verify installation
uv run python -c "from gpq_tiles import convert; print(convert.__doc__)"
```

### Running Python Tests

```bash
cd crates/python

# Run tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ -v --cov=gpq_tiles
```

### Linting

```bash
cd crates/python

# Check
uv run ruff check tests/

# Fix auto-fixable issues
uv run ruff check --fix tests/

# Format
uv run ruff format tests/
```

### Building Wheels

```bash
cd crates/python

# Build wheel for current platform
uv run maturin build --release

# Wheel will be in target/wheels/
ls ../../target/wheels/
```

### Usage Example

```python
from gpq_tiles import convert

# Basic conversion
convert(
    input="buildings.parquet",
    output="buildings.pmtiles",
    min_zoom=0,
    max_zoom=14,
)

# With feature dropping options
convert(
    input="buildings.parquet",
    output="buildings.pmtiles",
    min_zoom=0,
    max_zoom=14,
    drop_density="high",  # "low", "medium", or "high"
)
```

## Debugging

### Rust

```bash
# Run with debug output
RUST_LOG=debug cargo run -- input.parquet output.pmtiles

# Run specific test with backtrace
RUST_BACKTRACE=1 cargo test test_name

# Run under debugger (requires lldb or gdb)
rust-lldb target/debug/gpq-tiles
```

### Common Issues

**Problem**: `protoc` not found during build
**Solution**: Install protobuf compiler (see Initial Setup)

**Problem**: Linker errors on macOS
**Solution**: `xcode-select --install`

**Problem**: Tests fail with file not found
**Solution**: Tests run from workspace root, use relative paths like `tests/fixtures/...`

## Performance Profiling

### Using cargo-flamegraph

```bash
# Install
cargo install flamegraph

# Profile a benchmark
cargo flamegraph --bench tiling

# Profile the CLI
cargo flamegraph -- target/release/gpq-tiles input.parquet output.pmtiles
```

### Using criterion

Criterion benchmarks automatically generate:
- HTML reports in `target/criterion/`
- Statistical analysis of performance
- Comparison with previous runs

```bash
cargo bench
open target/criterion/report/index.html
```

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes, run tests
cargo test

# Format and lint
cargo fmt && cargo clippy

# Commit
git add .
git commit -m "feat: your feature description"

# Push
git push origin feature/your-feature
```

## CI Checks (What Runs on GitHub)

Before pushing, ensure these pass locally:

```bash
# Formatting
cargo fmt --check

# Linting
cargo clippy -- -D warnings

# Tests
cargo test

# (Optional) Coverage
cargo tarpaulin --out xml
```

## Useful Cargo Commands

```bash
# Update dependencies
cargo update

# Check for outdated dependencies
cargo outdated

# Add a dependency
cargo add rayon

# Remove a dependency
cargo rm rayon

# Show dependency tree
cargo tree

# Clean build artifacts
cargo clean

# Show package information
cargo metadata
```

## Troubleshooting

### Build is slow

```bash
# Use mold linker (Linux)
cargo install -f cargo-binutils
rustup component add llvm-tools-preview

# Or lld (cross-platform)
# Add to .cargo/config.toml:
# [target.x86_64-unknown-linux-gnu]
# linker = "clang"
# rustflags = ["-C", "link-arg=-fuse-ld=lld"]
```

### Tests are slow

```bash
# Run only unit tests (skip integration)
cargo test --lib

# Run tests in parallel (default) or sequentially
cargo test -- --test-threads=1
```

### Disk space issues

```bash
# Clean old build artifacts
cargo clean

# Clean cargo cache
rm -rf ~/.cargo/registry
rm -rf ~/.cargo/git
```

## Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [Rust by Example](https://doc.rust-lang.org/rust-by-example/)
- [Criterion.rs Docs](https://bheisler.github.io/criterion.rs/book/)
- [pyo3 Guide](https://pyo3.rs/)
- [MVT Spec](https://github.com/mapbox/vector-tile-spec)
- [PMTiles Spec](https://github.com/protomaps/PMTiles)
