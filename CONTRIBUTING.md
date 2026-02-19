# Contributing to gpq-tiles

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Rust 1.75+ (`rustup install stable`)
- protoc (Protocol Buffers compiler)
  - macOS: `brew install protobuf`
  - Ubuntu: `apt-get install protobuf-compiler`
- Git

### Setting Up Your Development Environment

```bash
# Clone the repository
git clone git@github.com:geoparquet-io/gpq-tiles.git
cd gpq-tiles

# Build the project
cargo build

# Run tests
cargo test

# Install development tools
cargo install cargo-watch cargo-tarpaulin cargo-mutants
```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

Follow these principles:
- **Library-first design**: Core logic goes in `crates/core`, not CLI/Python bindings
- **Test-driven development**: Write failing tests before implementation
- **Type safety**: Use `thiserror` for library errors, `anyhow` for CLI
- **Documentation**: Add doc comments for public APIs

### 3. Write Tests

Every feature needs tests at multiple levels:

```rust
// Unit tests (inline with code)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_functionality() {
        // Test implementation
    }
}

// Property-based tests
#[cfg(test)]
mod proptests {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn mvt_round_trip(coords in prop::collection::vec(0..100i32, 1..10)) {
            // Property test implementation
        }
    }
}
```

### 4. Run the Test Suite

```bash
# Fast feedback loop
cargo watch -x "test --lib"

# Full test suite
cargo test

# With coverage
cargo tarpaulin --out html

# Benchmarks
cargo bench
```

### 5. Format and Lint

```bash
# Format code
cargo fmt

# Lint
cargo clippy -- -D warnings
```

### 6. Commit Your Changes

Use descriptive commit messages:

```bash
git add .
git commit -m "feat: add feature dropping based on density

- Implement spacing calculation per zoom level
- Add threshold tuning for feature visibility
- Include tests for edge cases (sparse/dense datasets)"
```

Commit message format:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `test:` - Test additions or modifications
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### 7. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a PR on GitHub with:
- Clear description of the changes
- Reference to any related issues
- Screenshots/examples if applicable

## Testing Philosophy

### Test Layers

1. **Unit Tests** (`cargo test --lib`)
   - Fast, focused tests for algorithmic correctness
   - Test pure functions in isolation
   - Run on every save with `cargo watch`

2. **Property-Based Tests** (`proptest`)
   - Catch edge cases you didn't think of
   - Use for geometry operations, coordinate transforms
   - Shrinks failing cases to minimal reproductions

3. **Integration Tests** (`tests/`)
   - Full pipeline: GeoParquet → PMTiles
   - Golden file comparisons
   - Spec compliance verification

4. **Benchmarks** (`benches/`)
   - `criterion` for statistical rigor
   - Run before/after performance changes
   - Compare against baseline

5. **Mutation Tests** (`cargo mutants`)
   - Find test suite blind spots
   - Run before releases or weekly in CI
   - Focus on algorithmic modules

### Writing Good Tests

**DO:**
- Test behavior, not implementation
- Use descriptive test names: `test_feature_dropping_removes_dense_clusters`
- Add comments explaining WHY you're testing something
- Test edge cases: empty inputs, extreme values, boundary conditions
- Use property-based tests for geometry operations

**DON'T:**
- Test private functions directly (test through public API)
- Duplicate tests (one test per behavior)
- Write tests that depend on execution order
- Mock things that are cheap to use directly

### Example Test Structure

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mvt_encoding_handles_empty_geometry() {
        // Arrange
        let geometry = Geometry::LineString(vec![]);

        // Act
        let result = encode_mvt(&geometry);

        // Assert
        assert!(result.is_ok());
        assert_eq!(result.unwrap().commands.len(), 0);
    }

    #[test]
    fn test_tile_coordinates_never_overlap() {
        // Property: for any two different tiles at the same zoom,
        // their bounding boxes should not overlap
        let zoom = 8;
        let tile1 = TileCoord { x: 10, y: 20, z: zoom };
        let tile2 = TileCoord { x: 11, y: 20, z: zoom };

        let bbox1 = tile1.bbox();
        let bbox2 = tile2.bbox();

        assert!(!bbox1.intersects(&bbox2));
    }
}
```

## Code Style

### Rust Conventions

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for formatting (automatic with `cargo fmt`)
- Use `clippy` for linting (fix all warnings)
- Maximum line length: 100 characters
- Prefer explicit types over `let x = ...` when clarity matters

### Documentation

Add doc comments for all public items:

```rust
/// Converts a GeoParquet file to PMTiles.
///
/// # Arguments
///
/// * `input` - Path to the input GeoParquet file
/// * `output` - Path to the output PMTiles file
/// * `config` - Configuration for the conversion process
///
/// # Errors
///
/// Returns an error if:
/// - The input file cannot be read or is not valid GeoParquet
/// - The output file cannot be written
/// - Geometry processing fails
///
/// # Examples
///
/// ```
/// use gpq_tiles_core::{convert, Config};
///
/// let config = Config::default();
/// convert("input.parquet", "output.pmtiles", &config)?;
/// ```
pub fn convert(input: &str, output: &str, config: &Config) -> Result<()> {
    // Implementation
}
```

### Error Handling

Use `thiserror` for library errors:

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TilingError {
    #[error("Failed to read GeoParquet: {0}")]
    GeoParquetRead(#[from] geoarrow::error::GeoArrowError),

    #[error("Invalid geometry at feature {feature_id}: {reason}")]
    InvalidGeometry {
        feature_id: usize,
        reason: String,
    },
}
```

Use `anyhow` for CLI:

```rust
use anyhow::{Context, Result};

fn main() -> Result<()> {
    let input = read_file(path)
        .context("Failed to read input file")?;
    Ok(())
}
```

## Project Structure

Understanding the layout:

```
gpq-tiles/
├── crates/
│   ├── core/              # Core library (gpq-tiles-core)
│   │   ├── src/
│   │   │   ├── lib.rs     # Public API
│   │   │   ├── mvt.rs     # MVT encoding logic
│   │   │   ├── tiling.rs  # Tile generation
│   │   │   └── ...
│   │   ├── tests/         # Integration tests
│   │   ├── benches/       # Benchmarks
│   │   └── Cargo.toml
│   ├── cli/               # CLI binary
│   │   ├── src/
│   │   │   └── main.rs    # Thin wrapper around core
│   │   └── Cargo.toml
│   └── python/            # Python bindings
│       ├── src/
│       │   └── lib.rs     # pyo3 bindings
│       └── Cargo.toml
├── tests/
│   └── fixtures/          # Test data
├── .github/
│   └── workflows/         # CI configuration
└── Cargo.toml             # Workspace root
```

**Key principle**: All logic lives in `core`. CLI and Python are thin consumers.

## Pull Request Process

1. **Before Submitting**:
   - All tests pass (`cargo test`)
   - Code is formatted (`cargo fmt`)
   - No clippy warnings (`cargo clippy`)
   - Documentation is updated
   - CHANGELOG.md is updated (if applicable)

2. **PR Description Should Include**:
   - What changes were made and why
   - How to test the changes
   - Any breaking changes
   - Screenshots/examples for user-facing changes

3. **Review Process**:
   - Maintainers will review your PR
   - Address feedback by pushing new commits
   - Once approved, maintainers will merge

4. **After Merge**:
   - Delete your feature branch
   - Pull the latest main: `git checkout main && git pull`

## Release Process (Maintainers Only)

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md`
3. Create a git tag: `git tag -a v0.1.0 -m "Release v0.1.0"`
4. Push tag: `git push origin v0.1.0`
5. Publish to crates.io: `cargo publish -p gpq-tiles-core && cargo publish -p gpq-tiles`
6. Build Python wheels and publish to PyPI

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue with reproduction steps
- **Security**: Email nlebovits@pm.me (do not open public issues)

## Code of Conduct

Be respectful and constructive. We're all here to build something useful together.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
