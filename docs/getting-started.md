# Getting Started

## Installation

### CLI (Cargo)

```bash
cargo install gpq-tiles
```

**Prerequisites:** Rust 1.75+, protoc

```bash
# macOS
brew install protobuf

# Ubuntu/Debian
apt install protobuf-compiler
```

### Python

```bash
pip install gpq-tiles
```

### From Source

```bash
git clone https://github.com/geoparquet-io/gpq-tiles.git
cd gpq-tiles
cargo build --release
```

## Basic Usage

### CLI

```bash
# Basic conversion
gpq-tiles input.parquet output.pmtiles --min-zoom 0 --max-zoom 14

# Property filtering (matches tippecanoe -y/-x/-X)
gpq-tiles input.parquet output.pmtiles --include name --include population
gpq-tiles input.parquet output.pmtiles --exclude internal_id
gpq-tiles input.parquet output.pmtiles --exclude-all  # geometry only

# Compression options
gpq-tiles input.parquet output.pmtiles --compression zstd  # fastest decompression
gpq-tiles input.parquet output.pmtiles --compression brotli  # best ratio

# Suppress optimization warnings
gpq-tiles input.parquet output.pmtiles --quiet
```

### Python

```python
from gpq_tiles import convert

convert(
    input="buildings.parquet",
    output="buildings.pmtiles",
    min_zoom=0,
    max_zoom=14,
    compression="zstd",
)
```

### Rust

```rust
use gpq_tiles_core::pipeline::{generate_tiles, TilerConfig};
use std::path::Path;

let config = TilerConfig::new(0, 14)
    .with_density_drop(true);

let tiles = generate_tiles(Path::new("input.parquet"), &config)?;
```

## Optimizing Input Files

For best performance, optimize your GeoParquet files first:

```bash
# Hilbert-sort and add row group bboxes
gpq optimize input.parquet -o optimized.parquet --hilbert
```

gpq-tiles warns if input files lack optimization. See [geoparquet-io](https://github.com/geoparquet-io/geoparquet-io) for details.

## Large Files

For files larger than available memory:

```bash
gpq-tiles large.parquet output.pmtiles --streaming-mode low-memory
```

This mode re-reads the file per zoom level but keeps memory bounded (~750MB for a 3GB file).
