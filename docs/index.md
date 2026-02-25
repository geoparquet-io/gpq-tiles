# gpq-tiles

Fast GeoParquet → PMTiles converter in Rust.

## Why gpq-tiles?

- **Fast** — Parallel tile generation with Rayon, space-filling curve sorting
- **Correct** — MVT spec compliance, golden tests against tippecanoe v2.49.0
- **Smart** — Density-based feature dropping, tiny polygon removal, point thinning
- **Flexible** — Property filtering, compression options (gzip/brotli/zstd)
- **Efficient** — Tile deduplication via XXH3 hashing

## Quick Example

```bash
# CLI
gpq-tiles input.parquet output.pmtiles --min-zoom 0 --max-zoom 14

# With property filtering
gpq-tiles input.parquet output.pmtiles --include name --include population
```

```python
# Python
from gpq_tiles import convert
convert("input.parquet", "output.pmtiles", min_zoom=0, max_zoom=14)
```

```rust
// Rust
use gpq_tiles_core::pipeline::{generate_tiles, TilerConfig};
let config = TilerConfig::new(0, 14);
let tiles = generate_tiles(Path::new("input.parquet"), &config)?;
```

## Next Steps

- [Getting Started](getting-started.md) — Installation and basic usage
- [Advanced Usage](advanced-usage.md) — Performance tuning, streaming, CI/CD
- [API Reference](api-reference.md) — CLI flags, Rust API, Python API
- [Architecture](architecture.md) — Design decisions and internals

## License

Apache 2.0 — [View on GitHub](https://github.com/geoparquet-io/gpq-tiles)
