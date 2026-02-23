# gpq-tiles Architecture

Design decisions and tippecanoe divergences.

## Design Principles

1. **Arrow-First**: Process geometries within Arrow batch scope for zero-copy benefits
2. **Semantic Comparison**: Golden tests use IoU and feature counts, not byte-exact comparison
3. **Reference Implementations**: All algorithms match tippecanoe behavior; document divergences
4. **PMTiles Writer**: The `pmtiles` crate is read-only; we implement our own v3 writer

## Known Divergences from Tippecanoe

| Area | Our Approach | Tippecanoe | Notes |
|------|--------------|------------|-------|
| Metadata | Basic `vector_layers` | Full layer/field/tilestats | Field metadata planned |
| Simplification | Custom tolerance formula | `tile.cpp` | Tuned to match output quality |
| Density dropping | Grid-cell limiting | Hilbert-gap selection | Simpler, similar results |

## Density-Based Dropping

**DIVERGENCE**: Tippecanoe uses Hilbert curve ordering with gap-based selection (squared distance between consecutive features). We use grid-cell limiting — simpler and faster, but doesn't preserve spatial distribution as well.

```rust
// Enable density dropping
let config = TilerConfig::new(0, 14)
    .with_density_drop(true)
    .with_density_cell_size(32);   // Pixels per cell
```

**Cell size reference** (Z8, 4096 extent):

| Cell Size | Grid | Typical Features |
|-----------|------|------------------|
| 16px | 256×256 | 34 |
| 32px | 128×128 | 23 |
| 64px | 64×64 | 13 |
| 128px | 32×32 | 9 |

## Spatial Indexing

**We use space-filling curve sorting, not R-tree.**

| Consideration | R-tree | Space-filling Sort |
|---------------|--------|-------------------|
| Memory | +30-50% overhead | None |
| Access pattern | Random | Sequential |
| Streaming | Difficult | Natural |
| Complexity | Tree balancing | Standard sort |

**Hilbert vs Z-order:**
- Z-order (Morton): Simple bit interleaving, has "jumps" at quadrant boundaries
- Hilbert: Better locality, spatially adjacent points always close in 1D index

Default: Hilbert (matches tippecanoe's `-ah` flag).

## Golden Comparison Results

Validated against tippecanoe v2.49.0 using `open-buildings.parquet` (~1000 buildings):

| Zoom | Tippecanoe | gpq-tiles | Ratio |
|------|------------|-----------|-------|
| Z10 | 484 | 392 | 0.81x |
| Z8 | 97 | 76 | 0.78x |
| Z5 | 1 | ~200 | Configurable |

**Analysis:**
- We drop more aggressively at high zoom due to diffuse probability
- Area preservation after clip+simplify: 88%
- All zoom levels produce valid MVT tiles

## Streaming Processing

**Design:** Row-group-based streaming where memory is bounded by the largest row group, not file size.

```
peak_memory ≈ row_group_decoded + active_tile_buffers
           ≈ 100MB            + 50MB (typical)
           ≈ 150MB per row group being processed
```

### Row Group Iterator

```rust
// Process one row group at a time
for row_group in row_group_iterator(path)? {
    let features = decode_row_group(&row_group)?;
    for feature in features {
        // Bucket by tile across all zoom levels
        for tile_coord in tiles_intersecting(&feature.geom, config) {
            let clipped = clip_to_tile(&feature.geom, &tile_coord)?;
            tile_buckets.entry(tile_coord).or_default().push(clipped);
        }
    }
    // Encode and yield tiles for this row group
}
```

### File Quality Detection

Before streaming, `assess_quality()` checks input files and warns about suboptimal formats:

| Check | Cost | Action |
|-------|------|--------|
| Missing `geo` metadata | O(1) | Warn: "File missing GeoParquet metadata" |
| No row group bboxes | O(1) | Warn: "Cannot skip spatially" |
| Few row groups for size | O(1) | Warn: "Large file limits streaming efficiency" |
| Not Hilbert-sorted | O(1000) | Warn: "File not spatially sorted" (sampled) |

Warnings recommend optimizing with [geoparquet-io](https://github.com/geoparquet-io/geoparquet-io):

```
gpq optimize input.parquet -o optimized.parquet --hilbert
```

Use `--quiet` to suppress warnings. See `quality.rs` for implementation.

## Module Structure

```
crates/core/src/
├── lib.rs              # Public API
├── tile.rs             # TileCoord, TileBounds
├── clip.rs             # Geometry clipping
├── simplify.rs         # RDP simplification
├── validate.rs         # Geometry validation
├── mvt.rs              # MVT encoding
├── pmtiles_writer.rs   # PMTiles v3 writer
├── feature_drop.rs     # Dropping algorithms
├── spatial_index.rs    # Hilbert/Z-order curves
├── pipeline.rs         # Tile generation (streaming + non-streaming)
├── batch_processor.rs  # GeoArrow iteration (row group support)
├── quality.rs          # File quality assessment
└── golden.rs           # Golden tests
```
