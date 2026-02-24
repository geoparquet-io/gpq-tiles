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

## GeoParquet File Structure: Critical Performance Factor

**Row group size dramatically affects performance.** Our pipeline has per-row-group overhead (memory tracking, progress reporting, sorter flushes), so files with many small row groups are pathologically slow.

| File | Size | Rows | Row Groups | Rows/Group | Performance |
|------|------|------|------------|------------|-------------|
| ADM4 | 3.2 GB | 363,783 | 364 | ~1,000 | ✅ Good (3 min) |
| ADM2 | 1.9 GB | 43,064 | 4,307 | ~10 | ❌ Very slow |

**Rule of thumb:** Aim for 1,000+ rows per row group. Files with <100 rows/group will have significant overhead.

**Why this happens:**
- Each row group triggers progress callbacks, memory tracking, and sorter operations
- The external sort flushes buffers based on record count, not row groups
- Small row groups = more overhead per feature processed

**Recommendation:** If you control file creation, use larger row groups. If processing files with small row groups, consider consolidating them first with tools like DuckDB or gpio.

## Streaming Processing

### The Challenge

For a geometry that spans multiple tiles across multiple zoom levels, we need to store/process it multiple times. With 363K features at z0-z6, this can mean millions of geometry instances.

**The core problem:** PMTiles requires ALL features for a tile (z,x,y) to be encoded together. We can't write a tile incrementally as we encounter each feature.

### Streaming Modes

We provide two modes with different memory/speed tradeoffs:

| Mode | Memory | Speed | How it Works |
|------|--------|-------|--------------|
| `LowMemory` | ~750MB | 2-3x slower | Re-reads file for each zoom level |
| `Fast` | ~1-2GB | 1x | Single pass, stores clipped geometries |

### StreamingMode::LowMemory ✅ WORKS

Processes one zoom level at a time, re-reading the input file for each:

```rust
for z in min_zoom..=max_zoom {
    // Re-read file, process only zoom z
    process_geometries_by_row_group(input_path, |_, geometries| {
        for geom in geometries {
            for tile in tiles_for_bbox(&geom.bbox(), z) {
                let clipped = clip(&geom, &tile);
                tile_features.entry((x, y)).push(clipped);
            }
        }
    })?;

    // Encode and write all tiles for zoom z
    for ((x, y), features) in tile_features.drain() {
        let tile = encode_tile(features);
        writer.add_tile(z, x, y, &tile)?;
    }
    // Memory freed before next zoom level
}
```

**Why it's slow:** For z0-z6 (7 levels), reads a 3.2GB file 7 times = 22.4GB I/O.

**Why it works:** Only one zoom level's worth of clipped geometries in memory at a time.

### StreamingMode::Fast ⚠️ MAY OOM ON VERY LARGE FILES

Single pass that stores clipped (not original!) geometries per tile:

```rust
process_geometries_by_row_group(input_path, |_, geometries| {
    for geom in geometries {
        for z in min_zoom..=max_zoom {
            for tile in tiles_for_bbox(&geom.bbox(), z) {
                // Store CLIPPED geometry, not original (~90% smaller)
                let clipped = clip(&geom, &tile);
                tile_features.entry((z, x, y)).push(clipped);
            }
        }
    }
})?;
// Encode all tiles at end
```

**Key optimization:** Stores clipped geometries (~90% smaller than originals) instead of cloning original geometries.

**Why it may OOM:** For very large files with features spanning many tiles, clipped geometry accumulation can still exceed memory. A 3.2GB file with 363K features at z0-z6 accumulates ~1-2GB of clipped geometries.

### Planned: External Sort Approach

For true bounded-memory fast processing:

1. **Sort phase:** Write `(tile_id, clipped_geometry)` pairs to temp file
2. **External sort:** Sort temp file by tile_id (can use memory-mapped merge sort)
3. **Encode phase:** Read sorted file, encode each tile's features together

This would give Fast mode's speed with LowMemory mode's memory bounds.

### StreamingPmtilesWriter

The `StreamingPmtilesWriter` solves the memory problem for **output** (tiles):

| Component | PmtilesWriter | StreamingPmtilesWriter |
|-----------|---------------|------------------------|
| Tile data | In-memory BTreeMap | Temp file (disk) |
| Directory | Calculated at end | Built incrementally |
| Memory (30K tiles) | ~1.2 GB | ~2-3 MB |
| Deduplication | Hash → in-memory data | Hash → file offset |

**Usage:**

```rust
use gpq_tiles_core::pipeline::{generate_tiles_to_writer, StreamingMode, TilerConfig};
use gpq_tiles_core::pmtiles_writer::StreamingPmtilesWriter;
use gpq_tiles_core::compression::Compression;

// Create streaming writer
let mut writer = StreamingPmtilesWriter::new(Compression::Gzip)?;

// Generate tiles - use LowMemory for very large files
let config = TilerConfig::new(0, 14)
    .with_streaming_mode(StreamingMode::LowMemory);
generate_tiles_to_writer(Path::new("large.parquet"), &config, &mut writer)?;

// Finalize assembles: header + directory + metadata + tile_data
writer.finalize(Path::new("output.pmtiles"))?;
```

**Memory breakdown:**

```
StreamingPmtilesWriter memory:
├── Directory entries: 24 bytes × total_tiles  (~720 KB for 30K tiles)
├── Dedup cache:       40 bytes × unique_tiles (~800 KB for 20K unique)
├── Temp file buffer:  64 KB
└── Total:             ~2-3 MB
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

Use `config.with_quiet(true)` to suppress warnings. See `quality.rs` for implementation.

## Module Structure

```
crates/core/src/
├── lib.rs              # Public API
├── tile.rs             # TileCoord, TileBounds
├── clip.rs             # Geometry clipping
├── simplify.rs         # RDP simplification
├── validate.rs         # Geometry validation
├── mvt.rs              # MVT encoding
├── pmtiles_writer.rs   # PMTiles v3 writer (PmtilesWriter + StreamingPmtilesWriter)
├── feature_drop.rs     # Dropping algorithms
├── spatial_index.rs    # Hilbert/Z-order curves
├── pipeline.rs         # Tile generation (streaming + non-streaming)
├── batch_processor.rs  # GeoArrow iteration (row group support)
├── memory.rs           # Memory tracking and estimation
├── quality.rs          # File quality assessment
├── dedup.rs            # Tile deduplication (XXH3)
├── compression.rs      # gzip/brotli/zstd compression
├── property_filter.rs  # Include/exclude field filtering
└── golden.rs           # Golden tests

crates/core/examples/
└── test_large_file.rs  # Large file streaming test
```
