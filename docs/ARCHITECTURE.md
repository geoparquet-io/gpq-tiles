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
| Metadata | Empty `{}` JSON | Full layer/field metadata | Future enhancement |
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
├── pipeline.rs         # Tile generation
├── batch_processor.rs  # GeoArrow iteration
└── golden.rs           # Golden tests
```
