# gpq-tiles Architecture

This document captures key design decisions and known divergences from tippecanoe.

## Design Principles

1. **Arrow-First**: Process geometries within Arrow batch scope to preserve zero-copy benefits. See CLAUDE.md for patterns.

2. **Semantic Comparison**: Golden tests use IoU and feature counts, not byte-exact comparison. Byte-exact is impossible for alternative implementations.

3. **Reference Implementations**: All algorithms match tippecanoe behavior. Consult [tippecanoe](https://github.com/felt/tippecanoe) for algorithm decisions.

4. **PMTiles Writer**: The `pmtiles` crate is read-only. We implement our own v3 writer from spec with Hilbert curve ordering.

## Known Divergences from Tippecanoe

| Area | Description | Tippecanoe Reference | Resolution |
|------|-------------|---------------------|------------|
| Metadata | Empty `{}` JSON vs full layer/field metadata | PMTiles spec §metadata | Future enhancement |
| Simplification tolerance | Our tolerance formula may differ slightly | `tile.cpp` | Tuned to match output quality |
| Clipping buffer | Configurable; tippecanoe uses 8 pixels default | `--buffer` flag | Match tippecanoe default |

## Golden Comparison Results

Validated against tippecanoe v2.49.0 using `open-buildings.parquet` (Andorra, ~1000 buildings):

| Zoom | Tippecanoe Features | gpq-tiles Features | Ratio | Notes |
|------|--------------------|--------------------|-------|-------|
| Z5 | 1 | 1000 | - | tippecanoe drops aggressively |
| Z8 | 97 | 1000 | 10.3x | tippecanoe drops 90% at this zoom |
| Z10 | 484 | 684 | 1.41x | Acceptable baseline |

**Analysis:**
- At high zoom (Z10), within 1.5x of tippecanoe - acceptable baseline
- At low zoom (Z5-Z8), tippecanoe's feature dropping dominates - addressed in Phase 3
- Area preservation after clip+simplify: 88% - good fidelity
- All zoom levels produce valid MVT tiles

**After Phase 3 integration**, the ratios should improve significantly as our dropping algorithms match tippecanoe's behavior.

## Critical Issues (Resolved)

| Issue | Status | Resolution |
|-------|--------|------------|
| Simplification coordinate space | ✅ FIXED | `simplify_in_tile_coords()` with pixel-based tolerance |
| Antimeridian crossing | ✅ FIXED | `tiles_for_bbox()` splits bbox at 180° |
| Polygon winding validation | ✅ FIXED | `orient_polygon_for_mvt()` auto-corrects winding order |
| Degenerate geometry handling | ✅ FIXED | `validate.rs` module filters invalid geometries post-simplification |
| Value deduplication | ✅ FIXED | `PropertyValue` implements proper `Hash`/`Eq` traits |

## Remaining Issues

| Issue | Description | Workaround |
|-------|-------------|------------|
| Memory for large files | `extract_geometries` loads all into Vec | Document limitation; Phase 4 adds streaming |

## Feature Dropping Algorithms

The `feature_drop` module implements tippecanoe-compatible dropping (Phase 3):

### Tiny Polygon Reduction
- Threshold: 4 square pixels (configurable)
- Algorithm: Diffuse probability - smaller polygons have higher drop probability
- Deterministic: Same polygon always produces same decision via geometry hash

### Line Dropping
- Algorithm: Coordinate quantization - drop when all vertices collapse to same pixel
- No explicit length threshold - based on visual extent at zoom level

### Point Thinning
- Drop rate: 1/2.5 per zoom level above base zoom
- At base_zoom: 100% retention
- At base_zoom - 1: 40% retention
- At base_zoom - 2: 16% retention
- Deterministic via Murmur3-style hash on feature index

## Testing Strategy

### Test Tiers

1. **Tier 1 (Structural)**: Every `cargo test` - PMTiles validity, MVT decoding, coordinate transforms
2. **Tier 2-4 (Golden)**: `cargo test --features golden` - IoU comparison, feature counts
3. **Tier 5 (Performance)**: `cargo bench` - Criterion benchmarks with regression detection

### Key Tests

- `test_golden_open_buildings_z10_feature_count` - verifies feature ratio
- `test_document_low_zoom_feature_dropping_difference` - documents expected differences
- `test_golden_polygon_area_preserved_z10` - verifies area preservation

## Module Structure

```
crates/core/src/
├── lib.rs              # Public API
├── tile.rs             # TileCoord, TileBounds, coordinate math
├── clip.rs             # Geometry clipping with buffer
├── simplify.rs         # RDP simplification in tile coordinates
├── validate.rs         # Geometry validation (degenerate detection, winding order)
├── mvt.rs              # MVT protobuf encoding (with auto winding correction)
├── pmtiles_writer.rs   # Custom PMTiles v3 writer
├── feature_drop.rs     # Tippecanoe-compatible dropping algorithms
├── pipeline.rs         # Tile generation orchestration
├── batch_processor.rs  # GeoArrow batch iteration
└── golden.rs           # Golden comparison tests
```
