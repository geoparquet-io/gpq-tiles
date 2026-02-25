# Quantization Risks in gpq-tiles

Research into where coordinate quantization happens and how valid polygons could become invalid after quantization.

## Coordinate Flow Summary

```
Input: float64 WGS84 (lng/lat degrees)
    ↓
[tile.rs:23-42] TileCoord::bounds() — calculate tile geographic bounds
    ↓
[clip.rs:47-75] clip_geometry() — clip to tile bounds + buffer (in degrees)
    ↓
[simplify.rs:42-88] simplify_for_zoom() — RDP simplification (in degrees)
    ↓
[validate.rs:90-120] filter_valid_geometry() — drop degenerate geometries
    ↓
[mvt.rs:134-146] geo_to_tile_coords() — QUANTIZATION HAPPENS HERE
    ↓
Output: i32 tile coordinates (0 to extent, typically 4096)
```

## The Critical Quantization Point

### Location: `crates/core/src/mvt.rs:134-146`

```rust
pub fn geo_to_tile_coords(lng: f64, lat: f64, bounds: &TileBounds, extent: u32) -> (i32, i32) {
    let extent_f = extent as f64;

    // Normalize to 0-1 within tile bounds
    let x_ratio = (lng - bounds.lng_min) / (bounds.lng_max - bounds.lng_min);
    let y_ratio = (lat - bounds.lat_min) / (bounds.lat_max - bounds.lat_min);

    // Scale to extent and flip Y (tile coords have Y increasing downward)
    let x = (x_ratio * extent_f).round() as i32;  // <-- QUANTIZATION
    let y = ((1.0 - y_ratio) * extent_f).round() as i32;  // <-- QUANTIZATION

    (x, y)
}
```

### What happens:
1. **Normalization**: Geographic coordinates are normalized to 0-1 range within tile bounds
2. **Scaling**: Multiplied by extent (default 4096) to get float tile coordinates
3. **Rounding**: `.round()` snaps to nearest integer
4. **Casting**: `as i32` converts to signed integer for MVT encoding

## Risk 1: Multiple Points Collapsing to Same Integer

### The Problem

Two distinct float64 coordinates that are close together may round to the **same** integer tile coordinate.

**Example:**
```
Geographic coordinates within a z14 tile (~15m wide):
  Point A: (-122.4194000, 37.7749000)  → tile coords (2047.4, 2047.6) → (2047, 2048)
  Point B: (-122.4194001, 37.7749001)  → tile coords (2047.3, 2047.7) → (2047, 2048)
```

Both points snap to (2047, 2048).

**Resolution at different zooms (4096 extent):**
| Zoom | Tile width | Pixel size | Features that collapse |
|------|-----------|------------|------------------------|
| Z0 | 360° | 0.088° (~9.7 km) | Very aggressive |
| Z6 | 5.625° | 0.00137° (~150 m) | Aggressive |
| Z10 | 0.352° | 0.000086° (~9.5 m) | Moderate |
| Z14 | 0.022° | 0.0000054° (~60 cm) | Minimal |

### Impact on Polygons

When multiple vertices of a polygon ring collapse to the same point:

**Before quantization (valid):**
```
Ring: A → B → C → D → A
      (distinct points forming valid polygon)
```

**After quantization (potentially invalid):**
```
Ring: A → A → C → D → A  (B snapped to A's location)
      ↑
      Degenerate: consecutive identical vertices
```

### Where this is NOT currently validated

The current validation in `validate.rs` checks:
- Minimum point count (4 for rings)
- Zero-area polygons

But it does **NOT** check:
- Consecutive identical vertices after quantization
- Self-intersections introduced by rounding

**Code reference:** `crates/core/src/validate.rs:148-177` validates polygons in geographic coordinates, BEFORE MVT encoding quantizes them.

## Risk 2: Edge Crossings from Rounding (Bowtie Polygons)

### The Problem

A valid (non-self-intersecting) polygon can become self-intersecting after quantization if rounding causes edges to cross.

**Example - Nearly parallel edges:**
```
                Before                          After Quantization

    A ─────────────── B              A ═══════════════ B
     \               /                  \           /
      \             /                    X         X  ← CROSSING!
       \           /                      \       /
    D ──────────── C              D ═════════════ C

    (Valid: edges don't cross)       (Invalid: edges cross at X)
```

This happens when:
1. Two edges are nearly parallel but don't intersect
2. Quantization snaps vertices in a way that reverses the relative position
3. The edges now cross, creating a "bowtie" or self-intersection

### Mathematical Explanation

Consider a quadrilateral with vertices A, B, C, D that forms a valid, non-self-intersecting polygon.

The edges AB and CD might be very close but not intersecting in float64 space:
```
Edge AB: from (100.4, 200.6) to (300.3, 200.4)
Edge CD: from (300.4, 200.3) to (100.3, 200.5)
```

After quantization (rounding to integers):
```
Edge AB: from (100, 201) to (300, 200)
Edge CD: from (300, 200) to (100, 201)  ← B and C now overlap, D and A overlap!
```

The polygon has collapsed into two overlapping line segments.

### Where this could occur

1. **Thin polygons**: Long, narrow polygons where the two long edges are close together
2. **Small features at low zoom**: A building polygon at z5 might be only 2-3 pixels wide
3. **Clipping artifacts**: When clipping creates a thin sliver along the tile boundary

## Current Mitigation (Partial)

### What gpq-tiles currently does:

1. **Douglas-Peucker simplification** (`simplify.rs:42-88`): Removes vertices below pixel tolerance, reducing vertex count

2. **Degenerate geometry filtering** (`validate.rs:148-177`): Drops polygons with:
   - Fewer than 4 points in a ring
   - Zero or near-zero area (`MIN_POLYGON_AREA = 1e-10`)

3. **Tiny polygon dropping** (`feature_drop.rs:150-166`): Uses diffuse probability to drop small polygons

### What is NOT currently handled:

1. **Post-quantization validation**: No check for self-intersections after MVT encoding
2. **Consecutive duplicate vertices**: Not detected after coordinate transformation
3. **Edge crossing detection**: Not implemented

## Tippecanoe's Approach

Tippecanoe handles quantization-induced invalidity in `tile.cpp`:

1. **Snap-rounding**: Uses consistent rounding rules
2. **Degenerate cleanup**: Removes consecutive duplicate points after quantization
3. **Coalesce**: Merges vertices that snap to same location
4. **Douglas-Peucker**: Aggressively simplifies before quantization

**Key difference**: Tippecanoe does NOT validate for self-intersections post-quantization. It relies on:
- Aggressive simplification to reduce the chance of edge crossings
- MVT renderers being tolerant of minor self-intersections

## Recommendations for Future Work

### Option 1: Post-Quantization Cleanup (Minimal)

Add duplicate vertex removal after `geo_to_tile_coords()`:

```rust
// After encoding ring, remove consecutive identical points
fn remove_consecutive_duplicates(commands: &mut Vec<u32>) {
    // Walk through LineTo commands and remove zero-delta moves
}
```

**Pros:** Simple, fast
**Cons:** Doesn't fix self-intersections

### Option 2: Post-Quantization Validation (Moderate)

Add validation after MVT encoding:

```rust
fn is_valid_mvt_polygon(commands: &[u32]) -> bool {
    // Decode commands back to points
    // Check for consecutive duplicates
    // Check for self-intersections using Shamos-Hoey or similar
}
```

**Pros:** Can detect problems
**Cons:** O(n log n) self-intersection check per polygon, may be expensive

### Option 3: Pre-Quantization Snapping (Thorough)

Quantize geometry to integer tile space BEFORE clipping/simplifying:

```rust
// Snap to grid before clip
let snapped = snap_to_tile_grid(&geom, &bounds, extent);
let clipped = clip_geometry(&snapped, &bounds, buffer);
```

**Pros:** All subsequent operations work in integer space, consistent
**Cons:** Changes the pipeline significantly, may affect simplification quality

### Option 4: Match Tippecanoe (Pragmatic)

Accept that some minor self-intersections may occur, similar to tippecanoe:
- Rely on aggressive simplification
- Accept that MVT renderers tolerate small issues
- Add metrics to monitor the frequency of potential problems

## Code References Summary

| Operation | File | Line | Function |
|-----------|------|------|----------|
| Tile bounds calculation | tile.rs | 23-42 | `TileCoord::bounds()` |
| Geometry clipping | clip.rs | 47-75 | `clip_geometry()` |
| Simplification | simplify.rs | 42-88 | `simplify_for_zoom()` |
| Validation (pre-quantization) | validate.rs | 90-120 | `validate_geometry()` |
| **QUANTIZATION** | mvt.rs | 134-146 | `geo_to_tile_coords()` |
| Ring encoding | mvt.rs | 273-317 | `encode_ring()` |
| Polygon encoding | mvt.rs | 325-350 | `encode_polygon()` |
| Feature addition | mvt.rs | 517-548 | `LayerBuilder::add_feature()` |
| Pipeline integration | pipeline.rs | 1215, 1264, 1491, 1647 | `layer_builder.add_feature()` |

## Related Issues

- No existing GitHub issues found for quantization-induced invalidity
- The `validate.rs` module documentation (line 1-12) explicitly mentions handling post-simplification degeneracy but not post-quantization issues
