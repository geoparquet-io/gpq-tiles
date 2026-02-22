# Real-World Test Fixtures

Production data samples for testing gpq-tiles tiling performance.

## Fixtures

| File | Features | Size | Source | Use Case |
|------|----------|------|--------|----------|
| `open-buildings.parquet` | 1,000 | 143KB | Google Open Buildings | Quick tests, golden comparisons |
| `fieldmaps-madagascar-adm4.parquet` | 17,465 | 28MB | [FieldMaps](https://fieldmaps.io) | **Parallelization benchmarks** |
| `fieldmaps-boundaries.parquet` | 3 | 2.2MB | FieldMaps | Large polygon tests |
| `road-detections.parquet` | ~1,000 | 90KB | Road detection ML | LineString tests |

## Attribution

- **FieldMaps data** courtesy of Maxym Malchenko ([fieldmaps.io](https://fieldmaps.io)) — edge-matched humanitarian admin boundaries
- **Google Open Buildings** — CC BY 4.0
- **Road detections** — derived from ML model outputs

## Git LFS

Large fixtures are tracked with Git LFS. After cloning:

```bash
git lfs pull
```
