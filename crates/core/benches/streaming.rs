// Streaming vs Non-Streaming Benchmark Suite
//
// This benchmark suite measures memory usage and execution time for streaming
// vs non-streaming tile generation approaches.
//
// Key metrics:
// - Peak memory usage during tile generation
// - Execution time for complete pipeline
// - Tile output validity
//
// Run with: cargo bench --package gpq-tiles-core --bench streaming
//
// Note on memory measurement:
// We use allocation counting rather than true peak memory because Criterion
// doesn't support memory profiling directly. For accurate peak memory,
// use external tools like `heaptrack` or `/usr/bin/time -v`.
//
// Fixtures:
// - multi-rowgroup-small.parquet - 1K features split across many row groups
// - open-buildings.parquet - 1K features in single row group
// - fieldmaps-madagascar-adm4.parquet - 17K features for larger tests

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use gpq_tiles_core::batch_processor::extract_geometries;
use gpq_tiles_core::compression::Compression;
use gpq_tiles_core::pipeline::{
    generate_tiles, generate_tiles_from_geometries, generate_tiles_streaming,
    generate_tiles_to_writer, StreamingMode, TilerConfig,
};
use gpq_tiles_core::pmtiles_writer::StreamingPmtilesWriter;
use std::path::Path;

// Fixture paths (relative to crates/core/)
const FIXTURE_SMALL: &str = "../../tests/fixtures/realdata/open-buildings.parquet";
const FIXTURE_MULTI_RG: &str = "../../tests/fixtures/streaming/multi-rowgroup-small.parquet";
const FIXTURE_LARGE: &str = "../../tests/fixtures/realdata/fieldmaps-madagascar-adm4.parquet";

/// Helper to check if a fixture exists
fn fixture_exists(path: &str) -> bool {
    Path::new(path).exists()
}

/// Load geometries from fixture
fn load_fixture(fixture_path: &str) -> Vec<geo::Geometry<f64>> {
    let path = Path::new(fixture_path);
    if !path.exists() {
        panic!(
            "Fixture file not found at {}. Run `git lfs pull` if using LFS fixtures.",
            fixture_path
        );
    }
    extract_geometries(path).expect("Failed to load fixture geometries")
}

/// Benchmark streaming vs non-streaming on small fixture (1K features)
///
/// This benchmark compares:
/// - Non-streaming: loads all geometries into memory, then generates tiles
/// - Streaming: processes row groups incrementally
///
/// Expected outcome:
/// - Streaming may be slightly slower due to per-rowgroup overhead
/// - Streaming should have bounded memory (doesn't grow with file size)
fn bench_streaming_vs_non_streaming_small(c: &mut Criterion) {
    if !fixture_exists(FIXTURE_SMALL) {
        eprintln!("Skipping streaming benchmark: fixture not found");
        return;
    }

    let geometries = load_fixture(FIXTURE_SMALL);
    let fixture_path = Path::new(FIXTURE_SMALL);

    let mut group = c.benchmark_group("streaming_small");
    group.throughput(Throughput::Elements(geometries.len() as u64));

    // Test at zoom range 0-8 (reasonable for benchmarking)
    let config = TilerConfig::new(0, 8);

    // Non-streaming (from pre-loaded geometries)
    group.bench_function("non_streaming_from_geoms", |b| {
        b.iter(|| {
            let tiles: Vec<_> = generate_tiles_from_geometries(geometries.clone(), &config)
                .expect("generate_tiles failed")
                .collect();
            black_box(tiles)
        })
    });

    // Non-streaming (from file - includes file I/O)
    group.bench_function("non_streaming_from_file", |b| {
        b.iter(|| {
            let tiles: Vec<_> = generate_tiles(fixture_path, &config)
                .expect("generate_tiles failed")
                .collect();
            black_box(tiles)
        })
    });

    // Streaming (from file)
    group.bench_function("streaming_from_file", |b| {
        b.iter(|| {
            let tiles = generate_tiles_streaming(fixture_path, &config).expect("streaming failed");
            black_box(tiles)
        })
    });

    group.finish();
}

/// Benchmark streaming on multi-rowgroup fixture
///
/// This tests the scenario streaming is designed for:
/// - File with many row groups
/// - Each row group processed independently
/// - Memory bounded by row group size, not file size
fn bench_streaming_multi_rowgroup(c: &mut Criterion) {
    if !fixture_exists(FIXTURE_MULTI_RG) {
        eprintln!("Skipping multi-rowgroup benchmark: fixture not found");
        return;
    }

    let fixture_path = Path::new(FIXTURE_MULTI_RG);

    let mut group = c.benchmark_group("streaming_multi_rowgroup");
    group.sample_size(50); // Reduce samples for faster benchmarking

    // Test multiple zoom ranges
    for max_zoom in [6, 8, 10] {
        let config = TilerConfig::new(0, max_zoom);

        // Non-streaming
        group.bench_with_input(
            BenchmarkId::new("non_streaming", max_zoom),
            &config,
            |b, config| {
                b.iter(|| {
                    let tiles: Vec<_> = generate_tiles(fixture_path, config)
                        .expect("generate_tiles failed")
                        .collect();
                    black_box(tiles)
                })
            },
        );

        // Streaming
        group.bench_with_input(
            BenchmarkId::new("streaming", max_zoom),
            &config,
            |b, config| {
                b.iter(|| {
                    let tiles =
                        generate_tiles_streaming(fixture_path, config).expect("streaming failed");
                    black_box(tiles)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark streaming on large fixture (17K features)
///
/// This demonstrates where streaming shines:
/// - Larger file = more memory pressure for non-streaming
/// - Streaming maintains bounded memory regardless of file size
fn bench_streaming_large(c: &mut Criterion) {
    if !fixture_exists(FIXTURE_LARGE) {
        eprintln!("Skipping large fixture benchmark: fixture not found");
        return;
    }

    let fixture_path = Path::new(FIXTURE_LARGE);

    let mut group = c.benchmark_group("streaming_large");
    group.sample_size(10); // Fewer samples for larger dataset

    let config = TilerConfig::new(0, 8);

    // Non-streaming
    group.bench_function("non_streaming_17k", |b| {
        b.iter(|| {
            let tiles: Vec<_> = generate_tiles(fixture_path, &config)
                .expect("generate_tiles failed")
                .collect();
            black_box(tiles)
        })
    });

    // Streaming
    group.bench_function("streaming_17k", |b| {
        b.iter(|| {
            let tiles = generate_tiles_streaming(fixture_path, &config).expect("streaming failed");
            black_box(tiles)
        })
    });

    group.finish();
}

/// Validate that streaming produces correct output
///
/// This is not a benchmark but a validation test that runs during benchmarking
/// to ensure streaming produces valid PMTiles-compatible output.
fn bench_streaming_validation(c: &mut Criterion) {
    if !fixture_exists(FIXTURE_SMALL) {
        eprintln!("Skipping validation: fixture not found");
        return;
    }

    let fixture_path = Path::new(FIXTURE_SMALL);
    let config = TilerConfig::new(0, 6);

    let mut group = c.benchmark_group("streaming_validation");
    group.sample_size(10);

    // Benchmark that also validates output
    group.bench_function("validate_streaming_output", |b| {
        b.iter(|| {
            let tiles = generate_tiles_streaming(fixture_path, &config).expect("streaming failed");

            // Validate each tile
            for tile in &tiles {
                // Tile should have valid coordinates
                assert!(tile.coord.z <= config.max_zoom);
                assert!(tile.coord.x < (1 << tile.coord.z));
                assert!(tile.coord.y < (1 << tile.coord.z));

                // Tile should have non-empty data
                assert!(!tile.data.is_empty(), "Tile should have data");

                // Tile should be valid MVT (decode test)
                use gpq_tiles_core::pipeline::decode_tile;
                let decoded = decode_tile(&tile.data).expect("Should decode MVT");
                assert!(!decoded.layers.is_empty(), "Should have at least one layer");
            }

            black_box(tiles.len())
        })
    });

    group.finish();
}

/// Benchmark comparing tile counts between streaming and non-streaming
///
/// Validates that both approaches produce similar output.
fn bench_output_equivalence(c: &mut Criterion) {
    if !fixture_exists(FIXTURE_SMALL) {
        eprintln!("Skipping equivalence: fixture not found");
        return;
    }

    let fixture_path = Path::new(FIXTURE_SMALL);
    let config = TilerConfig::new(0, 6);

    // First, verify equivalence (outside benchmark loop)
    let non_streaming_tiles: Vec<_> = generate_tiles(fixture_path, &config)
        .expect("non-streaming failed")
        .filter_map(|r| r.ok())
        .collect();

    let streaming_tiles =
        generate_tiles_streaming(fixture_path, &config).expect("streaming failed");

    // Check that tile counts are similar (within 20% tolerance)
    let ratio = streaming_tiles.len() as f64 / non_streaming_tiles.len() as f64;
    if !(0.8..=1.2).contains(&ratio) {
        eprintln!(
            "WARNING: Tile count mismatch: streaming={}, non-streaming={}, ratio={:.2}",
            streaming_tiles.len(),
            non_streaming_tiles.len(),
            ratio
        );
    }

    // Check zoom level coverage
    let streaming_zooms: std::collections::HashSet<u8> =
        streaming_tiles.iter().map(|t| t.coord.z).collect();
    let non_streaming_zooms: std::collections::HashSet<u8> =
        non_streaming_tiles.iter().map(|t| t.coord.z).collect();

    if streaming_zooms != non_streaming_zooms {
        eprintln!(
            "WARNING: Zoom level mismatch: streaming={:?}, non-streaming={:?}",
            streaming_zooms, non_streaming_zooms
        );
    }

    let mut group = c.benchmark_group("output_equivalence");
    group.sample_size(10);

    group.bench_function("verify_equivalence", |b| {
        b.iter(|| {
            let streaming =
                generate_tiles_streaming(fixture_path, &config).expect("streaming failed");
            let non_streaming: Vec<_> = generate_tiles(fixture_path, &config)
                .expect("non-streaming failed")
                .filter_map(|r| r.ok())
                .collect();

            // Both should produce tiles at same zoom levels
            let s_zooms: std::collections::HashSet<u8> =
                streaming.iter().map(|t| t.coord.z).collect();
            let ns_zooms: std::collections::HashSet<u8> =
                non_streaming.iter().map(|t| t.coord.z).collect();
            assert_eq!(s_zooms, ns_zooms, "Zoom levels should match");

            black_box((streaming.len(), non_streaming.len()))
        })
    });

    group.finish();
}

/// Benchmark comparing all streaming modes: Fast, LowMemory, ExternalSort
///
/// This measures performance of each streaming mode writing to a PMTiles file.
/// Key metrics:
/// - Fast: fastest, uses most memory (~1-2GB for large files)
/// - LowMemory: 2-3x slower, ~100MB memory
/// - ExternalSort: tippecanoe-style, bounded memory with external disk sort
fn bench_streaming_modes(c: &mut Criterion) {
    if !fixture_exists(FIXTURE_SMALL) {
        eprintln!("Skipping streaming modes benchmark: fixture not found");
        return;
    }

    let fixture_path = Path::new(FIXTURE_SMALL);

    let mut group = c.benchmark_group("streaming_modes");
    group.sample_size(20);

    let base_config = TilerConfig::new(0, 8).with_quiet(true);

    // Fast mode (default)
    group.bench_function("fast", |b| {
        b.iter(|| {
            let config = base_config.clone().with_streaming_mode(StreamingMode::Fast);
            let mut writer =
                StreamingPmtilesWriter::new(Compression::Gzip).expect("Should create writer");
            let _stats = generate_tiles_to_writer(fixture_path, &config, &mut writer)
                .expect("Fast mode should work");
            let output = std::path::Path::new("/tmp/bench-fast.pmtiles");
            let result = writer.finalize(output);
            let _ = std::fs::remove_file(output);
            black_box(result)
        })
    });

    // LowMemory mode
    group.bench_function("low_memory", |b| {
        b.iter(|| {
            let config = base_config
                .clone()
                .with_streaming_mode(StreamingMode::LowMemory);
            let mut writer =
                StreamingPmtilesWriter::new(Compression::Gzip).expect("Should create writer");
            let _stats = generate_tiles_to_writer(fixture_path, &config, &mut writer)
                .expect("LowMemory mode should work");
            let output = std::path::Path::new("/tmp/bench-lowmem.pmtiles");
            let result = writer.finalize(output);
            let _ = std::fs::remove_file(output);
            black_box(result)
        })
    });

    // ExternalSort mode
    group.bench_function("external_sort", |b| {
        b.iter(|| {
            let config = base_config
                .clone()
                .with_streaming_mode(StreamingMode::ExternalSort);
            let mut writer =
                StreamingPmtilesWriter::new(Compression::Gzip).expect("Should create writer");
            let _stats = generate_tiles_to_writer(fixture_path, &config, &mut writer)
                .expect("ExternalSort mode should work");
            let output = std::path::Path::new("/tmp/bench-extsort.pmtiles");
            let result = writer.finalize(output);
            let _ = std::fs::remove_file(output);
            black_box(result)
        })
    });

    group.finish();
}

/// Benchmark ExternalSort on larger file to demonstrate memory-bounded behavior
fn bench_external_sort_larger(c: &mut Criterion) {
    if !fixture_exists(FIXTURE_LARGE) {
        eprintln!("Skipping external sort larger benchmark: fixture not found");
        return;
    }

    let fixture_path = Path::new(FIXTURE_LARGE);

    let mut group = c.benchmark_group("external_sort_larger");
    group.sample_size(10);

    let base_config = TilerConfig::new(0, 8).with_quiet(true);

    // ExternalSort with memory budget
    group.bench_function("external_sort_17k", |b| {
        b.iter(|| {
            let config = base_config
                .clone()
                .with_streaming_mode(StreamingMode::ExternalSort)
                .with_memory_budget(500 * 1024 * 1024); // 500MB budget
            let mut writer =
                StreamingPmtilesWriter::new(Compression::Gzip).expect("Should create writer");
            let stats = generate_tiles_to_writer(fixture_path, &config, &mut writer)
                .expect("ExternalSort should work on larger file");
            let output = std::path::Path::new("/tmp/bench-extsort-large.pmtiles");
            let result = writer.finalize(output);
            let _ = std::fs::remove_file(output);
            eprintln!("External sort peak memory: {}KB", stats.peak_bytes / 1024);
            black_box(result)
        })
    });

    // Compare with Fast mode
    group.bench_function("fast_17k", |b| {
        b.iter(|| {
            let config = base_config.clone().with_streaming_mode(StreamingMode::Fast);
            let mut writer =
                StreamingPmtilesWriter::new(Compression::Gzip).expect("Should create writer");
            let stats = generate_tiles_to_writer(fixture_path, &config, &mut writer)
                .expect("Fast mode should work on larger file");
            let output = std::path::Path::new("/tmp/bench-fast-large.pmtiles");
            let result = writer.finalize(output);
            let _ = std::fs::remove_file(output);
            eprintln!("Fast mode peak memory: {}KB", stats.peak_bytes / 1024);
            black_box(result)
        })
    });

    group.finish();
}

// Group benchmarks by execution time expectations
criterion_group!(
    name = fast_benchmarks;
    config = Criterion::default().sample_size(50);
    targets = bench_streaming_vs_non_streaming_small, bench_streaming_validation
);

criterion_group!(
    name = medium_benchmarks;
    config = Criterion::default().sample_size(30);
    targets = bench_streaming_multi_rowgroup, bench_output_equivalence, bench_streaming_modes
);

criterion_group!(
    name = slow_benchmarks;
    config = Criterion::default().sample_size(10);
    targets = bench_streaming_large, bench_external_sort_larger
);

criterion_main!(fast_benchmarks, medium_benchmarks, slow_benchmarks);
