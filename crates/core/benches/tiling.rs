// Benchmark suite for tiling performance
//
// Uses the golden data fixtures (open-buildings.parquet) for realistic benchmarks.
//
// Run with: cargo bench --package gpq-tiles-core

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use gpq_tiles_core::batch_processor::extract_geometries;
use gpq_tiles_core::pipeline::{generate_single_tile, generate_tiles_from_geometries, TilerConfig};
use gpq_tiles_core::tile::TileCoord;
use std::path::Path;

// Path to the golden fixture (relative to crates/core/)
const FIXTURE_PATH: &str = "../../tests/fixtures/realdata/open-buildings.parquet";

/// Load geometries from the golden fixture file
fn load_fixture_geometries() -> Vec<geo::Geometry<f64>> {
    let path = Path::new(FIXTURE_PATH);
    if !path.exists() {
        panic!(
            "Fixture file not found at {}. Run from crates/core/ or project root.",
            FIXTURE_PATH
        );
    }
    extract_geometries(path).expect("Failed to load fixture geometries")
}

/// Benchmark single tile generation at various zoom levels
fn bench_single_tile(c: &mut Criterion) {
    let geometries = load_fixture_geometries();
    let config = TilerConfig::new(0, 14);

    let mut group = c.benchmark_group("single_tile");
    group.throughput(Throughput::Elements(geometries.len() as u64));

    // Benchmark at different zoom levels
    // Z10/516/377 is the main tile covering our fixture
    for (z, x, y) in [(8, 129, 94), (10, 516, 377)] {
        let coord = TileCoord::new(x, y, z);
        group.bench_with_input(BenchmarkId::new("z", z), &coord, |b, coord| {
            b.iter(|| {
                let result = generate_single_tile(&geometries, *coord, &config);
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark full pipeline at various zoom ranges
fn bench_full_pipeline(c: &mut Criterion) {
    let geometries = load_fixture_geometries();

    let mut group = c.benchmark_group("full_pipeline");
    group.throughput(Throughput::Elements(geometries.len() as u64));

    // Benchmark different zoom ranges
    for max_zoom in [8, 10] {
        let config = TilerConfig::new(0, max_zoom);
        group.bench_with_input(
            BenchmarkId::new("max_zoom", max_zoom),
            &config,
            |b, config| {
                b.iter(|| {
                    let tiles: Vec<_> = generate_tiles_from_geometries(geometries.clone(), &config)
                        .expect("generate_tiles failed")
                        .collect();
                    black_box(tiles)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark parallel vs sequential tile generation
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let geometries = load_fixture_geometries();

    let mut group = c.benchmark_group("parallel_vs_sequential");
    group.throughput(Throughput::Elements(geometries.len() as u64));

    // Sequential
    let config_seq = TilerConfig::new(0, 10).with_parallel(false);
    group.bench_function("sequential_z0_10", |b| {
        b.iter(|| {
            let tiles: Vec<_> = generate_tiles_from_geometries(geometries.clone(), &config_seq)
                .expect("generate_tiles failed")
                .collect();
            black_box(tiles)
        })
    });

    // Parallel
    let config_par = TilerConfig::new(0, 10).with_parallel(true);
    group.bench_function("parallel_z0_10", |b| {
        b.iter(|| {
            let tiles: Vec<_> = generate_tiles_from_geometries(geometries.clone(), &config_par)
                .expect("generate_tiles failed")
                .collect();
            black_box(tiles)
        })
    });

    group.finish();
}

/// Benchmark with density dropping enabled vs disabled
fn bench_density_dropping(c: &mut Criterion) {
    let geometries = load_fixture_geometries();

    let mut group = c.benchmark_group("density_dropping");
    group.throughput(Throughput::Elements(geometries.len() as u64));

    // Without density dropping (default)
    let config_no_drop = TilerConfig::new(0, 10);
    group.bench_function("no_density_drop", |b| {
        b.iter(|| {
            let tiles: Vec<_> = generate_tiles_from_geometries(geometries.clone(), &config_no_drop)
                .expect("generate_tiles failed")
                .collect();
            black_box(tiles)
        })
    });

    // With density dropping
    let config_with_drop = TilerConfig::new(0, 10)
        .with_density_drop(true)
        .with_density_cell_size(32);
    group.bench_function("with_density_drop", |b| {
        b.iter(|| {
            let tiles: Vec<_> =
                generate_tiles_from_geometries(geometries.clone(), &config_with_drop)
                    .expect("generate_tiles failed")
                    .collect();
            black_box(tiles)
        })
    });

    group.finish();
}

/// Benchmark Hilbert vs Z-order sorting
fn bench_hilbert_vs_zorder(c: &mut Criterion) {
    let geometries = load_fixture_geometries();

    let mut group = c.benchmark_group("hilbert_vs_zorder");
    group.throughput(Throughput::Elements(geometries.len() as u64));

    // Hilbert (default)
    let config_hilbert = TilerConfig::new(0, 10).with_hilbert(true);
    group.bench_function("hilbert", |b| {
        b.iter(|| {
            let tiles: Vec<_> = generate_tiles_from_geometries(geometries.clone(), &config_hilbert)
                .expect("generate_tiles failed")
                .collect();
            black_box(tiles)
        })
    });

    // Z-order
    let config_zorder = TilerConfig::new(0, 10).with_hilbert(false);
    group.bench_function("zorder", |b| {
        b.iter(|| {
            let tiles: Vec<_> = generate_tiles_from_geometries(geometries.clone(), &config_zorder)
                .expect("generate_tiles failed")
                .collect();
            black_box(tiles)
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_single_tile,
    bench_full_pipeline,
    bench_parallel_vs_sequential,
    bench_density_dropping,
    bench_hilbert_vs_zorder,
);
criterion_main!(benches);
