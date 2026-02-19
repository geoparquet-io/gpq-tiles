// Benchmark suite for tiling performance
// Phase 1: Stub benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gpq_tiles_core::{Config, Converter};

fn bench_converter_creation(c: &mut Criterion) {
    c.bench_function("converter_creation", |b| {
        b.iter(|| {
            let config = Config::default();
            black_box(Converter::new(config))
        })
    });
}

// TODO: Add benchmarks for:
// - Single tile encoding
// - Full pipeline at zoom 0-8
// - Feature dropping at various densities

criterion_group!(benches, bench_converter_creation);
criterion_main!(benches);
