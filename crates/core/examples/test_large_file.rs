//! Test streaming PMTiles writer with a large GeoParquet file.
//!
//! Usage:
//!   cargo run --example test_large_file --release [path_to_large_file.parquet] [--deterministic]
//!
//! Options:
//!   --deterministic    Use deterministic (sequential) processing for reproducible output.
//!                      Default is parallel processing (faster).
//!
//! If no path is provided, uses the default test fixture location.

use std::path::Path;
use std::time::Instant;

use gpq_tiles_core::compression::Compression;
use gpq_tiles_core::pipeline::{generate_tiles_to_writer, TilerConfig};
use gpq_tiles_core::pmtiles_writer::StreamingPmtilesWriter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    // Parse args
    let args: Vec<String> = std::env::args().collect();
    let deterministic = args.iter().any(|a| a == "--deterministic");
    let input_path = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .map(|s| Path::new(s).to_path_buf())
        .unwrap_or_else(|| {
            // Try multiple potential locations
            let paths = [
                "tests/fixtures/realdata/adm4_polygons.parquet",
                "../../tests/fixtures/realdata/adm4_polygons.parquet",
                "../tests/fixtures/realdata/adm4_polygons.parquet",
            ];
            paths
                .iter()
                .map(|p| Path::new(p).to_path_buf())
                .find(|p| p.exists())
                .unwrap_or_else(|| {
                    eprintln!("Error: No input file found. Please provide a path:");
                    eprintln!(
                        "  cargo run --example test_large_file --release /path/to/file.parquet"
                    );
                    std::process::exit(1);
                })
        });

    if !input_path.exists() {
        eprintln!("Error: Input file not found: {:?}", input_path);
        std::process::exit(1);
    }

    let file_size = std::fs::metadata(&input_path)?.len();
    let mode_name = if deterministic {
        "Deterministic (sequential)"
    } else {
        "Parallel"
    };

    println!("=== Streaming PMTiles Writer Test ===");
    println!("Input: {:?}", input_path);
    println!(
        "File size: {:.2} GB",
        file_size as f64 / 1024.0 / 1024.0 / 1024.0
    );
    println!("Processing mode: {}", mode_name);
    println!();

    let config = TilerConfig::new(0, 6)
        .with_layer_name("features")
        .with_deterministic(deterministic)
        .with_memory_budget(512 * 1024 * 1024); // 512MB budget

    let output_path = Path::new("/tmp/streaming-test-output.pmtiles");
    let _ = std::fs::remove_file(output_path);

    println!("Generating tiles (z0-z6)...");
    let start = Instant::now();

    // Create streaming writer
    let mut writer = StreamingPmtilesWriter::new(Compression::Gzip)?;

    // Generate tiles directly to the streaming writer
    let memory_stats = generate_tiles_to_writer(&input_path, &config, &mut writer)?;

    let generation_time = start.elapsed();
    println!("  Generation: {:?}", generation_time);

    // Get write stats before finalize
    let write_stats = writer.stats().clone();

    println!("Finalizing PMTiles...");
    let finalize_start = Instant::now();
    let final_stats = writer.finalize(output_path)?;
    let finalize_time = finalize_start.elapsed();
    println!("  Finalization: {:?}", finalize_time);

    let total_time = start.elapsed();

    println!();
    println!("=== Results ===");
    println!("Total time: {:?}", total_time);
    println!();
    println!("Tiles:");
    println!("  Total tiles: {}", final_stats.total_tiles);
    println!("  Unique tiles: {}", final_stats.unique_tiles);
    println!(
        "  Bytes written: {:.2} MB",
        final_stats.bytes_written as f64 / 1024.0 / 1024.0
    );
    println!(
        "  Bytes saved (dedup): {:.2} MB",
        final_stats.bytes_saved_dedup as f64 / 1024.0 / 1024.0
    );
    println!();
    println!("Memory:");
    println!("  Peak usage: {}", memory_stats.peak_formatted());
    if let Some(budget_str) = memory_stats.budget_formatted() {
        println!("  Budget: {}", budget_str);
        println!("  Within budget: {}", memory_stats.within_budget());
        println!(
            "  Budget exceeded: {} times",
            memory_stats.budget_exceeded_count
        );
    }
    println!(
        "  Estimated streaming overhead: {:.2} MB",
        write_stats.estimated_memory_bytes() as f64 / 1024.0 / 1024.0
    );
    println!();

    // Output file info
    let output_size = std::fs::metadata(output_path)?.len();
    println!("Output:");
    println!("  Path: {:?}", output_path);
    println!("  Size: {:.2} MB", output_size as f64 / 1024.0 / 1024.0);

    // Verify output
    let output_data = std::fs::read(output_path)?;
    assert_eq!(&output_data[0..7], b"PMTiles", "Valid PMTiles magic");
    assert_eq!(output_data[7], 3, "PMTiles v3");
    println!("  Verification: OK");

    Ok(())
}
