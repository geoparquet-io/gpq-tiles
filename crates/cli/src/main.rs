//! CLI for gpq-tiles - Convert GeoParquet to PMTiles
//!
//! This is a thin wrapper around the gpq-tiles-core library.

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use gpq_tiles_core::compression::Compression;
use gpq_tiles_core::pipeline::{
    generate_tiles_to_writer, generate_tiles_to_writer_with_progress, ProgressEvent, StreamingMode,
    TilerConfig,
};
use gpq_tiles_core::pmtiles_writer::StreamingPmtilesWriter;
use gpq_tiles_core::PropertyFilter;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "gpq-tiles",
    about = "Convert GeoParquet to PMTiles vector tiles",
    version
)]
struct Args {
    /// Input GeoParquet file
    #[arg(value_name = "INPUT")]
    input: PathBuf,

    /// Output PMTiles file
    #[arg(value_name = "OUTPUT")]
    output: PathBuf,

    /// Minimum zoom level
    #[arg(long, default_value = "0")]
    min_zoom: u8,

    /// Maximum zoom level
    #[arg(long, default_value = "14")]
    max_zoom: u8,

    /// Feature dropping density (low, medium, high)
    #[arg(long, default_value = "medium")]
    drop_density: String,

    /// Layer name for the output tiles (default: derived from input filename)
    #[arg(long)]
    layer_name: Option<String>,

    /// Include only specified properties in output tiles (whitelist).
    /// Can be specified multiple times. Matches tippecanoe's -y flag.
    /// Example: --include name --include population
    #[arg(short = 'y', long = "include", value_name = "FIELD")]
    include: Vec<String>,

    /// Exclude specified properties from output tiles (blacklist).
    /// Can be specified multiple times. Matches tippecanoe's -x flag.
    /// Example: --exclude internal_id --exclude temp_field
    #[arg(short = 'x', long = "exclude", value_name = "FIELD")]
    exclude: Vec<String>,

    /// Exclude all properties, keeping only geometry.
    /// Matches tippecanoe's -X flag.
    #[arg(short = 'X', long = "exclude-all")]
    exclude_all: bool,

    /// Compression algorithm for tiles (gzip, brotli, zstd, none)
    #[arg(long, default_value = "gzip")]
    compression: String,

    /// Enable verbose logging with progress bars
    #[arg(short, long)]
    verbose: bool,

    /// Streaming mode for processing large files
    #[arg(long, value_enum, default_value = "fast")]
    streaming_mode: CliStreamingMode,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliStreamingMode {
    /// Fast mode: single pass, ~1-2GB memory for large files
    Fast,
    /// Low memory mode: re-reads file per zoom level, ~100MB memory
    LowMemory,
    /// External sort: tippecanoe-style disk-based sort, bounded memory
    ExternalSort,
}

impl Args {
    fn parse_property_filter(&self) -> Result<PropertyFilter> {
        // Check for conflicting options
        let has_include = !self.include.is_empty();
        let has_exclude = !self.exclude.is_empty();

        if self.exclude_all && (has_include || has_exclude) {
            anyhow::bail!("Cannot use --exclude-all with --include or --exclude");
        }

        if has_include && has_exclude {
            anyhow::bail!("Cannot use --include and --exclude together");
        }

        if self.exclude_all {
            Ok(PropertyFilter::ExcludeAll)
        } else if has_include {
            Ok(PropertyFilter::include(self.include.clone()))
        } else if has_exclude {
            Ok(PropertyFilter::exclude(self.exclude.clone()))
        } else {
            Ok(PropertyFilter::None)
        }
    }

    fn parse_compression(&self) -> Result<Compression> {
        Compression::from_str(&self.compression).ok_or_else(|| {
            anyhow::anyhow!(
                "Invalid compression: '{}'. Valid options: none, gzip, brotli, zstd",
                self.compression
            )
        })
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging - suppress when verbose (we use progress bars instead)
    let log_level = if args.verbose { "warn" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();

    // Parse options
    let property_filter = args
        .parse_property_filter()
        .context("Failed to parse property filter")?;
    let compression = args
        .parse_compression()
        .context("Failed to parse compression")?;

    // Derive layer name from input filename if not specified
    let layer_name = args.layer_name.clone().unwrap_or_else(|| {
        args.input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("layer")
            .to_string()
    });

    // Map CLI streaming mode to core streaming mode
    let streaming_mode = match args.streaming_mode {
        CliStreamingMode::Fast => StreamingMode::Fast,
        CliStreamingMode::LowMemory => StreamingMode::LowMemory,
        CliStreamingMode::ExternalSort => StreamingMode::ExternalSort,
    };

    // Build TilerConfig - quiet when using progress bars
    let tiler_config = TilerConfig::new(args.min_zoom, args.max_zoom)
        .with_extent(4096)
        .with_layer_name(&layer_name)
        .with_property_filter(property_filter)
        .with_streaming_mode(streaming_mode)
        .with_quiet(args.verbose); // Suppress log output when we have progress bars

    // Print configuration in verbose mode
    if args.verbose {
        eprintln!("Configuration:");
        eprintln!("  Input: {}", args.input.display());
        eprintln!("  Output: {}", args.output.display());
        eprintln!("  Zoom: {}-{}", args.min_zoom, args.max_zoom);
        eprintln!("  Streaming mode: {:?}", args.streaming_mode);
        eprintln!("  Compression: {}", args.compression);
        eprintln!();
    }

    let total_start = Instant::now();

    // Create streaming writer
    let mut writer =
        StreamingPmtilesWriter::new(compression).context("Failed to create PMTiles writer")?;

    // Run the pipeline with or without progress bars
    let stats = if args.verbose && matches!(args.streaming_mode, CliStreamingMode::ExternalSort) {
        // Use progress callback for ExternalSort with verbose mode
        run_with_progress(&args.input, &tiler_config, &mut writer)?
    } else {
        // Standard execution
        if args.verbose {
            eprintln!("Starting tile generation...");
        }
        generate_tiles_to_writer(&args.input, &tiler_config, &mut writer)
            .context("Failed to generate tiles")?
    };

    if args.verbose {
        eprintln!();
        eprintln!("Tile generation complete:");
        eprintln!(
            "  Peak memory tracked: {} MB",
            stats.peak_bytes / (1024 * 1024)
        );
        eprintln!();
        eprintln!("Finalizing PMTiles file...");
    }

    let finalize_start = Instant::now();
    let write_stats = writer
        .finalize(&args.output)
        .context("Failed to write PMTiles file")?;
    let finalize_duration = finalize_start.elapsed();

    let total_duration = total_start.elapsed();

    // Print results
    if args.verbose {
        eprintln!();
        eprintln!("Results:");
        eprintln!("  Total tiles: {}", write_stats.total_tiles);
        eprintln!("  Unique tiles: {}", write_stats.unique_tiles);
        eprintln!(
            "  Bytes written: {} MB",
            write_stats.bytes_written / (1024 * 1024)
        );
        eprintln!(
            "  Bytes saved (dedup): {} MB",
            write_stats.bytes_saved_dedup / (1024 * 1024)
        );
        eprintln!();
        eprintln!("Timing:");
        eprintln!("  Finalize: {:.2}s", finalize_duration.as_secs_f64());
        eprintln!("  Total: {:.2}s", total_duration.as_secs_f64());
    }

    println!(
        "✓ Converted {} to {} ({} tiles in {:.1}s)",
        args.input.display(),
        args.output.display(),
        write_stats.total_tiles,
        total_duration.as_secs_f64()
    );

    Ok(())
}

/// Run tile generation with progress bars for ExternalSort mode
fn run_with_progress(
    input: &PathBuf,
    config: &TilerConfig,
    writer: &mut StreamingPmtilesWriter,
) -> Result<gpq_tiles_core::memory::MemoryStats> {
    // Shared state for progress bar
    let progress_bar: Arc<Mutex<Option<ProgressBar>>> = Arc::new(Mutex::new(None));
    let pb_clone = Arc::clone(&progress_bar);

    // Track totals for Phase 3
    let total_records: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    let total_records_clone = Arc::clone(&total_records);

    let progress_callback = Box::new(move |event: ProgressEvent| {
        let mut pb_guard = pb_clone.lock().unwrap();

        match event {
            ProgressEvent::PhaseStart { phase, name } => {
                // Finish previous progress bar if any
                if let Some(ref pb) = *pb_guard {
                    pb.finish_and_clear();
                }

                if phase == 1 {
                    // Phase 1: Reading row groups - indeterminate at first
                    let pb = ProgressBar::new_spinner();
                    pb.set_style(
                        ProgressStyle::default_spinner()
                            .template("{spinner:.green} Phase 1: {msg}")
                            .unwrap(),
                    );
                    pb.set_message(format!("{} - starting...", name));
                    *pb_guard = Some(pb);
                } else if phase == 3 {
                    // Phase 3: Encoding - we know total from Phase 1
                    let total = *total_records_clone.lock().unwrap();
                    let pb = ProgressBar::new(total);
                    pb.set_style(
                        ProgressStyle::default_bar()
                            .template("{spinner:.green} Phase 3: [{bar:40.cyan/blue}] {pos}/{len} records ({percent}%) | {msg}")
                            .unwrap()
                            .progress_chars("█▉▊▋▌▍▎▏  "),
                    );
                    pb.set_message("Encoding tiles");
                    *pb_guard = Some(pb);
                }
            }

            ProgressEvent::Phase1Progress {
                row_group,
                total_row_groups,
                features_in_group,
                records_written,
            } => {
                if let Some(ref pb) = *pb_guard {
                    pb.set_message(format!(
                        "Row group {}/{} ({} features) | {} records written",
                        row_group + 1,
                        total_row_groups,
                        features_in_group,
                        records_written
                    ));
                }
            }

            ProgressEvent::Phase1Complete {
                total_records: total,
                peak_memory_bytes,
            } => {
                // Store total for Phase 3
                *total_records.lock().unwrap() = total;

                if let Some(ref pb) = *pb_guard {
                    pb.finish_with_message(format!(
                        "Complete: {} records, peak mem {} MB",
                        total,
                        peak_memory_bytes / (1024 * 1024)
                    ));
                }
                *pb_guard = None;
            }

            ProgressEvent::Phase2Start => {
                let pb = ProgressBar::new_spinner();
                pb.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner:.green} Phase 2: {msg}")
                        .unwrap(),
                );
                pb.set_message("External merge sort by tile_id...");
                pb.enable_steady_tick(std::time::Duration::from_millis(100));
                *pb_guard = Some(pb);
            }

            ProgressEvent::Phase2Complete => {
                if let Some(ref pb) = *pb_guard {
                    pb.finish_with_message("Sort complete");
                }
                *pb_guard = None;
            }

            ProgressEvent::Phase3Progress {
                tiles_written,
                records_processed,
                total_records: _,
            } => {
                if let Some(ref pb) = *pb_guard {
                    pb.set_position(records_processed);
                    pb.set_message(format!("{} tiles written", tiles_written));
                }
            }

            ProgressEvent::Complete {
                total_tiles,
                peak_memory_bytes,
                duration_secs,
            } => {
                if let Some(ref pb) = *pb_guard {
                    pb.finish_and_clear();
                }
                eprintln!(
                    "✓ Generated {} tiles in {:.1}s (peak memory: {} MB)",
                    total_tiles,
                    duration_secs,
                    peak_memory_bytes / (1024 * 1024)
                );
            }
        }
    });

    generate_tiles_to_writer_with_progress(input, config, writer, progress_callback)
        .context("Failed to generate tiles")
}
