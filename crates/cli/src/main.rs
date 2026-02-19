//! CLI for gpq-tiles - Convert GeoParquet to PMTiles
//!
//! This is a thin wrapper around the gpq-tiles-core library.

use anyhow::{Context, Result};
use clap::Parser;
use gpq_tiles_core::{Config, Converter, DropDensity};
use std::path::PathBuf;

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

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

impl Args {
    fn parse_drop_density(&self) -> Result<DropDensity> {
        match self.drop_density.to_lowercase().as_str() {
            "low" => Ok(DropDensity::Low),
            "medium" => Ok(DropDensity::Medium),
            "high" => Ok(DropDensity::High),
            _ => anyhow::bail!("Invalid drop density: {}", self.drop_density),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose {
        "debug"
    } else {
        "info"
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .init();

    // Build configuration
    let config = Config {
        min_zoom: args.min_zoom,
        max_zoom: args.max_zoom,
        extent: 4096,
        drop_density: args
            .parse_drop_density()
            .context("Failed to parse drop density")?,
    };

    // Create converter and run
    let converter = Converter::new(config);

    converter
        .convert(&args.input, &args.output)
        .context("Failed to convert GeoParquet to PMTiles")?;

    println!("✓ Converted {} to {}", args.input.display(), args.output.display());

    Ok(())
}
