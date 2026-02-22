//! CLI for gpq-tiles - Convert GeoParquet to PMTiles
//!
//! This is a thin wrapper around the gpq-tiles-core library.

use anyhow::{Context, Result};
use clap::Parser;
use gpq_tiles_core::{Config, Converter, DropDensity, PropertyFilter};
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
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();

    // Parse property filter first (before moving args fields)
    let property_filter = args
        .parse_property_filter()
        .context("Failed to parse property filter")?;

    // Build configuration
    let config = Config {
        min_zoom: args.min_zoom,
        max_zoom: args.max_zoom,
        extent: 4096,
        drop_density: args
            .parse_drop_density()
            .context("Failed to parse drop density")?,
        layer_name: args.layer_name,
        property_filter,
    };

    // Create converter and run
    let converter = Converter::new(config);

    converter
        .convert(&args.input, &args.output)
        .context("Failed to convert GeoParquet to PMTiles")?;

    println!(
        "✓ Converted {} to {}",
        args.input.display(),
        args.output.display()
    );

    Ok(())
}
