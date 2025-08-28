//! # hSNN CLI - Command Line Interface for Neuromorphic Research
//!
//! The CLI-first interface to the hSNN neuromorphic simulation platform.
//! Provides easy, reproducible, and flexible access to SNN training,
//! hypergraph manipulation, visualization, and experiment management.

use clap::Parser;
use tracing::{error, info};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

mod commands;
mod config;
mod error;
mod workspace;

use commands::HsnnCli;
use error::CliResult;

#[tokio::main]
async fn main() -> CliResult<()> {
    // Initialize logging with environment variable support
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));
    
    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(filter)
        .init();

    // Parse CLI arguments
    let cli = HsnnCli::parse();
    
    // Execute the command
    if let Err(err) = cli.execute().await {
        error!("Command failed: {}", err);
        std::process::exit(1);
    }
    
    Ok(())
}