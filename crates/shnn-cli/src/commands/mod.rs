//! CLI command implementations for hSNN

use clap::{Parser, Subcommand};
use crate::error::CliResult;

pub mod init;
pub mod train;
pub mod study;
pub mod viz;
pub mod ttr;
pub mod hg;
pub mod inspect;
pub mod nir;
pub mod snapshot;

/// hSNN - CLI-first neuromorphic research substrate
#[derive(Parser, Debug)]
#[command(
    name = "snn",
    version,
    about = "CLI-first neuromorphic research substrate",
    long_about = "hSNN provides easy, reproducible, and flexible neuromorphic simulation \
                  through a powerful command-line interface. Train SNNs, manage hypergraph \
                  databases, run parameter studies, and visualize results."
)]
pub struct HsnnCli {
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    pub verbose: bool,
    
    /// Workspace directory (defaults to current directory)
    #[arg(short, long, global = true)]
    pub workspace: Option<std::path::PathBuf>,
    
    /// Configuration file path
    #[arg(short, long, global = true)]
    pub config: Option<std::path::PathBuf>,
    
    #[command(subcommand)]
    pub command: Commands,
}

/// Available CLI commands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Initialize a new hSNN workspace
    #[command(alias = "new")]
    Init(init::InitCommand),
    
    /// Train spiking neural networks
    Train(train::TrainCommand),
    
    /// Run parameter studies and optimization
    Study(study::StudyCommand),
    
    /// Visualization and analysis tools
    #[command(alias = "vis")]
    Viz(viz::VizCommand),

    /// Task-Aware Topology Reshaping (TTR)
    Ttr(ttr::TtrCommand),
    
    /// Hypergraph database operations
    Hg(hg::HgCommand),
    
    /// Inspect workspace and data
    Inspect(inspect::InspectCommand),

    /// NIR-related commands (compile, run, op list)
    Nir(nir::NirCommand),

    /// Snapshot export/import of weights
    Snapshot(snapshot::SnapshotCommand),
}

impl HsnnCli {
    /// Execute the CLI command
    pub async fn execute(self) -> CliResult<()> {
        // Set up workspace and config
        let workspace = self.workspace.unwrap_or_else(|| std::env::current_dir().unwrap());
        let config = self.config;
        
        // Execute the appropriate subcommand
        match self.command {
            Commands::Init(cmd) => cmd.execute(workspace, config).await,
            Commands::Train(cmd) => cmd.execute(workspace, config).await,
            Commands::Study(cmd) => cmd.execute(workspace, config).await,
            Commands::Viz(cmd) => cmd.execute(workspace, config).await,
            Commands::Ttr(cmd) => cmd.execute(workspace, config).await,
            Commands::Hg(cmd) => cmd.execute(workspace, config).await,
            Commands::Inspect(cmd) => cmd.execute(workspace, config).await,
            Commands::Nir(cmd) => cmd.execute().await,
            Commands::Snapshot(cmd) => cmd.execute(workspace, config).await,
        }
    }
}