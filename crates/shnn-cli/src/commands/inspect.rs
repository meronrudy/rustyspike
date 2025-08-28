//! Workspace and data inspection commands

use clap::Args;
use std::path::PathBuf;
use tracing::info;

use crate::error::CliResult;

/// Inspect workspace and data
#[derive(Args, Debug)]
pub struct InspectCommand {
    /// What to inspect
    #[arg(default_value = "workspace")]
    pub target: String,
    
    /// Show detailed information
    #[arg(short, long)]
    pub detailed: bool,
    
    /// Show statistics
    #[arg(long)]
    pub stats: bool,
    
    /// Verify data integrity
    #[arg(long)]
    pub verify: bool,
}

impl InspectCommand {
    pub async fn execute(
        self,
        workspace: PathBuf,
        _config: Option<PathBuf>,
    ) -> CliResult<()> {
        info!("Inspecting {}", self.target);
        
        match self.target.as_str() {
            "workspace" => {
                self.inspect_workspace(workspace).await?;
            }
            "data" => {
                info!("Data inspection functionality coming soon");
            }
            "network" => {
                info!("Network inspection functionality coming soon");
            }
            _ => {
                info!("Unknown inspection target: {}", self.target);
            }
        }
        
        Ok(())
    }
    
    async fn inspect_workspace(&self, workspace: PathBuf) -> CliResult<()> {
        info!("Workspace: {}", workspace.display());
        
        // Check for hSNN.toml
        let config_path = workspace.join("hSNN.toml");
        if config_path.exists() {
            info!("✓ Configuration file found");
            if self.detailed {
                let config_content = std::fs::read_to_string(&config_path)?;
                println!("Configuration:\n{}", config_content);
            }
        } else {
            info!("✗ No hSNN.toml configuration file found");
            info!("  Run 'snn init <workspace_name>' to initialize");
        }
        
        // Check directory structure
        let dirs = ["experiments", "data", "results", "configs"];
        for dir in &dirs {
            let dir_path = workspace.join(dir);
            if dir_path.exists() {
                let file_count = std::fs::read_dir(&dir_path)?.count();
                info!("✓ {}: {} items", dir, file_count);
            } else {
                info!("✗ Missing directory: {}", dir);
            }
        }
        
        if self.stats {
            // Show workspace statistics
            info!("\nWorkspace Statistics:");
            if let Ok(entries) = std::fs::read_dir(&workspace) {
                let total_files = entries.count();
                info!("Total files/directories: {}", total_files);
            }
        }
        
        Ok(())
    }
}