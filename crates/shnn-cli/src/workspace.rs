//! Workspace management utilities

use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

use crate::error::{CliError, CliResult};

/// hSNN workspace configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkspaceConfig {
    pub workspace: WorkspaceInfo,
    pub defaults: DefaultParams,
    pub lif: LIFConfig,
    pub stdp: STDPConfig,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkspaceInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DefaultParams {
    pub neuron_type: String,
    pub plasticity_type: String,
    pub dt_us: u64,
    pub seed: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LIFConfig {
    pub tau_m: f32,
    pub tau_ref: f32,
    pub v_rest: f32,
    pub v_reset: f32,
    pub v_thresh: f32,
    pub r_m: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct STDPConfig {
    pub a_plus: f32,
    pub a_minus: f32,
    pub tau_plus: f32,
    pub tau_minus: f32,
    pub w_min: f32,
    pub w_max: f32,
}

/// Workspace management utilities
pub struct Workspace {
    pub root: PathBuf,
    pub config: Option<WorkspaceConfig>,
}

impl Workspace {
    /// Create a new workspace instance
    pub fn new(root: PathBuf) -> Self {
        Self { root, config: None }
    }
    
    /// Load workspace configuration
    pub fn load_config(&mut self) -> CliResult<()> {
        let config_path = self.root.join("hSNN.toml");
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            self.config = Some(toml::from_str(&content)?);
        }
        Ok(())
    }
    
    /// Check if this is a valid hSNN workspace
    pub fn is_valid(&self) -> bool {
        self.root.join("hSNN.toml").exists()
    }
    
    /// Get the experiments directory
    pub fn experiments_dir(&self) -> PathBuf {
        self.root.join("experiments")
    }
    
    /// Get the data directory
    pub fn data_dir(&self) -> PathBuf {
        self.root.join("data")
    }
    
    /// Get the results directory
    pub fn results_dir(&self) -> PathBuf {
        self.root.join("results")
    }
    
    /// Get the configs directory
    pub fn configs_dir(&self) -> PathBuf {
        self.root.join("configs")
    }
    
    /// Find workspace root by walking up the directory tree
    pub fn find_workspace_root(start: &Path) -> Option<PathBuf> {
        let mut current = start;
        loop {
            if current.join("hSNN.toml").exists() {
                return Some(current.to_path_buf());
            }
            
            match current.parent() {
                Some(parent) => current = parent,
                None => return None,
            }
        }
    }
    
    /// Ensure workspace directories exist
    pub fn ensure_directories(&self) -> CliResult<()> {
        let dirs = [
            self.experiments_dir(),
            self.data_dir(),
            self.results_dir(),
            self.configs_dir(),
        ];
        
        for dir in &dirs {
            if !dir.exists() {
                std::fs::create_dir_all(dir)
                    .map_err(|e| CliError::workspace(format!("Failed to create directory {}: {}", dir.display(), e)))?;
            }
        }
        
        Ok(())
    }
}