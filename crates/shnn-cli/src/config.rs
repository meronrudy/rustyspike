//! Configuration management for hSNN CLI

use std::path::Path;
use serde::{Deserialize, Serialize};

use crate::error::{CliError, CliResult};

/// Global CLI configuration
#[derive(Debug, Serialize, Deserialize)]
pub struct CliConfig {
    /// Default workspace directory
    pub default_workspace: Option<String>,
    
    /// Default logging level
    pub log_level: Option<String>,
    
    /// Default number of parallel jobs
    pub default_jobs: Option<u32>,
    
    /// User preferences
    pub preferences: UserPreferences,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Show progress bars
    pub show_progress: bool,
    
    /// Use colors in output
    pub use_colors: bool,
    
    /// Default output format
    pub output_format: String,
    
    /// Auto-save results
    pub auto_save: bool,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            default_workspace: None,
            log_level: Some("info".to_string()),
            default_jobs: Some(1),
            preferences: UserPreferences {
                show_progress: true,
                use_colors: true,
                output_format: "json".to_string(),
                auto_save: true,
            },
        }
    }
}

impl CliConfig {
    /// Load configuration from file
    pub fn load_from_file(path: &Path) -> CliResult<Self> {
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            toml::from_str(&content).map_err(|e| CliError::config(format!("Invalid config file: {}", e)))
        } else {
            Ok(Self::default())
        }
    }
    
    /// Save configuration to file
    pub fn save_to_file(&self, path: &Path) -> CliResult<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| CliError::config(format!("Failed to serialize config: {}", e)))?;
        
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Get the default configuration file path
    pub fn default_config_path() -> CliResult<std::path::PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| CliError::config("Could not determine config directory"))?;
        Ok(config_dir.join("hsnn").join("config.toml"))
    }
}