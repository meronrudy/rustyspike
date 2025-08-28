//! Error handling for the hSNN CLI

use thiserror::Error;

/// Result type for CLI operations
pub type CliResult<T> = Result<T, CliError>;

/// CLI-specific errors
#[derive(Error, Debug)]
pub enum CliError {
    /// Storage layer error
    #[error("Storage error: {0}")]
    Storage(#[from] shnn_storage::error::StorageError),
    
    /// Runtime layer error  
    #[error("Runtime error: {0}")]
    Runtime(#[from] shnn_runtime::error::RuntimeError),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// Workspace error
    #[error("Workspace error: {0}")]
    Workspace(String),
    
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serde(#[from] toml::de::Error),
    
    /// Generic error
    #[error("Error: {0}")]
    Generic(#[from] anyhow::Error),
    
    /// User interrupted operation
    #[error("Operation cancelled by user")]
    Cancelled,
    
    /// Invalid command arguments
    #[error("Invalid arguments: {0}")]
    InvalidArgs(String),
    
    /// Missing required file or resource
    #[error("Missing resource: {0}")]
    MissingResource(String),
}

impl CliError {
    /// Create a configuration error
    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }
    
    /// Create a workspace error
    pub fn workspace(msg: impl Into<String>) -> Self {
        Self::Workspace(msg.into())
    }
    
    /// Create an invalid arguments error
    pub fn invalid_args(msg: impl Into<String>) -> Self {
        Self::InvalidArgs(msg.into())
    }
    
    /// Create a missing resource error
    pub fn missing_resource(msg: impl Into<String>) -> Self {
        Self::MissingResource(msg.into())
    }
}