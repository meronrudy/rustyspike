//! Error handling for H-SNN
//!
//! This module provides comprehensive error handling adapted from SHNN
//! with H-SNN specific error types.

use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Result type for H-SNN operations
pub type Result<T> = core::result::Result<T, HSNNError>;

/// Main error type for H-SNN operations
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HSNNError {
    /// Invalid spike data
    InvalidSpike(String),
    /// Invalid neuron configuration
    InvalidNeuron(String),
    /// Hypergraph structure error
    HypergraphError(String),
    /// Hyperpath traversal error
    HyperpathError(String),
    /// Spike walk error
    SpikeWalkError(String),
    /// Learning mechanism error
    LearningError(String),
    /// Credit assignment error
    CreditAssignmentError(String),
    /// Time-related error
    TimeError(String),
    /// Network configuration error
    NetworkError(String),
    /// Memory allocation error
    MemoryError(String),
    /// Serialization error
    #[cfg(feature = "serde")]
    SerializationError(String),
    /// Mathematical computation error
    MathError(String),
    /// Resource exhaustion
    ResourceExhausted(String),
    /// Invalid operation
    InvalidOperation(String),
    /// Generic error with message
    Other(String),
}

impl HSNNError {
    /// Create a spike-related error
    pub fn invalid_spike<S: Into<String>>(msg: S) -> Self {
        Self::InvalidSpike(msg.into())
    }
    
    /// Create a neuron-related error
    pub fn invalid_neuron<S: Into<String>>(msg: S) -> Self {
        Self::InvalidNeuron(msg.into())
    }
    
    /// Create a hypergraph-related error
    pub fn hypergraph_error<S: Into<String>>(msg: S) -> Self {
        Self::HypergraphError(msg.into())
    }
    
    /// Create a hyperpath-related error
    pub fn hyperpath_error<S: Into<String>>(msg: S) -> Self {
        Self::HyperpathError(msg.into())
    }
    
    /// Create a spike walk error
    pub fn spike_walk_error<S: Into<String>>(msg: S) -> Self {
        Self::SpikeWalkError(msg.into())
    }
    
    /// Create a learning-related error
    pub fn learning_error<S: Into<String>>(msg: S) -> Self {
        Self::LearningError(msg.into())
    }
    
    /// Create a credit assignment error
    pub fn credit_assignment_error<S: Into<String>>(msg: S) -> Self {
        Self::CreditAssignmentError(msg.into())
    }
    
    /// Create a time-related error
    pub fn time_error<S: Into<String>>(msg: S) -> Self {
        Self::TimeError(msg.into())
    }
    
    /// Create a network-related error
    pub fn network_error<S: Into<String>>(msg: S) -> Self {
        Self::NetworkError(msg.into())
    }
    
    /// Create a memory-related error
    pub fn memory_error<S: Into<String>>(msg: S) -> Self {
        Self::MemoryError(msg.into())
    }
    
    /// Create a math-related error
    pub fn math_error<S: Into<String>>(msg: S) -> Self {
        Self::MathError(msg.into())
    }
    
    /// Create a resource exhaustion error
    pub fn resource_exhausted<S: Into<String>>(msg: S) -> Self {
        Self::ResourceExhausted(msg.into())
    }
    
    /// Create an invalid operation error
    pub fn invalid_operation<S: Into<String>>(msg: S) -> Self {
        Self::InvalidOperation(msg.into())
    }
    
    /// Get the error category
    pub fn category(&self) -> &'static str {
        match self {
            Self::InvalidSpike(_) => "spike",
            Self::InvalidNeuron(_) => "neuron",
            Self::HypergraphError(_) => "hypergraph",
            Self::HyperpathError(_) => "hyperpath",
            Self::SpikeWalkError(_) => "spike_walk",
            Self::LearningError(_) => "learning",
            Self::CreditAssignmentError(_) => "credit_assignment",
            Self::TimeError(_) => "time",
            Self::NetworkError(_) => "network",
            Self::MemoryError(_) => "memory",
            #[cfg(feature = "serde")]
            Self::SerializationError(_) => "serialization",
            Self::MathError(_) => "math",
            Self::ResourceExhausted(_) => "resource",
            Self::InvalidOperation(_) => "operation",
            Self::Other(_) => "other",
        }
    }
    
    /// Get the error message
    pub fn message(&self) -> &str {
        match self {
            Self::InvalidSpike(msg) => msg,
            Self::InvalidNeuron(msg) => msg,
            Self::HypergraphError(msg) => msg,
            Self::HyperpathError(msg) => msg,
            Self::SpikeWalkError(msg) => msg,
            Self::LearningError(msg) => msg,
            Self::CreditAssignmentError(msg) => msg,
            Self::TimeError(msg) => msg,
            Self::NetworkError(msg) => msg,
            Self::MemoryError(msg) => msg,
            #[cfg(feature = "serde")]
            Self::SerializationError(msg) => msg,
            Self::MathError(msg) => msg,
            Self::ResourceExhausted(msg) => msg,
            Self::InvalidOperation(msg) => msg,
            Self::Other(msg) => msg,
        }
    }
    
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::InvalidSpike(_) | Self::InvalidNeuron(_) => false,
            Self::HypergraphError(_) | Self::HyperpathError(_) => false,
            Self::SpikeWalkError(_) => true, // Walk can be retried
            Self::LearningError(_) => true, // Learning can continue
            Self::CreditAssignmentError(_) => true,
            Self::TimeError(_) => false,
            Self::NetworkError(_) => false,
            Self::MemoryError(_) => false, // Usually fatal
            #[cfg(feature = "serde")]
            Self::SerializationError(_) => false,
            Self::MathError(_) => false,
            Self::ResourceExhausted(_) => true, // Can retry later
            Self::InvalidOperation(_) => false,
            Self::Other(_) => false,
        }
    }
}

impl fmt::Display for HSNNError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.category(), self.message())
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HSNNError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

/// Error context for debugging
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ErrorContext {
    /// Source location (file:line)
    pub location: Option<String>,
    /// Additional context information
    pub context: Option<String>,
    /// Timestamp when error occurred
    #[cfg(feature = "std")]
    pub timestamp: Option<std::time::SystemTime>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new() -> Self {
        Self {
            location: None,
            context: None,
            #[cfg(feature = "std")]
            timestamp: Some(std::time::SystemTime::now()),
        }
    }
    
    /// Add location information
    pub fn with_location(mut self, location: String) -> Self {
        self.location = Some(location);
        self
    }
    
    /// Add context information
    pub fn with_context(mut self, context: String) -> Self {
        self.context = Some(context);
        self
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced error with context
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ContextualError {
    /// The base error
    pub error: HSNNError,
    /// Additional context
    pub context: ErrorContext,
}

impl ContextualError {
    /// Create a new contextual error
    pub fn new(error: HSNNError, context: ErrorContext) -> Self {
        Self { error, context }
    }
    
    /// Create from base error with automatic context
    pub fn from_error(error: HSNNError) -> Self {
        Self {
            error,
            context: ErrorContext::new(),
        }
    }
}

impl fmt::Display for ContextualError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)?;
        
        if let Some(ref location) = self.context.location {
            write!(f, " at {}", location)?;
        }
        
        if let Some(ref context) = self.context.context {
            write!(f, " ({})", context)?;
        }
        
        Ok(())
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ContextualError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Macro for creating errors with location information
#[macro_export]
macro_rules! hsnn_error {
    ($kind:expr, $msg:expr) => {
        HSNNError::$kind($msg.to_string())
    };
    ($kind:expr, $fmt:expr, $($arg:tt)*) => {
        HSNNError::$kind(format!($fmt, $($arg)*))
    };
}

/// Macro for creating contextual errors with location
#[macro_export]
macro_rules! contextual_error {
    ($error:expr) => {
        ContextualError::new(
            $error,
            ErrorContext::new().with_location(format!("{}:{}", file!(), line!()))
        )
    };
    ($error:expr, $context:expr) => {
        ContextualError::new(
            $error,
            ErrorContext::new()
                .with_location(format!("{}:{}", file!(), line!()))
                .with_context($context.to_string())
        )
    };
}

/// Ensure macro for validation
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $error:expr) => {
        if !($cond) {
            return Err($error);
        }
    };
    ($cond:expr, $kind:ident, $msg:expr) => {
        if !($cond) {
            return Err(HSNNError::$kind($msg.to_string()));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let error = HSNNError::invalid_spike("test spike error");
        assert_eq!(error.category(), "spike");
        assert_eq!(error.message(), "test spike error");
        assert!(!error.is_recoverable());
    }
    
    #[test]
    fn test_error_display() {
        let error = HSNNError::hypergraph_error("invalid structure");
        let error_string = format!("{}", error);
        assert_eq!(error_string, "hypergraph: invalid structure");
    }
    
    #[test]
    fn test_contextual_error() {
        let base_error = HSNNError::spike_walk_error("walk failed");
        let context = ErrorContext::new()
            .with_location("test.rs:123".to_string())
            .with_context("during spike processing".to_string());
        
        let contextual = ContextualError::new(base_error, context);
        let error_string = format!("{}", contextual);
        assert!(error_string.contains("spike_walk: walk failed"));
        assert!(error_string.contains("test.rs:123"));
        assert!(error_string.contains("during spike processing"));
    }
    
    #[test]
    fn test_recoverable_errors() {
        assert!(!HSNNError::invalid_spike("test").is_recoverable());
        assert!(HSNNError::spike_walk_error("test").is_recoverable());
        assert!(HSNNError::learning_error("test").is_recoverable());
        assert!(HSNNError::resource_exhausted("test").is_recoverable());
    }
    
    #[test]
    fn test_error_macros() {
        let error = hsnn_error!(HypergraphError, "test {}", "message");
        match error {
            HSNNError::HypergraphError(msg) => assert_eq!(msg, "test message"),
            _ => panic!("Wrong error type"),
        }
    }
}