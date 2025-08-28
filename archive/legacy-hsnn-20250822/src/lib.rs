//! # H-SNN: Hypergraph Spiking Neural Networks
//!
//! A lightweight implementation of Hypergraph Spiking Neural Networks that addresses
//! fundamental limitations of traditional SNNs through hypergraph-based connectivity
//! and group-level learning mechanisms.
//!
//! ## Architecture Overview
//!
//! H-SNN reconceptualizes neural computation using:
//! - **Hypergraphs** for multi-way neural connections
//! - **Spike walks** along hyperpaths for inference  
//! - **Group-level learning** to bypass spike non-differentiability
//! - **Non-local credit assignment** along structured pathways
//!
//! ## Quick Start
//!
//! ```rust
//! use hsnn::prelude::*;
//!
//! // Create H-SNN network
//! let mut network = HSNNBuilder::new()
//!     .neurons(100)
//!     .hyperedges(200)
//!     .build()?;
//!
//! // Add hyperedge for group processing
//! let group_edge = Hyperedge::many_to_many(
//!     HyperedgeId::new(0),
//!     vec![NeuronId::new(0), NeuronId::new(1)],
//!     vec![NeuronId::new(10), NeuronId::new(11)],
//! )?;
//!
//! network.add_hyperedge(group_edge)?;
//!
//! // Process spike and trigger spike walk
//! let spike = Spike::new(NeuronId::new(0), Time::from_millis(10), 1.0)?;
//! let walks = network.process_spike_walk(spike)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::type_complexity)]

// Platform-specific imports
#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate core as std;

// Core modules - adapted from SHNN with H-SNN extensions
pub mod core;
pub mod hypergraph;
pub mod learning;
pub mod inference;
pub mod network;
pub mod utils;

// Prelude module for convenient imports
pub mod prelude;

// Re-export the most commonly used types for convenience
pub use crate::{
    core::{
        neuron::{Neuron, NeuronId, NeuronType},
        spike::{Spike, SpikeWalk, SpikeWalkId},
        time::{Time, Duration},
    },
    hypergraph::{
        Hyperedge, HyperedgeId, HypergraphNetwork,
        hyperpath::{Hyperpath, HyperpathId},
        activation::{ActivationRule, ActivationState},
    },
    learning::{
        group_plasticity::{GroupSTDP, GroupPlasticityRule},
        credit_assignment::{CreditAssignment, CausalTrace},
    },
    inference::{
        spike_walk::{SpikeWalkEngine, WalkContext},
        engine::InferenceEngine,
    },
    network::{
        hsnn_network::HSNNNetwork,
        builder::HSNNBuilder,
    },
    utils::error::{HSNNError, Result},
};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build information
pub const BUILD_INFO: BuildInfo = BuildInfo {
    version: VERSION,
    git_hash: option_env!("GIT_HASH"),
    build_timestamp: option_env!("BUILD_TIMESTAMP").unwrap_or("unknown"),
    features: &[
        #[cfg(feature = "std")]
        "std",
        #[cfg(feature = "no-std")]
        "no-std",
        #[cfg(feature = "serde")]
        "serde",
        #[cfg(feature = "simd")]
        "simd",
        #[cfg(feature = "parallel")]
        "parallel",
        #[cfg(feature = "optimized")]
        "optimized",
    ],
};

/// Build information structure
#[derive(Debug, Clone)]
pub struct BuildInfo {
    /// Version string
    pub version: &'static str,
    /// Git commit hash
    pub git_hash: Option<&'static str>,
    /// Build timestamp
    pub build_timestamp: &'static str,
    /// Enabled features
    pub features: &'static [&'static str],
}

impl BuildInfo {
    /// Get a formatted build string
    pub fn build_string(&self) -> String {
        format!(
            "H-SNN v{} ({}), built on {} with features: [{}]",
            self.version,
            self.git_hash.unwrap_or("unknown"),
            self.build_timestamp,
            self.features.join(", ")
        )
    }
}

/// Initialize the H-SNN library
///
/// This function should be called once at the beginning of your application.
/// It performs any necessary global initialization.
pub fn init() -> Result<()> {
    #[cfg(feature = "std")]
    {
        // Initialize logging if available
        if std::env::var("RUST_LOG").is_err() {
            std::env::set_var("RUST_LOG", "info");
        }
        
        // Print build info in debug mode
        #[cfg(debug_assertions)]
        {
            println!("{}", BUILD_INFO.build_string());
        }
    }
    
    Ok(())
}

/// Configuration for library initialization
#[derive(Debug, Default)]
pub struct InitConfig {
    /// Whether to print build information
    pub print_build_info: bool,
    /// Custom log level to set
    pub log_level: Option<&'static str>,
}

/// Initialize H-SNN with custom configuration
pub fn init_with_config(config: InitConfig) -> Result<()> {
    #[cfg(feature = "std")]
    {
        if let Some(log_level) = config.log_level {
            std::env::set_var("RUST_LOG", log_level);
        }
        
        if config.print_build_info {
            println!("{}", BUILD_INFO.build_string());
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_build_info() {
        let build_string = BUILD_INFO.build_string();
        assert!(build_string.contains("H-SNN"));
        assert!(build_string.contains(VERSION));
    }
    
    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }
    
    #[test]
    fn test_init_with_config() {
        let config = InitConfig {
            print_build_info: false,
            log_level: Some("debug"),
        };
        assert!(init_with_config(config).is_ok());
    }
}