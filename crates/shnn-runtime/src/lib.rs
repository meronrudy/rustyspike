//! SNN runtime engine for the CLI-first neuromorphic research substrate
//!
//! This crate provides the core simulation engine for spiking neural networks,
//! designed specifically for CLI-driven research workflows with emphasis on
//! reproducibility and performance.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![warn(clippy::all)]

// Re-export essential types from storage
pub use shnn_storage::{
    NeuronId, HyperedgeId, Time, Spike,
    GenerationId, HypergraphStore, EventStore,
    Result as StorageResult, StorageError,
};

// Core modules
pub mod error;
pub mod neuron;
pub mod plasticity;
pub mod network;
pub mod simulation;

// Re-export essential types
pub use error::{RuntimeError, Result};
pub use neuron::{LIFNeuron, LIFParams, NeuronState};
pub use plasticity::{STDPRule, STDPParams, PlasticityRule};
pub use network::{SNNNetwork, NetworkBuilder, NetworkConfig};
pub use simulation::{SimulationEngine, SimulationParams, SimulationResult};

/// Runtime crate version for compatibility checking
pub const RUNTIME_VERSION: u32 = 1;

/// Default simulation time step (1 millisecond in nanoseconds)
pub const DEFAULT_TIMESTEP_NS: u64 = 1_000_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_integration() {
        // Test that all components can be imported and basic objects created
        let params = LIFParams::default();
        assert!(params.tau_m > 0.0);
        
        let stdp_params = STDPParams::default();
        assert!(stdp_params.a_plus > 0.0);
        
        let sim_params = SimulationParams::default();
        assert!(sim_params.dt_ns > 0);
    }
}