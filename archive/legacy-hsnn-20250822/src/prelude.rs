//! H-SNN Prelude
//!
//! This module provides convenient access to the most commonly used types
//! and functions in H-SNN for easy importing.

// Re-export core types
pub use crate::core::{
    spike::{Spike, SpikeWalk, SpikeWalkId, NeuronId, TimedSpike, SpikeTrain, WalkContext},
    time::{Time, Duration, TimeStep, TimeUtils},
    neuron::{Neuron, NeuronType, LIFNeuron, NeuronPool},
    encoding::{SpikeEncoder, EncodingType, RateEncoder},
};

// Re-export hypergraph types
pub use crate::hypergraph::{
    Hyperedge, HyperedgeId, HypergraphNetwork,
    hyperpath::{Hyperpath, HyperpathId},
    activation::{ActivationRule, ActivationState, ActivationEngine},
    routing::SpikeRoute,
};

// Re-export learning types
pub use crate::learning::{
    group_plasticity::{GroupSTDP, GroupPlasticityRule},
    credit_assignment::{CreditAssignment, CausalTrace, CreditAssignmentEngine},
    stdp::STDPRule,
};

// Re-export inference types
pub use crate::inference::{
    spike_walk::{SpikeWalkEngine, WalkStepResult},
    engine::InferenceEngine,
    temporal::{TemporalHypergraph, TemporalLayer},
    motifs::{HyperMotif, ConceptualEngine, ConceptId},
};

// Re-export network types
pub use crate::network::{
    hsnn_network::HSNNNetwork,
    builder::{HSNNBuilder, NetworkConfig},
    simulation::SimulationEngine,
};

// Re-export utility types
pub use crate::utils::{
    error::{HSNNError, Result, ContextualError},
    math::MathUtils,
    memory::MemoryPool,
    metrics::{NetworkMetrics, PerformanceCounter},
};

// Re-export main library functions
pub use crate::{init, init_with_config, InitConfig, VERSION, BUILD_INFO};

// Type aliases for convenience
pub type HSNNResult<T> = Result<T>;
pub type BasicNetwork = HSNNNetwork;

// Builder convenience function
/// Create a new H-SNN network builder
pub fn network() -> HSNNBuilder {
    HSNNBuilder::new()
}

// Common activation rules for convenience
pub mod activation_rules {
    use super::*;
    
    /// Simple threshold activation rule
    pub fn simple_threshold(min_spikes: u32, time_window_ms: u64) -> ActivationRule {
        ActivationRule::SimpleThreshold {
            min_spikes,
            time_window: Duration::from_millis(time_window_ms),
        }
    }
    
    /// Weighted sum activation rule
    pub fn weighted_sum(threshold: f32, decay_constant: f32) -> ActivationRule {
        ActivationRule::WeightedSum {
            threshold,
            decay_constant,
        }
    }
    
    /// Factory activation rule (all inputs required)
    pub fn factory_rule(input_neurons: Vec<NeuronId>, causal_window_ms: u64) -> ActivationRule {
        ActivationRule::FactoryRule {
            input_subset: input_neurons,
            require_all: true,
            causal_window: Duration::from_millis(causal_window_ms),
        }
    }
}

// Common encoding functions
pub mod encoding {
    use super::*;
    
    /// Create Poisson spike train
    pub fn poisson_spikes(
        neuron_id: NeuronId,
        rate_hz: f32,
        duration: Duration,
        start_time: Time,
    ) -> Result<SpikeTrain> {
        let encoder = RateEncoder::new(rate_hz);
        encoder.encode_duration(neuron_id, duration, start_time)
    }
    
    /// Create regular spike train
    pub fn regular_spikes(
        neuron_id: NeuronId,
        interval: Duration,
        count: usize,
        start_time: Time,
    ) -> Result<SpikeTrain> {
        let mut train = SpikeTrain::new(neuron_id);
        for i in 0..count {
            let spike_time = start_time + interval * (i as u64);
            train.add_spike(spike_time, None);
        }
        Ok(train)
    }
}

// Common network creation patterns
pub mod patterns {
    use super::*;
    
    /// Create a simple feedforward H-SNN
    pub fn feedforward_hsnn(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
    ) -> Result<HSNNNetwork> {
        HSNNBuilder::new()
            .neurons(input_size + hidden_size + output_size)
            .hyperedges(input_size * hidden_size + hidden_size * output_size)
            .activation_rule(activation_rules::simple_threshold(2, 5))
            .build()
    }
    
    /// Create a recurrent H-SNN with temporal processing
    pub fn recurrent_hsnn(
        size: usize,
        recurrent_probability: f32,
    ) -> Result<HSNNNetwork> {
        let mut builder = HSNNBuilder::new()
            .neurons(size)
            .activation_rule(activation_rules::weighted_sum(0.7, 0.1));
            
        // Add recurrent connections
        let recurrent_edges = (size as f32 * size as f32 * recurrent_probability) as usize;
        builder = builder.hyperedges(recurrent_edges);
        
        builder.build()
    }
    
    /// Create a hierarchical H-SNN for concept learning
    pub fn hierarchical_hsnn(
        input_size: usize,
        hierarchy_levels: usize,
    ) -> Result<HSNNNetwork> {
        let mut total_neurons = input_size;
        for level in 1..hierarchy_levels {
            total_neurons += input_size / (2_usize.pow(level as u32));
        }
        
        HSNNBuilder::new()
            .neurons(total_neurons)
            .activation_rule(activation_rules::factory_rule(
                (0..input_size).map(NeuronId::new).collect(),
                10
            ))
            .build()
    }
}

// Testing utilities
#[cfg(test)]
pub mod test_utils {
    use super::*;
    
    /// Create a simple test spike
    pub fn test_spike(neuron_id: u32, time_ms: u64) -> Spike {
        Spike::binary(NeuronId::new(neuron_id), Time::from_millis(time_ms)).unwrap()
    }
    
    /// Create a test network with specified size
    pub fn test_network(neurons: usize, hyperedges: usize) -> HSNNNetwork {
        HSNNBuilder::new()
            .neurons(neurons)
            .hyperedges(hyperedges)
            .build()
            .unwrap()
    }
    
    /// Create a sequence of test spikes
    pub fn test_spike_sequence(
        neuron_id: u32,
        count: usize,
        interval_ms: u64,
        start_ms: u64,
    ) -> Vec<Spike> {
        (0..count)
            .map(|i| test_spike(neuron_id, start_ms + i as u64 * interval_ms))
            .collect()
    }
}