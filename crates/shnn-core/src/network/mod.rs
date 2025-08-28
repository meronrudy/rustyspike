//! Generic neural network container
//!
//! This module provides a generic network container that abstracts over
//! different connectivity implementations and neuron types, enabling
//! flexible neural network construction.

use crate::{
    connectivity::{NetworkConnectivity, PlasticConnectivity, types::SpikeRoute},
    neuron::{Neuron, NeuronPool, NeuronId},
    spike::{Spike, TimedSpike},
    plasticity::{PlasticityRule, STDPConfig},
    encoding::{SpikeEncoder, MultiModalEncoder},
    time::Time,
    error::{SHNNError, Result},
};
use core::fmt;

#[cfg(feature = "std")]
use std::collections::VecDeque;

#[cfg(not(feature = "std"))]
use heapless::Deque as VecDeque;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub mod builder;
pub mod stats;

pub use builder::NetworkBuilder;
pub use stats::NetworkStats;

/// Generic spiking neural network container
///
/// This container abstracts over connectivity structures and neuron types,
/// allowing users to choose the optimal combination for their use case.
#[derive(Debug)]
pub struct SpikeNetwork<C, N> 
where 
    C: NetworkConnectivity<NeuronId>,
    N: Neuron,
{
    /// Connectivity structure (hypergraph, graph, matrix, etc.)
    connectivity: C,
    
    /// Neuron collection
    neurons: NeuronPool<N>,
    
    /// Plasticity configuration and state
    plasticity: PlasticityManager,
    
    /// Spike encoding/decoding
    encoder: MultiModalEncoder,
    
    /// Pending spikes (for delayed delivery)
    pending_spikes: VecDeque<TimedSpike>,
    
    /// Runtime statistics
    stats: NetworkStats,
    
    /// Current simulation time
    current_time: Time,
    
    /// Simulation step size
    time_step: crate::time::Duration,
    
    /// Maximum number of pending spikes
    max_pending_spikes: usize,
}

impl<C, N> SpikeNetwork<C, N> 
where 
    C: NetworkConnectivity<NeuronId>,
    N: Neuron,
{
    /// Create a new network with specified connectivity and neurons
    pub fn new(
        connectivity: C,
        neurons: NeuronPool<N>,
        plasticity: PlasticityManager,
        encoder: MultiModalEncoder,
    ) -> Self {
        Self {
            connectivity,
            neurons,
            plasticity,
            encoder,
            pending_spikes: VecDeque::new(),
            stats: NetworkStats::default(),
            current_time: Time::ZERO,
            time_step: crate::time::Duration::from_millis(1), // Default 1ms time step
            max_pending_spikes: 10000, // Default limit
        }
    }
    
    /// Set the simulation time step
    pub fn set_time_step(mut self, time_step: Time) -> Self {
        self.time_step = time_step.into();
        self
    }
    
    /// Set the maximum number of pending spikes
    pub fn set_max_pending_spikes(mut self, max_pending: usize) -> Self {
        self.max_pending_spikes = max_pending;
        self
    }
    
    /// Get current simulation time
    pub fn current_time(&self) -> Time {
        self.current_time
    }
    
    /// Get network statistics
    pub fn get_stats(&self) -> &NetworkStats {
        &self.stats
    }
    
    /// Get connectivity statistics
    pub fn get_connectivity_stats(&self) -> crate::connectivity::types::ConnectivityStats {
        self.connectivity.get_stats()
    }
    
    /// Get reference to the connectivity structure
    pub fn connectivity(&self) -> &C {
        &self.connectivity
    }
    
    /// Get mutable reference to the connectivity structure
    pub fn connectivity_mut(&mut self) -> &mut C {
        &mut self.connectivity
    }
    
    /// Get reference to the neuron pool
    pub fn neurons(&self) -> &NeuronPool<N> {
        &self.neurons
    }
    
    /// Get mutable reference to the neuron pool
    pub fn neurons_mut(&mut self) -> &mut NeuronPool<N> {
        &mut self.neurons
    }
    
    /// Process input spikes through the network
    pub fn process_spikes(&mut self, input_spikes: &[Spike]) -> Result<Vec<Spike>> {
        let mut output_spikes = Vec::new();
        
        // Add input spikes to processing queue
        for spike in input_spikes {
            self.add_spike(spike.clone())?;
        }
        
        // Process all spikes scheduled for the current time
        while let Some(timed_spike) = self.get_next_spike() {
            if timed_spike.delivery_time > self.current_time {
                // Put it back - not time yet
                self.pending_spikes.push_front(timed_spike);
                break;
            }
            
            // Route the spike through the connectivity structure
            let routes = self.connectivity.route_spike(&timed_spike.spike, self.current_time)
                .map_err(|e| SHNNError::generic("Connectivity routing failed"))?;
            
            // Process each route
            for route in routes {
                for (&target, &weight) in route.targets.iter().zip(route.weights.iter()) {
                    if let Some(neuron) = self.neurons.get_neuron_mut(target.raw() as usize) {
                        // Integrate weighted input
                        neuron.integrate(weight as f64, self.time_step.as_nanos() as u64);
                        
                        // Check for output spike
                        if let Some(output_spike) = neuron.update(self.time_step.as_nanos() as u64) {
                            output_spikes.push(output_spike.clone());
                            
                            // Apply plasticity if enabled
                            self.apply_plasticity(&timed_spike.spike, &output_spike, weight)?;
                        }
                    }
                }
            }
            
            // Update statistics
            self.stats.total_spikes_processed += 1;
        }
        
        // Update stats with output spikes
        self.stats.total_spikes_generated += output_spikes.len();
        
        Ok(output_spikes)
    }
    
    /// Step the simulation forward by one time step
    pub fn step(&mut self) -> Result<Vec<Spike>> {
        let output_spikes = self.process_spikes(&[])?;
        self.current_time = self.current_time + self.time_step;
        self.stats.simulation_steps += 1;
        Ok(output_spikes)
    }
    
    /// Run simulation for a specified duration
    pub fn run_for(&mut self, duration: Time) -> Result<Vec<Spike>> {
        let end_time = self.current_time + duration.into();
        let mut all_output_spikes = Vec::new();
        
        while self.current_time < end_time {
            let output_spikes = self.step()?;
            all_output_spikes.extend(output_spikes);
            
            // Check for runaway simulation
            if self.pending_spikes.len() > self.max_pending_spikes {
                return Err(SHNNError::generic("Too many pending spikes - simulation unstable"));
            }
        }
        
        Ok(all_output_spikes)
    }
    
    /// Add a spike to the network
    pub fn add_spike(&mut self, spike: Spike) -> Result<()> {
        let timed_spike = TimedSpike::new(spike, self.current_time);
        
        if self.pending_spikes.len() >= self.max_pending_spikes {
            return Err(SHNNError::generic("Pending spike queue full"));
        }
        
        // Insert spike in temporal order
        self.insert_spike_ordered(timed_spike);
        Ok(())
    }
    
    /// Add a delayed spike to the network
    pub fn add_delayed_spike(&mut self, spike: Spike, delay: crate::time::Duration) -> Result<()> {
        let timed_spike = TimedSpike::with_delay(spike, delay);
        
        if self.pending_spikes.len() >= self.max_pending_spikes {
            return Err(SHNNError::generic("Pending spike queue full"));
        }
        
        self.insert_spike_ordered(timed_spike);
        Ok(())
    }
    
    /// Reset network state
    pub fn reset(&mut self) {
        self.neurons.reset_all();
        self.plasticity.reset();
        self.pending_spikes.clear();
        self.current_time = Time::ZERO;
        self.stats = NetworkStats::default();
    }
    
    /// Validate network configuration
    pub fn validate(&self) -> Result<()> {
        // Validate connectivity
        self.connectivity.validate()
            .map_err(|e| SHNNError::generic("Connectivity validation failed"))?;
        
        // Validate neuron pool
        if self.neurons.is_empty() {
            return Err(SHNNError::generic("Network has no neurons"));
        }
        
        Ok(())
    }
    
    /// Get the next spike to process
    fn get_next_spike(&mut self) -> Option<TimedSpike> {
        self.pending_spikes.pop_front()
    }
    
    /// Insert spike in temporal order
    fn insert_spike_ordered(&mut self, timed_spike: TimedSpike) {
        // Simple insertion - for better performance, could use a priority queue
        let mut inserted = false;
        
        #[cfg(feature = "std")]
        {
            for (i, existing_spike) in self.pending_spikes.iter().enumerate() {
                if timed_spike.delivery_time <= existing_spike.delivery_time {
                    self.pending_spikes.insert(i, timed_spike.clone());
                    inserted = true;
                    break;
                }
            }
        }
        
        #[cfg(not(feature = "std"))]
        {
            // For no-std, we'll just append and sort periodically
            self.pending_spikes.push_back(timed_spike);
            inserted = true;
        }
        
        if !inserted {
            self.pending_spikes.push_back(timed_spike);
        }
    }
    
    /// Apply plasticity rules if connectivity supports it
    fn apply_plasticity(&mut self, pre_spike: &Spike, post_spike: &Spike, weight: f32) -> Result<()> {
        if !self.plasticity.is_enabled() {
            return Ok(());
        }
        
        // Split borrowing to avoid multiple mutable borrows
        let weight_delta = self.plasticity.compute_weight_change(
            pre_spike.timestamp,
            post_spike.timestamp,
            weight,
        );
        
        // Check if connectivity supports plasticity
        if let Ok(plastic_connectivity) = self.try_as_plastic_connectivity() {
            if weight_delta.abs() > 1e-6 { // Only apply if change is significant
                plastic_connectivity.apply_plasticity(
                    pre_spike.source,
                    post_spike.source,
                    weight_delta,
                ).map_err(|_e| SHNNError::generic("Plasticity application failed"))?;
                
                self.stats.plasticity_updates += 1;
            }
        }
        
        Ok(())
    }
    
    /// Try to get a reference to the connectivity as a plastic connectivity
    fn try_as_plastic_connectivity(&mut self) -> Result<&mut dyn PlasticConnectivity<NeuronId, ConnectionId = (NeuronId, NeuronId), RouteInfo = crate::connectivity::types::SpikeRoute, Error = SHNNError>> {
        // This is a bit tricky due to Rust's type system
        // We need to use dynamic dispatch here
        Err(SHNNError::generic("Plasticity not supported by this connectivity type"))
    }
}

/// Plasticity management for the network
#[derive(Debug, Clone)]
pub struct PlasticityManager {
    /// Whether plasticity is enabled
    enabled: bool,
    /// STDP configuration
    stdp_config: Option<STDPConfig>,
    /// Learning rate multiplier
    learning_rate: f32,
    /// Number of plasticity updates
    update_count: u64,
}

impl PlasticityManager {
    /// Create a new plasticity manager
    pub fn new() -> Self {
        Self {
            enabled: false,
            stdp_config: None,
            learning_rate: 1.0,
            update_count: 0,
        }
    }
    
    /// Create with STDP enabled
    pub fn with_stdp(config: STDPConfig) -> Self {
        Self {
            enabled: true,
            stdp_config: Some(config),
            learning_rate: 1.0,
            update_count: 0,
        }
    }
    
    /// Enable plasticity with STDP
    pub fn enable_stdp(&mut self, config: STDPConfig) {
        self.enabled = true;
        self.stdp_config = Some(config);
    }
    
    /// Disable plasticity
    pub fn disable(&mut self) {
        self.enabled = false;
    }
    
    /// Check if plasticity is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Set learning rate
    pub fn set_learning_rate(&mut self, rate: f32) {
        self.learning_rate = rate;
    }
    
    /// Compute weight change based on spike timing
    pub fn compute_weight_change(
        &mut self,
        pre_time: Time,
        post_time: Time,
        current_weight: f32,
    ) -> f32 {
        if !self.enabled {
            return 0.0;
        }
        
        if let Some(ref config) = self.stdp_config {
            let dt = (post_time.as_nanos() as i64) - (pre_time.as_nanos() as i64);
            let dt_ms = dt as f32 / 1_000_000.0; // Convert to milliseconds
            
            let weight_change = if dt_ms > 0.0 {
                // Post-synaptic spike after pre-synaptic (potentiation)
                config.a_plus * (-dt_ms / config.tau_plus).exp()
            } else if dt_ms < 0.0 {
                // Pre-synaptic spike after post-synaptic (depression)
                -config.a_minus * (dt_ms / config.tau_minus).exp()
            } else {
                0.0
            };
            
            self.update_count += 1;
            weight_change * self.learning_rate
        } else {
            0.0
        }
    }
    
    /// Reset plasticity state
    pub fn reset(&mut self) {
        self.update_count = 0;
    }
    
    /// Get number of plasticity updates
    pub fn update_count(&self) -> u64 {
        self.update_count
    }
}

impl Default for PlasticityManager {
    fn default() -> Self {
        Self::new()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        connectivity::graph::GraphNetwork,
        neuron::{LIFNeuron, LIFConfig},
        encoding::TemporalEncoder,
    };

    #[test]
    fn test_spike_network_basic() {
        let connectivity = GraphNetwork::new();
        let neurons = NeuronPool::new();
        let plasticity = PlasticityManager::new();
        let encoder = MultiModalEncoder::new();
        
        let mut network = SpikeNetwork::new(connectivity, neurons, plasticity, encoder);
        
        assert_eq!(network.current_time(), Time::ZERO);
        assert_eq!(network.get_stats().total_spikes_processed, 0);
    }
    
    #[test]
    fn test_spike_processing() {
        let mut connectivity = GraphNetwork::new();
        let mut neurons = NeuronPool::new();
        
        // Add some neurons
        let neuron1 = LIFNeuron::new(NeuronId::new(0), LIFConfig::default());
        let neuron2 = LIFNeuron::new(NeuronId::new(1), LIFConfig::default());
        
        neurons.add_neuron(neuron1);
        neurons.add_neuron(neuron2);
        
        // Add connection
        let edge = crate::connectivity::graph::GraphEdge::new(
            NeuronId::new(0), 
            NeuronId::new(1), 
            1.5
        );
        connectivity.add_edge(edge).expect("Should add edge");
        
        let plasticity = PlasticityManager::new();
        let encoder = MultiModalEncoder::new();
        
        let mut network = SpikeNetwork::new(connectivity, neurons, plasticity, encoder);
        
        // Create input spike
        let spike = Spike::new(NeuronId::new(0), Time::from_millis(1), 1.0)
            .expect("Should create spike");
        
        let output = network.process_spikes(&[spike]).expect("Should process");
        // Output depends on neuron dynamics and may be empty initially
        
        assert!(network.get_stats().total_spikes_processed > 0);
    }
    
    #[test]
    fn test_network_reset() {
        let connectivity = GraphNetwork::new();
        let neurons = NeuronPool::new();
        let plasticity = PlasticityManager::new();
        let encoder = MultiModalEncoder::new();
        
        let mut network = SpikeNetwork::new(connectivity, neurons, plasticity, encoder);
        
        // Add some state
        let spike = Spike::new(NeuronId::new(0), Time::from_millis(1), 1.0)
            .expect("Should create spike");
        network.add_spike(spike).expect("Should add spike");
        
        network.reset();
        
        assert_eq!(network.current_time(), Time::ZERO);
        assert_eq!(network.pending_spikes.len(), 0);
    }
    
    #[test]
    fn test_plasticity_manager() {
        let mut plasticity = PlasticityManager::new();
        assert!(!plasticity.is_enabled());
        
        plasticity.enable_stdp(STDPConfig::default());
        assert!(plasticity.is_enabled());
        
        let weight_change = plasticity.compute_weight_change(
            Time::from_millis(10),
            Time::from_millis(15),
            0.5,
        );
        
        // Should be positive for potentiation
        assert!(weight_change > 0.0);
        assert_eq!(plasticity.update_count(), 1);
    }
}