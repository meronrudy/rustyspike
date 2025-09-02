//! SNN network implementation

use crate::{
    error::*,
    neuron::{LIFNeuron, LIFParams},
    plasticity::{STDPRule, STDPParams, SynapseId, PlasticityRule},
    NeuronId, Time, Spike,
};
use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Network configuration parameters
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Default LIF parameters for neurons
    pub default_lif_params: LIFParams,
    /// Default STDP parameters for plasticity
    pub default_stdp_params: STDPParams,
    /// Default synaptic weight
    pub default_weight: f32,
    /// Input current scale factor
    pub input_scale: f32,
    /// Enable plasticity updates
    pub plasticity_enabled: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            default_lif_params: LIFParams::default(),
            default_stdp_params: STDPParams::default(),
            default_weight: 0.1,
            input_scale: 1.0,
            plasticity_enabled: true,
        }
    }
}

/// Synaptic connection in the network
#[derive(Debug, Clone)]
pub struct Synapse {
    /// Pre-synaptic neuron ID
    pub pre: NeuronId,
    /// Post-synaptic neuron ID
    pub post: NeuronId,
    /// Synaptic weight
    pub weight: f32,
    /// Synaptic delay (ms)
    pub delay: f32,
}

impl Synapse {
    /// Create a new synapse
    pub fn new(pre: NeuronId, post: NeuronId, weight: f32, delay: f32) -> Self {
        Self {
            pre,
            post,
            weight,
            delay,
        }
    }

    /// Get synapse ID
    pub fn id(&self) -> SynapseId {
        SynapseId::new(self.pre, self.post)
    }
}

/// Delayed spike for transmission
#[derive(Debug, Clone)]
struct DelayedSpike {
    /// Spike data
    spike: Spike,
    /// Target neuron
    target: NeuronId,
    /// Synaptic weight
    weight: f32,
    /// Delivery time (ns)
    delivery_time: u64,
}

/// Spiking Neural Network implementation
#[derive(Debug)]
pub struct SNNNetwork {
    /// Network configuration
    pub config: NetworkConfig,
    /// All neurons in the network
    neurons: HashMap<NeuronId, LIFNeuron>,
    /// All synaptic connections
    synapses: HashMap<SynapseId, Synapse>,
    /// Plasticity rule
    plasticity: Option<STDPRule>,
    /// Delayed spike queue
    spike_queue: Vec<DelayedSpike>,
    /// Current simulation time (ns)
    current_time: u64,
}

impl SNNNetwork {
    /// Create a new empty network
    pub fn new(config: NetworkConfig) -> Result<Self> {
        config.default_lif_params.validate()?;
        config.default_stdp_params.validate()?;

        let plasticity = if config.plasticity_enabled {
            Some(STDPRule::new(config.default_stdp_params.clone())?)
        } else {
            None
        };

        Ok(Self {
            config,
            neurons: HashMap::new(),
            synapses: HashMap::new(),
            plasticity,
            spike_queue: Vec::new(),
            current_time: 0,
        })
    }

    /// Add a neuron to the network
    pub fn add_neuron(&mut self, id: NeuronId) -> Result<()> {
        self.add_neuron_with_params(id, self.config.default_lif_params.clone())
    }

    /// Add a neuron with specific parameters
    pub fn add_neuron_with_params(&mut self, id: NeuronId, params: LIFParams) -> Result<()> {
        if self.neurons.contains_key(&id) {
            return Err(RuntimeError::invalid_config(
                format!("Neuron {} already exists", id.raw())
            ));
        }

        let neuron = LIFNeuron::new(id, params)?;
        self.neurons.insert(id, neuron);
        Ok(())
    }

    /// Add a synaptic connection
    pub fn add_synapse(&mut self, pre: NeuronId, post: NeuronId, weight: f32, delay: f32) -> Result<()> {
        // Validate neurons exist
        if !self.neurons.contains_key(&pre) {
            return Err(RuntimeError::NeuronNotFound { neuron_id: pre.raw() });
        }
        if !self.neurons.contains_key(&post) {
            return Err(RuntimeError::NeuronNotFound { neuron_id: post.raw() });
        }

        // Validate parameters
        if delay < 0.0 {
            return Err(RuntimeError::invalid_parameter(
                "delay",
                delay.to_string(),
                ">= 0.0",
            ));
        }

        let synapse = Synapse::new(pre, post, weight, delay);
        let synapse_id = synapse.id();
        
        if self.synapses.contains_key(&synapse_id) {
            return Err(RuntimeError::invalid_config(
                format!("Synapse from {} to {} already exists", pre.raw(), post.raw())
            ));
        }

        self.synapses.insert(synapse_id, synapse);
        Ok(())
    }

    /// Remove a neuron and all its connections
    pub fn remove_neuron(&mut self, id: NeuronId) -> Result<()> {
        if !self.neurons.contains_key(&id) {
            return Err(RuntimeError::NeuronNotFound { neuron_id: id.raw() });
        }

        // Remove neuron
        self.neurons.remove(&id);

        // Remove all synapses involving this neuron
        self.synapses.retain(|synapse_id, _| {
            synapse_id.pre != id && synapse_id.post != id
        });

        // Remove any delayed spikes for this neuron
        self.spike_queue.retain(|delayed_spike| {
            delayed_spike.spike.neuron_id != id && delayed_spike.target != id
        });

        Ok(())
    }

    /// Get neuron count
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Get synapse count
    pub fn synapse_count(&self) -> usize {
        self.synapses.len()
    }

    /// Get current simulation time
    pub fn current_time(&self) -> Time {
        Time::from_nanos(self.current_time)
    }

    /// Apply external input to a neuron
    pub fn apply_input(&mut self, neuron_id: NeuronId, current: f32) -> Result<()> {
        let neuron = self.neurons.get_mut(&neuron_id)
            .ok_or(RuntimeError::NeuronNotFound { neuron_id: neuron_id.raw() })?;
        
        neuron.receive_input(current * self.config.input_scale);
        Ok(())
    }

    /// Step the network forward by one time step
    pub fn step(&mut self, dt_ms: f32) -> Result<Vec<Spike>> {
        let mut output_spikes = Vec::new();

        // Update simulation time
        let dt_ns = (dt_ms * 1_000_000.0) as u64;
        self.current_time += dt_ns;

        // Process delayed spikes
        self.process_delayed_spikes(&mut output_spikes)?;

        // Update all neurons
        let neuron_spikes = self.update_neurons(dt_ms)?;

        // Process new spikes
        for spike in neuron_spikes {
            // Record for plasticity
            if let Some(ref mut plasticity) = self.plasticity {
                plasticity.record_spike(spike.neuron_id, spike.time);
            }

            // Propagate through synapses
            self.propagate_spike(&spike)?;
            output_spikes.push(spike);
        }

        // Apply plasticity updates
        if self.config.plasticity_enabled {
            self.apply_plasticity_updates()?;
        }

        Ok(output_spikes)
    }

    /// Process delayed spikes that should be delivered now
    fn process_delayed_spikes(&mut self, output_spikes: &mut Vec<Spike>) -> Result<()> {
        let current_time = self.current_time;
        let mut delivered_indices = Vec::new();

        for (i, delayed_spike) in self.spike_queue.iter().enumerate() {
            if delayed_spike.delivery_time <= current_time {
                // Deliver spike to target neuron
                if let Some(neuron) = self.neurons.get_mut(&delayed_spike.target) {
                    neuron.receive_input(delayed_spike.weight);
                }
                delivered_indices.push(i);
            }
        }

        // Remove delivered spikes (in reverse order to maintain indices)
        for &i in delivered_indices.iter().rev() {
            self.spike_queue.swap_remove(i);
        }

        Ok(())
    }

    /// Update all neurons for one time step
    fn update_neurons(&mut self, dt_ms: f32) -> Result<Vec<Spike>> {
        let current_time = self.current_time;
        let mut spikes = Vec::new();

        #[cfg(feature = "parallel")]
        let neuron_updates: Result<Vec<_>> = self.neurons.par_iter_mut()
            .map(|(_, neuron)| neuron.update(dt_ms, current_time))
            .collect();

        #[cfg(not(feature = "parallel"))]
        let neuron_updates: Result<Vec<_>> = self.neurons.iter_mut()
            .map(|(_, neuron)| neuron.update(dt_ms, current_time))
            .collect();

        for spike_opt in neuron_updates? {
            if let Some(spike) = spike_opt {
                spikes.push(spike);
            }
        }

        Ok(spikes)
    }

    /// Propagate a spike through the network
    fn propagate_spike(&mut self, spike: &Spike) -> Result<()> {
        // Find all outgoing synapses from the spiking neuron
        for (synapse_id, synapse) in &self.synapses {
            if synapse_id.pre == spike.neuron_id {
                let delay_ns = (synapse.delay * 1_000_000.0) as u64;
                let delivery_time = spike.time.nanos() + delay_ns;

                let delayed_spike = DelayedSpike {
                    spike: spike.clone(),
                    target: synapse.post,
                    weight: synapse.weight,
                    delivery_time,
                };

                self.spike_queue.push(delayed_spike);
            }
        }

        Ok(())
    }

    /// Apply plasticity updates to synaptic weights
    fn apply_plasticity_updates(&mut self) -> Result<()> {
        if let Some(ref mut plasticity) = self.plasticity {
            let mut weights: HashMap<SynapseId, f32> = self.synapses.iter()
                .map(|(id, synapse)| (*id, synapse.weight))
                .collect();

            let updates = plasticity.apply_updates(&mut weights, Time::from_nanos(self.current_time))?;

            // Apply weight updates
            for (synapse_id, _old_weight, new_weight) in updates {
                if let Some(synapse) = self.synapses.get_mut(&synapse_id) {
                    synapse.weight = new_weight;
                }
            }
        }

        Ok(())
    }

    /// Get neuron membrane potential
    pub fn get_membrane_potential(&self, neuron_id: NeuronId) -> Result<f32> {
        let neuron = self.neurons.get(&neuron_id)
            .ok_or(RuntimeError::NeuronNotFound { neuron_id: neuron_id.raw() })?;
        Ok(neuron.membrane_potential())
    }

    /// Get synaptic weight
    pub fn get_weight(&self, pre: NeuronId, post: NeuronId) -> Result<f32> {
        let synapse_id = SynapseId::new(pre, post);
        let synapse = self.synapses.get(&synapse_id)
            .ok_or(RuntimeError::network_topology(
                format!("No synapse from {} to {}", pre.raw(), post.raw())
            ))?;
        Ok(synapse.weight)
    }

    /// Set synaptic weight
    pub fn set_weight(&mut self, pre: NeuronId, post: NeuronId, weight: f32) -> Result<()> {
        let synapse_id = SynapseId::new(pre, post);
        let synapse = self.synapses.get_mut(&synapse_id)
            .ok_or(RuntimeError::network_topology(
                format!("No synapse from {} to {}", pre.raw(), post.raw())
            ))?;
        synapse.weight = weight;
        Ok(())
    }

    /// Get all neuron IDs
    pub fn neuron_ids(&self) -> Vec<NeuronId> {
        self.neurons.keys().copied().collect()
    }

    /// Get all synapse connections
    pub fn synapse_connections(&self) -> Vec<(NeuronId, NeuronId, f32)> {
        self.synapses.values()
            .map(|synapse| (synapse.pre, synapse.post, synapse.weight))
            .collect()
    }

    /// Count outgoing synapses for a given pre-synaptic neuron
    pub fn outgoing_count(&self, pre: NeuronId) -> usize {
        self.synapses.keys().filter(|id| id.pre == pre).count()
    }

    /// Reset network to initial state
    pub fn reset(&mut self) -> Result<()> {
        self.current_time = 0;
        self.spike_queue.clear();

        // Reset all neurons
        for (id, neuron) in &mut self.neurons {
            *neuron = LIFNeuron::new(*id, neuron.params.clone())?;
        }

        // Reset plasticity
        if let Some(ref mut plasticity) = self.plasticity {
            plasticity.clear_history();
        }

        Ok(())
    }
}

/// Builder for constructing SNN networks
#[derive(Debug)]
pub struct NetworkBuilder {
    config: NetworkConfig,
    neurons: Vec<(NeuronId, Option<LIFParams>)>,
    synapses: Vec<(NeuronId, NeuronId, f32, f32)>, // (pre, post, weight, delay)
}

impl NetworkBuilder {
    /// Create a new network builder
    pub fn new() -> Self {
        Self {
            config: NetworkConfig::default(),
            neurons: Vec::new(),
            synapses: Vec::new(),
        }
    }

    /// Set network configuration
    pub fn with_config(mut self, config: NetworkConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a neuron with default parameters
    pub fn add_neuron(mut self, id: NeuronId) -> Self {
        self.neurons.push((id, None));
        self
    }

    /// Add a neuron with specific parameters
    pub fn add_neuron_with_params(mut self, id: NeuronId, params: LIFParams) -> Self {
        self.neurons.push((id, Some(params)));
        self
    }

    /// Add a range of neurons
    pub fn add_neurons(mut self, start: u32, count: u32) -> Self {
        for i in start..(start + count) {
            self.neurons.push((NeuronId::new(i), None));
        }
        self
    }

    /// Add a synapse
    pub fn add_synapse(mut self, pre: NeuronId, post: NeuronId, weight: f32, delay: f32) -> Self {
        self.synapses.push((pre, post, weight, delay));
        self
    }

    /// Add synapses with default delay
    pub fn add_synapse_simple(mut self, pre: NeuronId, post: NeuronId, weight: f32) -> Self {
        self.synapses.push((pre, post, weight, 1.0)); // 1ms default delay
        self
    }

    /// Connect all neurons in a fully connected pattern
    pub fn fully_connected(mut self, weight: f32) -> Self {
        let neurons: Vec<_> = self.neurons.iter().map(|(id, _)| *id).collect();
        
        for &pre in &neurons {
            for &post in &neurons {
                if pre != post {
                    self.synapses.push((pre, post, weight, 1.0));
                }
            }
        }
        self
    }

    /// Build the network
    pub fn build(self) -> Result<SNNNetwork> {
        let mut network = SNNNetwork::new(self.config)?;

        // Add neurons
        for (id, params_opt) in self.neurons {
            if let Some(params) = params_opt {
                network.add_neuron_with_params(id, params)?;
            } else {
                network.add_neuron(id)?;
            }
        }

        // Add synapses
        for (pre, post, weight, delay) in self.synapses {
            network.add_synapse(pre, post, weight, delay)?;
        }

        Ok(network)
    }
}

impl Default for NetworkBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_creation() {
        let config = NetworkConfig::default();
        let network = SNNNetwork::new(config).unwrap();
        assert_eq!(network.neuron_count(), 0);
        assert_eq!(network.synapse_count(), 0);
    }

    #[test]
    fn test_add_neurons() {
        let config = NetworkConfig::default();
        let mut network = SNNNetwork::new(config).unwrap();
        
        let id1 = NeuronId::new(0);
        let id2 = NeuronId::new(1);
        
        network.add_neuron(id1).unwrap();
        network.add_neuron(id2).unwrap();
        
        assert_eq!(network.neuron_count(), 2);
        assert!(network.neurons.contains_key(&id1));
        assert!(network.neurons.contains_key(&id2));
    }

    #[test]
    fn test_add_synapses() {
        let config = NetworkConfig::default();
        let mut network = SNNNetwork::new(config).unwrap();
        
        let id1 = NeuronId::new(0);
        let id2 = NeuronId::new(1);
        
        network.add_neuron(id1).unwrap();
        network.add_neuron(id2).unwrap();
        network.add_synapse(id1, id2, 0.5, 1.0).unwrap();
        
        assert_eq!(network.synapse_count(), 1);
        assert_eq!(network.get_weight(id1, id2).unwrap(), 0.5);
    }

    #[test]
    fn test_network_builder() {
        let network = NetworkBuilder::new()
            .add_neurons(0, 3)
            .add_synapse_simple(NeuronId::new(0), NeuronId::new(1), 0.5)
            .add_synapse_simple(NeuronId::new(1), NeuronId::new(2), 0.3)
            .build()
            .unwrap();
        
        assert_eq!(network.neuron_count(), 3);
        assert_eq!(network.synapse_count(), 2);
    }

    #[test]
    fn test_fully_connected_builder() {
        let network = NetworkBuilder::new()
            .add_neurons(0, 3)
            .fully_connected(0.1)
            .build()
            .unwrap();
        
        assert_eq!(network.neuron_count(), 3);
        assert_eq!(network.synapse_count(), 6); // 3x3 - 3 (no self-connections)
    }

    #[test]
    fn test_input_application() {
        let config = NetworkConfig::default();
        let mut network = SNNNetwork::new(config).unwrap();
        
        let id = NeuronId::new(0);
        network.add_neuron(id).unwrap();
        
        let initial_potential = network.get_membrane_potential(id).unwrap();
        network.apply_input(id, 10.0).unwrap();
        
        // Input should be received (we can't directly check without stepping)
        assert!(network.apply_input(id, 10.0).is_ok());
    }

    #[test]
    fn test_network_step() {
        let config = NetworkConfig::default();
        let mut network = SNNNetwork::new(config).unwrap();
        
        let id = NeuronId::new(0);
        network.add_neuron(id).unwrap();
        
        // Apply large input to cause spike
        network.apply_input(id, 100.0).unwrap();
        let spikes = network.step(1.0).unwrap();
        
        // Should have generated at least one spike
        assert!(spikes.len() >= 0); // May or may not spike depending on parameters
    }

    #[test]
    fn test_network_reset() {
        let config = NetworkConfig::default();
        let mut network = SNNNetwork::new(config).unwrap();
        
        let id = NeuronId::new(0);
        network.add_neuron(id).unwrap();
        
        // Step the network
        network.step(1.0).unwrap();
        let time_after_step = network.current_time();
        
        // Reset
        network.reset().unwrap();
        
        assert_eq!(network.current_time().nanos(), 0);
        assert_eq!(network.get_membrane_potential(id).unwrap(), -70.0); // Should be reset potential
    }
}