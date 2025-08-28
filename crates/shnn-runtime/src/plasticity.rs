//! Plasticity rules for synaptic learning

use crate::{error::*, NeuronId, Time};
use std::collections::HashMap;

/// Parameters for STDP (Spike-Timing Dependent Plasticity) rule
#[derive(Debug, Clone, PartialEq)]
pub struct STDPParams {
    /// Learning rate for potentiation (weight increase)
    pub a_plus: f32,
    /// Learning rate for depression (weight decrease)  
    pub a_minus: f32,
    /// Time constant for potentiation (ms)
    pub tau_plus: f32,
    /// Time constant for depression (ms)
    pub tau_minus: f32,
    /// Maximum weight value
    pub w_max: f32,
    /// Minimum weight value
    pub w_min: f32,
    /// Maximum time window for STDP (ms)
    pub max_window: f32,
}

impl Default for STDPParams {
    fn default() -> Self {
        Self {
            a_plus: 0.01,      // 1% potentiation rate
            a_minus: 0.012,    // 1.2% depression rate (slightly stronger)
            tau_plus: 20.0,    // 20ms potentiation window
            tau_minus: 20.0,   // 20ms depression window
            w_max: 1.0,        // Maximum weight
            w_min: 0.0,        // Minimum weight
            max_window: 100.0, // 100ms maximum time window
        }
    }
}

impl STDPParams {
    /// Create new STDP parameters with validation
    pub fn new(
        a_plus: f32,
        a_minus: f32,
        tau_plus: f32,
        tau_minus: f32,
        w_max: f32,
        w_min: f32,
        max_window: f32,
    ) -> Result<Self> {
        if a_plus <= 0.0 {
            return Err(RuntimeError::invalid_parameter(
                "a_plus",
                a_plus.to_string(),
                "> 0.0",
            ));
        }
        if a_minus <= 0.0 {
            return Err(RuntimeError::invalid_parameter(
                "a_minus", 
                a_minus.to_string(),
                "> 0.0",
            ));
        }
        if tau_plus <= 0.0 {
            return Err(RuntimeError::invalid_parameter(
                "tau_plus",
                tau_plus.to_string(),
                "> 0.0",
            ));
        }
        if tau_minus <= 0.0 {
            return Err(RuntimeError::invalid_parameter(
                "tau_minus",
                tau_minus.to_string(),
                "> 0.0",
            ));
        }
        if w_max <= w_min {
            return Err(RuntimeError::invalid_parameter(
                "w_max",
                format!("{} (with w_min={})", w_max, w_min),
                "> w_min",
            ));
        }
        if max_window <= 0.0 {
            return Err(RuntimeError::invalid_parameter(
                "max_window",
                max_window.to_string(),
                "> 0.0",
            ));
        }

        Ok(Self {
            a_plus,
            a_minus,
            tau_plus,
            tau_minus,
            w_max,
            w_min,
            max_window,
        })
    }

    /// Validate parameters
    pub fn validate(&self) -> Result<()> {
        Self::new(
            self.a_plus,
            self.a_minus,
            self.tau_plus,
            self.tau_minus,
            self.w_max,
            self.w_min,
            self.max_window,
        )?;
        Ok(())
    }
}

/// Synaptic connection identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SynapseId {
    /// Pre-synaptic neuron
    pub pre: NeuronId,
    /// Post-synaptic neuron
    pub post: NeuronId,
}

impl SynapseId {
    /// Create a new synapse ID
    pub fn new(pre: NeuronId, post: NeuronId) -> Self {
        Self { pre, post }
    }
}

/// Spike timing record for plasticity
#[derive(Debug, Clone)]
struct SpikeRecord {
    /// Neuron ID
    neuron_id: NeuronId,
    /// Spike time (ns)
    time_ns: u64,
}

/// Trait for plasticity rules
pub trait PlasticityRule {
    /// Update synaptic weight based on spike timing
    fn update_weight(
        &self,
        current_weight: f32,
        pre_spike_time: Time,
        post_spike_time: Time,
    ) -> Result<f32>;

    /// Get learning rate parameters
    fn learning_rates(&self) -> (f32, f32); // (a_plus, a_minus)

    /// Get time constants
    fn time_constants(&self) -> (f32, f32); // (tau_plus, tau_minus)

    /// Get weight bounds
    fn weight_bounds(&self) -> (f32, f32); // (w_min, w_max)
}

/// STDP plasticity rule implementation
#[derive(Debug, Clone)]
pub struct STDPRule {
    /// STDP parameters
    pub params: STDPParams,
    /// Recent spike history for neurons
    spike_history: HashMap<NeuronId, Vec<SpikeRecord>>,
}

impl STDPRule {
    /// Create a new STDP rule
    pub fn new(params: STDPParams) -> Result<Self> {
        params.validate()?;
        Ok(Self {
            params,
            spike_history: HashMap::new(),
        })
    }

    /// Record a spike for STDP calculation
    pub fn record_spike(&mut self, neuron_id: NeuronId, spike_time: Time) {
        let spike_record = SpikeRecord {
            neuron_id,
            time_ns: spike_time.nanos(),
        };

        // Add to history
        self.spike_history
            .entry(neuron_id)
            .or_insert_with(Vec::new)
            .push(spike_record);

        // Clean old spikes outside the time window
        self.cleanup_old_spikes(spike_time.nanos());
    }

    /// Clean up spike history outside the time window
    fn cleanup_old_spikes(&mut self, current_time_ns: u64) {
        let max_window_ns = (self.params.max_window * 1_000_000.0) as u64;
        let cutoff_time = current_time_ns.saturating_sub(max_window_ns);

        for spikes in self.spike_history.values_mut() {
            spikes.retain(|spike| spike.time_ns >= cutoff_time);
        }

        // Remove empty entries
        self.spike_history.retain(|_, spikes| !spikes.is_empty());
    }

    /// Apply STDP updates for all relevant synapses
    pub fn apply_updates(
        &mut self,
        weights: &mut HashMap<SynapseId, f32>,
        current_time: Time,
    ) -> Result<Vec<(SynapseId, f32, f32)>> { // (synapse, old_weight, new_weight)
        let mut updates = Vec::new();

        // For each pair of neurons with recent spikes
        for (&pre_id, pre_spikes) in &self.spike_history {
            for (&post_id, post_spikes) in &self.spike_history {
                if pre_id == post_id {
                    continue;
                }

                let synapse_id = SynapseId::new(pre_id, post_id);
                if let Some(&current_weight) = weights.get(&synapse_id) {
                    if let Some(new_weight) = self.calculate_weight_update(
                        current_weight,
                        pre_spikes,
                        post_spikes,
                        current_time,
                    )? {
                        weights.insert(synapse_id, new_weight);
                        updates.push((synapse_id, current_weight, new_weight));
                    }
                }
            }
        }

        Ok(updates)
    }

    /// Calculate weight update for a specific synapse
    fn calculate_weight_update(
        &self,
        current_weight: f32,
        pre_spikes: &[SpikeRecord],
        post_spikes: &[SpikeRecord],
        current_time: Time,
    ) -> Result<Option<f32>> {
        let mut total_delta = 0.0;
        let current_time_ms = current_time.nanos() as f32 / 1_000_000.0;
        let max_window_ms = self.params.max_window;

        // Calculate potentiation (pre before post)
        for pre_spike in pre_spikes {
            let pre_time_ms = pre_spike.time_ns as f32 / 1_000_000.0;
            
            for post_spike in post_spikes {
                let post_time_ms = post_spike.time_ns as f32 / 1_000_000.0;
                let dt = post_time_ms - pre_time_ms;

                // Skip if outside time window
                if dt.abs() > max_window_ms {
                    continue;
                }

                if dt > 0.0 {
                    // Potentiation: pre before post
                    let factor = (-dt / self.params.tau_plus).exp();
                    total_delta += self.params.a_plus * factor;
                } else if dt < 0.0 {
                    // Depression: post before pre
                    let factor = (dt / self.params.tau_minus).exp();
                    total_delta -= self.params.a_minus * factor;
                }
            }
        }

        if total_delta.abs() < f32::EPSILON {
            return Ok(None);
        }

        // Apply weight update with bounds
        let new_weight = (current_weight + total_delta)
            .max(self.params.w_min)
            .min(self.params.w_max);

        if (new_weight - current_weight).abs() < f32::EPSILON {
            Ok(None)
        } else {
            Ok(Some(new_weight))
        }
    }

    /// Clear spike history
    pub fn clear_history(&mut self) {
        self.spike_history.clear();
    }

    /// Get spike count for a neuron
    pub fn spike_count(&self, neuron_id: NeuronId) -> usize {
        self.spike_history
            .get(&neuron_id)
            .map(|spikes| spikes.len())
            .unwrap_or(0)
    }
}

impl PlasticityRule for STDPRule {
    fn update_weight(
        &self,
        current_weight: f32,
        pre_spike_time: Time,
        post_spike_time: Time,
    ) -> Result<f32> {
        let dt_ns = post_spike_time.nanos() as i64 - pre_spike_time.nanos() as i64;
        let dt_ms = dt_ns as f32 / 1_000_000.0;

        // Check if within time window
        if dt_ms.abs() > self.params.max_window {
            return Ok(current_weight);
        }

        let delta = if dt_ms > 0.0 {
            // Potentiation: pre before post
            self.params.a_plus * (-dt_ms / self.params.tau_plus).exp()
        } else {
            // Depression: post before pre
            -self.params.a_minus * (dt_ms / self.params.tau_minus).exp()
        };

        let new_weight = (current_weight + delta)
            .max(self.params.w_min)
            .min(self.params.w_max);

        Ok(new_weight)
    }

    fn learning_rates(&self) -> (f32, f32) {
        (self.params.a_plus, self.params.a_minus)
    }

    fn time_constants(&self) -> (f32, f32) {
        (self.params.tau_plus, self.params.tau_minus)
    }

    fn weight_bounds(&self) -> (f32, f32) {
        (self.params.w_min, self.params.w_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdp_params_default() {
        let params = STDPParams::default();
        assert!(params.validate().is_ok());
        assert!(params.a_plus > 0.0);
        assert!(params.a_minus > 0.0);
        assert!(params.w_max > params.w_min);
    }

    #[test]
    fn test_stdp_params_validation() {
        // Invalid a_plus
        let result = STDPParams::new(-0.01, 0.012, 20.0, 20.0, 1.0, 0.0, 100.0);
        assert!(result.is_err());

        // Invalid weight bounds
        let result = STDPParams::new(0.01, 0.012, 20.0, 20.0, 0.0, 1.0, 100.0);
        assert!(result.is_err());

        // Valid parameters
        let result = STDPParams::new(0.01, 0.012, 20.0, 20.0, 1.0, 0.0, 100.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_synapse_id() {
        let pre = NeuronId::new(0);
        let post = NeuronId::new(1);
        let syn_id = SynapseId::new(pre, post);
        
        assert_eq!(syn_id.pre, pre);
        assert_eq!(syn_id.post, post);
    }

    #[test]
    fn test_stdp_rule_creation() {
        let params = STDPParams::default();
        let rule = STDPRule::new(params).unwrap();
        assert_eq!(rule.spike_count(NeuronId::new(0)), 0);
    }

    #[test]
    fn test_spike_recording() {
        let params = STDPParams::default();
        let mut rule = STDPRule::new(params).unwrap();
        
        let neuron_id = NeuronId::new(0);
        let spike_time = Time::from_millis(10);
        
        rule.record_spike(neuron_id, spike_time);
        assert_eq!(rule.spike_count(neuron_id), 1);
    }

    #[test]
    fn test_plasticity_rule_trait() {
        let params = STDPParams::default();
        let rule = STDPRule::new(params).unwrap();
        
        let pre_time = Time::from_millis(10);
        let post_time = Time::from_millis(15); // 5ms later
        
        let new_weight = rule.update_weight(0.5, pre_time, post_time).unwrap();
        assert!(new_weight > 0.5); // Should be potentiated
        
        let (a_plus, a_minus) = rule.learning_rates();
        assert!(a_plus > 0.0 && a_minus > 0.0);
    }

    #[test]
    fn test_potentiation_vs_depression() {
        let params = STDPParams::default();
        let rule = STDPRule::new(params).unwrap();
        
        let initial_weight = 0.5;
        
        // Potentiation: pre before post
        let pre_time = Time::from_millis(10);
        let post_time = Time::from_millis(15);
        let potentiated = rule.update_weight(initial_weight, pre_time, post_time).unwrap();
        assert!(potentiated > initial_weight);
        
        // Depression: post before pre
        let pre_time = Time::from_millis(15);
        let post_time = Time::from_millis(10);
        let depressed = rule.update_weight(initial_weight, pre_time, post_time).unwrap();
        assert!(depressed < initial_weight);
    }

    #[test]
    fn test_weight_bounds() {
        let params = STDPParams::new(1.0, 1.0, 20.0, 20.0, 1.0, 0.0, 100.0).unwrap();
        let rule = STDPRule::new(params).unwrap();
        
        // Test upper bound
        let pre_time = Time::from_millis(10);
        let post_time = Time::from_millis(11);
        let clamped_high = rule.update_weight(0.99, pre_time, post_time).unwrap();
        assert!(clamped_high <= 1.0);
        
        // Test lower bound
        let pre_time = Time::from_millis(11);
        let post_time = Time::from_millis(10);
        let clamped_low = rule.update_weight(0.01, pre_time, post_time).unwrap();
        assert!(clamped_low >= 0.0);
    }

    #[test]
    fn test_spike_history_cleanup() {
        let mut params = STDPParams::default();
        params.max_window = 10.0; // 10ms window
        let mut rule = STDPRule::new(params).unwrap();
        
        let neuron_id = NeuronId::new(0);
        
        // Add old spike
        rule.record_spike(neuron_id, Time::from_millis(5));
        assert_eq!(rule.spike_count(neuron_id), 1);
        
        // Add new spike (should cleanup old one)
        rule.record_spike(neuron_id, Time::from_millis(20));
        assert_eq!(rule.spike_count(neuron_id), 1);
    }
}