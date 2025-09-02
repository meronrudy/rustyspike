//! Simulation engine for SNN networks

use crate::{
    error::*,
    network::{SNNNetwork, NetworkConfig},
    NeuronId, Time, Spike,
};
use std::collections::HashMap;
use std::time::Instant;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Simulation parameters
#[derive(Debug, Clone)]
pub struct SimulationParams {
    /// Time step duration (ns)
    pub dt_ns: u64,
    /// Total simulation duration (ns)
    pub duration_ns: u64,
    /// Record spikes from these neurons (None = all)
    pub record_neurons: Option<Vec<NeuronId>>,
    /// Record membrane potentials (expensive)
    pub record_potentials: bool,
    /// Apply random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Maximum spikes to record (prevents memory issues)
    pub max_recorded_spikes: Option<usize>,
    /// Enable performance sampling
    pub perf_enabled: bool,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            dt_ns: 100_000,              // 0.1ms timestep
            duration_ns: 1_000_000_000,  // 1 second
            record_neurons: None,        // Record all neurons
            record_potentials: false,    // Don't record potentials by default
            random_seed: None,           // No deterministic seed
            max_recorded_spikes: Some(1_000_000), // 1M spike limit
            perf_enabled: false,
        }
    }
}

impl SimulationParams {
    /// Create new simulation parameters with validation
    pub fn new(dt_ns: u64, duration_ns: u64) -> Result<Self> {
        if dt_ns == 0 {
            return Err(RuntimeError::invalid_parameter(
                "dt_ns",
                dt_ns.to_string(),
                "> 0",
            ));
        }
        if duration_ns == 0 {
            return Err(RuntimeError::invalid_parameter(
                "duration_ns", 
                duration_ns.to_string(),
                "> 0",
            ));
        }
        if duration_ns < dt_ns {
            return Err(RuntimeError::invalid_parameter(
                "duration_ns",
                format!("{} (with dt_ns={})", duration_ns, dt_ns),
                ">= dt_ns",
            ));
        }

        Ok(Self {
            dt_ns,
            duration_ns,
            ..Default::default()
        })
    }

    /// Set neurons to record
    pub fn with_recorded_neurons(mut self, neurons: Vec<NeuronId>) -> Self {
        self.record_neurons = Some(neurons);
        self
    }

    /// Enable membrane potential recording
    pub fn with_potential_recording(mut self, enabled: bool) -> Self {
        self.record_potentials = enabled;
        self
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Set maximum spike recording limit
    pub fn with_spike_limit(mut self, limit: usize) -> Self {
        self.max_recorded_spikes = Some(limit);
        self
    }

    /// Enable or disable performance sampling
    pub fn with_perf(mut self, enabled: bool) -> Self {
        self.perf_enabled = enabled;
        self
    }

    /// Get timestep in milliseconds
    pub fn dt_ms(&self) -> f32 {
        self.dt_ns as f32 / 1_000_000.0
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> f32 {
        self.duration_ns as f32 / 1_000_000.0
    }

    /// Get number of simulation steps
    pub fn num_steps(&self) -> usize {
        (self.duration_ns / self.dt_ns) as usize
    }

    /// Validate parameters
    pub fn validate(&self) -> Result<()> {
        Self::new(self.dt_ns, self.duration_ns)?;
        Ok(())
    }
}

/// Input stimulus pattern
#[derive(Debug, Clone)]
pub enum StimulusPattern {
    /// Constant current injection
    Constant {
        /// Target neuron
        neuron: NeuronId,
        /// Current amplitude (nA)
        amplitude: f32,
        /// Start time (ns)
        start_time: u64,
        /// Duration (ns)
        duration: u64,
    },
    /// Poisson spike train
    Poisson {
        /// Target neuron
        neuron: NeuronId,
        /// Firing rate (Hz)
        rate: f32,
        /// Current amplitude per spike (nA)
        amplitude: f32,
        /// Start time (ns)
        start_time: u64,
        /// Duration (ns)
        duration: u64,
    },
    /// Custom spike times
    SpikeTrain {
        /// Target neuron
        neuron: NeuronId,
        /// Current amplitude per spike (nA)
        amplitude: f32,
        /// Spike times (ns)
        spike_times: Vec<u64>,
    },
}

/// Recorded membrane potential sample
#[derive(Debug, Clone)]
pub struct PotentialSample {
    /// Neuron ID
    pub neuron_id: NeuronId,
    /// Sample time (ns)
    pub time_ns: u64,
    /// Membrane potential (mV)
    pub potential: f32,
}

/// Simulation results
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// All recorded spikes
    pub spikes: Vec<Spike>,
    /// Membrane potential traces (if recorded)
    pub potentials: Vec<PotentialSample>,
    /// Final synaptic weights
    pub final_weights: HashMap<(NeuronId, NeuronId), f32>,
    /// Simulation duration (ns)
    pub duration_ns: u64,
    /// Number of steps executed
    pub steps_executed: usize,
    /// Total spike count
    pub total_spikes: usize,
    /// Optional performance report
    pub perf: Option<PerfReport>,
}

impl SimulationResult {
    /// Create a new empty result
    pub fn new(duration_ns: u64) -> Self {
        Self {
            spikes: Vec::new(),
            potentials: Vec::new(),
            final_weights: HashMap::new(),
            duration_ns,
            steps_executed: 0,
            total_spikes: 0,
            perf: None,
        }
    }

    /// Get spikes for a specific neuron
    pub fn spikes_for_neuron(&self, neuron_id: NeuronId) -> Vec<&Spike> {
        self.spikes.iter()
            .filter(|spike| spike.neuron_id == neuron_id)
            .collect()
    }

    /// Get firing rate for a neuron (Hz)
    pub fn firing_rate(&self, neuron_id: NeuronId) -> f32 {
        let spike_count = self.spikes_for_neuron(neuron_id).len();
        let duration_s = self.duration_ns as f32 / 1_000_000_000.0;
        spike_count as f32 / duration_s
    }

    /// Get average firing rate across all neurons (Hz)
    pub fn average_firing_rate(&self) -> f32 {
        let duration_s = self.duration_ns as f32 / 1_000_000_000.0;
        self.total_spikes as f32 / duration_s
    }

    /// Get potentials for a specific neuron
    pub fn potentials_for_neuron(&self, neuron_id: NeuronId) -> Vec<&PotentialSample> {
        self.potentials.iter()
            .filter(|sample| sample.neuron_id == neuron_id)
            .collect()
    }

    /// Export spikes to simple format (time_ns, neuron_id)
    pub fn export_spikes(&self) -> Vec<(u64, u32)> {
        self.spikes.iter()
            .map(|spike| (spike.time.nanos(), spike.neuron_id.raw()))
            .collect()
    }
}

/// Performance metrics collected during simulation steps.
/// Present when SimulationParams::with_perf(true) is used.
#[derive(Debug, Clone)]
pub struct PerfReport {
    /// Average step time in nanoseconds
    pub avg_step_ns: u64,
    /// Max step time in nanoseconds
    pub max_step_ns: u64,
    /// Steps sampled
    pub steps: usize,
}

/// Simulation engine
#[derive(Debug)]
pub struct SimulationEngine {
    /// Network being simulated
    network: SNNNetwork,
    /// Simulation parameters
    params: SimulationParams,
    /// Input stimuli
    stimuli: Vec<StimulusPattern>,
    /// Current simulation results
    results: SimulationResult,
    /// Random number generator state
    rng_state: u64,
    /// Per-step timing samples (ns), captured when perf_enabled
    perf_samples: Vec<u64>,
}

impl SimulationEngine {
    /// Create a new simulation engine
    pub fn new(network: SNNNetwork, params: SimulationParams) -> Result<Self> {
        params.validate()?;
        
        let results = SimulationResult::new(params.duration_ns);
        let rng_state = params.random_seed.unwrap_or(42);

        Ok(Self {
            network,
            params,
            stimuli: Vec::new(),
            results,
            rng_state,
            perf_samples: Vec::new(),
        })
    }

    /// Add an input stimulus
    pub fn add_stimulus(&mut self, stimulus: StimulusPattern) {
        self.stimuli.push(stimulus);
    }


    /// Run the complete simulation
    pub fn run(&mut self) -> Result<SimulationResult> {
        log::info!("Starting simulation: {}ms with {}ms timestep", 
                   self.params.duration_ms(), self.params.dt_ms());

        // Reset network
        self.network.reset()?;
        
        // Reset results
        self.results = SimulationResult::new(self.params.duration_ns);

        let num_steps = self.params.num_steps();
        let dt_ms = self.params.dt_ms();

        // Main simulation loop
        for step in 0..num_steps {
            let current_time_ns = step as u64 * self.params.dt_ns;

            // Step timing start
            let step_start = Instant::now();
            
            // Apply stimuli
            self.apply_stimuli(current_time_ns)?;

            // Step the network
            let step_spikes = self.network.step(dt_ms)
                .map_err(|e| RuntimeError::simulation_step(current_time_ns, e.to_string()))?;

            // Record spikes
            self.record_spikes(step_spikes)?;

            // Record membrane potentials
            if self.params.record_potentials {
                self.record_potentials(current_time_ns)?;
            }

            // Check spike limit
            if let Some(max_spikes) = self.params.max_recorded_spikes {
                if self.results.spikes.len() >= max_spikes {
                    log::warn!("Spike recording limit reached: {}", max_spikes);
                    // Capture timing before early break if perf enabled
                    if self.params.perf_enabled {
                        let elapsed_ns = step_start.elapsed().as_nanos() as u64;
                        self.perf_samples.push(elapsed_ns);
                    }
                    break;
                }
            }

            // Capture step timing
            if self.params.perf_enabled {
                let elapsed_ns = step_start.elapsed().as_nanos() as u64;
                self.perf_samples.push(elapsed_ns);
            }

            // Progress logging
            if step % (num_steps / 10).max(1) == 0 {
                let progress = (step as f32 / num_steps as f32) * 100.0;
                log::debug!("Simulation progress: {:.1}%", progress);
            }
        }

        // Record final weights
        self.record_final_weights();

        // Update final statistics
        self.results.steps_executed = num_steps;
        self.results.total_spikes = self.results.spikes.len();

        log::info!("Simulation completed: {} spikes in {} steps",
                   self.results.total_spikes, self.results.steps_executed);

        // Build performance report if enabled
        if self.params.perf_enabled && !self.perf_samples.is_empty() {
            let steps = self.perf_samples.len();
            let sum: u128 = self.perf_samples.iter().map(|v| *v as u128).sum();
            let avg = (sum / steps as u128) as u64;
            let max = *self.perf_samples.iter().max().unwrap_or(&0);
            self.results.perf = Some(PerfReport {
                avg_step_ns: avg,
                max_step_ns: max,
                steps,
            });
        }

        Ok(self.results.clone())
    }

    /// Apply input stimuli at current time
    fn apply_stimuli(&mut self, current_time_ns: u64) -> Result<()> {
        // Clone stimuli to avoid borrowing issues
        let stimuli = self.stimuli.clone();
        
        for stimulus in &stimuli {
            match stimulus {
                StimulusPattern::Constant { neuron, amplitude, start_time, duration } => {
                    if current_time_ns >= *start_time &&
                       current_time_ns < start_time + duration {
                        self.network.apply_input(*neuron, *amplitude)?;
                    }
                }
                StimulusPattern::Poisson { neuron, rate, amplitude, start_time, duration } => {
                    if current_time_ns >= *start_time &&
                       current_time_ns < start_time + duration {
                        let dt_s = self.params.dt_ns as f32 / 1_000_000_000.0;
                        let spike_prob = rate * dt_s;
                        
                        if self.random_uniform() < spike_prob {
                            self.network.apply_input(*neuron, *amplitude)?;
                        }
                    }
                }
                StimulusPattern::SpikeTrain { neuron, amplitude, spike_times } => {
                    for &spike_time in spike_times {
                        if spike_time == current_time_ns {
                            self.network.apply_input(*neuron, *amplitude)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Record spikes from network step
    fn record_spikes(&mut self, spikes: Vec<Spike>) -> Result<()> {
        for spike in spikes {
            // Check if we should record this neuron
            let should_record = match &self.params.record_neurons {
                Some(recorded) => recorded.contains(&spike.neuron_id),
                None => true,
            };

            if should_record {
                self.results.spikes.push(spike);
            }
        }
        Ok(())
    }

    /// Record membrane potentials
    fn record_potentials(&mut self, current_time_ns: u64) -> Result<()> {
        let neurons_to_record = match &self.params.record_neurons {
            Some(neurons) => neurons.clone(),
            None => self.network.neuron_ids(),
        };

        for neuron_id in neurons_to_record {
            if let Ok(potential) = self.network.get_membrane_potential(neuron_id) {
                let sample = PotentialSample {
                    neuron_id,
                    time_ns: current_time_ns,
                    potential,
                };
                self.results.potentials.push(sample);
            }
        }
        Ok(())
    }

    /// Record final synaptic weights
    fn record_final_weights(&mut self) {
        for (pre, post, weight) in self.network.synapse_connections() {
            self.results.final_weights.insert((pre, post), weight);
        }
    }

    /// Generate random uniform value [0, 1)
    fn random_uniform(&mut self) -> f32 {
        // Simple LCG for reproducibility
        self.rng_state = self.rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.rng_state as f32) / (u64::MAX as f32)
    }

    /// Get reference to network
    pub fn network(&self) -> &SNNNetwork {
        &self.network
    }

    /// Get mutable reference to network
    pub fn network_mut(&mut self) -> &mut SNNNetwork {
        &mut self.network
    }

    /// Get simulation parameters
    pub fn params(&self) -> &SimulationParams {
        &self.params
    }

    /// Get current results
    pub fn results(&self) -> &SimulationResult {
        &self.results
    }
}

/// Run a fixed-step deterministic simulation with a provided network
pub fn run_fixed_step(
    mut network: SNNNetwork,
    dt_ns: u64,
    duration_ns: u64,
    seed: Option<u64>,
) -> Result<SimulationResult> {
    let params = SimulationParams::new(dt_ns, duration_ns)?
        .with_spike_limit(1_000_000);
    let mut engine = SimulationEngine::new(network, SimulationParams { random_seed: seed, ..params })?;
    engine.run()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::NetworkBuilder;

    #[test]
    fn test_simulation_params_default() {
        let params = SimulationParams::default();
        assert!(params.validate().is_ok());
        assert!(params.dt_ns > 0);
        assert!(params.duration_ns > 0);
        assert!(params.duration_ns >= params.dt_ns);
    }

    #[test]
    fn test_simulation_params_validation() {
        // Zero timestep
        let result = SimulationParams::new(0, 1000000);
        assert!(result.is_err());

        // Zero duration
        let result = SimulationParams::new(100000, 0);
        assert!(result.is_err());

        // Duration < timestep
        let result = SimulationParams::new(1000000, 100000);
        assert!(result.is_err());

        // Valid parameters
        let result = SimulationParams::new(100000, 1000000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simulation_params_conversions() {
        let params = SimulationParams::new(100_000, 1_000_000).unwrap(); // 0.1ms, 1ms
        assert_eq!(params.dt_ms(), 0.1);
        assert_eq!(params.duration_ms(), 1.0);
        assert_eq!(params.num_steps(), 10);
    }

    #[test]
    fn test_simulation_result() {
        let mut result = SimulationResult::new(1_000_000_000); // 1 second
        
        // Add some test spikes
        let neuron_id = NeuronId::new(0);
        result.spikes.push(Spike::new(neuron_id, Time::from_millis(100)));
        result.spikes.push(Spike::new(neuron_id, Time::from_millis(200)));
        result.total_spikes = 2;
        
        assert_eq!(result.spikes_for_neuron(neuron_id).len(), 2);
        assert_eq!(result.firing_rate(neuron_id), 2.0); // 2 spikes in 1 second
    }

    #[test]
    fn test_stimulus_patterns() {
        let constant = StimulusPattern::Constant {
            neuron: NeuronId::new(0),
            amplitude: 10.0,
            start_time: 1000000,
            duration: 5000000,
        };

        let poisson = StimulusPattern::Poisson {
            neuron: NeuronId::new(1),
            rate: 100.0,
            amplitude: 5.0,
            start_time: 0,
            duration: 10000000,
        };

        let spike_train = StimulusPattern::SpikeTrain {
            neuron: NeuronId::new(2),
            amplitude: 15.0,
            spike_times: vec![1000000, 2000000, 3000000],
        };

        // Just test that patterns can be created
        match constant {
            StimulusPattern::Constant { amplitude, .. } => assert_eq!(amplitude, 10.0),
            _ => panic!("Wrong pattern type"),
        }
    }

    #[test]
    fn test_simulation_engine_creation() {
        let network = NetworkBuilder::new()
            .add_neurons(0, 2)
            .build()
            .unwrap();
        
        let params = SimulationParams::new(100_000, 1_000_000).unwrap();
        let engine = SimulationEngine::new(network, params).unwrap();
        
        assert_eq!(engine.network().neuron_count(), 2);
        assert_eq!(engine.params().dt_ns, 100_000);
    }

    #[test]
    fn test_run_fixed_step_smoke() {
        let network = NetworkBuilder::new()
            .add_neurons(0, 2)
            .add_synapse_simple(NeuronId::new(0), NeuronId::new(1), 0.2)
            .build()
            .unwrap();

        let result = super::run_fixed_step(network, 100_000, 1_000_000, Some(1234)).unwrap();
        assert!(result.steps_executed >= 1);
    }

    fn test_simple_simulation() {
        let network = NetworkBuilder::new()
            .add_neurons(0, 1)
            .build()
            .unwrap();
        
        let params = SimulationParams::new(100_000, 1_000_000).unwrap(); // 0.1ms steps, 1ms total
        let mut engine = SimulationEngine::new(network, params).unwrap();
        
        // Add constant input to make neuron spike
        engine.add_stimulus(StimulusPattern::Constant {
            neuron: NeuronId::new(0),
            amplitude: 100.0, // Large current to ensure spike
            start_time: 0,
            duration: 1_000_000,
        });
        
        let result = engine.run().unwrap();
        assert_eq!(result.steps_executed, 10);
        // May or may not have spikes depending on parameters, but should complete
    }

    #[test]
    fn test_determinism_reproducibility() {
        // Build identical networks and run twice with the same seed
        let network1 = NetworkBuilder::new()
            .add_neurons(0, 2)
            .add_synapse_simple(NeuronId::new(0), NeuronId::new(1), 0.2)
            .build()
            .unwrap();

        let network2 = NetworkBuilder::new()
            .add_neurons(0, 2)
            .add_synapse_simple(NeuronId::new(0), NeuronId::new(1), 0.2)
            .build()
            .unwrap();

        let res1 = super::run_fixed_step(network1, 100_000, 1_000_000, Some(9999)).unwrap();
        let res2 = super::run_fixed_step(network2, 100_000, 1_000_000, Some(9999)).unwrap();

        // Compare simple invariants (steps, spike counts). For stronger checks,
        // compare exported spike tuples when available.
        assert_eq!(res1.steps_executed, res2.steps_executed);
        assert_eq!(res1.total_spikes, res2.total_spikes);
        assert_eq!(res1.export_spikes(), res2.export_spikes());
    }

    fn test_potential_recording() {
        let network = NetworkBuilder::new()
            .add_neurons(0, 1)
            .build()
            .unwrap();
        
        let params = SimulationParams::new(100_000, 500_000) // 0.1ms steps, 0.5ms total
            .unwrap()
            .with_potential_recording(true);
            
        let mut engine = SimulationEngine::new(network, params).unwrap();
        let result = engine.run().unwrap();
        
        // Should have recorded potentials
        assert!(!result.potentials.is_empty());
    }
}