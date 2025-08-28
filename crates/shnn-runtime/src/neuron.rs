//! Neuron models for SNN simulation

use crate::{error::*, NeuronId, Time, Spike};

/// Parameters for Leaky Integrate-and-Fire neurons
#[derive(Debug, Clone, PartialEq)]
pub struct LIFParams {
    /// Membrane time constant (ms)
    pub tau_m: f32,
    /// Resting potential (mV)
    pub v_rest: f32,
    /// Reset potential (mV)
    pub v_reset: f32,
    /// Threshold potential (mV)
    pub v_thresh: f32,
    /// Refractory period (ms)
    pub t_refrac: f32,
    /// Membrane resistance (MΩ)
    pub r_m: f32,
    /// Capacitance (nF)
    pub c_m: f32,
}

impl Default for LIFParams {
    fn default() -> Self {
        Self {
            tau_m: 20.0,      // 20ms membrane time constant
            v_rest: -70.0,    // -70mV resting potential
            v_reset: -70.0,   // -70mV reset potential
            v_thresh: -50.0,  // -50mV threshold
            t_refrac: 2.0,    // 2ms refractory period
            r_m: 10.0,        // 10MΩ resistance
            c_m: 1.0,         // 1nF capacitance
        }
    }
}

impl LIFParams {
    /// Create new LIF parameters with validation
    pub fn new(
        tau_m: f32,
        v_rest: f32,
        v_reset: f32,
        v_thresh: f32,
        t_refrac: f32,
        r_m: f32,
        c_m: f32,
    ) -> Result<Self> {
        if tau_m <= 0.0 {
            return Err(RuntimeError::invalid_parameter(
                "tau_m",
                tau_m.to_string(),
                "> 0.0",
            ));
        }
        if v_thresh <= v_rest {
            return Err(RuntimeError::invalid_parameter(
                "v_thresh",
                format!("{} (with v_rest={})", v_thresh, v_rest),
                "> v_rest",
            ));
        }
        if t_refrac < 0.0 {
            return Err(RuntimeError::invalid_parameter(
                "t_refrac",
                t_refrac.to_string(),
                ">= 0.0",
            ));
        }
        if r_m <= 0.0 {
            return Err(RuntimeError::invalid_parameter(
                "r_m",
                r_m.to_string(),
                "> 0.0",
            ));
        }
        if c_m <= 0.0 {
            return Err(RuntimeError::invalid_parameter(
                "c_m",
                c_m.to_string(),
                "> 0.0",
            ));
        }

        Ok(Self {
            tau_m,
            v_rest,
            v_reset,
            v_thresh,
            t_refrac,
            r_m,
            c_m,
        })
    }

    /// Validate parameters
    pub fn validate(&self) -> Result<()> {
        Self::new(
            self.tau_m,
            self.v_rest,
            self.v_reset,
            self.v_thresh,
            self.t_refrac,
            self.r_m,
            self.c_m,
        )?;
        Ok(())
    }
}

/// Runtime state of a neuron
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronState {
    /// Membrane potential (mV)
    pub v_m: f32,
    /// Last spike time (ns)
    pub last_spike_time: Option<u64>,
    /// Input current accumulator (nA)
    pub i_input: f32,
    /// Neuron ID
    pub id: NeuronId,
}

impl NeuronState {
    /// Create a new neuron state
    pub fn new(id: NeuronId, v_rest: f32) -> Self {
        Self {
            v_m: v_rest,
            last_spike_time: None,
            i_input: 0.0,
            id,
        }
    }

    /// Check if neuron is in refractory period
    pub fn is_refractory(&self, current_time_ns: u64, t_refrac_ms: f32) -> bool {
        if let Some(last_spike) = self.last_spike_time {
            let dt_ns = current_time_ns.saturating_sub(last_spike);
            let dt_ms = dt_ns as f32 / 1_000_000.0;
            dt_ms < t_refrac_ms
        } else {
            false
        }
    }

    /// Reset neuron after spike
    pub fn reset(&mut self, v_reset: f32, spike_time_ns: u64) {
        self.v_m = v_reset;
        self.last_spike_time = Some(spike_time_ns);
        self.i_input = 0.0;
    }

    /// Add input current
    pub fn add_current(&mut self, current: f32) {
        self.i_input += current;
    }
}

/// Leaky Integrate-and-Fire neuron implementation
#[derive(Debug, Clone)]
pub struct LIFNeuron {
    /// Neuron parameters
    pub params: LIFParams,
    /// Current state
    pub state: NeuronState,
}

impl LIFNeuron {
    /// Create a new LIF neuron
    pub fn new(id: NeuronId, params: LIFParams) -> Result<Self> {
        params.validate()?;
        let state = NeuronState::new(id, params.v_rest);
        Ok(Self { params, state })
    }

    /// Update neuron for one time step
    pub fn update(&mut self, dt_ms: f32, current_time_ns: u64) -> Result<Option<Spike>> {
        // Check refractory period
        if self.state.is_refractory(current_time_ns, self.params.t_refrac) {
            return Ok(None);
        }

        // Update membrane potential using Euler integration
        // dV/dt = (v_rest - v_m + R*I) / tau_m
        let dv_dt = (self.params.v_rest - self.state.v_m + self.params.r_m * self.state.i_input) 
                    / self.params.tau_m;
        
        self.state.v_m += dv_dt * dt_ms;

        // Check for spike
        if self.state.v_m >= self.params.v_thresh {
            let spike = Spike::new(self.state.id, Time::from_nanos(current_time_ns));
            self.state.reset(self.params.v_reset, current_time_ns);
            Ok(Some(spike))
        } else {
            // Decay input current
            self.state.i_input = 0.0;
            Ok(None)
        }
    }

    /// Add synaptic input current
    pub fn receive_input(&mut self, current: f32) {
        self.state.add_current(current);
    }

    /// Get current membrane potential
    pub fn membrane_potential(&self) -> f32 {
        self.state.v_m
    }

    /// Get neuron ID
    pub fn id(&self) -> NeuronId {
        self.state.id
    }

    /// Check if neuron spiked recently
    pub fn last_spike_time(&self) -> Option<Time> {
        self.state.last_spike_time.map(Time::from_nanos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lif_params_default() {
        let params = LIFParams::default();
        assert!(params.validate().is_ok());
        assert!(params.tau_m > 0.0);
        assert!(params.v_thresh > params.v_rest);
    }

    #[test]
    fn test_lif_params_validation() {
        // Invalid tau_m
        let result = LIFParams::new(-1.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0);
        assert!(result.is_err());

        // Invalid threshold
        let result = LIFParams::new(20.0, -70.0, -70.0, -80.0, 2.0, 10.0, 1.0);
        assert!(result.is_err());

        // Valid parameters
        let result = LIFParams::new(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_neuron_state() {
        let id = NeuronId::new(0);
        let mut state = NeuronState::new(id, -70.0);
        
        assert_eq!(state.v_m, -70.0);
        assert_eq!(state.id, id);
        assert_eq!(state.last_spike_time, None);
        
        // Test input accumulation
        state.add_current(5.0);
        state.add_current(3.0);
        assert_eq!(state.i_input, 8.0);
        
        // Test reset
        state.reset(-70.0, 1000000);
        assert_eq!(state.v_m, -70.0);
        assert_eq!(state.last_spike_time, Some(1000000));
        assert_eq!(state.i_input, 0.0);
    }

    #[test]
    fn test_lif_neuron_creation() {
        let id = NeuronId::new(0);
        let params = LIFParams::default();
        let neuron = LIFNeuron::new(id, params).unwrap();
        
        assert_eq!(neuron.id(), id);
        assert_eq!(neuron.membrane_potential(), -70.0);
    }

    #[test]
    fn test_lif_neuron_update_no_spike() {
        let id = NeuronId::new(0);
        let params = LIFParams::default();
        let mut neuron = LIFNeuron::new(id, params).unwrap();
        
        // Small current, should not cause spike
        neuron.receive_input(1.0);
        let spike = neuron.update(1.0, 1000000).unwrap();
        assert!(spike.is_none());
        assert!(neuron.membrane_potential() > -70.0);
    }

    #[test]
    fn test_lif_neuron_spike() {
        let id = NeuronId::new(0);
        let params = LIFParams::default();
        let mut neuron = LIFNeuron::new(id, params).unwrap();
        
        // Large current to cause spike
        neuron.receive_input(100.0);
        let spike = neuron.update(1.0, 1000000).unwrap();
        assert!(spike.is_some());
        assert_eq!(neuron.membrane_potential(), -70.0); // Reset
        assert_eq!(neuron.last_spike_time(), Some(Time::from_nanos(1000000)));
    }

    #[test]
    fn test_refractory_period() {
        let id = NeuronId::new(0);
        let params = LIFParams::default();
        let mut neuron = LIFNeuron::new(id, params).unwrap();
        
        // Cause spike
        neuron.receive_input(100.0);
        let spike1 = neuron.update(1.0, 1000000).unwrap();
        assert!(spike1.is_some());
        
        // Try to spike again immediately (should be blocked by refractory period)
        neuron.receive_input(100.0);
        let spike2 = neuron.update(1.0, 1000000 + 1000000).unwrap(); // +1ms
        assert!(spike2.is_none());
        
        // After refractory period
        neuron.receive_input(100.0);
        let spike3 = neuron.update(1.0, 1000000 + 3000000).unwrap(); // +3ms
        assert!(spike3.is_some());
    }
}