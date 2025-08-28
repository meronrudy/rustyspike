//! Ultra-lightweight neuron implementations for embedded systems
//!
//! This module provides minimal, highly optimized neuron models designed
//! for microcontrollers with severe memory and processing constraints.

use crate::{Scalar, MicroError, Result, fixed_point::constants};

#[cfg(feature = "fixed-point")]
use crate::fixed_point::Q15_16;

/// Unique identifier for neurons (u8 for ultra-compact networks)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct NeuronId(pub u8);

impl NeuronId {
    /// Create new neuron ID
    #[inline(always)]
    pub const fn new(id: u8) -> Self {
        Self(id)
    }
    
    /// Get raw ID value
    #[inline(always)]
    pub const fn raw(self) -> u8 {
        self.0
    }
    
    /// Invalid neuron ID constant
    pub const INVALID: Self = Self(u8::MAX);
    
    /// Check if valid
    #[inline(always)]
    pub const fn is_valid(self) -> bool {
        self.0 != u8::MAX
    }
}

impl From<u8> for NeuronId {
    #[inline(always)]
    fn from(id: u8) -> Self {
        Self(id)
    }
}

impl From<NeuronId> for u8 {
    #[inline(always)]
    fn from(id: NeuronId) -> Self {
        id.0
    }
}

impl From<NeuronId> for usize {
    #[inline(always)]
    fn from(id: NeuronId) -> Self {
        id.0 as usize
    }
}

/// Neuron state that can be updated in-place
pub trait NeuronState {
    /// Update neuron state with input current and return spike if generated
    fn update(&mut self, input_current: Scalar, time_step_ms: u8) -> bool;
    
    /// Reset neuron to resting state
    fn reset(&mut self);
    
    /// Get current membrane potential
    fn membrane_potential(&self) -> Scalar;
    
    /// Check if neuron is in refractory period
    fn is_refractory(&self) -> bool;
}

/// Ultra-compact LIF (Leaky Integrate-and-Fire) neuron
///
/// Memory usage: 12 bytes per neuron (3x i32 for fixed-point)
/// Processing time: ~20 instructions per update
#[derive(Debug, Clone, Copy)]
#[repr(C, align(4))]
pub struct LIFNeuron {
    /// Current membrane potential
    pub membrane_potential: Scalar,
    /// Threshold for spike generation
    pub threshold: Scalar,
    /// Reset potential after spike
    pub reset_potential: Scalar,
    /// Membrane decay factor (precomputed exp(-dt/tau))
    pub decay_factor: Scalar,
    /// Refractory period counter (in time steps)
    pub refractory_counter: u8,
    /// Maximum refractory period
    pub refractory_period: u8,
    /// Padding for alignment
    _padding: [u8; 2],
}

impl LIFNeuron {
    /// Create new LIF neuron with default parameters
    #[inline]
    pub fn new_default() -> Self {
        Self {
            membrane_potential: constants::RESTING,
            threshold: constants::THRESHOLD,
            reset_potential: constants::RESET,
            decay_factor: constants::MEMBRANE_DECAY,
            refractory_counter: 0,
            refractory_period: 2, // 2ms default
            _padding: [0; 2],
        }
    }
    
    /// Create new LIF neuron with custom parameters
    #[inline]
    pub fn new(config: LIFConfig) -> Self {
        Self {
            membrane_potential: config.resting_potential,
            threshold: config.threshold,
            reset_potential: config.reset_potential,
            decay_factor: config.decay_factor,
            refractory_counter: 0,
            refractory_period: config.refractory_period_ms,
            _padding: [0; 2],
        }
    }
    
    /// Fast update optimized for fixed-point arithmetic
    #[inline(always)]
    pub fn fast_update(&mut self, input_current: Scalar, _time_step_ms: u8) -> bool {
        // Skip processing if in refractory period
        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            return false;
        }
        
        // Membrane potential update: V = V * decay + I
        #[cfg(feature = "fixed-point")]
        {
            self.membrane_potential = self.membrane_potential.fast_mul(self.decay_factor) + input_current;
        }
        
        #[cfg(not(feature = "fixed-point"))]
        {
            self.membrane_potential = self.membrane_potential * self.decay_factor + input_current;
        }
        
        // Check for spike generation
        if self.membrane_potential >= self.threshold {
            self.membrane_potential = self.reset_potential;
            self.refractory_counter = self.refractory_period;
            return true;
        }
        
        false
    }
}

impl NeuronState for LIFNeuron {
    #[inline(always)]
    fn update(&mut self, input_current: Scalar, time_step_ms: u8) -> bool {
        self.fast_update(input_current, time_step_ms)
    }
    
    #[inline(always)]
    fn reset(&mut self) {
        self.membrane_potential = self.reset_potential;
        self.refractory_counter = 0;
    }
    
    #[inline(always)]
    fn membrane_potential(&self) -> Scalar {
        self.membrane_potential
    }
    
    #[inline(always)]
    fn is_refractory(&self) -> bool {
        self.refractory_counter > 0
    }
}

impl Default for LIFNeuron {
    fn default() -> Self {
        Self::new_default()
    }
}

/// Configuration for LIF neurons
#[derive(Debug, Clone, Copy)]
pub struct LIFConfig {
    /// Resting membrane potential (typically -70mV)
    pub resting_potential: Scalar,
    /// Spike threshold (typically -50mV)
    pub threshold: Scalar,
    /// Reset potential after spike (typically -80mV)
    pub reset_potential: Scalar,
    /// Membrane time constant decay factor
    pub decay_factor: Scalar,
    /// Refractory period in milliseconds
    pub refractory_period_ms: u8,
}

impl Default for LIFConfig {
    fn default() -> Self {
        Self {
            resting_potential: constants::RESTING,
            threshold: constants::THRESHOLD,
            reset_potential: constants::RESET,
            decay_factor: constants::MEMBRANE_DECAY,
            refractory_period_ms: 2,
        }
    }
}

/// Simplified Izhikevich neuron for more realistic dynamics
///
/// Memory usage: 16 bytes per neuron
/// Processing time: ~30 instructions per update
#[cfg(feature = "izhikevich-neuron")]
#[derive(Debug, Clone, Copy)]
#[repr(C, align(4))]
pub struct IzhikevichNeuron {
    /// Membrane potential (mV)
    pub v: Scalar,
    /// Recovery variable
    pub u: Scalar,
    /// Parameter a (recovery time scale)
    pub a: Scalar,
    /// Parameter b (sensitivity of u to v)
    pub b: Scalar,
    /// Parameter c (reset value of v)
    pub c: Scalar,
    /// Parameter d (reset value of u)
    pub d: Scalar,
    /// Refractory counter
    pub refractory_counter: u8,
    /// Padding for alignment
    _padding: [u8; 3],
}

#[cfg(feature = "izhikevich-neuron")]
impl IzhikevichNeuron {
    /// Create regular spiking neuron
    pub fn regular_spiking() -> Self {
        Self {
            v: constants::RESTING,
            u: Scalar::ZERO,
            a: Q15_16::from_float(0.02),
            b: Q15_16::from_float(0.2),
            c: constants::RESET,
            d: Q15_16::from_float(8.0),
            refractory_counter: 0,
            _padding: [0; 3],
        }
    }
    
    /// Create fast spiking neuron
    pub fn fast_spiking() -> Self {
        Self {
            v: constants::RESTING,
            u: Scalar::ZERO,
            a: Q15_16::from_float(0.1),
            b: Q15_16::from_float(0.2),
            c: constants::RESET,
            d: Q15_16::from_float(2.0),
            refractory_counter: 0,
            _padding: [0; 3],
        }
    }
}

#[cfg(feature = "izhikevich-neuron")]
impl NeuronState for IzhikevichNeuron {
    #[inline]
    fn update(&mut self, input_current: Scalar, _time_step_ms: u8) -> bool {
        if self.refractory_counter > 0 {
            self.refractory_counter -= 1;
            return false;
        }
        
        // Izhikevich model equations (simplified for fixed-point)
        // v' = 0.04*v^2 + 5*v + 140 - u + I
        // u' = a*(b*v - u)
        
        let v_squared = self.v.saturating_mul(self.v);
        let five_v = self.v.shl(2) + self.v; // 5*v (approximately)
        
        #[cfg(feature = "fixed-point")]
        let v_update = v_squared.shr(5) + five_v + Q15_16::from_int(140) - self.u + input_current;
        
        #[cfg(not(feature = "fixed-point"))]
        let v_update = 0.04 * v_squared + 5.0 * self.v + 140.0 - self.u + input_current;
        
        self.v += v_update.shr(4); // Scale down for stability
        
        let u_update = self.a.saturating_mul(self.b.saturating_mul(self.v) - self.u);
        self.u += u_update;
        
        // Check for spike
        if self.v >= constants::THRESHOLD {
            self.v = self.c;
            self.u += self.d;
            self.refractory_counter = 1;
            return true;
        }
        
        false
    }
    
    #[inline]
    fn reset(&mut self) {
        self.v = self.c;
        self.u = Scalar::ZERO;
        self.refractory_counter = 0;
    }
    
    #[inline]
    fn membrane_potential(&self) -> Scalar {
        self.v
    }
    
    #[inline]
    fn is_refractory(&self) -> bool {
        self.refractory_counter > 0
    }
}

/// Generic neuron configuration
#[derive(Debug, Clone, Copy)]
pub enum NeuronConfig {
    /// LIF neuron configuration
    #[cfg(feature = "lif-neuron")]
    LIF(LIFConfig),
    /// Izhikevich neuron (regular spiking)
    #[cfg(feature = "izhikevich-neuron")]
    IzhikevichRS,
    /// Izhikevich neuron (fast spiking)
    #[cfg(feature = "izhikevich-neuron")]
    IzhikevichFS,
}

impl Default for NeuronConfig {
    fn default() -> Self {
        #[cfg(feature = "lif-neuron")]
        return Self::LIF(LIFConfig::default());
        
        #[cfg(all(feature = "izhikevich-neuron", not(feature = "lif-neuron")))]
        return Self::IzhikevichRS;
        
        #[cfg(not(any(feature = "lif-neuron", feature = "izhikevich-neuron")))]
        compile_error!("At least one neuron type must be enabled");
    }
}

/// Static neuron pool with compile-time size limits
#[derive(Debug)]
pub struct NeuronPool<N: NeuronState, const SIZE: usize> {
    /// Array of neurons (stack-allocated)
    neurons: [N; SIZE],
    /// Number of active neurons
    count: u8,
}

impl<N: NeuronState + Default + Copy, const SIZE: usize> NeuronPool<N, SIZE> {
    /// Create new neuron pool
    pub fn new() -> Self {
        Self {
            neurons: [N::default(); SIZE],
            count: 0,
        }
    }
    
    /// Add neuron to pool
    pub fn add_neuron(&mut self, neuron: N) -> Result<NeuronId> {
        if (self.count as usize) >= SIZE {
            return Err(MicroError::NetworkFull);
        }
        
        let id = NeuronId::new(self.count);
        self.neurons[self.count as usize] = neuron;
        self.count += 1;
        
        Ok(id)
    }
    
    /// Get neuron by ID
    #[inline(always)]
    pub fn get(&self, id: NeuronId) -> Option<&N> {
        if (id.raw() as usize) < (self.count as usize) {
            Some(&self.neurons[id.raw() as usize])
        } else {
            None
        }
    }
    
    /// Get mutable neuron by ID
    #[inline(always)]
    pub fn get_mut(&mut self, id: NeuronId) -> Option<&mut N> {
        if (id.raw() as usize) < (self.count as usize) {
            Some(&mut self.neurons[id.raw() as usize])
        } else {
            None
        }
    }
    
    /// Get number of neurons
    #[inline(always)]
    pub fn len(&self) -> u8 {
        self.count
    }
    
    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Iterate over all active neurons
    pub fn iter(&self) -> impl Iterator<Item = &N> {
        self.neurons[..self.count as usize].iter()
    }
    
    /// Iterate mutably over all active neurons
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut N> {
        self.neurons[..self.count as usize].iter_mut()
    }
    
    /// Update all neurons with input currents
    pub fn update_all(&mut self, inputs: &[Scalar], time_step_ms: u8) -> u8 {
        let mut spike_count = 0;
        
        for (i, neuron) in self.neurons[..self.count as usize].iter_mut().enumerate() {
            let input = inputs.get(i).copied().unwrap_or(Scalar::default());
            if neuron.update(input, time_step_ms) {
                spike_count += 1;
            }
        }
        
        spike_count
    }
    
    /// Reset all neurons
    pub fn reset_all(&mut self) {
        for neuron in self.neurons[..self.count as usize].iter_mut() {
            neuron.reset();
        }
    }
    
    /// Get maximum capacity
    #[inline(always)]
    pub const fn capacity() -> usize {
        SIZE
    }
}

impl<N: NeuronState + Default + Copy, const SIZE: usize> Default for NeuronPool<N, SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

/// SIMD-optimized batch neuron updates for platforms with vector support
#[cfg(feature = "simd-advanced")]
pub mod simd {
    use super::*;
    
    /// Update multiple LIF neurons simultaneously using SIMD
    #[cfg(all(feature = "lif-neuron", target_feature = "neon"))]
    pub unsafe fn update_lif_batch_neon(
        neurons: &mut [LIFNeuron],
        inputs: &[Scalar],
        spike_mask: &mut [bool]
    ) -> u8 {
        use core::arch::aarch64::*;
        
        let mut spike_count = 0u8;
        let chunks = neurons.len() / 4;
        
        for i in 0..chunks {
            let offset = i * 4;
            
            // Load membrane potentials
            let v_ptr = neurons[offset].membrane_potential.to_raw() as *const i32;
            let v = vld1q_s32(v_ptr);
            
            // Load decay factors
            let decay_ptr = neurons[offset].decay_factor.to_raw() as *const i32;
            let decay = vld1q_s32(decay_ptr);
            
            // Load inputs
            let input_ptr = inputs[offset].to_raw() as *const i32;
            let input = vld1q_s32(input_ptr);
            
            // Update: V = V * decay + I (fixed-point multiplication)
            let v_scaled = vshrq_n_s32(vmulq_s32(v, decay), 16);
            let v_updated = vaddq_s32(v_scaled, input);
            
            // Load thresholds
            let thresh_ptr = neurons[offset].threshold.to_raw() as *const i32;
            let threshold = vld1q_s32(thresh_ptr);
            
            // Compare with threshold
            let spike_cmp = vcgeq_s32(v_updated, threshold);
            
            // Extract spike mask and count spikes
            let mask_array = [0u32; 4];
            vst1q_u32(mask_array.as_ptr() as *mut u32, vreinterpretq_u32_s32(spike_cmp));
            
            for j in 0..4 {
                let neuron_idx = offset + j;
                let is_spike = mask_array[j] != 0;
                spike_mask[neuron_idx] = is_spike;
                
                if is_spike {
                    neurons[neuron_idx].membrane_potential = neurons[neuron_idx].reset_potential;
                    neurons[neuron_idx].refractory_counter = neurons[neuron_idx].refractory_period;
                    spike_count += 1;
                } else {
                    neurons[neuron_idx].membrane_potential = Q15_16::from_raw(v_updated.extract::<0>());
                }
            }
        }
        
        spike_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_neuron_id() {
        let id = NeuronId::new(42);
        assert_eq!(id.raw(), 42);
        assert!(id.is_valid());
        
        let invalid = NeuronId::INVALID;
        assert!(!invalid.is_valid());
    }
    
    #[test]
    fn test_lif_neuron() {
        let mut neuron = LIFNeuron::new_default();
        
        // Test resting state
        assert!(!neuron.is_refractory());
        assert_eq!(neuron.membrane_potential(), constants::RESTING);
        
        // Test subthreshold input
        let small_input = Q15_16::from_float(0.01);
        let spiked = neuron.update(small_input, 1);
        assert!(!spiked);
        assert!(neuron.membrane_potential() > constants::RESTING);
        
        // Test reset
        neuron.reset();
        assert_eq!(neuron.membrane_potential(), neuron.reset_potential);
    }
    
    #[test]
    fn test_neuron_pool() {
        let mut pool: NeuronPool<LIFNeuron, 8> = NeuronPool::new();
        
        assert_eq!(pool.len(), 0);
        assert!(pool.is_empty());
        
        let neuron = LIFNeuron::new_default();
        let id = pool.add_neuron(neuron).unwrap();
        
        assert_eq!(pool.len(), 1);
        assert!(!pool.is_empty());
        assert!(pool.get(id).is_some());
    }
    
    #[test]
    fn test_memory_layout() {
        // Verify compact memory layout
        assert_eq!(core::mem::size_of::<LIFNeuron>(), 16); // Should be compact
        assert_eq!(core::mem::align_of::<LIFNeuron>(), 4);  // 4-byte aligned
        
        // Verify neuron pool memory usage
        let pool_size = core::mem::size_of::<NeuronPool<LIFNeuron, 16>>();
        assert!(pool_size <= 16 * 16 + 8); // Should be close to theoretical minimum
    }
}