//! Ultra-lightweight neural network implementation for embedded systems
//!
//! This module provides a complete neural network that fits in microcontroller
//! memory with deterministic execution time and zero heap allocation.

use crate::{
    Scalar, MicroError, Result, MicroConfig,
    neuron::{NeuronId, NeuronState, NeuronPool, LIFNeuron},
    connectivity::{Connectivity, Connection},
    time::{MicroTime, Duration},
};

/// Ultra-compact neural network with compile-time size limits
///
/// Memory layout optimized for cache efficiency and minimal footprint.
/// All memory is statically allocated at compile time.
#[derive(Debug)]
pub struct MicroNetwork<const N: usize, const C: usize> {
    /// Neuron pool (stack allocated)
    neurons: NeuronPool<LIFNeuron, N>,
    /// Connection matrix (sparse representation)
    connectivity: Connectivity<C>,
    /// Input buffer for external signals
    input_buffer: [Scalar; MicroConfig::INPUT_BUFFER_SIZE],
    /// Output buffer for motor commands/actuators
    output_buffer: [Scalar; MicroConfig::OUTPUT_BUFFER_SIZE],
    /// Current simulation time
    current_time: MicroTime,
    /// Time step in milliseconds
    time_step_ms: u8,
    /// Network statistics
    stats: NetworkStats,
    /// Processing state
    state: ProcessingState,
}

impl<const N: usize, const C: usize> MicroNetwork<N, C> {
    /// Create new micro network
    pub fn new() -> Self {
        // Compile-time validation
        const _: () = {
            assert!(N <= 255, "Maximum 255 neurons supported (u8 indexing)");
            assert!(C <= N * N, "Too many connections for network size");
            assert!(N > 0, "Network must have at least one neuron");
        };
        
        Self {
            neurons: NeuronPool::new(),
            connectivity: Connectivity::new(),
            input_buffer: [Scalar::default(); MicroConfig::INPUT_BUFFER_SIZE],
            output_buffer: [Scalar::default(); MicroConfig::OUTPUT_BUFFER_SIZE],
            current_time: MicroTime::ZERO,
            time_step_ms: 1, // 1ms default
            stats: NetworkStats::new(),
            state: ProcessingState::Ready,
        }
    }
    
    /// Initialize network with configuration
    pub fn with_config(config: NetworkConfig) -> Result<Self> {
        let mut network = Self::new();
        
        // Add neurons according to configuration
        for i in 0..config.num_neurons.min(N as u8) {
            let neuron = LIFNeuron::new(config.neuron_config);
            network.neurons.add_neuron(neuron)?;
        }
        
        // Add connections according to configuration
        for conn in config.connections.iter().take(C) {
            network.connectivity.add_connection(*conn)?;
        }
        
        network.time_step_ms = config.time_step_ms;
        
        Ok(network)
    }
    
    /// Add neuron to network
    pub fn add_neuron(&mut self, neuron: LIFNeuron) -> Result<NeuronId> {
        self.neurons.add_neuron(neuron)
    }
    
    /// Add connection between neurons
    pub fn add_connection(&mut self, from: NeuronId, to: NeuronId, weight: Scalar) -> Result<()> {
        let connection = Connection::new(from, to, weight);
        self.connectivity.add_connection(connection)
    }
    
    /// Set input values (from sensors)
    pub fn set_inputs(&mut self, inputs: &[Scalar]) -> Result<()> {
        if inputs.len() > self.input_buffer.len() {
            return Err(MicroError::BufferOverflow);
        }
        
        for (i, &input) in inputs.iter().enumerate() {
            if i < self.input_buffer.len() {
                self.input_buffer[i] = input;
            }
        }
        
        Ok(())
    }
    
    /// Get output values (for actuators/motors)
    pub fn get_outputs(&self) -> &[Scalar] {
        &self.output_buffer
    }
    
    /// Process one time step of the neural network
    ///
    /// This is the main processing function optimized for deterministic
    /// execution time and minimal memory allocation.
    pub fn step(&mut self) -> Result<ProcessingResult> {
        if self.state != ProcessingState::Ready {
            return Err(MicroError::NetworkFull); // Using as "busy" error
        }
        
        self.state = ProcessingState::Processing;
        let start_time = self.current_time;
        
        // Clear output buffer
        self.output_buffer.fill(Scalar::default());
        
        // Step 1: Propagate input signals through connectivity matrix
        let mut neuron_inputs = [Scalar::default(); N];
        
        // Add external inputs to input neurons
        for (i, &input) in self.input_buffer.iter().enumerate() {
            if i < neuron_inputs.len() {
                neuron_inputs[i] += input;
            }
        }
        
        // Step 2: Update all neurons and detect spikes
        let mut spike_buffer = [false; N];
        let spike_count = self.update_neurons(&neuron_inputs, &mut spike_buffer);
        
        // Step 3: Propagate spikes through network
        self.propagate_spikes(&spike_buffer, &mut neuron_inputs);
        
        // Step 4: Generate outputs from output neurons
        self.generate_outputs(&neuron_inputs);
        
        // Step 5: Update statistics
        self.stats.total_steps += 1;
        self.stats.total_spikes += spike_count as u32;
        
        // Step 6: Advance time
        self.current_time = self.current_time.add_ms(self.time_step_ms);
        
        self.state = ProcessingState::Ready;
        
        Ok(ProcessingResult {
            spike_count,
            processing_time: self.current_time.since(start_time),
            outputs_changed: spike_count > 0,
        })
    }
    
    /// Process inputs for specified duration
    pub fn process_inputs(&mut self, inputs: &[Scalar], duration_ms: u16) -> Result<ProcessingResult> {
        self.set_inputs(inputs)?;
        
        let mut total_result = ProcessingResult::default();
        let steps = (duration_ms / self.time_step_ms as u16).max(1);
        
        for _ in 0..steps {
            let result = self.step()?;
            total_result.spike_count += result.spike_count;
            total_result.processing_time = total_result.processing_time.add(result.processing_time);
            total_result.outputs_changed |= result.outputs_changed;
        }
        
        Ok(total_result)
    }
    
    /// Reset network to initial state
    pub fn reset(&mut self) {
        self.neurons.reset_all();
        self.input_buffer.fill(Scalar::default());
        self.output_buffer.fill(Scalar::default());
        self.current_time = MicroTime::ZERO;
        self.stats = NetworkStats::new();
        self.state = ProcessingState::Ready;
    }
    
    /// Get network statistics
    pub fn stats(&self) -> &NetworkStats {
        &self.stats
    }
    
    /// Get current time
    pub fn current_time(&self) -> MicroTime {
        self.current_time
    }
    
    /// Get number of neurons
    pub fn neuron_count(&self) -> u8 {
        self.neurons.len()
    }
    
    /// Get number of connections
    pub fn connection_count(&self) -> u8 {
        self.connectivity.len()
    }
    
    /// Check if network is ready for processing
    pub fn is_ready(&self) -> bool {
        self.state == ProcessingState::Ready
    }
    
    /// Get memory usage estimate in bytes
    pub const fn memory_usage() -> usize {
        core::mem::size_of::<Self>()
    }
    
    // Private helper methods
    
    /// Update all neurons with current inputs
    fn update_neurons(&mut self, inputs: &[Scalar], spike_buffer: &mut [bool]) -> u8 {
        let mut spike_count = 0;
        
        for i in 0..self.neurons.len() {
            let neuron_id = NeuronId::new(i);
            if let Some(neuron) = self.neurons.get_mut(neuron_id) {
                let input = inputs.get(i as usize).copied().unwrap_or(Scalar::default());
                let spiked = neuron.update(input, self.time_step_ms);
                
                if i < spike_buffer.len() {
                    spike_buffer[i] = spiked;
                    if spiked {
                        spike_count += 1;
                    }
                }
            }
        }
        
        spike_count
    }
    
    /// Propagate spikes through connectivity matrix
    fn propagate_spikes(&self, spike_buffer: &[bool], neuron_inputs: &mut [Scalar]) {
        for (i, &spiked) in spike_buffer.iter().enumerate() {
            if spiked {
                let source_id = NeuronId::new(i as u8);
                
                // Find all connections from this neuron
                for connection in self.connectivity.get_outgoing(source_id) {
                    let target_idx = connection.target.raw() as usize;
                    if target_idx < neuron_inputs.len() {
                        neuron_inputs[target_idx] += connection.weight;
                    }
                }
            }
        }
    }
    
    /// Generate outputs from output neurons
    fn generate_outputs(&mut self, neuron_inputs: &[Scalar]) {
        // Simple strategy: map last few neurons to outputs
        let output_neuron_start = self.neurons.len().saturating_sub(self.output_buffer.len() as u8);
        
        for (i, output) in self.output_buffer.iter_mut().enumerate() {
            let neuron_idx = output_neuron_start as usize + i;
            if neuron_idx < neuron_inputs.len() {
                *output = neuron_inputs[neuron_idx];
            }
        }
    }
}

impl<const N: usize, const C: usize> Default for MicroNetwork<N, C> {
    fn default() -> Self {
        Self::new()
    }
}

/// Network configuration for initialization
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Number of neurons to create
    pub num_neurons: u8,
    /// Configuration for individual neurons
    pub neuron_config: crate::neuron::LIFConfig,
    /// Initial connections
    pub connections: heapless::Vec<Connection, 64>, // Fixed-size vector
    /// Time step in milliseconds
    pub time_step_ms: u8,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            num_neurons: 8,
            neuron_config: crate::neuron::LIFConfig::default(),
            connections: heapless::Vec::new(),
            time_step_ms: 1,
        }
    }
}

/// Result of network processing
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessingResult {
    /// Number of spikes generated this step
    pub spike_count: u8,
    /// Time spent processing
    pub processing_time: Duration,
    /// Whether outputs changed
    pub outputs_changed: bool,
}

/// Network processing state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessingState {
    /// Ready for processing
    Ready,
    /// Currently processing
    Processing,
    /// Error state
    Error,
}

/// Network performance statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct NetworkStats {
    /// Total processing steps
    pub total_steps: u32,
    /// Total spikes generated
    pub total_spikes: u32,
    /// Maximum spikes in single step
    pub max_spikes_per_step: u8,
    /// Average spikes per step
    pub avg_spikes_per_step: f32,
}

impl NetworkStats {
    /// Create new statistics
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Update statistics with new spike count
    pub fn update(&mut self, spike_count: u8) {
        self.total_spikes += spike_count as u32;
        self.max_spikes_per_step = self.max_spikes_per_step.max(spike_count);
        
        if self.total_steps > 0 {
            self.avg_spikes_per_step = self.total_spikes as f32 / self.total_steps as f32;
        }
    }
    
    /// Get spike rate (spikes per second)
    pub fn spike_rate(&self, time_step_ms: u8) -> f32 {
        if self.total_steps == 0 {
            0.0
        } else {
            let total_time_s = (self.total_steps * time_step_ms as u32) as f32 / 1000.0;
            self.total_spikes as f32 / total_time_s
        }
    }
}

/// Factory functions for common network configurations
impl<const N: usize, const C: usize> MicroNetwork<N, C> {
    /// Create simple feedforward network
    pub fn feedforward(input_size: u8, hidden_size: u8, output_size: u8) -> Result<Self> {
        let total_neurons = input_size + hidden_size + output_size;
        if total_neurons > N as u8 {
            return Err(MicroError::NetworkFull);
        }
        
        let mut network = Self::new();
        
        // Add neurons
        for _ in 0..total_neurons {
            let neuron = LIFNeuron::new_default();
            network.add_neuron(neuron)?;
        }
        
        // Connect input to hidden layer
        for i in 0..input_size {
            for j in 0..hidden_size {
                let from = NeuronId::new(i);
                let to = NeuronId::new(input_size + j);
                let weight = Scalar::from_float(0.5); // Default weight
                network.add_connection(from, to, weight)?;
            }
        }
        
        // Connect hidden to output layer
        for i in 0..hidden_size {
            for j in 0..output_size {
                let from = NeuronId::new(input_size + i);
                let to = NeuronId::new(input_size + hidden_size + j);
                let weight = Scalar::from_float(0.5);
                network.add_connection(from, to, weight)?;
            }
        }
        
        Ok(network)
    }
    
    /// Create recurrent network with random connectivity
    pub fn recurrent(num_neurons: u8, connectivity: f32) -> Result<Self> {
        if num_neurons > N as u8 {
            return Err(MicroError::NetworkFull);
        }
        
        let mut network = Self::new();
        
        // Add neurons
        for _ in 0..num_neurons {
            let neuron = LIFNeuron::new_default();
            network.add_neuron(neuron)?;
        }
        
        // Add random connections
        let total_possible = (num_neurons as u16 * num_neurons as u16) as f32;
        let num_connections = (total_possible * connectivity) as u8;
        
        for _ in 0..num_connections.min(C as u8) {
            let from = NeuronId::new(0); // Simplified - would use proper random in real implementation
            let to = NeuronId::new(1);
            let weight = Scalar::from_float(0.1);
            network.add_connection(from, to, weight)?;
        }
        
        Ok(network)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_micro_network_creation() {
        let network: MicroNetwork<16, 32> = MicroNetwork::new();
        
        assert_eq!(network.neuron_count(), 0);
        assert_eq!(network.connection_count(), 0);
        assert!(network.is_ready());
        assert_eq!(network.current_time(), MicroTime::ZERO);
    }
    
    #[test]
    fn test_neuron_addition() {
        let mut network: MicroNetwork<8, 16> = MicroNetwork::new();
        
        let neuron = LIFNeuron::new_default();
        let id = network.add_neuron(neuron).unwrap();
        
        assert_eq!(network.neuron_count(), 1);
        assert_eq!(id.raw(), 0);
    }
    
    #[test]
    fn test_connection_addition() {
        let mut network: MicroNetwork<8, 16> = MicroNetwork::new();
        
        // Add two neurons first
        let neuron1 = LIFNeuron::new_default();
        let neuron2 = LIFNeuron::new_default();
        let id1 = network.add_neuron(neuron1).unwrap();
        let id2 = network.add_neuron(neuron2).unwrap();
        
        // Add connection
        let weight = Scalar::from_float(0.5);
        network.add_connection(id1, id2, weight).unwrap();
        
        assert_eq!(network.connection_count(), 1);
    }
    
    #[test]
    fn test_step_processing() {
        let mut network: MicroNetwork<4, 8> = MicroNetwork::new();
        
        // Add a simple neuron
        let neuron = LIFNeuron::new_default();
        network.add_neuron(neuron).unwrap();
        
        // Set input
        let inputs = [Scalar::from_float(1.0)];
        network.set_inputs(&inputs).unwrap();
        
        // Process one step
        let result = network.step().unwrap();
        
        assert_eq!(network.stats().total_steps, 1);
        assert!(result.processing_time.as_ms() == 1); // 1ms time step
    }
    
    #[test]
    fn test_memory_usage() {
        let size = MicroNetwork::<16, 32>::memory_usage();
        
        // Should be reasonable for embedded systems
        assert!(size < 2048); // Less than 2KB
        
        // Should be deterministic
        let size2 = MicroNetwork::<16, 32>::memory_usage();
        assert_eq!(size, size2);
    }
    
    #[test]
    fn test_feedforward_factory() {
        let network: MicroNetwork<16, 32> = MicroNetwork::feedforward(4, 6, 2).unwrap();
        
        assert_eq!(network.neuron_count(), 12); // 4 + 6 + 2
        assert!(network.connection_count() > 0);
    }
    
    #[test]
    fn test_compile_time_constraints() {
        // These should compile
        let _small: MicroNetwork<8, 16> = MicroNetwork::new();
        let _medium: MicroNetwork<32, 64> = MicroNetwork::new();
        
        // Test memory usage calculation
        assert!(MicroNetwork::<8, 16>::memory_usage() < MicroNetwork::<32, 64>::memory_usage());
    }
}