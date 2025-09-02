//! Network builder for easy construction of neural networks
//!
//! This module provides a fluent builder interface for constructing
//! neural networks with different connectivity types and configurations.

use crate::{
    connectivity::{NetworkConnectivity, types::ConnectivityStats},
    network::{SpikeNetwork, PlasticityManager, NetworkStats},
    neuron::{Neuron, NeuronPool, NeuronId, LIFNeuron, LIFConfig, NeuronType},
    plasticity::STDPConfig,
    encoding::{MultiModalEncoder, EncodingConfig},
    time::Time,
    error::{SHNNError, Result},
};
use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Builder for constructing neural networks
#[derive(Debug)]
pub struct NetworkBuilder<C> 
where
    C: NetworkConnectivity<NeuronId>,
{
    connectivity: Option<C>,
    plasticity_config: PlasticityConfig,
    encoding_config: EncodingConfig,
    time_step: Time,
    max_pending_spikes: usize,
    neuron_config: NeuronConfig,
}

impl<C> NetworkBuilder<C>
where
    C: NetworkConnectivity<NeuronId> + 'static,
{
    /// Create a new network builder
    pub fn new() -> Self {
        Self {
            connectivity: None,
            plasticity_config: PlasticityConfig::default(),
            encoding_config: EncodingConfig::default(),
            time_step: Time::from_millis(1),
            max_pending_spikes: 10000,
            neuron_config: NeuronConfig::default(),
        }
    }
    
    /// Set the connectivity structure
    pub fn with_connectivity(mut self, connectivity: C) -> Self {
        self.connectivity = Some(connectivity);
        self
    }
    
    /// Configure plasticity
    pub fn with_plasticity(mut self, config: PlasticityConfig) -> Self {
        self.plasticity_config = config;
        self
    }
    
    /// Configure encoding
    pub fn with_encoding(mut self, config: EncodingConfig) -> Self {
        self.encoding_config = config;
        self
    }
    
    /// Set simulation time step
    pub fn with_time_step(mut self, time_step: Time) -> Self {
        self.time_step = time_step;
        self
    }
    
    /// Set maximum pending spikes
    pub fn with_max_pending_spikes(mut self, max_pending: usize) -> Self {
        self.max_pending_spikes = max_pending;
        self
    }
    
    /// Configure neurons
    pub fn with_neurons(mut self, config: NeuronConfig) -> Self {
        self.neuron_config = config;
        self
    }
    
    /// Enable STDP plasticity with default configuration
    pub fn enable_stdp(mut self) -> Self {
        self.plasticity_config.enabled = true;
        self.plasticity_config.stdp = Some(STDPConfig::default());
        self
    }
    
    /// Enable STDP plasticity with custom configuration
    pub fn enable_stdp_with_config(mut self, stdp_config: STDPConfig) -> Self {
        self.plasticity_config.enabled = true;
        self.plasticity_config.stdp = Some(stdp_config);
        self
    }
    
    /// Disable plasticity
    pub fn disable_plasticity(mut self) -> Self {
        self.plasticity_config.enabled = false;
        self
    }
    
    /// Build the network with LIF neurons
    pub fn build_lif(mut self) -> Result<SpikeNetwork<C, LIFNeuron>> {
        let connectivity = self.connectivity.take().ok_or_else(||
            SHNNError::generic("Connectivity structure not specified"))?;
        
        // Create neuron pool based on configuration
        let neurons = self.create_lif_neuron_pool()?;
        
        // Create plasticity manager
        let plasticity = self.create_plasticity_manager();
        
        // Create encoder
        let encoder = MultiModalEncoder::from_config(self.encoding_config);
        
        // Build the network
        let network = SpikeNetwork::new(connectivity, neurons, plasticity, encoder)
            .set_time_step(self.time_step.into())
            .set_max_pending_spikes(self.max_pending_spikes);
        
        Ok(network)
    }
    
    /// Build the network with custom neuron type
    pub fn build<N: Neuron>(mut self, neurons: NeuronPool<N>) -> Result<SpikeNetwork<C, N>> {
        let connectivity = self.connectivity.take().ok_or_else(||
            SHNNError::generic("Connectivity structure not specified"))?;
        
        // Create plasticity manager
        let plasticity = self.create_plasticity_manager();
        
        // Create encoder
        let encoder = MultiModalEncoder::from_config(self.encoding_config);
        
        // Build the network
        let network = SpikeNetwork::new(connectivity, neurons, plasticity, encoder)
            .set_time_step(self.time_step.into())
            .set_max_pending_spikes(self.max_pending_spikes);
        
        Ok(network)
    }
    
    /// Create LIF neuron pool based on configuration
    fn create_lif_neuron_pool(&self) -> Result<NeuronPool<LIFNeuron>> {
        let mut neuron_pool = NeuronPool::with_capacity(self.neuron_config.count);
        
        let lif_config = self.neuron_config.lif_config.clone().unwrap_or_default();
        
        for i in 0..self.neuron_config.count {
            let neuron_id = NeuronId::new(i as u32);
            let neuron = LIFNeuron::new(neuron_id);
            neuron_pool.add_neuron(neuron);
        }
        
        Ok(neuron_pool)
    }
    
    /// Create plasticity manager based on configuration
    fn create_plasticity_manager(&self) -> PlasticityManager {
        if self.plasticity_config.enabled {
            if let Some(stdp_config) = &self.plasticity_config.stdp {
                let mut manager = PlasticityManager::with_stdp(stdp_config.clone());
                manager.set_learning_rate(self.plasticity_config.learning_rate);
                manager
            } else {
                PlasticityManager::new()
            }
        } else {
            PlasticityManager::new()
        }
    }
}

impl<C> Default for NetworkBuilder<C>
where
    C: NetworkConnectivity<NeuronId> + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for plasticity
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PlasticityConfig {
    /// Whether plasticity is enabled
    pub enabled: bool,
    /// STDP configuration
    pub stdp: Option<STDPConfig>,
    /// Learning rate multiplier
    pub learning_rate: f32,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            stdp: None,
            learning_rate: 1.0,
        }
    }
}

impl PlasticityConfig {
    /// Create configuration with STDP enabled
    pub fn with_stdp(stdp_config: STDPConfig) -> Self {
        Self {
            enabled: true,
            stdp: Some(stdp_config),
            learning_rate: 1.0,
        }
    }
    
    /// Disable plasticity
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            stdp: None,
            learning_rate: 1.0,
        }
    }
    
    /// Set learning rate
    pub fn with_learning_rate(mut self, rate: f32) -> Self {
        self.learning_rate = rate;
        self
    }
}

/// Configuration for neurons
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NeuronConfig {
    /// Number of neurons to create
    pub count: usize,
    /// Type of neurons
    pub neuron_type: NeuronType,
    /// LIF-specific configuration
    pub lif_config: Option<LIFConfig>,
}

impl Default for NeuronConfig {
    fn default() -> Self {
        Self {
            count: 100,
            neuron_type: NeuronType::LIF,
            lif_config: Some(LIFConfig::default()),
        }
    }
}

impl NeuronConfig {
    /// Create configuration for LIF neurons
    pub fn lif(count: usize, config: LIFConfig) -> Self {
        Self {
            count,
            neuron_type: NeuronType::LIF,
            lif_config: Some(config),
        }
    }
    
    /// Create configuration with default LIF neurons
    pub fn lif_default(count: usize) -> Self {
        Self::lif(count, LIFConfig::default())
    }
}

/// Convenient factory functions for common network types
pub struct NetworkFactory;

impl NetworkFactory {
    /// Create a simple feedforward network with graph connectivity
    pub fn feedforward_graph(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weight: f32,
    ) -> Result<SpikeNetwork<crate::connectivity::graph::GraphNetwork, LIFNeuron>> {
        use crate::connectivity::graph::{GraphNetwork, GraphEdge};
        
        let total_neurons = input_size + hidden_size + output_size;
        let mut connectivity = GraphNetwork::new();
        
        // Connect input to hidden
        for i in 0..input_size {
            for j in input_size..(input_size + hidden_size) {
                let edge = GraphEdge::new(
                    NeuronId::new(i as u32),
                    NeuronId::new(j as u32),
                    weight,
                );
                connectivity.add_edge(edge)?;
            }
        }
        
        // Connect hidden to output
        for i in input_size..(input_size + hidden_size) {
            for j in (input_size + hidden_size)..total_neurons {
                let edge = GraphEdge::new(
                    NeuronId::new(i as u32),
                    NeuronId::new(j as u32),
                    weight,
                );
                connectivity.add_edge(edge)?;
            }
        }
        
        let neuron_config = NeuronConfig::lif_default(total_neurons);
        
        NetworkBuilder::new()
            .with_connectivity(connectivity)
            .with_neurons(neuron_config)
            .enable_stdp()
            .build_lif()
    }
    
    /// Create a fully connected network with matrix connectivity
    pub fn fully_connected_matrix(
        size: usize,
        weight: f32,
    ) -> Result<SpikeNetwork<crate::connectivity::matrix::MatrixNetwork, LIFNeuron>> {
        use crate::connectivity::matrix::MatrixNetwork;
        
        let neurons: Vec<NeuronId> = (0..size).map(|i| NeuronId::new(i as u32)).collect();
        let connectivity = MatrixNetwork::fully_connected(&neurons, weight)?;
        
        let neuron_config = NeuronConfig::lif_default(size);
        
        NetworkBuilder::new()
            .with_connectivity(connectivity)
            .with_neurons(neuron_config)
            .enable_stdp()
            .build_lif()
    }

    
    /// Create a sparse random network
    pub fn sparse_random(
        size: usize,
        connection_probability: f32,
        weight_range: (f32, f32),
    ) -> Result<SpikeNetwork<crate::connectivity::sparse::SparseMatrixNetwork, LIFNeuron>> {
        use crate::connectivity::sparse::SparseMatrixNetwork;
        
        let neurons: Vec<NeuronId> = (0..size).map(|i| NeuronId::new(i as u32)).collect();
        let connectivity = SparseMatrixNetwork::random_sparse(
            &neurons,
            connection_probability,
            weight_range
        )?;
        
        let neuron_config = NeuronConfig::lif_default(size);
        
        NetworkBuilder::new()
            .with_connectivity(connectivity)
            .with_neurons(neuron_config)
            .enable_stdp()
            .build_lif()
    }

    
    /// Create a hypergraph network with complex connectivity patterns
    pub fn hypergraph_network(
        size: usize,
        hyperedge_count: usize,
    ) -> Result<SpikeNetwork<crate::hypergraph::HypergraphNetwork, LIFNeuron>> {
        use crate::hypergraph::{HypergraphNetwork, Hyperedge, HyperedgeId, HyperedgeType};
        
        let mut connectivity = HypergraphNetwork::with_capacity(hyperedge_count);
        
        // Create random hyperedges
        for i in 0..hyperedge_count {
            let edge_size = 2 + (i % 4); // Vary hyperedge size from 2 to 5
            let source_count = 1 + (i % 2); // 1-2 sources to ensure targets >= 1
            let target_count = edge_size - source_count;
            
            // Ensure we always have at least 1 target
            let target_count = target_count.max(1);
            
            let sources: Vec<NeuronId> = (0..source_count)
                .map(|j| NeuronId::new(((i + j) % size) as u32))
                .collect();
            
            let targets: Vec<NeuronId> = (source_count..(source_count + target_count))
                .map(|j| NeuronId::new(((i + j) % size) as u32))
                .collect();
            
            let edge_type = match (source_count, target_count) {
                (1, 1) => HyperedgeType::OneToMany,
                (_, 1) => HyperedgeType::ManyToOne,
                _ => HyperedgeType::ManyToMany,
            };
            
            let hyperedge = Hyperedge::new(
                HyperedgeId::new(i as u32),
                sources,
                targets,
                edge_type,
            )?;
            
            connectivity.add_hyperedge(hyperedge)?;
        }
        
        let neuron_config = NeuronConfig::lif_default(size);
        
        NetworkBuilder::new()
            .with_connectivity(connectivity)
            .with_neurons(neuron_config)
            .enable_stdp()
            .build_lif()
    }
}

/// Error type for network building operations
#[derive(Debug, Clone, PartialEq)]
pub enum BuildError {
    /// Missing required component
    MissingComponent(String),
    /// Invalid configuration
    InvalidConfiguration(String),
    /// Connectivity creation failed
    ConnectivityError(String),
    /// Neuron creation failed
    NeuronError(String),
    /// General SHNN error
    Core(SHNNError),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingComponent(comp) => write!(f, "Missing component: {}", comp),
            Self::InvalidConfiguration(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::ConnectivityError(msg) => write!(f, "Connectivity error: {}", msg),
            Self::NeuronError(msg) => write!(f, "Neuron error: {}", msg),
            Self::Core(err) => write!(f, "Core error: {}", err),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for BuildError {}

impl From<SHNNError> for BuildError {
    fn from(err: SHNNError) -> Self {
        Self::Core(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectivity::graph::GraphNetwork;

    #[test]
    fn test_network_builder_basic() {
        let connectivity = GraphNetwork::new();
        let neuron_config = NeuronConfig::lif_default(10);
        
        let network = NetworkBuilder::new()
            .with_connectivity(connectivity)
            .with_neurons(neuron_config)
            .enable_stdp()
            .build_lif();
        
        assert!(network.is_ok());
        let network = network.unwrap();
        assert_eq!(network.neurons().len(), 10);
    }
    
    #[test]
    fn test_plasticity_config() {
        let config = PlasticityConfig::with_stdp(STDPConfig::default())
            .with_learning_rate(0.5);
        
        assert!(config.enabled);
        assert_eq!(config.learning_rate, 0.5);
        assert!(config.stdp.is_some());
    }
    
    #[test]
    fn test_neuron_config() {
        let config = NeuronConfig::lif_default(50);
        
        assert_eq!(config.count, 50);
        assert_eq!(config.neuron_type, NeuronType::LIF);
        assert!(config.lif_config.is_some());
    }
    
    #[test]
    fn test_network_factory_feedforward() {
        let network = NetworkFactory::feedforward_graph(10, 20, 5, 0.5);
        
        assert!(network.is_ok());
        let network = network.unwrap();
        assert_eq!(network.neurons().len(), 35); // 10 + 20 + 5
    }
    
    #[test]
    fn test_network_factory_fully_connected() {
        let network = NetworkFactory::fully_connected_matrix(20, 1.0);
        
        assert!(network.is_ok());
        let network = network.unwrap();
        assert_eq!(network.neurons().len(), 20);
    }
    
    #[test]
    fn test_network_factory_sparse() {
        let network = NetworkFactory::sparse_random(100, 0.1, (0.1, 1.0));
        
        assert!(network.is_ok());
        let network = network.unwrap();
        assert_eq!(network.neurons().len(), 100);
        
        // Check sparsity
        let stats = network.get_connectivity_stats();
        let density = stats.density;
        assert!(density < 0.2); // Should be quite sparse
    }
    
    #[test]
    fn test_build_error_missing_connectivity() {
        let neuron_config = NeuronConfig::lif_default(10);
        
        let result = NetworkBuilder::<GraphNetwork>::new()
            .with_neurons(neuron_config)
            .build_lif();
        
        assert!(result.is_err());
    }
}
#[cfg(feature = "plastic-sum")]
impl NetworkFactory {
    /// Create a feedforward network using PlasticConn (Graph variant)
    ///
    /// Layout:
    /// - Inputs: [0, input_size)
    /// - Hidden: [input_size, input_size + hidden_size)
    /// - Outputs: [input_size + hidden_size, total)
    pub fn feedforward_graph_plastic(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        weight: f32,
    ) -> Result<crate::network::SpikeNetwork<crate::connectivity::plastic_enum::PlasticConn, LIFNeuron>> {
        use crate::connectivity::graph::{GraphNetwork, GraphEdge};
        use crate::spike::NeuronId;

        let total_neurons = input_size + hidden_size + output_size;
        let mut connectivity = GraphNetwork::new();

        // Connect input to hidden
        for i in 0..input_size {
            for j in input_size..(input_size + hidden_size) {
                let edge = GraphEdge::new(
                    NeuronId::new(i as u32),
                    NeuronId::new(j as u32),
                    weight,
                );
                connectivity.add_edge(edge)?;
            }
        }

        // Connect hidden to output
        for i in input_size..(input_size + hidden_size) {
            for j in (input_size + hidden_size)..total_neurons {
                let edge = GraphEdge::new(
                    NeuronId::new(i as u32),
                    NeuronId::new(j as u32),
                    weight,
                );
                connectivity.add_edge(edge)?;
            }
        }

        let connectivity = crate::connectivity::plastic_enum::PlasticConn::from_graph(connectivity);
        let neuron_config = NeuronConfig::lif_default(total_neurons);

        NetworkBuilder::new()
            .with_connectivity(connectivity)
            .with_neurons(neuron_config)
            .enable_stdp()
            .build_lif()
    }

    /// Create a fully connected network using PlasticConn (Matrix variant)
    pub fn fully_connected_matrix_plastic(
        size: usize,
        weight: f32,
    ) -> Result<crate::network::SpikeNetwork<crate::connectivity::plastic_enum::PlasticConn, LIFNeuron>> {
        use crate::connectivity::matrix::MatrixNetwork;
        use crate::spike::NeuronId;

        let neurons: Vec<NeuronId> = (0..size).map(|i| NeuronId::new(i as u32)).collect();
        let m = MatrixNetwork::fully_connected(&neurons, weight)?;
        let connectivity = crate::connectivity::plastic_enum::PlasticConn::from_matrix(m);

        let neuron_config = NeuronConfig::lif_default(size);

        NetworkBuilder::new()
            .with_connectivity(connectivity)
            .with_neurons(neuron_config)
            .enable_stdp()
            .build_lif()
    }

    /// Create a sparse random network using PlasticConn (Sparse variant)
    pub fn sparse_random_plastic(
        size: usize,
        connection_probability: f32,
        weight_range: (f32, f32),
    ) -> Result<crate::network::SpikeNetwork<crate::connectivity::plastic_enum::PlasticConn, LIFNeuron>> {
        use crate::connectivity::sparse::SparseMatrixNetwork;
        use crate::spike::NeuronId;

        let neurons: Vec<NeuronId> = (0..size).map(|i| NeuronId::new(i as u32)).collect();
        let s = SparseMatrixNetwork::random_sparse(&neurons, connection_probability, weight_range)?;
        let connectivity = crate::connectivity::plastic_enum::PlasticConn::from_sparse(s);

        let neuron_config = NeuronConfig::lif_default(size);

        NetworkBuilder::new()
            .with_connectivity(connectivity)
            .with_neurons(neuron_config)
            .enable_stdp()
            .build_lif()
    }
}

#[cfg(all(test, feature = "plastic-sum"))]
mod plastic_builder_tests {
    use super::*;
    use crate::error::Result;

    #[test]
    fn build_graph_plastic_compiles_and_validates() -> Result<()> {
        let mut net = NetworkFactory::feedforward_graph_plastic(2, 1, 1, 1.0)?;
        net.validate()?;
        Ok(())
    }

    #[test]
    fn build_matrix_plastic_compiles_and_validates() -> Result<()> {
        let mut net = NetworkFactory::fully_connected_matrix_plastic(4, 0.5)?;
        net.validate()?;
        Ok(())
    }

    #[test]
    fn build_sparse_plastic_compiles_and_validates() -> Result<()> {
        let mut net = NetworkFactory::sparse_random_plastic(6, 0.2, (0.1, 0.9))?;
        net.validate()?;
        Ok(())
    }
}