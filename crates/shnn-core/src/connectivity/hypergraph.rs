//! Hypergraph-based connectivity implementation
//!
//! This module adapts the existing HypergraphNetwork to implement the generic
//! NetworkConnectivity trait, enabling it to be used with the modular network system.

use crate::{
    connectivity::{
        NetworkConnectivity, BatchConnectivity, PlasticConnectivity,
        types::{SpikeRoute, ConnectivityStats, ConnectivityError, ConnectionId},
    },
    hypergraph::{HypergraphNetwork, HyperedgeId, Hyperedge},
    spike::{NeuronId, Spike},
    time::Time,
    error::{SHNNError, Result},
};
use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Connection identifier for hypergraph edges
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HypergraphConnectionId {
    /// The hyperedge ID
    pub edge_id: HyperedgeId,
}

impl ConnectionId for HypergraphConnectionId {
    fn from_raw(data: &[u8]) -> core::result::Result<Self, &'static str> {
        if data.len() != 4 {
            return Err("Hypergraph connection ID requires exactly 4 bytes");
        }
        
        let id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        Ok(Self {
            edge_id: HyperedgeId::new(id),
        })
    }
    
    fn to_raw(&self) -> Vec<u8> {
        self.edge_id.raw().to_le_bytes().to_vec()
    }
    
    fn display_string(&self) -> String {
        format!("HE{}", self.edge_id.raw())
    }
}

impl fmt::Display for HypergraphConnectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_string())
    }
}

impl From<HyperedgeId> for HypergraphConnectionId {
    fn from(edge_id: HyperedgeId) -> Self {
        Self { edge_id }
    }
}

impl From<HypergraphConnectionId> for HyperedgeId {
    fn from(conn_id: HypergraphConnectionId) -> Self {
        conn_id.edge_id
    }
}

/// Route information specific to hypergraph connections
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct HypergraphRouteInfo {
    /// The hyperedge that generated this route
    pub hyperedge: Hyperedge,
    /// Number of targets reached
    pub target_count: usize,
    /// Total weight transmitted
    pub total_weight: f32,
}

/// Error type for hypergraph connectivity operations
#[derive(Debug, Clone, PartialEq)]
pub enum HypergraphConnectivityError {
    /// Wraps an SHNN error
    Core(SHNNError),
    /// Connectivity-specific error
    Connectivity(ConnectivityError),
    /// Invalid hyperedge
    InvalidHyperedge(String),
    /// Hyperedge not found
    HyperedgeNotFound(HyperedgeId),
}

impl fmt::Display for HypergraphConnectivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => write!(f, "Core error: {}", err),
            Self::Connectivity(err) => write!(f, "Connectivity error: {}", err),
            Self::InvalidHyperedge(msg) => write!(f, "Invalid hyperedge: {}", msg),
            Self::HyperedgeNotFound(id) => write!(f, "Hyperedge not found: {}", id),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for HypergraphConnectivityError {}

impl From<SHNNError> for HypergraphConnectivityError {
    fn from(err: SHNNError) -> Self {
        Self::Core(err)
    }
}

impl From<ConnectivityError> for HypergraphConnectivityError {
    fn from(err: ConnectivityError) -> Self {
        Self::Connectivity(err)
    }
}

/// Implementation of NetworkConnectivity for HypergraphNetwork
impl NetworkConnectivity<NeuronId> for HypergraphNetwork {
    type ConnectionId = HypergraphConnectionId;
    type RouteInfo = HypergraphRouteInfo;
    type Error = HypergraphConnectivityError;
    
    fn route_spike(
        &self,
        spike: &Spike,
        current_time: Time,
    ) -> Result<Vec<SpikeRoute>> {
        // Find hyperedges that have the spike source as input
        let mut spike_routes = Vec::new();
        
        for hyperedge_id in self.hyperedge_ids() {
            if let Some(hyperedge) = self.get_hyperedge(hyperedge_id) {
                // Check if this hyperedge has the spiking neuron as a source
                if hyperedge.has_source(spike.source) {
                    let connection_id = HypergraphConnectionId::from(hyperedge_id);
                    
                    // Calculate weights for all targets
                    let targets = hyperedge.targets.clone();
                    let weights: Vec<f32> = targets.iter()
                        .map(|_| hyperedge.weight_function.default_weight() * spike.amplitude)
                        .collect();
                    
                    let spike_route = SpikeRoute::new(
                        connection_id.to_raw(),
                        targets,
                        weights,
                        current_time,
                    ).map_err(|e| HypergraphConnectivityError::InvalidHyperedge(e.to_string()))?;
                    
                    spike_routes.push(spike_route);
                }
            }
        }
        
        Ok(spike_routes)
    }
    
    fn get_targets(&self, source: NeuronId) -> Result<Vec<NeuronId>> {
        let mut targets = Vec::new();
        
        for hyperedge_id in self.hyperedge_ids() {
            if let Some(hyperedge) = self.get_hyperedge(hyperedge_id) {
                if hyperedge.has_source(source) {
                    targets.extend(&hyperedge.targets);
                }
            }
        }
        
        // Remove duplicates
        targets.sort();
        targets.dedup();
        Ok(targets)
    }
    
    fn get_sources(&self, target: NeuronId) -> Result<Vec<NeuronId>> {
        let mut sources = Vec::new();
        
        for hyperedge_id in self.hyperedge_ids() {
            if let Some(hyperedge) = self.get_hyperedge(hyperedge_id) {
                if hyperedge.has_target(target) {
                    sources.extend(&hyperedge.sources);
                }
            }
        }
        
        // Remove duplicates
        sources.sort();
        sources.dedup();
        Ok(sources)
    }
    
    fn add_connection(
        &mut self,
        connection: Self::ConnectionId,
    ) -> Result<()> {
        // This is a placeholder - in practice, you'd need the full hyperedge data
        // This method is more conceptual for the hypergraph case
        Err(HypergraphConnectivityError::InvalidHyperedge(
            "Cannot add hyperedge with only connection ID - use add_hyperedge instead".to_string()
        ).into())
    }
    
    fn remove_connection(
        &mut self,
        connection: Self::ConnectionId,
    ) -> Result<Option<Self::RouteInfo>> {
        let edge_id = connection.edge_id;
        
        if let Some(hyperedge) = self.remove_hyperedge(edge_id) {
            let route_info = HypergraphRouteInfo {
                target_count: hyperedge.targets.len(),
                total_weight: hyperedge.weight_function.default_weight() * hyperedge.targets.len() as f32,
                hyperedge,
            };
            Ok(Some(route_info))
        } else {
            Ok(None)
        }
    }
    
    fn update_weight(
        &mut self,
        connection: Self::ConnectionId,
        new_weight: f32,
    ) -> Result<Option<f32>> {
        let edge_id = connection.edge_id;
        
        if let Some(hyperedge) = self.get_hyperedge_mut(edge_id) {
            let old_weight = hyperedge.weight_function.default_weight();
            
            // Update weight function to uniform with new weight
            hyperedge.weight_function = crate::hypergraph::WeightFunction::Uniform(new_weight);
            
            Ok(Some(old_weight))
        } else {
            Err(HypergraphConnectivityError::HyperedgeNotFound(edge_id).into())
        }
    }
    
    fn get_stats(&self) -> ConnectivityStats {
        let network_stats = self.stats();
        
        ConnectivityStats {
            connection_count: network_stats.edge_count,
            node_count: network_stats.neuron_count,
            average_degree: network_stats.average_degree,
            max_degree: network_stats.max_degree,
            min_degree: 0, // Would need more detailed analysis
            total_weight: 0.0, // Would need to sum all weights
            average_weight: 1.0, // Default assumption
            memory_usage: self.estimate_memory_usage(),
            has_cycles: None, // Would need cycle detection algorithm
            density: if network_stats.neuron_count > 1 {
                network_stats.connection_count as f32 / 
                (network_stats.neuron_count * (network_stats.neuron_count - 1)) as f32
            } else {
                0.0
            },
            custom_stats: vec![
                ("hyperedge_count".to_string(), network_stats.edge_count as f32),
                ("max_hyperedge_size".to_string(), self.max_hyperedge_size() as f32),
            ],
        }
    }
    
    fn validate(&self) -> Result<()> {
        // Check for invalid hyperedges
        for hyperedge_id in self.hyperedge_ids() {
            if let Some(hyperedge) = self.get_hyperedge(hyperedge_id) {
                if hyperedge.sources.is_empty() {
                    return Err(HypergraphConnectivityError::InvalidHyperedge(
                        format!("Hyperedge {} has no sources", hyperedge_id)
                    ).into());
                }
                if hyperedge.targets.is_empty() {
                    return Err(HypergraphConnectivityError::InvalidHyperedge(
                        format!("Hyperedge {} has no targets", hyperedge_id)
                    ).into());
                }
            }
        }
        
        Ok(())
    }
    
    fn reset(&mut self) {
        *self = HypergraphNetwork::new();
    }
    
    fn connection_count(&self) -> usize {
        self.stats().edge_count
    }
    
    fn neuron_count(&self) -> usize {
        self.stats().neuron_count
    }
}

/// Batch operations for hypergraph connectivity
impl BatchConnectivity<NeuronId> for HypergraphNetwork {
    fn add_connections<I>(
        &mut self,
        connections: I,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = Self::ConnectionId>,
    {
        // This is conceptual - you'd need the full hyperedge data
        let count = connections.into_iter().count();
        Err(HypergraphConnectivityError::InvalidHyperedge(
            format!("Cannot add {} hyperedges with only connection IDs - use add_hyperedge instead", count)
        ).into())
    }
    
    fn remove_connections<I>(
        &mut self,
        connections: I,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = Self::ConnectionId>,
    {
        let mut removed_count = 0;
        
        for connection in connections {
            if self.remove_hyperedge(connection.edge_id).is_some() {
                removed_count += 1;
            }
        }
        
        Ok(removed_count)
    }
}

/// Plasticity support for hypergraph connectivity
impl PlasticConnectivity<NeuronId> for HypergraphNetwork {
    fn apply_plasticity(
        &mut self,
        pre_neuron: NeuronId,
        post_neuron: NeuronId,
        weight_delta: f32,
    ) -> Result<Option<f32>> {
        // Find hyperedges connecting pre_neuron to post_neuron
        for hyperedge_id in self.hyperedge_ids() {
            if let Some(hyperedge) = self.get_hyperedge_mut(hyperedge_id) {
                if hyperedge.has_source(pre_neuron) && hyperedge.has_target(post_neuron) {
                    let old_weight = hyperedge.weight_function.default_weight();
                    let new_weight = (old_weight + weight_delta).max(0.0).min(10.0); // Clamp weights
                    
                    hyperedge.weight_function = crate::hypergraph::WeightFunction::Uniform(new_weight);
                    return Ok(Some(new_weight));
                }
            }
        }
        
        Ok(None)
    }
    
    fn get_weight(
        &self,
        pre_neuron: NeuronId,
        post_neuron: NeuronId,
    ) -> Result<Option<f32>> {
        // Find hyperedges connecting pre_neuron to post_neuron
        for hyperedge_id in self.hyperedge_ids() {
            if let Some(hyperedge) = self.get_hyperedge(hyperedge_id) {
                if hyperedge.has_source(pre_neuron) && hyperedge.has_target(post_neuron) {
                    return Ok(Some(hyperedge.weight_function.default_weight()));
                }
            }
        }
        
        Ok(None)
    }
}

/// Helper methods for HypergraphNetwork
impl HypergraphNetwork {
    /// Estimate memory usage of the hypergraph
    fn estimate_memory_usage(&self) -> usize {
        // Rough estimation
        let base_size = core::mem::size_of::<Self>();
        let hyperedge_size = core::mem::size_of::<Option<Hyperedge>>();
        let hyperedges_memory = self.hyperedge_ids().len() * hyperedge_size;
        
        #[cfg(feature = "std")]
        {
            // Add estimated map overhead
            let map_overhead = self.stats().neuron_count * 16; // Rough estimate
            base_size + hyperedges_memory + map_overhead
        }
        
        #[cfg(not(feature = "std"))]
        {
            base_size + hyperedges_memory
        }
    }
    
    /// Get the size of the largest hyperedge
    fn max_hyperedge_size(&self) -> usize {
        self.hyperedge_ids()
            .iter()
            .filter_map(|&id| self.get_hyperedge(id))
            .map(|edge| edge.sources.len() + edge.targets.len())
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hypergraph::{HyperedgeType, WeightFunction},
        time::Duration,
    };

    #[test]
    fn test_hypergraph_connectivity_basic() {
        let mut network = HypergraphNetwork::new();
        
        // Create a simple hyperedge
        let sources = vec![NeuronId::new(0), NeuronId::new(1)];
        let targets = vec![NeuronId::new(2), NeuronId::new(3)];
        let hyperedge = Hyperedge::new(
            HyperedgeId::new(0),
            sources.clone(),
            targets.clone(),
            HyperedgeType::ManyToMany,
        ).expect("Should create valid hyperedge");
        
        network.add_hyperedge(hyperedge).expect("Should add hyperedge");
        
        // Test connectivity trait methods
        assert_eq!(network.connection_count(), 1);
        assert_eq!(network.neuron_count(), 4);
        
        let spike = Spike::new(NeuronId::new(0), Time::from_millis(10), 1.0)
            .expect("Should create valid spike");
        
        let routes = network.route_spike(&spike, Time::from_millis(10))
            .expect("Should route spike");
        
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].targets, targets);
    }
    
    #[test]
    fn test_hypergraph_connection_id() {
        let edge_id = HyperedgeId::new(42);
        let conn_id = HypergraphConnectionId::from(edge_id);
        
        let raw = conn_id.to_raw();
        let reconstructed = HypergraphConnectionId::from_raw(&raw)
            .expect("Should reconstruct from raw");
        
        assert_eq!(conn_id, reconstructed);
        assert_eq!(conn_id.display_string(), "HE42");
    }
    
    #[test]
    fn test_plasticity_operations() {
        let mut network = HypergraphNetwork::new();
        
        let sources = vec![NeuronId::new(0)];
        let targets = vec![NeuronId::new(1)];
        let mut hyperedge = Hyperedge::new(
            HyperedgeId::new(0),
            sources,
            targets,
            HyperedgeType::OneToMany,
        ).expect("Should create valid hyperedge");
        
        hyperedge.weight_function = WeightFunction::Uniform(0.5);
        network.add_hyperedge(hyperedge).expect("Should add hyperedge");
        
        // Test weight getting
        let weight = network.get_weight(NeuronId::new(0), NeuronId::new(1))
            .expect("Should get weight");
        assert_eq!(weight, Some(0.5));
        
        // Test plasticity application
        let new_weight = network.apply_plasticity(
            NeuronId::new(0),
            NeuronId::new(1),
            0.2,
        ).expect("Should apply plasticity");
        assert_eq!(new_weight, Some(0.7));
    }
}