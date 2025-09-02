//! Traditional graph-based connectivity implementation
//!
//! This module implements pairwise neural connections using a traditional
//! directed graph structure, providing an efficient alternative for standard
//! neural network topologies.

use crate::{
    connectivity::{
        NetworkConnectivity, BatchConnectivity, PlasticConnectivity,
        types::{SpikeRoute, ConnectivityStats, ConnectivityError, ConnectionId},
    },
    spike::{NeuronId, Spike},
    time::Time,
    error::{SHNNError, Result},
};
use core::fmt;

#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet};

#[cfg(not(feature = "std"))]
use heapless::{FnvIndexMap as HashMap, FnvIndexSet as HashSet};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Connection identifier for graph edges
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphConnectionId {
    /// Source neuron ID
    pub source: NeuronId,
    /// Target neuron ID
    pub target: NeuronId,
}

impl ConnectionId for GraphConnectionId {
    fn from_raw(data: &[u8]) -> core::result::Result<Self, &'static str> {
        if data.len() != 8 {
            return Err("Graph connection ID requires exactly 8 bytes");
        }
        
        let source_bytes = [data[0], data[1], data[2], data[3]];
        let target_bytes = [data[4], data[5], data[6], data[7]];
        
        let source = u32::from_le_bytes(source_bytes);
        let target = u32::from_le_bytes(target_bytes);
        
        Ok(Self {
            source: NeuronId::new(source),
            target: NeuronId::new(target),
        })
    }
    
    fn to_raw(&self) -> Vec<u8> {
        let mut raw = Vec::with_capacity(8);
        raw.extend_from_slice(&self.source.raw().to_le_bytes());
        raw.extend_from_slice(&self.target.raw().to_le_bytes());
        raw
    }
    
    fn display_string(&self) -> String {
        format!("{}→{}", self.source, self.target)
    }
}

impl fmt::Display for GraphConnectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_string())
    }
}

impl From<(NeuronId, NeuronId)> for GraphConnectionId {
    fn from((source, target): (NeuronId, NeuronId)) -> Self {
        Self { source, target }
    }
}

/// A directed edge in the graph
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphEdge {
    /// Connection identifier
    pub id: GraphConnectionId,
    /// Synaptic weight
    pub weight: f32,
    /// Transmission delay
    pub delay: Time,
    /// Whether this edge is active
    pub active: bool,
    /// Whether this edge supports plasticity
    pub plastic: bool,
}

impl GraphEdge {
    /// Create a new graph edge
    pub fn new(source: NeuronId, target: NeuronId, weight: f32) -> Self {
        Self {
            id: GraphConnectionId { source, target },
            weight,
            delay: Time::ZERO,
            active: true,
            plastic: true,
        }
    }
    
    /// Create an edge with delay
    pub fn with_delay(source: NeuronId, target: NeuronId, weight: f32, delay: Time) -> Self {
        Self {
            id: GraphConnectionId { source, target },
            weight,
            delay,
            active: true,
            plastic: true,
        }
    }
    
    /// Set whether this edge is plastic
    pub fn set_plastic(mut self, plastic: bool) -> Self {
        self.plastic = plastic;
        self
    }
    
    /// Set whether this edge is active
    pub fn set_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }
}

/// Route information for graph connectivity
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphRouteInfo {
    /// The edge that was used
    pub edge: GraphEdge,
    /// Number of spikes transmitted
    pub spike_count: usize,
    /// Total weight transmitted
    pub total_weight: f32,
}

/// Error type for graph connectivity operations
#[derive(Debug, Clone, PartialEq)]
pub enum GraphConnectivityError {
    /// Wraps an SHNN error
    Core(SHNNError),
    /// Connectivity-specific error
    Connectivity(ConnectivityError),
    /// Edge not found
    EdgeNotFound(GraphConnectionId),
    /// Invalid edge parameters
    InvalidEdge(String),
    /// Neuron not found
    NeuronNotFound(NeuronId),
}

impl fmt::Display for GraphConnectivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => write!(f, "Core error: {}", err),
            Self::Connectivity(err) => write!(f, "Connectivity error: {}", err),
            Self::EdgeNotFound(id) => write!(f, "Edge not found: {}", id),
            Self::InvalidEdge(msg) => write!(f, "Invalid edge: {}", msg),
            Self::NeuronNotFound(id) => write!(f, "Neuron not found: {}", id),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for GraphConnectivityError {}

impl From<SHNNError> for GraphConnectivityError {
    fn from(err: SHNNError) -> Self {
        Self::Core(err)
    }
}

impl From<ConnectivityError> for GraphConnectivityError {
    fn from(err: ConnectivityError) -> Self {
        Self::Connectivity(err)
    }
}

/// Traditional graph-based neural network connectivity
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GraphNetwork {
    /// All edges in the network
    edges: HashMap<GraphConnectionId, GraphEdge>,
    /// Outgoing edges from each neuron (adjacency list)
    outgoing: HashMap<NeuronId, Vec<GraphConnectionId>>,
    /// Incoming edges to each neuron
    incoming: HashMap<NeuronId, Vec<GraphConnectionId>>,
    /// Set of all neurons in the network
    neurons: HashSet<NeuronId>,
}

impl GraphNetwork {
    /// Create a new empty graph network
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
            neurons: HashSet::new(),
        }
    }
    
    /// Create with initial capacity
    pub fn with_capacity(edge_capacity: usize, neuron_capacity: usize) -> Self {
        Self {
            edges: HashMap::with_capacity(edge_capacity),
            outgoing: HashMap::with_capacity(neuron_capacity),
            incoming: HashMap::with_capacity(neuron_capacity),
            neurons: HashSet::with_capacity(neuron_capacity),
        }
    }
    
    /// Add an edge to the network
    pub fn add_edge(&mut self, edge: GraphEdge) -> Result<()> {
        let id = edge.id.clone();
        
        // Add neurons to the set
        self.neurons.insert(id.source);
        self.neurons.insert(id.target);
        
        // Add to adjacency lists
        self.outgoing.entry(id.source).or_insert_with(Vec::new).push(id.clone());
        self.incoming.entry(id.target).or_insert_with(Vec::new).push(id.clone());
        
        // Store the edge
        self.edges.insert(id, edge);
        
        Ok(())
    }
    
    /// Remove an edge from the network
    pub fn remove_edge(&mut self, connection_id: GraphConnectionId) -> Option<GraphEdge> {
        if let Some(edge) = self.edges.remove(&connection_id) {
            // Remove from adjacency lists
            if let Some(outgoing_edges) = self.outgoing.get_mut(&connection_id.source) {
                outgoing_edges.retain(|id| *id != connection_id);
                if outgoing_edges.is_empty() {
                    self.outgoing.remove(&connection_id.source);
                }
            }
            
            if let Some(incoming_edges) = self.incoming.get_mut(&connection_id.target) {
                incoming_edges.retain(|id| *id != connection_id);
                if incoming_edges.is_empty() {
                    self.incoming.remove(&connection_id.target);
                }
            }
            
            // Check if neurons should be removed (no longer connected)
            self.cleanup_isolated_neurons();
            
            Some(edge)
        } else {
            None
        }
    }
    
    /// Get an edge by connection ID
    pub fn get_edge(&self, connection_id: &GraphConnectionId) -> Option<&GraphEdge> {
        self.edges.get(connection_id)
    }
    
    /// Get a mutable edge by connection ID
    pub fn get_edge_mut(&mut self, connection_id: &GraphConnectionId) -> Option<&mut GraphEdge> {
        self.edges.get_mut(connection_id)
    }
    
    /// Get all outgoing edges from a neuron
    pub fn get_outgoing_edges(&self, neuron: NeuronId) -> Vec<&GraphEdge> {
        if let Some(edge_ids) = self.outgoing.get(&neuron) {
            edge_ids.iter()
                .filter_map(|id| self.edges.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get all incoming edges to a neuron
    pub fn get_incoming_edges(&self, neuron: NeuronId) -> Vec<&GraphEdge> {
        if let Some(edge_ids) = self.incoming.get(&neuron) {
            edge_ids.iter()
                .filter_map(|id| self.edges.get(id))
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Remove neurons that have no connections
    fn cleanup_isolated_neurons(&mut self) {
        let connected_neurons: HashSet<NeuronId> = self.edges.keys()
            .flat_map(|id| [id.source, id.target])
            .collect();
        
        self.neurons.retain(|neuron| connected_neurons.contains(neuron));
    }
    
    /// Create a fully connected network between neurons
    pub fn fully_connected(
        neurons: &[NeuronId], 
        weight: f32,
        allow_self_loops: bool
    ) -> Self {
        let mut network = Self::with_capacity(
            neurons.len() * neurons.len(),
            neurons.len()
        );
        
        for &source in neurons {
            for &target in neurons {
                if source != target || allow_self_loops {
                    let edge = GraphEdge::new(source, target, weight);
                    network.add_edge(edge).expect("Should be able to add edge");
                }
            }
        }
        
        network
    }
    
    /// Create a random network with specified connection probability
    #[cfg(feature = "std")]
    pub fn random(
        neurons: &[NeuronId],
        connection_probability: f32,
        weight_range: (f32, f32),
    ) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut network = Self::with_capacity(
            (neurons.len() * neurons.len()) / 2,
            neurons.len()
        );
        
        // Simple deterministic "random" based on neuron IDs
        for &source in neurons {
            for &target in neurons {
                if source != target {
                    let mut hasher = DefaultHasher::new();
                    (source.raw(), target.raw()).hash(&mut hasher);
                    let hash_val = hasher.finish();
                    let prob = (hash_val % 1000) as f32 / 1000.0;
                    
                    if prob < connection_probability {
                        let weight_hash = (hash_val >> 32) % 1000;
                        let weight = weight_range.0 + 
                            (weight_range.1 - weight_range.0) * (weight_hash as f32 / 1000.0);
                        
                        let edge = GraphEdge::new(source, target, weight);
                        network.add_edge(edge).expect("Should be able to add edge");
                    }
                }
            }
        }
        
        network
    }
}

impl Default for GraphNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Implementation of NetworkConnectivity for GraphNetwork
impl NetworkConnectivity<NeuronId> for GraphNetwork {
    type ConnectionId = GraphConnectionId;
    type RouteInfo = GraphRouteInfo;
    type Error = GraphConnectivityError;
    
    fn route_spike(
        &self,
        spike: &Spike,
        current_time: Time,
    ) -> Result<Vec<SpikeRoute>> {
        let mut routes = Vec::new();
        
        if let Some(edge_ids) = self.outgoing.get(&spike.source) {
            for edge_id in edge_ids {
                if let Some(edge) = self.edges.get(edge_id) {
                    if edge.active {
                        let delivery_time = current_time + crate::time::Duration::from_nanos(edge.delay.as_nanos());
                        
                        let spike_route = SpikeRoute::new(
                            edge_id.to_raw(),
                            vec![edge_id.target],
                            vec![edge.weight * spike.amplitude],
                            delivery_time,
                        ).map_err(|e| GraphConnectivityError::InvalidEdge(e.to_string()))?;
                        
                        routes.push(spike_route);
                    }
                }
            }
        }
        
        Ok(routes)
    }
    
    fn get_targets(&self, source: NeuronId) -> Result<Vec<NeuronId>> {
        if let Some(edge_ids) = self.outgoing.get(&source) {
            let targets = edge_ids.iter()
                .map(|id| id.target)
                .collect();
            Ok(targets)
        } else {
            Ok(Vec::new())
        }
    }
    
    fn get_sources(&self, target: NeuronId) -> Result<Vec<NeuronId>> {
        if let Some(edge_ids) = self.incoming.get(&target) {
            let sources = edge_ids.iter()
                .map(|id| id.source)
                .collect();
            Ok(sources)
        } else {
            Ok(Vec::new())
        }
    }
    
    fn add_connection(
        &mut self,
        connection: Self::ConnectionId,
    ) -> Result<()> {
        let edge = GraphEdge::new(connection.source, connection.target, 1.0);
        self.add_edge(edge)
    }
    
    fn remove_connection(
        &mut self,
        connection: Self::ConnectionId,
    ) -> Result<Option<Self::RouteInfo>> {
        if let Some(edge) = self.remove_edge(connection) {
            let route_info = GraphRouteInfo {
                total_weight: edge.weight,
                spike_count: 1,
                edge,
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
        if let Some(edge) = self.edges.get_mut(&connection) {
            let old_weight = edge.weight;
            edge.weight = new_weight;
            Ok(Some(old_weight))
        } else {
            Ok(None)
        }
    }
    
    fn get_stats(&self) -> ConnectivityStats {
        let connection_count = self.edges.len();
        let node_count = self.neurons.len();
        
        let total_weight: f32 = self.edges.values().map(|edge| edge.weight).sum();
        let average_weight = if connection_count > 0 {
            total_weight / connection_count as f32
        } else {
            0.0
        };
        
        let (max_degree, min_degree) = if node_count > 0 {
            let degrees: Vec<u32> = self.neurons.iter()
                .map(|&neuron| {
                    let out_degree = self.outgoing.get(&neuron).map_or(0, |v| v.len());
                    let in_degree = self.incoming.get(&neuron).map_or(0, |v| v.len());
                    (out_degree + in_degree) as u32
                })
                .collect();
            
            (*degrees.iter().max().unwrap_or(&0), *degrees.iter().min().unwrap_or(&0))
        } else {
            (0, 0)
        };
        
        let average_degree = if node_count > 0 {
            (connection_count * 2) as f32 / node_count as f32 // Each edge contributes to two degrees
        } else {
            0.0
        };
        
        let density = if node_count > 1 {
            connection_count as f32 / (node_count * (node_count - 1)) as f32
        } else {
            0.0
        };
        
        let memory_usage = core::mem::size_of::<Self>() +
            connection_count * core::mem::size_of::<GraphEdge>() +
            node_count * core::mem::size_of::<NeuronId>() * 4; // Rough estimate
        
        ConnectivityStats {
            connection_count,
            node_count,
            average_degree,
            max_degree,
            min_degree,
            total_weight,
            average_weight,
            memory_usage,
            has_cycles: None, // Would need cycle detection
            density,
            custom_stats: vec![
                ("active_edges".to_string(), self.edges.values().filter(|e| e.active).count() as f32),
                ("plastic_edges".to_string(), self.edges.values().filter(|e| e.plastic).count() as f32),
            ],
        }
    }
    
    fn validate(&self) -> Result<()> {
        // Check consistency between edges and adjacency lists
        for (id, _) in &self.edges {
            if !self.outgoing.get(&id.source).map_or(false, |edges| edges.contains(id)) {
                return Err(GraphConnectivityError::InvalidEdge(
                    format!("Edge {} missing from outgoing list", id)
                ).into());
            }
            
            if !self.incoming.get(&id.target).map_or(false, |edges| edges.contains(id)) {
                return Err(GraphConnectivityError::InvalidEdge(
                    format!("Edge {} missing from incoming list", id)
                ).into());
            }
        }
        
        Ok(())
    }
    
    fn reset(&mut self) {
        self.edges.clear();
        self.outgoing.clear();
        self.incoming.clear();
        self.neurons.clear();
    }
    
    fn connection_count(&self) -> usize {
        self.edges.len()
    }
    
    fn neuron_count(&self) -> usize {
        self.neurons.len()
    }
}

/// Batch operations for graph connectivity
impl BatchConnectivity<NeuronId> for GraphNetwork {
    fn add_connections<I>(
        &mut self,
        connections: I,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = Self::ConnectionId>,
    {
        let mut added_count = 0;
        
        for connection in connections {
            let edge = GraphEdge::new(connection.source, connection.target, 1.0);
            self.add_edge(edge)?;
            added_count += 1;
        }
        
        Ok(added_count)
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
            if self.remove_edge(connection).is_some() {
                removed_count += 1;
            }
        }
        
        Ok(removed_count)
    }
}

/// Plasticity support for graph connectivity
impl PlasticConnectivity<NeuronId> for GraphNetwork {
    fn apply_plasticity(
        &mut self,
        pre_neuron: NeuronId,
        post_neuron: NeuronId,
        weight_delta: f32,
    ) -> Result<Option<f32>> {
        let connection_id = GraphConnectionId {
            source: pre_neuron,
            target: post_neuron,
        };
        
        if let Some(edge) = self.edges.get_mut(&connection_id) {
            if edge.plastic {
                edge.weight = (edge.weight + weight_delta).max(0.0).min(10.0); // Clamp weights
                Ok(Some(edge.weight))
            } else {
                Ok(Some(edge.weight)) // Return current weight but don't change it
            }
        } else {
            Ok(None)
        }
    }
    
    fn get_weight(
        &self,
        pre_neuron: NeuronId,
        post_neuron: NeuronId,
    ) -> Result<Option<f32>> {
        let connection_id = GraphConnectionId {
            source: pre_neuron,
            target: post_neuron,
        };
        
        Ok(self.edges.get(&connection_id).map(|edge| edge.weight))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_network_basic() {
        let mut network = GraphNetwork::new();
        
        let edge = GraphEdge::new(NeuronId::new(0), NeuronId::new(1), 0.5);
        network.add_edge(edge).expect("Should add edge");
        
        assert_eq!(network.connection_count(), 1);
        assert_eq!(network.neuron_count(), 2);
        
        let targets = network.get_targets(NeuronId::new(0)).expect("Should get targets");
        assert_eq!(targets, vec![NeuronId::new(1)]);
        
        let sources = network.get_sources(NeuronId::new(1)).expect("Should get sources");
        assert_eq!(sources, vec![NeuronId::new(0)]);
    }
    
    #[test]
    fn test_spike_routing() {
        let mut network = GraphNetwork::new();
        
        let edge = GraphEdge::new(NeuronId::new(0), NeuronId::new(1), 0.8);
        network.add_edge(edge).expect("Should add edge");
        
        let spike = Spike::new(NeuronId::new(0), Time::from_millis(10), 1.0)
            .expect("Should create spike");
        
        let routes = network.route_spike(&spike, Time::from_millis(10))
            .expect("Should route spike");
        
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].targets, vec![NeuronId::new(1)]);
        assert_eq!(routes[0].weights, vec![0.8]); // weight * amplitude
    }
    
    #[test]
    fn test_connection_id_serialization() {
        let connection_id = GraphConnectionId {
            source: NeuronId::new(42),
            target: NeuronId::new(84),
        };
        
        let raw = connection_id.to_raw();
        let reconstructed = GraphConnectionId::from_raw(&raw)
            .expect("Should reconstruct from raw");
        
        assert_eq!(connection_id, reconstructed);
        assert_eq!(connection_id.display_string(), "N42→N84");
    }
    
    #[test]
    fn test_plasticity() {
        let mut network = GraphNetwork::new();
        
        let edge = GraphEdge::new(NeuronId::new(0), NeuronId::new(1), 0.5);
        network.add_edge(edge).expect("Should add edge");
        
        let weight = network.get_weight(NeuronId::new(0), NeuronId::new(1))
            .expect("Should get weight");
        assert_eq!(weight, Some(0.5));
        
        let new_weight = network.apply_plasticity(
            NeuronId::new(0),
            NeuronId::new(1),
            0.2,
        ).expect("Should apply plasticity");
        assert_eq!(new_weight, Some(0.7));
    }
    
    #[test]
    fn test_fully_connected_network() {
        let neurons = vec![NeuronId::new(0), NeuronId::new(1), NeuronId::new(2)];
        let network = GraphNetwork::fully_connected(&neurons, 1.0, false);
        
        assert_eq!(network.connection_count(), 6); // 3 * (3-1) = 6 directed edges
        assert_eq!(network.neuron_count(), 3);
    }

    // Phase 6 tests: snapshot/apply on GraphNetwork
    #[test]
    fn test_graph_snapshot_and_apply() {
        use crate::connectivity::WeightSnapshotConnectivity;
        let mut net = GraphNetwork::new();
        net.add_edge(GraphEdge::new(NeuronId::new(0), NeuronId::new(1), 0.5)).unwrap();

        let snap = <GraphNetwork as WeightSnapshotConnectivity<NeuronId>>::snapshot_weights(&net);
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].0, NeuronId::new(0));
        assert_eq!(snap[0].1, NeuronId::new(1));
        assert!((snap[0].2 - 0.5).abs() < 1e-6);

        let updates = [(NeuronId::new(0), NeuronId::new(1), 0.9)];
        let applied = <GraphNetwork as WeightSnapshotConnectivity<NeuronId>>::apply_weight_updates(&mut net, &updates).unwrap();
        assert_eq!(applied, 1);

        let new_w = net.get_weight(NeuronId::new(0), NeuronId::new(1)).unwrap();
        assert_eq!(new_w, Some(0.9));
    }
}
// Phase 6: weight snapshot/apply for GraphNetwork
impl crate::connectivity::WeightSnapshotConnectivity<NeuronId> for GraphNetwork {
    fn snapshot_weights(&self) -> Vec<(NeuronId, NeuronId, f32)> {
        let mut out = Vec::with_capacity(self.edges.len());
        for (id, edge) in self.edges.iter() {
            out.push((id.source, id.target, edge.weight));
        }
        out
    }

    fn apply_weight_updates(&mut self, updates: &[(NeuronId, NeuronId, f32)]) -> Result<usize> {
        let mut applied = 0usize;
        for &(pre, post, w) in updates.iter() {
            let id = GraphConnectionId { source: pre, target: post };
            if let Some(edge) = self.edges.get_mut(&id) {
                edge.weight = w.max(0.0).min(10.0);
                applied += 1;
            }
        }
        Ok(applied)
    }
}