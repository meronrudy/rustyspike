//! Sparse matrix-based connectivity implementation
//!
//! This module implements neural connectivity using sparse matrices,
//! optimized for large networks with low connection density.

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

#[cfg(feature = "math")]
use shnn_math::{SparseMatrix, Float};

#[cfg(not(feature = "math"))]
use crate::math::{SparseMatrix, Float};

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use heapless::FnvIndexMap as HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Type alias for compatibility
#[cfg(not(feature = "math"))]
type Float = f32;

/// Connection identifier for sparse matrix entries
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseConnectionId {
    /// Row index (source neuron index)
    pub row: usize,
    /// Column index (target neuron index)
    pub col: usize,
}

impl ConnectionId for SparseConnectionId {
    fn from_raw(data: &[u8]) -> core::result::Result<Self, &'static str> {
        if data.len() != 16 {
            return Err("Sparse connection ID requires exactly 16 bytes");
        }
        
        let row_bytes = [
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ];
        let col_bytes = [
            data[8], data[9], data[10], data[11],
            data[12], data[13], data[14], data[15],
        ];
        
        let row = usize::from_le_bytes(row_bytes);
        let col = usize::from_le_bytes(col_bytes);
        
        Ok(Self { row, col })
    }
    
    fn to_raw(&self) -> Vec<u8> {
        let mut raw = Vec::with_capacity(16);
        raw.extend_from_slice(&self.row.to_le_bytes());
        raw.extend_from_slice(&self.col.to_le_bytes());
        raw
    }
    
    fn display_string(&self) -> String {
        format!("S[{},{}]", self.row, self.col)
    }
}

impl fmt::Display for SparseConnectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_string())
    }
}

impl From<(usize, usize)> for SparseConnectionId {
    fn from((row, col): (usize, usize)) -> Self {
        Self { row, col }
    }
}

/// Route information specific to sparse matrix connections
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseRouteInfo {
    /// Source neuron index
    pub source_index: usize,
    /// Target neurons reached
    pub target_indices: Vec<usize>,
    /// Weights transmitted
    pub weights: Vec<f32>,
    /// Sparsity ratio
    pub sparsity: f32,
}

/// Error type for sparse matrix connectivity operations
#[derive(Debug, Clone, PartialEq)]
pub enum SparseConnectivityError {
    /// Wraps an SHNN error
    Core(SHNNError),
    /// Connectivity-specific error
    Connectivity(ConnectivityError),
    /// Index out of bounds
    IndexOutOfBounds {
        /// The index that was out of bounds
        index: usize,
        /// The boundary that was exceeded
        bound: usize
    },
    /// Invalid neuron ID
    InvalidNeuronId(NeuronId),
    /// Sparse matrix operation failed
    SparseMatrixError(String),
    /// Memory allocation failed
    MemoryError(String),
}

impl fmt::Display for SparseConnectivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => write!(f, "Core error: {}", err),
            Self::Connectivity(err) => write!(f, "Connectivity error: {}", err),
            Self::IndexOutOfBounds { index, bound } => {
                write!(f, "Index {} out of bounds (max: {})", index, bound)
            }
            Self::InvalidNeuronId(id) => write!(f, "Invalid neuron ID: {}", id),
            Self::SparseMatrixError(msg) => write!(f, "Sparse matrix error: {}", msg),
            Self::MemoryError(msg) => write!(f, "Memory error: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SparseConnectivityError {}

impl From<SHNNError> for SparseConnectivityError {
    fn from(err: SHNNError) -> Self {
        Self::Core(err)
    }
}

impl From<ConnectivityError> for SparseConnectivityError {
    fn from(err: ConnectivityError) -> Self {
        Self::Connectivity(err)
    }
}

/// Sparse matrix-based neural network connectivity
/// 
/// This implementation uses a sparse matrix to represent connections
/// between neurons. It's optimal for large networks with low connection
/// density, providing memory-efficient storage for sparse connectivity patterns.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseMatrixNetwork {
    /// Sparse adjacency matrix
    adjacency_matrix: SparseMatrix,
    /// Delay information stored separately for sparse access
    delays: HashMap<(usize, usize), Time>,
    /// Maximum number of neurons supported
    max_neurons: usize,
    /// Mapping from NeuronId to matrix index
    neuron_id_to_index: HashMap<NeuronId, usize>,
    /// Mapping from matrix index to NeuronId
    index_to_neuron_id: Vec<Option<NeuronId>>,
    /// Number of active neurons
    neuron_count: usize,
    /// Next available index
    next_index: usize,
}

impl SparseMatrixNetwork {
    /// Create a new sparse matrix network
    pub fn new(max_neurons: usize) -> Self {
        Self {
            adjacency_matrix: SparseMatrix::new(max_neurons, max_neurons),
            delays: HashMap::new(),
            max_neurons,
            neuron_id_to_index: HashMap::new(),
            index_to_neuron_id: vec![None; max_neurons],
            neuron_count: 0,
            next_index: 0,
        }
    }
    
    /// Create with estimated capacity for non-zero entries
    pub fn with_capacity(max_neurons: usize, estimated_connections: usize) -> Self {
        Self {
            adjacency_matrix: SparseMatrix::with_capacity(max_neurons, max_neurons, estimated_connections),
            delays: HashMap::with_capacity(estimated_connections),
            max_neurons,
            neuron_id_to_index: HashMap::with_capacity(max_neurons),
            index_to_neuron_id: vec![None; max_neurons],
            neuron_count: 0,
            next_index: 0,
        }
    }
    
    /// Create a random sparse network
    pub fn random_sparse(
        neurons: &[NeuronId],
        connection_probability: f32,
        weight_range: (f32, f32),
    ) -> Result<Self> {
        let size = neurons.len();
        let estimated_connections = (size * size) as f32 * connection_probability;
        let mut network = Self::with_capacity(size, estimated_connections as usize);
        
        // Add all neurons
        for &neuron_id in neurons {
            network.add_neuron(neuron_id)?;
        }
        
        // Create sparse random connections
        for &source in neurons {
            for &target in neurons {
                if source != target {
                    // Simple deterministic "random" based on neuron IDs
                    let hash_input = (source.raw() as u64) << 32 | (target.raw() as u64);
                    let hash_val = hash_input.wrapping_mul(0x9e3779b97f4a7c15_u64);
                    let prob = (hash_val % 1000) as f32 / 1000.0;
                    
                    if prob < connection_probability {
                        let weight_hash = (hash_val >> 32) % 1000;
                        let weight = weight_range.0 + 
                            (weight_range.1 - weight_range.0) * (weight_hash as f32 / 1000.0);
                        
                        network.set_weight(source, target, weight)?;
                    }
                }
            }
        }
        
        Ok(network)
    }
    
    /// Add a neuron to the network
    pub fn add_neuron(&mut self, neuron_id: NeuronId) -> Result<usize> {
        if let Some(&existing_index) = self.neuron_id_to_index.get(&neuron_id) {
            return Ok(existing_index);
        }
        
        if self.next_index >= self.max_neurons {
            return Err(SparseConnectivityError::IndexOutOfBounds {
                index: self.next_index,
                bound: self.max_neurons,
            }.into());
        }
        
        let index = self.next_index;
        self.neuron_id_to_index.insert(neuron_id, index);
        self.index_to_neuron_id[index] = Some(neuron_id);
        self.next_index += 1;
        self.neuron_count += 1;
        
        Ok(index)
    }
    
    /// Get the matrix index for a neuron ID
    pub fn get_neuron_index(&self, neuron_id: NeuronId) -> Option<usize> {
        self.neuron_id_to_index.get(&neuron_id).copied()
    }
    
    /// Get the neuron ID for a matrix index
    pub fn get_neuron_id(&self, index: usize) -> Option<NeuronId> {
        if index < self.index_to_neuron_id.len() {
            self.index_to_neuron_id[index]
        } else {
            None
        }
    }
    
    /// Set the weight between two neurons
    pub fn set_weight(
        &mut self,
        source: NeuronId,
        target: NeuronId,
        weight: f32,
    ) -> Result<()> {
        let source_idx = self.get_neuron_index(source)
            .ok_or(SparseConnectivityError::InvalidNeuronId(source))?;
        let target_idx = self.get_neuron_index(target)
            .ok_or(SparseConnectivityError::InvalidNeuronId(target))?;
        
        self.adjacency_matrix.set(source_idx, target_idx, weight as Float)
            .map_err(|e| SparseConnectivityError::SparseMatrixError(format!("{:?}", e)))?;
        
        Ok(())
    }
    
    /// Get the weight between two neurons
    pub fn get_weight(
        &self, 
        source: NeuronId, 
        target: NeuronId
    ) -> Result<Option<f32>> {
        let source_idx = self.get_neuron_index(source)
            .ok_or(SparseConnectivityError::InvalidNeuronId(source))?;
        let target_idx = self.get_neuron_index(target)
            .ok_or(SparseConnectivityError::InvalidNeuronId(target))?;
        
        match self.adjacency_matrix.get(source_idx, target_idx) {
            Ok(weight) => Ok(Some(weight as f32)),
            Err(_) => Ok(None), // Element not found means no connection
        }
    }
    
    /// Set the delay between two neurons
    pub fn set_delay(
        &mut self,
        source: NeuronId,
        target: NeuronId,
        delay: Time,
    ) -> Result<()> {
        let source_idx = self.get_neuron_index(source)
            .ok_or(SparseConnectivityError::InvalidNeuronId(source))?;
        let target_idx = self.get_neuron_index(target)
            .ok_or(SparseConnectivityError::InvalidNeuronId(target))?;
        
        self.delays.insert((source_idx, target_idx), delay);
        Ok(())
    }
    
    /// Get the delay between two neurons
    pub fn get_delay(
        &self, 
        source: NeuronId, 
        target: NeuronId
    ) -> Result<Time> {
        let source_idx = self.get_neuron_index(source)
            .ok_or(SparseConnectivityError::InvalidNeuronId(source))?;
        let target_idx = self.get_neuron_index(target)
            .ok_or(SparseConnectivityError::InvalidNeuronId(target))?;
        
        Ok(self.delays.get(&(source_idx, target_idx)).copied().unwrap_or(Time::ZERO))
    }
    
    /// Get sparsity ratio (fraction of zero entries)
    pub fn sparsity(&self) -> f32 {
        let total_possible = self.max_neurons * self.max_neurons;
        let non_zero_count = self.adjacency_matrix.nnz();
        
        if total_possible > 0 {
            1.0 - (non_zero_count as f32 / total_possible as f32)
        } else {
            1.0
        }
    }
    
    /// Get number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.adjacency_matrix.nnz()
    }
    
    /// Get maximum capacity
    pub fn capacity(&self) -> usize {
        self.max_neurons
    }
    
    /// Get all outgoing connections from a neuron
    pub fn get_outgoing_connections(&self, neuron: NeuronId) -> Result<Vec<(NeuronId, f32)>> {
        let source_idx = self.get_neuron_index(neuron)
            .ok_or(SparseConnectivityError::InvalidNeuronId(neuron))?;
        
        let mut connections = Vec::new();
        
        // Iterate through the sparse row
        for (col_idx, weight) in self.adjacency_matrix.row_iter(source_idx) {
            if let Some(target_neuron) = self.get_neuron_id(col_idx) {
                connections.push((target_neuron, weight as f32));
            }
        }
        
        Ok(connections)
    }
    
    /// Get all incoming connections to a neuron
    pub fn get_incoming_connections(&self, neuron: NeuronId) -> Result<Vec<(NeuronId, f32)>> {
        let target_idx = self.get_neuron_index(neuron)
            .ok_or(SparseConnectivityError::InvalidNeuronId(neuron))?;
        
        let mut connections = Vec::new();
        
        // Iterate through the sparse column
        for (row_idx, weight) in self.adjacency_matrix.col_iter(target_idx) {
            if let Some(source_neuron) = self.get_neuron_id(row_idx) {
                connections.push((source_neuron, weight as f32));
            }
        }
        
        Ok(connections)
    }
}

impl Default for SparseMatrixNetwork {
    fn default() -> Self {
        Self::new(10000) // Default capacity of 10,000 neurons
    }
}

// Phase 6: weight snapshot/apply for SparseMatrixNetwork
impl crate::connectivity::WeightSnapshotConnectivity<NeuronId> for SparseMatrixNetwork {
    fn snapshot_weights(&self) -> Vec<(NeuronId, NeuronId, f32)> {
        let mut out = Vec::new();
        for row in 0..self.max_neurons {
            // Skip rows for non-existent neuron ids
            let pre = match self.get_neuron_id(row) {
                Some(id) => id,
                None => continue,
            };
            for (col, w) in self.adjacency_matrix.row_iter(row) {
                if let Some(post) = self.get_neuron_id(col) {
                    if w != 0.0 {
                        out.push((pre, post, w as f32));
                    }
                }
            }
        }
        out
    }

    fn apply_weight_updates(&mut self, updates: &[(NeuronId, NeuronId, f32)]) -> Result<usize> {
        let mut applied = 0usize;
        for &(pre, post, w) in updates.iter() {
            // Only apply if both neurons exist in current mapping
            if self.get_neuron_index(pre).is_some() && self.get_neuron_index(post).is_some() {
                let wclamp = w.max(0.0).min(10.0);
                self.set_weight(pre, post, wclamp)?;
                applied += 1;
            }
        }
        Ok(applied)
    }
}

/// Implementation of NetworkConnectivity for SparseMatrixNetwork
impl NetworkConnectivity<NeuronId> for SparseMatrixNetwork {
    type ConnectionId = SparseConnectionId;
    type RouteInfo = SparseRouteInfo;
    type Error = SparseConnectivityError;
    
    fn route_spike(
        &self,
        spike: &Spike,
        current_time: Time,
    ) -> Result<Vec<SpikeRoute>> {
        let source_idx = self.get_neuron_index(spike.source)
            .ok_or(SparseConnectivityError::InvalidNeuronId(spike.source))?;
        
        let mut routes = Vec::new();
        
        // Iterate through sparse row to find targets
        for (target_idx, weight) in self.adjacency_matrix.row_iter(source_idx) {
            let target_neuron_id = self.get_neuron_id(target_idx)
                .ok_or(SparseConnectivityError::IndexOutOfBounds {
                    index: target_idx,
                    bound: self.max_neurons,
                })?;
            
            let delay = self.delays.get(&(source_idx, target_idx))
                .copied()
                .unwrap_or(Time::ZERO);
            
            let delivery_time = current_time + crate::time::Duration::from_nanos(delay.as_nanos());
            let connection_id = SparseConnectionId { row: source_idx, col: target_idx };
            
            let spike_route = SpikeRoute::new(
                connection_id.to_raw(),
                vec![target_neuron_id],
                vec![(weight as f32) * spike.amplitude],
                delivery_time,
            ).map_err(|e| SparseConnectivityError::SparseMatrixError(e.to_string()))?;
            
            routes.push(spike_route);
        }
        
        Ok(routes)
    }
    
    fn get_targets(&self, source: NeuronId) -> Result<Vec<NeuronId>> {
        let connections = self.get_outgoing_connections(source)?;
        Ok(connections.into_iter().map(|(neuron, _)| neuron).collect())
    }
    
    fn get_sources(&self, target: NeuronId) -> Result<Vec<NeuronId>> {
        let connections = self.get_incoming_connections(target)?;
        Ok(connections.into_iter().map(|(neuron, _)| neuron).collect())
    }
    
    fn add_connection(
        &mut self,
        connection: Self::ConnectionId,
    ) -> Result<()> {
        if connection.row >= self.neuron_count || connection.col >= self.neuron_count {
            return Err(SparseConnectivityError::IndexOutOfBounds {
                index: connection.row.max(connection.col),
                bound: self.neuron_count,
            }.into());
        }
        
        self.adjacency_matrix.set(connection.row, connection.col, 1.0 as Float)
            .map_err(|e| SparseConnectivityError::SparseMatrixError(format!("{:?}", e)))?;
        
        Ok(())
    }
    
    fn remove_connection(
        &mut self,
        connection: Self::ConnectionId,
    ) -> Result<Option<Self::RouteInfo>> {
        if connection.row >= self.neuron_count || connection.col >= self.neuron_count {
            return Ok(None);
        }
        
        let weight = match self.adjacency_matrix.get(connection.row, connection.col) {
            Ok(weight) => weight as f32,
            Err(_) => return Ok(None), // Element not found means no connection
        };
        
        // Remove the entry by setting it to zero
        self.adjacency_matrix.set(connection.row, connection.col, 0.0 as Float)
            .map_err(|e| SparseConnectivityError::SparseMatrixError(format!("{:?}", e)))?;
        
        // Remove delay if present
        self.delays.remove(&(connection.row, connection.col));
        
        let route_info = SparseRouteInfo {
            source_index: connection.row,
            target_indices: vec![connection.col],
            weights: vec![weight],
            sparsity: self.sparsity(),
        };
        
        Ok(Some(route_info))
    }
    
    fn update_weight(
        &mut self,
        connection: Self::ConnectionId,
        new_weight: f32,
    ) -> Result<Option<f32>> {
        if connection.row >= self.neuron_count || connection.col >= self.neuron_count {
            return Ok(None);
        }
        
        let old_weight = match self.adjacency_matrix.get(connection.row, connection.col) {
            Ok(weight) => Some(weight as f32),
            Err(_) => None, // Element not found means no connection
        };
        
        self.adjacency_matrix.set(connection.row, connection.col, new_weight as Float)
            .map_err(|e| SparseConnectivityError::SparseMatrixError(format!("{:?}", e)))?;
        
        Ok(old_weight)
    }
    
    fn get_stats(&self) -> ConnectivityStats {
        let connection_count = self.nnz();
        let node_count = self.neuron_count;
        
        // Calculate statistics efficiently using sparse structure
        let mut total_weight = 0.0f32;
        let mut degree_counts = vec![0u32; node_count];
        
        // Count connections and weights efficiently
        for i in 0..node_count {
            for (j, weight) in self.adjacency_matrix.row_iter(i) {
                    total_weight += weight as f32;
                    degree_counts[i] += 1; // Out-degree
                    if j < degree_counts.len() {
                        degree_counts[j] += 1; // In-degree
                    }
                }
            }
        
        let average_weight = if connection_count > 0 {
            total_weight / connection_count as f32
        } else {
            0.0
        };
        
        let (max_degree, min_degree) = if !degree_counts.is_empty() {
            (*degree_counts.iter().max().unwrap_or(&0), *degree_counts.iter().min().unwrap_or(&0))
        } else {
            (0, 0)
        };
        
        let average_degree = if node_count > 0 {
            degree_counts.iter().sum::<u32>() as f32 / node_count as f32
        } else {
            0.0
        };
        
        let density = if node_count > 1 {
            connection_count as f32 / (node_count * (node_count - 1)) as f32
        } else {
            0.0
        };
        
        let memory_usage = core::mem::size_of::<Self>() +
            self.adjacency_matrix.memory_usage() +
            self.delays.len() * (core::mem::size_of::<(usize, usize)>() + core::mem::size_of::<Time>()) +
            self.neuron_id_to_index.len() * (core::mem::size_of::<NeuronId>() + core::mem::size_of::<usize>()) +
            self.index_to_neuron_id.len() * core::mem::size_of::<Option<NeuronId>>();
        
        ConnectivityStats {
            connection_count,
            node_count,
            average_degree,
            max_degree,
            min_degree,
            total_weight,
            average_weight,
            memory_usage,
            has_cycles: None, // Would require cycle detection
            density,
            custom_stats: vec![
                ("sparsity".to_string(), self.sparsity()),
                ("nnz".to_string(), self.nnz() as f32),
                ("capacity".to_string(), self.capacity() as f32),
                ("memory_efficiency".to_string(), 
                 if self.capacity() > 0 { 
                     memory_usage as f32 / (self.capacity() * self.capacity() * core::mem::size_of::<f32>()) as f32 
                 } else { 
                     0.0 
                 }),
            ],
        }
    }
    
    fn validate(&self) -> Result<()> {
        // Check neuron count consistency
        if self.neuron_count > self.capacity() {
            return Err(SparseConnectivityError::IndexOutOfBounds {
                index: self.neuron_count,
                bound: self.capacity(),
            }.into());
        }
        
        // Validate sparse matrix dimensions
        let (rows, cols) = self.adjacency_matrix.dims();
        if rows != self.max_neurons || cols != self.max_neurons {
            return Err(SparseConnectivityError::SparseMatrixError(
                format!("Matrix dimensions {}x{} don't match capacity {}", rows, cols, self.max_neurons)
            ).into());
        }
        
        Ok(())
    }
    
    fn reset(&mut self) {
        self.adjacency_matrix = SparseMatrix::new(self.max_neurons, self.max_neurons);
        self.delays.clear();
        self.neuron_id_to_index.clear();
        self.index_to_neuron_id.iter_mut().for_each(|entry| *entry = None);
        self.neuron_count = 0;
        self.next_index = 0;
    }
    
    fn connection_count(&self) -> usize {
        self.nnz()
    }
    
    fn neuron_count(&self) -> usize {
        self.neuron_count
    }
}

/// Batch operations for sparse matrix connectivity
impl BatchConnectivity<NeuronId> for SparseMatrixNetwork {
    fn add_connections<I>(
        &mut self,
        connections: I,
    ) -> Result<usize>
    where
        I: IntoIterator<Item = Self::ConnectionId>,
    {
        let mut added_count = 0;
        
        for connection in connections {
            self.add_connection(connection)?;
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
            if self.remove_connection(connection)?.is_some() {
                removed_count += 1;
            }
        }
        
        Ok(removed_count)
    }
}

/// Plasticity support for sparse matrix connectivity
impl PlasticConnectivity<NeuronId> for SparseMatrixNetwork {
    fn apply_plasticity(
        &mut self,
        pre_neuron: NeuronId,
        post_neuron: NeuronId,
        weight_delta: f32,
    ) -> Result<Option<f32>> {
        match self.get_weight(pre_neuron, post_neuron)? {
            Some(current_weight) => {
                let new_weight = (current_weight + weight_delta).max(0.0).min(10.0);
                self.set_weight(pre_neuron, post_neuron, new_weight)?;
                Ok(Some(new_weight))
            }
            None => {
                // Create new connection if weight delta is positive
                if weight_delta > 0.0 {
                    let new_weight = weight_delta.max(0.0).min(10.0);
                    self.set_weight(pre_neuron, post_neuron, new_weight)?;
                    Ok(Some(new_weight))
                } else {
                    Ok(None)
                }
            }
        }
    }
    
    fn get_weight(
        &self,
        pre_neuron: NeuronId,
        post_neuron: NeuronId,
    ) -> Result<Option<f32>> {
        // Avoid recursive dispatch by directly querying adjacency
        let source_idx = self.get_neuron_index(pre_neuron)
            .ok_or(SparseConnectivityError::InvalidNeuronId(pre_neuron))?;
        let target_idx = self.get_neuron_index(post_neuron)
            .ok_or(SparseConnectivityError::InvalidNeuronId(post_neuron))?;
        match self.adjacency_matrix.get(source_idx, target_idx) {
            Ok(weight) => Ok(Some(weight as f32)),
            Err(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectivity::WeightSnapshotConnectivity;

    #[test]
    fn test_sparse_network_basic() {
        let mut network = SparseMatrixNetwork::new(100);
        
        let neuron1 = NeuronId::new(1);
        let neuron2 = NeuronId::new(2);
        
        network.add_neuron(neuron1).expect("Should add neuron");
        network.add_neuron(neuron2).expect("Should add neuron");
        
        network.set_weight(neuron1, neuron2, 0.5).expect("Should set weight");
        
        assert_eq!(network.neuron_count(), 2);
        assert_eq!(network.connection_count(), 1);
        
        let weight = network.get_weight(neuron1, neuron2).expect("Should get weight");
        assert_eq!(weight, Some(0.5));
        
        assert!(network.sparsity() > 0.99); // Very sparse
    }
    
    #[test]
    fn test_sparse_spike_routing() {
        let mut network = SparseMatrixNetwork::new(100);
        
        let source = NeuronId::new(0);
        let target = NeuronId::new(1);
        
        network.add_neuron(source).expect("Should add neuron");
        network.add_neuron(target).expect("Should add neuron");
        network.set_weight(source, target, 0.8).expect("Should set weight");
        
        let spike = Spike::new(source, Time::from_millis(10), 1.0)
            .expect("Should create spike");
        
        let routes = network.route_spike(&spike, Time::from_millis(10))
            .expect("Should route spike");
        
        assert_eq!(routes.len(), 1);
        assert_eq!(routes[0].targets, vec![target]);
        assert_eq!(routes[0].weights, vec![0.8]);
    }
    
    #[test]
    fn test_sparse_connection_id() {
        let connection_id = SparseConnectionId { row: 42, col: 84 };
        
        let raw = connection_id.to_raw();
        let reconstructed = SparseConnectionId::from_raw(&raw)
            .expect("Should reconstruct from raw");
        
        assert_eq!(connection_id, reconstructed);
        assert_eq!(connection_id.display_string(), "S[42,84]");
    }
    
    #[test]
    fn test_sparse_plasticity() {
        let mut network = SparseMatrixNetwork::new(100);
        
        let source = NeuronId::new(0);
        let target = NeuronId::new(1);
        
        network.add_neuron(source).expect("Should add neuron");
        network.add_neuron(target).expect("Should add neuron");
        network.set_weight(source, target, 0.5).expect("Should set weight");
        
        let weight = network.get_weight(source, target)
            .expect("Should get weight");
        assert_eq!(weight, Some(0.5));
        
        let new_weight = network.apply_plasticity(source, target, 0.2)
            .expect("Should apply plasticity");
        assert_eq!(new_weight, Some(0.7));
    }
    
    #[test]
    fn test_sparse_memory_efficiency() {
        let dense_network = crate::connectivity::matrix::MatrixNetwork::new(1000);
        let sparse_network = SparseMatrixNetwork::new(1000);
        
        let dense_stats = dense_network.get_stats();
        let sparse_stats = sparse_network.get_stats();
        
        // Sparse should use much less memory for empty networks
        assert!(sparse_stats.memory_usage < dense_stats.memory_usage);
    }
}