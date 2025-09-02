//! Dense matrix-based connectivity implementation
//!
//! This module implements neural connectivity using dense adjacency matrices,
//! optimized for fully connected or near-fully connected neural networks.

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
use shnn_math::{Matrix, Float};

#[cfg(not(feature = "math"))]
use crate::math::{Matrix, Float};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Type alias for compatibility
#[cfg(not(feature = "math"))]
type Float = f32;

/// Connection identifier for matrix entries
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MatrixConnectionId {
    /// Row index (source neuron index)
    pub row: usize,
    /// Column index (target neuron index)  
    pub col: usize,
}

impl ConnectionId for MatrixConnectionId {
    fn from_raw(data: &[u8]) -> core::result::Result<Self, &'static str> {
        if data.len() != 16 {
            return Err("Matrix connection ID requires exactly 16 bytes");
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
        format!("M[{},{}]", self.row, self.col)
    }
}

impl fmt::Display for MatrixConnectionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.display_string())
    }
}

impl From<(usize, usize)> for MatrixConnectionId {
    fn from((row, col): (usize, usize)) -> Self {
        Self { row, col }
    }
}

/// Route information specific to matrix connections
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MatrixRouteInfo {
    /// Source neuron index
    pub source_index: usize,
    /// Target neurons reached
    pub target_indices: Vec<usize>,
    /// Weights transmitted
    pub weights: Vec<f32>,
    /// Total activation transmitted
    pub total_activation: f32,
}

/// Error type for matrix connectivity operations
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixConnectivityError {
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
    /// Dimension mismatch
    DimensionMismatch {
        /// Expected dimension
        expected: usize,
        /// Actual dimension received
        got: usize
    },
    /// Invalid neuron ID
    InvalidNeuronId(NeuronId),
    /// Matrix operation failed
    MatrixError(String),
}

impl fmt::Display for MatrixConnectivityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => write!(f, "Core error: {}", err),
            Self::Connectivity(err) => write!(f, "Connectivity error: {}", err),
            Self::IndexOutOfBounds { index, bound } => {
                write!(f, "Index {} out of bounds (max: {})", index, bound)
            }
            Self::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            Self::InvalidNeuronId(id) => write!(f, "Invalid neuron ID: {}", id),
            Self::MatrixError(msg) => write!(f, "Matrix error: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for MatrixConnectivityError {}

impl From<SHNNError> for MatrixConnectivityError {
    fn from(err: SHNNError) -> Self {
        Self::Core(err)
    }
}

impl From<ConnectivityError> for MatrixConnectivityError {
    fn from(err: ConnectivityError) -> Self {
        Self::Connectivity(err)
    }
}

/// Dense matrix-based neural network connectivity
/// 
/// This implementation uses a dense adjacency matrix to represent connections
/// between neurons. It's optimal for fully connected or near-fully connected
/// networks but may be memory-inefficient for sparse networks.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MatrixNetwork {
    /// Adjacency matrix where entry (i,j) represents weight from neuron i to neuron j
    adjacency_matrix: Matrix,
    /// Delay matrix for transmission delays (optional)
    delay_matrix: Option<Matrix>,
    /// Number of neurons in the network
    neuron_count: usize,
    /// Mapping from NeuronId to matrix index
    neuron_id_to_index: Vec<Option<NeuronId>>,
    /// Mapping from matrix index to NeuronId  
    index_to_neuron_id: Vec<NeuronId>,
    /// Next available index for new neurons
    next_index: usize,
}

impl MatrixNetwork {
    /// Create a new matrix network with specified capacity
    pub fn new(max_neurons: usize) -> Self {
        Self {
            adjacency_matrix: Matrix::zeros(max_neurons, max_neurons),
            delay_matrix: None,
            neuron_count: 0,
            neuron_id_to_index: vec![None; max_neurons],
            index_to_neuron_id: Vec::with_capacity(max_neurons),
            next_index: 0,
        }
    }
    
    /// Create with delays enabled
    pub fn with_delays(max_neurons: usize) -> Self {
        Self {
            adjacency_matrix: Matrix::zeros(max_neurons, max_neurons),
            delay_matrix: Some(Matrix::zeros(max_neurons, max_neurons)),
            neuron_count: 0,
            neuron_id_to_index: vec![None; max_neurons],
            index_to_neuron_id: Vec::with_capacity(max_neurons),
            next_index: 0,
        }
    }
    
    /// Create a fully connected network
    pub fn fully_connected(neurons: &[NeuronId], weight: f32) -> Result<Self> {
        let size = neurons.len();
        let mut network = Self::new(size);
        
        // Add all neurons
        for &neuron_id in neurons {
            network.add_neuron(neuron_id)?;
        }
        
        // Connect all pairs
        for &source in neurons {
            for &target in neurons {
                if source != target {
                    network.set_weight(source, target, weight)?;
                }
            }
        }
        
        Ok(network)
    }
    
    /// Create a random network with given connection probability
    pub fn random(
        neurons: &[NeuronId],
        connection_probability: f32,
        weight_range: (f32, f32),
    ) -> Result<Self> {
        let size = neurons.len();
        let mut network = Self::new(size);
        
        // Add all neurons
        for &neuron_id in neurons {
            network.add_neuron(neuron_id)?;
        }
        
        // Create random connections using deterministic hash-based approach
        for &source in neurons {
            for &target in neurons {
                if source != target {
                    // Simple deterministic "random" based on neuron IDs
                    let hash_input = (source.raw() as u64) << 32 | (target.raw() as u64);
                    let hash_val = hash_input.wrapping_mul(0x9e3779b97f4a7c15_u64); // Simple hash
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
        if self.next_index >= self.neuron_id_to_index.len() {
            return Err(MatrixConnectivityError::IndexOutOfBounds {
                index: self.next_index,
                bound: self.neuron_id_to_index.len(),
            }.into());
        }
        
        // Check if neuron already exists
        if let Some(existing_index) = self.get_neuron_index(neuron_id) {
            return Ok(existing_index);
        }
        
        let index = self.next_index;
        self.neuron_id_to_index[index] = Some(neuron_id);
        self.index_to_neuron_id.push(neuron_id);
        self.next_index += 1;
        self.neuron_count += 1;
        
        Ok(index)
    }
    
    /// Get the matrix index for a neuron ID
    pub fn get_neuron_index(&self, neuron_id: NeuronId) -> Option<usize> {
        self.index_to_neuron_id
            .iter()
            .position(|&id| id == neuron_id)
    }
    
    /// Get the neuron ID for a matrix index
    pub fn get_neuron_id(&self, index: usize) -> Option<NeuronId> {
        self.index_to_neuron_id.get(index).copied()
    }
    
    /// Set the weight between two neurons
    pub fn set_weight(
        &mut self,
        source: NeuronId,
        target: NeuronId,
        weight: f32,
    ) -> Result<()> {
        let source_idx = self.get_neuron_index(source)
            .ok_or(MatrixConnectivityError::InvalidNeuronId(source))?;
        let target_idx = self.get_neuron_index(target)
            .ok_or(MatrixConnectivityError::InvalidNeuronId(target))?;
        
        self.adjacency_matrix.set(source_idx, target_idx, weight)
            .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
        
        Ok(())
    }
    
    /// Get the weight between two neurons
    pub fn get_weight(&self, source: NeuronId, target: NeuronId) -> Result<f32> {
        let source_idx = self.get_neuron_index(source)
            .ok_or(MatrixConnectivityError::InvalidNeuronId(source))?;
        let target_idx = self.get_neuron_index(target)
            .ok_or(MatrixConnectivityError::InvalidNeuronId(target))?;
        
        Ok(self.adjacency_matrix.get(source_idx, target_idx)
            .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?)
    }
    
    /// Set the delay between two neurons
    pub fn set_delay(
        &mut self,
        source: NeuronId,
        target: NeuronId,
        delay: Time,
    ) -> Result<()> {
        let source_idx = self.get_neuron_index(source)
            .ok_or(MatrixConnectivityError::InvalidNeuronId(source))?;
        let target_idx = self.get_neuron_index(target)
            .ok_or(MatrixConnectivityError::InvalidNeuronId(target))?;
        
        if let Some(ref mut delay_matrix) = self.delay_matrix {
            delay_matrix.set(source_idx, target_idx, delay.as_nanos() as f32)
                .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
        }
        
        Ok(())
    }
    
    /// Get the delay between two neurons
    pub fn get_delay(&self, source: NeuronId, target: NeuronId) -> Result<Time> {
        if let Some(ref delay_matrix) = self.delay_matrix {
            let source_idx = self.get_neuron_index(source)
                .ok_or(MatrixConnectivityError::InvalidNeuronId(source))?;
            let target_idx = self.get_neuron_index(target)
                .ok_or(MatrixConnectivityError::InvalidNeuronId(target))?;
            
            let delay_ns = delay_matrix.get(source_idx, target_idx)
                .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
            
            Ok(Time::from_nanos(delay_ns as u64))
        } else {
            Ok(Time::ZERO)
        }
    }
    
    /// Get maximum capacity
    pub fn capacity(&self) -> usize {
        self.neuron_id_to_index.len()
    }
    
    /// Check if the network is at capacity
    pub fn is_full(&self) -> bool {
        self.neuron_count >= self.capacity()
    }
}

impl Default for MatrixNetwork {
    fn default() -> Self {
        Self::new(1000) // Default capacity of 1000 neurons
    }
}

// Phase 6: weight snapshot/apply for MatrixNetwork
impl crate::connectivity::WeightSnapshotConnectivity<NeuronId> for MatrixNetwork {
    fn snapshot_weights(&self) -> Vec<(NeuronId, NeuronId, f32)> {
        let mut out = Vec::new();
        for i in 0..self.neuron_count {
            for j in 0..self.neuron_count {
                if let Ok(w) = self.adjacency_matrix.get(i, j) {
                    if w != 0.0 {
                        if let (Some(pre), Some(post)) = (self.get_neuron_id(i), self.get_neuron_id(j)) {
                            out.push((pre, post, w));
                        }
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
                // clamp to [0,10] similar to other paths
                let wclamp = w.max(0.0).min(10.0);
                self.set_weight(pre, post, wclamp)?;
                applied += 1;
            }
        }
        Ok(applied)
    }
}

/// Implementation of NetworkConnectivity for MatrixNetwork
impl NetworkConnectivity<NeuronId> for MatrixNetwork {
    type ConnectionId = MatrixConnectionId;
    type RouteInfo = MatrixRouteInfo;
    type Error = MatrixConnectivityError;
    
    fn route_spike(
        &self,
        spike: &Spike,
        current_time: Time,
    ) -> Result<Vec<SpikeRoute>> {
        let source_idx = self.get_neuron_index(spike.source)
            .ok_or(MatrixConnectivityError::InvalidNeuronId(spike.source))?;
        
        let mut routes = Vec::new();
        
        // Iterate through all potential targets
        for target_idx in 0..self.neuron_count {
            let weight = self.adjacency_matrix.get(source_idx, target_idx)
                .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
            
            if weight > 0.0 {
                let target_neuron_id = self.get_neuron_id(target_idx)
                    .ok_or(MatrixConnectivityError::IndexOutOfBounds {
                        index: target_idx,
                        bound: self.neuron_count,
                    })?;
                
                let delay = if let Some(ref delay_matrix) = self.delay_matrix {
                    let delay_ns = delay_matrix.get(source_idx, target_idx)
                        .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
                    Time::from_nanos(delay_ns as u64)
                } else {
                    Time::ZERO
                };
                
                let delivery_time = current_time + crate::time::Duration::from_nanos(delay.as_nanos());
                let connection_id = MatrixConnectionId { row: source_idx, col: target_idx };
                
                let spike_route = SpikeRoute::new(
                    connection_id.to_raw(),
                    vec![target_neuron_id],
                    vec![weight * spike.amplitude],
                    delivery_time,
                ).map_err(|e| MatrixConnectivityError::MatrixError(e.to_string()))?;
                
                routes.push(spike_route);
            }
        }
        
        Ok(routes)
    }
    
    fn get_targets(&self, source: NeuronId) -> Result<Vec<NeuronId>> {
        let source_idx = self.get_neuron_index(source)
            .ok_or(MatrixConnectivityError::InvalidNeuronId(source))?;
        
        let mut targets = Vec::new();
        
        for target_idx in 0..self.neuron_count {
            let weight = self.adjacency_matrix.get(source_idx, target_idx)
                .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
            
            if weight > 0.0 {
                let target_neuron_id = self.get_neuron_id(target_idx)
                    .ok_or(MatrixConnectivityError::IndexOutOfBounds {
                        index: target_idx,
                        bound: self.neuron_count,
                    })?;
                targets.push(target_neuron_id);
            }
        }
        
        Ok(targets)
    }
    
    fn get_sources(&self, target: NeuronId) -> Result<Vec<NeuronId>> {
        let target_idx = self.get_neuron_index(target)
            .ok_or(MatrixConnectivityError::InvalidNeuronId(target))?;
        
        let mut sources = Vec::new();
        
        for source_idx in 0..self.neuron_count {
            let weight = self.adjacency_matrix.get(source_idx, target_idx)
                .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
            
            if weight > 0.0 {
                let source_neuron_id = self.get_neuron_id(source_idx)
                    .ok_or(MatrixConnectivityError::IndexOutOfBounds {
                        index: source_idx,
                        bound: self.neuron_count,
                    })?;
                sources.push(source_neuron_id);
            }
        }
        
        Ok(sources)
    }
    
    fn add_connection(
        &mut self,
        connection: Self::ConnectionId,
    ) -> Result<()> {
        if connection.row >= self.neuron_count || connection.col >= self.neuron_count {
            return Err(MatrixConnectivityError::IndexOutOfBounds {
                index: connection.row.max(connection.col),
                bound: self.neuron_count,
            }.into());
        }
        
        self.adjacency_matrix.set(connection.row, connection.col, 1.0)
            .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
        
        Ok(())
    }
    
    fn remove_connection(
        &mut self,
        connection: Self::ConnectionId,
    ) -> Result<Option<Self::RouteInfo>> {
        if connection.row >= self.neuron_count || connection.col >= self.neuron_count {
            return Ok(None);
        }
        
        let weight = self.adjacency_matrix.get(connection.row, connection.col)
            .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
        
        if weight > 0.0 {
            self.adjacency_matrix.set(connection.row, connection.col, 0.0)
                .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
            
            let route_info = MatrixRouteInfo {
                source_index: connection.row,
                target_indices: vec![connection.col],
                weights: vec![weight],
                total_activation: weight,
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
        if connection.row >= self.neuron_count || connection.col >= self.neuron_count {
            return Ok(None);
        }
        
        let old_weight = self.adjacency_matrix.get(connection.row, connection.col)
            .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
        
        self.adjacency_matrix.set(connection.row, connection.col, new_weight)
            .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
        
        Ok(Some(old_weight))
    }
    
    fn get_stats(&self) -> ConnectivityStats {
        let mut connection_count = 0;
        let mut total_weight = 0.0;
        let mut degrees = vec![0u32; self.neuron_count];
        
        // Count connections and calculate statistics
        for i in 0..self.neuron_count {
            for j in 0..self.neuron_count {
                if let Ok(weight) = self.adjacency_matrix.get(i, j) {
                    if weight > 0.0 {
                        connection_count += 1;
                        total_weight += weight;
                        degrees[i] += 1; // Out-degree
                        degrees[j] += 1; // In-degree (counted separately)
                    }
                }
            }
        }
        
        let average_weight = if connection_count > 0 {
            total_weight / connection_count as f32
        } else {
            0.0
        };
        
        let (max_degree, min_degree) = if !degrees.is_empty() {
            (*degrees.iter().max().unwrap_or(&0), *degrees.iter().min().unwrap_or(&0))
        } else {
            (0, 0)
        };
        
        let average_degree = if self.neuron_count > 0 {
            degrees.iter().sum::<u32>() as f32 / self.neuron_count as f32
        } else {
            0.0
        };
        
        let density = if self.neuron_count > 1 {
            connection_count as f32 / (self.neuron_count * (self.neuron_count - 1)) as f32
        } else {
            0.0
        };
        
        let memory_usage = core::mem::size_of::<Self>() +
            self.adjacency_matrix.as_slice().len() * core::mem::size_of::<Float>() +
            self.delay_matrix.as_ref().map_or(0, |dm| dm.as_slice().len() * core::mem::size_of::<Float>()) +
            self.neuron_id_to_index.len() * core::mem::size_of::<Option<NeuronId>>() +
            self.index_to_neuron_id.len() * core::mem::size_of::<NeuronId>();
        
        ConnectivityStats {
            connection_count,
            node_count: self.neuron_count,
            average_degree,
            max_degree,
            min_degree,
            total_weight,
            average_weight,
            memory_usage,
            has_cycles: None, // Would require cycle detection
            density,
            custom_stats: vec![
                ("matrix_capacity".to_string(), self.capacity() as f32),
                ("matrix_utilization".to_string(), 
                 if self.capacity() > 0 { self.neuron_count as f32 / self.capacity() as f32 } else { 0.0 }),
                ("has_delays".to_string(), if self.delay_matrix.is_some() { 1.0 } else { 0.0 }),
            ],
        }
    }
    
    fn validate(&self) -> Result<()> {
        // Check matrix dimensions
        let (rows, cols) = self.adjacency_matrix.dims();
        if rows != cols {
            return Err(MatrixConnectivityError::DimensionMismatch { expected: rows, got: cols }.into());
        }
        
        // Check neuron count consistency
        if self.neuron_count > self.capacity() {
            return Err(MatrixConnectivityError::IndexOutOfBounds {
                index: self.neuron_count,
                bound: self.capacity(),
            }.into());
        }
        
        // Validate delay matrix if present
        if let Some(ref delay_matrix) = self.delay_matrix {
            let (delay_rows, delay_cols) = delay_matrix.dims();
            if delay_rows != rows || delay_cols != cols {
                return Err(MatrixConnectivityError::DimensionMismatch {
                    expected: rows,
                    got: delay_rows,
                }.into());
            }
        }
        
        Ok(())
    }
    
    fn reset(&mut self) {
        self.adjacency_matrix = Matrix::zeros(self.capacity(), self.capacity());
        let capacity = self.capacity();
        if let Some(ref mut delay_matrix) = self.delay_matrix {
            *delay_matrix = Matrix::zeros(capacity, capacity);
        }
        self.neuron_count = 0;
        self.neuron_id_to_index.iter_mut().for_each(|entry| *entry = None);
        self.index_to_neuron_id.clear();
        self.next_index = 0;
    }
    
    fn connection_count(&self) -> usize {
        let mut count = 0;
        for i in 0..self.neuron_count {
            for j in 0..self.neuron_count {
                if let Ok(weight) = self.adjacency_matrix.get(i, j) {
                    if weight > 0.0 {
                        count += 1;
                    }
                }
            }
        }
        count
    }
    
    fn neuron_count(&self) -> usize {
        self.neuron_count
    }
}

/// Batch operations for matrix connectivity
impl BatchConnectivity<NeuronId> for MatrixNetwork {
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

/// Plasticity support for matrix connectivity
impl PlasticConnectivity<NeuronId> for MatrixNetwork {
    fn apply_plasticity(
        &mut self,
        pre_neuron: NeuronId,
        post_neuron: NeuronId,
        weight_delta: f32,
    ) -> Result<Option<f32>> {
        let source_idx = self.get_neuron_index(pre_neuron)
            .ok_or(MatrixConnectivityError::InvalidNeuronId(pre_neuron))?;
        let target_idx = self.get_neuron_index(post_neuron)
            .ok_or(MatrixConnectivityError::InvalidNeuronId(post_neuron))?;
        
        let current_weight = self.adjacency_matrix.get(source_idx, target_idx)
            .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
        
        let new_weight = (current_weight + weight_delta).max(0.0).min(10.0); // Clamp weights
        
        self.adjacency_matrix.set(source_idx, target_idx, new_weight)
            .map_err(|e| MatrixConnectivityError::MatrixError(format!("{:?}", e)))?;
        
        Ok(Some(new_weight))
    }
    
    fn get_weight(
        &self,
        pre_neuron: NeuronId,
        post_neuron: NeuronId,
    ) -> Result<Option<f32>> {
        match self.get_weight(pre_neuron, post_neuron) {
            Ok(weight) => Ok(Some(weight)),
            Err(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Time;

    #[test]
    fn test_matrix_network_basic() {
        let mut network = MatrixNetwork::new(10);
        
        let neuron1 = NeuronId::new(1);
        let neuron2 = NeuronId::new(2);
        
        network.add_neuron(neuron1).expect("Should add neuron");
        network.add_neuron(neuron2).expect("Should add neuron");
        
        network.set_weight(neuron1, neuron2, 0.5).expect("Should set weight");
        
        assert_eq!(network.neuron_count(), 2);
        assert_eq!(network.connection_count(), 1);
        
        let weight = network.get_weight(neuron1, neuron2).expect("Should get weight");
        assert_eq!(weight, 0.5);
    }
    
    #[test]
    fn test_spike_routing() {
        let mut network = MatrixNetwork::new(10);
        
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
    fn test_matrix_connection_id() {
        let connection_id = MatrixConnectionId { row: 42, col: 84 };
        
        let raw = connection_id.to_raw();
        let reconstructed = MatrixConnectionId::from_raw(&raw)
            .expect("Should reconstruct from raw");
        
        assert_eq!(connection_id, reconstructed);
        assert_eq!(connection_id.display_string(), "M[42,84]");
    }
    
    #[test]
    fn test_fully_connected_network() {
        let neurons = vec![NeuronId::new(0), NeuronId::new(1), NeuronId::new(2)];
        let network = MatrixNetwork::fully_connected(&neurons, 1.0)
            .expect("Should create fully connected network");
        
        assert_eq!(network.neuron_count(), 3);
        assert_eq!(network.connection_count(), 6); // 3 * (3-1) = 6 directed edges
    }
    
    #[test]
    fn test_plasticity() {
        let mut network = MatrixNetwork::new(10);
        
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

    // Phase 6 tests: snapshot/apply on MatrixNetwork
    #[test]
    fn test_matrix_snapshot_and_apply() {
        use crate::connectivity::WeightSnapshotConnectivity;
        let mut net = MatrixNetwork::new(10);
        let pre = NeuronId::new(0);
        let post = NeuronId::new(1);
        net.add_neuron(pre).unwrap();
        net.add_neuron(post).unwrap();
        net.set_weight(pre, post, 0.3).unwrap();

        let snap = <MatrixNetwork as WeightSnapshotConnectivity<NeuronId>>::snapshot_weights(&net);
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].0, pre);
        assert_eq!(snap[0].1, post);
        assert!((snap[0].2 - 0.3).abs() < 1e-6);

        let updates = [(pre, post, 0.95)];
        let applied = <MatrixNetwork as WeightSnapshotConnectivity<NeuronId>>::apply_weight_updates(&mut net, &updates).unwrap();
        assert_eq!(applied, 1);

        let new_w = net.get_weight(pre, post).unwrap();
        assert_eq!(new_w, Some(0.95));
    }
}