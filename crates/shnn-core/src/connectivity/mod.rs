//! Network connectivity abstraction layer
//!
//! This module provides a generic interface for different network connectivity
//! structures, allowing users to choose optimal data structures for their use cases.

use crate::{
    spike::{NeuronId, Spike},
    time::Time,
    error::{Result, SHNNError},
};
use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Re-export connectivity implementations
pub mod hypergraph;
pub mod graph;
pub mod matrix;
pub mod sparse;
pub mod types;
#[cfg(feature = "plastic-sum")]
pub mod plastic_enum;

pub use types::{SpikeRoute, ConnectivityStats, ConnectionId};
#[cfg(feature = "plastic-sum")]
pub use plastic_enum::PlasticConn;

/// Generic trait for network connectivity structures
///
/// This trait abstracts the underlying data structure used for neural connectivity,
/// enabling pluggable implementations for different use cases and performance
/// requirements.
pub trait NetworkConnectivity<NodeId> {
    /// Type representing connection identifiers
    type ConnectionId: Clone + fmt::Debug + PartialEq;
    
    /// Type containing routing information for connections
    type RouteInfo: Clone + fmt::Debug;
    
    /// Error type for connectivity operations
    type Error: From<SHNNError> + fmt::Debug + fmt::Display;
    
    /// Route a spike through the connectivity structure
    ///
    /// Given a spike from a source neuron, determine all target neurons
    /// and their corresponding weights and delivery times.
    ///
    /// # Arguments
    /// * `spike` - The input spike to route
    /// * `current_time` - Current simulation time
    ///
    /// # Returns
    /// Vector of spike routes representing all targets and their parameters
    fn route_spike(
        &self, 
        spike: &Spike, 
        current_time: Time
    ) -> Result<Vec<SpikeRoute>>;
    
    /// Get all target neurons for a given source neuron
    ///
    /// # Arguments
    /// * `source` - Source neuron ID
    ///
    /// # Returns
    /// Vector of target neuron IDs
    fn get_targets(&self, source: NodeId) -> Result<Vec<NodeId>>;
    
    /// Get all source neurons for a given target neuron
    ///
    /// # Arguments
    /// * `target` - Target neuron ID
    ///
    /// # Returns
    /// Vector of source neuron IDs
    fn get_sources(&self, target: NodeId) -> Result<Vec<NodeId>>;
    
    /// Add a connection to the network
    ///
    /// # Arguments
    /// * `connection` - Connection identifier to add
    ///
    /// # Returns
    /// Result indicating success or failure
    fn add_connection(
        &mut self, 
        connection: Self::ConnectionId
    ) -> Result<()>;
    
    /// Remove a connection from the network
    ///
    /// # Arguments
    /// * `connection` - Connection identifier to remove
    ///
    /// # Returns
    /// Optional route info for the removed connection
    fn remove_connection(
        &mut self, 
        connection: Self::ConnectionId
    ) -> Result<Option<Self::RouteInfo>>;
    
    /// Update the weight of an existing connection
    ///
    /// # Arguments
    /// * `connection` - Connection to update
    /// * `new_weight` - New weight value
    ///
    /// # Returns
    /// Previous weight value if connection exists
    fn update_weight(
        &mut self,
        connection: Self::ConnectionId,
        new_weight: f32,
    ) -> Result<Option<f32>>;
    
    /// Get network statistics
    ///
    /// # Returns
    /// Statistics about the connectivity structure
    fn get_stats(&self) -> ConnectivityStats;
    
    /// Validate the connectivity structure
    ///
    /// Checks for consistency and potential issues in the network structure.
    ///
    /// # Returns
    /// Result indicating if the structure is valid
    fn validate(&self) -> Result<()>;
    
    /// Reset the connectivity structure
    ///
    /// Clears all connections while preserving allocated capacity where possible.
    fn reset(&mut self);
    
    /// Get the number of connections in the network
    fn connection_count(&self) -> usize;
    
    /// Get the number of unique neurons in the network
    fn neuron_count(&self) -> usize;
    
    /// Check if the network contains any connections
    fn is_empty(&self) -> bool {
        self.connection_count() == 0
    }
}

/// Extension trait for connectivity structures that support batch operations
pub trait BatchConnectivity<NodeId>: NetworkConnectivity<NodeId> {
    /// Add multiple connections in a batch operation
    ///
    /// This can be more efficient than adding connections one by one.
    ///
    /// # Arguments
    /// * `connections` - Iterator of connections to add
    ///
    /// # Returns
    /// Number of connections successfully added
    fn add_connections<I>(&mut self, connections: I) -> Result<usize>
    where 
        I: IntoIterator<Item = Self::ConnectionId>;
    
    /// Remove multiple connections in a batch operation
    ///
    /// # Arguments
    /// * `connections` - Iterator of connections to remove
    ///
    /// # Returns
    /// Number of connections successfully removed
    fn remove_connections<I>(&mut self, connections: I) -> Result<usize>
    where 
        I: IntoIterator<Item = Self::ConnectionId>;
}

/// Extension trait for connectivity structures that support plasticity
pub trait PlasticConnectivity<NodeId>: NetworkConnectivity<NodeId> {
    /// Update weights based on plasticity rules
    ///
    /// # Arguments
    /// * `pre_neuron` - Presynaptic neuron ID
    /// * `post_neuron` - Postsynaptic neuron ID
    /// * `weight_delta` - Change in weight
    ///
    /// # Returns
    /// New weight value if connection exists
    fn apply_plasticity(
        &mut self,
        pre_neuron: NodeId,
        post_neuron: NodeId,
        weight_delta: f32,
    ) -> Result<Option<f32>>;
    
    /// Get current weight between two neurons
    ///
    /// # Arguments
    /// * `pre_neuron` - Presynaptic neuron ID
    /// * `post_neuron` - Postsynaptic neuron ID
    ///
    /// # Returns
    /// Current weight if connection exists
    fn get_weight(
        &self,
        pre_neuron: NodeId,
        post_neuron: NodeId,
    ) -> Result<Option<f32>>;
}

/// Extension trait for connectivity structures that support serialization
#[cfg(feature = "serde")]
pub trait SerializableConnectivity<NodeId>: NetworkConnectivity<NodeId> {
    /// Serialize the connectivity structure to bytes
    fn serialize(&self) -> Result<Vec<u8>, Self::Error>;
    
    /// Deserialize the connectivity structure from bytes
    fn deserialize(data: &[u8]) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spike::NeuronId;

    /// Test trait to verify trait object compatibility
    fn _test_trait_object_compatibility() {
        let _: Box<dyn NetworkConnectivity<
            NeuronId,
            ConnectionId = u32,
            RouteInfo = (),
            Error = SHNNError
        >> = todo!();
    }
}

//// ===== Phase 6: Weight snapshot/apply extension trait =====
/// Extension trait providing bulk weight extraction and application.
///
/// This trait enables:
/// - Snapshots of all existing directed connections and their weights as (pre, post, weight) triples
/// - Efficient bulk application of weight updates without rebuilding connectivity
///
/// Typical use cases:
/// - External optimizers performing batch updates after an evaluation epoch
/// - Checkpointing or exporting the current synaptic state
/// - Cross-backend weight synchronization
pub trait WeightSnapshotConnectivity<NodeId>: NetworkConnectivity<NodeId> {
    /// Create a snapshot of all connections as (pre, post, weight) triples.
    ///
    /// The returned vector is implementation-defined in ordering but must contain
    /// one entry per existing directed connection whose weight is non-zero (or
    /// otherwise considered active by the backend). Neuron identifiers and weights
    /// are represented in the same numeric domain used by the backend.
    fn snapshot_weights(&self) -> Vec<(NodeId, NodeId, f32)>;

    /// Apply a batch of weight updates.
    ///
    /// Each tuple (pre, post, new_weight) sets the weight of the directed connection
    /// from `pre` to `post` to `new_weight` when such a connection exists. Backends
    /// may clamp the weight to their supported range. Connections not present are
    /// ignored. Returns the number of successful updates applied.
    fn apply_weight_updates(&mut self, updates: &[(NodeId, NodeId, f32)]) -> Result<usize>;
}