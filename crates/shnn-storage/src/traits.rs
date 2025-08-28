//! Core trait definitions for the storage layer

use crate::{
    error::{Result, StorageError},
    ids::{GenerationId, MaskId, StreamId},
    NeuronId, HyperedgeId, Time,
};

use core::fmt;

/// Graph statistics for analysis and debugging
#[derive(Debug, Clone, PartialEq)]
pub struct GraphStats {
    /// Number of neurons in the graph
    pub num_neurons: u32,
    /// Number of hyperedges in the graph
    pub num_hyperedges: u32,
    /// Total number of incidences (neuron-hyperedge connections)
    pub num_incidences: u64,
    /// Average degree of neurons
    pub avg_degree: f32,
    /// Maximum degree of any neuron
    pub max_degree: u32,
    /// Graph density (0.0 to 1.0)
    pub density: f32,
    /// Generation this statistics represent
    pub generation: GenerationId,
    /// Timestamp when statistics were computed
    pub timestamp: Time,
}

/// Primary interface for hypergraph storage and retrieval
pub trait HypergraphStore: Send + Sync {
    /// Snapshot type for graph state at a specific generation
    type Snapshot: HypergraphSnapshot;
    
    /// Error type for storage operations
    type Error: From<StorageError> + Send + Sync + fmt::Debug + fmt::Display;
    
    /// Get a snapshot of the hypergraph at a specific generation
    fn get_snapshot(&self, generation: GenerationId) -> Result<Self::Snapshot>;
    
    /// Get the latest available generation
    fn latest_generation(&self) -> Result<GenerationId>;
    
    /// Get available generations in a range
    fn list_generations(
        &self, 
        start: Option<GenerationId>, 
        end: Option<GenerationId>
    ) -> Result<Vec<GenerationId>>;
    
    /// Create a new generation from a previous one with modifications
    fn create_generation(
        &mut self, 
        base: GenerationId, 
        operations: &[MorphologyOp]
    ) -> Result<GenerationId>;
    
    /// Compact storage by removing intermediate generations
    fn compact(&mut self, keep_generations: &[GenerationId]) -> Result<()>;
}

/// Interface for individual hypergraph snapshots
pub trait HypergraphSnapshot: Send + Sync {
    /// Subview type for masked/filtered views
    type Subview: HypergraphSubview;
    
    /// Iterator type for graph traversal
    type NeighborIter: Iterator<Item = (NeuronId, f32)> + Send;
    /// Iterator over hyperedges with their vertices and weights
    type HyperedgeIter: Iterator<Item = (HyperedgeId, Vec<NeuronId>, f32)> + Send;
    
    /// Get basic graph statistics
    fn stats(&self) -> GraphStats;
    
    /// Get neighbors of a neuron with weights
    fn neighbors(&self, neuron: NeuronId) -> Result<Self::NeighborIter>;
    
    /// Get all hyperedges involving a neuron
    fn hyperedges(&self, neuron: NeuronId) -> Result<Self::HyperedgeIter>;
    
    /// Apply a mask to create a subview
    fn apply_mask(&self, mask: &dyn Mask) -> Result<Self::Subview>;
    
    /// Get k-hop neighborhood around a set of neurons
    fn k_hop(&self, seeds: &[NeuronId], k: u32) -> Result<Self::Subview>;
    
    /// Check if an edge exists between neurons
    fn has_edge(&self, source: NeuronId, target: NeuronId) -> bool;
    
    /// Get edge weight if it exists
    fn edge_weight(&self, source: NeuronId, target: NeuronId) -> Option<f32>;
}

/// Interface for filtered subviews of hypergraphs
pub trait HypergraphSubview: Send + Sync {
    /// Get list of active neurons in this subview
    fn active_neurons(&self) -> &[NeuronId];
    
    /// Get list of active hyperedges in this subview
    fn active_hyperedges(&self) -> &[HyperedgeId];
    
    /// Get statistics for this subview
    fn stats(&self) -> GraphStats;
    
    /// Export subview to VCSR format
    fn export_vcsr(&self) -> Result<Vec<u8>>;
    
    /// Export subview to GraphML format
    fn export_graphml(&self) -> Result<String>;
}

/// Interface for temporal event streams
pub trait EventStore: Send + Sync {
    /// Event type stored in this stream
    type Event: Event + Send + Sync;
    
    /// Iterator type for event traversal
    type EventIter: Iterator<Item = Self::Event> + Send;
    
    /// Error type for event operations
    type Error: From<StorageError> + Send + Sync + fmt::Debug + fmt::Display;
    
    /// Append events to the stream
    fn append_events(&mut self, events: &[Self::Event]) -> Result<()>;
    
    /// Get events in a time window
    fn time_window(&self, start: Time, end: Time) -> Result<Self::EventIter>;
    
    /// Get events for specific neurons in a time window
    fn neuron_events(
        &self, 
        neurons: &[NeuronId], 
        start: Time, 
        end: Time
    ) -> Result<Self::EventIter>;
    
    /// Get the total number of events
    fn event_count(&self) -> u64;
    
    /// Get the time range covered by events
    fn time_range(&self) -> Option<(Time, Time)>;
    
    /// Export events to VEVT format
    fn export_vevt(&self, start: Time, end: Time) -> Result<Vec<u8>>;
}

/// Base trait for all events in the system
pub trait Event: Clone + Send + Sync {
    /// Get the timestamp of this event
    fn timestamp(&self) -> Time;
    
    /// Get the type of this event
    fn event_type(&self) -> EventType;
    
    /// Get the source neuron ID if applicable
    fn source_id(&self) -> Option<NeuronId>;
    
    /// Serialize this event to bytes
    fn serialize(&self) -> Result<Vec<u8>>;
}

/// Event types supported by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    /// Neural spike event
    Spike,
    /// TTR phase entry
    PhaseEnter,
    /// TTR phase exit
    PhaseExit,
    /// Neuromodulation event
    Neuromodulation,
    /// Reward signal
    Reward,
    /// System control event
    Control,
    /// Temporal marker
    Marker,
}

/// Interface for masks used in subviews and TTR
pub trait Mask: Send + Sync {
    /// Get the mask ID
    fn mask_id(&self) -> MaskId;
    
    /// Get the type of this mask
    fn mask_type(&self) -> MaskType;
    
    /// Check if a given ID is active in this mask
    fn is_active(&self, id: u32) -> bool;
    
    /// Get the number of active items
    fn active_count(&self) -> u64;
    
    /// Get the total number of items this mask covers
    fn total_count(&self) -> u64;
    
    /// Combine with another mask using intersection
    fn intersect(&self, other: &dyn Mask) -> Result<Box<dyn Mask>>;
    
    /// Combine with another mask using union
    fn union(&self, other: &dyn Mask) -> Result<Box<dyn Mask>>;
    
    /// Combine with another mask using difference
    fn difference(&self, other: &dyn Mask) -> Result<Box<dyn Mask>>;
    
    /// Export to VMSK format
    fn export_vmsk(&self) -> Result<Vec<u8>>;
}

/// Types of masks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskType {
    /// Mask for vertices/neurons
    VertexMask,
    /// Mask for edges/hyperedges
    EdgeMask,
    /// Mask for TTR modules
    ModuleMask,
    /// Mask for temporal windows
    TemporalMask,
    /// Mask based on activity patterns
    ActivityMask,
}

/// Morphology operation for graph modification
#[derive(Debug, Clone, PartialEq)]
pub enum MorphologyOp {
    /// Add a vertex
    AddVertex {
        /// Vertex ID
        id: NeuronId,
        /// Vertex type/properties
        properties: VertexProperties,
    },
    /// Remove a vertex
    RemoveVertex {
        /// Vertex ID
        id: NeuronId,
    },
    /// Add a hyperedge
    AddHyperedge {
        /// Hyperedge ID
        id: HyperedgeId,
        /// Connected vertices
        vertices: Vec<NeuronId>,
        /// Edge weight
        weight: f32,
    },
    /// Remove a hyperedge
    RemoveHyperedge {
        /// Hyperedge ID
        id: HyperedgeId,
    },
    /// Modify edge weight
    ModifyWeight {
        /// Hyperedge ID
        edge_id: HyperedgeId,
        /// New weight
        weight: f32,
    },
}

/// Properties for vertices
#[derive(Debug, Clone, PartialEq)]
pub struct VertexProperties {
    /// Vertex type (neuron model)
    pub vertex_type: u8,
    /// Additional flags
    pub flags: u8,
}

/// Base capability trait for feature detection
pub trait Capability {
    /// Name of this capability
    const NAME: &'static str;
    /// Version of this capability
    const VERSION: u32;
    
    /// Check if this capability is available
    fn is_available(&self) -> bool;
    
    /// Get required capabilities for this feature
    fn required_capabilities(&self) -> &[&'static str] {
        &[]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_type() {
        assert_eq!(EventType::Spike, EventType::Spike);
        assert_ne!(EventType::Spike, EventType::Control);
    }

    #[test]
    fn test_mask_type() {
        assert_eq!(MaskType::VertexMask, MaskType::VertexMask);
        assert_ne!(MaskType::VertexMask, MaskType::EdgeMask);
    }

    #[test]
    fn test_graph_stats() {
        let stats = GraphStats {
            num_neurons: 1000,
            num_hyperedges: 5000,
            num_incidences: 10000,
            avg_degree: 10.0,
            max_degree: 50,
            density: 0.01,
            generation: GenerationId::new(1),
            timestamp: Time::from_millis(1000),
        };
        
        assert_eq!(stats.num_neurons, 1000);
        assert_eq!(stats.generation, GenerationId::new(1));
    }
}