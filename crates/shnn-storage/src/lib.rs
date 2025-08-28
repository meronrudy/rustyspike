//! Storage layer and binary schemas for the CLI-first SNN framework
//!
//! This crate provides the fundamental storage infrastructure for the CLI-first
//! neuromorphic research substrate, including binary schemas, temporal snapshots,
//! and efficient data access patterns.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(missing_docs)]
#![warn(clippy::all)]

// Temporary local type definitions until shnn-core compilation is fixed
/// Neuron identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NeuronId(pub u32);

impl NeuronId {
    /// Create a new neuron ID
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
    
    /// Get the raw ID value
    pub const fn raw(&self) -> u32 {
        self.0
    }
}

/// Hyperedge identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HyperedgeId(pub u32);

impl HyperedgeId {
    /// Create a new hyperedge ID
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
    
    /// Get the raw ID value
    pub const fn raw(&self) -> u32 {
        self.0
    }
}

/// Time representation (nanoseconds since epoch)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Time(pub u64);

impl Time {
    /// Create a new time value
    pub const fn new(ns: u64) -> Self {
        Self(ns)
    }
    
    /// Create time from nanoseconds
    pub const fn from_nanos(ns: u64) -> Self {
        Self(ns)
    }
    
    /// Create time from milliseconds
    pub const fn from_millis(ms: u64) -> Self {
        Self(ms * 1_000_000)
    }
    
    /// Get nanoseconds since epoch
    pub const fn nanos(&self) -> u64 {
        self.0
    }
    
    /// Get nanoseconds since epoch (alias for nanos)
    pub const fn as_nanos(&self) -> u64 {
        self.0
    }
    
    /// Zero time constant
    pub const ZERO: Self = Self(0);
}

/// Spike event
#[derive(Debug, Clone, PartialEq)]
pub struct Spike {
    /// Neuron that spiked
    pub neuron_id: NeuronId,
    /// Time of spike
    pub time: Time,
    /// Spike amplitude (optional)
    pub amplitude: f32,
}

impl Spike {
    /// Create a new spike
    pub fn new(neuron_id: NeuronId, time: Time) -> Self {
        Self {
            neuron_id,
            time,
            amplitude: 1.0,
        }
    }
    
    /// Create a spike with specific amplitude
    pub fn with_amplitude(neuron_id: NeuronId, time: Time, amplitude: f32) -> Self {
        Self {
            neuron_id,
            time,
            amplitude,
        }
    }
}

// Core modules
pub mod error;
pub mod ids;
pub mod schemas;
pub mod traits;

// Storage backends
pub mod memory;
pub mod file;

// Specific format implementations
pub mod vcsr;
pub mod vevt;
pub mod vmsk;

// Re-export essential types
pub use error::{StorageError, Result};
pub use ids::{GenerationId, MaskId, StreamId};
pub use traits::{
    HypergraphStore, EventStore, Mask, HypergraphSnapshot, HypergraphSubview,
    Event, EventType, MaskType, GraphStats, MorphologyOp, VertexProperties
};

// Re-export implementations
pub use memory::{MemoryStore, MemorySnapshot};
pub use file::FileStore;
pub use vcsr::{VCSRSnapshot, VCSRHeader, VCSRVertex};
pub use vevt::{MemoryEventStore, VEVTEvent, VEVTHeader};
pub use vmsk::{BitmapMask, VMSKHeader};

/// Storage crate version for compatibility checking
pub const STORAGE_VERSION: u32 = 1;

/// Magic numbers for all binary formats
pub mod magic {
    /// VCSR magic number: "VCSR"
    pub const VCSR: [u8; 4] = [0x56, 0x43, 0x53, 0x52];
    /// VEVT magic number: "VEVT"
    pub const VEVT: [u8; 4] = [0x56, 0x45, 0x56, 0x54];
    /// VMSK magic number: "VMSK"
    pub const VMSK: [u8; 4] = [0x56, 0x4D, 0x53, 0x4B];
    /// VMORF magic number: "MORF"
    pub const VMORF: [u8; 4] = [0x4D, 0x4F, 0x52, 0x46];
    /// VGRF magic number: "GRAF"
    pub const VGRF: [u8; 4] = [0x47, 0x52, 0x41, 0x46];
    /// VRAS magic number: "RAST"
    pub const VRAS: [u8; 4] = [0x52, 0x41, 0x53, 0x54];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_numbers() {
        // Verify magic numbers are distinct
        let magics = [
            magic::VCSR,
            magic::VEVT,
            magic::VMSK,
            magic::VMORF,
            magic::VGRF,
            magic::VRAS,
        ];
        
        for (i, &magic1) in magics.iter().enumerate() {
            for (j, &magic2) in magics.iter().enumerate() {
                if i != j {
                    assert_ne!(magic1, magic2, "Magic numbers must be distinct");
                }
            }
        }
    }
}