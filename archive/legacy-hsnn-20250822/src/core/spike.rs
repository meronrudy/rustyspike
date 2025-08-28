//! Spike representation and processing for H-SNN
//!
//! This module provides the fundamental spike data structures adapted from SHNN
//! with H-SNN specific extensions for spike walks and hyperpath traversal.

use crate::utils::error::{HSNNError, Result};
use crate::core::time::{Time, Duration};
use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use crate::utils::HashMap;

/// Unique identifier for a neuron in the network
/// 
/// Adapted from SHNN with same interface for compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
    
    /// Invalid neuron ID constant
    pub const INVALID: Self = Self(u32::MAX);
    
    /// Check if this is a valid neuron ID
    pub const fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }
}

impl fmt::Display for NeuronId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "N{}", self.0)
    }
}

impl From<u32> for NeuronId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<NeuronId> for u32 {
    fn from(id: NeuronId) -> Self {
        id.0
    }
}

impl From<usize> for NeuronId {
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

impl From<NeuronId> for usize {
    fn from(id: NeuronId) -> Self {
        id.0 as usize
    }
}

/// Unique identifier for spike walks in H-SNN
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpikeWalkId(pub u64);

impl SpikeWalkId {
    /// Create a new spike walk ID
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
    
    /// Get the raw ID value
    pub const fn raw(&self) -> u64 {
        self.0
    }
    
    /// Invalid spike walk ID constant
    pub const INVALID: Self = Self(u64::MAX);
    
    /// Check if this is a valid spike walk ID
    pub const fn is_valid(&self) -> bool {
        self.0 != u64::MAX
    }
}

impl fmt::Display for SpikeWalkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SW{}", self.0)
    }
}

/// A neural spike event - core primitive adapted from SHNN
///
/// Represents a discrete spike event with source neuron, timestamp, and amplitude.
/// This is the fundamental unit of information in neuromorphic computation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Spike {
    /// The neuron that generated this spike
    pub source: NeuronId,
    /// When the spike occurred
    pub timestamp: Time,
    /// Spike amplitude (typically 1.0 for binary spikes)
    pub amplitude: f32,
}

impl Spike {
    /// Create a new spike
    pub fn new(source: NeuronId, timestamp: Time, amplitude: f32) -> Result<Self> {
        if !source.is_valid() {
            return Err(HSNNError::InvalidSpike("Invalid source neuron ID".into()));
        }
        
        if !amplitude.is_finite() || amplitude < 0.0 {
            return Err(HSNNError::InvalidSpike("Invalid spike amplitude".into()));
        }
        
        Ok(Self {
            source,
            timestamp,
            amplitude,
        })
    }
    
    /// Create a binary spike (amplitude = 1.0)
    pub fn binary(source: NeuronId, timestamp: Time) -> Result<Self> {
        Self::new(source, timestamp, 1.0)
    }
    
    /// Create a weighted spike with custom amplitude
    pub fn weighted(source: NeuronId, timestamp: Time, amplitude: f32) -> Result<Self> {
        Self::new(source, timestamp, amplitude)
    }
    
    /// Check if this spike is valid
    pub fn is_valid(&self) -> bool {
        self.source.is_valid() && self.amplitude.is_finite() && self.amplitude >= 0.0
    }
}

impl fmt::Display for Spike {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Spike({} @ {} with amplitude {})",
            self.source, self.timestamp, self.amplitude
        )
    }
}

/// Context information carried during spike walks
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WalkContext {
    /// Information accumulated during the walk
    pub accumulated_info: f32,
    /// Walk-specific metadata
    pub metadata: HashMap<String, f32>,
    /// Temporal window for this walk
    pub temporal_window: Duration,
}

impl WalkContext {
    /// Create a new empty walk context
    pub fn new() -> Self {
        Self {
            accumulated_info: 0.0,
            metadata: HashMap::new(),
            temporal_window: Duration::from_millis(10), // Default 10ms window
        }
    }
    
    /// Create context with initial information
    pub fn with_info(info: f32) -> Self {
        Self {
            accumulated_info: info,
            metadata: HashMap::new(),
            temporal_window: Duration::from_millis(10),
        }
    }
    
    /// Add information to the context
    pub fn add_info(&mut self, info: f32) {
        self.accumulated_info += info;
    }
    
    /// Set metadata value
    pub fn set_metadata(&mut self, key: String, value: f32) {
        self.metadata.insert(key, value);
    }
    
    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).copied()
    }
}

impl Default for WalkContext {
    fn default() -> Self {
        Self::new()
    }
}

/// A spike walk through the hypergraph - NEW for H-SNN
///
/// Represents the traversal of a spike through hyperpath structures,
/// enabling structured inference and group-level computation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpikeWalk {
    /// Unique identifier for this walk
    pub id: SpikeWalkId,
    /// The initiating spike that started this walk
    pub initiating_spike: Spike,
    /// Current position in the hyperpath
    pub current_hyperedge: crate::hypergraph::HyperedgeId,
    /// History of traversed hyperedges with timestamps
    pub traversal_history: Vec<(crate::hypergraph::HyperedgeId, Time)>,
    /// Context information carried during the walk
    pub context: WalkContext,
    /// Whether this walk is still active
    pub active: bool,
}

impl SpikeWalk {
    /// Create a new spike walk
    pub fn new(
        id: SpikeWalkId,
        initiating_spike: Spike,
        starting_hyperedge: crate::hypergraph::HyperedgeId,
    ) -> Self {
        Self {
            id,
            initiating_spike,
            current_hyperedge: starting_hyperedge,
            traversal_history: vec![(starting_hyperedge, initiating_spike.timestamp)],
            context: WalkContext::new(),
            active: true,
        }
    }
    
    /// Move the walk to a new hyperedge
    pub fn traverse_to(&mut self, hyperedge: crate::hypergraph::HyperedgeId, time: Time) {
        self.traversal_history.push((hyperedge, time));
        self.current_hyperedge = hyperedge;
    }
    
    /// Terminate the walk
    pub fn terminate(&mut self) {
        self.active = false;
    }
    
    /// Get the path length (number of hyperedges traversed)
    pub fn path_length(&self) -> usize {
        self.traversal_history.len()
    }
    
    /// Get the total duration of the walk
    pub fn duration(&self) -> Duration {
        if let (Some(first), Some(last)) = (
            self.traversal_history.first(),
            self.traversal_history.last(),
        ) {
            last.1 - first.1
        } else {
            Duration::ZERO
        }
    }
    
    /// Check if the walk has visited a specific hyperedge
    pub fn has_visited(&self, hyperedge: crate::hypergraph::HyperedgeId) -> bool {
        self.traversal_history.iter().any(|(id, _)| *id == hyperedge)
    }
}

impl fmt::Display for SpikeWalk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SpikeWalk({}, {} steps, {} active)",
            self.id,
            self.path_length(),
            if self.active { "active" } else { "terminated" }
        )
    }
}

/// A timed spike with precise temporal information
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimedSpike {
    /// The spike data
    pub spike: Spike,
    /// Precise delivery time
    pub delivery_time: Time,
    /// Optional delay from original spike time
    pub delay: Option<Duration>,
}

impl TimedSpike {
    /// Create a new timed spike
    pub fn new(spike: Spike, delivery_time: Time) -> Self {
        Self {
            spike,
            delivery_time,
            delay: None,
        }
    }
    
    /// Create a timed spike with delay
    pub fn with_delay(spike: Spike, delay: Duration) -> Self {
        let delivery_time = spike.timestamp + delay;
        Self {
            spike,
            delivery_time,
            delay: Some(delay),
        }
    }
    
    /// Get the effective timestamp (delivery time)
    pub fn effective_time(&self) -> Time {
        self.delivery_time
    }
}

/// A collection of spikes for batch processing
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SpikeTrain {
    /// Source neuron for all spikes in this train
    pub source: NeuronId,
    /// Spike timestamps
    pub timestamps: Vec<Time>,
    /// Optional amplitudes (if None, all spikes have amplitude 1.0)
    pub amplitudes: Option<Vec<f32>>,
}

impl SpikeTrain {
    /// Create a new spike train
    pub fn new(source: NeuronId) -> Self {
        Self {
            source,
            timestamps: Vec::new(),
            amplitudes: None,
        }
    }
    
    /// Add a spike to the train
    pub fn add_spike(&mut self, timestamp: Time, amplitude: Option<f32>) {
        self.timestamps.push(timestamp);
        
        if let Some(amp) = amplitude {
            if self.amplitudes.is_none() {
                // Initialize amplitudes vector with 1.0 for existing spikes
                self.amplitudes = Some(vec![1.0; self.timestamps.len() - 1]);
            }
            self.amplitudes.as_mut().unwrap().push(amp);
        } else if self.amplitudes.is_some() {
            // Add default amplitude if we're tracking amplitudes
            self.amplitudes.as_mut().unwrap().push(1.0);
        }
    }
    
    /// Get the number of spikes in this train
    pub fn len(&self) -> usize {
        self.timestamps.len()
    }
    
    /// Check if the train is empty
    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }
    
    /// Convert to individual spikes
    pub fn to_spikes(&self) -> Result<Vec<Spike>> {
        let mut spikes = Vec::with_capacity(self.len());
        
        for (i, &timestamp) in self.timestamps.iter().enumerate() {
            let amplitude = if let Some(ref amps) = self.amplitudes {
                amps.get(i).copied().unwrap_or(1.0)
            } else {
                1.0
            };
            
            spikes.push(Spike::new(self.source, timestamp, amplitude)?);
        }
        
        Ok(spikes)
    }
    
    /// Sort spikes by timestamp
    pub fn sort_by_time(&mut self) {
        if let Some(ref mut amplitudes) = self.amplitudes {
            // Sort both vectors together
            let mut pairs: Vec<_> = self.timestamps.iter().zip(amplitudes.iter()).collect();
            pairs.sort_by_key(|(time, _)| *time);
            
            self.timestamps = pairs.iter().map(|(time, _)| **time).collect();
            *amplitudes = pairs.iter().map(|(_, amp)| **amp).collect();
        } else {
            self.timestamps.sort();
        }
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
    fn test_spike_creation() {
        let spike = Spike::new(NeuronId::new(0), Time::from_millis(10), 1.0).unwrap();
        assert_eq!(spike.source, NeuronId::new(0));
        assert_eq!(spike.timestamp, Time::from_millis(10));
        assert_eq!(spike.amplitude, 1.0);
        assert!(spike.is_valid());
    }
    
    #[test]
    fn test_spike_walk() {
        let spike = Spike::binary(NeuronId::new(0), Time::from_millis(10)).unwrap();
        let hyperedge_id = crate::hypergraph::HyperedgeId::new(0);
        let mut walk = SpikeWalk::new(SpikeWalkId::new(1), spike, hyperedge_id);
        
        assert_eq!(walk.path_length(), 1);
        assert!(walk.active);
        assert_eq!(walk.current_hyperedge, hyperedge_id);
        
        let next_hyperedge = crate::hypergraph::HyperedgeId::new(1);
        walk.traverse_to(next_hyperedge, Time::from_millis(15));
        
        assert_eq!(walk.path_length(), 2);
        assert_eq!(walk.current_hyperedge, next_hyperedge);
        assert!(walk.has_visited(hyperedge_id));
        assert!(walk.has_visited(next_hyperedge));
    }
    
    #[test]
    fn test_spike_train() {
        let mut train = SpikeTrain::new(NeuronId::new(0));
        train.add_spike(Time::from_millis(10), None);
        train.add_spike(Time::from_millis(20), Some(0.5));
        train.add_spike(Time::from_millis(5), Some(2.0));
        
        assert_eq!(train.len(), 3);
        
        train.sort_by_time();
        let spikes = train.to_spikes().unwrap();
        
        assert_eq!(spikes.len(), 3);
        assert_eq!(spikes[0].timestamp, Time::from_millis(5));
        assert_eq!(spikes[1].timestamp, Time::from_millis(10));
        assert_eq!(spikes[2].timestamp, Time::from_millis(20));
    }
    
    #[test]
    fn test_walk_context() {
        let mut context = WalkContext::new();
        context.add_info(1.5);
        context.set_metadata("key1".to_string(), 2.0);
        
        assert_eq!(context.accumulated_info, 1.5);
        assert_eq!(context.get_metadata("key1"), Some(2.0));
        assert_eq!(context.get_metadata("key2"), None);
    }
}