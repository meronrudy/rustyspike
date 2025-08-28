//! Core neuromorphic primitives for H-SNN
//!
//! This module provides the fundamental building blocks adapted from the SHNN
//! codebase with H-SNN specific extensions:
//!
//! - **Neurons**: Biologically realistic neuron models
//! - **Spikes**: Event-driven spike processing with walk extensions
//! - **Time**: High-precision temporal operations
//! - **Encoding**: Spike encoding schemes for input processing

pub mod neuron;
pub mod spike;
pub mod time;
pub mod encoding;

// Re-export commonly used types
pub use self::{
    neuron::{Neuron, NeuronId, NeuronType, LIFNeuron, NeuronPool},
    spike::{Spike, SpikeWalk, SpikeWalkId, TimedSpike},
    time::{Time, Duration, TimeStep},
    encoding::{SpikeEncoder, EncodingType, RateEncoder},
};