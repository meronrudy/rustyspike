//! Utility modules for H-SNN
//!
//! This module provides supporting utilities adapted from SHNN
//! with H-SNN specific extensions.

pub mod error;
pub mod math;
pub mod memory;
pub mod metrics;

// Re-export commonly used types
pub use self::{
    error::{HSNNError, Result},
    math::MathUtils,
    memory::MemoryPool,
    metrics::{NetworkMetrics, PerformanceCounter},
};

// Conditional re-exports based on features
#[cfg(feature = "std")]
pub use std::collections::HashMap;

#[cfg(not(feature = "std"))]
pub use heapless::FnvIndexMap as HashMap;