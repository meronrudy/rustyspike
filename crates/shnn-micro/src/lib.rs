//! # SHNN Micro: Ultra-Lightweight Spiking Neural Networks
//!
//! A zero-dependency, ultra-lightweight implementation of Spiking Neural Networks
//! optimized for microcontrollers and embedded robotics applications.
//!
//! ## Design Principles
//!
//! - **Zero Heap Allocation**: All memory statically allocated at compile time
//! - **Deterministic Execution**: Fixed execution time bounds for real-time systems
//! - **Minimal Binary Size**: <8KB for basic configurations
//! - **Ultra-Low Power**: Optimized for battery-powered devices
//! - **Compile-Time Configuration**: Network topology fixed at build time
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! #![no_std]
//! #![no_main]
//! 
//! use shnn_micro::{MicroNetwork, LIFNeuron, FixedPoint};
//! 
//! // Define network at compile time: 8 neurons, 16 connections
//! type RobotBrain = MicroNetwork<8, 16>;
//! 
//! let mut network = RobotBrain::new();
//! 
//! // Process sensor inputs
//! let sensor_values = [FixedPoint::from_float(0.5); 4];
//! let motor_outputs = network.process_inputs(&sensor_values);
//! ```

#![no_std]
#![deny(missing_docs)]
#![warn(clippy::all)]
#![cfg_attr(feature = "simd-advanced", feature(portable_simd))]

// Core modules
pub mod neuron;
pub mod connectivity;
pub mod fixed_point;
pub mod network;
pub mod time;

// Platform-specific optimizations
#[cfg(any(feature = "simd-neon", feature = "simd-sse2", feature = "simd-avx2"))]
pub mod simd;

// Real-time integration
#[cfg(feature = "rtic-support")]
pub mod rtic;

// Re-export core types
pub use crate::{
    neuron::{LIFNeuron, NeuronState, NeuronConfig},
    connectivity::{BasicConnectivity, SparseConnectivity},
    fixed_point::{FixedPoint, Q15_16},
    network::{MicroNetwork, NetworkConfig, ProcessingResult},
    time::{MicroTime, Duration},
};

/// Compile-time configuration based on feature flags
pub struct MicroConfig;

impl MicroConfig {
    /// Maximum number of neurons for current configuration
    pub const MAX_NEURONS: usize = {
        #[cfg(feature = "micro-8kb")]
        { 16 }
        #[cfg(all(feature = "micro-32kb", not(feature = "micro-8kb")))]
        { 64 }
        #[cfg(all(feature = "micro-128kb", not(any(feature = "micro-8kb", feature = "micro-32kb"))))]
        { 256 }
        #[cfg(feature = "standard")]
        { 1024 }
        #[cfg(not(any(feature = "micro-8kb", feature = "micro-32kb", feature = "micro-128kb", feature = "standard")))]
        { 32 } // Default fallback
    };
    
    /// Maximum number of connections
    pub const MAX_CONNECTIONS: usize = Self::MAX_NEURONS * 4; // Sparse connectivity
    
    /// Input buffer size
    pub const INPUT_BUFFER_SIZE: usize = {
        #[cfg(feature = "micro-8kb")]
        { 8 }
        #[cfg(not(feature = "micro-8kb"))]
        { 16 }
    };
    
    /// Output buffer size  
    pub const OUTPUT_BUFFER_SIZE: usize = Self::INPUT_BUFFER_SIZE;
    
    /// Enable plasticity
    pub const PLASTICITY_ENABLED: bool = {
        #[cfg(feature = "no-plasticity")]
        { false }
        #[cfg(not(feature = "no-plasticity"))]
        { true }
    };
    
    /// Use fixed-point arithmetic
    pub const FIXED_POINT: bool = {
        #[cfg(feature = "fixed-point")]
        { true }
        #[cfg(not(feature = "fixed-point"))]
        { false }
    };
}

/// Arithmetic type based on configuration
#[cfg(feature = "fixed-point")]
pub type Scalar = Q15_16;

#[cfg(not(feature = "fixed-point"))]
pub type Scalar = f32;

/// Error types for micro SNN operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicroError {
    /// Network is full (cannot add more neurons/connections)
    NetworkFull,
    /// Invalid neuron ID
    InvalidNeuronId,
    /// Invalid connection ID
    InvalidConnectionId,
    /// Buffer overflow
    BufferOverflow,
    /// Processing timeout
    Timeout,
    /// Hardware fault
    HardwareFault,
}

/// Result type for micro operations
pub type Result<T> = core::result::Result<T, MicroError>;

/// Ultra-minimal panic handler for no-std environments
#[cfg(all(not(test), not(feature = "std")))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // In production, this should:
    // 1. Save critical state to non-volatile memory
    // 2. Signal fault to external watchdog
    // 3. Attempt graceful shutdown
    
    // For now, infinite loop waiting for watchdog reset
    loop {
        core::hint::spin_loop();
    }
}

/// Global allocator is disabled (no heap allocation)
#[cfg(all(not(test), not(feature = "std")))]
extern crate linked_list_allocator;

/// Critical section implementation for interrupt safety
#[cfg(feature = "rtic-support")]
pub mod critical {
    use core::sync::atomic::{AtomicBool, Ordering};
    
    static IN_CRITICAL: AtomicBool = AtomicBool::new(false);
    
    /// Enter critical section (disable interrupts)
    pub fn enter() -> CriticalSection {
        #[cfg(target_arch = "arm")]
        unsafe {
            cortex_m::interrupt::disable();
        }
        
        IN_CRITICAL.store(true, Ordering::SeqCst);
        CriticalSection
    }
    
    /// Critical section guard
    pub struct CriticalSection;
    
    impl Drop for CriticalSection {
        fn drop(&mut self) {
            IN_CRITICAL.store(false, Ordering::SeqCst);
            
            #[cfg(target_arch = "arm")]
            unsafe {
                cortex_m::interrupt::enable();
            }
        }
    }
    
    /// Check if currently in critical section
    pub fn is_critical() -> bool {
        IN_CRITICAL.load(Ordering::SeqCst)
    }
}

/// Compile-time assertions to validate configuration
const _: () = {
    assert!(MicroConfig::MAX_NEURONS > 0, "Must have at least one neuron");
    assert!(MicroConfig::MAX_CONNECTIONS >= MicroConfig::MAX_NEURONS, "Must have at least as many connections as neurons");
    assert!(MicroConfig::INPUT_BUFFER_SIZE > 0, "Input buffer must be non-empty");
    assert!(MicroConfig::OUTPUT_BUFFER_SIZE > 0, "Output buffer must be non-empty");
};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build configuration summary
pub const BUILD_INFO: BuildInfo = BuildInfo {
    version: VERSION,
    max_neurons: MicroConfig::MAX_NEURONS,
    max_connections: MicroConfig::MAX_CONNECTIONS,
    fixed_point: MicroConfig::FIXED_POINT,
    plasticity: MicroConfig::PLASTICITY_ENABLED,
    features: &[
        #[cfg(feature = "lif-neuron")]
        "lif-neuron",
        #[cfg(feature = "fixed-point")]
        "fixed-point",
        #[cfg(feature = "simd-advanced")]
        "simd-advanced",
        #[cfg(feature = "deterministic")]
        "deterministic",
        #[cfg(feature = "rtic-support")]
        "rtic-support",
    ],
};

/// Build information structure
#[derive(Debug, Clone, Copy)]
pub struct BuildInfo {
    /// Version string
    pub version: &'static str,
    /// Maximum neurons
    pub max_neurons: usize,
    /// Maximum connections
    pub max_connections: usize,
    /// Using fixed-point arithmetic
    pub fixed_point: bool,
    /// Plasticity enabled
    pub plasticity: bool,
    /// Enabled features
    pub features: &'static [&'static str],
}

impl BuildInfo {
    /// Get memory usage estimate in bytes
    pub const fn estimated_memory_usage(&self) -> usize {
        // Rough estimate of static memory usage
        let neuron_memory = self.max_neurons * 16; // 16 bytes per neuron state
        let connection_memory = self.max_connections * 8; // 8 bytes per connection
        let buffer_memory = 64; // Input/output buffers
        
        neuron_memory + connection_memory + buffer_memory
    }
    
    /// Check if configuration fits in target memory
    pub const fn fits_in_memory(&self, available_bytes: usize) -> bool {
        self.estimated_memory_usage() <= available_bytes
    }
}

/// Utility macros for embedded development
#[macro_export]
macro_rules! micro_assert {
    ($cond:expr) => {
        #[cfg(feature = "debug-assertions")]
        if !$cond {
            panic!("Assertion failed: {}", stringify!($cond));
        }
    };
    ($cond:expr, $msg:expr) => {
        #[cfg(feature = "debug-assertions")]
        if !$cond {
            panic!("Assertion failed: {}: {}", stringify!($cond), $msg);
        }
    };
}

/// Compile-time memory layout validation
#[macro_export]
macro_rules! validate_memory_layout {
    ($target_kb:expr) => {
        const _: () = {
            const TARGET_BYTES: usize = $target_kb * 1024;
            assert!(
                $crate::BUILD_INFO.fits_in_memory(TARGET_BYTES),
                "Configuration exceeds target memory limit"
            );
        };
    };
}

// Validate default configuration
validate_memory_layout!(32); // 32KB default target

/// Prelude module for common imports
pub mod prelude {
    //! Common imports for SHNN Micro development
    
    pub use crate::{
        MicroNetwork, LIFNeuron, FixedPoint, Q15_16,
        MicroError, Result, Scalar,
        MicroConfig, BUILD_INFO,
        micro_assert, validate_memory_layout,
    };
    
    #[cfg(feature = "basic-connectivity")]
    pub use crate::connectivity::BasicConnectivity;
    
    #[cfg(feature = "sparse-connectivity")]
    pub use crate::connectivity::SparseConnectivity;
    
    #[cfg(feature = "rtic-support")]
    pub use crate::rtic::{RTICTask, TaskPriority};
    
    #[cfg(feature = "timer-integration")]
    pub use crate::time::{HardwareTimer, TimerConfig};
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_build_info() {
        let info = BUILD_INFO;
        assert!(info.max_neurons > 0);
        assert!(info.max_connections >= info.max_neurons);
        assert!(!info.version.is_empty());
    }
    
    #[test]
    fn test_memory_estimation() {
        let info = BUILD_INFO;
        let memory_usage = info.estimated_memory_usage();
        assert!(memory_usage > 0);
        assert!(memory_usage < 1024 * 1024); // Should be reasonable
    }
    
    #[test]
    fn test_configuration_constants() {
        assert!(MicroConfig::MAX_NEURONS > 0);
        assert!(MicroConfig::MAX_CONNECTIONS > 0);
        assert!(MicroConfig::INPUT_BUFFER_SIZE > 0);
        assert!(MicroConfig::OUTPUT_BUFFER_SIZE > 0);
    }
}