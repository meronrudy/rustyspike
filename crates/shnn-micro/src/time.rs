//! Time handling for embedded neural networks
//!
//! Provides ultra-lightweight time representation optimized for deterministic
//! execution on microcontrollers without floating-point or 64-bit arithmetic.

use crate::{MicroError, Result};

/// Ultra-compact time representation for embedded systems
///
/// Uses 32-bit integer with microsecond resolution for deterministic timing.
/// Range: 0 to ~4294 seconds (71 minutes) with 1μs precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct MicroTime(u32);

impl MicroTime {
    /// Zero time constant
    pub const ZERO: Self = Self(0);
    /// Maximum representable time
    pub const MAX: Self = Self(u32::MAX);
    
    /// Create from microseconds
    #[inline(always)]
    pub const fn from_us(microseconds: u32) -> Self {
        Self(microseconds)
    }
    
    /// Create from milliseconds
    #[inline(always)]
    pub const fn from_ms(milliseconds: u32) -> Self {
        Self(milliseconds.saturating_mul(1000))
    }
    
    /// Create from seconds
    #[inline(always)]
    pub const fn from_s(seconds: u32) -> Self {
        Self(seconds.saturating_mul(1_000_000))
    }
    
    /// Get microseconds
    #[inline(always)]
    pub const fn as_us(self) -> u32 {
        self.0
    }
    
    /// Get milliseconds (truncated)
    #[inline(always)]
    pub const fn as_ms(self) -> u32 {
        self.0 / 1000
    }
    
    /// Get seconds (truncated)
    #[inline(always)]
    pub const fn as_s(self) -> u32 {
        self.0 / 1_000_000
    }
    
    /// Add microseconds
    #[inline(always)]
    pub const fn add_us(self, microseconds: u32) -> Self {
        Self(self.0.saturating_add(microseconds))
    }
    
    /// Add milliseconds
    #[inline(always)]
    pub const fn add_ms(self, milliseconds: u32) -> Self {
        Self(self.0.saturating_add(milliseconds.saturating_mul(1000)))
    }
    
    /// Add duration
    #[inline(always)]
    pub const fn add(self, duration: Duration) -> Self {
        Self(self.0.saturating_add(duration.0))
    }
    
    /// Subtract time (returns duration)
    #[inline(always)]
    pub const fn since(self, earlier: Self) -> Duration {
        Duration(self.0.saturating_sub(earlier.0))
    }
    
    /// Current time from system timer (platform-specific)
    #[cfg(feature = "timer-integration")]
    pub fn now() -> Self {
        #[cfg(target_arch = "arm")]
        {
            // Use DWT cycle counter if available
            unsafe {
                let dwt = &*cortex_m::peripheral::DWT::ptr();
                let cyccnt = dwt.cyccnt.read();
                // Assume 80MHz clock for conversion to microseconds
                Self::from_us(cyccnt / 80)
            }
        }
        
        #[cfg(not(target_arch = "arm"))]
        {
            // Fallback for non-ARM platforms
            Self::ZERO
        }
    }
    
    /// Get current time without timer integration (returns zero)
    #[cfg(not(feature = "timer-integration"))]
    pub fn now() -> Self {
        Self::ZERO
    }
}

impl Default for MicroTime {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl From<u32> for MicroTime {
    #[inline(always)]
    fn from(microseconds: u32) -> Self {
        Self::from_us(microseconds)
    }
}

impl From<MicroTime> for u32 {
    #[inline(always)]
    fn from(time: MicroTime) -> Self {
        time.as_us()
    }
}

/// Duration representation for time intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Duration(u32);

impl Duration {
    /// Zero duration
    pub const ZERO: Self = Self(0);
    /// Maximum duration
    pub const MAX: Self = Self(u32::MAX);
    
    /// Create from microseconds
    #[inline(always)]
    pub const fn from_us(microseconds: u32) -> Self {
        Self(microseconds)
    }
    
    /// Create from milliseconds
    #[inline(always)]
    pub const fn from_ms(milliseconds: u32) -> Self {
        Self(milliseconds.saturating_mul(1000))
    }
    
    /// Create from seconds
    #[inline(always)]
    pub const fn from_s(seconds: u32) -> Self {
        Self(seconds.saturating_mul(1_000_000))
    }
    
    /// Get microseconds
    #[inline(always)]
    pub const fn as_us(self) -> u32 {
        self.0
    }
    
    /// Get milliseconds
    #[inline(always)]
    pub const fn as_ms(self) -> u32 {
        self.0 / 1000
    }
    
    /// Get seconds
    #[inline(always)]
    pub const fn as_s(self) -> u32 {
        self.0 / 1_000_000
    }
    
    /// Add duration
    #[inline(always)]
    pub const fn add(self, other: Duration) -> Self {
        Self(self.0.saturating_add(other.0))
    }
    
    /// Subtract duration
    #[inline(always)]
    pub const fn sub(self, other: Duration) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
    
    /// Multiply by integer
    #[inline(always)]
    pub const fn mul(self, factor: u32) -> Self {
        Self(self.0.saturating_mul(factor))
    }
    
    /// Divide by integer
    #[inline(always)]
    pub const fn div(self, divisor: u32) -> Self {
        if divisor == 0 {
            Self::MAX
        } else {
            Self(self.0 / divisor)
        }
    }
}

impl Default for Duration {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

impl From<u32> for Duration {
    #[inline(always)]
    fn from(microseconds: u32) -> Self {
        Self::from_us(microseconds)
    }
}

impl From<Duration> for u32 {
    #[inline(always)]
    fn from(duration: Duration) -> Self {
        duration.as_us()
    }
}

/// Hardware timer abstraction for embedded platforms
#[cfg(feature = "timer-integration")]
pub trait HardwareTimer {
    /// Timer error type
    type Error;
    
    /// Initialize timer
    fn init(&mut self) -> Result<(), Self::Error>;
    
    /// Get current timer value in microseconds
    fn get_time_us(&self) -> u32;
    
    /// Set timer interrupt for specific time
    fn set_interrupt(&mut self, time: MicroTime) -> Result<(), Self::Error>;
    
    /// Clear timer interrupt
    fn clear_interrupt(&mut self);
    
    /// Check if timer interrupt is pending
    fn is_interrupt_pending(&self) -> bool;
}

/// Timer configuration for different platforms
#[cfg(feature = "timer-integration")]
#[derive(Debug, Clone, Copy)]
pub struct TimerConfig {
    /// Timer frequency in Hz
    pub frequency_hz: u32,
    /// Enable interrupts
    pub enable_interrupts: bool,
    /// Auto-reload value
    pub auto_reload: u32,
}

#[cfg(feature = "timer-integration")]
impl Default for TimerConfig {
    fn default() -> Self {
        Self {
            frequency_hz: 1_000_000, // 1MHz for microsecond resolution
            enable_interrupts: false,
            auto_reload: u32::MAX,
        }
    }
}

/// Simple timer implementation for Cortex-M
#[cfg(all(feature = "timer-integration", target_arch = "arm"))]
pub struct CortexMTimer {
    config: TimerConfig,
    start_time: u32,
}

#[cfg(all(feature = "timer-integration", target_arch = "arm"))]
impl CortexMTimer {
    /// Create new timer
    pub fn new(config: TimerConfig) -> Self {
        Self {
            config,
            start_time: 0,
        }
    }
}

#[cfg(all(feature = "timer-integration", target_arch = "arm"))]
impl HardwareTimer for CortexMTimer {
    type Error = ();
    
    fn init(&mut self) -> Result<(), Self::Error> {
        // Initialize DWT cycle counter
        unsafe {
            let mut core = cortex_m::Peripherals::steal();
            core.DWT.enable_cycle_counter();
            self.start_time = core.DWT.cyccnt.read();
        }
        Ok(())
    }
    
    fn get_time_us(&self) -> u32 {
        unsafe {
            let dwt = &*cortex_m::peripheral::DWT::ptr();
            let current_cycles = dwt.cyccnt.read();
            let elapsed_cycles = current_cycles.wrapping_sub(self.start_time);
            // Convert cycles to microseconds (assuming known CPU frequency)
            elapsed_cycles / (self.config.frequency_hz / 1_000_000)
        }
    }
    
    fn set_interrupt(&mut self, _time: MicroTime) -> Result<(), Self::Error> {
        // Would configure timer interrupt here
        Ok(())
    }
    
    fn clear_interrupt(&mut self) {
        // Would clear timer interrupt here
    }
    
    fn is_interrupt_pending(&self) -> bool {
        // Would check timer interrupt status here
        false
    }
}

/// Timing utilities for deterministic execution
pub struct TimingUtils;

impl TimingUtils {
    /// Measure execution time of a function
    #[cfg(feature = "timer-integration")]
    pub fn measure<F, R>(f: F) -> (R, Duration)
    where
        F: FnOnce() -> R,
    {
        let start = MicroTime::now();
        let result = f();
        let end = MicroTime::now();
        (result, end.since(start))
    }
    
    /// Sleep for specified duration (busy wait)
    pub fn delay_us(microseconds: u32) {
        let target_cycles = microseconds * 80; // Assuming 80MHz CPU
        let start = Self::get_cycle_count();
        
        while Self::get_cycle_count().wrapping_sub(start) < target_cycles {
            core::hint::spin_loop();
        }
    }
    
    /// Sleep for milliseconds
    pub fn delay_ms(milliseconds: u32) {
        Self::delay_us(milliseconds * 1000);
    }
    
    /// Get CPU cycle count (platform-specific)
    #[cfg(target_arch = "arm")]
    fn get_cycle_count() -> u32 {
        unsafe {
            let dwt = &*cortex_m::peripheral::DWT::ptr();
            dwt.cyccnt.read()
        }
    }
    
    /// Fallback cycle count for non-ARM platforms
    #[cfg(not(target_arch = "arm"))]
    fn get_cycle_count() -> u32 {
        0
    }
    
    /// Check if deadline is met
    pub fn check_deadline(start_time: MicroTime, deadline_us: u32) -> bool {
        let current = MicroTime::now();
        current.since(start_time).as_us() <= deadline_us
    }
    
    /// Wait until specific time
    pub fn wait_until(target_time: MicroTime) {
        while MicroTime::now() < target_time {
            core::hint::spin_loop();
        }
    }
}

/// Real-time constraints for deterministic execution
#[cfg(feature = "deterministic")]
pub struct RTConstraints {
    /// Maximum allowed execution time per step
    pub max_step_time_us: u32,
    /// Maximum allowed jitter
    pub max_jitter_us: u32,
    /// Target step frequency
    pub target_frequency_hz: u32,
}

#[cfg(feature = "deterministic")]
impl Default for RTConstraints {
    fn default() -> Self {
        Self {
            max_step_time_us: 800,  // 800μs for 1kHz with margin
            max_jitter_us: 50,      // ±50μs jitter tolerance
            target_frequency_hz: 1000, // 1kHz default
        }
    }
}

#[cfg(feature = "deterministic")]
impl RTConstraints {
    /// Check if execution meets real-time constraints
    pub fn check_constraints(&self, execution_time: Duration, jitter: Duration) -> Result<()> {
        if execution_time.as_us() > self.max_step_time_us {
            return Err(MicroError::Timeout);
        }
        
        if jitter.as_us() > self.max_jitter_us {
            return Err(MicroError::Timeout); // Using Timeout as jitter error
        }
        
        Ok(())
    }
    
    /// Get target step duration
    pub fn step_duration(&self) -> Duration {
        Duration::from_us(1_000_000 / self.target_frequency_hz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_micro_time_basic() {
        let time = MicroTime::from_ms(100);
        assert_eq!(time.as_ms(), 100);
        assert_eq!(time.as_us(), 100_000);
        
        let time2 = time.add_ms(50);
        assert_eq!(time2.as_ms(), 150);
    }
    
    #[test]
    fn test_duration_arithmetic() {
        let d1 = Duration::from_ms(100);
        let d2 = Duration::from_ms(50);
        
        let sum = d1.add(d2);
        assert_eq!(sum.as_ms(), 150);
        
        let diff = d1.sub(d2);
        assert_eq!(diff.as_ms(), 50);
        
        let product = d1.mul(3);
        assert_eq!(product.as_ms(), 300);
        
        let quotient = d1.div(2);
        assert_eq!(quotient.as_ms(), 50);
    }
    
    #[test]
    fn test_time_since() {
        let start = MicroTime::from_ms(100);
        let end = MicroTime::from_ms(150);
        
        let duration = end.since(start);
        assert_eq!(duration.as_ms(), 50);
    }
    
    #[test]
    fn test_saturation() {
        let max_time = MicroTime::MAX;
        let added = max_time.add_ms(1000);
        assert_eq!(added, MicroTime::MAX); // Should saturate
        
        let zero_time = MicroTime::ZERO;
        let duration = zero_time.since(MicroTime::from_ms(100));
        assert_eq!(duration, Duration::ZERO); // Should saturate to zero
    }
    
    #[test]
    fn test_conversions() {
        let time_us: u32 = MicroTime::from_ms(5).into();
        assert_eq!(time_us, 5000);
        
        let time_from_us = MicroTime::from(10000u32);
        assert_eq!(time_from_us.as_ms(), 10);
    }
    
    #[test]
    fn test_duration_constants() {
        assert_eq!(Duration::ZERO.as_us(), 0);
        assert_eq!(Duration::MAX.as_us(), u32::MAX);
        
        assert_eq!(MicroTime::ZERO.as_us(), 0);
        assert_eq!(MicroTime::MAX.as_us(), u32::MAX);
    }
    
    #[cfg(feature = "deterministic")]
    #[test]
    fn test_rt_constraints() {
        let constraints = RTConstraints::default();
        
        let good_execution = Duration::from_us(500);
        let good_jitter = Duration::from_us(20);
        assert!(constraints.check_constraints(good_execution, good_jitter).is_ok());
        
        let bad_execution = Duration::from_us(1000);
        let bad_jitter = Duration::from_us(100);
        assert!(constraints.check_constraints(bad_execution, bad_jitter).is_err());
    }
    
    #[test]
    fn test_timing_utils() {
        // Test delay functions (they should complete without errors)
        TimingUtils::delay_us(10);
        TimingUtils::delay_ms(1);
        
        // Test cycle count (should return consistent values)
        let cycles1 = TimingUtils::get_cycle_count();
        let cycles2 = TimingUtils::get_cycle_count();
        assert!(cycles2 >= cycles1); // Should be monotonic
    }
}