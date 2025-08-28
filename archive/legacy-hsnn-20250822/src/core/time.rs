//! Time handling and temporal processing for H-SNN
//!
//! This module provides precise time representation adapted from SHNN
//! with additional temporal operations optimized for H-SNN spike walks.

use crate::utils::error::{HSNNError, Result};
use core::fmt;
use core::ops::{Add, Sub, AddAssign, SubAssign};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// High-precision time representation for neuromorphic computation
///
/// Time is represented in nanoseconds to provide sufficient precision
/// for biological time constants while maintaining efficient arithmetic.
/// Adapted directly from SHNN with same interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Time(u64);

impl Time {
    /// Create a new time from nanoseconds
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }
    
    /// Create a new time from microseconds
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros * 1_000)
    }
    
    /// Create a new time from milliseconds
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis * 1_000_000)
    }
    
    /// Create a new time from seconds
    pub const fn from_secs(secs: u64) -> Self {
        Self(secs * 1_000_000_000)
    }
    
    /// Create a new time from floating-point seconds
    pub fn from_secs_f64(secs: f64) -> Result<Self> {
        if secs < 0.0 || !secs.is_finite() {
            return Err(HSNNError::TimeError("Invalid time value".into()));
        }
        Ok(Self((secs * 1_000_000_000.0) as u64))
    }
    
    /// Create a new time from floating-point milliseconds
    pub fn from_millis_f64(millis: f64) -> Result<Self> {
        if millis < 0.0 || !millis.is_finite() {
            return Err(HSNNError::TimeError("Invalid time value".into()));
        }
        Ok(Self((millis * 1_000_000.0) as u64))
    }
    
    /// Zero time constant
    pub const ZERO: Self = Self(0);
    
    /// Maximum representable time
    pub const MAX: Self = Self(u64::MAX);
    
    /// Get nanoseconds
    pub const fn as_nanos(&self) -> u64 {
        self.0
    }
    
    /// Get microseconds
    pub const fn as_micros(&self) -> u64 {
        self.0 / 1_000
    }
    
    /// Get milliseconds
    pub const fn as_millis(&self) -> u64 {
        self.0 / 1_000_000
    }
    
    /// Get seconds
    pub const fn as_secs(&self) -> u64 {
        self.0 / 1_000_000_000
    }
    
    /// Get floating-point seconds
    pub fn as_secs_f64(&self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }
    
    /// Get floating-point milliseconds
    pub fn as_millis_f64(&self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }
    
    /// Get floating-point microseconds
    pub fn as_micros_f64(&self) -> f64 {
        self.0 as f64 / 1_000.0
    }
    
    /// Get nanoseconds as f64
    pub fn as_nanos_f64(&self) -> f64 {
        self.0 as f64
    }
    
    /// Check if time is zero
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }
    
    /// Saturating addition
    pub fn saturating_add(&self, duration: Duration) -> Self {
        Self(self.0.saturating_add(duration.0))
    }
    
    /// Saturating subtraction
    pub fn saturating_sub(&self, duration: Duration) -> Self {
        Self(self.0.saturating_sub(duration.0))
    }
    
    /// Checked addition
    pub fn checked_add(&self, duration: Duration) -> Option<Self> {
        self.0.checked_add(duration.0).map(Self)
    }
    
    /// Checked subtraction
    pub fn checked_sub(&self, duration: Duration) -> Option<Self> {
        self.0.checked_sub(duration.0).map(Self)
    }
}

impl fmt::Display for Time {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display in most appropriate unit
        if self.0 >= 1_000_000_000 {
            write!(f, "{:.3}s", self.as_secs_f64())
        } else if self.0 >= 1_000_000 {
            write!(f, "{:.3}ms", self.as_millis_f64())
        } else if self.0 >= 1_000 {
            write!(f, "{:.3}μs", self.as_micros_f64())
        } else {
            write!(f, "{}ns", self.0)
        }
    }
}

impl Add<Duration> for Time {
    type Output = Self;
    
    fn add(self, rhs: Duration) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub<Duration> for Time {
    type Output = Self;
    
    fn sub(self, rhs: Duration) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Sub<Time> for Time {
    type Output = Duration;
    
    fn sub(self, rhs: Time) -> Self::Output {
        Duration(self.0 - rhs.0)
    }
}

impl AddAssign<Duration> for Time {
    fn add_assign(&mut self, rhs: Duration) {
        self.0 += rhs.0;
    }
}

impl SubAssign<Duration> for Time {
    fn sub_assign(&mut self, rhs: Duration) {
        self.0 -= rhs.0;
    }
}

/// Duration type for representing time intervals
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Duration(u64);

impl Duration {
    /// Create a new duration from nanoseconds
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }
    
    /// Create a new duration from microseconds
    pub const fn from_micros(micros: u64) -> Self {
        Self(micros * 1_000)
    }
    
    /// Create a new duration from milliseconds
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis * 1_000_000)
    }
    
    /// Create a new duration from seconds
    pub const fn from_secs(secs: u64) -> Self {
        Self(secs * 1_000_000_000)
    }
    
    /// Create a new duration from floating-point seconds
    pub fn from_secs_f64(secs: f64) -> Result<Self> {
        if secs < 0.0 || !secs.is_finite() {
            return Err(HSNNError::TimeError("Invalid duration value".into()));
        }
        Ok(Self((secs * 1_000_000_000.0) as u64))
    }
    
    /// Zero duration constant
    pub const ZERO: Self = Self(0);
    
    /// Maximum representable duration
    pub const MAX: Self = Self(u64::MAX);
    
    /// Get nanoseconds
    pub const fn as_nanos(&self) -> u64 {
        self.0
    }
    
    /// Get microseconds
    pub const fn as_micros(&self) -> u64 {
        self.0 / 1_000
    }
    
    /// Get milliseconds
    pub const fn as_millis(&self) -> u64 {
        self.0 / 1_000_000
    }
    
    /// Get seconds
    pub const fn as_secs(&self) -> u64 {
        self.0 / 1_000_000_000
    }
    
    /// Get floating-point seconds
    pub fn as_secs_f64(&self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }
    
    /// Get floating-point milliseconds
    pub fn as_millis_f64(&self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }
    
    /// Check if duration is zero
    pub const fn is_zero(&self) -> bool {
        self.0 == 0
    }
    
    /// Multiply duration by a factor
    pub fn mul_f32(&self, factor: f32) -> Self {
        Self((self.0 as f64 * factor as f64) as u64)
    }
    
    /// Divide duration by a factor
    pub fn div_f32(&self, factor: f32) -> Self {
        Self((self.0 as f64 / factor as f64) as u64)
    }
    
    /// Saturating addition
    pub fn saturating_add(&self, other: Duration) -> Self {
        Self(self.0.saturating_add(other.0))
    }
    
    /// Saturating subtraction
    pub fn saturating_sub(&self, other: Duration) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display in most appropriate unit
        if self.0 >= 1_000_000_000 {
            write!(f, "{:.3}s", self.as_secs_f64())
        } else if self.0 >= 1_000_000 {
            write!(f, "{:.3}ms", self.as_millis_f64())
        } else if self.0 >= 1_000 {
            write!(f, "{:.3}μs", self.0 as f64 / 1_000.0)
        } else {
            write!(f, "{}ns", self.0)
        }
    }
}

impl Add for Duration {
    type Output = Self;
    
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub for Duration {
    type Output = Self;
    
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl AddAssign for Duration {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl SubAssign for Duration {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

/// Time step type for discrete simulation steps
pub type TimeStep = u64;

/// Time utilities for H-SNN specific operations
pub struct TimeUtils;

impl TimeUtils {
    /// Check if two times are within a temporal window
    pub fn within_window(time1: Time, time2: Time, window: Duration) -> bool {
        let diff = if time1 > time2 {
            time1 - time2
        } else {
            time2 - time1
        };
        diff <= window
    }
    
    /// Calculate temporal distance between two times
    pub fn temporal_distance(time1: Time, time2: Time) -> Duration {
        if time1 > time2 {
            time1 - time2
        } else {
            time2 - time1
        }
    }
    
    /// Check if time1 is before time2 within a tolerance
    pub fn is_before_within(time1: Time, time2: Time, tolerance: Duration) -> bool {
        time1 <= time2 && (time2 - time1) <= tolerance
    }
    
    /// Calculate overlap between two temporal windows
    pub fn window_overlap(
        start1: Time,
        end1: Time,
        start2: Time,
        end2: Time,
    ) -> Option<Duration> {
        let overlap_start = start1.max(start2);
        let overlap_end = end1.min(end2);
        
        if overlap_start < overlap_end {
            Some(overlap_end - overlap_start)
        } else {
            None
        }
    }
    
    /// Create a time range iterator
    pub fn time_range(start: Time, end: Time, step: Duration) -> TimeRangeIterator {
        TimeRangeIterator {
            current: start,
            end,
            step,
        }
    }
}

/// Iterator over time ranges
pub struct TimeRangeIterator {
    current: Time,
    end: Time,
    step: Duration,
}

impl Iterator for TimeRangeIterator {
    type Item = Time;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let current = self.current;
            self.current += self.step;
            Some(current)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_time_creation() {
        let time_ns = Time::from_nanos(1_000_000);
        let time_ms = Time::from_millis(1);
        assert_eq!(time_ns, time_ms);
        
        let time_secs = Time::from_secs_f64(0.001).unwrap();
        assert_eq!(time_secs, time_ms);
    }
    
    #[test]
    fn test_time_arithmetic() {
        let time1 = Time::from_millis(100);
        let duration = Duration::from_millis(50);
        
        let time2 = time1 + duration;
        assert_eq!(time2, Time::from_millis(150));
        
        let time3 = time2 - duration;
        assert_eq!(time3, time1);
        
        let diff = time2 - time1;
        assert_eq!(diff, duration);
    }
    
    #[test]
    fn test_duration_operations() {
        let dur1 = Duration::from_millis(100);
        let dur2 = Duration::from_millis(50);
        
        assert_eq!(dur1 + dur2, Duration::from_millis(150));
        assert_eq!(dur1 - dur2, Duration::from_millis(50));
        
        let scaled = dur1.mul_f32(2.0);
        assert_eq!(scaled, Duration::from_millis(200));
    }
    
    #[test]
    fn test_time_utils() {
        let time1 = Time::from_millis(100);
        let time2 = Time::from_millis(105);
        let window = Duration::from_millis(10);
        
        assert!(TimeUtils::within_window(time1, time2, window));
        assert_eq!(TimeUtils::temporal_distance(time1, time2), Duration::from_millis(5));
        assert!(TimeUtils::is_before_within(time1, time2, Duration::from_millis(10)));
    }
    
    #[test]
    fn test_time_range_iterator() {
        let start = Time::from_millis(0);
        let end = Time::from_millis(10);
        let step = Duration::from_millis(2);
        
        let times: Vec<_> = TimeUtils::time_range(start, end, step).collect();
        assert_eq!(times.len(), 5);
        assert_eq!(times[0], Time::from_millis(0));
        assert_eq!(times[4], Time::from_millis(8));
    }
    
    #[test]
    fn test_window_overlap() {
        let overlap = TimeUtils::window_overlap(
            Time::from_millis(10),
            Time::from_millis(20),
            Time::from_millis(15),
            Time::from_millis(25),
        );
        assert_eq!(overlap, Some(Duration::from_millis(5)));
        
        let no_overlap = TimeUtils::window_overlap(
            Time::from_millis(10),
            Time::from_millis(15),
            Time::from_millis(20),
            Time::from_millis(25),
        );
        assert_eq!(no_overlap, None);
    }
}