//! Network statistics and metrics
//!
//! This module provides comprehensive statistics for monitoring
//! neural network performance and behavior.

use crate::time::Time;
use core::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Comprehensive network statistics
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NetworkStats {
    /// Total number of spikes processed
    pub total_spikes_processed: usize,
    /// Total number of spikes generated
    pub total_spikes_generated: usize,
    /// Number of simulation steps executed
    pub simulation_steps: u64,
    /// Number of plasticity updates applied
    pub plasticity_updates: u64,
    /// Current simulation time
    pub current_time: Time,
    /// Performance metrics
    pub performance: PerformanceStats,
    /// Activity metrics
    pub activity: ActivityStats,
    /// Memory usage metrics
    pub memory: MemoryStats,
}

impl NetworkStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Get spike processing rate (spikes per time unit)
    pub fn spike_rate(&self) -> f32 {
        if self.current_time.as_nanos() > 0 {
            (self.total_spikes_processed as f32) / (self.current_time.as_secs_f32())
        } else {
            0.0
        }
    }
    
    /// Get spike generation rate
    pub fn generation_rate(&self) -> f32 {
        if self.current_time.as_nanos() > 0 {
            (self.total_spikes_generated as f32) / (self.current_time.as_secs_f32())
        } else {
            0.0
        }
    }
    
    /// Get plasticity update rate
    pub fn plasticity_rate(&self) -> f32 {
        if self.current_time.as_nanos() > 0 {
            (self.plasticity_updates as f32) / (self.current_time.as_secs_f32())
        } else {
            0.0
        }
    }
    
    /// Get average spikes per step
    pub fn spikes_per_step(&self) -> f32 {
        if self.simulation_steps > 0 {
            (self.total_spikes_processed as f32) / (self.simulation_steps as f32)
        } else {
            0.0
        }
    }
    
    /// Get network efficiency (output/input ratio)
    pub fn efficiency(&self) -> f32 {
        if self.total_spikes_processed > 0 {
            (self.total_spikes_generated as f32) / (self.total_spikes_processed as f32)
        } else {
            0.0
        }
    }
    
    /// Update performance stats
    pub fn update_performance(&mut self, step_duration: Time, memory_usage: usize) {
        self.performance.update_step_time(step_duration);
        self.memory.current_usage = memory_usage;
        
        if memory_usage > self.memory.peak_usage {
            self.memory.peak_usage = memory_usage;
        }
    }
    
    /// Record spike activity
    pub fn record_spike_activity(&mut self, input_count: usize, output_count: usize) {
        self.activity.update_spike_counts(input_count, output_count);
    }
    
    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    
    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "NetworkStats {{ processed: {}, generated: {}, steps: {}, rate: {:.2} Hz, efficiency: {:.2} }}",
            self.total_spikes_processed,
            self.total_spikes_generated,
            self.simulation_steps,
            self.spike_rate(),
            self.efficiency()
        )
    }
}

impl fmt::Display for NetworkStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Performance-related statistics
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceStats {
    /// Average step execution time
    pub avg_step_time: Time,
    /// Minimum step execution time
    pub min_step_time: Time,
    /// Maximum step execution time
    pub max_step_time: Time,
    /// Total computation time
    pub total_computation_time: Time,
    /// Number of performance samples
    sample_count: u64,
    /// Running sum for average calculation
    time_sum: Time,
}

impl PerformanceStats {
    /// Update step timing statistics
    pub fn update_step_time(&mut self, step_time: Time) {
        self.sample_count += 1;
        self.time_sum += step_time.into();
        self.total_computation_time += step_time.into();
        
        self.avg_step_time = Time::from_nanos(self.time_sum.as_nanos() / self.sample_count);
        
        if self.sample_count == 1 || step_time < self.min_step_time {
            self.min_step_time = step_time;
        }
        
        if self.sample_count == 1 || step_time > self.max_step_time {
            self.max_step_time = step_time;
        }
    }
    
    /// Get steps per second
    pub fn steps_per_second(&self) -> f32 {
        if self.avg_step_time.as_nanos() > 0 {
            1.0 / self.avg_step_time.as_secs_f32()
        } else {
            0.0
        }
    }
    
    /// Get performance efficiency (ratio of min to max step time)
    pub fn efficiency(&self) -> f32 {
        if self.max_step_time.as_nanos() > 0 {
            self.min_step_time.as_nanos() as f32 / self.max_step_time.as_nanos() as f32
        } else {
            1.0
        }
    }
}

/// Activity-related statistics
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ActivityStats {
    /// Average input spike count per step
    pub avg_input_spikes: f32,
    /// Average output spike count per step
    pub avg_output_spikes: f32,
    /// Peak input spike count in a single step
    pub peak_input_spikes: usize,
    /// Peak output spike count in a single step
    pub peak_output_spikes: usize,
    /// Total input spike count
    pub total_input_spikes: usize,
    /// Total output spike count
    pub total_output_spikes: usize,
    /// Number of activity samples
    sample_count: u64,
}

impl ActivityStats {
    /// Update spike count statistics
    pub fn update_spike_counts(&mut self, input_count: usize, output_count: usize) {
        self.sample_count += 1;
        self.total_input_spikes += input_count;
        self.total_output_spikes += output_count;
        
        // Update averages
        self.avg_input_spikes = self.total_input_spikes as f32 / self.sample_count as f32;
        self.avg_output_spikes = self.total_output_spikes as f32 / self.sample_count as f32;
        
        // Update peaks
        if input_count > self.peak_input_spikes {
            self.peak_input_spikes = input_count;
        }
        
        if output_count > self.peak_output_spikes {
            self.peak_output_spikes = output_count;
        }
    }
    
    /// Get activity ratio (output/input)
    pub fn activity_ratio(&self) -> f32 {
        if self.total_input_spikes > 0 {
            self.total_output_spikes as f32 / self.total_input_spikes as f32
        } else {
            0.0
        }
    }
    
    /// Get burstiness index (peak/average ratio)
    pub fn burstiness_input(&self) -> f32 {
        if self.avg_input_spikes > 0.0 {
            self.peak_input_spikes as f32 / self.avg_input_spikes
        } else {
            0.0
        }
    }
    
    /// Get output burstiness index
    pub fn burstiness_output(&self) -> f32 {
        if self.avg_output_spikes > 0.0 {
            self.peak_output_spikes as f32 / self.avg_output_spikes
        } else {
            0.0
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
    /// Baseline memory usage (initial allocation)
    pub baseline_usage: usize,
}

impl MemoryStats {
    /// Set baseline memory usage
    pub fn set_baseline(&mut self, baseline: usize) {
        self.baseline_usage = baseline;
        if self.current_usage == 0 {
            self.current_usage = baseline;
        }
        if self.peak_usage < baseline {
            self.peak_usage = baseline;
        }
    }
    
    /// Get memory growth factor
    pub fn growth_factor(&self) -> f32 {
        if self.baseline_usage > 0 {
            self.current_usage as f32 / self.baseline_usage as f32
        } else {
            1.0
        }
    }
    
    /// Get peak memory growth factor
    pub fn peak_growth_factor(&self) -> f32 {
        if self.baseline_usage > 0 {
            self.peak_usage as f32 / self.baseline_usage as f32
        } else {
            1.0
        }
    }
    
    /// Format memory usage as human-readable string
    pub fn format_usage(&self) -> String {
        format_bytes(self.current_usage)
    }
    
    /// Format peak usage as human-readable string
    pub fn format_peak(&self) -> String {
        format_bytes(self.peak_usage)
    }
}

/// Format bytes as human-readable string
fn format_bytes(bytes: usize) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_index = 0;
    
    while size >= 1024.0 && unit_index < UNITS.len() - 1 {
        size /= 1024.0;
        unit_index += 1;
    }
    
    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

/// Statistics collector for running averages and metrics
#[derive(Debug, Clone)]
pub struct StatsCollector {
    /// Network statistics
    stats: NetworkStats,
    /// Collection interval
    collection_interval: Time,
    /// Last collection time
    last_collection: Time,
    /// Whether to collect detailed metrics
    detailed_collection: bool,
}

impl StatsCollector {
    /// Create new statistics collector
    pub fn new(collection_interval: Time) -> Self {
        Self {
            stats: NetworkStats::new(),
            collection_interval,
            last_collection: Time::ZERO,
            detailed_collection: true,
        }
    }
    
    /// Set whether to collect detailed metrics
    pub fn set_detailed_collection(&mut self, detailed: bool) {
        self.detailed_collection = detailed;
    }
    
    /// Update statistics if collection interval has passed
    pub fn update_if_time(&mut self, current_time: Time, network_stats: &NetworkStats) -> bool {
        if current_time >= self.last_collection + crate::time::Duration::from_secs(self.collection_interval.as_secs()) {
            self.stats = network_stats.clone();
            self.last_collection = current_time;
            true
        } else {
            false
        }
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> &NetworkStats {
        &self.stats
    }
    
    /// Generate performance report
    pub fn generate_report(&self) -> String {
        let stats = &self.stats;
        
        format!(
            "Network Performance Report\n\
            ===========================\n\
            Simulation Steps: {}\n\
            Current Time: {:.3}s\n\
            \n\
            Spike Activity:\n\
            - Total Processed: {}\n\
            - Total Generated: {}\n\
            - Processing Rate: {:.2} Hz\n\
            - Generation Rate: {:.2} Hz\n\
            - Efficiency: {:.2}\n\
            \n\
            Performance:\n\
            - Avg Step Time: {:.3}ms\n\
            - Min Step Time: {:.3}ms\n\
            - Max Step Time: {:.3}ms\n\
            - Steps/Second: {:.1}\n\
            \n\
            Memory:\n\
            - Current Usage: {}\n\
            - Peak Usage: {}\n\
            - Growth Factor: {:.2}x\n\
            \n\
            Plasticity:\n\
            - Updates: {}\n\
            - Update Rate: {:.2} Hz\n",
            stats.simulation_steps,
            stats.current_time.as_secs_f32(),
            stats.total_spikes_processed,
            stats.total_spikes_generated,
            stats.spike_rate(),
            stats.generation_rate(),
            stats.efficiency(),
            stats.performance.avg_step_time.as_secs_f32() * 1000.0,
            stats.performance.min_step_time.as_secs_f32() * 1000.0,
            stats.performance.max_step_time.as_secs_f32() * 1000.0,
            stats.performance.steps_per_second(),
            stats.memory.format_usage(),
            stats.memory.format_peak(),
            stats.memory.growth_factor(),
            stats.plasticity_updates,
            stats.plasticity_rate()
        )
    }
}

impl Default for StatsCollector {
    fn default() -> Self {
        Self::new(Time::from_millis(100)) // Collect every 100ms by default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_stats_basic() {
        let mut stats = NetworkStats::new();
        
        assert_eq!(stats.total_spikes_processed, 0);
        assert_eq!(stats.total_spikes_generated, 0);
        assert_eq!(stats.simulation_steps, 0);
        
        stats.total_spikes_processed = 100;
        stats.current_time = Time::from_secs(10);
        
        assert_eq!(stats.spike_rate(), 10.0);
    }
    
    #[test]
    fn test_performance_stats() {
        let mut perf = PerformanceStats::default();
        
        perf.update_step_time(Time::from_millis(1));
        perf.update_step_time(Time::from_millis(2));
        perf.update_step_time(Time::from_millis(3));
        
        assert_eq!(perf.min_step_time, Time::from_millis(1));
        assert_eq!(perf.max_step_time, Time::from_millis(3));
        assert_eq!(perf.avg_step_time, Time::from_millis(2));
    }
    
    #[test]
    fn test_activity_stats() {
        let mut activity = ActivityStats::default();
        
        activity.update_spike_counts(10, 5);
        activity.update_spike_counts(20, 15);
        
        assert_eq!(activity.total_input_spikes, 30);
        assert_eq!(activity.total_output_spikes, 20);
        assert_eq!(activity.peak_input_spikes, 20);
        assert_eq!(activity.avg_input_spikes, 15.0);
    }
    
    #[test]
    fn test_memory_stats() {
        let mut memory = MemoryStats::default();
        
        memory.set_baseline(1000);
        memory.current_usage = 1500;
        memory.peak_usage = 2000;
        
        assert_eq!(memory.growth_factor(), 1.5);
        assert_eq!(memory.peak_growth_factor(), 2.0);
    }
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
    }
    
    #[test]
    fn test_stats_collector() {
        let mut collector = StatsCollector::new(Time::from_millis(100));
        let stats = NetworkStats::new();
        
        // Should not update immediately
        assert!(!collector.update_if_time(Time::from_millis(50), &stats));
        
        // Should update after interval
        assert!(collector.update_if_time(Time::from_millis(100), &stats));
    }
}