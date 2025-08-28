# Thin-Waist Trait Definitions

## Overview

This document defines the core trait system that serves as the "thin waist" of the CLI-first SNN framework. These traits provide stable, minimal interfaces between major system components while enabling flexibility in implementation and optimization.

## Design Philosophy

### Minimal Surface Area
- Traits expose only essential operations
- Complex operations built from simple primitives
- Clear separation of concerns between layers

### Zero-Cost Abstractions
- Traits designed for compile-time optimization
- No runtime overhead for common operations
- Efficient default implementations where possible

### Capability-Based Design
- Features gated behind capability traits
- Graceful degradation when capabilities unavailable
- Runtime capability discovery and negotiation

### Future-Proof Interfaces
- Extensible through associated types
- Backward compatibility through versioning
- Clear migration paths for interface evolution

## Core Trait Hierarchy

```rust
// Core type definitions used throughout the system
pub use crate::ids::{NeuronId, HyperedgeId, GenerationId, MaskId};
pub use crate::time::Time;
pub use crate::error::{Result, SHNNError};

/// Base capability trait for feature detection
pub trait Capability {
    const NAME: &'static str;
    const VERSION: u32;
    
    fn is_available(&self) -> bool;
    fn required_capabilities(&self) -> &[&'static str] { &[] }
}
```

## Hypergraph Database Traits

### Core Storage Interface

```rust
/// Primary interface for hypergraph storage and retrieval
pub trait HypergraphStore: Send + Sync {
    /// Snapshot type for graph state at a specific generation
    type Snapshot: HypergraphSnapshot;
    
    /// Error type for storage operations
    type Error: From<SHNNError> + Send + Sync;
    
    /// Get a snapshot of the hypergraph at a specific generation
    fn get_snapshot(&self, generation: GenerationId) -> Result<Self::Snapshot, Self::Error>;
    
    /// Get the latest available generation
    fn latest_generation(&self) -> Result<GenerationId, Self::Error>;
    
    /// Get available generations in a range
    fn list_generations(&self, start: Option<GenerationId>, end: Option<GenerationId>) 
        -> Result<Vec<GenerationId>, Self::Error>;
    
    /// Create a new generation from a previous one with modifications
    fn create_generation(&mut self, base: GenerationId, operations: &[MorphologyOp]) 
        -> Result<GenerationId, Self::Error>;
    
    /// Compact storage by removing intermediate generations
    fn compact(&mut self, keep_generations: &[GenerationId]) -> Result<(), Self::Error>;
}

/// Interface for individual hypergraph snapshots
pub trait HypergraphSnapshot: Send + Sync {
    /// Subview type for masked/filtered views
    type Subview: HypergraphSubview;
    
    /// Iterator type for graph traversal
    type NeighborIter: Iterator<Item = (NeuronId, f32)> + Send;
    type HyperedgeIter: Iterator<Item = (HyperedgeId, &[NeuronId], f32)> + Send;
    
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
    fn active_neurons(&self) -> &[NeuronId];
    fn active_hyperedges(&self) -> &[HyperedgeId];
    fn stats(&self) -> GraphStats;
    
    /// Export subview to various formats
    fn export_vcsr(&self) -> Result<Vec<u8>>;
    fn export_graphml(&self) -> Result<String>;
}

/// Graph statistics for analysis and debugging
#[derive(Debug, Clone, PartialEq)]
pub struct GraphStats {
    pub num_neurons: u32,
    pub num_hyperedges: u32,
    pub num_incidences: u64,
    pub avg_degree: f32,
    pub max_degree: u32,
    pub density: f32,
    pub generation: GenerationId,
    pub timestamp: Time,
}
```

### Temporal Event Interface

```rust
/// Interface for temporal event streams
pub trait EventStore: Send + Sync {
    /// Event type stored in this stream
    type Event: Event + Send + Sync;
    
    /// Iterator type for event traversal
    type EventIter: Iterator<Item = Self::Event> + Send;
    
    /// Error type for event operations
    type Error: From<SHNNError> + Send + Sync;
    
    /// Append events to the stream
    fn append_events(&mut self, events: &[Self::Event]) -> Result<(), Self::Error>;
    
    /// Get events in a time window
    fn time_window(&self, start: Time, end: Time) -> Result<Self::EventIter, Self::Error>;
    
    /// Get events for specific neurons in a time window
    fn neuron_events(&self, neurons: &[NeuronId], start: Time, end: Time) 
        -> Result<Self::EventIter, Self::Error>;
    
    /// Get the total number of events
    fn event_count(&self) -> u64;
    
    /// Get the time range covered by events
    fn time_range(&self) -> Option<(Time, Time)>;
    
    /// Export events to VEVT format
    fn export_vevt(&self, start: Time, end: Time) -> Result<Vec<u8>, Self::Error>;
}

/// Base trait for all events in the system
pub trait Event: Clone + Send + Sync {
    fn timestamp(&self) -> Time;
    fn event_type(&self) -> EventType;
    fn source_id(&self) -> Option<NeuronId>;
    fn serialize(&self) -> Result<Vec<u8>>;
}

/// Event types supported by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    Spike,
    PhaseEnter,
    PhaseExit,
    Neuromodulation,
    Reward,
    Control,
    Marker,
}
```

### Mask Interface

```rust
/// Interface for masks used in subviews and TTR
pub trait Mask: Send + Sync {
    fn mask_id(&self) -> MaskId;
    fn mask_type(&self) -> MaskType;
    fn is_active(&self, id: u32) -> bool;
    fn active_count(&self) -> u64;
    fn total_count(&self) -> u64;
    
    /// Combine with another mask
    fn intersect(&self, other: &dyn Mask) -> Result<Box<dyn Mask>>;
    fn union(&self, other: &dyn Mask) -> Result<Box<dyn Mask>>;
    fn difference(&self, other: &dyn Mask) -> Result<Box<dyn Mask>>;
    
    /// Export to VMSK format
    fn export_vmsk(&self) -> Result<Vec<u8>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskType {
    VertexMask,
    EdgeMask,
    ModuleMask,
    TemporalMask,
    ActivityMask,
}
```

## SNN Runtime Traits

### Core Runtime Interface

```rust
/// Main interface for SNN simulation runtime
pub trait SNNRuntime: Send + Sync {
    /// Configuration type for this runtime
    type Config: RuntimeConfig;
    
    /// State type for runtime inspection
    type State: RuntimeState;
    
    /// Error type for runtime operations
    type Error: From<SHNNError> + Send + Sync;
    
    /// Initialize the runtime with a configuration
    fn initialize(&mut self, config: Self::Config) -> Result<(), Self::Error>;
    
    /// Execute a single simulation step
    fn step(&mut self, dt: Time, inputs: &[SpikeEvent]) -> Result<Vec<SpikeEvent>, Self::Error>;
    
    /// Execute multiple steps efficiently
    fn multi_step(&mut self, steps: u32, dt: Time, input_stream: &mut dyn Iterator<Item = SpikeEvent>) 
        -> Result<Vec<SpikeEvent>, Self::Error>;
    
    /// Get current runtime state
    fn get_state(&self) -> Self::State;
    
    /// Set runtime configuration
    fn set_config(&mut self, config: Self::Config) -> Result<(), Self::Error>;
    
    /// Reset runtime to initial state
    fn reset(&mut self) -> Result<(), Self::Error>;
    
    /// Get performance metrics
    fn metrics(&self) -> RuntimeMetrics;
}

/// Configuration trait for SNN runtimes
pub trait RuntimeConfig: Clone + Send + Sync {
    fn validate(&self) -> Result<()>;
    fn merge(&mut self, other: &Self) -> Result<()>;
    fn to_toml(&self) -> Result<String>;
    fn from_toml(toml: &str) -> Result<Self>;
}

/// State inspection trait for SNN runtimes
pub trait RuntimeState: Clone + Send + Sync {
    fn current_time(&self) -> Time;
    fn step_count(&self) -> u64;
    fn active_neurons(&self) -> &[NeuronId];
    fn neuron_state(&self, id: NeuronId) -> Option<NeuronState>;
    fn network_stats(&self) -> NetworkStats;
}

/// Spike event for runtime processing
#[derive(Debug, Clone, PartialEq)]
pub struct SpikeEvent {
    pub neuron_id: NeuronId,
    pub timestamp: Time,
    pub amplitude: f32,
    pub payload: Vec<u8>,
}

/// Neuron state for inspection
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronState {
    pub membrane_potential: f32,
    pub recovery_variable: f32,
    pub last_spike_time: Option<Time>,
    pub refractory_until: Option<Time>,
}

/// Network statistics for monitoring
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkStats {
    pub total_spikes: u64,
    pub avg_firing_rate: f32,
    pub active_fraction: f32,
    pub total_energy: f32,
}

/// Performance metrics for optimization
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeMetrics {
    pub steps_per_second: f32,
    pub spikes_per_second: f32,
    pub memory_usage_mb: f32,
    pub cpu_utilization: f32,
}
```

### Plasticity Interface

```rust
/// Interface for plasticity rules and adaptation
pub trait PlasticityRule: Send + Sync {
    type Config: Send + Sync + Clone;
    type State: Send + Sync + Clone;
    
    /// Update synaptic weight based on spike timing
    fn update_weight(&self, current_weight: f32, pre_time: Time, post_time: Time, 
                    config: &Self::Config, state: &mut Self::State) -> f32;
    
    /// Initialize plasticity state
    fn init_state(&self, config: &Self::Config) -> Self::State;
    
    /// Reset plasticity state
    fn reset_state(&self, state: &mut Self::State);
    
    /// Get rule name and parameters
    fn name(&self) -> &'static str;
    fn parameters(&self) -> Vec<(&'static str, f32)>;
}

/// Manager for plasticity rules across the network
pub trait PlasticityManager: Send + Sync {
    type Rule: PlasticityRule;
    type Error: From<SHNNError> + Send + Sync;
    
    /// Add a plasticity rule for specific connections
    fn add_rule(&mut self, rule: Self::Rule, mask: Box<dyn Mask>) -> Result<(), Self::Error>;
    
    /// Update all plastic connections
    fn update_weights(&mut self, pre_spikes: &[SpikeEvent], post_spikes: &[SpikeEvent]) 
        -> Result<(), Self::Error>;
    
    /// Get weight changes since last update
    fn weight_changes(&self) -> Vec<(NeuronId, NeuronId, f32)>;
    
    /// Apply weight bounds and constraints
    fn apply_constraints(&mut self) -> Result<(), Self::Error>;
}
```

## Visualization Engine Traits

### Core Visualization Interface

```rust
/// Main interface for visualization engines
pub trait VizEngine: Send + Sync {
    type Frame: VizFrame;
    type Error: From<SHNNError> + Send + Sync;
    
    /// Render a graph frame for visualization
    fn render_graph(&mut self, snapshot: &dyn HypergraphSnapshot, config: &RenderConfig) 
        -> Result<Self::Frame, Self::Error>;
    
    /// Render a temporal raster frame
    fn render_raster(&mut self, events: &dyn EventStore, config: &RasterConfig) 
        -> Result<Self::Frame, Self::Error>;
    
    /// Start real-time streaming visualization
    fn start_stream(&mut self, config: &StreamConfig) -> Result<StreamHandle, Self::Error>;
    
    /// Update streaming visualization with new data
    fn update_stream(&mut self, handle: StreamHandle, data: &StreamData) 
        -> Result<(), Self::Error>;
    
    /// Export frame to various formats
    fn export_frame(&self, frame: &Self::Frame, format: ExportFormat) 
        -> Result<Vec<u8>, Self::Error>;
}

/// Visualization frame interface
pub trait VizFrame: Send + Sync {
    fn frame_id(&self) -> u64;
    fn timestamp(&self) -> Time;
    fn dimensions(&self) -> (u32, u32);
    fn render_mode(&self) -> RenderMode;
    
    /// Export to VGRF format
    fn export_vgrf(&self) -> Result<Vec<u8>>;
    
    /// Export to VRAS format (for raster frames)
    fn export_vras(&self) -> Result<Vec<u8>>;
}

/// Rendering configuration
#[derive(Debug, Clone, PartialEq)]
pub struct RenderConfig {
    pub render_mode: RenderMode,
    pub layout_type: LayoutType,
    pub lod_level: u8,
    pub viewport: (u32, u32),
    pub color_scheme: ColorScheme,
    pub show_weights: bool,
    pub show_labels: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    Structural,
    Incidence,
    Star,
    Bipartite,
    Temporal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutType {
    ForceDirected,
    Hierarchical,
    Circular,
    Grid,
    Spectral,
}

/// Real-time streaming configuration
#[derive(Debug, Clone, PartialEq)]
pub struct StreamConfig {
    pub buffer_size: usize,
    pub update_rate_hz: f32,
    pub compression: bool,
    pub format: StreamFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFormat {
    Binary,
    Json,
    MessagePack,
}

/// Handle for streaming visualization sessions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamHandle(pub u64);

/// Data for streaming updates
pub enum StreamData {
    SpikeEvents(Vec<SpikeEvent>),
    GraphUpdate(Box<dyn HypergraphSnapshot>),
    StateUpdate(Vec<NeuronState>),
    Marker(String),
}
```

## TTR (Task-Aware Topology Reshaping) Traits

### TTR Management Interface

```rust
/// Interface for task-aware topology reshaping
pub trait TTRManager: Send + Sync {
    type Program: TTRProgram;
    type Error: From<SHNNError> + Send + Sync;
    
    /// Load a TTR program
    fn load_program(&mut self, program: Self::Program) -> Result<(), Self::Error>;
    
    /// Execute the current phase of the program
    fn execute_phase(&mut self, runtime: &mut dyn SNNRuntime, 
                    store: &mut dyn HypergraphStore) -> Result<PhaseResult, Self::Error>;
    
    /// Transition to the next phase
    fn next_phase(&mut self) -> Result<Option<PhaseInfo>, Self::Error>;
    
    /// Get current phase information
    fn current_phase(&self) -> Option<PhaseInfo>;
    
    /// Get all available masks
    fn available_masks(&self) -> Vec<MaskId>;
    
    /// Create a new mask from criteria
    fn create_mask(&mut self, criteria: &MaskCriteria) -> Result<MaskId, Self::Error>;
}

/// TTR program definition
pub trait TTRProgram: Send + Sync + Clone {
    fn phases(&self) -> &[Phase];
    fn transitions(&self) -> &[Transition];
    fn validate(&self) -> Result<()>;
    
    /// Export to TOML format
    fn to_toml(&self) -> Result<String>;
    
    /// Import from TOML format
    fn from_toml(toml: &str) -> Result<Self>;
}

/// Phase definition in TTR programs
#[derive(Debug, Clone, PartialEq)]
pub struct Phase {
    pub name: String,
    pub duration: Option<Time>,
    pub max_steps: Option<u64>,
    pub active_masks: Vec<MaskId>,
    pub learning_config: Option<LearningConfig>,
    pub termination_criteria: Vec<TerminationCriterion>,
}

/// Transition between phases
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    pub from_phase: String,
    pub to_phase: String,
    pub trigger: TransitionTrigger,
    pub actions: Vec<TransitionAction>,
}

/// Current phase information
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseInfo {
    pub name: String,
    pub start_time: Time,
    pub elapsed_time: Time,
    pub progress: f32,
    pub active_masks: Vec<MaskId>,
}

/// Result of phase execution
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseResult {
    pub phase_completed: bool,
    pub metrics: PhaseMetrics,
    pub next_phase: Option<String>,
}

/// Metrics collected during phase execution
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseMetrics {
    pub spikes_processed: u64,
    pub weights_updated: u64,
    pub topology_changes: u32,
    pub performance_score: f32,
}
```

## Study Runner Traits

### Experiment Automation Interface

```rust
/// Interface for automated experiment studies
pub trait StudyRunner: Send + Sync {
    type Study: Study;
    type Trial: Trial;
    type Error: From<SHNNError> + Send + Sync;
    
    /// Initialize a new study
    fn init_study(&mut self, config: StudyConfig) -> Result<Self::Study, Self::Error>;
    
    /// Run a study to completion
    fn run_study(&mut self, study: &mut Self::Study) -> Result<StudyResults, Self::Error>;
    
    /// Run a single trial
    fn run_trial(&mut self, study: &Self::Study, params: &TrialParams) 
        -> Result<Self::Trial, Self::Error>;
    
    /// Get study progress
    fn study_progress(&self, study: &Self::Study) -> StudyProgress;
    
    /// Get best trials from a study
    fn best_trials(&self, study: &Self::Study, metric: &str, count: usize) 
        -> Result<Vec<Self::Trial>, Self::Error>;
    
    /// Generate study report
    fn generate_report(&self, study: &Self::Study, format: ReportFormat) 
        -> Result<Vec<u8>, Self::Error>;
}

/// Study configuration and search space
#[derive(Debug, Clone, PartialEq)]
pub struct StudyConfig {
    pub name: String,
    pub search_space: SearchSpace,
    pub optimization: OptimizationConfig,
    pub budget: Budget,
    pub parallelism: u32,
    pub random_seed: Option<u64>,
}

/// Search space definition
#[derive(Debug, Clone, PartialEq)]
pub struct SearchSpace {
    pub parameters: Vec<Parameter>,
    pub constraints: Vec<Constraint>,
}

/// Parameter definition for studies
#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub param_type: ParameterType,
    pub range: ParameterRange,
    pub scale: ParameterScale,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParameterType {
    Float,
    Integer,
    Categorical,
    Boolean,
}

/// Study and trial traits
pub trait Study: Send + Sync + Clone {
    fn study_id(&self) -> String;
    fn config(&self) -> &StudyConfig;
    fn trials(&self) -> &[Box<dyn Trial>];
    fn is_complete(&self) -> bool;
}

pub trait Trial: Send + Sync {
    fn trial_id(&self) -> u64;
    fn parameters(&self) -> &TrialParams;
    fn metrics(&self) -> &TrialMetrics;
    fn status(&self) -> TrialStatus;
    fn duration(&self) -> Time;
}
```

## Integration and Composition

### Capability System

```rust
/// System for managing and discovering capabilities
pub trait CapabilityRegistry: Send + Sync {
    /// Register a capability provider
    fn register<C: Capability + 'static>(&mut self, capability: C) -> Result<()>;
    
    /// Check if a capability is available
    fn has_capability(&self, name: &str) -> bool;
    
    /// Get all available capabilities
    fn available_capabilities(&self) -> Vec<&str>;
    
    /// Check capability dependencies
    fn check_dependencies(&self, required: &[&str]) -> Result<()>;
}

/// Main system compositor that brings all traits together
pub trait SystemCompositor: Send + Sync {
    type Store: HypergraphStore;
    type Runtime: SNNRuntime;
    type VizEngine: VizEngine;
    type TTR: TTRManager;
    type Studies: StudyRunner;
    
    /// Get the hypergraph store
    fn store(&self) -> &Self::Store;
    fn store_mut(&mut self) -> &mut Self::Store;
    
    /// Get the SNN runtime
    fn runtime(&self) -> &Self::Runtime;
    fn runtime_mut(&mut self) -> &mut Self::Runtime;
    
    /// Get the visualization engine
    fn viz_engine(&self) -> &Self::VizEngine;
    fn viz_engine_mut(&mut self) -> &mut Self::VizEngine;
    
    /// Get the TTR manager
    fn ttr_manager(&self) -> &Self::TTR;
    fn ttr_manager_mut(&mut self) -> &mut Self::TTR;
    
    /// Get the study runner
    fn study_runner(&self) -> &Self::Studies;
    fn study_runner_mut(&mut self) -> &mut Self::Studies;
    
    /// Execute a complete experimental workflow
    fn execute_workflow(&mut self, workflow: &Workflow) -> Result<WorkflowResults>;
}
```

## Implementation Guidelines

### Performance Considerations
- All traits designed for zero-cost abstractions
- Async variants available where beneficial
- Batch operations preferred over individual calls
- Memory-efficient default implementations

### Error Handling
- Consistent error types across trait boundaries
- Detailed error context for debugging
- Graceful degradation for missing capabilities
- Clear error recovery strategies

### Testing Strategy
- Mock implementations for all traits
- Property-based testing for trait contracts
- Performance benchmarks for trait implementations
- Integration tests for trait composition

### Documentation Requirements
- Complete API documentation for all traits
- Usage examples for common patterns
- Performance characteristics documentation
- Migration guides for trait evolution

This trait system provides the stable foundation for the CLI-first SNN framework while enabling flexibility in implementation and optimization strategies.