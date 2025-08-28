# CLI-First SNN Framework Architecture

## Executive Summary

This document outlines the architectural transformation of hSNN into a CLI-first neuromorphic research substrate. The design enables researchers to conduct SNN experiments entirely through command-line interfaces, making the framework accessible to non-programmers while maintaining the performance and flexibility of the underlying Rust implementation.

## Vision Alignment

**Core Principle**: CLI as the primary interface for SNN research, with "invisible infrastructure" handling complexity.

**Key Goals**:
- **Ease**: Experiments runnable without code modification
- **Reproducibility**: Every run generates RUNINFO bundles
- **Flexibility**: Parameter sweeps and ablations via command line
- **Performance**: Real-time visualization and efficient computation

## Current Foundation Analysis

### Reusable Components
- **Type-safe ID system**: `NeuronId(u32)` and `HyperedgeId(u32)` - perfect for unified ID space
- **High-precision timing**: Nanosecond `Time` representation enables precise temporal dynamics
- **Modular connectivity**: `NetworkConnectivity` trait supports pluggable structures
- **Plasticity framework**: `PlasticityRule` trait with existing STDP implementation
- **Zero-dependency architecture**: Custom implementations maximize compatibility
- **Cross-platform support**: no-std compatibility enables embedded deployment

### Architecture Gaps to Address
1. **Storage layer**: Need temporal hypergraph database with CSR snapshots
2. **CLI interface**: No command-line interface currently exists
3. **Visualization**: No real-time rendering or WebGL support
4. **Experiment automation**: No study runner or parameter sweeps
5. **Serialization formats**: Need standardized binary schemas
6. **TTR support**: Task-aware topology reshaping not implemented

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLI Interface Layer                      │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────────────┐ │
│  │   snn   │   hg    │   ttr   │   viz   │      study      │ │
│  │  core   │ hyper-  │ topology│ visual- │   experiment    │ │
│  │commands │ graph   │reshape  │ization  │   automation    │ │
│  └─────────┴─────────┴─────────┴─────────┴─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                 Thin-Waist Contracts                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Binary Schemas: VCSR│VEVT│VMSK│VMORF│VGRF│VRAS        │ │
│  │  Traits: HypergraphStore │ SNNRuntime │ VizEngine      │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     Core Systems                           │
│  ┌─────────────┬─────────────────┬─────────────────────────┐ │
│  │ Hypergraph  │   SNN Runtime   │    Visualization        │ │
│  │  Database   │   (Enhanced)    │       Engine            │ │
│  │             │                 │                         │ │
│  │ • CSR       │ • LIF/STDP      │ • WebGL2 Renderer      │ │
│  │   Snapshots │ • Event-driven  │ • Real-time Streaming  │ │
│  │ • Temporal  │ • Deterministic │ • LOD Management       │ │
│  │   Queries   │ • TTR Support   │ • Binary Formats       │ │
│  └─────────────┴─────────────────┴─────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                Existing shnn-core Foundation               │
│  ┌─────────────┬─────────────┬─────────────┬─────────────┐  │
│  │   Neurons   │   Spikes    │Connectivity │ Plasticity  │  │
│  │   (LIF,     │  (NeuronId, │  (Traits,   │  (STDP,     │  │
│  │ Izhikevich) │   Time)     │ Hypergraph) │  Rules)     │  │
│  └─────────────┴─────────────┴─────────────┴─────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Principles

### 1. Single ID Space
- `neuron_id == node_id` and `synapse_id == edge_id`
- Unified addressing across all system layers
- Efficient indexing and lookup operations

### 2. Thin-Waist Contracts
- Stable binary schemas for zero-copy operations
- Minimal APIs between layers to reduce coupling
- Versioned interfaces for forward compatibility

### 3. Zero/Low-Copy Paths
- Memory-mapped CSR structures
- Append-only event logs for temporal consistency
- Streaming interfaces for real-time processing

### 4. Capability Negotiation
- Features gated behind capability flags
- Runtime detection of available functionality
- Graceful degradation when features unavailable

### 5. Deterministic Options
- Reproducible random number generation
- Ordered computation for consistency
- RUNINFO bundles for experiment provenance

## Binary Schema Specifications

See `BINARY_SCHEMAS.md` for complete specifications of:
- **VCSR**: Versioned Compressed Sparse Row format
- **VEVT**: Event stream format for spikes and control events
- **VMSK**: Mask format for subviews and TTR
- **VMORF**: Morphology operations log
- **VGRF**: Graph frame for visualization
- **VRAS**: Raster/heatmap frame for temporal visualization

## CLI Command Structure

### Core SNN Commands
```bash
# Network initialization and configuration
snn init --neurons 1000 --topology random --connectivity 0.1
snn config --show                # Display current configuration
snn config --set learning_rate=0.01 plasticity=stdp

# Training and evaluation
snn train --epochs 100 --input spikes.dat --output model.bin
snn eval --model model.bin --input test_spikes.dat --metrics accuracy
snn run --model model.bin --realtime --input_stream tcp:8080

# Data import/export
snn export --format vcsr --output network.vcsr
snn import --format spike_times --input spikes.csv
```

### Hypergraph Operations
```bash
# Snapshot management
snn hg snapshot --gen latest --format vcsr --output snapshot.vcsr
snn hg list-snapshots --show-stats
snn hg diff --from gen:100 --to gen:200 --format summary

# Analysis and inspection
snn hg inspect --show-stats --mask active-neurons
snn hg k-hop --source neuron:123 --hops 3 --output subgraph.json
snn hg stats --time-range 0:1000ms --output stats.json
```

### TTR (Task-Aware Topology Reshaping)
```bash
# Phase programs and masks
snn ttr plan --phases phases.toml --output program.ttr
snn ttr apply --program program.ttr --mask active-modules
snn ttr inspect --show-phases --show-masks
snn ttr diff --before snapshot1.vcsr --after snapshot2.vcsr
```

### Visualization
```bash
# Interactive visualization
snn viz serve --port 8080 --realtime --mode structural
snn viz serve --mode temporal --time-range 0:5000ms

# Data export for visualization
snn viz dump --format vgrf --time-range 0:1000ms --output frames.vgrf
snn viz record --duration 10s --output session.vrec
snn viz replay --session session.vrec --speed 2x
```

### Experiment Automation
```bash
# Study management
snn study init --space params.toml --algo bayesian --output study.json
snn study run --budget trials:200 --parallel 4
snn study resume --study study.json --additional-trials 100

# Analysis and reporting
snn study best --study study.json --metric accuracy
snn study report --study study.json --format html --output report.html
snn study compare --studies study1.json study2.json
```

## Implementation Phases

### Phase 1: Foundation (Weeks 1-3)
- **Binary schema implementation**
- **Thin-waist trait definitions**
- **Basic hypergraph database structure**
- **CLI framework with core commands**

### Phase 2: Core Functionality (Weeks 4-7)
- **Enhanced SNN runtime with CLI integration**
- **Event streaming and temporal queries**
- **RUNINFO bundle generation**
- **Basic visualization support**

### Phase 3: Advanced Features (Weeks 8-12)
- **Full visualization engine with WebGL2**
- **TTR implementation with masks and phases**
- **Experiment automation system**
- **Performance optimization**

### Phase 4: Polish and Deploy (Weeks 13-16)
- **Embedded deployment path**
- **Comprehensive testing framework**
- **Documentation and examples**
- **Performance benchmarking**

## Migration Strategy

### Backward Compatibility
- Existing `shnn-core` APIs remain unchanged
- New CLI interfaces supplement rather than replace
- Gradual migration path for users
- Clear deprecation timeline for old patterns

### Integration Points
```rust
// Existing API continues to work
use shnn_core::prelude::*;
let network = NetworkBuilder::new().build()?;

// New CLI-accessible functionality
let db = HypergraphDB::from_network(&network)?;
let cli_config = CLIConfig::from_network_config(&config);
```

### Data Migration
- Automatic conversion from existing data structures
- Export utilities for current users
- Import utilities for external data sources
- Validation tools for data integrity

## Success Metrics

### Technical Metrics
- **Performance**: Real-time visualization for 10K+ neuron networks
- **Scalability**: Handle networks up to 1M neurons efficiently
- **Memory usage**: <10% overhead for storage and visualization
- **Latency**: <1ms response time for interactive commands

### User Experience Metrics
- **CLI completeness**: All one-pager examples work via command line
- **Reproducibility**: 100% of runs generate verifiable RUNINFO bundles
- **Learning curve**: New users can run experiments within 30 minutes
- **Documentation coverage**: >95% of functionality documented with examples

### Research Impact Metrics
- **Iteration speed**: 10x faster experiment setup compared to code-based approaches
- **Reproducibility**: Elimination of "works on my machine" issues
- **Collaboration**: Easier sharing of experimental protocols
- **Discovery**: Reduced friction for trying new approaches

## Conclusion

This architecture transforms hSNN from a library into a complete research substrate where the CLI becomes the experimental language. By building on the solid foundation of the existing codebase while adding the infrastructure for storage, visualization, and automation, we create a system that is both powerful for experts and accessible for newcomers to neuromorphic computing research.

The modular design ensures that each component can be developed and tested independently, while the thin-waist contracts provide stable interfaces that enable future extensions and optimizations.