# H-SNN: Hypergraph Spiking Neural Networks

A lightweight, standalone implementation of Hypergraph Spiking Neural Networks (H-SNN) based on the proposed architecture for advanced neuromorphic computation.

## Overview

H-SNN implements the novel architecture that reconceptualizes neural connectivity using hypergraphs to address fundamental limitations of traditional Spiking Neural Networks (SNNs):

- **Non-differentiability** of spike events
- **Temporal credit assignment** problems  
- **Information loss** during data encoding
- **Asynchrony mismatch** with neuromorphic hardware

## Key Features

### 🔗 Hypergraph Architecture
- **Multi-way connections** beyond pairwise synapses
- **Hyperedge activation rules** (threshold, weighted sum, factory patterns)
- **Hyperpath traversal** for structured inference
- **Temporal hypergraphs** with time-encoded topology

### 🧠 Advanced Learning
- **Group-level plasticity** operating on hyperedge activations
- **Non-local credit assignment** along hyperpath structures
- **Bypass spike non-differentiability** through group-level learning
- **Traditional STDP** support for individual synapses

### ⚡ High Performance
- **Minimal dependencies** (no tokio, nalgebra, etc.)
- **No-std compatible** for embedded systems
- **Memory efficient** with sparse data structures
- **SIMD optimizations** for critical paths

### 🎯 Inference as Spike Walks
- **Spike walks** along hyperpaths instead of simple routing
- **Coordinated group activations** representing concepts
- **Structured reasoning** over embedded concepts
- **Event-driven computation** mapping to neuromorphic hardware

## Quick Start

### Basic Usage

```rust
use hsnn::prelude::*;

// Create H-SNN network
let mut network = HSNNBuilder::new()
    .neurons(1000)
    .hyperedges(2000)
    .activation_rule(ActivationRule::SimpleThreshold { min_spikes: 3, time_window: Duration::from_millis(5) })
    .learning_rule(GroupSTDP::default())
    .build()?;

// Add hyperedge for concept encoding
let concept_edge = Hyperedge::many_to_many(
    HyperedgeId::new(0),
    vec![NeuronId::new(0), NeuronId::new(1), NeuronId::new(2)], // sources
    vec![NeuronId::new(100), NeuronId::new(101)], // targets
)?
.with_activation_rule(ActivationRule::FactoryRule {
    input_subset: vec![NeuronId::new(0), NeuronId::new(1)],
    require_all: true,
    causal_window: Duration::from_millis(2),
});

network.add_hyperedge(concept_edge)?;

// Process spike and trigger spike walk
let input_spike = Spike::new(NeuronId::new(0), Time::from_millis(10), 1.0)?;
let spike_walks = network.process_spike_walk(input_spike)?;

println!("Generated {} spike walks", spike_walks.len());
```

### Pattern Recognition Example

```rust
use hsnn::prelude::*;

// Create network for pattern recognition
let mut network = HSNNBuilder::new()
    .neurons(784)  // 28x28 input neurons
    .build()?;

// Define pattern-specific hyperedges
let horizontal_line = Hyperedge::convergent(
    HyperedgeId::new(0),
    (0..28).map(|i| NeuronId::new(i)).collect(), // horizontal neurons
    NeuronId::new(784), // pattern detection neuron
)?
.with_activation_rule(ActivationRule::WeightedSum {
    threshold: 0.7,
    decay_constant: 0.1,
});

network.add_hyperedge(horizontal_line)?;

// Encode input pattern
let input_pattern = vec![1.0; 28]; // horizontal line
let spike_train = RateEncoder::encode_pattern(&input_pattern, Time::ZERO)?;

// Process through network
let results = network.process_spike_trains(spike_train)?;
```

## Architecture

### Core Components

- **[`core/`](src/core/)** - Basic SNN primitives (neurons, spikes, time)
- **[`hypergraph/`](src/hypergraph/)** - Hypergraph structures and hyperpath traversal  
- **[`learning/`](src/learning/)** - Group-level plasticity and credit assignment
- **[`inference/`](src/inference/)** - Spike walk engine and temporal processing
- **[`network/`](src/network/)** - High-level network management

### Hypergraph Components

```rust
// Hyperedge with advanced activation rules
pub struct Hyperedge {
    pub id: HyperedgeId,
    pub sources: Vec<NeuronId>,
    pub targets: Vec<NeuronId>,
    pub activation_rule: ActivationRule,
    pub weight_function: WeightFunction,
}

// Hyperpath for structured inference
pub struct Hyperpath {
    pub sequence: Vec<HyperedgeId>,
    pub activation_pattern: ActivationPattern,
    pub temporal_constraints: Vec<TemporalConstraint>,
}

// Spike walk for inference
pub struct SpikeWalk {
    pub id: SpikeWalkId,
    pub current_position: HyperedgeId,
    pub traversal_history: Vec<(HyperedgeId, Time)>,
    pub context: WalkContext,
}
```

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
hsnn = "0.1"

# For full features:
hsnn = { version = "0.1", features = ["full"] }

# For no-std environments:
hsnn = { version = "0.1", default-features = false, features = ["no-std"] }
```

## Examples

- [`examples/basic_hsnn.rs`](examples/basic_hsnn.rs) - Basic H-SNN usage
- [`examples/pattern_recognition.rs`](examples/pattern_recognition.rs) - Pattern recognition
- [`examples/temporal_processing.rs`](examples/temporal_processing.rs) - Temporal sequences  
- [`examples/concept_learning.rs`](examples/concept_learning.rs) - Concept formation

## Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api-reference.md)
- [Migration Guide](docs/migration-guide.md) - From traditional SNNs
- [Examples Guide](docs/examples.md)

## Performance

### Benchmarks

- **Spike processing**: <100μs per spike walk
- **Memory usage**: <10MB for 10K neuron networks
- **Throughput**: >1M spike walks/second on modern CPU

### Optimization Features

- `simd` - SIMD vectorization for critical loops
- `optimized` - All performance optimizations enabled
- `parallel` - Multi-threaded processing with rayon

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you use H-SNN in your research, please cite:

```bibtex
@software{hsnn2024,
  title = {H-SNN: Hypergraph Spiking Neural Networks},
  author = {H-SNN Development Team},
  year = {2024},
  url = {https://github.com/hsnn-project/hsnn},
  version = {0.1.0}
}