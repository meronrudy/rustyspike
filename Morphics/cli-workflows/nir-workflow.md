# Chapter 3: The NIR Workflow 🧠📝

NIR (Neuromorphic Intermediate Representation) is hSNN's secret weapon—a universal language for describing spiking neural networks that's both human-readable and machine-optimizable. Think of it as the "assembly language" of neuromorphic computing, but designed for humans.

## What Makes NIR Special

Traditional neuromorphic platforms use:
- **Binary formats** you can't read or debug
- **Proprietary languages** that lock you into one tool
- **GUI-only interfaces** that can't be version controlled
- **Platform-specific formats** that don't transfer between systems

NIR is different:
- **Human-readable text** you can edit in any editor
- **Open standard** that works across platforms
- **Version-controllable** like any source code
- **Round-trip guaranteed** - no information loss between text and binary

## Your First NIR Network

Let's start with the simplest possible spiking network:

```nir
// hello-world.nirt - Your first neuromorphic network
module @hello_world {
  // Create a single LIF neuron
  %neuron = neuron.lif<v_th=1.0, v_reset=0.0, tau_mem=20.0>() -> (1,)
  
  // Add input stimulus
  %stimulus = stimuli.poisson<rate=10.0>() -> (1,)
  
  // Connect stimulus to neuron
  %output = connectivity.synapse_connect<weight=0.5>(%stimulus, %neuron) -> (1,)
}
```

**What this does:**
- Creates one LIF neuron with biologically realistic parameters
- Generates random spike input (Poisson process at 10Hz)
- Connects input to neuron with weight 0.5
- Results in irregular output spikes when input accumulates above threshold

**Try it yourself:**
```bash
# Save the above as hello-world.nirt, then:
snn nir verify hello-world.nirt   # Check it's valid
snn nir run hello-world.nirt --duration 100ms --output hello.json
snn viz serve --results-dir . &   # Visualize the spikes
```

## NIR Syntax: The Grammar of Spikes

### Basic Structure
```nir
module @module_name {
  // Operations go here
  %result = operation.type<parameters>(%inputs) -> (shape,)
}
```

**Key elements:**
- `module @name` - Container for your network
- `%variable` - Named results you can reference later
- `operation.type` - What computation to perform
- `<parameters>` - Configuration for the operation
- `(%inputs)` - What data flows into this operation
- `-> (shape,)` - Output shape (number of neurons/channels)

### Core Operations

#### 1. **Neurons** - The computing elements
```nir
// Leaky Integrate-and-Fire neuron
%lif_layer = neuron.lif<
  v_th=1.0,        // Spike threshold
  v_reset=0.0,     // Reset potential after spike
  tau_mem=20.0,    // Membrane time constant (ms)
  tau_ref=2.0      // Refractory period (ms)
>() -> (10,)       // Create 10 neurons

// Izhikevich neuron (more biologically detailed)
%izh_layer = neuron.izhikevich<
  a=0.02,          // Recovery time constant
  b=0.2,           // Sensitivity to sub-threshold oscillations
  c=-65.0,         // After-spike reset value
  d=8.0            // After-spike reset recovery
>() -> (5,)
```

#### 2. **Connectivity** - How neurons connect
```nir
// All-to-all connectivity
%fully_connected = connectivity.fully_connected<
  weight=0.3       // Connection strength
>(%input_layer, %hidden_layer) -> (50,)

// Sparse random connectivity  
%random_sparse = connectivity.random<
  weight=0.5,      // Connection strength
  probability=0.1  // Connection probability (10% connectivity)
>(%hidden_layer, %output_layer) -> (10,)

// Convolutional connectivity (for spatial patterns)
%conv_layer = connectivity.conv2d<
  kernel_size=3,   // 3x3 kernels
  stride=1,        // Step size
  weight=0.4       // Kernel weights
>(%input_2d, %feature_maps) -> (32, 28, 28)
```

#### 3. **Plasticity** - How connections learn
```nir
// Spike-Timing Dependent Plasticity
%plastic_synapses = plasticity.stdp<
  lr_plus=0.01,    // Learning rate for strengthening
  lr_minus=0.008,  // Learning rate for weakening  
  tau_plus=20.0,   // Time constant for strengthening
  tau_minus=20.0   // Time constant for weakening
>(%pre_layer, %post_layer) -> (100,)

// Homeostatic plasticity (maintains stability)
%homeostatic = plasticity.homeostatic<
  target_rate=10.0, // Target firing rate (Hz)
  lr=0.001         // Adaptation learning rate
>(%layer) -> (50,)
```

#### 4. **Stimuli** - Input generation
```nir
// Poisson spike trains (random)
%random_input = stimuli.poisson<
  rate=15.0        // Average spike rate (Hz)
>() -> (20,)

// Regular spike trains (rhythmic)
%regular_input = stimuli.regular<
  interval=50.0    // Spike every 50ms
>() -> (5,)

// Data-driven spikes (from files)
%data_input = stimuli.from_file<
  file="input_data.json",
  format="spike_times"
>() -> (100,)
```

## Building Complex Networks

### Pattern 1: Feedforward Classification
```nir
module @mnist_classifier {
  // Input layer: 28x28 = 784 pixels
  %input = stimuli.from_file<
    file="mnist_spikes.json"
  >() -> (784,)
  
  // Hidden layer: feature detection
  %hidden = neuron.lif<
    v_th=1.0, tau_mem=20.0
  >() -> (128,)
  
  // Output layer: 10 digit classes
  %output = neuron.lif<
    v_th=1.5, tau_mem=30.0  // Higher threshold for decision
  >() -> (10,)
  
  // Plastic connections for learning
  %input_to_hidden = plasticity.stdp<
    lr_plus=0.01, lr_minus=0.008
  >(%input, %hidden) -> (128,)
  
  %hidden_to_output = plasticity.stdp<
    lr_plus=0.005, lr_minus=0.004  // Slower learning in output
  >(%hidden, %output) -> (10,)
}
```

### Pattern 2: Recurrent Memory Network
```nir
module @sequence_memory {
  // Input sequence
  %sequence_input = stimuli.from_file<
    file="temporal_patterns.json"
  >() -> (50,)
  
  // Recurrent layer for temporal memory
  %memory_layer = neuron.lif<
    v_th=1.0, tau_mem=50.0  // Longer memory for temporal integration
  >() -> (100,)
  
  // Feedforward connections
  %input_connections = connectivity.fully_connected<
    weight=0.3
  >(%sequence_input, %memory_layer) -> (100,)
  
  // Recurrent connections (memory)
  %recurrent_connections = connectivity.random<
    weight=0.2, probability=0.2  // Sparse recurrence
  >(%memory_layer, %memory_layer) -> (100,)
  
  // Output predictions
  %prediction_layer = neuron.lif<>() -> (50,)
  %prediction_connections = plasticity.stdp<>(%memory_layer, %prediction_layer) -> (50,)
}
```

### Pattern 3: Reservoir Computing
```nir
module @liquid_state_machine {
  // Input projection
  %input = stimuli.poisson<rate=20.0>() -> (10,)
  
  // Reservoir: random recurrent network
  %reservoir = neuron.lif<
    v_th=1.0, tau_mem=30.0
  >() -> (200,)
  
  // Sparse input connections
  %input_proj = connectivity.random<
    weight=0.5, probability=0.3
  >(%input, %reservoir) -> (200,)
  
  // Dense recurrent connections
  %recurrent = connectivity.random<
    weight=0.1, probability=0.1
  >(%reservoir, %reservoir) -> (200,)
  
  // Trainable readout
  %readout = neuron.lif<>() -> (5,)
  %output_weights = plasticity.stdp<
    lr_plus=0.01
  >(%reservoir, %readout) -> (5,)
}
```

## NIR Best Practices

### 1. **Naming Conventions**
```nir
// Good: Descriptive, hierarchical names
%visual_input_layer = ...
%feature_detection_conv = ...
%decision_output_neurons = ...

// Bad: Cryptic abbreviations
%l1 = ...
%x = ...
%net = ...
```

### 2. **Parameter Documentation**
```nir
module @documented_network {
  // Primary visual input: 32x32 grayscale images
  %visual_input = stimuli.from_file<
    file="camera_data.json"
  >() -> (1024,)  // 32*32 pixels
  
  // Edge detection layer: fast response for motion
  %edge_detectors = neuron.lif<
    v_th=0.8,     // Low threshold for sensitivity
    tau_mem=10.0  // Fast response time
  >() -> (256,)   // 4x fewer neurons than input
}
```

### 3. **Modular Design**
```nir
module @modular_network {
  // Input processing module
  %preprocessed = call @input_preprocessing(%raw_input) -> (100,)
  
  // Feature extraction module  
  %features = call @feature_extraction(%preprocessed) -> (50,)
  
  // Decision module
  %decision = call @classification_head(%features) -> (10,)
}

// Separate module definitions
module @input_preprocessing(%input) -> (100,) {
  %filtered = neuron.lif<v_th=0.5>() -> (100,)
  %connections = connectivity.fully_connected<weight=0.3>(%input, %filtered) -> (100,)
}
```

### 4. **Performance Annotations**
```nir
module @optimized_network {
  // Mark critical paths for optimization
  %critical_layer = neuron.lif<
    v_th=1.0, tau_mem=20.0
  >() -> (1000,) @optimize(parallel=true, memory=low)
  
  // Sparse operations save computation
  %sparse_connections = connectivity.random<
    weight=0.1, probability=0.05  // Only 5% connectivity
  >(%input, %critical_layer) -> (1000,) @optimize(sparse=true)
}
```

## Working with NIR Files

### Development Workflow
```bash
# 1. Write NIR by hand or generate from templates
snn workspace new-network --template feedforward > network.nirt

# 2. Verify syntax and semantics
snn nir verify network.nirt --strict

# 3. Test with small simulation
snn nir run network.nirt --duration 10ms --output test.json

# 4. Iterate on parameters
vim network.nirt  # Edit parameters
snn nir verify network.nirt && snn nir run network.nirt --duration 10ms

# 5. Scale up for full experiment
snn nir run network.nirt --duration 1000ms --output full-run.json
```

### Version Control
```bash
# NIR files are text - perfect for git
git add network.nirt
git commit -m "Add STDP learning to visual layer"

# Compare versions
git diff HEAD~1 network.nirt

# Branch for experiments
git checkout -b experiment/higher-thresholds
# Edit network.nirt to try higher thresholds
git commit -am "Experiment: increase thresholds for sparsity"
```

### Collaboration
```nir
// Leave comments for teammates
module @team_project {
  // TODO: @alice - tune these parameters for stability
  %unstable_layer = neuron.lif<
    v_th=1.2,  // Maybe too high? Causing silence.
    tau_mem=15.0
  >() -> (50,)
  
  // FIXME: @bob - this connectivity seems too dense
  %dense_connections = connectivity.fully_connected<
    weight=0.8  // Very strong - might cause instability
  >(%input, %unstable_layer) -> (50,)
  
  // NOTE: @charlie - this works well, don't change
  %stable_output = neuron.lif<
    v_th=1.0, tau_mem=20.0  // Validated parameters
  >() -> (10,)
}
```

## Advanced NIR Features

### 1. **Conditional Networks**
```nir
module @adaptive_network {
  // Different behaviors based on input statistics
  %input_stats = analyze.statistics<window=100ms>(%input) -> (1,)
  
  %low_activity_path = neuron.lif<v_th=0.5>() -> (50,)
  %high_activity_path = neuron.lif<v_th=1.5>() -> (50,)
  
  // Route based on activity level
  %output = control.conditional<
    condition="input_rate < 20.0"
  >(%input_stats, %low_activity_path, %high_activity_path) -> (50,)
}
```

### 2. **Hierarchical Composition**
```nir
// Import other NIR modules
import @visual_cortex from "modules/visual.nirt"
import @motor_cortex from "modules/motor.nirt"

module @complete_brain {
  %visual_features = call @visual_cortex(%camera_input) -> (100,)
  %motor_commands = call @motor_cortex(%visual_features) -> (10,)
}
```

### 3. **Hardware Targeting**
```nir
module @embedded_network {
  // Constrain for specific hardware
  %efficient_layer = neuron.lif<
    v_th=1.0, tau_mem=20.0
  >() -> (100,) @target(
    platform="neuromorphic_chip",
    cores=4,
    memory_kb=64
  )
}
```

## Debugging NIR Networks

### Common Errors and Solutions

**Syntax Error: "Expected ')' after parameters"**
```nir
// Wrong: Missing comma
%bad = neuron.lif<v_th=1.0 tau_mem=20.0>() -> (10,)

// Right: Comma-separated parameters  
%good = neuron.lif<v_th=1.0, tau_mem=20.0>() -> (10,)
```

**Semantic Error: "Shape mismatch"**
```nir
// Wrong: Output shape doesn't match connection
%layer1 = neuron.lif<>() -> (10,)
%layer2 = neuron.lif<>() -> (20,)
%bad_connection = connectivity.fully_connected<>(%layer1, %layer2) -> (15,)  // Should be (20,)

// Right: Output shape matches post-synaptic layer
%good_connection = connectivity.fully_connected<>(%layer1, %layer2) -> (20,)
```

**Runtime Error: "No spikes generated"**
```nir
// Wrong: Threshold too high, input too weak
%silent = neuron.lif<v_th=10.0>() -> (10,)  // Very high threshold
%weak_input = stimuli.poisson<rate=1.0>() -> (10,)  // Very low rate
%weak_connection = connectivity.fully_connected<weight=0.1>(%weak_input, %silent) -> (10,)

// Right: Balanced parameters
%active = neuron.lif<v_th=1.0>() -> (10,)
%strong_input = stimuli.poisson<rate=20.0>() -> (10,)
%good_connection = connectivity.fully_connected<weight=0.5>(%strong_input, %active) -> (10,)
```

### Debugging Tools
```bash
# Verbose verification
snn nir verify network.nirt --verbose --debug

# Step-by-step execution
snn nir run network.nirt --debug --step-by-step

# Profile performance
snn nir run network.nirt --profile --output profile.json

# Visualize network structure
snn viz export network.nirt --format dot | dot -Tpng > network.png
```

## NIR Extensions and Customization

### Creating Custom Operations
```rust
// In your Rust code, register custom operations
use shnn_ir::OpRegistry;

#[derive(Debug, Clone)]
pub struct CustomNeuron {
    pub threshold: f64,
    pub custom_param: f64,
}

// Register with NIR compiler
OpRegistry::register("neuron.custom", CustomNeuron::from_nir);
```

```nir
// Now use in NIR files
%custom_layer = neuron.custom<
  threshold=1.5,
  custom_param=0.7
>() -> (25,)
```

### Domain-Specific Languages
```nir
// Audio processing DSL
module @audio_pipeline {
  %audio_in = stimuli.microphone<sample_rate=44100>() -> (1,)
  %cochlea = audio.cochlear_filter<frequencies=128>(%audio_in) -> (128,)
  %features = audio.mfcc<coefficients=13>(%cochlea) -> (13,)
}

// Vision processing DSL  
module @vision_pipeline {
  %camera_in = stimuli.camera<resolution="640x480">() -> (307200,)
  %retina = vision.retina_filter<>(%camera_in) -> (307200,)
  %edges = vision.edge_detection<>(%retina) -> (153600,)
}
```

## Performance Optimization

### Memory Optimization
```nir
module @memory_efficient {
  // Use smaller data types where possible
  %compact_layer = neuron.lif<
    v_th=1.0f, tau_mem=20.0f  // 32-bit instead of 64-bit
  >() -> (1000,) @memory(precision=f32)
  
  // Streaming for large datasets
  %streaming_input = stimuli.from_file<
    file="large_dataset.json",
    streaming=true,
    buffer_size=1024
  >() -> (10000,)
}
```

### Compute Optimization
```nir
module @compute_efficient {
  // Sparse operations
  %sparse_layer = connectivity.random<
    weight=0.1, probability=0.01  // 1% connectivity
  >(%input, %hidden) -> (1000,) @compute(sparse=true)
  
  // Parallel processing
  %parallel_layers = neuron.lif<>() -> (10000,) @compute(
    parallel=true,
    threads=8
  )
}
```

## What's Next?

You now understand NIR as both a language and a development tool. You can:
- **Write networks by hand** for precise control
- **Generate networks programmatically** for large-scale experiments
- **Version control your networks** like any source code
- **Debug systematically** using NIR tools
- **Optimize for performance** through annotations

**[Next: Compile, Run, Visualize →](compile-run-viz.md)**

In the next chapter, you'll master the core development cycle: turning your NIR descriptions into running simulations and understanding what your networks are doing through visualization.

---

**Key Takeaways:**
- 🧠 **NIR is human-readable** assembly language for neuromorphic computing
- 📝 **Text-based format** enables version control and collaboration
- 🔧 **Modular design** supports reusable network components
- 🐛 **Built-in verification** catches errors before simulation
- ⚡ **Performance annotations** guide optimization
- 🎯 **Domain-specific extensions** enable specialized applications

**Try This:**
- Write your own NIR network for a specific task
- Convert the networks from Part I into NIR format
- Experiment with different parameter combinations
- Create reusable modules for common patterns
