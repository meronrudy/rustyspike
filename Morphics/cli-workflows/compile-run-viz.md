# Chapter 4: Compile, Run, Visualize ⚡🔍

This is the heartbeat of neuromorphic development—the rapid cycle that turns ideas into insights. In traditional AI, this cycle takes hours or days. With hSNN's CLI, it takes seconds.

## The Three-Step Dance

Every neuromorphic developer learns this rhythm:

```bash
# 1. COMPILE: Turn description into executable network
snn nir compile --output network.nirt

# 2. RUN: Execute the network and collect data  
snn nir run network.nirt --output results.json

# 3. VISUALIZE: See what happened
snn viz serve --results-dir .
```

**Why this works:**
- **Fast feedback loop:** See results in under 30 seconds
- **Incremental development:** Change one thing, test immediately
- **Visual debugging:** Spot problems instantly in spike patterns
- **Reproducible:** Every step is documented and repeatable

## Step 1: Compile - From Idea to Executable

The compile step transforms high-level descriptions into optimized neuromorphic networks.

### Basic Compilation
```bash
# Generate a simple feedforward network
snn nir compile \
  --neurons lif \
  --topology "10->20->5" \
  --plasticity stdp \
  --output basic-network.nirt

# What this creates:
# - 10 input neurons (LIF type)
# - 20 hidden neurons (LIF type) 
# - 5 output neurons (LIF type)
# - Fully connected layers with STDP learning
# - All with sensible default parameters
```

### Advanced Compilation Options
```bash
# Detailed parameter control
snn nir compile \
  --neurons lif \
  --neuron-params "v_th=1.2,tau_mem=25.0,tau_ref=3.0" \
  --topology "100->50->10" \
  --plasticity stdp \
  --plasticity-params "lr_plus=0.01,lr_minus=0.008" \
  --connectivity sparse \
  --sparsity 0.1 \
  --output detailed-network.nirt

# This creates:
# - Custom neuron parameters for specific behavior
# - Sparse connectivity (10% connections) for efficiency
# - Tuned learning rates for stability
```

### Compilation from Configuration Files
```bash
# Use workspace configuration
snn nir compile \
  --config experiments/pattern-recognition.toml \
  --output networks/pattern-net.nirt

# Override specific parameters
snn nir compile \
  --config experiments/pattern-recognition.toml \
  --override "plasticity.lr_plus=0.02" \
  --output networks/fast-learning.nirt
```

**Example configuration file** ([`experiments/pattern-recognition.toml`](file://crates/shnn-cli/test_workspace/experiments/example.toml)):
```toml
[experiment]
name = "pattern_recognition"
description = "Learn visual patterns with STDP"

[network]
inputs = 784      # 28x28 image
hidden = 100      # Feature detectors
outputs = 10      # Digit classes
topology = "fully-connected"

[neurons]
type = "lif"
v_th = 1.0
tau_mem = 20.0
tau_ref = 2.0

[plasticity] 
type = "stdp"
lr_plus = 0.01
lr_minus = 0.008
tau_plus = 20.0
tau_minus = 20.0

[simulation]
dt_us = 100       # 100 microsecond timesteps
duration_ms = 1000
```

### Verification During Compilation
```bash
# Strict verification (catches more potential issues)
snn nir compile \
  --config my-experiment.toml \
  --verify strict \
  --output verified-network.nirt

# What gets checked:
# - Parameter ranges are biologically plausible
# - Network connectivity won't cause instabilities  
# - Memory requirements are reasonable
# - Performance estimates are provided
```

**Example verification output:**
```
✅ Network structure valid
✅ Parameter ranges acceptable
⚠️  Warning: High connectivity may cause synchronization
✅ Estimated memory usage: 12.3 MB
✅ Estimated simulation speed: 1.2x real-time
📊 Network statistics:
   - Total neurons: 894
   - Total synapses: 87,400
   - Sparsity: 10.8%
   - Critical path depth: 3 layers
```

## Step 2: Run - Execute and Collect

The run step executes your compiled network and collects all the data you need for analysis.

### Basic Execution
```bash
# Run for 1 second of simulated time
snn nir run network.nirt \
  --duration 1000ms \
  --output results/run-001.json

# Quick test run
snn nir run network.nirt \
  --duration 100ms \
  --output quick-test.json
```

### Input Stimuli
```bash
# Use built-in stimulus patterns
snn nir run network.nirt \
  --stimulus poisson \
  --stimulus-rate 20.0 \
  --duration 500ms \
  --output poisson-input.json

# Load stimulus from file
snn nir run network.nirt \
  --stimulus-file data/mnist-spikes.json \
  --duration 1000ms \
  --output mnist-results.json

# Real-time input (from sensors)
snn nir run network.nirt \
  --stimulus realtime \
  --input-device /dev/audio0 \
  --duration 5000ms \
  --output realtime-audio.json
```

### Data Collection Options
```bash
# Collect everything (for debugging)
snn nir run network.nirt \
  --duration 1000ms \
  --record spikes,potentials,weights,plasticity \
  --output debug-run.json

# Minimal collection (for performance)
snn nir run network.nirt \
  --duration 10000ms \
  --record spikes \
  --subsample 10 \
  --output performance-run.json

# Targeted collection (specific neurons)
snn nir run network.nirt \
  --duration 1000ms \
  --record spikes \
  --neurons "output_layer,hidden_layer[0:10]" \
  --output targeted-run.json
```

### Performance Monitoring
```bash
# Profile execution performance
snn nir run network.nirt \
  --duration 1000ms \
  --profile cpu,memory,cache \
  --output results.json \
  --profile-output performance.json

# Parallel execution
snn nir run network.nirt \
  --duration 1000ms \
  --threads 8 \
  --batch-size 1000 \
  --output parallel-results.json

# Real-time execution constraints
snn nir run network.nirt \
  --duration 1000ms \
  --realtime \
  --max-latency 1ms \
  --output realtime-results.json
```

### Output Data Format

The results JSON contains structured data about your simulation:

```json
{
  "metadata": {
    "network_file": "network.nirt",
    "duration_ms": 1000,
    "dt_us": 100,
    "timestamp": "2025-01-15T10:30:00Z",
    "version": "hSNN-0.1.0"
  },
  "network_info": {
    "total_neurons": 135,
    "total_synapses": 1340,
    "layers": ["input", "hidden", "output"]
  },
  "simulation_stats": {
    "total_spikes": 12847,
    "spike_rate_hz": 12.847,
    "simulation_time_ms": 342,
    "real_time_factor": 2.92
  },
  "spikes": [
    {"neuron_id": 0, "time_ms": 23.4, "layer": "input"},
    {"neuron_id": 15, "time_ms": 24.1, "layer": "hidden"},
    {"neuron_id": 132, "time_ms": 26.7, "layer": "output"}
  ],
  "final_weights": {
    "input_to_hidden": [[0.34, 0.12, ...], ...],
    "hidden_to_output": [[0.67, 0.23, ...], ...]
  }
}
```

## Step 3: Visualize - Understand What Happened

Visualization turns data into insights. hSNN's visualization tools let you see your network thinking in real-time.

### Quick Visualization
```bash
# Start visualization server
snn viz serve --results-dir results/ --port 8080 &

# Open browser to http://localhost:8080
# Instantly see:
# - Spike raster plots
# - Network activity over time  
# - Weight evolution
# - Performance metrics
```

### Advanced Visualization Options
```bash
# High-resolution visualization
snn viz serve \
  --results-dir results/ \
  --resolution high \
  --frame-rate 60 \
  --port 8080

# Custom visualization templates
snn viz serve \
  --results-dir results/ \
  --template neuroscience \
  --colormap viridis \
  --port 8080

# Export static plots
snn viz plot results/experiment.json \
  --type raster,weights,activity \
  --output plots/ \
  --format png,svg
```

### Real-Time Visualization
```bash
# Visualize as simulation runs
snn nir run network.nirt \
  --duration 10000ms \
  --streaming \
  --output-stream tcp://localhost:9090 &

snn viz serve \
  --input-stream tcp://localhost:9090 \
  --realtime \
  --port 8080
```

### Understanding Visualization Outputs

#### 1. **Spike Raster Plot**
```
Neuron ID
    ↑
134 |  •     •        •     •  •
133 |     •     •  •     •      
132 |  •  •  •     •        •  •
131 |     •        •  •     •   
...
 15 |  •        •     •  •     •
 14 |     •  •     •        •  
 13 |  •     •        •     •  •
...
  2 |  •  •     •        •     •
  1 |     •        •  •        •
  0 |  •     •  •     •     •  •
    +---------------------------→
    0   100  200  300  400  500  Time (ms)
```

**What to look for:**
- **Dense vertical lines:** Network-wide synchronization (often bad)
- **Horizontal bands:** Individual neurons firing regularly
- **Sparse, random dots:** Healthy asynchronous activity
- **Silent regions:** Neurons not receiving enough input

#### 2. **Network Activity Heatmap**
```
Layer Activity Over Time
       0ms   100ms  200ms  300ms  400ms  500ms
Input  ████   ████   ████   ████   ████   ████  
Hidden ████   ████   █████  █████  ████   ████
Output ████   █████  █████  ████   █████  ████

Color: █ = 0-5 Hz, ██ = 5-15 Hz, ███ = 15-30 Hz, ████ = 30+ Hz
```

**What to look for:**
- **Gradual activity propagation:** Information flowing through layers
- **Sudden activity bursts:** Network responding to strong inputs
- **Oscillating patterns:** Rhythmic processing or instabilities
- **Layer-specific patterns:** Different processing in each layer

#### 3. **Weight Evolution Plot**
```
Average Synaptic Weights Over Time
Weight
  1.0 ┤                          
  0.9 ┤      ╭─────────────────╮
  0.8 ┤    ╭─╯                 ╰╮
  0.7 ┤  ╭─╯                   ╰╮
  0.6 ┤╭─╯                     ╰─╮
  0.5 ┼╯                        ╰─
  0.4 ┤
  0.3 ┤  Input→Hidden
  0.2 ┤  Hidden→Output
  0.1 ┤
  0.0 ┤
      └─────────────────────────────→
      0   200   400   600   800  1000 Time (ms)
```

**What to look for:**
- **Gradual weight increases:** Successful learning
- **Rapid weight changes:** Instability or overfitting
- **Plateauing weights:** Learning has converged
- **Weight oscillations:** Network struggling to learn

### Debugging Through Visualization

#### Problem: "Network is completely silent"
**Symptoms in visualization:**
- Empty spike raster plot
- Zero activity in all layers
- No weight changes

**Debugging steps:**
```bash
# Check input stimulus
snn nir run network.nirt --duration 100ms --record spikes --stimulus-debug

# Look for:
# - Are input neurons receiving spikes?
# - Are neuron thresholds too high?
# - Are initial weights too small?
```

#### Problem: "All neurons fire at the same time"
**Symptoms in visualization:**
- Vertical lines in raster plot
- Synchronized activity bursts
- Unstable learning

**Debugging steps:**
```bash
# Check connectivity and parameters
snn nir verify network.nirt --analyze-stability

# Common fixes:
# - Reduce connection weights
# - Add more randomness to initial conditions
# - Increase membrane time constants
```

#### Problem: "Learning isn't happening"
**Symptoms in visualization:**
- Flat weight evolution
- No performance improvement
- Random-looking activity

**Debugging steps:**
```bash
# Check plasticity parameters
snn nir run network.nirt --duration 1000ms --record plasticity --debug-learning

# Look for:
# - Are pre and post spikes correlated?
# - Are learning rates appropriate?
# - Is the network getting useful feedback?
```

## Complete Example Workflow

Let's put it all together with a realistic example: building a sound pattern classifier.

### 1. Design and Compile
```bash
# Create project structure
mkdir sound-classifier && cd sound-classifier
mkdir networks data results plots

# Define network configuration
cat > config/sound-net.toml << EOF
[experiment]
name = "sound_pattern_classifier"
description = "Classify different audio patterns using STDP"

[network]
inputs = 64       # Frequency bins from audio preprocessing  
hidden = 128      # Feature detection layer
outputs = 4       # Four sound categories

[neurons]
type = "lif"
v_th = 1.0
tau_mem = 20.0    # Good for audio timescales
tau_ref = 2.0

[plasticity]
type = "stdp"
lr_plus = 0.01
lr_minus = 0.008
tau_plus = 15.0   # Faster for audio patterns
tau_minus = 15.0
EOF

# Compile the network
snn nir compile \
  --config config/sound-net.toml \
  --verify strict \
  --output networks/sound-classifier.nirt
```

### 2. Prepare Training Data
```bash
# Convert audio files to spike trains
snn data convert \
  --input-dir audio/training/ \
  --format spike-times \
  --preprocessing cochlear \
  --output data/training-spikes.json

# Verify data format
snn data validate data/training-spikes.json
```

### 3. Training Phase
```bash
# Initial training run
snn nir run networks/sound-classifier.nirt \
  --stimulus-file data/training-spikes.json \
  --duration 5000ms \
  --record spikes,weights \
  --output results/training-phase1.json

# Check initial results
snn viz serve --results-dir results/ --port 8080 &
# → Open browser, look at learning progress

# If learning looks good, continue training
snn nir run networks/sound-classifier.nirt \
  --stimulus-file data/training-spikes.json \
  --duration 20000ms \
  --load-weights results/training-phase1.json \
  --record spikes,weights \
  --output results/training-final.json
```

### 4. Testing Phase
```bash
# Test on unseen data
snn nir run networks/sound-classifier.nirt \
  --stimulus-file data/test-spikes.json \
  --duration 2000ms \
  --load-weights results/training-final.json \
  --record spikes \
  --output results/test-performance.json

# Generate classification report
snn analyze classification results/test-performance.json \
  --ground-truth data/test-labels.json \
  --output reports/classification-report.html
```

### 5. Visualization and Analysis
```bash
# Create comprehensive visualization
snn viz serve \
  --results-dir results/ \
  --template classification \
  --comparison-mode training,testing \
  --port 8080

# Export publication-quality plots
snn viz plot results/training-final.json \
  --type learning-curve,confusion-matrix,spike-raster \
  --style publication \
  --output plots/ \
  --format pdf

# Generate summary report
snn study report results/ \
  --template sound-classification \
  --include-plots plots/ \
  --output reports/final-report.html
```

### 6. Optimization Iteration
```bash
# Try different parameters
for lr in 0.005 0.01 0.02; do
  # Modify config
  sed "s/lr_plus = .*/lr_plus = $lr/" config/sound-net.toml > config/sound-net-lr${lr}.toml
  
  # Compile and test
  snn nir compile --config config/sound-net-lr${lr}.toml --output networks/sound-net-lr${lr}.nirt
  snn nir run networks/sound-net-lr${lr}.nirt --stimulus-file data/training-spikes.json --duration 5000ms --output results/lr${lr}-training.json
done

# Compare all results
snn study compare results/lr*-training.json \
  --metric learning-speed,final-accuracy \
  --output reports/parameter-sweep.html
```

## Performance Tips

### Compilation Optimization
```bash
# Pre-compile for faster iteration
snn nir compile --config base-config.toml --cache --output base-network.nirt

# Use cached compilation for parameter sweeps
snn nir compile --config base-config.toml --override "plasticity.lr_plus=0.02" --use-cache --output variant.nirt
```

### Execution Optimization
```bash
# Use binary format for faster loading
snn nir run network.nirt --binary-output results.bin

# Parallel execution for batch processing
snn batch run experiments/*.toml --parallel 8 --output-dir batch-results/
```

### Visualization Optimization
```bash
# Pre-process for faster visualization
snn viz preprocess results/large-simulation.json --output results/large-simulation.viz

# Use streaming for real-time display
snn viz serve --streaming --buffer-size 1000 --port 8080
```

## What You've Mastered

The compile-run-visualize cycle is now your superpower:

- **⚡ Rapid iteration:** Ideas to insights in seconds
- **🔍 Visual debugging:** See problems immediately  
- **📊 Data-driven development:** Let the spikes guide your decisions
- **🔄 Systematic optimization:** Methodical parameter exploration
- **📈 Performance tracking:** Quantitative development progress

**[Next: Working with Results →](working-with-results.md)**

In the next chapter, you'll learn to extract maximum insight from your simulation data, automate analysis workflows, and prepare results for publication or deployment.

---

**Key Takeaways:**
- 🔄 **Three-step rhythm:** Compile → Run → Visualize
- ⚡ **Fast feedback:** Complete cycle in under 30 seconds
- 🎯 **Visual debugging:** Spike patterns reveal network behavior
- 📊 **Rich data collection:** Spikes, weights, potentials, and more
- 🔧 **Flexible execution:** From quick tests to long experiments
- 📈 **Real-time monitoring:** Watch your network learn live

**Try This:**
- Implement the sound classifier example
- Experiment with different visualization styles
- Try real-time visualization during simulation
- Create your own analysis templates
