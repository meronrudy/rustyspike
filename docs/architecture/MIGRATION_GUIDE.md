# Migration Guide: From Library to CLI-First Framework

## Overview

This guide provides a comprehensive migration path from the current hSNN library-based approach to the new CLI-first research substrate. The migration is designed to be gradual, maintaining backward compatibility while enabling users to adopt new capabilities incrementally.

## Migration Philosophy

### Zero-Breaking Changes Initially
- All existing APIs remain functional
- New CLI capabilities supplement existing workflows
- Users can migrate at their own pace
- Clear deprecation timeline with advance notice

### Incremental Value Addition
- Each migration step provides immediate benefits
- Users can adopt CLI features selectively
- Mixed workflows (API + CLI) are supported
- Clear ROI for each migration step

### Automated Migration Tools
- Code transformation utilities
- Configuration migration scripts
- Data format converters
- Validation and testing tools

## Migration Timeline

| Phase | Duration | Changes | Impact |
|-------|----------|---------|---------|
| **Phase 0: Coexistence** | Months 1-6 | CLI added alongside existing API | No breaking changes |
| **Phase 1: Soft Deprecation** | Months 7-12 | Existing API marked as "legacy" | Warnings, but no breaks |
| **Phase 2: Hard Deprecation** | Months 13-18 | Legacy API requires explicit opt-in | Minor friction added |
| **Phase 3: Legacy Removal** | Months 19-24 | Legacy API moved to separate crate | Clean architecture |

## Current vs. New Architecture

### Before: Library-First Approach

```rust
// Current approach: Everything in code
use shnn_core::prelude::*;

fn main() -> Result<()> {
    // Manual network construction
    let mut network = NetworkBuilder::new()
        .with_connectivity(HypergraphNetwork::new())
        .with_neurons(NeuronConfig::lif_default(1000))
        .enable_stdp()
        .build_lif()?;
    
    // Manual spike generation
    let mut encoder = PoissonEncoder::new(50.0, Some(42));
    let spikes = encoder.encode(1.0, TimeStep::from_secs(1.0), 0)?;
    
    // Manual simulation loop
    let mut output_spikes = Vec::new();
    for step in 0..1000 {
        let step_output = network.process_step(&spikes, TimeStep::from_millis(1.0))?;
        output_spikes.extend(step_output);
    }
    
    // Manual analysis
    println!("Generated {} spikes", output_spikes.len());
    
    Ok(())
}
```

### After: CLI-First Approach

```bash
# New approach: CLI commands handle complexity
snn init my_experiment --neurons 1000 --topology hypergraph
cd my_experiment

# Configuration in version-controlled files
snn config --set neuron_model=lif plasticity=stdp learning_rate=0.01

# Training with automatic progress tracking and reproducibility
snn train --epochs 100 --input poisson:rate=50 --output model.bin --seed 42

# Automatic analysis and visualization
snn eval --model model.bin --metrics firing_rate,connectivity --format json
snn viz serve --model model.bin --realtime &

# Experiment automation
snn study init --space params.toml --algo bayesian
snn study run --budget trials:50 --parallel 4
```

## Migration Strategies

### Strategy 1: CLI Wrapper (Immediate)

Keep existing code but add CLI wrapper for common operations.

**Before:**
```rust
// Existing research script
use shnn_core::prelude::*;

fn run_experiment(learning_rate: f32, connectivity: f32) -> Result<f32> {
    let network = NetworkBuilder::new()
        .with_learning_rate(learning_rate)
        .with_connectivity(connectivity)
        .build()?;
    
    // ... simulation logic ...
    
    Ok(accuracy)
}

fn main() {
    for lr in [0.01, 0.02, 0.03] {
        for conn in [0.1, 0.2, 0.3] {
            let acc = run_experiment(lr, conn)?;
            println!("LR: {}, Conn: {}, Acc: {}", lr, conn, acc);
        }
    }
}
```

**After (Phase 0):**
```bash
# Convert parameter sweep to CLI study
cat > params.toml << EOF
[parameters]
learning_rate = { type = "float", min = 0.01, max = 0.03, step = 0.01 }
connectivity = { type = "float", min = 0.1, max = 0.3, step = 0.1 }
EOF

snn study init --space params.toml --algo grid
snn study run --budget trials:9 --parallel 3
snn study report --format table
```

### Strategy 2: Hybrid Approach (Gradual)

Use CLI for experiment management, keep custom code for specialized logic.

**Before:**
```rust
// Custom neuron model
struct CustomNeuron {
    // ... custom implementation
}

impl Neuron for CustomNeuron {
    fn update(&mut self, input: f32, dt: TimeStep) -> Option<Spike> {
        // ... custom dynamics
    }
}

fn main() {
    let mut network = NetworkBuilder::new()
        .with_custom_neurons(CustomNeuron::new)
        .build()?;
    
    // Run experiment with custom model
}
```

**After (Phase 1):**
```rust
// Register custom model with CLI system
use shnn_cli::registry::ModelRegistry;

#[derive(CLICompatible)]  // Proc macro for CLI integration
struct CustomNeuron {
    // ... same implementation
}

fn main() {
    // Register model
    ModelRegistry::register("custom", CustomNeuron::factory);
    
    // Model now available via CLI
    // No other code changes needed
}
```

```bash
# Use custom model via CLI
snn train --neuron-model custom --config custom_params.toml
```

### Strategy 3: Full Migration (Target)

Complete transition to CLI-first workflow with configuration files.

**Before (everything in code):**
```rust
// 200+ lines of experiment setup and execution
```

**After (declarative configuration):**
```toml
# experiment.toml
[network]
neurons = 1000
topology = "hypergraph"
connectivity = 0.15

[training]
epochs = 100
learning_rate = 0.01
plasticity = "stdp"

[evaluation]
metrics = ["accuracy", "spike_rate", "energy"]
test_split = 0.2

[visualization]
realtime = true
export_format = "html"
```

```bash
# Single command runs entire experiment
snn run --config experiment.toml --output results/
```

## Code Migration Examples

### Example 1: Simple Network Training

**Before:**
```rust
use shnn_core::prelude::*;

fn train_network() -> Result<()> {
    let mut network = NetworkBuilder::new()
        .with_neurons(1000)
        .with_connectivity(0.1)
        .with_plasticity(STDPRule::default())
        .build()?;
    
    let input_spikes = generate_poisson_spikes(100, 50.0, 1.0, 42)?;
    
    for epoch in 0..100 {
        let output = network.process_spikes(&input_spikes)?;
        network.update_weights()?;
        
        if epoch % 10 == 0 {
            println!("Epoch {}: {} spikes", epoch, output.len());
        }
    }
    
    network.save("model.bin")?;
    Ok(())
}
```

**After:**
```bash
# Create configuration
cat > train.toml << EOF
[network]
neurons = 1000
connectivity = 0.1

[plasticity]
rule = "stdp"
learning_rate = 0.01

[training]
epochs = 100
input = "poisson:rate=50,duration=1.0,seed=42"
progress_every = 10

[output]
model = "model.bin"
EOF

# Run training
snn train --config train.toml
```

### Example 2: Parameter Sweep

**Before:**
```rust
use shnn_core::prelude::*;

fn parameter_sweep() -> Result<()> {
    let mut results = Vec::new();
    
    for &lr in &[0.01, 0.02, 0.03, 0.04, 0.05] {
        for &conn in &[0.05, 0.1, 0.15, 0.2] {
            let config = NetworkConfig::new()
                .with_learning_rate(lr)
                .with_connectivity(conn);
            
            let network = NetworkBuilder::from_config(config).build()?;
            let accuracy = evaluate_network(&network)?;
            
            results.push((lr, conn, accuracy));
            println!("LR: {:.3}, Conn: {:.3}, Acc: {:.3}", lr, conn, accuracy);
        }
    }
    
    // Find best parameters
    let best = results.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    println!("Best: LR={:.3}, Conn={:.3}, Acc={:.3}", best.0, best.1, best.2);
    
    Ok(())
}
```

**After:**
```bash
# Define search space
cat > sweep.toml << EOF
[parameters]
learning_rate = { type = "float", values = [0.01, 0.02, 0.03, 0.04, 0.05] }
connectivity = { type = "float", values = [0.05, 0.1, 0.15, 0.2] }

[optimization]
metric = "accuracy"
direction = "maximize"
EOF

# Run sweep
snn study init --space sweep.toml --algo grid
snn study run --budget trials:20
snn study best --count 5
```

### Example 3: Custom Analysis

**Before:**
```rust
use shnn_core::prelude::*;

fn analyze_network_dynamics() -> Result<()> {
    let network = Network::load("model.bin")?;
    let spikes = load_spike_data("test_spikes.dat")?;
    
    // Custom analysis code
    let mut firing_rates = HashMap::new();
    let mut isi_distributions = HashMap::new();
    
    for spike in &spikes {
        // Complex analysis logic...
    }
    
    // Generate plots
    plot_firing_rates(&firing_rates)?;
    plot_isi_distributions(&isi_distributions)?;
    
    Ok(())
}
```

**After (Hybrid):**
```rust
// analysis.rs - Custom analysis plugin
use shnn_cli::plugins::AnalysisPlugin;

#[plugin]
struct CustomAnalysis;

impl AnalysisPlugin for CustomAnalysis {
    fn analyze(&self, data: &AnalysisData) -> Result<AnalysisResult> {
        // Same analysis logic, but integrated with CLI
        let firing_rates = compute_firing_rates(&data.spikes)?;
        let isi_distributions = compute_isi_distributions(&data.spikes)?;
        
        Ok(AnalysisResult {
            plots: vec![
                Plot::firing_rates(firing_rates),
                Plot::isi_distributions(isi_distributions),
            ],
            metrics: HashMap::new(),
        })
    }
}
```

```bash
# Register and use plugin
snn plugin register analysis.so
snn analyze --model model.bin --data test_spikes.dat --plugin custom_analysis
```

## Migration Tools

### Automatic Code Conversion

```bash
# Convert existing Rust code to CLI configuration
snn migrate convert --input src/main.rs --output experiment.toml

# Generate CLI commands from existing code
snn migrate commands --input src/ --output run.sh

# Validate migration completeness
snn migrate validate --old-code src/ --new-config configs/
```

### Data Format Migration

```bash
# Convert existing data to new formats
snn migrate data --input old_spikes.json --output spikes.vevt --format vevt

# Batch convert directory
snn migrate batch --input data/ --output migrated/ --format vcsr
```

### Configuration Migration

```bash
# Extract configuration from code
snn migrate extract-config --input src/main.rs --output config.toml

# Merge multiple configurations
snn migrate merge-configs --inputs config1.toml config2.toml --output merged.toml
```

## Backward Compatibility

### API Preservation

During the transition period, all existing APIs remain available:

```rust
// This continues to work unchanged
use shnn_core::prelude::*;

let network = NetworkBuilder::new()
    .with_neurons(1000)
    .build()?;
```

### Gradual Deprecation

```rust
// Phase 1: Soft deprecation with warnings
#[deprecated(
    since = "0.2.0",
    note = "Consider using `snn train` CLI command for better reproducibility"
)]
pub fn train_network() -> Result<()> {
    // ... existing implementation
}

// Phase 2: Hard deprecation requiring opt-in
#[cfg(feature = "legacy-api")]
pub fn train_network() -> Result<()> {
    // ... existing implementation
}

// Phase 3: Move to separate crate
// Available in `shnn-legacy` crate
```

### Feature Flags

Users can control which APIs are available:

```toml
# Cargo.toml
[dependencies]
shnn = { version = "0.3", features = ["cli", "legacy-api"] }

# Only new CLI-based APIs
shnn = { version = "0.3", features = ["cli"] }

# Only legacy APIs (for gradual migration)
shnn = { version = "0.3", features = ["legacy-api"] }
```

## Testing Migration

### Validation Framework

```rust
// Test that CLI and legacy APIs produce identical results
#[test]
fn test_cli_legacy_equivalence() {
    // Legacy approach
    let legacy_result = run_legacy_experiment(params)?;
    
    // CLI approach
    let cli_result = run_cli_experiment("config.toml")?;
    
    // Validate equivalence
    assert_eq!(legacy_result.spikes, cli_result.spikes);
    assert_eq!(legacy_result.weights, cli_result.weights);
}
```

### Regression Testing

```bash
# Ensure migration doesn't break existing functionality
snn test migration --legacy-code tests/legacy/ --new-configs tests/configs/

# Benchmark performance comparison
snn benchmark migration --runs 10 --compare legacy,cli
```

## Best Practices for Migration

### Start Small
1. Begin with simple experiments
2. Use CLI for new projects first
3. Gradually convert existing workflows
4. Keep complex custom code initially

### Leverage Strengths
1. Use CLI for experiment management
2. Keep custom algorithms in code initially
3. Use CLI visualization for all projects
4. Adopt CLI automation for parameter sweeps

### Incremental Adoption
1. **Week 1**: Install CLI, try basic commands
2. **Week 2**: Convert one simple experiment
3. **Week 3**: Use CLI visualization for existing projects
4. **Week 4**: Try parameter sweeps with study runner
5. **Month 2**: Convert main research workflows
6. **Month 3**: Full CLI adoption for new work

### Common Migration Patterns

**Pattern 1: Experiment Scripts → Configuration Files**
- Extract parameters to TOML files
- Use CLI commands instead of custom scripts
- Leverage built-in progress tracking and logging

**Pattern 2: Manual Analysis → Automated Pipelines**
- Use CLI visualization instead of custom plotting
- Leverage built-in metrics and statistics
- Add custom analysis as CLI plugins

**Pattern 3: Ad-hoc Experiments → Reproducible Studies**
- Convert parameter loops to study configurations
- Use RUNINFO bundles for reproducibility
- Leverage experiment tracking and comparison

## Troubleshooting Migration

### Common Issues

**Issue 1: Custom Neuron Models**
```rust
// Problem: Custom model not CLI-compatible
struct MyNeuron { /* custom fields */ }

// Solution: Add CLI compatibility
#[derive(CLICompatible)]
struct MyNeuron { 
    #[cli(parameter = "threshold")]
    threshold: f32,
    /* other fields */
}
```

**Issue 2: Complex Workflows**
```bash
# Problem: Workflow too complex for single CLI command
# Solution: Break into multiple commands with intermediate files

snn preprocess --input raw_data/ --output processed/
snn train --input processed/ --config train.toml --output model.bin
snn evaluate --model model.bin --test processed/test/ --output results.json
snn visualize --results results.json --output plots/
```

**Issue 3: Performance Differences**
```bash
# Problem: CLI version slower than optimized code
# Solution: Use performance profiling and optimization flags

snn train --config train.toml --optimize --profile performance.json
snn optimize analyze --profile performance.json --suggest
```

### Getting Help

```bash
# Built-in migration assistance
snn migrate help
snn migrate check --input src/
snn migrate suggest --current-workflow workflow.rs

# Community resources
snn community forum
snn community examples
```

## Timeline and Support

### Transition Timeline
- **Months 1-6**: Full backward compatibility, CLI available as addition
- **Months 7-12**: Soft deprecation warnings, CLI becomes recommended approach
- **Months 13-18**: Legacy API requires explicit opt-in, migration tools mature
- **Months 19-24**: Legacy API moved to separate crate, clean architecture achieved

### Support Commitment
- **Bug fixes**: Legacy APIs receive bug fixes for 24 months
- **Security updates**: Legacy APIs receive security updates for 36 months
- **Migration assistance**: Free migration consulting available for first 12 months
- **Documentation**: Legacy documentation maintained for 24 months

### Success Metrics
- **Adoption rate**: >50% of users try CLI within 6 months
- **Migration rate**: >75% of active users migrate within 18 months
- **Satisfaction**: >90% of migrated users prefer CLI approach
- **Support load**: <10% increase in support requests during migration

This migration guide ensures a smooth transition from the current library-based approach to the CLI-first research substrate while maintaining user productivity and satisfaction throughout the process.