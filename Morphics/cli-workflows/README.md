# Part II: CLI-First Workflows 🛠️

Welcome to the practical side of neuromorphic computing! In this section, you'll master the command-line tools that make hSNN the most developer-friendly neuromorphic platform available.

## Why CLI-First?

Most neuromorphic platforms force you to learn complex GUIs, proprietary IDEs, or arcane configuration systems. hSNN takes a different approach: **everything you need is available through simple, composable command-line tools**.

### The CLI-First Philosophy

```bash
# Traditional neuromorphic workflow
1. Launch proprietary IDE
2. Click through complex menus
3. Configure in GUI forms
4. Export to proprietary format
5. Import into simulation environment
6. Debug through GUI debugger
7. Visualize in separate tool

# hSNN workflow
snn nir compile --output network.nirt    # Build your network
snn nir run network.nirt --output results.json  # Simulate
snn viz serve --results-dir ./            # Visualize
```

**Benefits:**
- 🤖 **Scriptable:** Automate your entire workflow
- 🔄 **Reproducible:** Every command is documented and repeatable
- 🧰 **Composable:** Chain commands together for complex workflows
- 📝 **Version controllable:** All configurations are text files
- 🏃‍♂️ **Fast:** No GUI overhead, direct to results

## Your Journey Through This Section

### 🏗️ **Chapter 1: [Command Line Mastery](command-line-mastery.md)**
Master the core `snn` command and its subcommands. Learn the patterns that make neuromorphic development fast and intuitive.

**Key Skills:**
- Navigate the command hierarchy
- Use help systems effectively
- Chain commands for complex workflows
- Debug when things go wrong

### 📁 **Chapter 2: [Project Setup and Structure](project-setup.md)**
Learn to organize neuromorphic projects for maintainability and collaboration. Understand workspace configuration and project templates.

**Key Skills:**
- Initialize new neuromorphic projects
- Configure workspace settings
- Organize experiments and results
- Share projects with teams

### 🧠 **Chapter 3: [The NIR Workflow](nir-workflow.md)**
Master the Neuromorphic IR (NIR) - the universal language for describing spiking networks. Build, verify, and optimize networks using textual descriptions.

**Key Skills:**
- Write NIR descriptions by hand
- Compile parameters to NIR
- Verify network correctness
- Optimize NIR for performance

### ⚡ **Chapter 4: [Compile, Run, Visualize](compile-run-viz.md)**
The core development cycle: from idea to working neuromorphic system in minutes. See your networks come alive with real-time visualization.

**Key Skills:**
- Compile networks from high-level descriptions
- Run simulations with different parameters
- Visualize spike activity and network dynamics
- Iterate rapidly on designs

### 📊 **Chapter 5: [Working with Results](working-with-results.md)**
Extract insights from your neuromorphic simulations. Analyze performance, debug behavior, and optimize for deployment.

**Key Skills:**
- Parse simulation results
- Generate analysis reports
- Export data for external tools
- Track performance metrics

## The Command Structure

hSNN's CLI follows a consistent, hierarchical structure:

```
snn [global-options] <command> [command-options] [subcommand] [args]

Commands:
├── nir          # Neuromorphic IR operations  
│   ├── compile  # Build NIR from parameters
│   ├── run      # Execute NIR simulation
│   ├── verify   # Check NIR correctness
│   └── op-list  # List available operations
├── viz          # Visualization and analysis
│   ├── serve    # Start visualization server
│   ├── export   # Export network diagrams
│   └── plot     # Generate static plots
├── workspace    # Project management
│   ├── init     # Initialize new project
│   ├── validate # Check project structure
│   └── clean    # Clean build artifacts
└── study        # Experiment orchestration
    ├── run      # Execute experiment suite
    ├── compare  # Compare experiment results
    └── report   # Generate analysis reports
```

**Design principles:**
- **Predictable:** Similar patterns across all commands
- **Self-documenting:** `--help` available at every level
- **Consistent:** Same options mean the same thing everywhere
- **Safe:** Destructive operations require confirmation

## Example: Complete Workflow in 5 Commands

Here's a taste of what you'll be able to do by the end of this section:

```bash
# 1. Create a new neuromorphic project
snn workspace init pattern-classifier --template classification

# 2. Build a network for pattern recognition
snn nir compile \
  --neurons lif \
  --plasticity stdp \
  --topology "10->50->3" \
  --output networks/classifier.nirt

# 3. Run simulation with training data
snn nir run networks/classifier.nirt \
  --stimulus patterns/training.json \
  --duration 1000ms \
  --output results/training-run.json

# 4. Visualize the results
snn viz serve \
  --results-dir results/ \
  --port 8080 &

# 5. Generate analysis report
snn study report results/training-run.json \
  --template classification-analysis \
  --output reports/training-analysis.html
```

**What this achieves:**
- Complete pattern classification system
- Trained neuromorphic network
- Real-time visualization
- Comprehensive analysis report
- All in under 2 minutes!

## Configuration Philosophy

hSNN uses a layered configuration system:

### 1. **Global Defaults** (Built-in)
```toml
# Built into hSNN
[defaults]
neuron_type = "lif"
dt_us = 100  # 100 microsecond timesteps
seed = 42
```

### 2. **Workspace Config** ([`hSNN.toml`](file://crates/shnn-cli/test_workspace/hSNN.toml))
```toml
# Project-specific defaults
[workspace]
name = "my_project"
version = "0.1.0"

[lif]
tau_m = 20.0      # membrane time constant
v_thresh = -50.0  # spike threshold
```

### 3. **Experiment Config** ([`experiments/`](file://crates/shnn-cli/test_workspace/experiments/))
```toml
# Specific experiment parameters
[experiment]
name = "pattern_learning"

[network]
inputs = 10
hidden = 50
outputs = 3
```

### 4. **Command Line** (Highest priority)
```bash
# Override anything at runtime
snn nir compile --dt-us 50 --seed 123
```

**Benefits:**
- **Hierarchical:** Specific settings override general ones
- **Reproducible:** All settings stored in version control
- **Flexible:** Override anything without changing files
- **Discoverable:** Self-documenting through examples

## Development Patterns

### Pattern 1: **Rapid Prototyping**
```bash
# Quick iteration cycle
snn nir compile --neurons lif --topology 5x5 -o test.nirt
snn nir run test.nirt --duration 100ms -o quick-test.json
snn viz serve --results-dir . --port 8080
# → See results in browser immediately
```

### Pattern 2: **Reproducible Research**
```bash
# Everything scripted and versioned
git clone neuromorphic-study
cd neuromorphic-study
./scripts/run-all-experiments.sh  # Runs dozens of experiments
./scripts/generate-paper-figures.sh  # Creates all plots
```

### Pattern 3: **Production Deployment**
```bash
# Optimize for deployment
snn nir compile config/production.toml -o production.nirt
snn nir verify production.nirt --strict
snn nir run production.nirt --profile performance
# → Ready for embedded deployment
```

### Pattern 4: **Collaborative Development**
```bash
# Share with teammates
snn workspace validate  # Check everything is correct
git add . && git commit -m "Add gesture recognition network"
# → Teammates can reproduce everything exactly
```

## Performance Tips

### CLI Performance
- **Use `--quiet`** for faster execution in scripts
- **Specify `--output`** to avoid default file discovery
- **Use `--parallel`** for multi-core simulations
- **Cache with `--cache-dir`** for repeated operations

### Workflow Performance
- **Compile once, run many:** NIR files are portable
- **Use incremental builds:** Only rebuild what changed
- **Profile before optimizing:** Measure actual bottlenecks
- **Visualize smartly:** Stream large datasets instead of loading all

## Debugging and Troubleshooting

### Common Issues and Solutions

**"Command not found"**
```bash
# Check installation
which snn
snn --version

# Rebuild if needed
cargo build --release
export PATH=$PATH:./target/release
```

**"NIR compilation failed"**
```bash
# Get detailed error info
snn nir verify my-network.nirt --verbose
snn nir compile --debug --output debug.nirt
```

**"Simulation runs but no spikes"**
```bash
# Check network activity
snn nir run network.nirt --debug-spikes
snn viz serve --debug-mode
```

**"Results look wrong"**
```bash
# Validate your workflow
snn workspace validate
snn nir verify network.nirt --strict
snn study compare results/ --baseline expected/
```

### Getting Help

hSNN's CLI is self-documenting:

```bash
# Top-level help
snn --help

# Command-specific help
snn nir --help
snn nir compile --help

# Example usage
snn nir compile --help --examples

# Man pages (if installed)
man snn
man snn-nir-compile
```

## What You'll Build

By the end of this section, you'll have created:

1. **A complete neuromorphic workspace** with proper organization
2. **Multiple working networks** for different tasks
3. **Automated experiment scripts** for reproducible research
4. **Real-time visualization dashboards** for network analysis
5. **Production-ready configurations** for deployment

More importantly, you'll understand the **philosophy** behind CLI-first development and be able to:
- Design neuromorphic systems efficiently
- Debug problems systematically
- Collaborate effectively with teams
- Deploy systems confidently

## Prerequisites

### Technical Requirements
- **hSNN installed** and in your PATH
- **Command line comfort:** Basic shell navigation and scripting
- **Text editor:** Any editor that handles TOML and NIR files
- **Web browser:** For visualization interfaces

### Optional but Helpful
- **Git:** For version control and collaboration
- **jq:** For JSON processing and analysis
- **curl:** For API interactions and testing

## Ready to Begin?

The command line is your gateway to neuromorphic computing mastery. Let's start with the fundamentals and work up to advanced workflows.

**[Start with Chapter 1: Command Line Mastery →](command-line-mastery.md)**

Or jump to specific topics:
- **[Project Setup →](project-setup.md)** if you want to start a new project
- **[NIR Workflow →](nir-workflow.md)** if you want to understand the IR language
- **[Compile, Run, Visualize →](compile-run-viz.md)** if you want to see results quickly

---

*"The command line is not a barrier to entry—it's a superpower for neuromorphic development."* — hSNN Philosophy
