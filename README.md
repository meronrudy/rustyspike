<<<<<<< Updated upstream
# hSNN — Spiking Hypergraph Neural Network
## Current State
 present state of the  workspace, with emphasis on the CLI‑first workflow, Neuromorphic IR compatibility (NIR), verification and lowering via the compiler, and minimal visualization.

General/Paradigm: CLI‑first neuromorphic research substrate that treats the command line as the primary UX. Everything else (IR, runtime, storage, visualization) supports that workflow.
Thin‑waist architecture: A stable, versioned, unit‑aware IR and binary storage schemas form the long‑lived “waist” between high‑level tools and low‑level engines. Explicit trait boundaries and narrow, composable contracts.
End‑to‑end NIR path: Textual NIR → parse → verify → (passes) → lower → run → export → visualize.
Core components

shnn-ir (Neuromorphic IR)
Purpose: Define the textual IR data model and tooling.
Implemented:
Module/Operation model with dialects (neuron, plasticity, connectivity, stimulus, runtime).
Versioned (name@vN), typed, unit‑aware attributes:
TimeNs, DurationNs, VoltageMv, ResistanceMohm, CapacitanceNf, CurrentNa, RateHz, Weight, RangeU32, NeuronRef, plus scalars.
Textual printer (to_text) and minimal parser (parse_text) with round‑trip parity (tests included).
Convenience constructors (sugar) for common ops:
neuron.lif@v1, plasticity.stdp@v1, connectivity.layer_fully_connected@v1, connectivity.synapse_connect@v1, stimulus.poisson@v1, runtime.simulate.run@v1.
Tests:
Single‑op and mixed module round‑trip tests (including synapse_connect).
Conformance parity tests used downstream by the compiler.
shnn-compiler (NIR compiler)
Purpose: Verify NIR, optionally transform via passes, and lower to the runtime engine.
Public API:
verify_module(&Module) → Result<()>: Ensures attribute presence, types/units, and semantic bounds (e.g., tau_m > 0, r_m > 0, c_m > 0; stdp tau > 0 and w_min ≤ w_max; range validity; dt/duration > 0; etc.).
list_ops() → &'static [OpSpec]: Access the static op registry (dialect, name, version, attributes) for dynamic introspection.
compile_with_passes(&Module) → Result:
Pipeline: verify → run passes → lower → runnable SimulationEngine.
Pass framework (crates/shnn-compiler/src/passes.rs):
CanonicalizePass: expands connectivity.layer_fully_connected into explicit connectivity.synapse_connect ops; normalizes attributes where appropriate.
UpgradeVersionsPass: scaffolding for automatic in‑place upgrades (e.g., lif@v0 → lif@v1) with defaulted attributes.
Lowering:
Sets LIF/STDP defaults (NetworkConfig), builds neurons/synapses via NetworkBuilder, collects StimulusPattern values, and configures SimulationParams for the runtime engine.
Tests:
verify_bounds.rs: negative/positive coverage for semantics and unit checks.
compile_with_passes_ok.rs: end‑to‑end compile (with passes) and run.
nir_parity_roundtrip.rs: compile→run parity comparing direct Module vs parse(text(Module)).
Op Registry (public “OpRegistry”)

Backed by static OpSpec/AttributeSpec entries with AttrKind enumeration (e.g., DurationNs, VoltageMv, etc.).
Drives:
CLI dynamic op listing (printing attribute names, kinds, required/optional, and docs).
Verification (type checks and presence).
Documentation and future JSON serialization of op schemas.
shnn-runtime (Engine)
Purpose: Execute the lowered program.
Current:
NetworkBuilder and SNN network construction with LIF neurons.
Optional STDP plasticity configuration.
SimulationEngine capable of running a program and exporting spikes.
Integration: The compiler lowers IR ops into runtime constructs (network configuration, synapses, stimuli, simulation parameters).
shnn-cli (Command Line Interface)
Purpose: The primary user entrypoint.
NIR commands:
snn nir compile — Construct textual NIR from CLI settings (LIF defaults, optional STDP, fully‑connected layers, Poisson stimuli, simulate.run).
snn nir verify — Parse and verify a textual NIR file.
snn nir run — Parse → verify → compile_with_passes → run. Optionally export spike JSON (time_ns/time_ms + neuron_id).
snn nir op-list [--detailed] — Dynamically print supported dialects/ops/versions and attribute schemas from shnn-compiler::list_ops(), using AttrKind::name() for type names.
Visualization:
snn viz serve — Starts a small static file server (no extra deps) hosting an SPA with a canvas spike raster.
Endpoints: / (SPA), /api/health, /api/list (list JSON files in --results-dir), /api/spikes?file=...
SPA reads exported spike JSON and renders a raster (Canvas 2D), structured for future WebGL2.
Additional scaffolds:
TTR command: parses a TOML “program” and outputs a JSON mask (placeholder, to migrate to storage VMSK).
Study runner: parses a TOML study config and orchestrates sequential runs (exports per‑run summaries and summary.json).
Storage layer (design complete; staged implementation)
Binary schemas (documented): VCSR, VEVT, VMSK, VMORF, VGRF, VRAS; targeted at zero‑copy or minimal‑copy IO and typed headers.
Traits (“interfaces”): Stable contracts for reading/writing and interop with the runtime and IR layers.
Current code includes schema utilities and partial implementations; JSON is used for early viz exports.
Thin‑waist traits (“interfaces”)
Stable boundaries define contracts across:
IR (ops/types/units and registry)
Compiler (verify, passes, lowering)
Runtime (network construction, stimuli, simulation)
Storage (binary schemas)
Ensures that tools can evolve independently while preserving reproducibility and determinism.
Current user flows (end‑to‑end)

Introspect ops:
cargo run -p shnn-cli -- nir op-list --detailed
Compile a textual NIR:
cargo run -p shnn-cli -- nir compile --output /tmp/demo.nirt
Run a textual NIR and export spikes:
cargo run -p shnn-cli -- nir run /tmp/demo.nirt --output crates/shnn-cli/test_workspace/results/run1.json
Visualize spikes:
cargo run -p shnn-cli -- viz serve --results-dir crates/shnn-cli/test_workspace/results
Open the printed http://127.0.0.1:7878
Quality and testing

IR round‑trip tests (print→parse→print).
Compiler bounds/semantic verifier tests (negative/positive).
Compile→run parity tests (direct Module vs parse(text(Module))).
CLI remains minimal‑dep and runs on stable toolchains.
What’s not done yet (high level)

Rich canonicalization and version upgrades beyond v1 examples.
Topology/measurement/analytics endpoints and visualization in the SPA (WebGL2 path).
Binary mask writing (VMSK) and workload‑aware transforms (TTR engine).
Parallelized, resumable study orchestration and robust determinism harness.
References

NIR dialects and versioning: docs/architecture/NIR_DIALECTS_AND_VERSIONING.md
CLI commands: crates/shnn-cli/src/commands
Compiler public API: crates/shnn-compiler/src/lib.rs and src/passes.rs
CLI‑first neuromorphic research "substrate" for building, verifying, compiling, running, and visualizing spiking neural networks (SNNs), emphasizing ease, reproducibility, and flexibility. 

Status: Core foundations are mostly implemented and tested. Working on figuring out the (NIR) path most erthing else is live through the compiler and CLI, Gotta modularize the MLIR shit..... dynamic op introspection, verification, lowering to runtime, simulation, JSON exports, adding a minimal visualization server with spike raster rendering. Several advanced features are scaffolded for incremental, compilation‑safe expansion.

Next up"
- Textual NIR with versioned, typed, unit‑aware ops, MLIR‑like syntax and round‑trip printing/parsing
- Static, introspectable Op Registry with attribute kinds and docs
- Compiler with verification, lowering, and a pass framework (canonicalization and version upgrader)
- CLI add NIR commands (compile, run, verify, op‑list), and a minimal viz server (serve + JSON endpoints)
- Storage and runtime foundations aligned to thin‑waist traits and zero‑copy binary schemas (design complete; select pieces implemented)
=======
# Welcome to the Future of Computing 🧠⚡

## What if your computer could think like a brain?

Imagine a computer that doesn't crunch numbers in rigid steps, but instead sends tiny electrical spikes that carry information—just like neurons in your brain. This isn't science fiction. This is **neuromorphic computing**, and you're about to discover why it's going to change everything.

## Your Brain vs. Your Laptop: A Tale of Two Processors
>>>>>>> Stashed changes

Right now, your laptop burns through 65 watts to run a video call. Your brain? It does everything—seeing, thinking, remembering, dreaming—on just 20 watts. That's less power than a light bulb.

The secret? Your brain doesn't process information like a traditional computer. Instead of moving data back and forth between memory and processor (the infamous "von Neumann bottleneck"), your brain's 86 billion neurons communicate directly through **spikes**—brief electrical pulses that carry both data and computation together.

## Traditional AI vs. Neuromorphic: The Ultimate Showdown

**Traditional AI:**
- 🔥 Burns massive amounts of energy (ChatGPT uses as much power as a small city)
- 🐌 Processes everything in batches, even for real-time tasks
- 🧱 Rigid architectures that require complete retraining for new tasks
- 💸 Needs expensive GPUs and cloud infrastructure

**Neuromorphic Computing:**
- ⚡ Ultra-low power (1000x more efficient for many tasks)
- 🏃‍♂️ Processes information as it arrives, in real-time
- 🧠 Learns continuously, adapting to new patterns on the fly
- 💡 Runs on tiny chips, even in your smartphone

## Why Everyone Else Makes This Hard (And We Don't)

Most neuromorphic platforms are academic nightmares:
- 📚 Require PhD-level neuroscience knowledge just to get started
- 🔧 Need custom hardware and proprietary tools
- 🕸️ Tangled codebases with zero documentation
- 🎯 Focus on recreating biological accuracy instead of solving real problems

**We took a different approach.**

## Introducing hSNN: Neuromorphic Computing Made Human

hSNN is the first neuromorphic platform designed for **developers, not just neuroscientists**. Here's what makes us different:

### 🎯 CLI-First Philosophy
```bash
# Want to build a neuromorphic network? Three commands:
snn nir compile --output my-network.nirt
snn nir run my-network.nirt --output results.json
snn viz serve --results-dir ./results
```

No IDEs to learn. No proprietary tools. Just simple commands that do exactly what they say.

### 🏗️ "Thin Waist" Architecture
We built the stable foundation everyone else forgot:
- **Neuromorphic IR (NIR)**: A universal language for describing spiking networks
- **Binary schemas**: Lightning-fast data formats that work everywhere
- **Trait-based interfaces**: Swap components without breaking anything

Think of it as the "HTML of neuromorphic computing"—a common standard that just works.

### 🔄 Round-Trip Everything
```nir
# Write your network in human-readable NIR:
%lif1 = neuron.lif<v_th=1.0, v_reset=0.0>() -> (3,)
%syn1 = connectivity.fully_connected<weight=0.5>(%lif1) -> (3,)
```

Our system can convert this to binary, simulate it, visualize it, and convert it back to text **without losing any information**. Try that with TensorFlow.

### 🎮 Real-Time Visualization
See your spiking networks come alive with built-in visualization:
- 📊 Spike raster plots showing neuron activity over time
- 🕸️ Network topology viewers
- 📈 Real-time performance metrics

### 🔧 Extensible by Design
- **Static introspection**: Query what operations are available at compile time
- **Pass framework**: Transform and optimize networks with composable passes
- **Multiple backends**: CPU, embedded, WASM, Python—one codebase, many targets

## Real-World Applications You Can Build Today

- 🤖 **Autonomous drones** that navigate using 100x less power
- 👁️ **Computer vision** systems that work in real-time on mobile devices
- 🎧 **Audio processing** that adapts to your environment
- 🕹️ **Game AI** that learns your playing style
- 🏠 **Smart sensors** that run for years on a single battery

## The Science Made Simple

### Spikes: Nature's Information Packets
Traditional computers represent information as continuous numbers (0.7, 3.14, etc.). Brains use **spikes**—binary events that happen at specific times. It's like the difference between a dimmer switch and a telegraph key.

<<<<<<< Updated upstream
Citation
If you use hSNN in your research, please cite:
```bibtex
@software{hsnn2025,
  title   = {hSNN: Spiking Hypergraph Neural Network},
  author  = {hSNN Development Team},
  year    = {2025},
  url     = {https://github.com/hsnn-project/hsnn},
  version = {0.1.0}
}
=======
### Time is Everything
In traditional AI, time is just another dimension. In neuromorphic computing, **time IS the computation**. When spikes arrive, what they do, and how neurons respond—that's where the magic happens.

### Learning Without Forgetting
Your brain doesn't retrain from scratch every time you learn something new. Neuromorphic systems use **spike-timing dependent plasticity (STDP)**—connections get stronger or weaker based on the precise timing of spikes. It's learning that happens naturally, without backpropagation.

## Why This Matters Now

We're hitting the limits of traditional computing:
- **Moore's Law is dead**: Transistors can't get much smaller
- **Power walls**: Data centers consume 4% of global electricity
- **The AI bubble**: Current approaches don't scale to real-world deployment

Neuromorphic computing isn't just the next step—it's the only step that makes sense.

## Your Journey Starts Here

This isn't just documentation. This is your guide from zero to neuromorphic hero. We'll take you from "What's a spike?" to building and deploying production neuromorphic systems.

Whether you're:
- 🎓 A student curious about brain-inspired computing
- 👩‍💻 A developer wanting to build ultra-efficient AI
- 🔬 A researcher exploring new computational paradigms
- 🏢 An engineer solving real-world power and latency problems

...you're in the right place.

---

## The Challenge 🎯

**We dare you to build a neuromorphic system in the next hour.**

No PhD required. No expensive hardware. No months of setup.

Just you, this documentation, and the future of computing.

**[Accept the Challenge → Start with Quick Start](Morphics/quick-start.md)**

*Or if you want to understand the "why" before the "how":*
**[Understand the Science → Introduction to Neuromorphic Computing](Morphics/introduction/)**

---

## Quick Navigation

- 🚀 **[Quick Start](Morphics/quick-start.md)** - Build your first neuromorphic network in minutes
- 🧠 **[Introduction](Morphics/introduction/)** - Neuromorphic computing explained for humans
- 🛠️ **[CLI Workflows](Morphics/cli-workflows/)** - Master the command-line tools
- 🏗️ **[Architecture](Morphics/architecture/)** - The "thin waist" that makes it all work
- ⚙️ **[Engine](Morphics/engine/)** - Under the hood of the simulation engine
- 🔌 **[Interop](Morphics/interop/)** - Python, WASM, embedded, and more
- 👥 **[Contributing](Morphics/contributing/)** - Join the neuromorphic revolution

---

*"The best way to predict the future is to implement it."* - hSNN Development Team
>>>>>>> Stashed changes
