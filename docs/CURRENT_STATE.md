# hSNN Current State

This document captures the present state of the hSNN workspace, with emphasis on the CLI‑first workflow, Neuromorphic IR (NIR), verification and lowering via the compiler, and minimal visualization.

Overview
- Paradigm: CLI‑first neuromorphic research substrate that treats the command line as the primary UX. Everything else (IR, runtime, storage, visualization) supports that workflow.
- Thin‑waist architecture: A stable, versioned, unit‑aware IR and binary storage schemas form the long‑lived “waist” between high‑level tools and low‑level engines. Explicit trait boundaries and narrow, composable contracts.
- End‑to‑end NIR path: Textual NIR → parse → verify → (passes) → lower → run → export → visualize.

Core components

1) shnn-ir (Neuromorphic IR)
- Purpose: Define the textual IR data model and tooling.
- Implemented:
  - Module/Operation model with dialects (neuron, plasticity, connectivity, stimulus, runtime).
  - Versioned (name@vN), typed, unit‑aware attributes:
    - TimeNs, DurationNs, VoltageMv, ResistanceMohm, CapacitanceNf, CurrentNa, RateHz, Weight, RangeU32, NeuronRef, plus scalars.
  - Textual printer (to_text) and minimal parser (parse_text) with round‑trip parity (tests included).
  - Convenience constructors (sugar) for common ops:
    - neuron.lif@v1, plasticity.stdp@v1, connectivity.layer_fully_connected@v1, connectivity.synapse_connect@v1, stimulus.poisson@v1, runtime.simulate.run@v1.
- Tests:
  - Single‑op and mixed module round‑trip tests (including synapse_connect).
  - Conformance parity tests used downstream by the compiler.

2) shnn-compiler (NIR compiler)
- Purpose: Verify NIR, optionally transform via passes, and lower to the runtime engine.
- Public API:
  - verify_module(&Module) → Result<()>: Ensures attribute presence, types/units, and semantic bounds (e.g., tau_m > 0, r_m > 0, c_m > 0; stdp tau > 0 and w_min ≤ w_max; range validity; dt/duration > 0; etc.).
  - list_ops() → &'static [OpSpec]: Access the static op registry (dialect, name, version, attributes) for dynamic introspection.
  - compile_with_passes(&Module) → Result<LoweredProgram>:
    - Pipeline: verify → run passes → lower → runnable SimulationEngine.
  - Pass framework (crates/shnn-compiler/src/passes.rs):
    - CanonicalizePass: expands connectivity.layer_fully_connected into explicit connectivity.synapse_connect ops; normalizes attributes where appropriate.
    - UpgradeVersionsPass: scaffolding for automatic in‑place upgrades (e.g., lif@v0 → lif@v1) with defaulted attributes.
  - Lowering:
    - Sets LIF/STDP defaults (NetworkConfig), builds neurons/synapses via NetworkBuilder, collects StimulusPattern values, and configures SimulationParams for the runtime engine.
- Tests:
  - verify_bounds.rs: negative/positive coverage for semantics and unit checks.
  - compile_with_passes_ok.rs: end‑to‑end compile (with passes) and run.
  - nir_parity_roundtrip.rs: compile→run parity comparing direct Module vs parse(text(Module)).

Op Registry (public “OpRegistry”)
- Backed by static OpSpec/AttributeSpec entries with AttrKind enumeration (e.g., DurationNs, VoltageMv, etc.).
- Drives:
  - CLI dynamic op listing (printing attribute names, kinds, required/optional, and docs).
  - Verification (type checks and presence).
  - Documentation and future JSON serialization of op schemas.

3) shnn-runtime (Engine)
- Purpose: Execute the lowered program.
- Current:
  - NetworkBuilder and SNN network construction with LIF neurons.
  - Optional STDP plasticity configuration.
  - SimulationEngine capable of running a program and exporting spikes.
- Integration: The compiler lowers IR ops into runtime constructs (network configuration, synapses, stimuli, simulation parameters).

4) shnn-cli (Command Line Interface)
- Purpose: The primary user entrypoint.
- NIR commands:
  - snn nir compile — Construct textual NIR from CLI settings (LIF defaults, optional STDP, fully‑connected layers, Poisson stimuli, simulate.run).
  - snn nir verify — Parse and verify a textual NIR file.
  - snn nir run — Parse → verify → compile_with_passes → run. Optionally export spike JSON (time_ns/time_ms + neuron_id).
  - snn nir op-list [--detailed] — Dynamically print supported dialects/ops/versions and attribute schemas from shnn-compiler::list_ops(), using AttrKind::name() for type names.
- Visualization:
  - snn viz serve — Starts a small static file server (no extra deps) hosting an SPA with a canvas spike raster.
    - Endpoints: / (SPA), /api/health, /api/list (list JSON files in --results-dir), /api/spikes?file=...
    - SPA reads exported spike JSON and renders a raster (Canvas 2D), structured for future WebGL2.
- Additional scaffolds:
  - TTR command: parses a TOML “program” and outputs a JSON mask (placeholder, to migrate to storage VMSK).
  - Study runner: parses a TOML study config and orchestrates sequential runs (exports per‑run summaries and summary.json).

5) Storage layer (design complete; staged implementation)
- Binary schemas (documented): VCSR, VEVT, VMSK, VMORF, VGRF, VRAS; targeted at zero‑copy or minimal‑copy IO and typed headers.
- Traits (“interfaces”): Stable contracts for reading/writing and interop with the runtime and IR layers.
- Current code includes schema utilities and partial implementations; JSON is used for early viz exports.

6) Thin‑waist traits (“interfaces”)
- Stable boundaries define contracts across:
  - IR (ops/types/units and registry)
  - Compiler (verify, passes, lowering)
  - Runtime (network construction, stimuli, simulation)
  - Storage (binary schemas)
- Ensures that tools can evolve independently while preserving reproducibility and determinism.

Current user flows (end‑to‑end)
- Introspect ops:
  - cargo run -p shnn-cli -- nir op-list --detailed
- Compile a textual NIR:
  - cargo run -p shnn-cli -- nir compile --output /tmp/demo.nirt
- Run a textual NIR and export spikes:
  - cargo run -p shnn-cli -- nir run /tmp/demo.nirt --output crates/shnn-cli/test_workspace/results/run1.json
- Visualize spikes:
  - cargo run -p shnn-cli -- viz serve --results-dir crates/shnn-cli/test_workspace/results
  - Open the printed http://127.0.0.1:7878

Quality and testing
- IR round‑trip tests (print→parse→print).
- Compiler bounds/semantic verifier tests (negative/positive).
- Compile→run parity tests (direct Module vs parse(text(Module))).
- CLI remains minimal‑dep and runs on stable toolchains.

What’s not done yet (high level)
- Rich canonicalization and version upgrades beyond v1 examples.
- Topology/measurement/analytics endpoints and visualization in the SPA (WebGL2 path).
- Binary mask writing (VMSK) and workload‑aware transforms (TTR engine).
- Parallelized, resumable study orchestration and robust determinism harness.

References
- NIR dialects and versioning: docs/architecture/NIR_DIALECTS_AND_VERSIONING.md
- CLI commands: crates/shnn-cli/src/commands
- Compiler public API: crates/shnn-compiler/src/lib.rs and src/passes.rs