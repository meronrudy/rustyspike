# hSNN — Spiking Hypergraph Neural Network

CLI‑first neuromorphic research "substrate" for building, verifying, compiling, running, and visualizing spiking neural networks (SNNs), emphasizing ease, reproducibility, and flexibility. 

Status: Core foundations are mostly implemented and tested. Working on figuring out the (NIR) path most erthing else is live through the compiler and CLI, Gotta modularize the MLIR shit..... dynamic op introspection, verification, lowering to runtime, simulation, JSON exports, adding a minimal visualization server with spike raster rendering. Several advanced features are scaffolded for incremental, compilation‑safe expansion.

Next up"
- Textual NIR with versioned, typed, unit‑aware ops, MLIR‑like syntax and round‑trip printing/parsing
- Static, introspectable Op Registry with attribute kinds and docs
- Compiler with verification, lowering, and a pass framework (canonicalization and version upgrader)
- CLI add NIR commands (compile, run, verify, op‑list), and a minimal viz server (serve + JSON endpoints)
- Storage and runtime foundations aligned to thin‑waist traits and zero‑copy binary schemas (design complete; select pieces implemented)

Workspace crates overview
- shnn-compiler
  - Purpose: Compile NIR to the runtime engine; verify modules; provide op registry and compiler passes.
  - Key APIs:
    - verify_module(&Module) — semantic/type/unit checks
    - list_ops() — expose OpSpec/AttributeSpec for dynamic introspection
    - compile_with_passes(&Module) — verify → run passes (canonicalize, upgrade versions) → lower → runnable program
  - Passes:
    - Canonicalize: expands connectivity.layer_fully_connected to explicit connectivity.synapse_connect ops
    - UpgradeVersions: scaffolding to evolve ops (e.g., lif@v0 → lif@v1) with defaulted attributes
- shnn-cli
  - Purpose: User‑facing CLI for the entire workflow (NIR compile/run/verify/op‑list; visualization server; study/TTR scaffolds).
  - Key commands:
    - snn nir compile, snn nir run, snn nir verify, snn nir op-list
    - snn viz serve (SPA + JSON endpoints) to visualize spike rasters from JSON outputs
  - Integration:
    - Drive shnn-compiler’s verify/list_ops/compile_with_passes and shnn-ir’s parse_text/to_text
- shnn-core
  - Purpose: Core SNN data structures and traits (thin‑waist “interfaces” for runtime/network behavior), LIF and STDP parameters, and network builder abstractions leveraged by the compiler runtime path.
- “Interfaces” (thin‑waist traits)
  - Purpose: The stable “contract” between layers; documented across crates (core/runtime/storage). These interfaces define how NIR lowering and storage I/O interoperate. (Conceptually “shnn-interfaces”; implemented as traits in shnn-core and related crates.)
- Additional crates (selected)
  - shnn-ir: Neuromorphic IR (Module/Operation, Attributes, printer/parser)
  - shnn-runtime: Network builder, simulation engine, stimuli, and execution
  - shnn-storage: Binary schema designs and utilities for masks/graphs/spikes (design complete; staged implementations)

Quick start
Prerequisites
- Rust toolchain (latest stable recommended)
- macOS, Linux, or Windows

Build
```bash
# From the repository root
cargo build --release
```

List registered NIR operations (dynamic op registry)
```bash
# Lists all dialects/ops/versions with attribute kinds
cargo run -p shnn-cli -- nir op-list --detailed
```

End‑to‑end example (compile → run → visualize)
```bash
# 1) Generate a textual NIR program
cargo run -p shnn-cli -- nir compile --output /tmp/demo.nirt

# 2) Run it and export spikes to JSON
cargo run -p shnn-cli -- nir run /tmp/demo.nirt --output crates/shnn-cli/test_workspace/results/run1.json

# 3) Serve the minimal viz app and browse the raster (default host:port shown)
cargo run -p shnn-cli -- viz serve --results-dir crates/shnn-cli/test_workspace/results
# Open the printed http://127.0.0.1:7878 in your browser
```

Project goals
- CLI‑first: All core workflows are command based (compile, verify, run, inspect, visualize)
- Thin‑waist: Typed, unit‑aware IR and stable storage schemas enable composability and evolution
- Determinism: Seeded execution with golden tests; reproducible studies and exports
- Extensibility: Static/dynamic introspection of ops; pass framework for canonicalization/upgrade

Project state
- Implemented (selected):
  - shnn-ir textual printer/parser (round‑trip)
  - shnn-compiler op registry, verification, lowering, pass manager
  - shnn-cli NIR commands and minimal viz server (SPA + JSON)
  - Conformance tests (round‑trip and compile‑run parity)
- Scaffolded (selected):
  - TTR (Task‑Aware Topology Reshaping) JSON mask emitter (placeholder for VMSK)
  - Study runner (sequential orchestrator)
  - Additional passes and endpoints for richer canonicalization and visualization

Documentation
- Architecture overview: [docs/architecture/README.md](docs/architecture/README.md)
- NIR dialects and versioning: [docs/architecture/NIR_DIALECTS_AND_VERSIONING.md](docs/architecture/NIR_DIALECTS_AND_VERSIONING.md)
- Current state summary: [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)
- Roadmap: [docs/ROADMAP.md](docs/ROADMAP.md)

License
Dual‑licensed under Apache‑2.0 and MIT. See LICENSE‑APACHE and LICENSE‑MIT.

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
