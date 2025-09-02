# hSNN Roadmap

This roadmap outlines the next implementation phases for hSNN. The project follows an incremental, CLI‑first approach with compilation‑safe steps and continuous test coverage. Each phase lists scope, deliverables, acceptance criteria, and risks.

Guiding principles
- Keep builds green and tests passing at each increment.
- Maintain the thin‑waist: typed, unit‑aware IR and stable storage schemas form the contract across layers.
- Prefer deterministic, reproducible workflows (seeded simulations, golden tests).
- Document public APIs and evolve via versioned ops and pass‑based upgraders.

Phase 1 — Flesh out Scaffolding (stabilize surfaces)
Goal: Solidify the newly scaffolded features and ensure they are useful to end users, while keeping core logic lean.

1) Visualization server
- Current: Minimal static server (no extra deps) + SPA with raster rendering; JSON endpoints /api/health, /api/list, /api/spikes.
- Next:
  - Add /api/topology endpoint to stream network metadata (counts, ranges) and, when available, topology edges (expanded or compressed).
  - Add SPA panels and controls:
    - File selection, basic zoom/pan, time windowing, neuron range filtering.
    - Palette/scales and render performance improvements (batching, offscreen canvas).
  - Optional WebGL2 (or wasm) path (behind a feature flag) for larger rasters.
- Deliverables:
  - Extended endpoints, SPA enhancements, examples documented in README + getting-started.
- Acceptance:
  - cargo build -p shnn-cli passes; interactive viz works for JSON exports from snn nir run.

2) TTR command (placeholder → stabilized JSON)
- Current: TTR apply reads TOML and emits a JSON mask of selected neuron IDs.
- Next:
  - Define TOML schema more concretely (program metadata, inputs/ranges, op list with selector/transform enums).
  - Add validation/diagnostics (duplicate ranges, empty masks).
  - Keep JSON mask until storage VMSK integration lands in Phase 2.
- Deliverables:
  - Documented schema and examples; improved CLI messages; unit tests for schema parsing and mask assembly.
- Acceptance:
  - cargo test -p shnn-cli exercises TTR schema parsing; CLI produces expected JSON masks.

3) Engine stubs (explicit surfaces for future transforms)
- Current: Lowering builds runtime network and stimuli; pass framework is in place.
- Next:
  - Introduce explicit “transform surfaces” in compiler/passes to enable future graph- and mask-based transforms (no-op until Phase 2).
  - Add small helper APIs to make pass composition cleaner and testable.
- Deliverables:
  - Pass utilities for pattern matching composite ops and emitting rewritten ops; examples in tests.
- Acceptance:
  - Tests demonstrate pass transforms on toy modules; no runtime behavior change unless passes are enabled by default.

4) Compiler passes (consolidation)
- Current: Canonicalize expands FC layers; Version upgrader is scaffolded.
- Next:
  - Add diagnostics and guardrails (size thresholds for expansion, optional skip for large FC).
  - Implement a minimal, well-documented upgrade rule (e.g., stdp@v1 → stdp@v2 with an additional attribute default).
- Deliverables:
  - Documented pass registry with stable pass names and ordering expectations.
- Acceptance:
  - Unit tests for controlled canonicalization; upgrade test creates identical runtime after default insertion.

Phase 2 — Core Logic Implementation (depth and performance)
Goal: Implement core logic where scaffolds exist, fleshing out deterministic pipelines and storage interop.

1) Compiler pass execution
- Implement a configurable pass pipeline:
  - Pass ordering constraints and feature flags (e.g., --no-canonicalize).
  - JSON/pretty print of the pipeline and transform stats (rewritten ops, timings, counts).
- Add canonicalization for additional composite ops (e.g., future layer types).
- Acceptance: Tests validating pipeline ordering and stats; CLI flag toggles take effect.

2) Engine runtime and determinism
- Improve simulation engine hooks for determinism tests:
  - Stable seeds; snapshot / golden outputs for tiny networks.
  - Performance guardrails (timestep divisibility, warnings for truncations).
- Acceptance: Golden deterministic tests added; docs on how to run and update goldens.

3) Storage integration (masks and events; early)
- Move TTR masks from JSON → VMSK using shnn-storage. Keep JSON export as convenience.
- Add event export/import (VEVT) for replay and visualization beyond spikes.
- Acceptance: CLI can emit and read VMSK/VEVT for small networks; docs updated; tests reading/writing binary artifacts.

Phase 3 — Study Runner and Higher‑Level Tooling
Goal: Make multi‑run experiments reproducible and easy.

1) Study runner
- Current: Sequential in‑process runner with summary export.
- Next:
  - Add sweeps (grid/list/linspace) and derived runs; seeds per repeat.
  - Folder structure for runs (e.g., out_dir/run_001_rep_01).
  - Optional parallelism (bounded worker pool).
- Acceptance: TOML schema supports sweeps; CLI orchestrates multiple jobs with clear logging; summary.json captures core metrics and metadata.

2) Rich CLI introspection
- JSON output option for op-list (including docs and attribute kinds).
- Print current pass pipeline and registry for reproducible reports.
- Acceptance: CLI subcommands print JSON suitable for tooling; covered by tests.

Phase 4 — Visualization features
Goal: Move from raster-only to multi-panel, topology‑aware visualization.

- Add topology/measure panels (neuron ranges, degree distribution, weight histograms).
- Add temporal filters and basic measurement queries (spike counts by range and time window).
- Move to WebGL2 or wasm for larger datasets with responsive UI (progressive streaming).
- Acceptance: SPA panels documented; endpoints optimized for typical experiments; optional GPU path behind a feature flag.

Phase 5 — Embedded/edge pathway (exploratory)
Goal: Provide a credible path to microcontroller or SoC deployment.

- Fixed‑point kernels and reduced‑precision LIF step in shnn-embedded.
- Minimal platform traits and CI builds for a Cortex‑M target (no device‑specific code in main repo).
- Acceptance: “hello lif” build for embedded; host shim tests; documentation.

Appendix — Risks and mitigations
- Pass expansion scale: Expand‑to‑synapses can explode for large layers. Mitigate with thresholds, streaming construction, or lazy kernels.
- Format churn: Keep IR and registry changes versioned; provide upgrade passes and CLI “upgrade” helpers.
- Visualization performance: Prefer Canvas/WebGL2 pivot; keep simple code paths for small cases.
- Binary schemas: Introduce cautiously with thorough round‑trip tests and versioned headers.

## Phase 7 — Documentation Consolidation & Storage/Data Interop (Completed)

Goal: Consolidate user/developer documentation, lock thin-waist contracts, and provide help-synced CLI examples for snapshot workflows and NIR pipelines.

Deliverables:
- Storage & Interop docs:
  - Snapshot Triples Contract and determinism guarantees in [docs/architecture/BINARY_SCHEMAS.md](docs/architecture/BINARY_SCHEMAS.md)
- CLI-first workflows:
  - Snapshot export/import and NIR pipeline examples in [docs/architecture/CLI_FIRST_ARCHITECTURE.md](docs/architecture/CLI_FIRST_ARCHITECTURE.md) and [docs/getting-started/README.md](docs/getting-started/README.md)
- FFI cookbook:
  - Snapshot/apply C API usage and ownership in [docs/development/FFI-ABI.md](docs/development/FFI-ABI.md)

Acceptance:
- Examples are help-synced with current CLI flags (snapshot export/import, nir run).
- Docs reference concrete, runnable commands producing JSON/VEVT outputs and serving via viz.

## Phase 8 — Release/Versioning (Completed)

Goal: Establish a reproducible, multi-crate release process with semver policy, CI artifacts (including cbindgen header), and a smoke validation matrix.

Deliverables:
- End-to-end release flow and checklists:
  - [docs/release/RELEASE_CHECKLIST.md](docs/release/RELEASE_CHECKLIST.md)
- FFI symbol/version policy, feature flags, and CI header artifact linkage:
  - [docs/development/FFI-ABI.md](docs/development/FFI-ABI.md)
- CHANGELOG discipline and cross-links:
  - [CHANGELOG.md](CHANGELOG.md)

Acceptance:
- CI job exists to generate and publish shnn_ffi.h via cbindgen (artifact).
- Local and CI instructions documented for HSNN_SKIP_CBINDGEN=1 test flows and a separate header-generation step.

## Status Sync (Phases 1–6)

- Phase 1 — Visualization server:
  - Status: Minimal static server + JSON endpoints implemented; interactive viz works with spikes JSON/VEVT.
  - Impacted by Phase 7/8 docs; examples updated and linked from Getting Started.
- Phase 2 — Storage integration (early):
  - Status: VEVT encode/decode implemented and used by CLI viz flow; VMSK migration planned later.
- Phase 3 — Study runner and higher-level tooling:
  - Status: Not yet implemented; Phase 8 adds reproducibility docs and placeholders; roadmap remains.
- Phase 4 — Visualization features:
  - Status: Advanced panels/WebGL2 remain future; minimal pathway documented and validated.
- Phase 5 — Embedded/edge pathway:
  - Status: Unchanged by 7/8.
- Phase 6 — Weight snapshot connectivity:
  - Status: Completed — trait, CLI snapshot export/import, and FFI snapshot/apply endpoints shipped and documented.

## References
- Current State: [docs/CURRENT_STATE.md](docs/CURRENT_STATE.md)
- NIR Dialects and Versioning: [docs/architecture/NIR_DIALECTS_AND_VERSIONING.md](docs/architecture/NIR_DIALECTS_AND_VERSIONING.md)
- CLI Architecture: [docs/architecture/CLI_FIRST_ARCHITECTURE.md](docs/architecture/CLI_FIRST_ARCHITECTURE.md)
- Binary Schemas: [docs/architecture/BINARY_SCHEMAS.md](docs/architecture/BINARY_SCHEMAS.md)
- Release Checklist: [docs/release/RELEASE_CHECKLIST.md](docs/release/RELEASE_CHECKLIST.md)
- FFI ABI: [docs/development/FFI-ABI.md](docs/development/FFI-ABI.md)
- Getting Started: [docs/getting-started/README.md](docs/getting-started/README.md)