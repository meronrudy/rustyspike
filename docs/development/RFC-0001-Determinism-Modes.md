# RFC-0001: Determinism Modes

Status: Draft
Authors: hSNN Maintainers
Last-Updated: 2025-08-28
Related:
- [shnn-core DeterminismConfig](crates/shnn-core/src/network/mod.rs)
- [shnn-runtime run_fixed_step()](crates/shnn-runtime/src/simulation.rs)

## Summary

This RFC specifies the determinism modes for hSNN. It defines requirements, terminology, toggles, and acceptance tests for deterministic operation across the stack, especially for the CLI-first flow and programmatic APIs. The intent is to ensure reproducible, verifiable runs under fixed conditions with clear contracts for ordering, RNG, and timing.

## Goals

- Provide a deterministic execution mode with explicit, documented semantics.
- Enable reproducible spike streams and statistics (golden tests).
- Ensure stable ordering of:
  - input spikes,
  - routing targets,
  - pending spike queues with identical delivery times.
- Specify RNG usage and seeding.
- Establish acceptance tests and CI gates.

Non-goals (for this RFC)

- Cross-OS bit-for-bit equivalence (tracked separately; see Notes).
- Distributed multi-node determinism.
- GPU/accelerator determinism.

## Definitions

- Deterministic: Given the same program (NIR), same seed, same config, and same platform class (OS/arch + Rust), the outputs (spike tuples exported via canonical serializer) are identical across runs.
- Fixed-step: A discrete time stepping simulation with a constant dt; events that align to a step are applied in deterministic order.
- Event-driven: Internal routing may compute delivery times between steps; the pending queues must enforce stable ordering.

## Determinism Modes and Toggles

Determinism is configured via the thin waist (core runtime with fixed-step) and exposed to higher layers (CLI). A conceptual configuration struct exists in core:

- DeterminismConfig
  - enabled: bool
  - sort_inputs: bool
    - When true and enabled, input spikes are sorted by (timestamp, source_id) before enqueuing.
  - stable_routing: bool
    - When true and enabled, per-route target application order is sorted by NeuronId.
  - seed: Option<u64>
    - RNG seed for any stochastic features (Poisson stimuli etc.).

Current implementation references:
- Sorting of routing target pairs in [SpikeNetwork::process_spikes](crates/shnn-core/src/network/mod.rs).
- Stable pending spike dequeue order via (delivery_time, source_id) in [SpikeNetwork::get_next_spike](crates/shnn-core/src/network/mod.rs).
- Fixed-step API entry point [run_fixed_step()](crates/shnn-runtime/src/simulation.rs) using a validated SimulationParams and seed for reproducibility.

CLI flags (planned in later phases):
- --deterministic (enable full determinism mode)
- --seed <u64>
- --dt-ns <u64>
- --duration-ns <u64>

## Time and Ordering Semantics

When DeterminismConfig.enabled:
- Input ordering: If multiple input spikes occur at the same timestamp, they are enqueued sorted by ascending source_id.
- Routing ordering: For a given route, targets are iterated in ascending NeuronId.
- Pending queue: If multiple TimedSpikes share the same delivery_time, they are processed by ascending source_id.
- Step increments: The fixed-step driver uses a constant dt_ns validated to be > 0 and duration_ns ≥ dt_ns.
- RNG: A 64-bit LCG is used in shnn-runtime’s SimulationEngine; seeds are supplied by SimulationParams.random_seed.

## Acceptance Tests

- Core deterministic ordering test: insert two spikes with identical delivery_time but different source_id in reverse order and assert pending queue order is (3, 5). See [test_deterministic_insertion_order](crates/shnn-core/src/network/mod.rs).
- Runtime reproducibility:
  - Build two identical networks; run `run_fixed_step` with the same seed.
  - Assert steps_executed, total_spikes, and exported spike tuple list are identical. See [test_determinism_reproducibility](crates/shnn-runtime/src/simulation.rs).

## CI Integration

- CI runs `cargo fmt --check` and `cargo clippy -D warnings` on Linux/macOS/Windows (stable).
- `cargo test --workspace --all-features` executes determinism tests.
- Future: Add a matrix job to re-run determinism golden tests and assert SHA256 of exported spike tuples for selected examples.

## Known Limitations and Notes

- Floating-point variation across OS/architectures can affect bitwise identity; exact match is required for same OS/arch. Cross-OS parity will be measured and documented with numeric tolerances in a later RFC.
- Determinism is guaranteed only when:
  - determinism.enabled is true,
  - no non-deterministic features (e.g., parallel reordering) bypass sorting,
  - fixed-step driver is used or the event-driven implementation preserves total ordering for ties.

## Migration and Usage Guidance

- Library users:
  - Enable determinism via builder-style API (core) and use fixed-step entry point (runtime).
  - Set seeds on any stochastic stimuli.
- CLI users (future):
  - `snn nir run --deterministic --seed 1234 --dt-ns 100000 --duration-ns 1000000`
- Tests:
  - Prefer “golden” spike tuple exports for stable examples and validate by hash.

## Future Work

- Export deterministic mode via CLI flags and persist determinism config in run metadata.
- Add golden tests for selected canonical networks in CI (hash-checked).
- Investigate cross-OS numeric tolerance reporting; document where tolerances are acceptable vs. strict bitwise matching.
- Integrate determinism config into NIR runtime.simulate ops as attributes.
