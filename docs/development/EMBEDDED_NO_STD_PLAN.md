# SHNN Embedded: No-Std Optimization and Hardening Plan

This plan upgrades the embedded crate to meet tight MCU constraints while keeping deterministic timing and sparse hypergraph efficiency. It addresses current build issues, organizes features for no-std correctness, and lays out performance-focused changes.

Key files to modify:
- Core crate wiring: [crates/shnn-embedded/src/lib.rs](crates/shnn-embedded/src/lib.rs)
- Memory/data layout: [crates/shnn-embedded/src/embedded_memory.rs](crates/shnn-embedded/src/embedded_memory.rs)
- Network scheduling: [crates/shnn-embedded/src/embedded_network.rs](crates/shnn-embedded/src/embedded_network.rs)
- HAL abstractions: [crates/shnn-embedded/src/hal.rs](crates/shnn-embedded/src/hal.rs)

Current issues observed (from host builds):
- Duplicate panic handler and missing cfg feature leading to E0152 and unexpected cfg warnings in [lib.rs](crates/shnn-embedded/src/lib.rs:240).
- `env!` at build-time for target arch causing error: replace with `cfg` selection in [lib.rs](crates/shnn-embedded/src/lib.rs:151).
- `heapless::pool` is gated on select targets; unused imports cause E0432 per host in [embedded_memory.rs](crates/shnn-embedded/src/embedded_memory.rs:11).
- `Box` usage in `HALFactory::create_hal()` not valid in no-std without `alloc` in [hal.rs](crates/shnn-embedded/src/hal.rs:616).

## Goals

1) Memory efficiency
- Adopt structure-of-arrays (SoA) for neuron states and synapses to improve locality.
- Use compact fixed-point (Q8.8 or Q12.4) when acceptable; keep Q16.16 default.
- Sparse hypergraph adjacency with fixed-capacity ring buffers; avoid dynamic allocation in no-std.

2) Determinism and timing
- Event-driven update: only active neurons/hyperedges are processed per tick.
- Keep ISRs minimal; push processing into main-loop tasks.
- Provide compile-time bounds for spike buffers, fan-in/fan-out.

3) Energy per spike
- Fixed-point kernels; prefer lookup tables over expensive functions.
- Batch hyperedge updates to coalesce memory traffic.

4) Graph partitioning
- Optional static partition descriptor to group nodes for cache/DMA alignment.
- Partition-local buffers for spikes to reduce cross-partition traffic.

5) Plasticity
- Event-driven STDP with quantized weights and periodic pruning of weak synapses.

6) Rust/no-std hygiene
- #![no_std] baseline.
- Make `std` an explicit feature, used only for tests and host examples.
- Gate `Box` and trait-object returns behind `alloc` feature.
- Avoid importing platform-gated modules unless cfg/feature enables them.

---

## Concrete Changes

A) Build hardening (no-std correctness)
- Add feature `std` to the crate and gate std-specific code accordingly.
- Replace build-time arch with cfg-derived const:
  - In [lib.rs](crates/shnn-embedded/src/lib.rs:151), replace `env!("CARGO_CFG_TARGET_ARCH")` with:
    - `const TARGET_ARCH: &str = "arm"` under `cfg(target_arch = "arm")`, etc.
- Gate panic handler so it compiles only in true no-std contexts:
  - `#[cfg(all(not(test), not(feature = "std")))]` on panic handler in [lib.rs](crates/shnn-embedded/src/lib.rs:241).
- Remove `heapless::pool` import and other unused imports in [embedded_memory.rs](crates/shnn-embedded/src/embedded_memory.rs:11-12).
- Gate `HALFactory::create_hal()` behind `alloc`:
  - Under `#[cfg(feature = "alloc")]`, provide `create_hal_box()` that returns `Box<dyn EmbeddedHAL<...>>`.
  - Under no-alloc, provide `create_hal()` that returns `EmbeddedError::UnsupportedPlatform` or static concrete HAL where available.

B) Memory layout upgrades
- Introduce SoA neuron container:
  - New type `EmbeddedNeuronSoA<T>` in [embedded_neuron.rs](crates/shnn-embedded/src/embedded_neuron.rs:1) maintaining separate arrays (v, u, refractory, params).
  - Provide batch update kernels specialized by neuron type (const generic for capacity).
- Refine sparse matrices:
  - In [embedded_memory.rs](crates/shnn-embedded/src/embedded_memory.rs:321), add small-index encoding for (row, col) pairs (u16/u8) where safe.
  - Add `set_weight_batch()` and a batched `multiply_vector_sparse()` that iterates non-zero triplets.

C) Event-driven pipeline
- In [embedded_network.rs](crates/shnn-embedded/src/embedded_network.rs:343):
  - Prepare `ActiveList` of neurons triggered by inputs and recent spikes.
  - Process synapses only from spiking presynaptic sources in the last tick (delay satisfied).
  - Use fixed-capacity ring buffers per synapse; saturating behavior with stats.

D) Plasticity (quantized STDP)
- New module (or extend synapse) for event-driven STDP:
  - Weight delta from pre/post spike timing in discrete ticks; quantize to Qx.y step.
  - Add `prune_below(threshold)` to decay and prune weak connections sparsely.

E) Scheduling and interrupts
- Keep `critical_section` minimal; add explicit docs on ISR usage timing in [lib.rs](crates/shnn-embedded/src/lib.rs:255).
- Add optional RTIC integration sample guarded behind feature `rtic`.

F) Partitioning (optional)
- Add `PartitionMap` type and helpers to assign neurons/hyperedges into partitions.
- Track cross-partition edges to allow deferred processing.

G) Tests and cross-compiles
- Host tests with `--features std,alloc`.
- Add cross-compile job (no-std) for Cortex-M:
  - `thumbv7em-none-eabihf` build of `shnn-embedded` with `--no-default-features` and feature set: `fixed-point`.
  - Ensure no panics/alloc in that path.

---

## Implementation Checklist

- Build hygiene
  - [ ] Add `std` feature in [crates/shnn-embedded/Cargo.toml](crates/shnn-embedded/Cargo.toml:50) and gate std-only code in [lib.rs](crates/shnn-embedded/src/lib.rs:43).
  - [ ] Replace `env!` arch with `cfg`-selected `const TARGET_ARCH` in [lib.rs](crates/shnn-embedded/src/lib.rs:151).
  - [ ] Gate panic handler as `#[cfg(all(not(test), not(feature = "std")))]` in [lib.rs](crates/shnn-embedded/src/lib.rs:241).
  - [ ] Remove `heapless::pool` and unused imports in [embedded_memory.rs](crates/shnn-embedded/src/embedded_memory.rs:11-12).
  - [ ] Gate `HALFactory::create_hal()` to `alloc` in [hal.rs](crates/shnn-embedded/src/hal.rs:616) and add `create_hal_box()` under `alloc`.

- Memory/Data layout
  - [ ] Add `EmbeddedNeuronSoA<T>` with SoA arrays in [embedded_neuron.rs](crates/shnn-embedded/src/embedded_neuron.rs:1) and batch kernels.
  - [ ] Add sparse triplet encoding and batched ops in [embedded_memory.rs](crates/shnn-embedded/src/embedded_memory.rs:321-473).

- Event-driven updates
  - [ ] Maintain `ActiveList` and only process active nodes and satisfied-delay synapses in [embedded_network.rs](crates/shnn-embedded/src/embedded_network.rs:343-384).
  - [ ] Spike buffer stats and thresholds for backpressure.

- Plasticity
  - [ ] Add quantized STDP trait and implementation on synapses; pruning function; update tests.

- Partitioning
  - [ ] Optional `PartitionMap` with per-partition buffers (behind feature `partitioning`).

- Tests and CI
  - [ ] Add unit tests for SoA kernels and sparse ops (host `std`).
  - [ ] Add cross-compile job for `thumbv7em-none-eabihf` no-std build.
  - [ ] Document ISR timing constraints and critical section rules.

---

## Operational Guidance

### ISR timing budgets

Purpose: guarantee bounded interrupt latency and predictable main-loop work on MCUs.

- Define your tick period (dt) and the maximum work allowed per tick in the main loop. Main-loop work should remain deterministic and capped via bounded processing of active neurons and routed spikes, as implemented in the event-driven scheduler described in [crates/shnn-embedded/src/embedded_network.rs](crates/shnn-embedded/src/embedded_network.rs).
- Split responsibilities:
  - ISR: keep to the minimum. Only timestamp/edge-capture hardware events, enqueue a compact event record into a fixed-capacity queue, and set a flag for the main loop. Avoid any math, graph traversal, or dynamic dispatch in ISR.
  - Main loop: drain ISR-produced events, update neurons that are currently active, route spikes for presynaptic sources that fired with delay satisfied, and record stats. Enforce strict per-tick budgets.
- Budgeting model (device-agnostic):
  - Inputs:
    - cycles_per_update: worst-case cycles to update one neuron state in the chosen fixed-point format.
    - cycles_per_route: worst-case cycles to route one spike across matching synapses.
    - updates_per_tick: configured cap of active neuron updates per tick.
    - spikes_per_tick: configured cap of routed spikes per tick.
    - f_cpu_hz: CPU clock frequency.
  - Estimated main-loop time per tick (microseconds):
    - time_us ≈ 1e6 * (updates_per_tick * cycles_per_update + spikes_per_tick * cycles_per_route) / f_cpu_hz + isr_overhead_us
  - Choose dt such that time_us <= 0.7 * dt_us to preserve margin for jitter, DMA, and occasional bursty activity.
- Measurement guidance:
  - On target, surround main-loop sections with GPIO toggle or DWT cycle counter sampling (on Cortex-M) and record min/avg/max across runs.
  - Validate that worst-case ISR latency remains below application thresholds under stress tests (max event rate, full buffers).
- Architectural notes:
  - Prefer compile-time capacities (const generics) for active lists and spike queues to keep stack/flash use explicit.
  - Avoid nested critical sections; keep any required section as short as possible and only around queue pointer updates.
  - If using an RTOS or RTIC, assign priorities so ISR only preempts low-priority background tasks; main loop should not be preempted by other compute-intensive tasks.

### Buffer saturation and backpressure

Bounded buffers guarantee determinism; establish explicit behavior when full.

- Active list semantics:
  - Fixed-capacity queue for indices of neurons scheduled to update. When full, new entries are dropped or deferred according to policy.
  - Recommended policy: drop-new and increment a saturating counter to record misses. This guarantees upper bound on work; missed updates are observable via telemetry.
- Spike routing queue semantics:
  - Fixed-capacity queue for spikes produced this tick. When full, prefer drop-oldest to preserve recent activity locality, or drop-new if preserving causality timelines is more important. Choose one policy and document it.
- Telemetry counters (saturating):
  - active_dropped, spikes_dropped, isr_events_dropped, queue_high_watermarks for each queue.
  - Provide a periodic dump or a debug readout path on host builds to verify tuning.
- Backpressure strategies:
  - Increase tick period dt slightly or reduce per-tick caps to keep within ISR/main-loop budgets.
  - Reduce network fan-out or prune weak synapses more aggressively to limit routing pressure.
  - Use optional partitioning to localize traffic; cross-partition spikes can be deferred or rate-limited (see partitioning section).

### Fixed-point Q-format selection

Select a Q-format to balance dynamic range, precision, code size, and throughput.

- Typical options:
  - Q8.8: range approximately [-128, +127.996], resolution approximately 0.0039. Best for small microcontrollers where memory and speed are critical.
  - Q12.4: range approximately [-2048, +2047.9375], resolution 0.0625. Middle ground when dynamics are moderate and lower precision is acceptable.
  - Q16.16: range approximately [-32768, +32767.99998], resolution approximately 1.5e-5. Default for maximum stability and easier porting from float models.
- Selection checklist:
  - Identify maximum expected membrane potential, synaptic current, and weight range across your model.
  - Ensure headroom for transient bursts (consider at least 2x safety margin).
  - Validate cumulative rounding behavior over long runs; use symmetric rounding where possible to limit bias.
  - Verify saturation behavior: ensure all arithmetic saturates rather than wraps. Clamp plasticity updates and neuron states to within model limits.
- Memory and performance:
  - Narrower Q formats reduce memory footprint and can improve cache locality and throughput on Cortex-M.
  - Instruction sequences for multiply-accumulate may differ; benchmark your kernels on target before finalizing.
- Portability:
  - Keep all constants and gains represented in the chosen Q format at compile time to avoid runtime conversion overhead.
  - Host builds can continue to use standard math features; keep embedded builds cleanly gated to avoid accidental float use.

### Sizing checklist (practical)

- Choose dt based on the physical model and device clock; reserve at least 30% margin.
- Estimate worst-case per-tick update and routing operations from expected activity levels; set compile-time caps accordingly.
- Size active list and spike queues for expected burst lengths (for example, 99.9th percentile). Confirm with high-watermark telemetry.
- Select Q-format and validate numerical stability under long-run tests with plasticity enabled.
- Enable pruning to keep sparse structure tractable and reduce traffic pressure.
- Re-measure ISR latency and main-loop time after each network or parameter change.

---

## Deterministic Update Flow (Mermaid)

```mermaid
flowchart TD
  A[Inputs (current tick)] --> B[Apply to input neurons (SoA)]
  B --> C[Fetch active list (spikes/delays satisfied)]
  C --> D[Update neurons (batch kernels, fixed-point)]
  D --> E[Generate spikes]
  E --> F[Enqueue to synapse ring buffers]
  F --> G[Event-driven plasticity (quantized STDP)]
  G --> H[Update stats, buffers, time]
```

---

## Notes

- All dynamic dispatch (`Box<dyn ...>`) must be behind `alloc`. For pure no-std, use concrete types or compile-time selection.
- Prefer const generics for capacities (neurons, synapses, buffers) over runtime-configured heap allocations.
- Document bounds and failure modes: saturating ring buffers, pruning policies, and how to tune Q formats.
