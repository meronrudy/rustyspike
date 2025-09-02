# SHNN FFI ABI: C API, Determinism, Ownership, and Versioning

This document describes the thin-waist C ABI exposed by the SHNN project for building networks, running deterministic fixed-step simulations, and retrieving simulation outputs (VEVT bytes). It also covers header generation and distribution, memory ownership rules, error handling, determinism guarantees, and versioning/stability policy.

References:
- Core C bindings implementation: [crates/shnn-ffi/src/c_bindings.rs](crates/shnn-ffi/src/c_bindings.rs)
- Build script (header generation): [crates/shnn-ffi/build.rs](crates/shnn-ffi/build.rs)
- Generated, distributable header path (repo): [crates/shnn-ffi/include/shnn_ffi.h](crates/shnn-ffi/include/shnn_ffi.h)
- cbindgen configuration: [crates/shnn-ffi/cbindgen.toml](crates/shnn-ffi/cbindgen.toml)
- CI workflow publishes header as artifact: [.github/workflows/ffi-header.yml](.github/workflows/ffi-header.yml)

## Overview

The FFI provides a minimal, stable surface that:
- Creates a network builder (opaque handle).
- Adds neurons and synapses.
- Builds an executable network handle.
- Runs a deterministic fixed-step simulation.
- Returns a buffer of VEVT bytes containing spike events.
- Provides an explicit deallocator for any buffers returned by the library.

The ABI uses opaque pointers to hide Rust internals and to keep the surface stable across internal refactoring.

## Header Generation & Distribution

- The header is generated at build time by cbindgen in two locations:
  1) In the Cargo build OUT_DIR for local consumers.
  2) Into the repo-distributed include directory: [crates/shnn-ffi/include/shnn_ffi.h](crates/shnn-ffi/include/shnn_ffi.h)

- CI publishes the generated header as an artifact on each push/PR to main/master:
  - See [.github/workflows/ffi-header.yml](.github/workflows/ffi-header.yml)

- The cbindgen settings are controlled by [crates/shnn-ffi/cbindgen.toml](crates/shnn-ffi/cbindgen.toml), targeting C with pragma once/include guards, and doxygen-style documentation.

## Build & Linking

- Library crate: [crates/shnn-ffi/Cargo.toml](crates/shnn-ffi/Cargo.toml)
- Crate builds cdylib/staticlib/rlib, name: `shnn_ffi`.
- Include the generated header and link to the built library as appropriate for your platform.

Example (Linux gcc):
```sh
# Build Rust library
cargo build -p shnn-ffi --release

# Headers live here after build:
ls ./crates/shnn-ffi/include/shnn_ffi.h

# Compile your C client
gcc -I./crates/shnn-ffi/include -L./target/release -lshnn_ffi -o client client.c
LD_LIBRARY_PATH=./target/release ./client
```

## ABI Conventions

- Opaque handles:
  - Network builder handle: created first, used to configure neurons/synapses.
  - Network handle: created by “build” from the builder; used to simulate.

- Memory Ownership:
  - All buffers returned by the library (e.g., VEVT bytes) are owned by the caller after return.
  - Callers must free such buffers using the provided free function.

- Error Handling:
  - Functions return integer codes (0 = OK, non-zero = error), and when applicable set an out-parameter length or pointer.
  - Errors are deterministic and stable; map to string categories or codes as our ABI grows.

- Thread Safety:
  - Builder and network handles are not thread-safe unless otherwise documented.
  - Use a single-threaded ownership model for the handles.

- Determinism:
  - Fixed-step simulations guarantee deterministic processing when determinism is configured in the runtime.
  - Input ordering is optionally normalized (by timestamp and source id) and routing order can be stabilized.
  - Binary output (VEVT) is reproducible for identical inputs and determinism configuration.

## VEVT: Binary Spike Event Format

- Magic/version header, checksums, followed by event records with:
  - timestamps (ns)
  - neuron ids
  - event payload (if any)

Integrity checks protect against truncated or corrupted data.

Decoding and encoding live in the storage crate:
- [crates/shnn-storage/src/vevt.rs](crates/shnn-storage/src/vevt.rs)

## Minimal C API (Conceptual)

The header exposes a small set of C functions. An indicative example of the surface:

- Builder Lifecycle:
  - `hsnn_network_builder_new(...) -> hsnn_builder_t*`
  - `hsnn_add_neuron_range(builder, first_id, count)`
  - `hsnn_add_synapse_simple(builder, src_id, dst_id, weight)`
  - `hsnn_network_build(builder) -> hsnn_network_t*`
  - `hsnn_network_free(network)`

- Simulation:
  - `hsnn_run_fixed_step_vevt_consume(network, steps, out_ptr, out_len) -> int`

- Buffer Management:
  - `hsnn_free_buffer(ptr)`

Note: See the generated header for the exact signatures for your build. We keep the names stable and add new APIs with suffixes when needed.

## Example C Usage

```c
#include "shnn_ffi.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

int main(void) {
    // 1) Create builder
    hsnn_builder_t* builder = hsnn_network_builder_new();
    if (!builder) {
        fprintf(stderr, "Failed to create builder\n");
        return 1;
    }

    // 2) Define neurons [0..9]
    if (hsnn_add_neuron_range(builder, 0, 10) != 0) {
        fprintf(stderr, "Failed to add neurons\n");
        return 1;
    }

    // 3) Connect 0 -> 1, weight 0.5
    if (hsnn_add_synapse_simple(builder, 0, 1, 0.5f) != 0) {
        fprintf(stderr, "Failed to add synapse\n");
        return 1;
    }

    // 4) Build network
    hsnn_network_t* net = hsnn_network_build(builder);
    if (!net) {
        fprintf(stderr, "Failed to build network\n");
        return 1;
    }

    // Optionally free/destroy builder here if API requires it
    // hsnn_network_builder_free(builder);

    // 5) Run deterministic fixed-step simulation and get VEVT bytes
    uint8_t* vevt_bytes = NULL;
    size_t vevt_len = 0;
    int rc = hsnn_run_fixed_step_vevt_consume(net, /*steps=*/100, &vevt_bytes, &vevt_len);
    if (rc != 0) {
        fprintf(stderr, "Simulation failed, rc=%d\n", rc);
        return 1;
    }

    printf("VEVT len: %zu\n", vevt_len);

    // 6) Free returned buffer
    hsnn_free_buffer(vevt_bytes);

    // 7) Destroy network
    hsnn_network_free(net);
    return 0;
}
```

## Determinism: Details

- With determinism enabled, the runtime:
  - Sorts input spikes by (timestamp_ns, source_id).
  - Optionally stabilizes routing order by target_id.
  - Uses fixed time step for updates.
- Results (VEVT) are bitwise reproducible, assuming identical inputs and configuration.

## Stability Policy

- The ABI is additive and versioned in the generated header via cbindgen.
- Breaking changes:
  - Avoided whenever possible; if needed, introduced with renamed functions (suffixes) while keeping legacy symbols for a deprecation window.
- Opaque handles:
  - Allow internal refactoring without breaking consumers.
- Binary format (VEVT):
  - Versioned and validated; decoders perform sanity checks.

## Troubleshooting

- Header not generated:
  - Ensure `cargo build -p shnn-ffi` ran, and check the build log for cbindgen output.
  - Verify [crates/shnn-ffi/cbindgen.toml](crates/shnn-ffi/cbindgen.toml) exists and is readable.

- Link errors:
  - Verify your compiler/toolchain is picking up `-I` for the include directory and `-L` for the Rust target directory.
  - Ensure the library name is `shnn_ffi` (cdylib/staticlib).

- Buffer leaks:
  - Always free buffers returned by the library with `hsnn_free_buffer`.

- Non-deterministic outputs:
  - Confirm determinism is enabled for the runtime and that inputs are fed identically.

## Weight Snapshot C API (Phase 7/8)

Alongside fixed-step simulation and VEVT export, the C ABI provides a minimal surface for weight snapshots to integrate with external optimizers.

Structures:
- HSNN_WeightTriple
  - typedef struct { uint32_t pre; uint32_t post; float weight; } HSNN_WeightTriple;

Functions:
- [hsnn_network_snapshot_weights()](crates/shnn-ffi/src/c_bindings.rs:0)
  - Export all current weights as a malloc-allocated array of HSNN_WeightTriple.
  - Signature (conceptual):
    - int hsnn_network_snapshot_weights(const HSNN_Network* net, HSNN_WeightTriple** out_ptr, size_t* out_len);
- [hsnn_network_apply_weight_updates()](crates/shnn-ffi/src/c_bindings.rs:0)
  - Apply weight updates from an array of HSNN_WeightTriple to existing connections.
  - Missing connections are ignored; returns number of updates applied.
  - Signature (conceptual):
    - int hsnn_network_apply_weight_updates(HSNN_Network* net, const HSNN_WeightTriple* ptr, size_t len, size_t* out_applied);

Memory ownership:
- The array returned by hsnn_network_snapshot_weights is owned by the caller after return and must be freed via:
  - hsnn_free_buffer((uint8_t*)ptr);

Example (C):
```c
#include "shnn_ffi.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  HSNN_NetworkBuilder* b = NULL;
  if (hsnn_network_builder_new(&b) != 0) return 1;
  hsnn_network_builder_add_neuron_range(b, 0, 2);
  hsnn_network_builder_add_synapse_simple(b, 0, 1, 0.2f);

  HSNN_Network* net = NULL;
  if (hsnn_network_build(b, &net) != 0 || !net) return 2;

  HSNN_WeightTriple* triples = NULL; size_t n = 0;
  if (hsnn_network_snapshot_weights(net, &triples, &n) != 0) return 3;
  if (n > 0) {
    printf("pre=%u post=%u w=%f\n", triples[0].pre, triples[0].post, triples[0].weight);
  }
  hsnn_free_buffer((uint8_t*)triples);

  HSNN_WeightTriple upd = { .pre = 0, .post = 1, .weight = 0.9f };
  size_t applied = 0;
  if (hsnn_network_apply_weight_updates(net, &upd, 1, &applied) != 0) return 4;
  printf("applied=%zu\n", applied);

  hsnn_network_free(net);
  return 0;
}
```

Notes:
- Headers are generated by cbindgen; see [crates/shnn-ffi/build.rs](crates/shnn-ffi/build.rs) and the distributed header under crates/shnn-ffi/include.
- On CI/toolchains that lack newer Cargo features, header generation can be skipped with HSNN_SKIP_CBINDGEN=1, while still testing the ABI.

## Roadmap Notes

- Additional APIs will be added to configure determinism from C and to feed input spikes and retrieve outputs incrementally.
- Hardware backend integration remains gated under opt-in features for the Rust side; the ABI surface will expose stable toggles progressively.

## Symbol & Versioning Policy (Phase 4)

- Symbol prefix: hsnn_* for C-facing functions, opaque types prefixed with HSNN_.
- Additive evolution:
  - New APIs are added without breaking existing symbols (MINOR).
  - Breaking changes introduce suffixed symbols (e.g., hsnn_run_fixed_step_vevt_consume_v2) and maintain legacy forms during a deprecation window (MAJOR when removed).
- ABI version macro in generated header:
  - Header includes a version define (e.g., HSNN_FFI_ABI_VERSION) controlled via cbindgen docs/comments in [crates/shnn-ffi/build.rs](crates/shnn-ffi/build.rs:0).
- Opaque handles and explicit sizes keep ABI stable across internal refactors.

## Feature Flags & Build Matrix

- c-bindings: enables the C ABI surface ([crates/shnn-ffi/src/c_bindings.rs](crates/shnn-ffi/src/c_bindings.rs:0))
- HSNN_SKIP_CBINDGEN=1: environment guard to skip cbindgen during tests on constrained toolchains ([crates/shnn-ffi/build.rs](crates/shnn-ffi/build.rs:0))
- Optional backends (cuda, opencl, fpga, rram, intel-loihi, spiNNaker) remain opt-in and do not affect the thin-waist C ABI by default.

## Testing & CI Flows

Local (tests, skipping header generation):
- HSNN_SKIP_CBINDGEN=1 cargo test -p shnn-ffi --features c-bindings -- --nocapture

Header generation (CI job on modern toolchain):
- cargo install cbindgen
- cargo build -p shnn-ffi --release
- cbindgen crates/shnn-ffi -o crates/shnn-ffi/include/shnn_ffi.h

Artifact:
- CI publishes header as artifact name "shnn_ffi.h" from [crates/shnn-ffi/include/shnn_ffi.h](crates/shnn-ffi/include/shnn_ffi.h:0)
- The FFI docs link to the header location and artifact in CI.

