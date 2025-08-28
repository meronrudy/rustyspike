# NIR Dialects and Versioning

Status: v0 (text-only, static registry), minimal but compiling.

Purpose:
- Define the current Neuromorphic IR (NIR) dialects, operations, and attributes
- Establish versioning policy for evolving ops and dialects
- Document units and canonical textual representation
- Provide guidance for upgrades and compatibility

Dialects (v0):
- neuron: neuron dynamics and defaults
- plasticity: learning rules and parameters
- connectivity: topology specifications (layers, explicit synapses)
- stimulus: external inputs and patterns
- runtime: simulation control and execution

Operation Specs (current registry)
See dynamic listing via CLI introspection:
- Run [shnn-cli nir.op_list()](crates/shnn-cli/src/commands/nir.rs:292) using: `snn nir op-list --detailed`
- Backed by the compiler registry [shnn_compiler::list_ops()](crates/shnn-compiler/src/lib.rs:232)

Current Ops (v1):
- neuron.lif@v1
  - attrs:
    - tau_m: DurationNs (ns)
    - v_rest: VoltageMv (mV)
    - v_reset: VoltageMv (mV)
    - v_thresh: VoltageMv (mV)
    - t_refrac: DurationNs (ns)
    - r_m: ResistanceMohm (MΩ)
    - c_m: CapacitanceNf (nF)
- plasticity.stdp@v1
  - attrs:
    - a_plus: f32
    - a_minus: f32
    - tau_plus: DurationNs (ns)
    - tau_minus: DurationNs (ns)
    - w_min: Weight (unitless f32)
    - w_max: Weight (unitless f32)
- connectivity.layer_fully_connected@v1
  - attrs:
    - in: RangeU32 (inclusive)
    - out: RangeU32 (inclusive)
    - weight: Weight (unitless f32)
    - delay: DurationNs (ns)
- connectivity.synapse_connect@v1
  - attrs:
    - pre: NeuronRef (%nX)
    - post: NeuronRef (%nX)
    - weight: Weight (unitless f32)
    - delay: DurationNs (ns)
- stimulus.poisson@v1
  - attrs:
    - neuron: NeuronRef (%nX)
    - rate: RateHz (Hz)
    - amplitude: CurrentNa (nA)
    - start: TimeNs (ns)
    - duration: DurationNs (ns)
- runtime.simulate.run@v1
  - attrs:
    - dt: DurationNs (ns)
    - duration: DurationNs (ns)
    - record_potentials: bool
    - seed: i64 (optional)

Textual Format (MLIR-like, v0):
- Header: dialect.name@vN
- Attributes: key = value, comma-separated, canonical units appended
- Module form:
  nir.module {
    neuron.lif@v1 { tau_m = 20000000 ns, v_rest = -70 mV, v_reset = -70 mV, v_thresh = -50 mV, t_refrac = 2000000 ns, r_m = 10 MΩ, c_m = 1 nF }
    connectivity.layer_fully_connected@v1 { in = 0..9, out = 10..59, weight = 1, delay = 1000000 ns }
    stimulus.poisson@v1 { neuron = %n0, rate = 20 Hz, amplitude = 10 nA, start = 0 ns, duration = 500000000 ns }
    runtime.simulate.run@v1 { dt = 100000 ns, duration = 500000000 ns, record_potentials = false, seed = 42 }
  }

Canonical Units (printer/parser v0):
- TimeNs, DurationNs: printed as "<u64> ns"
- VoltageMv: "<f32> mV"
- ResistanceMohm: "<f32> MΩ"
- CapacitanceNf: "<f32> nF"
- CurrentNa: "<f32> nA"
- RateHz: "<f32> Hz"
- Weight: "<f32>" (dimensionless)
- RangeU32: "start..end"
- NeuronRef: "%n<id>"

Verification (v0):
- Presence and type/units check per registry spec [shnn_compiler::verify_module()](crates/shnn-compiler/src/lib.rs:239)
- Semantic/bounds checks:
  - neuron.lif: tau_m > 0; r_m > 0; c_m > 0
  - plasticity.stdp: tau_plus, tau_minus > 0; w_min <= w_max
  - connectivity.layer_fully_connected: each RangeU32 start <= end
  - stimulus.poisson: rate_hz, amplitude >= 0
  - runtime.simulate.run: dt, duration > 0
- Tests: see
  - [verify_bounds.rs](crates/shnn-compiler/tests/verify_bounds.rs:1)
  - [nir_parity_roundtrip.rs](crates/shnn-compiler/tests/nir_parity_roundtrip.rs:1)
  - [roundtrip_synapse_connect.rs](crates/shnn-ir/tests/roundtrip_synapse_connect.rs:1)

Versioning Policy:
- SemVer-like for ops: name@vN, where N increases on any backward-incompatible change
- Allowed changes without bump:
  - Documentation clarifications
  - Narrowly relaxing verifier checks that do not break existing valid programs
- Requires version bump:
  - Changing required attribute sets
  - Changing attribute kinds or units
  - Altering runtime semantics that affects results deterministically
- Deprecation:
  - Older versions remain in the registry during a grace period
  - The compiler provides a version upgrader pass to transform vK → vK+1 where feasible
  - The CLI will expose `snn nir upgrade --in in.nirt --out out.nirt` (planned)

Upgrade and Canonicalization Passes (scaffold v0):
- Pass framework defined in [shnn-compiler passes](crates/shnn-compiler/src/passes.rs:1)
- Registered no-op passes:
  - canonicalize: intended to expand composite ops (e.g., layer_fully_connected → explicit synapses) and normalize attrs
  - upgrade_versions: intended to add defaulted attributes and rewrite older op versions to current ones
- Entry point: [compile_with_passes()](crates/shnn-compiler/src/lib.rs:537)
  - Runs verify → passes → lowering

CLI Introspection:
- `snn nir op-list [--detailed]` prints the dynamically registered op schemas
- Implemented at [NirOpList.execute()](crates/shnn-cli/src/commands/nir.rs:292)
- Backed by [list_ops()](crates/shnn-compiler/src/lib.rs:233) and human-readable kinds via [AttrKind::name()](crates/shnn-compiler/src/lib.rs:109)

Conformance and Round-trip Tests:
- IR round-trip parity including connectivity.synapse_connect:
  - [roundtrip_synapse_connect.rs](crates/shnn-ir/tests/roundtrip_synapse_connect.rs:1)
- Compile→run parity between Module and parse(text(Module)):
  - [nir_parity_roundtrip.rs](crates/shnn-compiler/tests/nir_parity_roundtrip.rs:1)
- Verifier negative/positive coverage:
  - [verify_bounds.rs](crates/shnn-compiler/tests/verify_bounds.rs:1)

Future Extensions:
- JSON op-list output in CLI
- Rich diagnostics with spans and suggestions
- Passes to expand layers conditionally (size thresholds) and eliminate redundancy
- Additional dialects (e.g., measurement, analysis), plus custom research.* namespaces
- Binary IR and MLIR interop

Change Log:
- v0.1: Initial dialect spec, version policy, registry-backed introspection, verifier semantics, pass framework scaffold, and conformance tests documented.