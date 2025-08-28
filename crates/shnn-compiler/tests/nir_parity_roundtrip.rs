//! NIR conformance: compile→run parity between original Module and parse(text(Module))

use shnn_compiler::compile_module;
use shnn_ir::{
    Module,
    lif_neuron_v1, stdp_rule_v1, layer_fully_connected_v1,
    stimulus_poisson_v1, runtime_simulate_run_v1, parse_text,
};

#[test]
fn nir_compile_run_parity_roundtrip() {
    // Build a small deterministic program
    let mut m = Module::new();
    m.push(lif_neuron_v1(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
    m.push(stdp_rule_v1(0.01, 0.012, 20.0, 20.0, 0.0, 1.0));
    m.push(layer_fully_connected_v1(0, 0, 1, 1, 1.0, 1.0));
    m.push(stimulus_poisson_v1(0, 20.0, 10.0, 0.0, 100.0));
    // Use a fixed seed for determinism
    m.push(runtime_simulate_run_v1(0.1, 10.0, false, Some(1234)));

    // Print → parse round-trip
    let text = m.to_text();
    let m2 = parse_text(&text).expect("parse");

    // Compile + run both
    let r1 = compile_module(&m).expect("compile m").run().expect("run m");
    let r2 = compile_module(&m2).expect("compile m2").run().expect("run m2");

    // Parity checks
    assert_eq!(r1.steps_executed, r2.steps_executed, "steps_executed mismatch");
    assert_eq!(r1.spikes.len(), r2.spikes.len(), "spike count mismatch");
}