//! Round-trip tests including connectivity.synapse_connect and mixed modules.

use shnn_ir::{
    Module,
    lif_neuron_v1, stdp_rule_v1, layer_fully_connected_v1, synapse_connect_v1,
    stimulus_poisson_v1, runtime_simulate_run_v1,
    parse_text,
};

#[test]
fn roundtrip_synapse_connect_single() {
    let mut m = Module::new();
    // Basic LIF defaults
    m.push(lif_neuron_v1(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
    // Single synapse 0 -> 1
    m.push(synapse_connect_v1(0, 1, 0.5, 1.0));
    // Poisson stimulus on neuron 0
    m.push(stimulus_poisson_v1(0, 20.0, 10.0, 0.0, 50.0));
    // Simulation
    m.push(runtime_simulate_run_v1(0.1, 50.0, false, Some(1234)));

    let text1 = m.to_text();
    let parsed = parse_text(&text1).expect("parse");
    let text2 = parsed.to_text();
    assert_eq!(text1, text2, "Textual IR did not round-trip identically");
}

#[test]
fn roundtrip_mixed_connectivity() {
    let mut m = Module::new();

    // LIF + STDP defaults
    m.push(lif_neuron_v1(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
    m.push(stdp_rule_v1(0.01, 0.012, 20.0, 20.0, 0.0, 1.0));

    // Layer FC: 0..1 -> 2..3 plus a single synapse 1 -> 2
    m.push(layer_fully_connected_v1(0, 1, 2, 3, 1.0, 1.0));
    m.push(synapse_connect_v1(1, 2, 0.75, 0.5));

    // Stimulus to all inputs
    m.push(stimulus_poisson_v1(0, 10.0, 5.0, 0.0, 100.0));
    m.push(stimulus_poisson_v1(1, 15.0, 5.0, 0.0, 100.0));

    m.push(runtime_simulate_run_v1(0.1, 100.0, true, Some(42)));

    let text1 = m.to_text();
    let parsed = parse_text(&text1).expect("parse");
    let text2 = parsed.to_text();
    assert_eq!(text1, text2, "Mixed connectivity IR did not round-trip identically");
}