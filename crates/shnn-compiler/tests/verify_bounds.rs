//! Verifier semantic/bounds checks

use shnn_compiler::verify_module;
use shnn_ir::{
    Module,
    lif_neuron_v1, stdp_rule_v1, layer_fully_connected_v1,
    stimulus_poisson_v1, runtime_simulate_run_v1,
};

#[test]
fn lif_tau_m_zero_fails() {
    let mut m = Module::new();
    // tau_m_ms = 0.0 -> 0 ns
    m.push(lif_neuron_v1(0.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
    // Need a connectivity op to ensure network has neurons (not required for verify, but ok)
    m.push(layer_fully_connected_v1(0, 0, 0, 0, 1.0, 1.0));
    m.push(stimulus_poisson_v1(0, 20.0, 10.0, 0.0, 1.0));
    m.push(runtime_simulate_run_v1(0.1, 1.0, false, Some(1)));
    let err = verify_module(&m).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("tau_m") && msg.contains("> 0"), "unexpected error: {}", msg);
}

#[test]
fn stdp_bounds_fail() {
    let mut m = Module::new();
    // valid lif to get past neuron checks
    m.push(lif_neuron_v1(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
    // w_min > w_max should fail
    m.push(stdp_rule_v1(0.01, 0.012, 20.0, 20.0, 1.0, 0.0));
    m.push(layer_fully_connected_v1(0, 0, 0, 0, 1.0, 1.0));
    m.push(stimulus_poisson_v1(0, 20.0, 10.0, 0.0, 1.0));
    m.push(runtime_simulate_run_v1(0.1, 1.0, false, None));
    let err = verify_module(&m).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("w_min") && msg.contains("<="), "unexpected error: {}", msg);
}

#[test]
fn layer_invalid_range_fails() {
    let mut m = Module::new();
    m.push(lif_neuron_v1(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
    // in range start > end
    m.push(layer_fully_connected_v1(1, 0, 0, 0, 1.0, 1.0));
    m.push(stimulus_poisson_v1(0, 20.0, 10.0, 0.0, 1.0));
    m.push(runtime_simulate_run_v1(0.1, 1.0, false, None));
    let err = verify_module(&m).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("in") && msg.contains("start <= end"), "unexpected error: {}", msg);
}

#[test]
fn poisson_negative_rate_fails() {
    let mut m = Module::new();
    m.push(lif_neuron_v1(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
    m.push(layer_fully_connected_v1(0, 0, 0, 0, 1.0, 1.0));
    // negative rate
    m.push(stimulus_poisson_v1(0, -1.0, 10.0, 0.0, 1.0));
    m.push(runtime_simulate_run_v1(0.1, 1.0, false, None));
    let err = verify_module(&m).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("rate") && msg.contains(">= 0"), "unexpected error: {}", msg);
}

#[test]
fn simulate_dt_zero_fails() {
    let mut m = Module::new();
    m.push(lif_neuron_v1(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
    m.push(layer_fully_connected_v1(0, 0, 0, 0, 1.0, 1.0));
    m.push(stimulus_poisson_v1(0, 20.0, 10.0, 0.0, 1.0));
    // dt_ms = 0.0 -> 0 ns
    m.push(runtime_simulate_run_v1(0.0, 1.0, false, None));
    let err = verify_module(&m).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("dt") && msg.contains("> 0"), "unexpected error: {}", msg);
}

#[test]
fn verify_ok_minimal_module() {
    let mut m = Module::new();
    m.push(lif_neuron_v1(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
    m.push(stdp_rule_v1(0.01, 0.012, 20.0, 20.0, 0.0, 1.0));
    m.push(layer_fully_connected_v1(0, 0, 0, 0, 1.0, 1.0));
    m.push(stimulus_poisson_v1(0, 20.0, 10.0, 0.0, 1.0));
    m.push(runtime_simulate_run_v1(0.1, 1.0, false, Some(42)));
    verify_module(&m).expect("verify ok");
}