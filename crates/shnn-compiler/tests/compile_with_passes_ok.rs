//! Ensure compile_with_passes runs the no-op pass pipeline and produces a runnable program.

use shnn_compiler::{compile_with_passes};
use shnn_ir::{
    Module,
    lif_neuron_v1, stdp_rule_v1, layer_fully_connected_v1,
    stimulus_poisson_v1, runtime_simulate_run_v1,
};

#[test]
fn compile_with_passes_runs_program() {
    let mut m = Module::new();
    m.push(lif_neuron_v1(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
    m.push(stdp_rule_v1(0.01, 0.012, 20.0, 20.0, 0.0, 1.0));
    m.push(layer_fully_connected_v1(0, 0, 1, 1, 1.0, 1.0));
    m.push(stimulus_poisson_v1(0, 20.0, 10.0, 0.0, 100.0));
    m.push(runtime_simulate_run_v1(0.1, 10.0, false, Some(42)));

    let res = compile_with_passes(&m).expect("compile with passes").run().expect("run");
    assert!(res.steps_executed > 0);
}