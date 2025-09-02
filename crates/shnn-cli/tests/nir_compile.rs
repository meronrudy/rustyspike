use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use tempfile::tempdir;
use assert_cmd::Command;

fn compile_nir_to(path: &PathBuf) -> Result<(), Box<dyn Error>> {
    let mut cmd = Command::cargo_bin("snn")?;
    let path_str = path.to_str().expect("temp path to UTF-8");
    cmd.args([
        "nir", "compile",
        "-o", path_str,
        "--neurons", "lif",
        "--plasticity", "stdp",
        "--inputs", "4",
        "--hidden", "4",
        "--outputs", "2",
        "--topology", "fully-connected",
        "--steps", "100",
        "--dt-us", "100",
        "--stimulus", "poisson",
        "--stimulus-rate", "5.0",
    ]);
    cmd.assert().success();
    Ok(())
}

#[test]
fn nir_compile_and_verify() -> Result<(), Box<dyn Error>> {
    let tmp = tempdir()?;
    let model = tmp.path().join("model.nirt");

    compile_nir_to(&model)?;

    assert!(model.exists(), "compiled NIR file should exist");

    let mut verify = Command::cargo_bin("snn")?;
    verify.args(["nir", "verify", &model.to_string_lossy()]);
    verify.assert().success();

    Ok(())
}

#[test]
fn nir_run_json_and_snapshot_smoke() -> Result<(), Box<dyn Error>> {
    let tmp = tempdir()?;
    let model = tmp.path().join("m.nirt");

    // Compile NIR first
    compile_nir_to(&model)?;
    assert!(model.exists(), "compiled NIR file should exist");

    // Run NIR and emit JSON spikes
    let out_json = tmp.path().join("spikes.json");
    let mut run_json = Command::cargo_bin("snn")?;
    let model_str = model.to_str().expect("temp path to UTF-8");
    let out_json_str = out_json.to_str().expect("temp path to UTF-8");
    run_json.args([
        "nir", "run",
        model_str,
        "--output", out_json_str,
        "--spikes-format", "json",
    ]);
    run_json.assert().success();
    assert!(out_json.exists(), "JSON spikes output should be created");

    // Snapshot export/import smoke (graph backend)
    let weights = tmp.path().join("weights.json");
    let mut export_cmd = Command::cargo_bin("snn")?;
    let weights_str = weights.to_str().expect("temp path to UTF-8");
    export_cmd.args([
        "snapshot", "export",
        "--backend", "graph",
        "--inputs", "4",
        "--hidden", "4",
        "--outputs", "2",
        "--weight", "1.0",
        "--format", "json",
        "--out", weights_str,
    ]);
    export_cmd.assert().success();
    assert!(weights.exists(), "weights snapshot should be created");

    let mut import_cmd = Command::cargo_bin("snn")?;
    let weights_str = weights.to_str().expect("temp path to UTF-8");
    import_cmd.args([
        "snapshot", "import",
        "--backend", "graph",
        "--inputs", "4",
        "--hidden", "4",
        "--outputs", "2",
        "--format", "json",
        "--input", weights_str,
    ]);
    import_cmd.assert().success();

    Ok(())
}