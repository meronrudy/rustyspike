use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::error::Error;
use tempfile::tempdir;
use assert_cmd::Command;

#[test]
fn nir_compile_missing_output_flag_fails() -> Result<(), Box<dyn Error>> {
    // No -o/--output provided, clap should fail fast
    let mut cmd = Command::cargo_bin("snn")?;
    cmd.args([
        "nir", "compile",
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
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--output").or(predicate::str::contains("-o")));
    Ok(())
}

#[test]
fn nir_compile_unsupported_topology_random_fails() -> Result<(), Box<dyn Error>> {
    // Compile with a topology we don't yet support in compile path: random
    let tmp = tempdir()?;
    let out = tmp.path().join("model.nirt");
    let out_str = out.to_str().expect("utf8");
    let mut cmd = Command::cargo_bin("snn")?;
    cmd.args([
        "nir", "compile",
        "-o", out_str,
        "--neurons", "lif",
        "--plasticity", "stdp",
        "--inputs", "4",
        "--hidden", "4",
        "--outputs", "2",
        "--topology", "random",
        "--steps", "100",
        "--dt-us", "100",
        "--stimulus", "poisson",
        "--stimulus-rate", "5.0",
    ]);
    let assert = cmd.assert().failure();
    let out = String::from_utf8_lossy(&assert.get_output().stdout);
    let err = String::from_utf8_lossy(&assert.get_output().stderr);
    assert!(
        out.contains("Only fully-connected topology supported")
            || err.contains("Only fully-connected topology supported"),
        "Expected message on stdout or stderr.\nstdout={}\nstderr={}",
        out,
        err
    );
    Ok(())
}

#[test]
fn nir_run_invalid_spikes_format_flag_fails_fast() -> Result<(), Box<dyn Error>> {
    // Invalid enum value for --spikes-format should be caught by clap ValueEnum before I/O
    // Provide a dummy path; parsing fails prior to file checks.
    let mut cmd = Command::cargo_bin("snn")?;
    cmd.args([
        "nir", "run",
        "does-not-matter.nirt",
        "--output", "out.json",
        "--spikes-format", "csv",
    ]);
    cmd.assert()
        .failure()
        // clap typically prints possible values in the error message; ensure we see our valid ones
        .stderr(predicate::str::contains("json").and(predicate::str::contains("vevt")));
    Ok(())
}