use assert_cmd::prelude::*;
use std::error::Error;
use tempfile::tempdir;
use assert_cmd::Command;
use serde::Deserialize;

#[derive(Deserialize)]
struct WeightRecord {
    pre: u32,
    post: u32,
    weight: f32,
}

fn run_cmd<I, S>(args: I) -> Result<(), Box<dyn Error>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let mut cmd = Command::cargo_bin("snn")?;
    cmd.args(args);
    cmd.assert().success();
    Ok(())
}

fn read_json_triples(path: &std::path::Path) -> Result<Vec<(u32, u32, f32)>, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut records: Vec<WeightRecord> = serde_json::from_str(&text)?;
    // stable sort
    records.sort_by_key(|r| (r.pre, r.post));
    Ok(records.into_iter().map(|r| (r.pre, r.post, r.weight)).collect())
}

fn read_bincode_triples(path: &std::path::Path) -> Result<Vec<(u32, u32, f32)>, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let mut records: Vec<WeightRecord> = bincode::deserialize(&bytes)?;
    records.sort_by_key(|r| (r.pre, r.post));
    Ok(records.into_iter().map(|r| (r.pre, r.post, r.weight)).collect())
}

#[test]
fn snapshot_roundtrip_graph_json() -> Result<(), Box<dyn Error>> {
    let tmp = tempdir()?;
    let out = tmp.path().join("weights_graph.json");
    let out_str = out.to_str().unwrap();

    run_cmd([
        "snapshot","export",
        "--backend","graph",
        "--inputs","4",
        "--hidden","4",
        "--outputs","2",
        "--format","json",
        "--out", out_str,
    ])?;
    assert!(out.exists(), "graph JSON snapshot should be created");

    run_cmd([
        "snapshot","import",
        "--backend","graph",
        "--inputs","4",
        "--hidden","4",
        "--outputs","2",
        "--format","json",
        "--input", out_str,
    ])?;

    Ok(())
}

#[test]
fn snapshot_roundtrip_matrix_bincode() -> Result<(), Box<dyn Error>> {
    let tmp = tempdir()?;
    let out = tmp.path().join("weights_matrix.bin");
    let out_str = out.to_str().unwrap();

    run_cmd([
        "snapshot","export",
        "--backend","matrix",
        "--size","16",
        "--format","bincode",
        "--out", out_str,
    ])?;
    assert!(out.exists(), "matrix bincode snapshot should be created");

    run_cmd([
        "snapshot","import",
        "--backend","matrix",
        "--size","16",
        "--format","bincode",
        "--input", out_str,
    ])?;

    Ok(())
}

#[test]
fn snapshot_roundtrip_sparse_json() -> Result<(), Box<dyn Error>> {
    let tmp = tempdir()?;
    let out = tmp.path().join("weights_sparse.json");
    let out_str = out.to_str().unwrap();

    run_cmd([
        "snapshot","export",
        "--backend","sparse",
        "--size","8",
        "--format","json",
        "--out", out_str,
    ])?;
    assert!(out.exists(), "sparse JSON snapshot should be created");

    run_cmd([
        "snapshot","import",
        "--backend","sparse",
        "--size","8",
        "--format","json",
        "--input", out_str,
    ])?;

    Ok(())
}

#[test]
fn snapshot_export_deterministic_graph_json() -> Result<(), Box<dyn Error>> {
    let tmp = tempdir()?;
    let a = tmp.path().join("a.json");
    let b = tmp.path().join("b.json");
    let a_str = a.to_str().unwrap();
    let b_str = b.to_str().unwrap();

    // First export
    run_cmd([
        "snapshot","export",
        "--backend","graph",
        "--inputs","4",
        "--hidden","4",
        "--outputs","2",
        "--format","json",
        "--out", a_str,
    ])?;
    // Second export (identical params)
    run_cmd([
        "snapshot","export",
        "--backend","graph",
        "--inputs","4",
        "--hidden","4",
        "--outputs","2",
        "--format","json",
        "--out", b_str,
    ])?;

    let mut t_a = read_json_triples(&a)?;
    let mut t_b = read_json_triples(&b)?;
    t_a.sort_by(|x, y| (x.0, x.1, x.2.to_bits()).cmp(&(y.0, y.1, y.2.to_bits())));
    t_b.sort_by(|x, y| (x.0, x.1, x.2.to_bits()).cmp(&(y.0, y.1, y.2.to_bits())));
    assert_eq!(t_a, t_b, "graph JSON triples (sorted by (pre,post,weight_bits)) should be identical for deterministic content");
    Ok(())
}

#[test]
fn snapshot_export_deterministic_matrix_bincode() -> Result<(), Box<dyn Error>> {
    let tmp = tempdir()?;
    let a = tmp.path().join("a.bin");
    let b = tmp.path().join("b.bin");
    let a_str = a.to_str().unwrap();
    let b_str = b.to_str().unwrap();

    // First export
    run_cmd([
        "snapshot","export",
        "--backend","matrix",
        "--size","16",
        "--format","bincode",
        "--out", a_str,
    ])?;
    // Second export (identical params)
    run_cmd([
        "snapshot","export",
        "--backend","matrix",
        "--size","16",
        "--format","bincode",
        "--out", b_str,
    ])?;

    let mut t_a = read_bincode_triples(&a)?;
    let mut t_b = read_bincode_triples(&b)?;
    t_a.sort_by(|x, y| (x.0, x.1, x.2.to_bits()).cmp(&(y.0, y.1, y.2.to_bits())));
    t_b.sort_by(|x, y| (x.0, x.1, x.2.to_bits()).cmp(&(y.0, y.1, y.2.to_bits())));
    assert_eq!(t_a, t_b, "matrix bincode triples (sorted by (pre,post,weight_bits)) should be identical for deterministic content");
    Ok(())
}