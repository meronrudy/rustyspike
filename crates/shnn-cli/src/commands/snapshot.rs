//! Snapshot export/import commands for synaptic weights across backends
//!
//! Provides a thin CLI around shnn-core's WeightSnapshotConnectivity and PlasticConn
//! to export/import weight triples [(pre, post, weight)] in JSON or bincode.
//!
//! Example:
//!   snn snapshot export --backend graph --inputs 10 --hidden 50 --outputs 5 --weight 1.0 --format json --out weights.json
//!   snn snapshot import --backend graph --inputs 10 --hidden 50 --outputs 5 --format json --input weights.json
//!
//! Notes:
//! - This constructs deterministic topologies locally (no persistence yet).
//! - Graph: fully-connected Input->Hidden and Hidden->Output with uniform weight.
//! - Matrix: fully connected (NxN, no self-connections) with uniform weight.
//! - Sparse: ring connectivity (i -> (i+1) mod N) with uniform weight.

use clap::{Args, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::info;

use crate::error::{CliError, CliResult};

use shnn_core::spike::NeuronId;
use shnn_core::connectivity::WeightSnapshotConnectivity;

use shnn_core::connectivity::plastic_enum::PlasticConn;
use shnn_core::connectivity::graph::{GraphEdge, GraphNetwork};
use shnn_core::connectivity::matrix::MatrixNetwork;
use shnn_core::connectivity::sparse::SparseMatrixNetwork;

#[derive(Args, Debug)]
pub struct SnapshotCommand {
    #[command(subcommand)]
    pub sub: SnapshotSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum SnapshotSubcommand {
    /// Export a weight snapshot from a deterministic locally constructed connectivity
    Export(SnapshotExport),
    /// Import weight updates and apply to a deterministic locally constructed connectivity
    Import(SnapshotImport),
}

#[derive(ValueEnum, Clone, Debug)]
pub enum SnapshotBackend {
    Graph,
    Matrix,
    Sparse,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum SnapshotFormat {
    Json,
    Bincode,
}

#[derive(Args, Debug)]
pub struct SnapshotExport {
    /// Connectivity backend
    #[arg(long, value_enum, default_value = "graph")]
    pub backend: SnapshotBackend,

    /// Output format
    #[arg(long, value_enum, default_value = "json")]
    pub format: SnapshotFormat,

    /// Output snapshot file path
    #[arg(short, long)]
    pub out: PathBuf,

    /// Initial uniform weight for constructed connectivity
    #[arg(long, default_value = "1.0")]
    pub weight: f32,

    /// Graph: number of input neurons
    #[arg(long)]
    pub inputs: Option<u32>,
    /// Graph: number of hidden neurons
    #[arg(long)]
    pub hidden: Option<u32>,
    /// Graph: number of output neurons
    #[arg(long)]
    pub outputs: Option<u32>,

    /// Matrix/Sparse: total neurons
    #[arg(long)]
    pub size: Option<u32>,
}

#[derive(Args, Debug)]
pub struct SnapshotImport {
    /// Connectivity backend
    #[arg(long, value_enum, default_value = "graph")]
    pub backend: SnapshotBackend,

    /// Input format
    #[arg(long, value_enum, default_value = "json")]
    pub format: SnapshotFormat,

    /// Input snapshot file path
    #[arg(long)]
    pub input: PathBuf,

    /// Initial uniform weight for constructed connectivity (used before applying updates)
    #[arg(long, default_value = "1.0")]
    pub weight: f32,

    /// Graph: number of input neurons
    #[arg(long)]
    pub inputs: Option<u32>,
    /// Graph: number of hidden neurons
    #[arg(long)]
    pub hidden: Option<u32>,
    /// Graph: number of output neurons
    #[arg(long)]
    pub outputs: Option<u32>,

    /// Matrix/Sparse: total neurons
    #[arg(long)]
    pub size: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WeightRecord {
    pre: u32,
    post: u32,
    weight: f32,
}

impl SnapshotCommand {
    pub async fn execute(self, _workspace: PathBuf, _config: Option<PathBuf>) -> CliResult<()> {
        match self.sub {
            SnapshotSubcommand::Export(cmd) => cmd.execute().await,
            SnapshotSubcommand::Import(cmd) => cmd.execute().await,
        }
    }
}

impl SnapshotExport {
    pub async fn execute(self) -> CliResult<()> {
        // Validate arguments based on backend
        match self.backend {
            SnapshotBackend::Graph => {
                ensure_all(&[self.inputs.is_some(), self.hidden.is_some(), self.outputs.is_some()])?
            }
            SnapshotBackend::Matrix | SnapshotBackend::Sparse => {
                if self.size.is_none() {
                    return Err(CliError::invalid_args(
                        "Missing required --size for matrix/sparse backend",
                    ));
                }
            }
        }

        // Build connectivity
        let conn = build_connectivity(
            &self.backend,
            self.inputs,
            self.hidden,
            self.outputs,
            self.size,
            self.weight,
        )?;

        // Snapshot weights
        let triples = <PlasticConn as WeightSnapshotConnectivity<NeuronId>>::snapshot_weights(&conn);
        let records: Vec<WeightRecord> = triples
            .into_iter()
            .map(|(pre, post, w)| WeightRecord {
                pre: pre.raw(),
                post: post.raw(),
                weight: w,
            })
            .collect();

        // Write to file
        if let Some(parent) = self.out.parent() {
            std::fs::create_dir_all(parent)?;
        }
        match self.format {
            SnapshotFormat::Json => {
                let text = serde_json::to_string_pretty(&records)
                    .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;
                std::fs::write(&self.out, text)?;
            }
            SnapshotFormat::Bincode => {
                let bytes = bincode::serialize(&records)
                    .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;
                std::fs::write(&self.out, bytes)?;
            }
        }

        info!(
            "Exported {} weight records to {}",
            records.len(),
            self.out.display()
        );
        Ok(())
    }
}

impl SnapshotImport {
    pub async fn execute(self) -> CliResult<()> {
        // Validate arguments based on backend
        match self.backend {
            SnapshotBackend::Graph => {
                ensure_all(&[self.inputs.is_some(), self.hidden.is_some(), self.outputs.is_some()])?
            }
            SnapshotBackend::Matrix | SnapshotBackend::Sparse => {
                if self.size.is_none() {
                    return Err(CliError::invalid_args(
                        "Missing required --size for matrix/sparse backend",
                    ));
                }
            }
        }

        // Read file
        let records: Vec<WeightRecord> = match self.format {
            SnapshotFormat::Json => {
                let text = std::fs::read_to_string(&self.input)?;
                serde_json::from_str(&text)
                    .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?
            }
            SnapshotFormat::Bincode => {
                let bytes = std::fs::read(&self.input)?;
                bincode::deserialize(&bytes)
                    .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?
            }
        };

        // Build connectivity (deterministic)
        let mut conn = build_connectivity(
            &self.backend,
            self.inputs,
            self.hidden,
            self.outputs,
            self.size,
            self.weight,
        )?;

        // Apply updates
        let updates: Vec<(NeuronId, NeuronId, f32)> = records
            .iter()
            .map(|r| (NeuronId::new(r.pre), NeuronId::new(r.post), r.weight))
            .collect();

        let applied =
            <PlasticConn as WeightSnapshotConnectivity<NeuronId>>::apply_weight_updates(
                &mut conn,
                &updates,
            )
            .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;

        info!(
            "Applied {}/{} weight updates from {}",
            applied,
            updates.len(),
            self.input.display()
        );
        Ok(())
    }
}

// Build a PlasticConn connectivity deterministically based on backend and params.
fn build_connectivity(
    backend: &SnapshotBackend,
    inputs: Option<u32>,
    hidden: Option<u32>,
    outputs: Option<u32>,
    size: Option<u32>,
    weight: f32,
) -> CliResult<PlasticConn> {
    match backend {
        SnapshotBackend::Graph => {
            let (inputs, hidden, outputs) = (
                inputs.ok_or_else(|| CliError::invalid_args("missing --inputs"))?,
                hidden.ok_or_else(|| CliError::invalid_args("missing --hidden"))?,
                outputs.ok_or_else(|| CliError::invalid_args("missing --outputs"))?,
            );
            let input_size = inputs as usize;
            let hidden_size = hidden as usize;
            let output_size = outputs as usize;

            let mut g = GraphNetwork::new();

            // Input -> Hidden
            for i in 0..input_size {
                for j in input_size..(input_size + hidden_size) {
                    let edge = GraphEdge::new(NeuronId::new(i as u32), NeuronId::new(j as u32), weight);
                    g.add_edge(edge)
                        .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;
                }
            }
            // Hidden -> Output
            let total_neurons = input_size + hidden_size + output_size;
            for i in input_size..(input_size + hidden_size) {
                for j in (input_size + hidden_size)..total_neurons {
                    let edge = GraphEdge::new(NeuronId::new(i as u32), NeuronId::new(j as u32), weight);
                    g.add_edge(edge)
                        .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;
                }
            }

            Ok(PlasticConn::from_graph(g))
        }
        SnapshotBackend::Matrix => {
            let n = size.ok_or_else(|| CliError::invalid_args("missing --size"))? as usize;
            let neurons: Vec<NeuronId> = (0..n).map(|i| NeuronId::new(i as u32)).collect();
            let m = MatrixNetwork::fully_connected(&neurons, weight)
                .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;
            Ok(PlasticConn::from_matrix(m))
        }
        SnapshotBackend::Sparse => {
            let n = size.ok_or_else(|| CliError::invalid_args("missing --size"))? as usize;
            let mut s = SparseMatrixNetwork::new(n);
            // Ensure neurons are registered
            for i in 0..n {
                let id = NeuronId::new(i as u32);
                if s.get_neuron_index(id).is_none() {
                    s.add_neuron(id)
                        .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;
                }
            }
            // Deterministic ring connectivity: i -> (i+1) mod n
            if n > 1 {
                for i in 0..n {
                    let pre = NeuronId::new(i as u32);
                    let post = NeuronId::new(((i + 1) % n) as u32);
                    s.set_weight(pre, post, weight)
                        .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;
                }
            }
            Ok(PlasticConn::from_sparse(s))
        }
    }
}

fn ensure_all(flags: &[bool]) -> CliResult<()> {
    if flags.iter().all(|b| *b) {
        Ok(())
    } else {
        Err(CliError::invalid_args(
            "Missing required arguments for selected backend",
        ))
    }
}