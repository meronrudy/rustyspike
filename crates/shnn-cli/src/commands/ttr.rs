//! Task-Aware Topology Reshaping (TTR) - minimal working implementation (JSON mask)

use clap::Args;
use serde::Deserialize;
use std::path::PathBuf;
use tracing::info;

use crate::error::{CliError, CliResult};

/// Task-Aware Topology Reshaping (TTR)
#[derive(Args, Debug)]
pub struct TtrCommand {
    /// Program file (TOML)
    #[arg(long)]
    pub program: PathBuf,
    /// Output mask path (JSON, placeholder)
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Debug, Deserialize)]
struct TtrProgram {
    #[serde(default)]
    program: ProgramSection,
    #[serde(default)]
    inputs: InputsSection,
    #[serde(default)]
    ops: Vec<Op>,
    #[serde(default)]
    output: OutputSection,
}

#[derive(Debug, Default, Deserialize)]
struct ProgramSection {
    #[serde(default)]
    name: String,
    #[serde(default)]
    version: String,
}

#[derive(Debug, Default, Deserialize)]
struct InputsSection {
    #[serde(default)]
    ranges: Vec<RangeSpec>,
}

#[derive(Debug, Deserialize)]
struct RangeSpec {
    start: u32,
    end: u32,
}

#[derive(Debug, Default, Deserialize)]
struct OutputSection {
    #[serde(default)]
    mask_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Op {
    #[serde(default)]
    op_type: String,
    #[serde(default)]
    params: toml::value::Table,
}

impl TtrCommand {
    pub async fn execute(self, _workspace: PathBuf, _config: Option<PathBuf>) -> CliResult<()> {
        // Parse program
        let text = std::fs::read_to_string(&self.program)?;
        let prog: TtrProgram = toml::from_str(&text)
            .map_err(|e| CliError::config(format!("bad TTR program: {}", e)))?;

        // Placeholder engine: just echoes input ranges into a JSON mask
        let mut neurons: Vec<u32> = Vec::new();
        for r in prog.inputs.ranges {
            for id in r.start..=r.end {
                neurons.push(id);
            }
        }
        neurons.sort_unstable();
        neurons.dedup();

        // For now, ignore ops/params and just emit the mask
        let json = serde_json::json!({
            "mask": {
                "neurons": neurons,
            },
            "meta": {
                "program": {
                    "name": prog.program.name,
                    "version": prog.program.version,
                },
                "source": self.program.display().to_string(),
            }
        });

        if let Some(parent) = self.output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.output, serde_json::to_string_pretty(&json)
            .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?)?;

        info!("Wrote JSON mask to {}", self.output.display());
        Ok(())
    }
}