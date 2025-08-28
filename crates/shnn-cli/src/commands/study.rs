#![allow(clippy::needless_collect)]
//! Parameter study and optimization command (minimal working runner)

use clap::Args;
use serde::Deserialize;
use std::path::PathBuf;
use tracing::{info, warn};

use crate::error::{CliError, CliResult};
use shnn_ir::parse_text;
use shnn_compiler::{verify_module, compile_with_passes};

/// Run parameter studies
#[derive(Args, Debug)]
pub struct StudyCommand {
    /// Study configuration file (.toml)
    pub config: PathBuf,

    /// Number of parallel jobs (unused in v0; sequential)
    #[arg(short, long, default_value = "1")]
    pub jobs: u32,
}

#[derive(Debug, Deserialize)]
struct StudyConfig {
    #[serde(default)]
    study: StudySection,
    #[serde(default)]
    runs: Vec<StudyRun>,
}

#[derive(Debug, Default, Deserialize)]
struct StudySection {
    #[serde(default)]
    name: String,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    out_dir: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StudyRun {
    nir: String,
    #[serde(default)]
    repeats: u32,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    record_potentials: Option<bool>,
}

impl StudyCommand {
    pub async fn execute(self, _workspace: PathBuf, _config: Option<PathBuf>) -> CliResult<()> {
        let text = std::fs::read_to_string(&self.config)?;
        let cfg: StudyConfig = toml::from_str(&text)
            .map_err(|e| CliError::config(format!("Invalid study config: {}", e)))?;

        let base_out = cfg.study.out_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("crates/shnn-cli/test_workspace/results/studies"));
        std::fs::create_dir_all(&base_out)?;

        info!("Study '{}' with {} runs", cfg.study.name, cfg.runs.len());
        let mut summary = Vec::new();

        for (i, run) in cfg.runs.iter().enumerate() {
            let repeats = run.repeats.max(1);
            info!("Run {}: {} ({} repeats)", i + 1, run.nir, repeats);

            let nir_path = PathBuf::from(&run.nir);
            let nir_txt = std::fs::read_to_string(&nir_path)?;
            let module = parse_text(&nir_txt)
                .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;
            verify_module(&module).map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;

            for r in 0..repeats {
                let mut program = compile_with_passes(&module)
                    .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;
                let result = program.run()?;

                let out_file = base_out.join(format!("run{}_rep{}.json", i + 1, r + 1));
                let json = serde_json::json!({
                    "study": cfg.study.name,
                    "run_index": i + 1,
                    "repeat_index": r + 1,
                    "steps_executed": result.steps_executed,
                    "spike_count": result.spikes.len(),
                });
                std::fs::write(&out_file, serde_json::to_string_pretty(&json)
                    .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?)?;
                summary.push(json);
            }
        }

        let summary_file = base_out.join("summary.json");
        std::fs::write(&summary_file, serde_json::to_string_pretty(&serde_json::json!({ "runs": summary }))
            .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?)?;
        warn!("Study summary: {}", summary_file.display());
        Ok(())
    }
}