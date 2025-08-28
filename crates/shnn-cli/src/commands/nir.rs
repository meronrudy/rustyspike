//! NIR-focused commands: compile (build textual NIR), run (from NIR - stub),
//! and op listing (dialects/ops/versions).

use clap::{Args, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracing::info;
use std::fs;

use crate::error::{CliError, CliResult};

use shnn_ir::{
    Module, parse_text,
    lif_neuron_v1, stdp_rule_v1, layer_fully_connected_v1,
    stimulus_poisson_v1, runtime_simulate_run_v1,
};

use shnn_compiler::{compile_with_passes, verify_module, list_ops};

/// NIR-related commands
#[derive(Args, Debug)]
pub struct NirCommand {
    #[command(subcommand)]
    pub sub: NirSubcommand,
}

#[derive(Subcommand, Debug)]
pub enum NirSubcommand {
    /// Build textual NIR from CLI parameters (no execution)
    Compile(NirCompile),
    /// Run from textual NIR (stub until parser lands)
    Run(NirRun),
    /// List available ops and versions
    OpList(NirOpList),
    /// Verify textual NIR file
    Verify(NirVerify),
}

/// Compile CLI params to textual NIR (.nirt), without executing
#[derive(Args, Debug)]
pub struct NirCompile {
    /// Output NIR file path (.nirt)
    #[arg(short, long)]
    pub output: PathBuf,

    /// Neuron model
    #[arg(long, default_value = "lif")]
    pub neurons: NeuronType,

    /// Plasticity
    #[arg(long, default_value = "stdp")]
    pub plasticity: PlasticityType,

    /// Inputs/hidden/outputs
    #[arg(long, default_value = "10")]
    pub inputs: u32,
    #[arg(long, default_value = "50")]
    pub hidden: u32,
    #[arg(long, default_value = "5")]
    pub outputs: u32,

    /// Topology
    #[arg(long, default_value = "fully-connected")]
    pub topology: TopologyType,

    /// Simulation control
    #[arg(long, default_value = "10000")]
    pub steps: u64,
    #[arg(long, default_value = "100")]
    pub dt_us: u64,

    /// Stimulus control
    #[arg(long, default_value = "poisson")]
    pub stimulus: StimulusType,
    #[arg(long, default_value = "20.0")]
    pub stimulus_rate: f32,

    /// Record potentials
    #[arg(long)]
    pub record_potentials: bool,

    /// Random seed
    #[arg(long)]
    pub seed: Option<u64>,
}

/// Run from textual NIR
#[derive(Args, Debug)]
pub struct NirRun {
    /// Input textual NIR file (.nirt)
    pub input: PathBuf,

    /// Output JSON file (optional)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// List available ops and versions
#[derive(Args, Debug)]
pub struct NirOpList {
    /// Show detailed attribute hints
    #[arg(long)]
    pub detailed: bool,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum NeuronType {
    Lif,
}
#[derive(ValueEnum, Clone, Debug)]
pub enum PlasticityType {
    Stdp,
    None,
}
#[derive(ValueEnum, Clone, Debug)]
pub enum TopologyType {
    FullyConnected,
    Random,
    Custom,
}
#[derive(ValueEnum, Clone, Debug)]
pub enum StimulusType {
    Poisson,
    Custom,
}

impl NirCommand {
    pub async fn execute(self) -> CliResult<()> {
        match self.sub {
            NirSubcommand::Compile(cmd) => cmd.execute().await,
            NirSubcommand::Run(cmd) => cmd.execute().await,
            NirSubcommand::OpList(cmd) => cmd.execute().await,
            NirSubcommand::Verify(cmd) => cmd.execute().await,
        }
    }
}

impl NirCompile {
    pub async fn execute(self) -> CliResult<()> {
        // Build NIR module from CLI params (mirrors train->NIR logic)
        let mut module = Module::new();

        // lif.neuron@v1 defaults (basic LIF; parameters can be lifted later via config)
        // Conservative defaults copied from runtime defaults
        let lif_tau_m = 20.0;
        let lif_v_rest = -70.0;
        let lif_v_reset = -70.0;
        let lif_v_thresh = -50.0;
        let lif_t_refrac = 2.0;
        let lif_r_m = 10.0;
        let lif_c_m = 1.0;

        module.push(lif_neuron_v1(
            lif_tau_m,
            lif_v_rest,
            lif_v_reset,
            lif_v_thresh,
            lif_t_refrac,
            lif_r_m,
            lif_c_m,
        ));

        // plasticity.stdp@v1 if requested
        match self.plasticity {
            PlasticityType::Stdp => {
                // Defaults from runtime STDP defaults
                let a_plus = 0.01;
                let a_minus = 0.012;
                let tau_plus = 20.0;
                let tau_minus = 20.0;
                let w_min = 0.0;
                let w_max = 1.0;

                module.push(stdp_rule_v1(
                    a_plus, a_minus, tau_plus, tau_minus, w_min, w_max,
                ));
            }
            PlasticityType::None => {}
        }

        // connectivity.layer_fully_connected@v1
        match self.topology {
            TopologyType::FullyConnected => {
                // Input (0..inputs-1) -> Hidden (inputs..inputs+hidden-1)
                if self.inputs > 0 && self.hidden > 0 {
                    module.push(layer_fully_connected_v1(
                        0,
                        self.inputs.saturating_sub(1),
                        self.inputs,
                        self.inputs + self.hidden - 1,
                        1.0,
                        1.0,
                    ));
                }
                // Hidden -> Output
                if self.hidden > 0 && self.outputs > 0 {
                    module.push(layer_fully_connected_v1(
                        self.inputs,
                        self.inputs + self.hidden - 1,
                        self.inputs + self.hidden,
                        self.inputs + self.hidden + self.outputs - 1,
                        1.0,
                        1.0,
                    ));
                }
            }
            _ => return Err(CliError::invalid_args("Only fully-connected topology supported for NIR compile v0")),
        }

        // Stimulus
        let dt_ms = (self.dt_us as f32) / 1000.0;
        let total_ms = dt_ms * (self.steps as f32);

        match self.stimulus {
            StimulusType::Poisson => {
                for i in 0..self.inputs {
                    module.push(stimulus_poisson_v1(
                        i,
                        self.stimulus_rate,
                        10.0,
                        0.0,
                        total_ms,
                    ));
                }
            }
            _ => return Err(CliError::invalid_args("Only Poisson stimulus supported for NIR compile v0")),
        }

        // runtime.simulate.run@v1
        module.push(runtime_simulate_run_v1(
            dt_ms,
            total_ms,
            self.record_potentials,
            self.seed,
        ));

        // Emit textual NIR
        let text = module.to_text();
        if let Some(parent) = self.output.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&self.output, text)?;
        info!("Emitted NIR to {}", self.output.display());
        Ok(())
    }
}

impl NirRun {
    pub async fn execute(self) -> CliResult<()> {
        // Read textual NIR, parse, verify, compile, and run
        let text = fs::read_to_string(&self.input)?;
        let module = parse_text(&text)
            .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;

        verify_module(&module).map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;

        info!("Compiling NIR from {}", self.input.display());
        let program = compile_with_passes(&module)
            .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;

        info!("Running simulation...");
        let result = program.run()?;
        info!("Simulation completed: {} spikes", result.spikes.len());

        // Optionally write results JSON (spike export only for now)
        if let Some(path) = &self.output {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let spike_data: Vec<_> = result.export_spikes().into_iter().map(|(time_ns, neuron_id)| {
                serde_json::json!({
                    "neuron_id": neuron_id,
                    "time_ns": time_ns,
                    "time_ms": time_ns as f64 / 1_000_000.0,
                })
            }).collect();

            let json = serde_json::json!({
                "results": {
                    "spike_count": result.spikes.len(),
                    "spikes": spike_data
                }
            });
            std::fs::write(path, serde_json::to_string_pretty(&json)
                .map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?)?;
            info!("Wrote results to {}", path.display());
        }

        Ok(())
    }
}

impl NirOpList {
    pub async fn execute(self) -> CliResult<()> {
        let ops = list_ops();
        
        // Group ops by dialect
        let mut dialects: std::collections::BTreeMap<&str, Vec<_>> = std::collections::BTreeMap::new();
        for op in ops {
            dialects.entry(op.dialect).or_default().push(op);
        }

        println!("Registered dialects and ops:");
        for (dialect, ops) in dialects {
            println!("- {}:", dialect);
            for op in ops {
                if self.detailed {
                    // Detailed mode: show attributes with types and docs
                    println!("  - {}@v{} {{", op.name, op.version);
                    for attr in op.attrs {
                        let required = if attr.required { "" } else { "?" };
                        println!("    {}{}: {} // {}", attr.name, required, attr.kind.name(), attr.doc);
                    }
                    println!("  }}");
                } else {
                    // Compact mode: just show signature
                    let attrs: Vec<String> = op.attrs.iter().map(|a| {
                        let required = if a.required { "" } else { "?" };
                        format!("{}{}: {}", a.name, required, a.kind.name())
                    }).collect();
                    println!("  - {}@v{} {{ {} }}", op.name, op.version, attrs.join(", "));
                }
            }
        }

        if self.detailed {
            println!("\nUnit reference:");
            println!("- DurationNs, TimeNs: nanoseconds");
            println!("- VoltageMv: millivolts, ResistanceMohm: megaohms, CapacitanceNf: nanofarads");
            println!("- CurrentNa: nanoamps, RateHz: hertz");
            println!("- Weight: dimensionless synaptic weight");
            println!("- RangeU32: inclusive start..end range");
            println!("- NeuronRef: reference to neuron by ID (%nX format in textual NIR)");
        }

        Ok(())
    }
}

// --- Added: NirVerify command (parse + verify textual NIR) ---

/// Verify textual NIR file
#[derive(clap::Args, Debug)]
pub struct NirVerify {
    /// Input textual NIR file (.nirt)
    pub input: std::path::PathBuf,
}

impl NirVerify {
    pub async fn execute(self) -> crate::error::CliResult<()> {
        let text = std::fs::read_to_string(&self.input)?;
        let module = shnn_ir::parse_text(&text)
            .map_err(|e| crate::error::CliError::Generic(anyhow::anyhow!(e)))?;
        shnn_compiler::verify_module(&module)
            .map_err(|e| crate::error::CliError::Generic(anyhow::anyhow!(e)))?;
        println!("Verification OK: {}", self.input.display());
        Ok(())
    }
}
