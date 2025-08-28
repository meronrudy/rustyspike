//! SNN training command implementation

use clap::{Args, ValueEnum};
use std::path::PathBuf;
use tracing::{info, warn};

use crate::error::{CliError, CliResult};
use shnn_runtime::{
    neuron::LIFParams,
    plasticity::STDPParams,
};
use shnn_ir::{
    Module,
    lif_neuron_v1, stdp_rule_v1, layer_fully_connected_v1,
    stimulus_poisson_v1, runtime_simulate_run_v1,
};
use shnn_compiler::compile_module;

/// Train spiking neural networks
#[derive(Args, Debug)]
pub struct TrainCommand {
    /// Neuron model type
    #[arg(long, default_value = "lif")]
    pub neurons: NeuronType,
    
    /// Plasticity rule type
    #[arg(long, default_value = "stdp")]
    pub plasticity: PlasticityType,
    
    /// Number of simulation steps
    #[arg(long, default_value = "10000")]
    pub steps: u64,
    
    /// Simulation time step in microseconds
    #[arg(long, default_value = "100")]
    pub dt_us: u64,
    
    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,
    
    /// Output file for training results
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// Emit textual NIR to this path (will still run unless you omit steps/params)
    #[arg(long)]
    pub emit_nir: Option<PathBuf>,
    
    /// Network topology specification
    #[arg(long, default_value = "fully-connected")]
    pub topology: TopologyType,
    
    /// Number of input neurons
    #[arg(long, default_value = "10")]
    pub inputs: u32,
    
    /// Number of hidden neurons
    #[arg(long, default_value = "50")]
    pub hidden: u32,
    
    /// Number of output neurons
    #[arg(long, default_value = "5")]
    pub outputs: u32,
    
    /// LIF neuron parameters (key=value pairs)
    #[arg(long = "lif", value_parser = parse_key_val::<String, f32>)]
    pub lif_params: Vec<(String, f32)>,
    
    /// STDP plasticity parameters (key=value pairs)
    #[arg(long = "stdp", value_parser = parse_key_val::<String, f32>)]
    pub stdp_params: Vec<(String, f32)>,
    
    /// Input stimulus pattern
    #[arg(long, default_value = "poisson")]
    pub stimulus: StimulusType,
    
    /// Stimulus rate in Hz (for Poisson)
    #[arg(long, default_value = "20.0")]
    pub stimulus_rate: f32,
    
    /// Record membrane potentials
    #[arg(long)]
    pub record_potentials: bool,
    
    /// Record spike times
    #[arg(long, default_value = "true")]
    pub record_spikes: bool,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum NeuronType {
    /// Leaky Integrate-and-Fire neurons
    Lif,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum PlasticityType {
    /// Spike-Timing Dependent Plasticity
    Stdp,
    /// No plasticity (static weights)
    None,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum TopologyType {
    /// Fully connected layers
    FullyConnected,
    /// Random sparse connectivity
    Random,
    /// Custom topology from file
    Custom,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum StimulusType {
    /// Poisson spike trains
    Poisson,
    /// Custom spike patterns
    Custom,
}

impl TrainCommand {
    pub async fn execute(
        self,
        workspace: PathBuf,
        _config: Option<PathBuf>,
    ) -> CliResult<()> {
        info!("Starting SNN training with {} neurons and {} plasticity", 
              format!("{:?}", self.neurons).to_lowercase(),
              format!("{:?}", self.plasticity).to_lowercase());
        
        // Create LIF neuron parameters from CLI args
        let mut lif_params = LIFParams::default();
        for (key, value) in &self.lif_params {
            match key.as_str() {
                "tau_m" => lif_params.tau_m = *value,
                "t_refrac" => lif_params.t_refrac = *value,
                "v_rest" => lif_params.v_rest = *value,
                "v_reset" => lif_params.v_reset = *value,
                "v_thresh" => lif_params.v_thresh = *value,
                "r_m" => lif_params.r_m = *value,
                "c_m" => lif_params.c_m = *value,
                _ => {
                    warn!("Unknown LIF parameter: {}", key);
                }
            }
        }
        
        // Create STDP parameters from CLI args
        let mut stdp_params = STDPParams::default();
        for (key, value) in &self.stdp_params {
            match key.as_str() {
                "a_plus" => stdp_params.a_plus = *value,
                "a_minus" => stdp_params.a_minus = *value,
                "tau_plus" => stdp_params.tau_plus = *value,
                "tau_minus" => stdp_params.tau_minus = *value,
                "w_min" => stdp_params.w_min = *value,
                "w_max" => stdp_params.w_max = *value,
                _ => {
                    warn!("Unknown STDP parameter: {}", key);
                }
            }
        }
        
        info!("Building NIR module...");
        let mut module = Module::new();

        // lif.neuron@v1
        module.push(lif_neuron_v1(
            lif_params.tau_m,
            lif_params.v_rest,
            lif_params.v_reset,
            lif_params.v_thresh,
            lif_params.t_refrac,
            lif_params.r_m,
            lif_params.c_m,
        ));

        // plasticity.stdp@v1 (if enabled)
        match self.plasticity {
            PlasticityType::Stdp => {
                module.push(stdp_rule_v1(
                    stdp_params.a_plus,
                    stdp_params.a_minus,
                    stdp_params.tau_plus,
                    stdp_params.tau_minus,
                    stdp_params.w_min,
                    stdp_params.w_max,
                ));
            }
            PlasticityType::None => { /* no-op */ }
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
                        1.0, // delay ms
                    ));
                }
                // Hidden (inputs..inputs+hidden-1) -> Output (inputs+hidden..end)
                if self.hidden > 0 && self.outputs > 0 {
                    module.push(layer_fully_connected_v1(
                        self.inputs,
                        self.inputs + self.hidden - 1,
                        self.inputs + self.hidden,
                        self.inputs + self.hidden + self.outputs - 1,
                        1.0,
                        1.0, // delay ms
                    ));
                }
            }
            _ => {
                return Err(CliError::invalid_args("Only fully-connected topology supported in v0"));
            }
        }

        // Stimuli
        let dt_ms = (self.dt_us as f32) / 1000.0;
        let total_ms = dt_ms * (self.steps as f32);

        match self.stimulus {
            StimulusType::Poisson => {
                for i in 0..self.inputs {
                    module.push(stimulus_poisson_v1(
                        i,
                        self.stimulus_rate,
                        10.0, // 10nA
                        0.0,  // start ms
                        total_ms,
                    ));
                }
            }
            _ => {
                return Err(CliError::invalid_args("Only Poisson stimulus supported in v0"));
            }
        }

        // runtime.simulate.run@v1
        module.push(runtime_simulate_run_v1(
            dt_ms,
            total_ms,
            self.record_potentials,
            self.seed,
        ));

        // Optionally emit textual NIR
        if let Some(path) = &self.emit_nir {
            let text = module.to_text();
            if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
            std::fs::write(path, text)?;
            info!("Emitted NIR to {}", path.display());
        }

        // Compile and run
        info!("Lowering NIR and running simulation for {} steps...", self.steps);
        let program = compile_module(&module).map_err(|e| CliError::Generic(anyhow::anyhow!(e)))?;
        let result = program.run()?;
        
        info!("Simulation completed!");
        info!("Total spikes recorded: {}", result.spikes.len());
        
        if self.record_potentials {
            info!("Membrane potentials recorded: {}", result.potentials.len());
        }
        
        // Save results if output specified
        if let Some(output_path) = self.output {
            let output_path = workspace.join(output_path);
            info!("Saving results to: {}", output_path.display());
            
            // Create output directory if needed
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            
            // Convert results to JSON and save
            let spike_data: Vec<_> = result.export_spikes().into_iter().map(|(time_ns, neuron_id)| {
                serde_json::json!({
                    "neuron_id": neuron_id,
                    "time_ns": time_ns,
                    "time_ms": time_ns as f64 / 1_000_000.0,
                })
            }).collect();
            
            let results_json = serde_json::json!({
                "simulation": {
                    "steps": self.steps,
                    "dt_us": self.dt_us,
                    "total_time_ms": (self.dt_us * self.steps) as f64 / 1000.0,
                },
                "network": {
                    "inputs": self.inputs,
                    "hidden": self.hidden,
                    "outputs": self.outputs,
                    "topology": format!("{:?}", self.topology),
                },
                "parameters": {
                    "lif": {
                        "tau_m": lif_params.tau_m,
                        "v_rest": lif_params.v_rest,
                        "v_reset": lif_params.v_reset,
                        "v_thresh": lif_params.v_thresh,
                        "t_refrac": lif_params.t_refrac,
                        "r_m": lif_params.r_m,
                        "c_m": lif_params.c_m,
                    },
                    "stdp": {
                        "a_plus": stdp_params.a_plus,
                        "a_minus": stdp_params.a_minus,
                        "tau_plus": stdp_params.tau_plus,
                        "tau_minus": stdp_params.tau_minus,
                        "w_min": stdp_params.w_min,
                        "w_max": stdp_params.w_max,
                    },
                },
                "results": {
                    "spike_count": result.spikes.len(),
                    "spikes": spike_data,
                }
            });
            
            let json_string = serde_json::to_string_pretty(&results_json)
                .map_err(|e| CliError::Generic(anyhow::anyhow!("JSON serialization failed: {}", e)))?;
            std::fs::write(output_path, json_string)?;
        }
        
        Ok(())
    }
}

/// Parse a single key-value pair
fn parse_key_val<T, U>(s: &str) -> Result<(T, U), Box<dyn std::error::Error + Send + Sync + 'static>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
    U: std::str::FromStr,
    U::Err: std::error::Error + Send + Sync + 'static,
{
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{s}`"))?;
    Ok((s[..pos].parse()?, s[pos + 1..].parse()?))
}