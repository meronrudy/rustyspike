//! Workspace initialization command

use clap::Args;
use std::path::PathBuf;
use tracing::info;

use crate::error::CliResult;

/// Initialize a new hSNN workspace
#[derive(Args, Debug)]
pub struct InitCommand {
    /// Workspace name
    pub name: String,
    
    /// Create with example configurations
    #[arg(long)]
    pub examples: bool,
    
    /// Initialize Git repository
    #[arg(long, default_value = "true")]
    pub git: bool,
}

impl InitCommand {
    pub async fn execute(
        self,
        workspace: PathBuf,
        _config: Option<PathBuf>,
    ) -> CliResult<()> {
        info!("Initializing hSNN workspace: {}", self.name);
        
        let workspace_dir = workspace.join(&self.name);
        std::fs::create_dir_all(&workspace_dir)?;
        
        // Create basic directory structure
        std::fs::create_dir_all(workspace_dir.join("experiments"))?;
        std::fs::create_dir_all(workspace_dir.join("data"))?;
        std::fs::create_dir_all(workspace_dir.join("results"))?;
        std::fs::create_dir_all(workspace_dir.join("configs"))?;
        
        // Create default config file
        let config_content = r#"# hSNN Workspace Configuration
[workspace]
name = "{name}"
version = "0.1.0"

[defaults]
neuron_type = "lif"
plasticity_type = "stdp"
dt_us = 100
seed = 42

[lif]
tau_m = 20.0      # membrane time constant (ms)
tau_ref = 2.0     # refractory period (ms) 
v_rest = -70.0    # resting potential (mV)
v_reset = -80.0   # reset potential (mV)
v_thresh = -50.0  # spike threshold (mV)
r_m = 10.0        # membrane resistance (MΩ)

[stdp]
a_plus = 0.01     # potentiation amplitude
a_minus = 0.012   # depression amplitude
tau_plus = 20.0   # potentiation time constant (ms)
tau_minus = 20.0  # depression time constant (ms)
w_min = 0.0       # minimum weight
w_max = 1.0       # maximum weight
"#;
        
        let config_content = config_content.replace("{name}", &self.name);
        std::fs::write(workspace_dir.join("hSNN.toml"), config_content)?;
        
        if self.examples {
            // Create example experiment
            let example_config = r#"# Example SNN Training Experiment
[experiment]
name = "basic_classification"
description = "Basic 3-layer classification network"

[network]
inputs = 10
hidden = 50  
outputs = 5
topology = "fully-connected"

[training]
steps = 20000
stimulus = "poisson"
stimulus_rate = 20.0
record_spikes = true
record_potentials = false
"#;
            std::fs::write(workspace_dir.join("experiments").join("example.toml"), example_config)?;
        }
        
        if self.git {
            // Initialize git repository
            std::process::Command::new("git")
                .arg("init")
                .current_dir(&workspace_dir)
                .output()?;
                
            // Create .gitignore
            let gitignore_content = r#"# Generated files
/results/
/data/*.bin
/data/*.h5

# Temporary files
*.tmp
*.log

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db
"#;
            std::fs::write(workspace_dir.join(".gitignore"), gitignore_content)?;
        }
        
        info!("Workspace initialized at: {}", workspace_dir.display());
        info!("Run 'cd {}' to enter the workspace", self.name);
        
        Ok(())
    }
}