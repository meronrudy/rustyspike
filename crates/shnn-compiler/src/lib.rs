#![doc = "Neuromorphic IR (NIR) compiler — verification, pass pipeline, and lowering to the runtime engine.\n\nPublic responsibilities:\n- Op Registry and schema introspection (list_ops) for dialects/ops/versions and attributes\n- Verification (verify_module): presence, type/unit validation, and semantic bounds\n- Pass pipeline (compile_with_passes): verify → canonicalize/upgrade → lower\n- Lowering (compile_module internal): build network + engine from NIR\n\nKey concepts:\n- Op Registry: Static OpSpec/AttributeSpec array with AttrKind describing attribute kinds/units\n- Verification: Ensures correctness (e.g., lif tau_m > 0; stdp w_min ≤ w_max; valid ranges; dt/duration > 0)\n- Passes: \n  * Canonicalize: expand composite connectivity (e.g., layer_fully_connected → synapse_connect)\n  * UpgradeVersions: scaffold to migrate older op versions to current ones with defaulted attrs\n- Lowering: Produces a runnable SimulationEngine by configuring NetworkBuilder, stimuli, and SimulationParams\n\nIntegration points:\n- shnn-ir: Provides Module/Operation and textual printer/parser\n- shnn-cli: Uses verify_module, list_ops, and compile_with_passes to power CLI commands\n\nSee also:\n- crates/shnn-compiler/src/passes.rs for Pass, PassManager, and built-in passes\n- docs/architecture/NIR_DIALECTS_AND_VERSIONING.md for dialects, ops, and versioning policy\n"]

#![deny(missing_docs)]

use std::collections::BTreeSet;

use shnn_ir::{
    AttributeValue, DialectKey, Module, Operation, OpVersion,
};
use shnn_runtime::{
    network::{NetworkBuilder, NetworkConfig},
    simulation::{SimulationEngine, SimulationParams, StimulusPattern, SimulationResult},
    neuron::LIFParams,
    plasticity::STDPParams,
    NeuronId, Result as RuntimeResult,
};

/// Public pass framework (no-op scaffolding)
pub mod passes;

/// Compiler error type
#[derive(thiserror::Error, Debug)]
pub enum CompilerError {
    /// Unsupported operation or version
    #[error("Unsupported op: {dialect}.{name}@{version}")]
    UnsupportedOp {
        /// Dialect
        dialect: String,
        /// Name
        name: String,
        /// Version
        version: String,
    },

    /// Missing required attribute
    #[error("Missing attribute '{0}' in {1}.{2}@{3}")]
    MissingAttr(String, String, String, String),

    /// Attribute type mismatch
    #[error("Bad attribute '{key}' in {dialect}.{name}@{version}: {reason}")]
    BadAttr {
        /// Key
        key: String,
        /// Dialect
        dialect: String,
        /// Name
        name: String,
        /// Version
        version: String,
        /// Reason
        reason: String,
    },

    /// Runtime layer error during lowering/building
    #[error("Runtime error: {0}")]
    Runtime(#[from] shnn_runtime::error::RuntimeError),

    /// Generic error
    #[error("{0}")]
    Message(String),
}

/// Result alias for compiler operations
pub type Result<T> = std::result::Result<T, CompilerError>;

/// Attribute kind spec for registry/type checking and introspection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttrKind {
    /// Boolean attribute
    Bool,
    /// 64-bit integer attribute
    I64,
    /// 32-bit floating point attribute
    F32,
    /// Duration in nanoseconds
    DurationNs,
    /// Absolute time in nanoseconds
    TimeNs,
    /// Voltage in millivolts
    VoltageMv,
    /// Resistance in megaohms
    ResistanceMohm,
    /// Capacitance in nanofarads
    CapacitanceNf,
    /// Current in nanoamps
    CurrentNa,
    /// Frequency in Hertz
    RateHz,
    /// Dimensionless synaptic weight (f32)
    Weight,
    /// Inclusive u32 range attribute
    RangeU32,
    /// Reference to a neuron by id
    NeuronRef,
}

impl AttrKind {
    /// Human-readable kind name for CLI/docs
    pub fn name(self) -> &'static str {
        match self {
            AttrKind::Bool => "bool",
            AttrKind::I64 => "i64",
            AttrKind::F32 => "f32",
            AttrKind::DurationNs => "DurationNs",
            AttrKind::TimeNs => "TimeNs",
            AttrKind::VoltageMv => "VoltageMv",
            AttrKind::ResistanceMohm => "ResistanceMohm",
            AttrKind::CapacitanceNf => "CapacitanceNf",
            AttrKind::CurrentNa => "CurrentNa",
            AttrKind::RateHz => "RateHz",
            AttrKind::Weight => "Weight(f32)",
            AttrKind::RangeU32 => "RangeU32",
            AttrKind::NeuronRef => "NeuronRef(%n<u32>)",
        }
    }
}

/// Attribute specification (name, kind, required)
#[derive(Debug, Clone, Copy)]
pub struct AttributeSpec {
    /// Attribute key
    pub name: &'static str,
    /// Expected kind/type
    pub kind: AttrKind,
    /// Required attribute (true) or optional (false)
    pub required: bool,
    /// Short doc string
    pub doc: &'static str,
}

/// Operation specification in the registry
#[derive(Debug, Clone, Copy)]
pub struct OpSpec {
    /// Dialect name
    pub dialect: &'static str,
    /// Op name
    pub name: &'static str,
    /// Version number
    pub version: u16,
    /// Attribute specs
    pub attrs: &'static [AttributeSpec],
}

/// Static registry of supported ops (v0)
static OPS: &[OpSpec] = &[
    OpSpec {
        dialect: "neuron",
        name: "lif",
        version: 1,
        attrs: &[
            AttributeSpec { name: "tau_m", kind: AttrKind::DurationNs, required: true, doc: "Membrane time constant (ns)" },
            AttributeSpec { name: "v_rest", kind: AttrKind::VoltageMv, required: true, doc: "Resting potential (mV)" },
            AttributeSpec { name: "v_reset", kind: AttrKind::VoltageMv, required: true, doc: "Reset potential (mV)" },
            AttributeSpec { name: "v_thresh", kind: AttrKind::VoltageMv, required: true, doc: "Threshold potential (mV)" },
            AttributeSpec { name: "t_refrac", kind: AttrKind::DurationNs, required: true, doc: "Refractory period (ns)" },
            AttributeSpec { name: "r_m", kind: AttrKind::ResistanceMohm, required: true, doc: "Membrane resistance (MΩ)" },
            AttributeSpec { name: "c_m", kind: AttrKind::CapacitanceNf, required: true, doc: "Capacitance (nF)" },
        ],
    },
    OpSpec {
        dialect: "plasticity",
        name: "stdp",
        version: 1,
        attrs: &[
            AttributeSpec { name: "a_plus", kind: AttrKind::F32, required: true, doc: "Potentiation amplitude" },
            AttributeSpec { name: "a_minus", kind: AttrKind::F32, required: true, doc: "Depression amplitude" },
            AttributeSpec { name: "tau_plus", kind: AttrKind::DurationNs, required: true, doc: "Potentiation time constant (ns)" },
            AttributeSpec { name: "tau_minus", kind: AttrKind::DurationNs, required: true, doc: "Depression time constant (ns)" },
            AttributeSpec { name: "w_min", kind: AttrKind::F32, required: true, doc: "Minimum weight" },
            AttributeSpec { name: "w_max", kind: AttrKind::F32, required: true, doc: "Maximum weight" },
        ],
    },
    OpSpec {
        dialect: "connectivity",
        name: "layer_fully_connected",
        version: 1,
        attrs: &[
            AttributeSpec { name: "in", kind: AttrKind::RangeU32, required: true, doc: "Inclusive input neuron range" },
            AttributeSpec { name: "out", kind: AttrKind::RangeU32, required: true, doc: "Inclusive output neuron range" },
            AttributeSpec { name: "weight", kind: AttrKind::Weight, required: true, doc: "Initial weight (unitless)" },
            AttributeSpec { name: "delay", kind: AttrKind::DurationNs, required: true, doc: "Synaptic delay (ns)" },
        ],
    },
    OpSpec {
        dialect: "connectivity",
        name: "synapse_connect",
        version: 1,
        attrs: &[
            AttributeSpec { name: "pre", kind: AttrKind::NeuronRef, required: true, doc: "Pre-synaptic neuron id" },
            AttributeSpec { name: "post", kind: AttrKind::NeuronRef, required: true, doc: "Post-synaptic neuron id" },
            AttributeSpec { name: "weight", kind: AttrKind::Weight, required: true, doc: "Synaptic weight (unitless)" },
            AttributeSpec { name: "delay", kind: AttrKind::DurationNs, required: true, doc: "Synaptic delay (ns)" },
        ],
    },
    OpSpec {
        dialect: "stimulus",
        name: "poisson",
        version: 1,
        attrs: &[
            AttributeSpec { name: "neuron", kind: AttrKind::NeuronRef, required: true, doc: "Target neuron id" },
            AttributeSpec { name: "rate", kind: AttrKind::RateHz, required: true, doc: "Firing rate (Hz)" },
            AttributeSpec { name: "amplitude", kind: AttrKind::CurrentNa, required: true, doc: "Current per spike (nA)" },
            AttributeSpec { name: "start", kind: AttrKind::TimeNs, required: true, doc: "Start time (ns)" },
            AttributeSpec { name: "duration", kind: AttrKind::DurationNs, required: true, doc: "Duration (ns)" },
        ],
    },
    OpSpec {
        dialect: "runtime",
        name: "simulate.run",
        version: 1,
        attrs: &[
            AttributeSpec { name: "dt", kind: AttrKind::DurationNs, required: true, doc: "Timestep (ns)" },
            AttributeSpec { name: "duration", kind: AttrKind::DurationNs, required: true, doc: "Total duration (ns)" },
            AttributeSpec { name: "record_potentials", kind: AttrKind::Bool, required: true, doc: "Record membrane potentials" },
            AttributeSpec { name: "seed", kind: AttrKind::I64, required: false, doc: "Optional RNG seed" },
        ],
    },
];

/// List op specifications for CLI introspection
pub fn list_ops() -> &'static [OpSpec] {
    OPS
}

/// Verify that a NIR module is semantically valid (v0 minimal checks).
/// Ensures required attributes exist and have acceptable types/units.
pub fn verify_module(module: &Module) -> Result<()> {
    for op in &module.ops {
        match (&op.dialect, op.name.as_str(), op.version) {
            (DialectKey::Neuron, "lif", OpVersion(1)) => {
                // Presence and type checks
                let tau_m_ns = duration_ns_from_attr(op, "tau_m")?;
                let _t_refrac_ns = duration_ns_from_attr(op, "t_refrac")?;
                let _ = f32_from_attr(op, "v_rest")?;
                let _ = f32_from_attr(op, "v_reset")?;
                let _ = f32_from_attr(op, "v_thresh")?;
                let r_m = f32_from_attr(op, "r_m")?;
                let c_m = f32_from_attr(op, "c_m")?;

                // Semantic/bounds checks
                if tau_m_ns == 0 {
                    return Err(CompilerError::BadAttr {
                        key: "tau_m".into(),
                        dialect: op.dialect.to_string(),
                        name: op.name.clone(),
                        version: op.version.to_string(),
                        reason: "must be > 0 ns".into(),
                    });
                }
                if r_m <= 0.0 {
                    return Err(CompilerError::BadAttr {
                        key: "r_m".into(),
                        dialect: op.dialect.to_string(),
                        name: op.name.clone(),
                        version: op.version.to_string(),
                        reason: "must be > 0 MΩ".into(),
                    });
                }
                if c_m <= 0.0 {
                    return Err(CompilerError::BadAttr {
                        key: "c_m".into(),
                        dialect: op.dialect.to_string(),
                        name: op.name.clone(),
                        version: op.version.to_string(),
                        reason: "must be > 0 nF".into(),
                    });
                }
            }
            (DialectKey::Plasticity, "stdp", OpVersion(1)) => {
                let _ = f32_from_attr(op, "a_plus")?;
                let _ = f32_from_attr(op, "a_minus")?;
                let tau_plus_ns = duration_ns_from_attr(op, "tau_plus")?;
                let tau_minus_ns = duration_ns_from_attr(op, "tau_minus")?;
                let w_min = f32_from_attr(op, "w_min")?;
                let w_max = f32_from_attr(op, "w_max")?;

                if tau_plus_ns == 0 {
                    return Err(CompilerError::BadAttr {
                        key: "tau_plus".into(),
                        dialect: op.dialect.to_string(),
                        name: op.name.clone(),
                        version: op.version.to_string(),
                        reason: "must be > 0 ns".into(),
                    });
                }
                if tau_minus_ns == 0 {
                    return Err(CompilerError::BadAttr {
                        key: "tau_minus".into(),
                        dialect: op.dialect.to_string(),
                        name: op.name.clone(),
                        version: op.version.to_string(),
                        reason: "must be > 0 ns".into(),
                    });
                }
                if w_min > w_max {
                    return Err(CompilerError::BadAttr {
                        key: "w_min".into(),
                        dialect: op.dialect.to_string(),
                        name: op.name.clone(),
                        version: op.version.to_string(),
                        reason: "must be <= w_max".into(),
                    });
                }
            }
            (DialectKey::Connectivity, "layer_fully_connected", OpVersion(1)) => {
                let (in_start, in_end) = range_from_attr(op, "in")?;
                let (out_start, out_end) = range_from_attr(op, "out")?;
                let _ = f32_from_attr(op, "weight")?;
                let _ = duration_ns_from_attr(op, "delay")?;
                if in_start > in_end {
                    return Err(CompilerError::BadAttr {
                        key: "in".into(),
                        dialect: op.dialect.to_string(),
                        name: op.name.clone(),
                        version: op.version.to_string(),
                        reason: "range must satisfy start <= end".into(),
                    });
                }
                if out_start > out_end {
                    return Err(CompilerError::BadAttr {
                        key: "out".into(),
                        dialect: op.dialect.to_string(),
                        name: op.name.clone(),
                        version: op.version.to_string(),
                        reason: "range must satisfy start <= end".into(),
                    });
                }
            }
            (DialectKey::Connectivity, "synapse_connect", OpVersion(1)) => {
                let _ = neuron_ref_from_attr(op, "pre")?;
                let _ = neuron_ref_from_attr(op, "post")?;
                let _ = f32_from_attr(op, "weight")?;
                let _ = duration_ns_from_attr(op, "delay")?;
                // Self-connections allowed; no further semantic checks here.
            }
            (DialectKey::Stimulus, "poisson", OpVersion(1)) => {
                let _ = neuron_ref_from_attr(op, "neuron")?;
                let rate = rate_hz_from_attr(op, "rate")?;
                let amp = current_na_from_attr(op, "amplitude")?;
                let _ = time_ns_from_attr(op, "start")?;
                let _ = duration_ns_from_attr(op, "duration")?;

                if rate < 0.0 {
                    return Err(CompilerError::BadAttr {
                        key: "rate".into(),
                        dialect: op.dialect.to_string(),
                        name: op.name.clone(),
                        version: op.version.to_string(),
                        reason: "must be >= 0 Hz".into(),
                    });
                }
                if amp < 0.0 {
                    return Err(CompilerError::BadAttr {
                        key: "amplitude".into(),
                        dialect: op.dialect.to_string(),
                        name: op.name.clone(),
                        version: op.version.to_string(),
                        reason: "must be >= 0 nA".into(),
                    });
                }
            }
            (DialectKey::Runtime, "simulate.run", OpVersion(1)) => {
                let dt = duration_ns_from_attr(op, "dt")?;
                let dur = duration_ns_from_attr(op, "duration")?;
                let _ = bool_from_attr(op, "record_potentials")?;
                let _ = i64_opt_from_attr(op, "seed")?;

                if dt == 0 {
                    return Err(CompilerError::BadAttr {
                        key: "dt".into(),
                        dialect: op.dialect.to_string(),
                        name: op.name.clone(),
                        version: op.version.to_string(),
                        reason: "must be > 0 ns".into(),
                    });
                }
                if dur == 0 {
                    return Err(CompilerError::BadAttr {
                        key: "duration".into(),
                        dialect: op.dialect.to_string(),
                        name: op.name.clone(),
                        version: op.version.to_string(),
                        reason: "must be > 0 ns".into(),
                    });
                }
                // Non-divisible duration is allowed in v0; engine may truncate last partial step.
            }
            (d, n, v) => {
                return Err(CompilerError::UnsupportedOp {
                    dialect: d.to_string(),
                    name: n.to_string(),
                    version: v.to_string(),
                });
            }
        }
    }
    Ok(())
}

/// Lowered program produced by the compiler
pub struct LoweredProgram {
    /// Built simulation engine ready to run
    pub engine: SimulationEngine,
    /// Stimuli collected during lowering (already added to engine)
    pub stimuli: Vec<StimulusPattern>,
}

impl LoweredProgram {
    /// Run simulation and get results
    pub fn run(mut self) -> RuntimeResult<SimulationResult> {
        self.engine.run()
    }
}

/// Compile a NIR module into a runnable program (builds network + simulation engine)
pub fn compile_module(module: &Module) -> Result<LoweredProgram> {
    // Defaults that can be overridden by ops
    let mut net_cfg = NetworkConfig::default();
    let mut builder = NetworkBuilder::new();
    let mut added_neurons: BTreeSet<u32> = BTreeSet::new();

    let mut sim_params: Option<SimulationParams> = None;
    let mut stimuli: Vec<StimulusPattern> = Vec::new();

    for op in &module.ops {
        match (&op.dialect, op.name.as_str(), op.version) {
            (DialectKey::Neuron, "lif", OpVersion(1)) => {
                let lif = lif_from_attrs(op)?;
                net_cfg.default_lif_params = lif;
            }
            (DialectKey::Plasticity, "stdp", OpVersion(1)) => {
                let stdp = stdp_from_attrs(op)?;
                net_cfg.default_stdp_params = stdp;
                net_cfg.plasticity_enabled = true;
            }
            (DialectKey::Connectivity, "layer_fully_connected", OpVersion(1)) => {
                let (in_start, in_end) = range_from_attr(op, "in")?;
                let (out_start, out_end) = range_from_attr(op, "out")?;
                let weight = f32_from_attr(op, "weight")?;
                let delay_ms = duration_ns_to_ms(op, "delay")?;

                // Ensure neurons exist for both ranges
                builder = add_range_if_missing(builder, &mut added_neurons, in_start, in_end);
                builder = add_range_if_missing(builder, &mut added_neurons, out_start, out_end);

                // Add synapses (fully connected)
                for pre in in_start..=in_end {
                    for post in out_start..=out_end {
                        builder = builder.add_synapse(NeuronId::new(pre), NeuronId::new(post), weight, delay_ms);
                    }
                }
            }
            (DialectKey::Connectivity, "synapse_connect", OpVersion(1)) => {
                let pre = neuron_ref_from_attr(op, "pre")?;
                let post = neuron_ref_from_attr(op, "post")?;
                let weight = f32_from_attr(op, "weight")?;
                let delay_ms = duration_ns_to_ms(op, "delay")?;

                // Ensure both neurons exist
                builder = add_range_if_missing(builder, &mut added_neurons, pre, pre);
                builder = add_range_if_missing(builder, &mut added_neurons, post, post);

                // Add single synapse
                builder = builder.add_synapse(NeuronId::new(pre), NeuronId::new(post), weight, delay_ms);
            }
            (DialectKey::Stimulus, "poisson", OpVersion(1)) => {
                let neuron = neuron_ref_from_attr(op, "neuron")?;
                let rate = rate_hz_from_attr(op, "rate")?;
                let amplitude = current_na_from_attr(op, "amplitude")?;
                let start_ns = time_ns_from_attr(op, "start")?;
                let dur_ns = duration_ns_from_attr(op, "duration")?;

                let pattern = StimulusPattern::Poisson {
                    neuron: NeuronId::new(neuron),
                    rate,
                    amplitude,
                    start_time: start_ns,
                    duration: dur_ns,
                };
                stimuli.push(pattern);
            }
            (DialectKey::Runtime, "simulate.run", OpVersion(1)) => {
                let dt_ns = duration_ns_from_attr(op, "dt")?;
                let duration_ns = duration_ns_from_attr(op, "duration")?;
                let record_potentials = bool_from_attr(op, "record_potentials")?;
                let seed = i64_opt_from_attr(op, "seed")?.map(|v| v as u64);

                let mut params = SimulationParams::new(dt_ns, duration_ns)
                    .map_err(CompilerError::Runtime)?;
                if record_potentials {
                    params = params.with_potential_recording(true);
                }
                if let Some(s) = seed {
                    params = params.with_seed(s);
                }
                sim_params = Some(params);
            }
            (d, n, v) => {
                return Err(CompilerError::UnsupportedOp {
                    dialect: d.to_string(),
                    name: n.to_string(),
                    version: v.to_string(),
                });
            }
        }
    }

    // Build network
    let network = builder.with_config(net_cfg).build()
        .map_err(CompilerError::Runtime)?;

    // Simulation params required
    let params = sim_params.ok_or_else(|| CompilerError::Message("Missing runtime.simulate.run@v1 op (no simulation params)".into()))?;

    // Create engine and add stimuli
    let mut engine = SimulationEngine::new(network, params)
        .map_err(CompilerError::Runtime)?;
    for s in &stimuli {
        engine.add_stimulus(s.clone());
    }

    Ok(LoweredProgram { engine, stimuli })
}

/// Compile with a (currently no-op) pass pipeline, then lower to runtime.
/// Runs verification before passes.
pub fn compile_with_passes(module: &Module) -> Result<LoweredProgram> {
    let mut m = module.clone();
    // Verify pre-pass
    verify_module(&m)?;
    // Run no-op passes (canonicalize, version upgrade)
    let mut pm = passes::PassManager::new();
    pm.add(Box::new(passes::CanonicalizePass));
    pm.add(Box::new(passes::UpgradeVersionsPass));
    pm.run(&mut m)?;
    // Lower
    compile_module(&m)
}
// ------------------------- Attribute helpers -------------------------

fn get_attr<'a>(op: &'a Operation, key: &str) -> Result<&'a AttributeValue> {
    op.attrs.get(key).ok_or_else(|| CompilerError::MissingAttr(
        key.to_string(),
        op.dialect.to_string(),
        op.name.clone(),
        op.version.to_string(),
    ))
}

fn bool_from_attr(op: &Operation, key: &str) -> Result<bool> {
    match get_attr(op, key)? {
        AttributeValue::Bool(b) => Ok(*b),
        other => Err(CompilerError::BadAttr {
            key: key.to_string(),
            dialect: op.dialect.to_string(),
            name: op.name.clone(),
            version: op.version.to_string(),
            reason: format!("expected Bool, got {:?}", other),
        }),
    }
}

fn i64_opt_from_attr(op: &Operation, key: &str) -> Result<Option<i64>> {
    match op.attrs.get(key) {
        None => Ok(None),
        Some(AttributeValue::I64(v)) => Ok(Some(*v)),
        Some(other) => Err(CompilerError::BadAttr {
            key: key.to_string(),
            dialect: op.dialect.to_string(),
            name: op.name.clone(),
            version: op.version.to_string(),
            reason: format!("expected I64, got {:?}", other),
        }),
    }
}

fn f32_from_attr(op: &Operation, key: &str) -> Result<f32> {
    match get_attr(op, key)? {
        AttributeValue::F32(v) => Ok(*v),
        AttributeValue::Weight(w) => Ok(*w),
        AttributeValue::VoltageMv(mv) => Ok(*mv),
        AttributeValue::ResistanceMohm(mohm) => Ok(*mohm),
        AttributeValue::CapacitanceNf(nf) => Ok(*nf),
        other => Err(CompilerError::BadAttr {
            key: key.to_string(),
            dialect: op.dialect.to_string(),
            name: op.name.clone(),
            version: op.version.to_string(),
            reason: format!("expected numeric f32-like attr, got {:?}", other),
        }),
    }
}

fn time_ns_from_attr(op: &Operation, key: &str) -> Result<u64> {
    match get_attr(op, key)? {
        AttributeValue::TimeNs(ns) => Ok(*ns),
        other => Err(CompilerError::BadAttr {
            key: key.to_string(),
            dialect: op.dialect.to_string(),
            name: op.name.clone(),
            version: op.version.to_string(),
            reason: format!("expected TimeNs, got {:?}", other),
        }),
    }
}

fn duration_ns_from_attr(op: &Operation, key: &str) -> Result<u64> {
    match get_attr(op, key)? {
        AttributeValue::DurationNs(ns) => Ok(*ns),
        other => Err(CompilerError::BadAttr {
            key: key.to_string(),
            dialect: op.dialect.to_string(),
            name: op.name.clone(),
            version: op.version.to_string(),
            reason: format!("expected DurationNs, got {:?}", other),
        }),
    }
}

fn duration_ns_to_ms(op: &Operation, key: &str) -> Result<f32> {
    let ns = duration_ns_from_attr(op, key)?;
    Ok(ns as f32 / 1_000_000.0)
}

fn rate_hz_from_attr(op: &Operation, key: &str) -> Result<f32> {
    match get_attr(op, key)? {
        AttributeValue::RateHz(hz) => Ok(*hz),
        other => Err(CompilerError::BadAttr {
            key: key.to_string(),
            dialect: op.dialect.to_string(),
            name: op.name.clone(),
            version: op.version.to_string(),
            reason: format!("expected RateHz, got {:?}", other),
        }),
    }
}

fn current_na_from_attr(op: &Operation, key: &str) -> Result<f32> {
    match get_attr(op, key)? {
        AttributeValue::CurrentNa(na) => Ok(*na),
        other => Err(CompilerError::BadAttr {
            key: key.to_string(),
            dialect: op.dialect.to_string(),
            name: op.name.clone(),
            version: op.version.to_string(),
            reason: format!("expected CurrentNa, got {:?}", other),
        }),
    }
}

fn neuron_ref_from_attr(op: &Operation, key: &str) -> Result<u32> {
    match get_attr(op, key)? {
        AttributeValue::NeuronRef(id) => Ok(*id),
        other => Err(CompilerError::BadAttr {
            key: key.to_string(),
            dialect: op.dialect.to_string(),
            name: op.name.clone(),
            version: op.version.to_string(),
            reason: format!("expected NeuronRef, got {:?}", other),
        }),
    }
}

fn range_from_attr(op: &Operation, key: &str) -> Result<(u32, u32)> {
    match get_attr(op, key)? {
        AttributeValue::RangeU32 { start, end } => Ok((*start, *end)),
        other => Err(CompilerError::BadAttr {
            key: key.to_string(),
            dialect: op.dialect.to_string(),
            name: op.name.clone(),
            version: op.version.to_string(),
            reason: format!("expected RangeU32, got {:?}", other),
        }),
    }
}

fn lif_from_attrs(op: &Operation) -> Result<LIFParams> {
    let tau_m_ns = duration_ns_from_attr(op, "tau_m")?;
    let t_refrac_ns = duration_ns_from_attr(op, "t_refrac")?;
    let v_rest = f32_from_attr(op, "v_rest")?;
    let v_reset = f32_from_attr(op, "v_reset")?;
    let v_thresh = f32_from_attr(op, "v_thresh")?;
    let r_m = f32_from_attr(op, "r_m")?;
    let c_m = f32_from_attr(op, "c_m")?;

    // Convert to existing runtime units (ms for times)
    let tau_m_ms = tau_m_ns as f32 / 1_000_000.0;
    let t_refrac_ms = t_refrac_ns as f32 / 1_000_000.0;

    let mut params = LIFParams::default();
    params.tau_m = tau_m_ms;
    params.t_refrac = t_refrac_ms;
    params.v_rest = v_rest;
    params.v_reset = v_reset;
    params.v_thresh = v_thresh;
    params.r_m = r_m;
    params.c_m = c_m;
    Ok(params)
}

fn stdp_from_attrs(op: &Operation) -> Result<STDPParams> {
    let a_plus = f32_from_attr(op, "a_plus")?;
    let a_minus = f32_from_attr(op, "a_minus")?;
    let tau_plus_ns = duration_ns_from_attr(op, "tau_plus")?;
    let tau_minus_ns = duration_ns_from_attr(op, "tau_minus")?;
    let w_min = f32_from_attr(op, "w_min")?;
    let w_max = f32_from_attr(op, "w_max")?;

    let tau_plus_ms = tau_plus_ns as f32 / 1_000_000.0;
    let tau_minus_ms = tau_minus_ns as f32 / 1_000_000.0;

    let mut params = STDPParams::default();
    params.a_plus = a_plus;
    params.a_minus = a_minus;
    params.tau_plus = tau_plus_ms;
    params.tau_minus = tau_minus_ms;
    params.w_min = w_min;
    params.w_max = w_max;

    Ok(params)
}

// Ensure neurons in [start..=end] exist; add if missing
fn add_range_if_missing(mut builder: NetworkBuilder, added: &mut BTreeSet<u32>, start: u32, end: u32) -> NetworkBuilder {
    for id in start..=end {
        if !added.contains(&id) {
            builder = builder.add_neuron(NeuronId::new(id));
            added.insert(id);
        }
    }
    builder
}

#[cfg(test)]
mod tests {
    use super::*;
    use shnn_ir::*;

    #[test]
    fn compile_minimal_program() {
        let mut m = Module::new();
        m.push(lif_neuron_v1(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
        m.push(stdp_rule_v1(0.01, 0.012, 20.0, 20.0, 0.0, 1.0));
        m.push(layer_fully_connected_v1(0, 0, 1, 1, 1.0, 1.0));
        m.push(stimulus_poisson_v1(0, 20.0, 10.0, 0.0, 100.0));
        m.push(runtime_simulate_run_v1(0.1, 10.0, false, Some(42)));

        let res = compile_module(&m).expect("compile").run().expect("run");
        assert!(res.steps_executed > 0);
    }
}