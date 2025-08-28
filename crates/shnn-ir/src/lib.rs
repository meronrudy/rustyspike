#![doc = "Neuromorphic IR (NIR) - minimal core with MLIR-like textual printing (text-only v0)\n\nGoals (initial milestone):\n- Versioned, strongly-typed, unit-aware ops\n- MLIR-like textual format: dialect.op@vN { attrs }\n- Minimal types/attributes, verifier stubs, and printer\n\nFollow-ups (next milestones):\n- Parser, pass manager, type inference, proper verification\n- Registry/Lowering located in shnn-compiler (next crate)\n"]
#![warn(missing_docs)]

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

/// IR-wide result type
pub type Result<T> = std::result::Result<T, IrError>;

/// IR errors
#[derive(thiserror::Error, Debug)]
pub enum IrError {
    /// Generic IR error
    #[error("IR error: {0}")]
    Message(String),
}

/// Dialect key (static for now; aligns to MLIR-like dialect grouping)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum DialectKey {
    /// Neuron dialect (e.g., lif.neuron)
    Neuron,
    /// Plasticity dialect (e.g., stdp.rule)
    Plasticity,
    /// Connectivity dialect (e.g., layer.fully_connected, synapse.connect)
    Connectivity,
    /// Stimulus dialect (e.g., poisson)
    Stimulus,
    /// Runtime dialect (e.g., simulate.run)
    Runtime,
    /// Research/experimental dialects can use a string key
    Research(String),
}

impl Display for DialectKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            DialectKey::Neuron => write!(f, "neuron"),
            DialectKey::Plasticity => write!(f, "plasticity"),
            DialectKey::Connectivity => write!(f, "connectivity"),
            DialectKey::Stimulus => write!(f, "stimulus"),
            DialectKey::Runtime => write!(f, "runtime"),
            DialectKey::Research(s) => write!(f, "research.{}", s),
        }
    }
}

/// A version number for an operation (e.g., @v1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct OpVersion(pub u16);

impl Display for OpVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Base scalar types and semantic units
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    // Plain scalars
    Bool(bool),
    I64(i64),
    F32(f32),
    String(String),

    // Unit-aware scalars (canonicalized units noted in docs below)
    // Times are expressed canonically in nanoseconds in text printer with suffix `ns`.
    TimeNs(u64),
    // Duration in nanoseconds
    DurationNs(u64),
    // Electrical/biophysical units
    VoltageMv(f32),
    ResistanceMohm(f32),
    CapacitanceNf(f32),
    CurrentNa(f32),
    RateHz(f32),
    Weight(f32),

    // Ranges and references
    RangeU32 { start: u32, end: u32 }, // inclusive start..end
    NeuronRef(u32),
}

impl Display for AttributeValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            AttributeValue::Bool(v) => write!(f, "{}", v),
            AttributeValue::I64(v) => write!(f, "{}", v),
            AttributeValue::F32(v) => {
                // Print with up to necessary precision
                write!(f, "{}", v)
            }
            AttributeValue::String(s) => {
                // Quote strings
                write!(f, "\"{}\"", s.escape_debug())
            }
            AttributeValue::TimeNs(ns) => write!(f, "{} ns", ns),
            AttributeValue::DurationNs(ns) => write!(f, "{} ns", ns),
            AttributeValue::VoltageMv(mv) => write!(f, "{} mV", mv),
            AttributeValue::ResistanceMohm(mohm) => write!(f, "{} MΩ", mohm),
            AttributeValue::CapacitanceNf(nf) => write!(f, "{} nF", nf),
            AttributeValue::CurrentNa(na) => write!(f, "{} nA", na),
            AttributeValue::RateHz(hz) => write!(f, "{} Hz", hz),
            AttributeValue::Weight(w) => write!(f, "{}", w),
            AttributeValue::RangeU32 { start, end } => write!(f, "{}..{}", start, end),
            AttributeValue::NeuronRef(id) => write!(f, "%n{}", id),
        }
    }
}

/// A minimal type system placeholder (for future type inference)
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Type {
    /// Neuron handle/type
    Neuron,
    /// Synapse handle/type
    Synapse,
    /// Simulation handle/type
    Simulation,
    /// Unit/No result
    None,
}

/// An operation in the IR module
#[derive(Debug, Clone)]
pub struct Operation {
    /// Dialect (e.g., neuron, plasticity)
    pub dialect: DialectKey,
    /// Operation name within the dialect (e.g., "lif", "stdp", "layer_fully_connected")
    pub name: String,
    /// Version tag (e.g., v1)
    pub version: OpVersion,
    /// Attributes (typed and unit-aware)
    pub attrs: BTreeMap<String, AttributeValue>,
    /// Operands (not used in this minimal milestone)
    pub operands: Vec<String>,
    /// Results (not used yet)
    pub results: Vec<Type>,
    /// Nested operations/regions (for future composite ops)
    pub regions: Vec<Operation>,
}

impl Operation {
    /// Create a new operation
    pub fn new(dialect: DialectKey, name: impl Into<String>, version: OpVersion) -> Self {
        Self {
            dialect,
            name: name.into(),
            version,
            attrs: BTreeMap::new(),
            operands: Vec::new(),
            results: Vec::new(),
            regions: Vec::new(),
        }
    }

    /// Set an attribute on the operation
    pub fn with_attr(mut self, key: impl Into<String>, val: AttributeValue) -> Self {
        self.attrs.insert(key.into(), val);
        self
    }

    /// Add a nested op (region)
    pub fn with_region(mut self, op: Operation) -> Self {
        self.regions.push(op);
        self
    }

    /// MLIR-like op header: "dialect.name@vN"
    fn header(&self) -> String {
        format!("{}.{}@{}", self.dialect, self.name, self.version)
    }
}

/// A module that contains operations
#[derive(Debug, Default, Clone)]
pub struct Module {
    /// Module operations (top-level)
    pub ops: Vec<Operation>,
}

impl Module {
    /// Create an empty module
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Push a top-level operation
    pub fn push(&mut self, op: Operation) {
        self.ops.push(op);
    }

    /// Print textual IR (MLIR-like) for the module
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str("nir.module {\n");
        for op in &self.ops {
            Self::print_op(&mut out, op, 2);
            out.push('\n');
        }
        out.push_str("}\n");
        out
    }

    fn print_indent(out: &mut String, n: usize) {
        for _ in 0..n {
            out.push(' ');
        }
    }

    fn print_kv_list(out: &mut String, attrs: &BTreeMap<String, AttributeValue>) {
        let mut first = true;
        for (k, v) in attrs {
            if !first {
                out.push_str(", ");
            }
            first = false;
            out.push_str(k);
            out.push_str(" = ");
            out.push_str(&v.to_string());
        }
    }

    fn print_op(out: &mut String, op: &Operation, indent: usize) {
        Self::print_indent(out, indent);
        out.push_str(&op.header());

        // Attributes
        if !op.attrs.is_empty() || !op.regions.is_empty() {
            out.push_str(" {");
        }

        if !op.attrs.is_empty() {
            out.push(' ');
            Self::print_kv_list(out, &op.attrs);
            if !op.regions.is_empty() {
                out.push(' ');
            }
        }

        // Regions (nested ops)
        if !op.regions.is_empty() {
            out.push('\n');
            for nested in &op.regions {
                Self::print_op(out, nested, indent + 2);
                out.push('\n');
            }
            Self::print_indent(out, indent);
        }

        if !op.attrs.is_empty() || !op.regions.is_empty() {
            out.push('}');
        }
    }
}

// Convenience constructors for initial v1 ops (sugar layer)

/// lif.neuron@v1
pub fn lif_neuron_v1(
    tau_m_ms: f32,
    v_rest_mv: f32,
    v_reset_mv: f32,
    v_thresh_mv: f32,
    t_refrac_ms: f32,
    r_m_mohm: f32,
    c_m_nf: f32,
) -> Operation {
    Operation::new(DialectKey::Neuron, "lif", OpVersion(1))
        .with_attr("tau_m", AttributeValue::DurationNs((tau_m_ms * 1_000_000.0) as u64))
        .with_attr("v_rest", AttributeValue::VoltageMv(v_rest_mv))
        .with_attr("v_reset", AttributeValue::VoltageMv(v_reset_mv))
        .with_attr("v_thresh", AttributeValue::VoltageMv(v_thresh_mv))
        .with_attr("t_refrac", AttributeValue::DurationNs((t_refrac_ms * 1_000_000.0) as u64))
        .with_attr("r_m", AttributeValue::ResistanceMohm(r_m_mohm))
        .with_attr("c_m", AttributeValue::CapacitanceNf(c_m_nf))
}

/// plasticity.stdp@v1
pub fn stdp_rule_v1(
    a_plus: f32,
    a_minus: f32,
    tau_plus_ms: f32,
    tau_minus_ms: f32,
    w_min: f32,
    w_max: f32,
) -> Operation {
    Operation::new(DialectKey::Plasticity, "stdp", OpVersion(1))
        .with_attr("a_plus", AttributeValue::F32(a_plus))
        .with_attr("a_minus", AttributeValue::F32(a_minus))
        .with_attr("tau_plus", AttributeValue::DurationNs((tau_plus_ms * 1_000_000.0) as u64))
        .with_attr("tau_minus", AttributeValue::DurationNs((tau_minus_ms * 1_000_000.0) as u64))
        .with_attr("w_min", AttributeValue::Weight(w_min))
        .with_attr("w_max", AttributeValue::Weight(w_max))
}

/// connectivity.layer_fully_connected@v1
pub fn layer_fully_connected_v1(
    in_start: u32,
    in_end: u32,
    out_start: u32,
    out_end: u32,
    weight: f32,
    delay_ms: f32,
) -> Operation {
    Operation::new(DialectKey::Connectivity, "layer_fully_connected", OpVersion(1))
        .with_attr("in", AttributeValue::RangeU32 { start: in_start, end: in_end })
        .with_attr("out", AttributeValue::RangeU32 { start: out_start, end: out_end })
        .with_attr("weight", AttributeValue::Weight(weight))
        .with_attr("delay", AttributeValue::DurationNs((delay_ms * 1_000_000.0) as u64))
}

/// stimulus.poisson@v1
pub fn stimulus_poisson_v1(
    neuron_id: u32,
    rate_hz: f32,
    amplitude_na: f32,
    start_ms: f32,
    duration_ms: f32,
) -> Operation {
    Operation::new(DialectKey::Stimulus, "poisson", OpVersion(1))
        .with_attr("neuron", AttributeValue::NeuronRef(neuron_id))
        .with_attr("rate", AttributeValue::RateHz(rate_hz))
        .with_attr("amplitude", AttributeValue::CurrentNa(amplitude_na))
        .with_attr("start", AttributeValue::TimeNs((start_ms * 1_000_000.0) as u64))
        .with_attr("duration", AttributeValue::DurationNs((duration_ms * 1_000_000.0) as u64))
}

/// connectivity.synapse_connect@v1
pub fn synapse_connect_v1(
    pre_neuron: u32,
    post_neuron: u32,
    weight: f32,
    delay_ms: f32,
) -> Operation {
    Operation::new(DialectKey::Connectivity, "synapse_connect", OpVersion(1))
        .with_attr("pre", AttributeValue::NeuronRef(pre_neuron))
        .with_attr("post", AttributeValue::NeuronRef(post_neuron))
        .with_attr("weight", AttributeValue::Weight(weight))
        .with_attr("delay", AttributeValue::DurationNs((delay_ms * 1_000_000.0) as u64))
}

/// runtime.simulate.run@v1
pub fn runtime_simulate_run_v1(
    dt_ms: f32,
    duration_ms: f32,
    record_potentials: bool,
    seed: Option<u64>,
) -> Operation {
    let mut op = Operation::new(DialectKey::Runtime, "simulate.run", OpVersion(1))
        .with_attr("dt", AttributeValue::DurationNs((dt_ms * 1_000_000.0) as u64))
        .with_attr("duration", AttributeValue::DurationNs((duration_ms * 1_000_000.0) as u64))
        .with_attr("record_potentials", AttributeValue::Bool(record_potentials));

    if let Some(s) = seed {
        op = op.with_attr("seed", AttributeValue::I64(s as i64));
    }
    op
}

/// Parse textual NIR (MLIR-like) into a Module (minimal subset).
/// Supports single-line ops with attribute list printed by to_text().
pub fn parse_text(input: &str) -> Result<Module> {
    let mut module = Module::new();
    for raw in input.lines() {
        let line = raw.trim();
        if line.is_empty() || line == "nir.module {" || line == "}" || line == "{" {
            continue;
        }

        // Expect single-line op: "dialect.name@vN { key = val, key = val }"
        // Regions are not supported in v0 parser.
        let (header, attrs) = if let Some(brace_pos) = line.find('{') {
            let header = line[..brace_pos].trim();
            // find closing '}'
            let close_pos = line.rfind('}').ok_or_else(|| IrError::Message("missing '}' in op line".into()))?;
            let attrs_str = line[brace_pos + 1..close_pos].trim();
            (header, attrs_str)
        } else {
            // allow ops without attributes (no braces)
            (line, "")
        };

        let op = parse_op_line(header, attrs)?;
        module.push(op);
    }
    Ok(module)
}

fn parse_op_line(header: &str, attrs: &str) -> Result<Operation> {
    // header: "dialect.name@vN"
    let at_pos = header.rfind('@').ok_or_else(|| IrError::Message(format!("missing @version in header '{}'", header)))?;
    let left = header[..at_pos].trim();
    let ver = header[at_pos + 1..].trim();
    if !ver.starts_with('v') {
        return Err(IrError::Message(format!("bad version '{}'", ver)));
    }
    let version_num: u16 = ver[1..].parse().map_err(|_| IrError::Message(format!("bad version number '{}'", ver)))?;

    let dot_pos = left.find('.').ok_or_else(|| IrError::Message(format!("missing '.' in op name '{}'", left)))?;
    let dialect_str = &left[..dot_pos];
    let name = left[dot_pos + 1..].to_string();

    let dialect = match dialect_str {
        "neuron" => DialectKey::Neuron,
        "plasticity" => DialectKey::Plasticity,
        "connectivity" => DialectKey::Connectivity,
        "stimulus" => DialectKey::Stimulus,
        "runtime" => DialectKey::Runtime,
        other => DialectKey::Research(other.to_string()),
    };

    let mut op = Operation::new(dialect, name, OpVersion(version_num));

    let attrs_str = attrs.trim();
    if !attrs_str.is_empty() {
        // split by ", " (printer uses comma+space)
        for pair in split_top_level(attrs_str, ',') {
            let part = pair.trim();
            if part.is_empty() {
                continue;
            }
            let eq_pos = part.find('=').ok_or_else(|| IrError::Message(format!("missing '=' in attr '{}'", part)))?;
            let key = part[..eq_pos].trim();
            let val_str = part[eq_pos + 1..].trim();
            let val = parse_attr_value(key, val_str)?;
            op = op.with_attr(key, val);
        }
    }
    Ok(op)
}

fn split_top_level(s: &str, delim: char) -> Vec<String> {
    // Because our values never contain commas except as separators,
    // a simple split is sufficient.
    s.split(delim).map(|t| t.to_string()).collect()
}

fn parse_attr_value(key: &str, s: &str) -> Result<AttributeValue> {
    // String: "...."
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        let inner = &s[1..s.len() - 1];
        return Ok(AttributeValue::String(inner.to_string()));
    }

    // NeuronRef: %n123
    if let Some(nstr) = s.strip_prefix("%n") {
        let id: u32 = nstr.trim().parse().map_err(|_| IrError::Message(format!("bad NeuronRef '{}'", s)))?;
        return Ok(AttributeValue::NeuronRef(id));
    }

    // Range: a..b
    if let Some(range_pos) = s.find("..") {
        let a = s[..range_pos].trim();
        let b = s[range_pos + 2..].trim();
        let start: u32 = a.parse().map_err(|_| IrError::Message(format!("bad range start '{}'", s)))?;
        let end: u32 = b.parse().map_err(|_| IrError::Message(format!("bad range end '{}'", s)))?;
        return Ok(AttributeValue::RangeU32 { start, end });
    }

    // Units: order matters (match longer suffixes first)
    if let Some(val) = s.strip_suffix(" ns") {
        let ns: u64 = val.trim().parse().map_err(|_| IrError::Message(format!("bad ns value '{}'", s)))?;
        // Heuristic: "start" is TimeNs, others are DurationNs in v0 printer
        if key == "start" {
            return Ok(AttributeValue::TimeNs(ns));
        } else {
            return Ok(AttributeValue::DurationNs(ns));
        }
    }
    if let Some(val) = s.strip_suffix(" mV") {
        let v: f32 = val.trim().parse().map_err(|_| IrError::Message(format!("bad mV value '{}'", s)))?;
        return Ok(AttributeValue::VoltageMv(v));
    }
    if let Some(val) = s.strip_suffix(" MΩ") {
        let v: f32 = val.trim().parse().map_err(|_| IrError::Message(format!("bad MΩ value '{}'", s)))?;
        return Ok(AttributeValue::ResistanceMohm(v));
    }
    if let Some(val) = s.strip_suffix(" nF") {
        let v: f32 = val.trim().parse().map_err(|_| IrError::Message(format!("bad nF value '{}'", s)))?;
        return Ok(AttributeValue::CapacitanceNf(v));
    }
    if let Some(val) = s.strip_suffix(" nA") {
        let v: f32 = val.trim().parse().map_err(|_| IrError::Message(format!("bad nA value '{}'", s)))?;
        return Ok(AttributeValue::CurrentNa(v));
    }
    if let Some(val) = s.strip_suffix(" Hz") {
        let v: f32 = val.trim().parse().map_err(|_| IrError::Message(format!("bad Hz value '{}'", s)))?;
        return Ok(AttributeValue::RateHz(v));
    }

    // Booleans
    if s == "true" {
        return Ok(AttributeValue::Bool(true));
    }
    if s == "false" {
        return Ok(AttributeValue::Bool(false));
    }

    // Numeric values: context-sensitive parsing
    // Special-case: seeds should be integers
    if key == "seed" {
        if let Ok(i) = s.parse::<i64>() {
            return Ok(AttributeValue::I64(i));
        } else {
            return Err(IrError::Message(format!("seed must be integer i64, got '{}'", s)));
        }
    }

    // Prefer f32 for unitless numeric to avoid integer-vs-float mismatches after round-trip.
    if let Ok(f) = s.parse::<f32>() {
        // Map weight-like keys to Weight
        if key == "weight" || key == "w_min" || key == "w_max" {
            return Ok(AttributeValue::Weight(f));
        }
        return Ok(AttributeValue::F32(f));
    }

    // Fallback: integer
    if let Ok(i) = s.parse::<i64>() {
        return Ok(AttributeValue::I64(i));
    }

    Err(IrError::Message(format!("unrecognized attribute value '{}'", s)))
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_minimal_module() {
        let mut m = Module::new();

        m.push(lif_neuron_v1(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
        m.push(stdp_rule_v1(0.01, 0.012, 20.0, 20.0, 0.0, 1.0));
        m.push(layer_fully_connected_v1(0, 9, 10, 59, 1.0, 1.0));
        m.push(stimulus_poisson_v1(0, 20.0, 10.0, 0.0, 500.0));
        m.push(runtime_simulate_run_v1(0.1, 500.0, false, Some(42)));

        let text = m.to_text();
        assert!(text.contains("nir.module {"));
        assert!(text.contains("neuron.lif@v1"));
        assert!(text.contains("plasticity.stdp@v1"));
        assert!(text.contains("connectivity.layer_fully_connected@v1"));
        assert!(text.contains("stimulus.poisson@v1"));
        assert!(text.contains("runtime.simulate.run@v1"));
    }
}
#[cfg(test)]
mod parser_roundtrip_tests {
    use super::*;

    #[test]
    fn parse_single_op_line() {
        let text = "nir.module {\n  neuron.lif@v1 { c_m = 1 nF, r_m = 10 MΩ, t_refrac = 2000000 ns, tau_m = 20000000 ns, v_reset = -70 mV, v_rest = -70 mV, v_thresh = -50 mV}\n}\n";
        let module = parse_text(text).expect("parse");
        assert_eq!(module.ops.len(), 1);
        let printed = module.to_text();
        assert_eq!(printed, text);
    }

    #[test]
    fn parse_and_roundtrip_full_example() {
        // Build with constructors, print, parse, print again
        let mut m = Module::new();
        m.push(lif_neuron_v1(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
        m.push(stdp_rule_v1(0.01, 0.012, 20.0, 20.0, 0.0, 1.0));
        m.push(layer_fully_connected_v1(0, 9, 10, 59, 1.0, 1.0));
        m.push(stimulus_poisson_v1(0, 20.0, 10.0, 0.0, 500.0));
        m.push(runtime_simulate_run_v1(0.1, 500.0, false, Some(42)));

        let text1 = m.to_text();
        let parsed = parse_text(&text1).expect("parse");
        let text2 = parsed.to_text();

        // The textual printer is canonical; round-trip must be identical.
        assert_eq!(text1, text2);
    }
}