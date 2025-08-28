//! Pass framework for shnn-compiler (no-op scaffolding v0)

use shnn_ir::Module;

use crate::Result;

/// A compiler pass over a NIR Module
pub trait Pass {
    /// Human-readable pass name
    fn name(&self) -> &'static str;
    /// Execute the pass, mutating the module in-place
    fn run(&self, module: &mut Module) -> Result<()>;
}

/// Simple pass manager that runs passes in sequence
pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    /// Create an empty pass manager
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    /// Append a pass to the pipeline
    pub fn add(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    /// Run all passes in order
    pub fn run(&mut self, module: &mut Module) -> Result<()> {
        for p in &self.passes {
            // In future, add tracing/logging here
            p.run(module)?;
        }
        Ok(())
    }
}

/// Canonicalization pass
/// - Expands connectivity.layer_fully_connected into explicit connectivity.synapse_connect ops
/// - Normalizes attribute units to canonical forms
pub struct CanonicalizePass;

impl Pass for CanonicalizePass {
    fn name(&self) -> &'static str { "canonicalize" }
    fn run(&self, module: &mut Module) -> Result<()> {
        let mut new_ops = Vec::new();
        
        for op in &module.ops {
            match (&op.dialect, op.name.as_str(), op.version) {
                (shnn_ir::DialectKey::Connectivity, "layer_fully_connected", shnn_ir::OpVersion(1)) => {
                    // Expand into synapse_connect ops
                    let expanded = expand_layer_fully_connected(op)?;
                    new_ops.extend(expanded);
                }
                _ => {
                    // Keep other ops as-is
                    new_ops.push(op.clone());
                }
            }
        }
        
        module.ops = new_ops;
        Ok(())
    }
}

fn expand_layer_fully_connected(op: &shnn_ir::Operation) -> Result<Vec<shnn_ir::Operation>> {
    use shnn_ir::{AttributeValue, DialectKey, OpVersion};
    
    // Extract attributes
    let in_range = match op.attrs.get("in") {
        Some(AttributeValue::RangeU32 { start, end }) => (*start, *end),
        _ => return Err(crate::CompilerError::Message("layer_fully_connected missing 'in' range".into())),
    };
    
    let out_range = match op.attrs.get("out") {
        Some(AttributeValue::RangeU32 { start, end }) => (*start, *end),
        _ => return Err(crate::CompilerError::Message("layer_fully_connected missing 'out' range".into())),
    };
    
    let weight = match op.attrs.get("weight") {
        Some(AttributeValue::Weight(w)) => *w,
        Some(AttributeValue::F32(w)) => *w,
        _ => return Err(crate::CompilerError::Message("layer_fully_connected missing 'weight'".into())),
    };
    
    let delay = match op.attrs.get("delay") {
        Some(AttributeValue::DurationNs(ns)) => *ns,
        _ => return Err(crate::CompilerError::Message("layer_fully_connected missing 'delay'".into())),
    };
    
    // Generate synapse_connect ops
    let mut synapse_ops = Vec::new();
    for pre in in_range.0..=in_range.1 {
        for post in out_range.0..=out_range.1 {
            let mut synapse_op = shnn_ir::Operation::new(
                DialectKey::Connectivity,
                "synapse_connect",
                OpVersion(1)
            );
            synapse_op = synapse_op
                .with_attr("pre", AttributeValue::NeuronRef(pre))
                .with_attr("post", AttributeValue::NeuronRef(post))
                .with_attr("weight", AttributeValue::Weight(weight))
                .with_attr("delay", AttributeValue::DurationNs(delay));
            
            synapse_ops.push(synapse_op);
        }
    }
    
    Ok(synapse_ops)
}

/// Version upgrade pass
/// - Upgrades older op versions to current versions by inserting defaulted attributes
/// - Currently handles hypothetical upgrades from v0 to v1 (for future compatibility)
pub struct UpgradeVersionsPass;

impl Pass for UpgradeVersionsPass {
    fn name(&self) -> &'static str { "upgrade_versions" }
    fn run(&self, module: &mut Module) -> Result<()> {
        for op in &mut module.ops {
            match (&op.dialect, op.name.as_str(), op.version) {
                // Example: upgrade hypothetical lif@v0 to lif@v1
                // In practice, this would handle real version transitions
                (shnn_ir::DialectKey::Neuron, "lif", shnn_ir::OpVersion(0)) => {
                    upgrade_lif_v0_to_v1(op)?;
                }
                // Example: upgrade hypothetical stdp@v0 to stdp@v1
                (shnn_ir::DialectKey::Plasticity, "stdp", shnn_ir::OpVersion(0)) => {
                    upgrade_stdp_v0_to_v1(op)?;
                }
                _ => {
                    // Op is already current version or no upgrade path defined
                }
            }
        }
        Ok(())
    }
}

fn upgrade_lif_v0_to_v1(op: &mut shnn_ir::Operation) -> Result<()> {
    use shnn_ir::{AttributeValue, OpVersion};
    
    // Hypothetical v0->v1 upgrade: add missing t_refrac with default if not present
    if !op.attrs.contains_key("t_refrac") {
        op.attrs.insert("t_refrac".to_string(), AttributeValue::DurationNs(2_000_000)); // 2ms default
    }
    
    // Update version
    op.version = OpVersion(1);
    Ok(())
}

fn upgrade_stdp_v0_to_v1(op: &mut shnn_ir::Operation) -> Result<()> {
    use shnn_ir::{AttributeValue, OpVersion};
    
    // Hypothetical v0->v1 upgrade: add missing w_min/w_max with defaults if not present
    if !op.attrs.contains_key("w_min") {
        op.attrs.insert("w_min".to_string(), AttributeValue::Weight(0.0));
    }
    if !op.attrs.contains_key("w_max") {
        op.attrs.insert("w_max".to_string(), AttributeValue::Weight(1.0));
    }
    
    // Update version
    op.version = OpVersion(1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shnn_ir::{Module, layer_fully_connected_v1, lif_neuron_v1};

    #[test]
    fn pass_manager_runs_passes() {
        let mut m = Module::new();
        let mut pm = PassManager::new();
        pm.add(Box::new(CanonicalizePass));
        pm.add(Box::new(UpgradeVersionsPass));
        pm.run(&mut m).expect("passes run");
    }
    
    #[test]
    fn canonicalize_expands_layer_fully_connected() {
        let mut m = Module::new();
        m.push(lif_neuron_v1(20.0, -70.0, -70.0, -50.0, 2.0, 10.0, 1.0));
        m.push(layer_fully_connected_v1(0, 1, 2, 3, 1.0, 1.0)); // 2x2 = 4 synapses
        
        let original_op_count = m.ops.len();
        
        let mut pass = CanonicalizePass;
        pass.run(&mut m).expect("canonicalize pass");
        
        // Should have 1 lif + 4 synapse_connect ops (expanded from 1 layer_fully_connected)
        assert_eq!(m.ops.len(), original_op_count + 3); // +3 because 1 removed, 4 added
        
        // Count synapse_connect ops
        let synapse_count = m.ops.iter()
            .filter(|op| op.name == "synapse_connect")
            .count();
        assert_eq!(synapse_count, 4);
        
        // Should have no layer_fully_connected ops left
        let layer_count = m.ops.iter()
            .filter(|op| op.name == "layer_fully_connected")
            .count();
        assert_eq!(layer_count, 0);
    }
    
    #[test]
    fn version_upgrade_handles_hypothetical_v0() {
        use shnn_ir::{Operation, DialectKey, OpVersion, AttributeValue};
        
        let mut m = Module::new();
        
        // Create a hypothetical lif@v0 op missing t_refrac
        let mut lif_v0 = Operation::new(DialectKey::Neuron, "lif", OpVersion(0));
        lif_v0 = lif_v0
            .with_attr("tau_m", AttributeValue::DurationNs(20_000_000))
            .with_attr("v_rest", AttributeValue::VoltageMv(-70.0))
            .with_attr("v_reset", AttributeValue::VoltageMv(-70.0))
            .with_attr("v_thresh", AttributeValue::VoltageMv(-50.0))
            .with_attr("r_m", AttributeValue::ResistanceMohm(10.0))
            .with_attr("c_m", AttributeValue::CapacitanceNf(1.0));
        // Note: missing t_refrac
        
        m.push(lif_v0);
        
        let mut pass = UpgradeVersionsPass;
        pass.run(&mut m).expect("upgrade pass");
        
        // Should be upgraded to v1
        assert_eq!(m.ops[0].version, OpVersion(1));
        
        // Should have t_refrac added
        assert!(m.ops[0].attrs.contains_key("t_refrac"));
    }
}