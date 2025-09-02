use anyhow::{Result, anyhow};
use shnn_storage::{
    ids::{MaskId, GenerationId},
    traits::{MaskType, Mask},
    vmsk::BitmapMask,
};

use std::fs;

fn main() -> Result<()> {
    // Create a simple bitmap mask over 128 bits
    let mut mask = BitmapMask::new(MaskId::new(1), MaskType::VertexMask, GenerationId::new(1), 128);

    // Activate a few bits
    let bits = [1u64, 3, 64, 127];
    for &b in &bits {
        mask.set_bit(b)?;
    }

    // Export to VMSK bytes and write to disk
    let bytes = mask.export_vmsk()?;
    fs::write("out.vmsk", &bytes)?;

    // Import from bytes and validate
    let roundtrip = fs::read("out.vmsk")?;
    let imported = BitmapMask::import_vmsk(&roundtrip)?;

    for &b in &bits {
        if !imported.is_active(b as u32) {
            return Err(anyhow!("bit {} not active after import", b));
        }
    }

    if imported.active_count() != bits.len() as u64 {
        return Err(anyhow!("active_count {} != {}", imported.active_count(), bits.len()));
    }

    println!(
        "VMSK OK: {} active bits, indices {:?}",
        imported.active_count(),
        imported.active_indices()
    );
    Ok(())
}