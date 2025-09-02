use anyhow::{anyhow, Result};
use shnn_storage::{
    vevt::{encode_vevt, decode_vevt, VEVTEvent},
    StreamId,
    Time as StorageTime,
};
use std::fs;

fn main() -> Result<()> {
    // Create a tiny synthetic spike stream (10 spikes)
    let mut events: Vec<VEVTEvent> = Vec::new();
    let mut t = 0u64;
    for i in 0..10u32 {
        // 100_000 ns between events (0.1 ms)
        t += 100_000;
        events.push(VEVTEvent {
            timestamp: t,
            event_type: 0, // 0 = spike
            source_id: i,
            target_id: u32::MAX,
            payload_size: 0,
            reserved: 0,
        });
    }

    let start_ns = events.first().map(|e| e.timestamp).unwrap_or(0);
    let end_ns = events.last().map(|e| e.timestamp).unwrap_or(0);

    // Encode to VEVT bytes
    let bytes = encode_vevt(
        StreamId::new(1),
        StorageTime::from_nanos(start_ns),
        StorageTime::from_nanos(end_ns),
        &events,
    )?;
    fs::write("out.vevt", &bytes)?;

    // Decode from bytes and validate
    let roundtrip = fs::read("out.vevt")?;
    let (_hdr, decoded) = decode_vevt(&roundtrip).map_err(|e| anyhow!("decode failed: {e}"))?;

    if decoded.len() != events.len() {
        return Err(anyhow!("decoded {} events, expected {}", decoded.len(), events.len()));
    }

    println!("VEVT OK: {} events from {} ns to {} ns", decoded.len(), start_ns, end_ns);
    Ok(())
}