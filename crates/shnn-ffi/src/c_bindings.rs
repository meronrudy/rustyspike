//! C ABI for thin-waist HSNN operations (network build, run, VEVT export)
//!
//! This module is compiled when the "c-bindings" feature is enabled.

use std::ptr;
use libc::{c_void};
use shnn_runtime::{NetworkBuilder, SNNNetwork, NeuronId};
use shnn_runtime::simulation::run_fixed_step;
use shnn_storage::vevt::{VEVTEvent, encode_vevt};
use shnn_storage::{StreamId, Time as StorageTime};

/// Opaque network builder handle
#[repr(C)]
pub struct HSNN_NetworkBuilder {
    inner: Option<NetworkBuilder>,
}

/// Opaque network handle
#[repr(C)]
pub struct HSNN_Network {
    inner: SNNNetwork,
}

/// Weight triple used for snapshot/import operations
#[repr(C)]
#[derive(Copy, Clone)]
pub struct HSNN_WeightTriple {
    pub pre: u32,
    pub post: u32,
    pub weight: f32,
}

/// Create a new network builder handle
#[no_mangle]
pub extern "C" fn hsnn_network_builder_new(out_builder: *mut *mut HSNN_NetworkBuilder) -> i32 {
    if out_builder.is_null() { return 1008; }
    let builder = HSNN_NetworkBuilder { inner: Some(NetworkBuilder::new()) };
    let boxed = Box::new(builder);
    unsafe { *out_builder = Box::into_raw(boxed); }
    0
}

/// Add a contiguous range of neurons [start, start+count)
#[no_mangle]
pub extern "C" fn hsnn_network_builder_add_neuron_range(builder: *mut HSNN_NetworkBuilder, start: u32, count: u32) -> i32 {
    if builder.is_null() { return 1008; }
    let b = unsafe { &mut *builder };
    let inner = b.inner.take().expect("builder consumed");
    let new_inner = inner.add_neurons(start, count);
    b.inner = Some(new_inner);
    0
}

/// Add a simple synapse with default delay (1ms)
#[no_mangle]
pub extern "C" fn hsnn_network_builder_add_synapse_simple(builder: *mut HSNN_NetworkBuilder, pre: u32, post: u32, weight: f32) -> i32 {
    if builder.is_null() { return 1008; }
    let b = unsafe { &mut *builder };
    let inner = b.inner.take().expect("builder consumed");
    let new_inner = inner.add_synapse_simple(NeuronId::new(pre), NeuronId::new(post), weight);
    b.inner = Some(new_inner);
    0
}

/// Finalize the builder into a network handle (consumes the builder)
#[no_mangle]
pub extern "C" fn hsnn_network_build(builder: *mut HSNN_NetworkBuilder, out_network: *mut *mut HSNN_Network) -> i32 {
    if builder.is_null() || out_network.is_null() { return 1008; }
    let boxed_builder = unsafe { Box::from_raw(builder) };
    let inner = boxed_builder.inner.expect("builder missing inner");
    let built = match inner.build() {
        Ok(n) => n,
        Err(_) => return 1008,
    };
    let net = HSNN_Network { inner: built };
    let boxed_net = Box::new(net);
    unsafe { *out_network = Box::into_raw(boxed_net); }
    0
}

/// Free a network handle
#[no_mangle]
pub extern "C" fn hsnn_network_free(network: *mut HSNN_Network) {
    if network.is_null() { return; }
    unsafe { drop(Box::from_raw(network)); }
}

/// Free a buffer allocated by this library (malloc-based)
#[no_mangle]
pub extern "C" fn hsnn_free_buffer(ptr: *mut u8) {
    if ptr.is_null() { return; }
    unsafe { libc::free(ptr as *mut c_void); }
}

/// Run a deterministic fixed-step simulation, consuming the network, and return VEVT bytes.
/// On success returns 0 and sets (*out_ptr, *out_len). The network handle is invalidated.
/// On failure, returns an error code and leaves outputs untouched.
#[no_mangle]
pub extern "C" fn hsnn_run_fixed_step_vevt_consume(
    network_ptr: *mut *mut HSNN_Network,
    dt_ns: u64,
    duration_ns: u64,
    seed: u64,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    if network_ptr.is_null() || out_ptr.is_null() || out_len.is_null() {
        return 1008;
    }
    let net_handle = unsafe { *network_ptr };
    if net_handle.is_null() {
        return 1008;
    }
    // Take ownership and null-out caller pointer
    let boxed_net = unsafe { Box::from_raw(net_handle) };
    unsafe { *network_ptr = ptr::null_mut(); }

    // Run simulation
    let result = match run_fixed_step(boxed_net.inner, dt_ns, duration_ns, Some(seed)) {
        Ok(r) => r,
        Err(_) => return 1006,
    };

    // Build VEVT events
    let spikes = result.export_spikes();
    let (start_ns, end_ns) = if spikes.is_empty() {
        (0u64, result.duration_ns)
    } else {
        let min_t = spikes.iter().map(|(t, _)| *t).min().unwrap();
        let max_t = spikes.iter().map(|(t, _)| *t).max().unwrap();
        (min_t, max_t)
    };
    let events: Vec<VEVTEvent> = spikes.into_iter().map(|(time_ns, neuron_id)| VEVTEvent {
        timestamp: time_ns,
        event_type: 0,
        source_id: neuron_id,
        target_id: u32::MAX,
        payload_size: 0,
        reserved: 0,
    }).collect();
    let bytes = match encode_vevt(
        StreamId::new(1),
        StorageTime::from_nanos(start_ns),
        StorageTime::from_nanos(end_ns),
        &events
    ) {
        Ok(b) => b,
        Err(_) => return 9999,
    };

    // Allocate C buffer and copy
    let len = bytes.len();
    let buf = unsafe { libc::malloc(len) as *mut u8 };
    if buf.is_null() {
        return 1005;
    }
    unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), buf, len); }
    unsafe {
        *out_ptr = buf;
        *out_len = len;
    }
    0
}

/// Export all synaptic weights as a C array of HSNN_WeightTriple (malloc-allocated).
/// Caller must free with hsnn_free_buffer((uint8_t*)ptr).
#[no_mangle]
pub extern "C" fn hsnn_network_snapshot_weights(
    network: *const HSNN_Network,
    out_ptr: *mut *mut HSNN_WeightTriple,
    out_len: *mut usize,
) -> i32 {
    if network.is_null() || out_ptr.is_null() || out_len.is_null() {
        return 1008; // invalid args
    }
    let net = unsafe { &*network };

    // Collect snapshot as triples
    let triples = net.inner.synapse_connections();
    let len = triples.len();

    if len == 0 {
        // No weights to export
        unsafe {
            *out_ptr = std::ptr::null_mut();
            *out_len = 0;
        }
        return 0;
    }

    // Allocate C buffer
    let bytes = len
        .checked_mul(std::mem::size_of::<HSNN_WeightTriple>())
        .unwrap_or(0);
    let buf = unsafe { libc::malloc(bytes) as *mut HSNN_WeightTriple };
    if buf.is_null() {
        return 1005; // allocation failed
    }

    // Fill buffer
    let slice = unsafe { std::slice::from_raw_parts_mut(buf, len) };
    for (i, (pre, post, w)) in triples.into_iter().enumerate() {
        slice[i] = HSNN_WeightTriple {
            pre: pre.raw(),
            post: post.raw(),
            weight: w,
        };
    }

    unsafe {
        *out_ptr = buf;
        *out_len = len;
    }
    0
}

/// Apply weight updates from a C array of HSNN_WeightTriple.
/// Only existing synapses are updated; missing connections are ignored.
/// Returns 0 on success and sets out_applied to the number of updates applied.
#[no_mangle]
pub extern "C" fn hsnn_network_apply_weight_updates(
    network: *mut HSNN_Network,
    updates_ptr: *const HSNN_WeightTriple,
    updates_len: usize,
    out_applied: *mut usize,
) -> i32 {
    if network.is_null() || out_applied.is_null() {
        return 1008; // invalid args
    }
    if updates_len > 0 && updates_ptr.is_null() {
        return 1008; // invalid updates buffer
    }

    let net = unsafe { &mut *network };

    if updates_len == 0 {
        unsafe { *out_applied = 0; }
        return 0;
    }

    let updates = unsafe { std::slice::from_raw_parts(updates_ptr, updates_len) };
    let mut applied = 0usize;

    for upd in updates {
        // Attempt to set weight; ignore errors for missing synapses
        if net.inner.set_weight(NeuronId::new(upd.pre), NeuronId::new(upd.post), upd.weight).is_ok() {
            applied += 1;
        }
    }

    unsafe { *out_applied = applied; }
    0
}