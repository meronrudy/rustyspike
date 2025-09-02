// Run with: cargo test -p shnn-ffi --features c-bindings
#![allow(clippy::missing_safety_doc)]

#[cfg(feature = "c-bindings")]
mod tests {
    use std::ptr;
    use shnn_storage::vevt::decode_vevt;

    // Import the C ABI symbols from the crate's c_bindings module
    use shnn_ffi::c_bindings::{
        HSNN_NetworkBuilder,
        HSNN_Network,
        HSNN_WeightTriple,
        hsnn_network_builder_new,
        hsnn_network_builder_add_neuron_range,
        hsnn_network_builder_add_synapse_simple,
        hsnn_network_build,
        hsnn_run_fixed_step_vevt_consume,
        hsnn_network_snapshot_weights,
        hsnn_network_apply_weight_updates,
        hsnn_network_free,
        hsnn_free_buffer,
    };

    // Helper to check error code == 0
    fn ok(code: i32) {
        assert_eq!(code, 0, "FFI returned error code {}", code);
    }

    #[test]
    fn c_abi_smoke_build_and_run_fixed_step() {
        unsafe {
            // Create builder
            let mut builder_ptr: *mut HSNN_NetworkBuilder = ptr::null_mut();
            ok(hsnn_network_builder_new(&mut builder_ptr as *mut _));
            assert!(!builder_ptr.is_null());

            // Add neurons [0, 2)
            ok(hsnn_network_builder_add_neuron_range(builder_ptr, 0, 2));
            // Add synapse 0 -> 1 with weight 0.2
            ok(hsnn_network_builder_add_synapse_simple(builder_ptr, 0, 1, 0.2));

            // Build network (consumes builder)
            let mut net_ptr: *mut HSNN_Network = ptr::null_mut();
            ok(hsnn_network_build(builder_ptr, &mut net_ptr as *mut _));
            // builder_ptr is invalid now; ensure net created
            assert!(!net_ptr.is_null());

            // Run fixed step and export VEVT
            let mut net_owner: *mut HSNN_Network = net_ptr; // pass pointer-to-pointer for consume
            let mut vevt_ptr: *mut u8 = ptr::null_mut();
            let mut vevt_len: usize = 0;

            ok(hsnn_run_fixed_step_vevt_consume(
                &mut net_owner as *mut _,
                100_000,     // dt_ns = 0.1 ms
                1_000_000,   // duration_ns = 1 ms
                1234,        // seed
                &mut vevt_ptr as *mut _,
                &mut vevt_len as *mut _,
            ));

            // After consume, network pointer should be nulled by callee
            assert!(net_owner.is_null(), "network pointer should be null after consume");

            // Validate VEVT bytes
            assert!(!vevt_ptr.is_null(), "VEVT pointer should be non-null");
            assert!(vevt_len > 0, "VEVT length should be > 0");

            // Copy VEVT bytes into a Rust Vec to decode
            let bytes: Vec<u8> = std::slice::from_raw_parts(vevt_ptr, vevt_len).to_vec();

            // Free C buffer
            hsnn_free_buffer(vevt_ptr);

            // Decode to verify format correctness
            let (_hdr, _events) = decode_vevt(&bytes).expect("decode_vevt should succeed");
            #[test]
            fn c_abi_weight_snapshot_and_apply() {
                unsafe {
                    // Build a minimal network with 2 neurons and one synapse 0->1 (0.2)
                    let mut builder_ptr: *mut HSNN_NetworkBuilder = ptr::null_mut();
                    ok(hsnn_network_builder_new(&mut builder_ptr as *mut _));
                    ok(hsnn_network_builder_add_neuron_range(builder_ptr, 0, 2));
                    ok(hsnn_network_builder_add_synapse_simple(builder_ptr, 0, 1, 0.2));
        
                    let mut net_ptr: *mut HSNN_Network = ptr::null_mut();
                    ok(hsnn_network_build(builder_ptr, &mut net_ptr as *mut _));
                    assert!(!net_ptr.is_null());
        
                    // Snapshot weights
                    let mut w_ptr: *mut HSNN_WeightTriple = std::ptr::null_mut();
                    let mut w_len: usize = 0;
                    ok(hsnn_network_snapshot_weights(
                        net_ptr as *const _,
                        &mut w_ptr as *mut _,
                        &mut w_len as *mut _,
                    ));
                    assert_eq!(w_len, 1, "expected one weight");
                    assert!(!w_ptr.is_null());
        
                    // Verify initial weight 0.2
                    let first = std::slice::from_raw_parts(w_ptr, w_len)[0];
                    assert_eq!(first.pre, 0);
                    assert_eq!(first.post, 1);
                    assert!((first.weight - 0.2).abs() < 1e-6);
        
                    // Free snapshot buffer
                    hsnn_free_buffer(w_ptr as *mut u8);
        
                    // Apply update: set 0->1 to 0.9
                    let updates = vec![HSNN_WeightTriple { pre: 0, post: 1, weight: 0.9 }];
                    let mut applied: usize = 0;
                    ok(hsnn_network_apply_weight_updates(
                        net_ptr,
                        updates.as_ptr(),
                        updates.len(),
                        &mut applied as *mut _,
                    ));
                    assert_eq!(applied, 1);
        
                    // Snapshot again and verify updated weight
                    let mut w2_ptr: *mut HSNN_WeightTriple = std::ptr::null_mut();
                    let mut w2_len: usize = 0;
                    ok(hsnn_network_snapshot_weights(
                        net_ptr as *const _,
                        &mut w2_ptr as *mut _,
                        &mut w2_len as *mut _,
                    ));
                    assert_eq!(w2_len, 1);
                    let updated = std::slice::from_raw_parts(w2_ptr, w2_len)[0];
                    assert!((updated.weight - 0.9).abs() < 1e-6);
        
                    // Cleanup
                    hsnn_free_buffer(w2_ptr as *mut u8);
                    hsnn_network_free(net_ptr);
                }
            }
        }
    }

    #[test]
    fn c_abi_resource_cleanup_is_safe() {
        unsafe {
            // Create builder and network
            let mut builder_ptr: *mut HSNN_NetworkBuilder = ptr::null_mut();
            ok(hsnn_network_builder_new(&mut builder_ptr as *mut _));
            ok(hsnn_network_builder_add_neuron_range(builder_ptr, 0, 1));

            let mut net_ptr: *mut HSNN_Network = ptr::null_mut();
            ok(hsnn_network_build(builder_ptr, &mut net_ptr as *mut _));
            assert!(!net_ptr.is_null());

            // Explicitly free without running simulation
            hsnn_network_free(net_ptr);
        }
    }
}