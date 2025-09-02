#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

template<typename T = void>
struct Option;

/// Opaque network builder handle
struct HSNN_NetworkBuilder {
  Option<NetworkBuilder> inner;
};

/// Opaque network handle
struct HSNN_Network {
  SNNNetwork inner;
};

/// Weight triple used for snapshot/import operations
struct HSNN_WeightTriple {
  uint32_t pre;
  uint32_t post;
  float weight;
};

extern "C" {

/// Create a new network builder handle
int32_t hsnn_network_builder_new(HSNN_NetworkBuilder **out_builder);

/// Add a contiguous range of neurons [start, start+count)
int32_t hsnn_network_builder_add_neuron_range(HSNN_NetworkBuilder *builder,
                                              uint32_t start,
                                              uint32_t count);

/// Add a simple synapse with default delay (1ms)
int32_t hsnn_network_builder_add_synapse_simple(HSNN_NetworkBuilder *builder,
                                                uint32_t pre,
                                                uint32_t post,
                                                float weight);

/// Finalize the builder into a network handle (consumes the builder)
int32_t hsnn_network_build(HSNN_NetworkBuilder *builder, HSNN_Network **out_network);

/// Free a network handle
void hsnn_network_free(HSNN_Network *network);

/// Free a buffer allocated by this library (malloc-based)
void hsnn_free_buffer(uint8_t *ptr);

/// Run a deterministic fixed-step simulation, consuming the network, and return VEVT bytes.
/// On success returns 0 and sets (*out_ptr, *out_len). The network handle is invalidated.
/// On failure, returns an error code and leaves outputs untouched.
int32_t hsnn_run_fixed_step_vevt_consume(HSNN_Network **network_ptr,
                                         uint64_t dt_ns,
                                         uint64_t duration_ns,
                                         uint64_t seed,
                                         uint8_t **out_ptr,
                                         uintptr_t *out_len);

/// Export all synaptic weights as a C array of HSNN_WeightTriple (malloc-allocated).
/// Caller must free with hsnn_free_buffer((uint8_t*)ptr).
int32_t hsnn_network_snapshot_weights(const HSNN_Network *network,
                                      HSNN_WeightTriple **out_ptr,
                                      uintptr_t *out_len);

/// Apply weight updates from a C array of HSNN_WeightTriple.
/// Only existing synapses are updated; missing connections are ignored.
/// Returns 0 on success and sets out_applied to the number of updates applied.
int32_t hsnn_network_apply_weight_updates(HSNN_Network *network,
                                          const HSNN_WeightTriple *updates_ptr,
                                          uintptr_t updates_len,
                                          uintptr_t *out_applied);

} // extern "C"
