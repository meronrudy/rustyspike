# Binary Schema Specifications

## Overview

This document defines the binary schemas used throughout the CLI-first SNN framework. These schemas enable zero-copy operations, efficient storage, and cross-language compatibility while maintaining version stability.

## Design Principles

### Zero-Copy Operations
- All schemas support memory-mapped access
- Aligned data structures for efficient processing
- Minimal parsing overhead for real-time operations

### Version Stability
- Each schema includes version and capability information
- Forward and backward compatibility through capability negotiation
- Clear migration paths between schema versions

### Cross-Platform Compatibility
- Little-endian byte order (consistent with modern processors)
- Explicit padding and alignment specifications
- No platform-specific data types

### Efficiency
- Compact representations for large-scale networks
- Compression support where appropriate
- Batch operations for bulk data processing

## Schema Definitions

### VCSR: Versioned Compressed Sparse Row

The VCSR format stores hypergraph snapshots using compressed sparse row representation, optimized for efficient traversal and memory usage.

```rust
#[repr(C)]
pub struct VCSRHeader {
    magic: [u8; 4],           // "VCSR" (0x52, 0x53, 0x43, 0x56)
    version: u32,             // Schema version (current: 1)
    generation: u64,          // Temporal generation number
    timestamp: u64,           // Creation timestamp (nanoseconds since epoch)
    
    // Graph structure
    num_vertices: u32,        // Number of vertices (neurons)
    num_hyperedges: u32,      // Number of hyperedges
    num_incidences: u64,      // Total number of incidences
    
    // Capability flags
    capabilities: u64,        // Feature flags (see VCSRCapabilities)
    
    // Data layout offsets (from start of file)
    vertex_offset: u64,       // Offset to vertex data
    row_ptr_offset: u64,      // Offset to CSR row pointers
    col_indices_offset: u64,  // Offset to column indices
    weights_offset: u64,      // Offset to weight data
    metadata_offset: u64,     // Offset to metadata section
    
    // Checksums for integrity
    header_checksum: u32,     // CRC32 of header
    data_checksum: u32,       // CRC32 of all data sections
    
    reserved: [u8; 32],       // Reserved for future extensions
}

// Capability flags for VCSR format
pub struct VCSRCapabilities;
impl VCSRCapabilities {
    pub const MMAP_COMPATIBLE: u64 = 1 << 0;     // Memory-mappable
    pub const COMPRESSED_WEIGHTS: u64 = 1 << 1;   // Weight compression
    pub const TEMPORAL_METADATA: u64 = 1 << 2;    // Temporal annotations
    pub const MASK_SUPPORT: u64 = 1 << 3;        // Subview masking
    pub const HYPEREDGE_METADATA: u64 = 1 << 4;  // Extended hyperedge info
    pub const BIDIRECTIONAL: u64 = 1 << 5;       // Bidirectional edges
}

// Vertex data structure
#[repr(C)]
pub struct VCSRVertex {
    id: u32,                  // Vertex ID (NeuronId)
    vertex_type: u8,          // Vertex type (neuron model)
    flags: u8,                // Status flags
    reserved: [u8; 2],        // Alignment padding
}

// CSR structure follows standard layout:
// - row_ptr: Vec<u64> of length (num_vertices + 1)
// - col_indices: Vec<u32> of length num_incidences  
// - weights: Vec<f32> of length num_incidences
```

### VEVT: Event Stream Format

The VEVT format stores temporal event streams including spikes, control events, and system messages.

```rust
#[repr(C)]
pub struct VEVTHeader {
    magic: [u8; 4],           // "VEVT" (0x54, 0x56, 0x45, 0x56)
    version: u32,             // Schema version (current: 1)
    stream_id: u64,           // Unique stream identifier
    
    // Temporal range
    time_start: u64,          // Start time (nanoseconds)
    time_end: u64,            // End time (nanoseconds)
    time_resolution: u32,     // Time resolution (nanoseconds per tick)
    
    // Event counts
    total_events: u64,        // Total number of events
    spike_events: u64,        // Number of spike events
    control_events: u64,      // Number of control events
    
    // Compression and encoding
    compression: u8,          // Compression type (see VEVTCompression)
    encoding: u8,             // Encoding type (see VEVTEncoding)
    flags: u16,               // Stream flags
    
    // Data layout
    events_offset: u64,       // Offset to event data
    index_offset: u64,        // Offset to time index
    metadata_offset: u64,     // Offset to metadata
    
    // Integrity
    header_checksum: u32,     // CRC32 of header
    data_checksum: u32,       // CRC32 of event data
    
    reserved: [u8; 24],       // Reserved space
}

// Compression types
pub struct VEVTCompression;
impl VEVTCompression {
    pub const NONE: u8 = 0;
    pub const LZ4: u8 = 1;
    pub const ZSTD: u8 = 2;
    pub const CUSTOM: u8 = 255;
}

// Event encoding types
pub struct VEVTEncoding;
impl VEVTEncoding {
    pub const BINARY: u8 = 0;        // Raw binary events
    pub const DELTA_COMPRESSED: u8 = 1;  // Delta compression
    pub const RLE: u8 = 2;           // Run-length encoding
}

// Base event structure
#[repr(C)]
pub struct VEVTEvent {
    timestamp: u64,           // Event timestamp (nanoseconds)
    event_type: u8,           // Event type (see VEVTEventType)
    source_id: u32,           // Source neuron/entity ID
    target_id: u32,           // Target neuron/entity ID (if applicable)
    payload_size: u16,        // Size of additional payload
    payload: u8,              // Start of variable-length payload
}

// Event types
pub struct VEVTEventType;
impl VEVTEventType {
    pub const SPIKE: u8 = 0;          // Neural spike
    pub const PHASE_ENTER: u8 = 1;    // TTR phase entry
    pub const PHASE_EXIT: u8 = 2;     // TTR phase exit
    pub const NEUROMOD: u8 = 3;       // Neuromodulation event
    pub const REWARD: u8 = 4;         // Reward signal
    pub const CONTROL: u8 = 5;        // System control event
    pub const MARKER: u8 = 6;         // Temporal marker
}
```

### VMSK: Mask Format

The VMSK format stores bitmasks for subviews, TTR modules, and selective operations.

```rust
#[repr(C)]
pub struct VMSKHeader {
    magic: [u8; 4],           // "VMSK" (0x4B, 0x53, 0x4D, 0x56)
    version: u32,             // Schema version (current: 1)
    mask_id: u32,             // Unique mask identifier
    generation: u64,          // Associated generation
    
    // Mask properties
    total_bits: u64,          // Total number of bits in mask
    active_bits: u64,         // Number of active (set) bits
    mask_type: u8,            // Mask type (see VMSKType)
    compression: u8,          // Compression type
    flags: u16,               // Mask flags
    
    // Data layout
    bitmap_offset: u64,       // Offset to bitmap data
    index_offset: u64,        // Offset to sparse index (if available)
    metadata_offset: u64,     // Offset to metadata
    
    // Integrity
    header_checksum: u32,     // CRC32 of header
    data_checksum: u32,       // CRC32 of mask data
    
    reserved: [u8; 20],       // Reserved space
}

// Mask types
pub struct VMSKType;
impl VMSKType {
    pub const VERTEX_MASK: u8 = 0;       // Vertex/neuron selection
    pub const EDGE_MASK: u8 = 1;         // Edge selection
    pub const MODULE_MASK: u8 = 2;       // TTR module mask
    pub const TEMPORAL_MASK: u8 = 3;     // Temporal window mask
    pub const ACTIVITY_MASK: u8 = 4;     // Activity-based mask
}

// Bitmap data follows header as packed u64 array
// Optional sparse index for very sparse masks
#[repr(C)]
pub struct VMSKSparseIndex {
    num_runs: u32,            // Number of run-length encoded runs
    runs_offset: u64,         // Offset to run data
}

#[repr(C)]
pub struct VMSKRun {
    start_bit: u64,           // Starting bit position
    length: u32,              // Length of run
    value: u8,                // Run value (0 or 1)
    reserved: [u8; 3],        // Alignment
}
```

### VMORF: Morphology Operations Log

The VMORF format stores topology modification operations for TTR and dynamic network evolution.

```rust
#[repr(C)]
pub struct VMORFHeader {
    magic: [u8; 4],           // "MORF" (0x46, 0x52, 0x4F, 0x4D)
    version: u32,             // Schema version (current: 1)
    log_id: u64,              // Unique log identifier
    
    // Temporal range
    time_start: u64,          // First operation timestamp
    time_end: u64,            // Last operation timestamp
    generation_start: u64,    // Starting generation
    generation_end: u64,      // Ending generation
    
    // Operation counts
    total_operations: u64,    // Total number of operations
    add_operations: u32,      // Number of add operations
    remove_operations: u32,   // Number of remove operations
    modify_operations: u32,   // Number of modify operations
    
    // Data layout
    operations_offset: u64,   // Offset to operations data
    index_offset: u64,        // Offset to temporal index
    metadata_offset: u64,     // Offset to metadata
    
    // Integrity
    header_checksum: u32,     // CRC32 of header
    data_checksum: u32,       // CRC32 of operations data
    
    reserved: [u8; 16],       // Reserved space
}

// Morphology operation
#[repr(C)]
pub struct VMORFOperation {
    timestamp: u64,           // Operation timestamp
    generation: u64,          // Target generation
    op_type: u8,              // Operation type (see VMORFOpType)
    entity_type: u8,          // Entity type (vertex/edge)
    flags: u16,               // Operation flags
    
    // Entity identifiers
    primary_id: u32,          // Primary entity ID
    secondary_id: u32,        // Secondary entity ID (for edges)
    
    // Operation data
    data_size: u16,           // Size of operation data
    reserved: [u8; 6],        // Alignment
    // Variable-length operation data follows
}

// Operation types
pub struct VMORFOpType;
impl VMORFOpType {
    pub const ADD_VERTEX: u8 = 0;        // Add vertex
    pub const REMOVE_VERTEX: u8 = 1;     // Remove vertex
    pub const ADD_EDGE: u8 = 2;          // Add edge/hyperedge
    pub const REMOVE_EDGE: u8 = 3;       // Remove edge/hyperedge
    pub const MODIFY_WEIGHT: u8 = 4;     // Modify edge weight
    pub const MODIFY_VERTEX: u8 = 5;     // Modify vertex properties
    pub const BATCH_OP: u8 = 6;          // Batch operation
}
```

### VGRF: Graph Frame for Visualization

The VGRF format stores pre-computed graph layout and rendering data for efficient visualization.

```rust
#[repr(C)]
pub struct VGRFHeader {
    magic: [u8; 4],           // "GRAF" (0x46, 0x41, 0x52, 0x47)
    version: u32,             // Schema version (current: 1)
    frame_id: u64,            // Unique frame identifier
    generation: u64,          // Source generation
    timestamp: u64,           // Frame timestamp
    
    // Rendering properties
    render_mode: u8,          // Render mode (see VGRFRenderMode)
    layout_type: u8,          // Layout algorithm used
    lod_level: u8,            // Level of detail
    flags: u8,                // Rendering flags
    
    // Graph properties
    num_vertices: u32,        // Number of vertices in frame
    num_edges: u32,           // Number of edges in frame
    num_segments: u32,        // Number of line segments
    
    // Viewport and transforms
    viewport_width: f32,      // Viewport width
    viewport_height: f32,     // Viewport height
    transform_matrix: [f32; 16], // 4x4 transform matrix
    
    // Data layout
    vertices_offset: u64,     // Offset to vertex data
    edges_offset: u64,        // Offset to edge data
    segments_offset: u64,     // Offset to line segments
    colors_offset: u64,       // Offset to color data
    metadata_offset: u64,     // Offset to metadata
    
    // Integrity
    header_checksum: u32,     // CRC32 of header
    data_checksum: u32,       // CRC32 of frame data
    
    reserved: [u8; 8],        // Reserved space
}

// Render modes
pub struct VGRFRenderMode;
impl VGRFRenderMode {
    pub const STRUCTURAL: u8 = 0;        // Structural graph view
    pub const INCIDENCE: u8 = 1;         // Incidence graph view
    pub const STAR: u8 = 2;              // Star graph view
    pub const BIPARTITE: u8 = 3;         // Bipartite layout
}

// Vertex data for rendering
#[repr(C)]
pub struct VGRFVertex {
    id: u32,                  // Vertex ID
    x: f32,                   // X coordinate
    y: f32,                   // Y coordinate
    z: f32,                   // Z coordinate (for 3D)
    size: f32,                // Vertex size
    color: u32,               // RGBA color (packed)
    flags: u16,               // Rendering flags
    vertex_type: u8,          // Vertex type
    reserved: u8,             // Alignment
}

// Edge data for rendering
#[repr(C)]
pub struct VGRFEdge {
    id: u32,                  // Edge ID
    source: u32,              // Source vertex ID
    target: u32,              // Target vertex ID
    weight: f32,              // Edge weight
    color: u32,               // RGBA color (packed)
    thickness: f32,           // Line thickness
    flags: u16,               // Rendering flags
    edge_type: u8,            // Edge type
    reserved: u8,             // Alignment
}

// Line segments for efficient rendering
#[repr(C)]
pub struct VGRFSegment {
    x1: f32, y1: f32,         // Start point
    x2: f32, y2: f32,         // End point
    color: u32,               // RGBA color (packed)
    thickness: f32,           // Line thickness
    flags: u16,               // Rendering flags
    reserved: [u8; 2],        // Alignment
}
```

### VRAS: Raster Format for Temporal Visualization

The VRAS format stores raster and heatmap data for temporal spike visualization.

```rust
#[repr(C)]
pub struct VRASHeader {
    magic: [u8; 4],           // "RAST" (0x54, 0x53, 0x41, 0x52)
    version: u32,             // Schema version (current: 1)
    raster_id: u64,           // Unique raster identifier
    generation: u64,          // Source generation
    
    // Temporal properties
    time_start: u64,          // Start time (nanoseconds)
    time_end: u64,            // End time (nanoseconds)
    time_resolution: u32,     // Time bin size (nanoseconds)
    
    // Raster dimensions
    num_neurons: u32,         // Number of neurons (rows)
    num_time_bins: u32,       // Number of time bins (columns)
    data_type: u8,            // Data type (see VRASDataType)
    compression: u8,          // Compression type
    flags: u16,               // Raster flags
    
    // Color mapping
    color_map: u8,            // Color map type
    value_min: f32,           // Minimum value for color mapping
    value_max: f32,           // Maximum value for color mapping
    
    // Data layout
    raster_offset: u64,       // Offset to raster data
    neuron_map_offset: u64,   // Offset to neuron ID mapping
    time_map_offset: u64,     // Offset to time mapping
    metadata_offset: u64,     // Offset to metadata
    
    // Integrity
    header_checksum: u32,     // CRC32 of header
    data_checksum: u32,       // CRC32 of raster data
    
    reserved: [u8; 12],       // Reserved space
}

// Data types
pub struct VRASDataType;
impl VRASDataType {
    pub const BINARY: u8 = 0;           // Binary spike raster
    pub const COUNT: u8 = 1;            // Spike count per bin
    pub const RATE: u8 = 2;             // Firing rate
    pub const MEMBRANE_V: u8 = 3;       // Membrane potential
    pub const WEIGHT: u8 = 4;           // Synaptic weights
    pub const ACTIVITY: u8 = 5;         // Activity level
}

// Neuron mapping (for sparse rasters)
#[repr(C)]
pub struct VRASNeuronMap {
    neuron_id: u32,           // Original neuron ID
    row_index: u32,           // Row in raster
}

// Time mapping (for non-uniform bins)
#[repr(C)]
pub struct VRASTimeMap {
    timestamp: u64,           // Actual timestamp
    bin_index: u32,           // Bin in raster
    reserved: u32,            // Alignment
}
```

## Implementation Guidelines

### Memory Layout
- All structures use `#[repr(C)]` for stable layout
- 64-bit alignment for optimal performance
- Reserved fields for future extensions
- Explicit padding to avoid compiler surprises

### Versioning Strategy
- Major version changes for incompatible changes
- Minor version changes for backward-compatible additions
- Capability flags for optional features
- Clear migration documentation

### Error Handling
- CRC32 checksums for data integrity
- Magic number validation
- Version compatibility checks
- Graceful degradation for unsupported features

### Performance Considerations
- Memory-mapped file support
- Batch processing interfaces
- Compression where beneficial
- Cache-friendly data access patterns

## Usage Examples

### Creating a VCSR Snapshot
```rust
use shnn_storage::vcsr::*;

let snapshot = VCSRSnapshot::new()
    .with_generation(gen_id)
    .with_capabilities(VCSRCapabilities::MMAP_COMPATIBLE | VCSRCapabilities::MASK_SUPPORT)
    .from_hypergraph(&graph)?;

snapshot.write_to_file("network.vcsr")?;
```

### Reading Event Stream
```rust
use shnn_storage::vevt::*;

let stream = VEVTStream::open("events.vevt")?;
for event in stream.time_range(start_time, end_time)? {
    match event.event_type {
        VEVTEventType::SPIKE => handle_spike(&event),
        VEVTEventType::PHASE_ENTER => handle_phase_enter(&event),
        _ => {}
    }
}
```

### Applying Masks
```rust
use shnn_storage::vmsk::*;

let mask = VMSKMask::load("active_neurons.vmsk")?;
let subview = snapshot.apply_mask(&mask)?;
// Work with filtered subview
```

This binary schema system provides the foundation for efficient, portable, and version-stable storage throughout the CLI-first SNN framework.