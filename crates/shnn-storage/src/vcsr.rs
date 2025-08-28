//! VCSR (Versioned Compressed Sparse Row) format implementation

use crate::{
    error::{Result, StorageError},
    ids::GenerationId,
    magic,
    schemas::{calculate_checksum, current_timestamp, validate_checksum, validate_magic},
    NeuronId, HyperedgeId, Time,
};

use core::mem;
use std::io::{Read, Write};

/// VCSR format header
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VCSRHeader {
    /// Magic number "VCSR"
    pub magic: [u8; 4],
    /// Schema version (current: 1)
    pub version: u32,
    /// Temporal generation number
    pub generation: u64,
    /// Creation timestamp (nanoseconds since epoch)
    pub timestamp: u64,
    
    // Graph structure
    /// Number of vertices (neurons)
    pub num_vertices: u32,
    /// Number of hyperedges
    pub num_hyperedges: u32,
    /// Total number of incidences
    pub num_incidences: u64,
    
    // Capability flags
    /// Feature flags (see VCSRCapabilities)
    pub capabilities: u64,
    
    // Data layout offsets (from start of file)
    /// Offset to vertex data
    pub vertex_offset: u64,
    /// Offset to CSR row pointers
    pub row_ptr_offset: u64,
    /// Offset to column indices
    pub col_indices_offset: u64,
    /// Offset to weight data
    pub weights_offset: u64,
    /// Offset to metadata section
    pub metadata_offset: u64,
    
    // Checksums for integrity
    /// CRC32 of header
    pub header_checksum: u32,
    /// CRC32 of all data sections
    pub data_checksum: u32,
    
    /// Reserved for future extensions
    pub reserved: [u8; 32],
}

impl VCSRHeader {
    /// Create a new VCSR header
    pub fn new(generation: GenerationId, num_vertices: u32, num_hyperedges: u32) -> Self {
        Self {
            magic: magic::VCSR,
            version: 1,
            generation: generation.raw(),
            timestamp: current_timestamp(),
            num_vertices,
            num_hyperedges,
            num_incidences: 0, // Will be updated when data is added
            capabilities: VCSRCapabilities::MMAP_COMPATIBLE,
            vertex_offset: 0,
            row_ptr_offset: 0,
            col_indices_offset: 0,
            weights_offset: 0,
            metadata_offset: 0,
            header_checksum: 0, // Will be calculated later
            data_checksum: 0,   // Will be calculated later
            reserved: [0; 32],
        }
    }
    
    /// Validate this header
    pub fn validate(&self) -> Result<()> {
        validate_magic(&self.magic, magic::VCSR)?;
        
        if self.version != 1 {
            return Err(StorageError::UnsupportedVersion {
                version: self.version,
                supported: 1,
            });
        }
        
        Ok(())
    }
    
    /// Calculate and update the header checksum
    pub fn update_header_checksum(&mut self) {
        // Temporarily set checksum to 0
        let original_checksum = self.header_checksum;
        self.header_checksum = 0;
        
        // Calculate checksum over the header
        let header_bytes = unsafe {
            core::slice::from_raw_parts(
                self as *const Self as *const u8,
                mem::size_of::<Self>(),
            )
        };
        
        self.header_checksum = calculate_checksum(header_bytes);
    }
    
    /// Verify the header checksum
    pub fn verify_header_checksum(&self) -> Result<()> {
        let mut temp_header = self.clone();
        let expected_checksum = temp_header.header_checksum;
        temp_header.header_checksum = 0;
        
        let header_bytes = unsafe {
            core::slice::from_raw_parts(
                &temp_header as *const Self as *const u8,
                mem::size_of::<Self>(),
            )
        };
        
        validate_checksum(header_bytes, expected_checksum)
    }
}

/// Capability flags for VCSR format
pub struct VCSRCapabilities;

impl VCSRCapabilities {
    /// Memory-mappable format
    pub const MMAP_COMPATIBLE: u64 = 1 << 0;
    /// Weight compression enabled
    pub const COMPRESSED_WEIGHTS: u64 = 1 << 1;
    /// Temporal metadata available
    pub const TEMPORAL_METADATA: u64 = 1 << 2;
    /// Subview masking supported
    pub const MASK_SUPPORT: u64 = 1 << 3;
    /// Extended hyperedge info
    pub const HYPEREDGE_METADATA: u64 = 1 << 4;
    /// Bidirectional edges
    pub const BIDIRECTIONAL: u64 = 1 << 5;
}

/// Vertex data structure
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VCSRVertex {
    /// Vertex ID (NeuronId)
    pub id: u32,
    /// Vertex type (neuron model)
    pub vertex_type: u8,
    /// Status flags
    pub flags: u8,
    /// Alignment padding
    pub reserved: [u8; 2],
}

impl VCSRVertex {
    /// Create a new vertex
    pub fn new(id: NeuronId, vertex_type: u8) -> Self {
        Self {
            id: id.raw(),
            vertex_type,
            flags: 0,
            reserved: [0; 2],
        }
    }
    
    /// Get the neuron ID
    pub fn neuron_id(&self) -> NeuronId {
        NeuronId::new(self.id)
    }
}

/// VCSR snapshot implementation
#[derive(Debug, Clone)]
pub struct VCSRSnapshot {
    /// Header information
    pub header: VCSRHeader,
    /// Vertex data
    pub vertices: Vec<VCSRVertex>,
    /// CSR row pointers (length: num_vertices + 1)
    pub row_ptr: Vec<u64>,
    /// Column indices (target vertices)
    pub col_indices: Vec<u32>,
    /// Edge weights
    pub weights: Vec<f32>,
}

impl VCSRSnapshot {
    /// Create a new empty VCSR snapshot
    pub fn new(generation: GenerationId, num_vertices: u32) -> Self {
        let header = VCSRHeader::new(generation, num_vertices, 0);
        let mut row_ptr = Vec::with_capacity((num_vertices + 1) as usize);
        // Initialize row pointers to 0
        row_ptr.resize((num_vertices + 1) as usize, 0);
        
        Self {
            header,
            vertices: Vec::new(),
            row_ptr,
            col_indices: Vec::new(),
            weights: Vec::new(),
        }
    }
    
    /// Add a vertex to the snapshot
    pub fn add_vertex(&mut self, vertex: VCSRVertex) {
        self.vertices.push(vertex);
    }
    
    /// Add an edge between vertices
    pub fn add_edge(&mut self, source: NeuronId, target: NeuronId, weight: f32) {
        let source_idx = source.raw() as usize;
        if source_idx >= self.row_ptr.len() - 1 {
            return; // Invalid source vertex
        }
        
        // Find insertion point for this edge
        let start = self.row_ptr[source_idx] as usize;
        let end = self.row_ptr[source_idx + 1] as usize;
        
        // Insert the new edge
        self.col_indices.insert(end, target.raw());
        self.weights.insert(end, weight);
        
        // Update row pointers for all vertices after this one
        for i in (source_idx + 1)..self.row_ptr.len() {
            self.row_ptr[i] += 1;
        }
        
        // Update header
        self.header.num_incidences += 1;
    }
    
    /// Get neighbors of a vertex
    pub fn neighbors(&self, vertex: NeuronId) -> Box<dyn Iterator<Item = (NeuronId, f32)> + '_> {
        let vertex_idx = vertex.raw() as usize;
        if vertex_idx >= self.row_ptr.len() - 1 {
            return Box::new(std::iter::empty());
        }
        
        let start = self.row_ptr[vertex_idx] as usize;
        let end = self.row_ptr[vertex_idx + 1] as usize;
        
        Box::new(
            self.col_indices[start..end]
                .iter()
                .zip(&self.weights[start..end])
                .map(|(&target, &weight)| (NeuronId::new(target), weight))
        )
    }
    
    /// Finalize the snapshot and update checksums
    pub fn finalize(&mut self) {
        // Update header with final counts
        self.header.num_vertices = self.vertices.len() as u32;
        self.header.num_incidences = self.col_indices.len() as u64;
        
        // Calculate data checksum
        let mut data_to_hash = Vec::new();
        
        // Add vertices
        for vertex in &self.vertices {
            data_to_hash.extend_from_slice(unsafe {
                core::slice::from_raw_parts(
                    vertex as *const VCSRVertex as *const u8,
                    mem::size_of::<VCSRVertex>(),
                )
            });
        }
        
        // Add row pointers
        for &ptr in &self.row_ptr {
            data_to_hash.extend_from_slice(&ptr.to_le_bytes());
        }
        
        // Add column indices
        for &idx in &self.col_indices {
            data_to_hash.extend_from_slice(&idx.to_le_bytes());
        }
        
        // Add weights
        for &weight in &self.weights {
            data_to_hash.extend_from_slice(&weight.to_le_bytes());
        }
        
        self.header.data_checksum = calculate_checksum(&data_to_hash);
        self.header.update_header_checksum();
    }
    
    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        
        // Write header
        let header_bytes = unsafe {
            core::slice::from_raw_parts(
                &self.header as *const VCSRHeader as *const u8,
                mem::size_of::<VCSRHeader>(),
            )
        };
        bytes.extend_from_slice(header_bytes);
        
        // Write vertices
        for vertex in &self.vertices {
            let vertex_bytes = unsafe {
                core::slice::from_raw_parts(
                    vertex as *const VCSRVertex as *const u8,
                    mem::size_of::<VCSRVertex>(),
                )
            };
            bytes.extend_from_slice(vertex_bytes);
        }
        
        // Write row pointers
        for &ptr in &self.row_ptr {
            bytes.extend_from_slice(&ptr.to_le_bytes());
        }
        
        // Write column indices
        for &idx in &self.col_indices {
            bytes.extend_from_slice(&idx.to_le_bytes());
        }
        
        // Write weights
        for &weight in &self.weights {
            bytes.extend_from_slice(&weight.to_le_bytes());
        }
        
        bytes
    }
    
    /// Load from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < mem::size_of::<VCSRHeader>() {
            return Err(StorageError::invalid_format("Data too short for header"));
        }
        
        // Parse header
        let header = unsafe {
            *(data.as_ptr() as *const VCSRHeader)
        };
        header.validate()?;
        header.verify_header_checksum()?;
        
        let mut offset = mem::size_of::<VCSRHeader>();
        
        // Parse vertices
        let vertices_size = header.num_vertices as usize * mem::size_of::<VCSRVertex>();
        if offset + vertices_size > data.len() {
            return Err(StorageError::invalid_format("Data too short for vertices"));
        }
        
        let mut vertices = Vec::with_capacity(header.num_vertices as usize);
        for i in 0..header.num_vertices as usize {
            let vertex_offset = offset + i * mem::size_of::<VCSRVertex>();
            let vertex = unsafe {
                *(data[vertex_offset..].as_ptr() as *const VCSRVertex)
            };
            vertices.push(vertex);
        }
        offset += vertices_size;
        
        // Parse row pointers
        let row_ptr_size = (header.num_vertices as usize + 1) * mem::size_of::<u64>();
        if offset + row_ptr_size > data.len() {
            return Err(StorageError::invalid_format("Data too short for row pointers"));
        }
        
        let mut row_ptr = Vec::with_capacity(header.num_vertices as usize + 1);
        for i in 0..=header.num_vertices as usize {
            let ptr_offset = offset + i * mem::size_of::<u64>();
            let ptr = u64::from_le_bytes([
                data[ptr_offset],
                data[ptr_offset + 1],
                data[ptr_offset + 2],
                data[ptr_offset + 3],
                data[ptr_offset + 4],
                data[ptr_offset + 5],
                data[ptr_offset + 6],
                data[ptr_offset + 7],
            ]);
            row_ptr.push(ptr);
        }
        offset += row_ptr_size;
        
        // Parse column indices
        let col_indices_size = header.num_incidences as usize * mem::size_of::<u32>();
        if offset + col_indices_size > data.len() {
            return Err(StorageError::invalid_format("Data too short for column indices"));
        }
        
        let mut col_indices = Vec::with_capacity(header.num_incidences as usize);
        for i in 0..header.num_incidences as usize {
            let idx_offset = offset + i * mem::size_of::<u32>();
            let idx = u32::from_le_bytes([
                data[idx_offset],
                data[idx_offset + 1],
                data[idx_offset + 2],
                data[idx_offset + 3],
            ]);
            col_indices.push(idx);
        }
        offset += col_indices_size;
        
        // Parse weights
        let weights_size = header.num_incidences as usize * mem::size_of::<f32>();
        if offset + weights_size > data.len() {
            return Err(StorageError::invalid_format("Data too short for weights"));
        }
        
        let mut weights = Vec::with_capacity(header.num_incidences as usize);
        for i in 0..header.num_incidences as usize {
            let weight_offset = offset + i * mem::size_of::<f32>();
            let weight = f32::from_le_bytes([
                data[weight_offset],
                data[weight_offset + 1],
                data[weight_offset + 2],
                data[weight_offset + 3],
            ]);
            weights.push(weight);
        }
        
        Ok(Self {
            header,
            vertices,
            row_ptr,
            col_indices,
            weights,
        })
    }
}

trait EitherIterator<L, R> {
    type Iter: Iterator;
    fn left_iter(self) -> Self::Iter;
    fn right_iter(self) -> Self::Iter;
}

impl<T, L, R> EitherIterator<L, R> for std::iter::Empty<T>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Iter = std::iter::Empty<T>;
    
    fn left_iter(self) -> Self::Iter {
        self
    }
    
    fn right_iter(self) -> Self::Iter {
        self
    }
}

impl<T, L, R> EitherIterator<L, R> for std::iter::Map<std::iter::Zip<std::slice::Iter<'_, u32>, std::slice::Iter<'_, f32>>, fn((&u32, &f32)) -> (NeuronId, f32)>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Iter = Self;
    
    fn left_iter(self) -> Self::Iter {
        self
    }
    
    fn right_iter(self) -> Self::Iter {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vcsr_header() {
        let mut header = VCSRHeader::new(GenerationId::new(1), 100, 500);
        header.update_header_checksum();
        
        assert_eq!(header.magic, magic::VCSR);
        assert_eq!(header.version, 1);
        assert_eq!(header.generation, 1);
        assert_eq!(header.num_vertices, 100);
        assert_eq!(header.num_hyperedges, 500);
        
        assert!(header.validate().is_ok());
        assert!(header.verify_header_checksum().is_ok());
    }

    #[test]
    fn test_vcsr_vertex() {
        let neuron_id = NeuronId::new(42);
        let vertex = VCSRVertex::new(neuron_id, 1);
        
        assert_eq!(vertex.id, 42);
        assert_eq!(vertex.vertex_type, 1);
        assert_eq!(vertex.neuron_id(), neuron_id);
    }

    #[test]
    fn test_vcsr_snapshot() {
        let mut snapshot = VCSRSnapshot::new(GenerationId::new(1), 3);
        
        // Add vertices
        snapshot.add_vertex(VCSRVertex::new(NeuronId::new(0), 1));
        snapshot.add_vertex(VCSRVertex::new(NeuronId::new(1), 1));
        snapshot.add_vertex(VCSRVertex::new(NeuronId::new(2), 1));
        
        // Add edges
        snapshot.add_edge(NeuronId::new(0), NeuronId::new(1), 0.5);
        snapshot.add_edge(NeuronId::new(0), NeuronId::new(2), 0.8);
        snapshot.add_edge(NeuronId::new(1), NeuronId::new(2), 0.3);
        
        snapshot.finalize();
        
        // Test neighbors
        let neighbors: Vec<_> = snapshot.neighbors(NeuronId::new(0)).collect();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&(NeuronId::new(1), 0.5)));
        assert!(neighbors.contains(&(NeuronId::new(2), 0.8)));
        
        // Test serialization
        let bytes = snapshot.to_bytes();
        let loaded = VCSRSnapshot::from_bytes(&bytes).unwrap();
        
        assert_eq!(loaded.header.num_vertices, 3);
        assert_eq!(loaded.header.num_incidences, 3);
        assert_eq!(loaded.vertices.len(), 3);
    }
}