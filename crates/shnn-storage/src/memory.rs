//! In-memory storage backend

use crate::{
    error::{Result, StorageError},
    ids::GenerationId,
    traits::{GraphStats, HypergraphSnapshot, HypergraphStore, HypergraphSubview, MorphologyOp},
    vcsr::VCSRSnapshot,
    NeuronId, HyperedgeId, Time,
};

use std::collections::BTreeMap;

/// In-memory hypergraph storage implementation
#[derive(Debug)]
pub struct MemoryStore {
    /// Stored snapshots indexed by generation
    snapshots: BTreeMap<GenerationId, VCSRSnapshot>,
    /// Next generation ID to assign
    next_generation: GenerationId,
}

impl MemoryStore {
    /// Create a new empty memory store
    pub fn new() -> Self {
        Self {
            snapshots: BTreeMap::new(),
            next_generation: GenerationId::new(1),
        }
    }
    
    /// Add a snapshot to the store
    pub fn add_snapshot(&mut self, generation: GenerationId, snapshot: VCSRSnapshot) {
        self.snapshots.insert(generation, snapshot);
        if generation >= self.next_generation {
            self.next_generation = GenerationId::new(generation.raw() + 1);
        }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl HypergraphStore for MemoryStore {
    type Snapshot = MemorySnapshot;
    type Error = StorageError;
    
    fn get_snapshot(&self, generation: GenerationId) -> Result<Self::Snapshot> {
        let snapshot = self.snapshots.get(&generation)
            .ok_or(StorageError::GenerationNotFound { 
                generation: generation.raw() 
            })?;
        
        Ok(MemorySnapshot {
            inner: snapshot.clone(),
        })
    }
    
    fn latest_generation(&self) -> Result<GenerationId> {
        self.snapshots.keys()
            .next_back()
            .copied()
            .ok_or(StorageError::invalid_format("No generations available"))
    }
    
    fn list_generations(
        &self, 
        start: Option<GenerationId>, 
        end: Option<GenerationId>
    ) -> Result<Vec<GenerationId>> {
        let start_gen = start.unwrap_or(GenerationId::new(0));
        let end_gen = end.unwrap_or(GenerationId::new(u64::MAX));
        
        Ok(self.snapshots.keys()
            .filter(|&&gen| gen >= start_gen && gen <= end_gen)
            .copied()
            .collect())
    }
    
    fn create_generation(
        &mut self, 
        base: GenerationId, 
        operations: &[MorphologyOp]
    ) -> Result<GenerationId> {
        let base_snapshot = self.snapshots.get(&base)
            .ok_or(StorageError::GenerationNotFound { 
                generation: base.raw() 
            })?;
        
        let mut new_snapshot = base_snapshot.clone();
        new_snapshot.header.generation = self.next_generation.raw();
        
        // Apply operations (simplified for now)
        for op in operations {
            match op {
                MorphologyOp::AddVertex { id, properties } => {
                    new_snapshot.add_vertex(crate::vcsr::VCSRVertex::new(*id, properties.vertex_type));
                }
                MorphologyOp::AddHyperedge { vertices, weight, .. } => {
                    // For simple edges, just connect first two vertices
                    if vertices.len() >= 2 {
                        new_snapshot.add_edge(vertices[0], vertices[1], *weight);
                    }
                }
                _ => {
                    // TODO: Implement other operations
                }
            }
        }
        
        new_snapshot.finalize();
        
        let new_gen = self.next_generation;
        self.snapshots.insert(new_gen, new_snapshot);
        self.next_generation = GenerationId::new(new_gen.raw() + 1);
        
        Ok(new_gen)
    }
    
    fn compact(&mut self, keep_generations: &[GenerationId]) -> Result<()> {
        let keep_set: std::collections::HashSet<_> = keep_generations.iter().collect();
        self.snapshots.retain(|gen, _| keep_set.contains(gen));
        Ok(())
    }
}

/// Memory-based snapshot implementation
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    inner: VCSRSnapshot,
}

impl HypergraphSnapshot for MemorySnapshot {
    type Subview = MemorySubview;
    type NeighborIter = MemoryNeighborIter;
    type HyperedgeIter = MemoryHyperedgeIter;
    
    fn stats(&self) -> GraphStats {
        GraphStats {
            num_neurons: self.inner.header.num_vertices,
            num_hyperedges: self.inner.header.num_hyperedges,
            num_incidences: self.inner.header.num_incidences,
            avg_degree: if self.inner.header.num_vertices > 0 {
                self.inner.header.num_incidences as f32 / self.inner.header.num_vertices as f32
            } else {
                0.0
            },
            max_degree: self.inner.vertices.len() as u32, // Simplified
            density: 0.0, // TODO: Calculate actual density
            generation: GenerationId::new(self.inner.header.generation),
            timestamp: Time::from_nanos(self.inner.header.timestamp),
        }
    }
    
    fn neighbors(&self, neuron: NeuronId) -> Result<Self::NeighborIter> {
        let neighbors: Vec<_> = self.inner.neighbors(neuron).collect();
        Ok(MemoryNeighborIter {
            iter: neighbors.into_iter(),
        })
    }
    
    fn hyperedges(&self, _neuron: NeuronId) -> Result<Self::HyperedgeIter> {
        // Simplified implementation - no actual hyperedges for now
        Ok(MemoryHyperedgeIter {
            iter: Vec::new().into_iter(),
        })
    }
    
    fn apply_mask(&self, _mask: &dyn crate::traits::Mask) -> Result<Self::Subview> {
        // Simplified implementation
        Ok(MemorySubview {
            active_neurons: self.inner.vertices.iter()
                .map(|v| v.neuron_id())
                .collect(),
            active_hyperedges: Vec::new(),
            stats: self.stats(),
        })
    }
    
    fn k_hop(&self, seeds: &[NeuronId], k: u32) -> Result<Self::Subview> {
        let mut visited = std::collections::HashSet::new();
        let mut current_level = seeds.to_vec();
        
        for _ in 0..k {
            let mut next_level = Vec::new();
            for &neuron in &current_level {
                if visited.insert(neuron) {
                    let neighbors: Vec<_> = self.inner.neighbors(neuron)
                        .map(|(neighbor, _)| neighbor)
                        .collect();
                    next_level.extend(neighbors);
                }
            }
            current_level = next_level;
        }
        
        Ok(MemorySubview {
            active_neurons: visited.into_iter().collect(),
            active_hyperedges: Vec::new(),
            stats: self.stats(),
        })
    }
    
    fn has_edge(&self, source: NeuronId, target: NeuronId) -> bool {
        self.inner.neighbors(source)
            .any(|(neighbor, _)| neighbor == target)
    }
    
    fn edge_weight(&self, source: NeuronId, target: NeuronId) -> Option<f32> {
        self.inner.neighbors(source)
            .find(|(neighbor, _)| *neighbor == target)
            .map(|(_, weight)| weight)
    }
}

/// Memory-based neighbor iterator
pub struct MemoryNeighborIter {
    iter: std::vec::IntoIter<(NeuronId, f32)>,
}

impl Iterator for MemoryNeighborIter {
    type Item = (NeuronId, f32);
    
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

/// Memory-based hyperedge iterator  
pub struct MemoryHyperedgeIter {
    iter: std::vec::IntoIter<(HyperedgeId, Vec<NeuronId>, f32)>,
}

impl Iterator for MemoryHyperedgeIter {
    type Item = (HyperedgeId, Vec<NeuronId>, f32);
    
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

/// Memory-based subview implementation
pub struct MemorySubview {
    active_neurons: Vec<NeuronId>,
    active_hyperedges: Vec<HyperedgeId>,
    stats: GraphStats,
}

impl HypergraphSubview for MemorySubview {
    fn active_neurons(&self) -> &[NeuronId] {
        &self.active_neurons
    }
    
    fn active_hyperedges(&self) -> &[HyperedgeId] {
        &self.active_hyperedges
    }
    
    fn stats(&self) -> GraphStats {
        self.stats.clone()
    }
    
    fn export_vcsr(&self) -> Result<Vec<u8>> {
        // TODO: Implement VCSR export for subview
        Ok(Vec::new())
    }
    
    fn export_graphml(&self) -> Result<String> {
        // TODO: Implement GraphML export
        Ok(String::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{traits::VertexProperties, vcsr::VCSRVertex};

    #[test]
    fn test_memory_store() {
        let mut store = MemoryStore::new();
        
        // Create a simple snapshot
        let mut snapshot = VCSRSnapshot::new(GenerationId::new(1), 2);
        snapshot.add_vertex(VCSRVertex::new(NeuronId::new(0), 1));
        snapshot.add_vertex(VCSRVertex::new(NeuronId::new(1), 1));
        snapshot.add_edge(NeuronId::new(0), NeuronId::new(1), 0.5);
        snapshot.finalize();
        
        store.add_snapshot(GenerationId::new(1), snapshot);
        
        // Test retrieval
        let retrieved = store.get_snapshot(GenerationId::new(1)).unwrap();
        let stats = retrieved.stats();
        assert_eq!(stats.num_neurons, 2);
        assert_eq!(stats.num_incidences, 1);
        
        // Test neighbors
        let neighbors: Vec<_> = retrieved.neighbors(NeuronId::new(0)).unwrap().collect();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], (NeuronId::new(1), 0.5));
    }

    #[test]
    fn test_generation_operations() {
        let mut store = MemoryStore::new();
        
        // Create initial snapshot
        let mut snapshot = VCSRSnapshot::new(GenerationId::new(1), 1);
        snapshot.add_vertex(VCSRVertex::new(NeuronId::new(0), 1));
        snapshot.finalize();
        
        store.add_snapshot(GenerationId::new(1), snapshot);
        
        // Create new generation with operations
        let operations = vec![
            MorphologyOp::AddVertex {
                id: NeuronId::new(1),
                properties: VertexProperties {
                    vertex_type: 1,
                    flags: 0,
                },
            },
        ];
        
        let new_gen = store.create_generation(GenerationId::new(1), &operations).unwrap();
        assert_eq!(new_gen, GenerationId::new(2));
        
        let new_snapshot = store.get_snapshot(new_gen).unwrap();
        let stats = new_snapshot.stats();
        assert_eq!(stats.num_neurons, 2);
    }
}