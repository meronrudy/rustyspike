//! File-based storage backend

use crate::{
    error::{Result, StorageError},
    ids::GenerationId,
    memory::MemoryStore,
    traits::{HypergraphSnapshot, HypergraphStore, MorphologyOp},
};

use std::path::Path;

/// File-based hypergraph storage implementation
pub struct FileStore {
    /// In-memory cache
    cache: MemoryStore,
    /// Base directory for storage
    base_dir: std::path::PathBuf,
}

impl FileStore {
    /// Create a new file-based store
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_dir)?;
        
        Ok(Self {
            cache: MemoryStore::new(),
            base_dir,
        })
    }
    
    /// Get the path for a generation file
    fn generation_path(&self, generation: GenerationId) -> std::path::PathBuf {
        self.base_dir.join(format!("gen_{:016x}.vcsr", generation.raw()))
    }
}

impl HypergraphStore for FileStore {
    type Snapshot = <MemoryStore as HypergraphStore>::Snapshot;
    type Error = StorageError;
    
    fn get_snapshot(&self, generation: GenerationId) -> Result<Self::Snapshot> {
        // Try cache first
        if let Ok(snapshot) = self.cache.get_snapshot(generation) {
            return Ok(snapshot);
        }
        
        // TODO: Load from file
        Err(StorageError::GenerationNotFound { 
            generation: generation.raw() 
        })
    }
    
    fn latest_generation(&self) -> Result<GenerationId> {
        self.cache.latest_generation()
    }
    
    fn list_generations(
        &self, 
        start: Option<GenerationId>, 
        end: Option<GenerationId>
    ) -> Result<Vec<GenerationId>> {
        self.cache.list_generations(start, end)
    }
    
    fn create_generation(
        &mut self, 
        base: GenerationId, 
        operations: &[MorphologyOp]
    ) -> Result<GenerationId> {
        self.cache.create_generation(base, operations)
    }
    
    fn compact(&mut self, keep_generations: &[GenerationId]) -> Result<()> {
        self.cache.compact(keep_generations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_store_creation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let store = FileStore::new(temp_dir.path()).unwrap();
        assert!(temp_dir.path().exists());
    }
}