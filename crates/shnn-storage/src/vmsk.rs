//! VMSK (Mask) format implementation

use crate::{
    error::{Result, StorageError},
    ids::{GenerationId, MaskId},
    magic,
    schemas::{calculate_checksum, current_timestamp, validate_magic},
    traits::{Mask, MaskType},
};

use core::mem;

/// VMSK format header
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VMSKHeader {
    /// Magic number "VMSK"
    pub magic: [u8; 4],
    /// Schema version (current: 1)
    pub version: u32,
    /// Unique mask identifier
    pub mask_id: u32,
    /// Associated generation
    pub generation: u64,
    
    // Mask properties
    /// Total number of bits in mask
    pub total_bits: u64,
    /// Number of active (set) bits
    pub active_bits: u64,
    /// Mask type
    pub mask_type: u8,
    /// Compression type
    pub compression: u8,
    /// Mask flags
    pub flags: u16,
    
    // Data layout
    /// Offset to bitmap data
    pub bitmap_offset: u64,
    /// Offset to sparse index (if available)
    pub index_offset: u64,
    /// Offset to metadata
    pub metadata_offset: u64,
    
    // Integrity
    /// CRC32 of header
    pub header_checksum: u32,
    /// CRC32 of mask data
    pub data_checksum: u32,
    
    /// Reserved space
    pub reserved: [u8; 20],
}

impl VMSKHeader {
    /// Create a new VMSK header
    pub fn new(mask_id: MaskId, generation: GenerationId, mask_type: MaskType, total_bits: u64) -> Self {
        Self {
            magic: magic::VMSK,
            version: 1,
            mask_id: mask_id.raw(),
            generation: generation.raw(),
            total_bits,
            active_bits: 0,
            mask_type: mask_type_to_u8(mask_type),
            compression: 0,
            flags: 0,
            bitmap_offset: 0,
            index_offset: 0,
            metadata_offset: 0,
            header_checksum: 0,
            data_checksum: 0,
            reserved: [0; 20],
        }
    }
    
    /// Validate this header
    pub fn validate(&self) -> Result<()> {
        validate_magic(&self.magic, magic::VMSK)?;
        
        if self.version != 1 {
            return Err(StorageError::UnsupportedVersion {
                version: self.version,
                supported: 1,
            });
        }
        
        Ok(())
    }
}

/// Convert MaskType to u8
fn mask_type_to_u8(mask_type: MaskType) -> u8 {
    match mask_type {
        MaskType::VertexMask => 0,
        MaskType::EdgeMask => 1,
        MaskType::ModuleMask => 2,
        MaskType::TemporalMask => 3,
        MaskType::ActivityMask => 4,
    }
}

/// Convert u8 to MaskType
fn u8_to_mask_type(value: u8) -> MaskType {
    match value {
        0 => MaskType::VertexMask,
        1 => MaskType::EdgeMask,
        2 => MaskType::ModuleMask,
        3 => MaskType::TemporalMask,
        4 => MaskType::ActivityMask,
        _ => MaskType::VertexMask, // Default fallback
    }
}

/// Simple bitmap-based mask implementation
pub struct BitmapMask {
    mask_id: MaskId,
    mask_type: MaskType,
    generation: GenerationId,
    bitmap: Vec<u64>,
    total_bits: u64,
    active_bits: u64,
}

impl BitmapMask {
    /// Create a new bitmap mask
    pub fn new(mask_id: MaskId, mask_type: MaskType, generation: GenerationId, total_bits: u64) -> Self {
        let bitmap_size = (total_bits + 63) / 64; // Round up to nearest u64
        Self {
            mask_id,
            mask_type,
            generation,
            bitmap: vec![0u64; bitmap_size as usize],
            total_bits,
            active_bits: 0,
        }
    }

    /// Import a VMSK binary into a BitmapMask
    pub fn import_vmsk(bytes: &[u8]) -> Result<Self> {
        use crate::schemas::cast_slice_to_struct;

        if bytes.len() < core::mem::size_of::<VMSKHeader>() {
            return Err(StorageError::InvalidFormat { reason: "VMSK too small".into() });
        }

        let header: &VMSKHeader = unsafe { cast_slice_to_struct(bytes)? };
        header.validate()?;

        // Compute expected bitmap length
        let words = ((header.total_bits + 63) / 64) as usize;
        let header_size = core::mem::size_of::<VMSKHeader>();
        let bitmap_bytes = words * core::mem::size_of::<u64>();

        if bytes.len() < header_size + bitmap_bytes {
            return Err(StorageError::InvalidFormat { reason: "VMSK bitmap truncated".into() });
        }

        let mut mask = BitmapMask::new(
            MaskId::new(header.mask_id),
            u8_to_mask_type(header.mask_type),
            GenerationId::new(header.generation),
            header.total_bits,
        );

        // Copy bitmap words
        for i in 0..words {
            let start = header_size + i * core::mem::size_of::<u64>();
            let end = start + core::mem::size_of::<u64>();
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&bytes[start..end]);
            mask.bitmap[i] = u64::from_le_bytes(arr);
        }

        // Recompute active_bits
        mask.active_bits = mask.bitmap.iter().map(|w| w.count_ones() as u64).sum();

        Ok(mask)
    }
    
    /// Set a bit in the mask
    pub fn set_bit(&mut self, index: u64) -> Result<()> {
        if index >= self.total_bits {
            return Err(StorageError::OutOfBounds {
                index: index as usize,
                max: self.total_bits as usize,
            });
        }
        
        let word_index = (index / 64) as usize;
        let bit_index = index % 64;
        
        let old_word = self.bitmap[word_index];
        self.bitmap[word_index] |= 1u64 << bit_index;
        
        // Update active count if bit was newly set
        if old_word != self.bitmap[word_index] {
            self.active_bits += 1;
        }
        
        Ok(())
    }
    
    /// Clear a bit in the mask
    pub fn clear_bit(&mut self, index: u64) -> Result<()> {
        if index >= self.total_bits {
            return Err(StorageError::OutOfBounds {
                index: index as usize,
                max: self.total_bits as usize,
            });
        }
        
        let word_index = (index / 64) as usize;
        let bit_index = index % 64;
        
        let old_word = self.bitmap[word_index];
        self.bitmap[word_index] &= !(1u64 << bit_index);
        
        // Update active count if bit was cleared
        if old_word != self.bitmap[word_index] {
            self.active_bits -= 1;
        }
        
        Ok(())
    }
    
    /// Get all active indices
    pub fn active_indices(&self) -> Vec<u32> {
        let mut indices = Vec::new();
        
        for (word_idx, &word) in self.bitmap.iter().enumerate() {
            if word != 0 {
                for bit_idx in 0..64 {
                    if word & (1u64 << bit_idx) != 0 {
                        let global_idx = (word_idx * 64 + bit_idx) as u64;
                        if global_idx < self.total_bits {
                            indices.push(global_idx as u32);
                        }
                    }
                }
            }
        }
        
        indices
    }
}

impl Mask for BitmapMask {
    fn mask_id(&self) -> MaskId {
        self.mask_id
    }
    
    fn mask_type(&self) -> MaskType {
        self.mask_type
    }
    
    fn is_active(&self, id: u32) -> bool {
        let index = id as u64;
        if index >= self.total_bits {
            return false;
        }
        
        let word_index = (index / 64) as usize;
        let bit_index = index % 64;
        
        self.bitmap[word_index] & (1u64 << bit_index) != 0
    }
    
    fn active_count(&self) -> u64 {
        self.active_bits
    }
    
    fn total_count(&self) -> u64 {
        self.total_bits
    }
    
    fn intersect(&self, other: &dyn Mask) -> Result<Box<dyn Mask>> {
        // Simplified implementation - assumes same type and size
        if other.mask_type() != self.mask_type || other.total_count() != self.total_bits {
            return Err(StorageError::invalid_format("Incompatible masks for intersection"));
        }
        
        let mut result = BitmapMask::new(
            MaskId::new(0), // Generate new ID
            self.mask_type,
            self.generation,
            self.total_bits,
        );
        
        for i in 0..self.total_bits {
            if self.is_active(i as u32) && other.is_active(i as u32) {
                result.set_bit(i)?;
            }
        }
        
        Ok(Box::new(result))
    }
    
    fn union(&self, other: &dyn Mask) -> Result<Box<dyn Mask>> {
        // Simplified implementation - assumes same type and size
        if other.mask_type() != self.mask_type || other.total_count() != self.total_bits {
            return Err(StorageError::invalid_format("Incompatible masks for union"));
        }
        
        let mut result = BitmapMask::new(
            MaskId::new(0), // Generate new ID
            self.mask_type,
            self.generation,
            self.total_bits,
        );
        
        for i in 0..self.total_bits {
            if self.is_active(i as u32) || other.is_active(i as u32) {
                result.set_bit(i)?;
            }
        }
        
        Ok(Box::new(result))
    }
    
    fn difference(&self, other: &dyn Mask) -> Result<Box<dyn Mask>> {
        // Simplified implementation - assumes same type and size
        if other.mask_type() != self.mask_type || other.total_count() != self.total_bits {
            return Err(StorageError::invalid_format("Incompatible masks for difference"));
        }
        
        let mut result = BitmapMask::new(
            MaskId::new(0), // Generate new ID
            self.mask_type,
            self.generation,
            self.total_bits,
        );
        
        for i in 0..self.total_bits {
            if self.is_active(i as u32) && !other.is_active(i as u32) {
                result.set_bit(i)?;
            }
        }
        
        Ok(Box::new(result))
    }
    
    fn export_vmsk(&self) -> Result<Vec<u8>> {
        let mut header = VMSKHeader::new(self.mask_id, self.generation, self.mask_type, self.total_bits);
        header.active_bits = self.active_bits;
        
        let mut bytes = Vec::new();
        
        // Write header
        let header_bytes = unsafe {
            core::slice::from_raw_parts(
                &header as *const VMSKHeader as *const u8,
                mem::size_of::<VMSKHeader>(),
            )
        };
        bytes.extend_from_slice(header_bytes);
        
        // Write bitmap data
        for &word in &self.bitmap {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vmsk_header() {
        let header = VMSKHeader::new(
            MaskId::new(1),
            GenerationId::new(5),
            MaskType::VertexMask,
            1000,
        );
        
        assert_eq!(header.magic, magic::VMSK);
        assert_eq!(header.version, 1);
        assert_eq!(header.mask_id, 1);
        assert_eq!(header.generation, 5);
        assert_eq!(header.total_bits, 1000);
        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_bitmap_mask() {
        let mut mask = BitmapMask::new(
            MaskId::new(1),
            MaskType::VertexMask,
            GenerationId::new(1),
            100,
        );
        
        assert_eq!(mask.total_count(), 100);
        assert_eq!(mask.active_count(), 0);
        assert!(!mask.is_active(42));
        
        mask.set_bit(42).unwrap();
        assert!(mask.is_active(42));
        assert_eq!(mask.active_count(), 1);
        
        mask.set_bit(99).unwrap();
        assert!(mask.is_active(99));
        assert_eq!(mask.active_count(), 2);
        
        mask.clear_bit(42).unwrap();
        assert!(!mask.is_active(42));
        assert_eq!(mask.active_count(), 1);
        
        let active_indices = mask.active_indices();
        assert_eq!(active_indices, vec![99]);
    }

    #[test]
    fn test_mask_operations() {
        let mut mask1 = BitmapMask::new(
            MaskId::new(1),
            MaskType::VertexMask,
            GenerationId::new(1),
            10,
        );
        
        let mut mask2 = BitmapMask::new(
            MaskId::new(2),
            MaskType::VertexMask,
            GenerationId::new(1),
            10,
        );
        
        mask1.set_bit(1).unwrap();
        mask1.set_bit(3).unwrap();
        mask1.set_bit(5).unwrap();
        
        mask2.set_bit(3).unwrap();
        mask2.set_bit(5).unwrap();
        mask2.set_bit(7).unwrap();
        
        // Test intersection
        let intersection = mask1.intersect(&mask2).unwrap();
        assert!(intersection.is_active(3));
        assert!(intersection.is_active(5));
        assert!(!intersection.is_active(1));
        assert!(!intersection.is_active(7));
        assert_eq!(intersection.active_count(), 2);
        
        // Test union
        let union = mask1.union(&mask2).unwrap();
        assert!(union.is_active(1));
        assert!(union.is_active(3));
        assert!(union.is_active(5));
        assert!(union.is_active(7));
        assert_eq!(union.active_count(), 4);
        
        // Test difference
        let difference = mask1.difference(&mask2).unwrap();
        assert!(difference.is_active(1));
        assert!(!difference.is_active(3));
        assert!(!difference.is_active(5));
        assert!(!difference.is_active(7));
        assert_eq!(difference.active_count(), 1);
    }

    #[test]
    fn test_mask_serialization() {
        let mut mask = BitmapMask::new(
            MaskId::new(1),
            MaskType::VertexMask,
            GenerationId::new(1),
            100,
        );
        
        mask.set_bit(10).unwrap();
        mask.set_bit(20).unwrap();
        mask.set_bit(30).unwrap();
        
        let bytes = mask.export_vmsk().unwrap();
        assert!(!bytes.is_empty());
        
        // Verify header is at the beginning
        assert_eq!(&bytes[0..4], &magic::VMSK);
    }
}