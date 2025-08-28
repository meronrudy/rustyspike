//! Binary schema definitions and utilities

use crate::{error::Result, magic};
use core::mem;

/// Validate magic number for a binary format
pub fn validate_magic(data: &[u8], expected: [u8; 4]) -> Result<()> {
    if data.len() < 4 {
        return Err(crate::error::StorageError::InvalidFormat {
            reason: "Data too short for magic number".to_string(),
        });
    }
    
    let found = [data[0], data[1], data[2], data[3]];
    if found != expected {
        return Err(crate::error::StorageError::InvalidMagic { expected, found });
    }
    
    Ok(())
}

/// Calculate CRC32 checksum
pub fn calculate_checksum(data: &[u8]) -> u32 {
    crc32fast::hash(data)
}

/// Validate checksum
pub fn validate_checksum(data: &[u8], expected: u32) -> Result<()> {
    let computed = calculate_checksum(data);
    if computed != expected {
        return Err(crate::error::StorageError::ChecksumMismatch { expected, computed });
    }
    Ok(())
}

/// Get current timestamp in nanoseconds since epoch
pub fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Safely cast a byte slice to a struct reference
/// 
/// # Safety
/// This function is safe as long as:
/// - The struct has `#[repr(C)]` for stable layout
/// - The alignment requirements are met
/// - The slice is large enough for the struct
pub unsafe fn cast_slice_to_struct<T>(data: &[u8]) -> Result<&T> {
    if data.len() < mem::size_of::<T>() {
        return Err(crate::error::StorageError::InvalidFormat {
            reason: format!(
                "Data too short: need {} bytes, got {}",
                mem::size_of::<T>(),
                data.len()
            ),
        });
    }
    
    // Check alignment
    if data.as_ptr() as usize % mem::align_of::<T>() != 0 {
        return Err(crate::error::StorageError::InvalidFormat {
            reason: "Invalid alignment for struct".to_string(),
        });
    }
    
    Ok(&*(data.as_ptr() as *const T))
}

/// Safely cast a byte slice to a mutable struct reference
/// 
/// # Safety
/// Same requirements as `cast_slice_to_struct`
pub unsafe fn cast_slice_to_struct_mut<T>(data: &mut [u8]) -> Result<&mut T> {
    if data.len() < mem::size_of::<T>() {
        return Err(crate::error::StorageError::InvalidFormat {
            reason: format!(
                "Data too short: need {} bytes, got {}",
                mem::size_of::<T>(),
                data.len()
            ),
        });
    }
    
    // Check alignment
    if data.as_ptr() as usize % mem::align_of::<T>() != 0 {
        return Err(crate::error::StorageError::InvalidFormat {
            reason: "Invalid alignment for struct".to_string(),
        });
    }
    
    Ok(&mut *(data.as_mut_ptr() as *mut T))
}

/// Convert any value to bytes
pub fn to_bytes<T>(value: &T) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(
            value as *const T as *const u8,
            mem::size_of::<T>(),
        )
    }
}

/// Capability flags utilities
pub mod capabilities {
    /// Check if a capability flag is set
    pub fn has_capability(flags: u64, capability: u64) -> bool {
        (flags & capability) != 0
    }
    
    /// Set a capability flag
    pub fn set_capability(flags: &mut u64, capability: u64) {
        *flags |= capability;
    }
    
    /// Clear a capability flag
    pub fn clear_capability(flags: &mut u64, capability: u64) {
        *flags &= !capability;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_validation() {
        let data = [0x56, 0x43, 0x53, 0x52, 0x00, 0x00]; // VCSR + padding
        assert!(validate_magic(&data, magic::VCSR).is_ok());
        
        let bad_data = [0x00, 0x00, 0x00, 0x00];
        assert!(validate_magic(&bad_data, magic::VCSR).is_err());
    }

    #[test]
    fn test_checksum() {
        let data = b"hello world";
        let checksum = calculate_checksum(data);
        assert!(validate_checksum(data, checksum).is_ok());
        assert!(validate_checksum(data, checksum + 1).is_err());
    }

    #[test]
    fn test_capabilities() {
        let mut flags = 0u64;
        let cap1 = 1u64;
        let cap2 = 2u64;
        
        assert!(!capabilities::has_capability(flags, cap1));
        
        capabilities::set_capability(&mut flags, cap1);
        assert!(capabilities::has_capability(flags, cap1));
        assert!(!capabilities::has_capability(flags, cap2));
        
        capabilities::clear_capability(&mut flags, cap1);
        assert!(!capabilities::has_capability(flags, cap1));
    }

    #[test]
    fn test_timestamp() {
        let ts = current_timestamp();
        assert!(ts > 0);
    }

    #[repr(C)]
    struct TestStruct {
        a: u32,
        b: u64,
    }

    #[test]
    fn test_cast_slice_to_struct() {
        let data = vec![0u8; mem::size_of::<TestStruct>()];
        let aligned_data = {
            let mut aligned = vec![0u8; mem::size_of::<TestStruct>() + mem::align_of::<TestStruct>()];
            let offset = aligned.as_ptr().align_offset(mem::align_of::<TestStruct>());
            aligned.drain(..offset);
            aligned.truncate(mem::size_of::<TestStruct>());
            aligned
        };
        
        unsafe {
            let result = cast_slice_to_struct::<TestStruct>(&aligned_data);
            assert!(result.is_ok());
        }
    }
}