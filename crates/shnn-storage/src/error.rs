//! Error types for the storage layer

use thiserror::Error;

/// Result type for storage operations
pub type Result<T> = std::result::Result<T, StorageError>;

/// Errors that can occur in the storage layer
#[derive(Error, Debug)]
pub enum StorageError {
    /// Invalid magic number in binary format
    #[error("Invalid magic number: expected {expected:?}, found {found:?}")]
    InvalidMagic {
        /// Expected magic number
        expected: [u8; 4],
        /// Found magic number
        found: [u8; 4]
    },

    /// Unsupported version
    #[error("Unsupported version: {version}, supported: {supported}")]
    UnsupportedVersion {
        /// Version found
        version: u32,
        /// Supported version
        supported: u32
    },

    /// Checksum verification failed
    #[error("Checksum verification failed: expected {expected:08x}, computed {computed:08x}")]
    ChecksumMismatch {
        /// Expected checksum
        expected: u32,
        /// Computed checksum
        computed: u32
    },

    /// Invalid file format or corrupted data
    #[error("Invalid format: {reason}")]
    InvalidFormat {
        /// Reason for invalid format
        reason: String
    },

    /// Generation not found
    #[error("Generation {generation} not found")]
    GenerationNotFound {
        /// Generation ID that was not found
        generation: u64
    },

    /// Mask not found
    #[error("Mask {mask_id} not found")]
    MaskNotFound {
        /// Mask ID that was not found
        mask_id: u32
    },

    /// I/O error
    #[error("I/O error: {source}")]
    Io {
        #[from]
        /// Source I/O error
        source: std::io::Error,
    },

    /// Memory mapping error
    #[error("Memory mapping error: {reason}")]
    MemoryMap {
        /// Reason for memory mapping failure
        reason: String
    },

    /// Compression/decompression error
    #[error("Compression error: {reason}")]
    Compression {
        /// Reason for compression failure
        reason: String
    },

    /// Generic operation error
    #[error("Operation failed: {message}")]
    OperationError {
        /// Error message
        message: String,
    },

    /// Out of bounds access
    #[error("Index {index} out of bounds (max: {max})")]
    OutOfBounds {
        /// Index that was out of bounds
        index: usize,
        /// Maximum allowed index
        max: usize
    },

    /// Incompatible capability
    #[error("Missing capability: {capability}")]
    MissingCapability {
        /// Missing capability name
        capability: String
    },
}

impl StorageError {
    /// Create an invalid format error
    pub fn invalid_format(reason: impl Into<String>) -> Self {
        Self::InvalidFormat {
            reason: reason.into(),
        }
    }

    /// Create a memory mapping error
    pub fn memory_map(reason: impl Into<String>) -> Self {
        Self::MemoryMap {
            reason: reason.into(),
        }
    }

    /// Create a compression error
    pub fn compression(reason: impl Into<String>) -> Self {
        Self::Compression {
            reason: reason.into(),
        }
    }

    /// Create a missing capability error
    pub fn missing_capability(capability: impl Into<String>) -> Self {
        Self::MissingCapability {
            capability: capability.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = StorageError::invalid_format("test reason");
        assert!(matches!(err, StorageError::InvalidFormat { .. }));
        
        let err = StorageError::memory_map("mmap failed");
        assert!(matches!(err, StorageError::MemoryMap { .. }));
    }

    #[test]
    fn test_error_display() {
        let err = StorageError::InvalidMagic {
            expected: [0x56, 0x43, 0x53, 0x52],
            found: [0x00, 0x00, 0x00, 0x00],
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid magic number"));
    }
}