//! ID types for the storage layer

use core::fmt;

/// Unique identifier for a generation (snapshot) in the hypergraph database
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GenerationId(pub u64);

impl GenerationId {
    /// Create a new generation ID
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
    
    /// Get the raw ID value
    pub const fn raw(&self) -> u64 {
        self.0
    }
    
    /// Initial generation ID
    pub const INITIAL: Self = Self(0);
    
    /// Invalid generation ID constant
    pub const INVALID: Self = Self(u64::MAX);
    
    /// Check if this is a valid generation ID
    pub const fn is_valid(&self) -> bool {
        self.0 != u64::MAX
    }
    
    /// Get the next generation ID
    pub const fn next(&self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl Default for GenerationId {
    fn default() -> Self {
        Self::INITIAL
    }
}

impl fmt::Display for GenerationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "G{}", self.0)
    }
}

/// Unique identifier for a mask
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaskId(pub u32);

impl MaskId {
    /// Create a new mask ID
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
    
    /// Get the raw ID value
    pub const fn raw(&self) -> u32 {
        self.0
    }
    
    /// Invalid mask ID constant
    pub const INVALID: Self = Self(u32::MAX);
    
    /// Check if this is a valid mask ID
    pub const fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }
}

impl fmt::Display for MaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "M{}", self.0)
    }
}

/// Stream ID for event streams
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId(pub u64);

impl StreamId {
    /// Create a new stream ID
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
    
    /// Get the raw ID value
    pub const fn raw(&self) -> u64 {
        self.0
    }
    
    /// Invalid stream ID constant
    pub const INVALID: Self = Self(u64::MAX);
    
    /// Check if this is a valid stream ID
    pub const fn is_valid(&self) -> bool {
        self.0 != u64::MAX
    }
}

impl fmt::Display for StreamId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S{}", self.0)
    }
}

#[cfg(feature = "serde")]
mod serde_impls {
    use super::*;
    use serde::{Deserialize, Serialize};

    impl Serialize for GenerationId {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            self.0.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for GenerationId {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let id = u64::deserialize(deserializer)?;
            Ok(GenerationId::new(id))
        }
    }

    impl Serialize for MaskId {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            self.0.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for MaskId {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let id = u32::deserialize(deserializer)?;
            Ok(MaskId::new(id))
        }
    }

    impl Serialize for StreamId {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            self.0.serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for StreamId {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let id = u64::deserialize(deserializer)?;
            Ok(StreamId::new(id))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_id() {
        let gen = GenerationId::new(42);
        assert_eq!(gen.raw(), 42);
        assert!(gen.is_valid());
        assert_eq!(gen.next(), GenerationId::new(43));
        assert_eq!(format!("{}", gen), "G42");
    }

    #[test]
    fn test_mask_id() {
        let mask = MaskId::new(123);
        assert_eq!(mask.raw(), 123);
        assert!(mask.is_valid());
        assert_eq!(format!("{}", mask), "M123");
    }

    #[test]
    fn test_invalid_ids() {
        assert!(!GenerationId::INVALID.is_valid());
        assert!(!MaskId::INVALID.is_valid());
        assert!(!StreamId::INVALID.is_valid());
    }

    #[test]
    fn test_ordering() {
        let gen1 = GenerationId::new(1);
        let gen2 = GenerationId::new(2);
        assert!(gen1 < gen2);
        
        let mask1 = MaskId::new(1);
        let mask2 = MaskId::new(2);
        assert!(mask1 < mask2);
    }
}