//! VEVT (Event Stream) format implementation

use crate::{
    error::{Result, StorageError},
    ids::StreamId,
    magic,
    schemas::{calculate_checksum, current_timestamp, validate_magic},
    traits::{Event, EventStore, EventType},
    NeuronId, Time,
};

use core::mem;

/// VEVT format header
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VEVTHeader {
    /// Magic number "VEVT"
    pub magic: [u8; 4],
    /// Schema version (current: 1)
    pub version: u32,
    /// Unique stream identifier
    pub stream_id: u64,
    
    // Temporal range
    /// Start time (nanoseconds)
    pub time_start: u64,
    /// End time (nanoseconds)
    pub time_end: u64,
    /// Time resolution (nanoseconds per tick)
    pub time_resolution: u32,
    
    // Event counts
    /// Total number of events
    pub total_events: u64,
    /// Number of spike events
    pub spike_events: u64,
    /// Number of control events
    pub control_events: u64,
    
    // Compression and encoding
    /// Compression type
    pub compression: u8,
    /// Encoding type
    pub encoding: u8,
    /// Stream flags
    pub flags: u16,
    
    // Data layout
    /// Offset to event data
    pub events_offset: u64,
    /// Offset to time index
    pub index_offset: u64,
    /// Offset to metadata
    pub metadata_offset: u64,
    
    // Integrity
    /// CRC32 of header
    pub header_checksum: u32,
    /// CRC32 of event data
    pub data_checksum: u32,
    
    /// Reserved space
    pub reserved: [u8; 24],
}

impl VEVTHeader {
    /// Create a new VEVT header
    pub fn new(stream_id: StreamId) -> Self {
        Self {
            magic: magic::VEVT,
            version: 1,
            stream_id: stream_id.raw(),
            time_start: 0,
            time_end: 0,
            time_resolution: 1, // 1 nanosecond resolution
            total_events: 0,
            spike_events: 0,
            control_events: 0,
            compression: VEVTCompression::NONE,
            encoding: VEVTEncoding::BINARY,
            flags: 0,
            events_offset: 0,
            index_offset: 0,
            metadata_offset: 0,
            header_checksum: 0,
            data_checksum: 0,
            reserved: [0; 24],
        }
    }
    
    /// Validate this header
    pub fn validate(&self) -> Result<()> {
        validate_magic(&self.magic, magic::VEVT)?;
        
        if self.version != 1 {
            return Err(StorageError::UnsupportedVersion {
                version: self.version,
                supported: 1,
            });
        }
        
        Ok(())
    }
}

/// Compression types
pub struct VEVTCompression;
impl VEVTCompression {
    /// No compression
    pub const NONE: u8 = 0;
    /// LZ4 compression
    pub const LZ4: u8 = 1;
    /// ZSTD compression
    pub const ZSTD: u8 = 2;
    /// Custom compression algorithm
    pub const CUSTOM: u8 = 255;
}

/// Event encoding types
pub struct VEVTEncoding;
impl VEVTEncoding {
    /// Binary encoding
    pub const BINARY: u8 = 0;
    /// Delta compressed encoding
    pub const DELTA_COMPRESSED: u8 = 1;
    /// Run-length encoding
    pub const RLE: u8 = 2;
}

/// Base event structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct VEVTEvent {
    /// Event timestamp (nanoseconds)
    pub timestamp: u64,
    /// Event type
    pub event_type: u8,
    /// Source neuron/entity ID
    pub source_id: u32,
    /// Target neuron/entity ID (if applicable)
    pub target_id: u32,
    /// Size of additional payload
    pub payload_size: u16,
    /// Reserved for alignment
    pub reserved: u8,
}

impl Event for VEVTEvent {
    fn timestamp(&self) -> Time {
        Time::from_nanos(self.timestamp)
    }
    
    fn event_type(&self) -> EventType {
        match self.event_type {
            0 => EventType::Spike,
            1 => EventType::PhaseEnter,
            2 => EventType::PhaseExit,
            3 => EventType::Neuromodulation,
            4 => EventType::Reward,
            5 => EventType::Control,
            6 => EventType::Marker,
            _ => EventType::Control, // Default fallback
        }
    }
    
    fn source_id(&self) -> Option<NeuronId> {
        if self.source_id != u32::MAX {
            Some(NeuronId::new(self.source_id))
        } else {
            None
        }
    }
    
    fn serialize(&self) -> Result<Vec<u8>> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes.push(self.event_type);
        bytes.extend_from_slice(&self.source_id.to_le_bytes());
        bytes.extend_from_slice(&self.target_id.to_le_bytes());
        bytes.extend_from_slice(&self.payload_size.to_le_bytes());
        bytes.push(self.reserved);
        Ok(bytes)
    }
}

/// Simple in-memory event store implementation
pub struct MemoryEventStore {
    events: Vec<VEVTEvent>,
    stream_id: StreamId,
}

impl MemoryEventStore {
    /// Create a new memory event store
    pub fn new(stream_id: StreamId) -> Self {
        Self {
            events: Vec::new(),
            stream_id,
        }
    }
}

impl EventStore for MemoryEventStore {
    type Event = VEVTEvent;
    type EventIter = std::vec::IntoIter<VEVTEvent>;
    type Error = StorageError;
    
    fn append_events(&mut self, events: &[Self::Event]) -> Result<()> {
        self.events.extend_from_slice(events);
        // Keep events sorted by timestamp
        self.events.sort_by_key(|e| e.timestamp);
        Ok(())
    }
    
    fn time_window(&self, start: Time, end: Time) -> Result<Self::EventIter> {
        let start_ns = start.as_nanos();
        let end_ns = end.as_nanos();
        
        let filtered_events: Vec<_> = self.events.iter()
            .filter(|e| e.timestamp >= start_ns && e.timestamp <= end_ns)
            .cloned()
            .collect();
        
        Ok(filtered_events.into_iter())
    }
    
    fn neuron_events(
        &self,
        neurons: &[NeuronId],
        start: Time,
        end: Time,
    ) -> Result<Self::EventIter> {
        let start_ns = start.as_nanos();
        let end_ns = end.as_nanos();
        let neuron_set: std::collections::HashSet<_> = neurons.iter()
            .map(|n| n.raw())
            .collect();
        
        let filtered_events: Vec<_> = self.events.iter()
            .filter(|e| {
                e.timestamp >= start_ns && 
                e.timestamp <= end_ns &&
                neuron_set.contains(&e.source_id)
            })
            .cloned()
            .collect();
        
        Ok(filtered_events.into_iter())
    }
    
    fn event_count(&self) -> u64 {
        self.events.len() as u64
    }
    
    fn time_range(&self) -> Option<(Time, Time)> {
        if self.events.is_empty() {
            return None;
        }
        
        let min_time = self.events.first().unwrap().timestamp;
        let max_time = self.events.last().unwrap().timestamp;
        
        Some((Time::from_nanos(min_time), Time::from_nanos(max_time)))
    }
    
    fn export_vevt(&self, start: Time, end: Time) -> Result<Vec<u8>> {
        let mut header = VEVTHeader::new(self.stream_id);
        header.time_start = start.as_nanos();
        header.time_end = end.as_nanos();
        
        let events_in_range: Vec<_> = self.time_window(start, end)?.collect();
        header.total_events = events_in_range.len() as u64;
        header.spike_events = events_in_range.iter()
            .filter(|e| e.event_type == 0)
            .count() as u64;
        
        let mut bytes = Vec::new();
        
        // Write header
        let header_bytes = unsafe {
            core::slice::from_raw_parts(
                &header as *const VEVTHeader as *const u8,
                mem::size_of::<VEVTHeader>(),
            )
        };
        bytes.extend_from_slice(header_bytes);
        
        // Write events
        for event in events_in_range {
            let event_bytes = unsafe {
                core::slice::from_raw_parts(
                    &event as *const VEVTEvent as *const u8,
                    mem::size_of::<VEVTEvent>(),
                )
            };
            bytes.extend_from_slice(event_bytes);
        }
        
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vevt_header() {
        let header = VEVTHeader::new(StreamId::new(1));
        assert_eq!(header.magic, magic::VEVT);
        assert_eq!(header.version, 1);
        assert_eq!(header.stream_id, 1);
        assert!(header.validate().is_ok());
    }

    #[test]
    fn test_vevt_event() {
        let event = VEVTEvent {
            timestamp: 1000,
            event_type: 0, // Spike
            source_id: 42,
            target_id: 43,
            payload_size: 0,
            reserved: 0,
        };
        
        assert_eq!(event.timestamp(), Time::from_nanos(1000));
        assert_eq!(event.event_type(), EventType::Spike);
        assert_eq!(event.source_id(), Some(NeuronId::new(42)));
    }

    #[test]
    fn test_memory_event_store() {
        let mut store = MemoryEventStore::new(StreamId::new(1));
        
        let events = vec![
            VEVTEvent {
                timestamp: 1000,
                event_type: 0,
                source_id: 1,
                target_id: 2,
                payload_size: 0,
                reserved: 0,
            },
            VEVTEvent {
                timestamp: 2000,
                event_type: 0,
                source_id: 2,
                target_id: 3,
                payload_size: 0,
                reserved: 0,
            },
        ];
        
        store.append_events(&events).unwrap();
        assert_eq!(store.event_count(), 2);
        
        let time_range = store.time_range().unwrap();
        assert_eq!(time_range.0, Time::from_nanos(1000));
        assert_eq!(time_range.1, Time::from_nanos(2000));
        
        let windowed_events: Vec<_> = store.time_window(
            Time::from_nanos(1500),
            Time::from_nanos(2500),
        ).unwrap().collect();
        assert_eq!(windowed_events.len(), 1);
        assert_eq!(windowed_events[0].timestamp, 2000);
    }
}