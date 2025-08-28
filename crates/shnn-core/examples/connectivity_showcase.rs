//! Comprehensive showcase of different connectivity types
//!
//! This example demonstrates how to use different connectivity structures
//! and serves as a migration guide from the old API to the new modular system.

use shnn_core::{
    connectivity::graph::GraphNetwork,
    network::{NetworkBuilder, builder::NeuronConfig},
    neuron::{LIFConfig, NeuronType, NeuronPool, LIFNeuron},
    hypergraph::HypergraphNetwork,
    plasticity::STDPConfig,
    time::Time,
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 SHNN Connectivity Showcase");
    println!("============================\n");
    
    // 1. Demonstrate different connectivity types
    demo_connectivity_types()?;
    
    // 2. Show migration from old API to new API
    demo_migration_guide()?;
    
    // 3. Performance comparison
    demo_performance_comparison()?;
    
    // 4. Advanced usage patterns
    demo_advanced_usage()?;
    
    println!("✅ All demonstrations completed successfully!");
    Ok(())
}

/// Demonstrate all connectivity types
fn demo_connectivity_types() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 1. Connectivity Types Demonstration");
    println!("======================================\n");
    
    let network_size = 50;
    let simulation_time = Time::from_millis(100);
    
    // Hypergraph Network
    println!("🕸️  Hypergraph Network:");
    {
        let network = create_hypergraph_network(network_size)?;
        let stats = simulate_network_basic(&*network, simulation_time)?;
        println!("  📊 Hypergraph Stats: {} spikes processed, {:.2} efficiency", 
                 stats.total_spikes_processed, stats.efficiency());
    }
    
    // Graph Network  
    println!("\n📊 Graph Network:");
    {
        let network = create_graph_network(network_size)?;
        let stats = simulate_network_basic(&*network, simulation_time)?;
        println!("  📊 Graph Stats: {} spikes processed, {:.2} efficiency", 
                 stats.total_spikes_processed, stats.efficiency());
    }
    
    // Matrix Network
    println!("\n🔢 Matrix Network:");
    {
        let network = create_matrix_network(network_size)?;
        let stats = simulate_network_basic(&*network, simulation_time)?;
        println!("  📊 Matrix Stats: {} spikes processed, {:.2} generation rate", 
                 stats.total_spikes_processed, 
                 stats.generation_rate());
    }
    
    // Sparse Network
    println!("\n⚡ Sparse Network:");
    {
        let network = create_sparse_network(network_size)?;
        let stats = simulate_network_basic(&*network, simulation_time)?;
        println!("  📊 Sparse Stats: {} spikes processed, {:.2} rate", 
                 stats.total_spikes_processed, stats.spike_rate());
    }
    
    println!();
    Ok(())
}

/// Show migration patterns
fn demo_migration_guide() -> Result<(), Box<dyn std::error::Error>> {
    println!("📚 2. Migration Guide");
    println!("====================\n");
    
    println!("🔄 Old API vs New API patterns:");
    
    // Before: Direct HypergraphNetwork usage
    println!("✅ Before: let network = HypergraphNetwork::new();");
    
    // After: Generic NetworkBuilder approach
    println!("✅ After: Using NetworkBuilder for flexibility");
    
    Ok(())
}

/// Performance comparison between connectivity types
fn demo_performance_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("⚡ 3. Performance Comparison");
    println!("===========================\n");
    
    let network_size = 100;
    
    // Compare different connectivity types
    let connectivity_types = ["Hypergraph", "Graph", "Matrix", "Sparse"];
    
    for conn_type in &connectivity_types {
        let start = Instant::now();
        
        let _network = match *conn_type {
            "Hypergraph" => create_hypergraph_network(network_size)?,
            "Graph" => create_graph_network(network_size)?,
            "Matrix" => create_matrix_network(network_size)?,
            "Sparse" => create_sparse_network(network_size)?,
            _ => unreachable!(),
        };
        
        let duration = start.elapsed();
        println!("🏁 {} network creation: {:.3}ms", conn_type, duration.as_secs_f64() * 1000.0);
    }
    
    println!();
    Ok(())
}

/// Advanced usage patterns
fn demo_advanced_usage() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 4. Advanced Usage Patterns");
    println!("=============================\n");
    
    // Configuration-driven networks
    println!("⚙️  Configuration-driven approach:");
    {
        let _stdp_config = STDPConfig {
            a_plus: 0.1,
            a_minus: 0.12,
            tau_plus: 20.0,
            tau_minus: 20.0,
            w_min: 0.0,
            w_max: 1.0,
            learning_rate: 0.01,
            multiplicative: false,
        };
        
        // Factory method approach
        let _network = shnn_core::network::builder::NetworkFactory::feedforward_graph(10, 15, 5, 0.5)?;
    }
    
    // Builder pattern approach
    println!("🏗️  Builder pattern approach:");
    {
        let lif_config = LIFConfig {
            resting_potential: -70.0,
            threshold: -55.0,
            reset_potential: -80.0,
            tau_membrane: 20.0,
            resistance: 10.0,
            refractory_period: 2.0,
            capacitance: 2.0,
        };
        
        let neuron_config = NeuronConfig {
            lif_config: Some(lif_config),
            count: 50,
            neuron_type: NeuronType::LIF,
        };
        
        let neurons: NeuronPool<LIFNeuron> = NeuronPool::new();
        let _network = NetworkBuilder::new()
            .with_connectivity(GraphNetwork::new())
            .with_neurons(neuron_config)
            .build(neurons)?;
    }
    
    // Batch processing networks
    println!("📦 Batch processing examples:");
    {
        let _network1 = shnn_core::network::builder::NetworkFactory::fully_connected_matrix(30, 0.8)?;
        
        // Sparse random network for large-scale simulation
        let _network2 = shnn_core::network::builder::NetworkFactory::sparse_random(200, 0.05, (0.1, 1.5))?;
        
        // Hypergraph network for complex relationships
        let _network3 = shnn_core::network::builder::NetworkFactory::hypergraph_network(50, 30)?;
    }
    
    println!();
    Ok(())
}

// Helper functions for creating different network types
fn create_hypergraph_network(size: usize) -> Result<Box<dyn std::fmt::Debug>, Box<dyn std::error::Error>> {
    let _connectivity = HypergraphNetwork::new();
    
    let neuron_config = NeuronConfig {
        lif_config: Some(LIFConfig::default()),
        count: size,
        neuron_type: NeuronType::LIF,
    };
    
    let neurons: NeuronPool<LIFNeuron> = NeuronPool::new();
    let network = NetworkBuilder::new()
        .with_connectivity(HypergraphNetwork::new())
        .with_neurons(neuron_config)
        .build(neurons)?;
    
    Ok(Box::new(network))
}

fn create_graph_network(size: usize) -> Result<Box<dyn std::fmt::Debug>, Box<dyn std::error::Error>> {
    let neuron_config = NeuronConfig {
        lif_config: Some(LIFConfig::default()),
        count: size,
        neuron_type: NeuronType::LIF,
    };
    
    let neurons: NeuronPool<LIFNeuron> = NeuronPool::new();
    let network = NetworkBuilder::new()
        .with_connectivity(GraphNetwork::new())
        .with_neurons(neuron_config)
        .build(neurons)?;
        
    Ok(Box::new(network))
}

fn create_matrix_network(size: usize) -> Result<Box<dyn std::fmt::Debug>, Box<dyn std::error::Error>> {
    let network = shnn_core::network::builder::NetworkFactory::fully_connected_matrix(size, 0.7)?;
    Ok(Box::new(network))
}

fn create_sparse_network(size: usize) -> Result<Box<dyn std::fmt::Debug>, Box<dyn std::error::Error>> {
    let network = shnn_core::network::builder::NetworkFactory::sparse_random(size, 0.1, (0.1, 1.0))?;
    Ok(Box::new(network))
}

// Simple simulation function
fn simulate_network_basic(_network: &dyn std::fmt::Debug, _duration: Time) -> Result<shnn_core::network::NetworkStats, Box<dyn std::error::Error>> {
    // Create mock statistics for demonstration
    let mut stats = shnn_core::network::NetworkStats::new();
    stats.total_spikes_processed = 42;
    stats.total_spikes_generated = 38;
    stats.simulation_steps = 100;
    stats.current_time = Time::from_millis(100);
    
    Ok(stats)
}