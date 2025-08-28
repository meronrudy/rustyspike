# hSNN Documentation

Welcome to the comprehensive documentation for hSNN (Hypergraph Spiking Neural Networks), a high-performance, modular framework for spiking neural network computation.

## 📚 Documentation Structure

### Getting Started
- **[Installation Guide](getting-started/README.md)** - System requirements and installation instructions
- **[Quick Start Tutorial](getting-started/quickstart.md)** - Your first hSNN network in 5 minutes
- **[Basic Concepts](getting-started/concepts.md)** - Understanding SNNs and hypergraphs

### User Guide
- **[Network Creation](user-guide/networks.md)** - Building and configuring networks
- **[Connectivity Types](user-guide/connectivity.md)** - Choosing the right data structure
- **[Neuron Models](user-guide/neurons.md)** - Available neuron types and parameters
- **[Plasticity Rules](user-guide/plasticity.md)** - Learning and adaptation mechanisms
- **[Spike Encoding](user-guide/encoding.md)** - Converting data to spike patterns

### API Reference
- **[Rust API](api/rust/)** - Complete Rust API documentation
- **[Python API](api/python/)** - Python bindings reference
- **[WebAssembly API](api/wasm/)** - Browser JavaScript API
- **[C FFI](api/c/)** - C foreign function interface

### Architecture & Design
- **[System Architecture](architecture/README.md)** - High-level system design
- **[Modular Connectivity](architecture/connectivity.md)** - Pluggable connectivity framework
- **[Performance Design](architecture/performance.md)** - Optimization strategies
- **[Memory Management](architecture/memory.md)** - Memory layout and allocation

### Migration & Development
- **[Migration Guide](migration/README.md)** - Upgrading from previous versions
- **[Breaking Changes](migration/breaking-changes.md)** - Version compatibility notes
- **[Development Plan](development/HYPERGRAPH_MODULARIZATION_PLAN.md)** - Original modularization strategy
- **[Implementation Guide](implementation/HYPERGRAPH_MODULARIZATION_IMPLEMENTATION.md)** - Complete implementation details

### Advanced Topics
- **[Hardware Acceleration](advanced/hardware.md)** - GPU and neuromorphic chip support
- **[Real-time Processing](advanced/realtime.md)** - Streaming and async processing
- **[Distributed Computing](advanced/distributed.md)** - Multi-node deployments
- **[Custom Extensions](advanced/extensions.md)** - Implementing custom components

### Examples & Tutorials
- **[Code Examples](../examples/README.md)** - Working code examples by category
- **[Connectivity Showcase](../crates/shnn-core/examples/connectivity_showcase.rs)** - Complete feature demonstration
- **[Performance Benchmarks](../examples/benchmarks/)** - Performance comparison studies
- **[Research Applications](../examples/research/)** - Academic use cases

## 🎯 Quick Navigation

### By Use Case
- **Learning hSNN**: Start with [Getting Started](getting-started/) → [Examples](../examples/basic/)
- **Migrating Code**: Check [Migration Guide](migration/) → [Breaking Changes](migration/breaking-changes.md)
- **Performance Tuning**: Read [Architecture](architecture/) → [Performance Guide](advanced/performance.md)
- **Research Applications**: Explore [Advanced Topics](advanced/) → [Research Examples](../examples/research/)

### By Programming Language
- **Rust Developers**: [Rust API](api/rust/) → [Rust Examples](../examples/rust/)
- **Python Users**: [Python API](api/python/) → [Python Examples](../examples/python/)
- **Web Developers**: [WebAssembly API](api/wasm/) → [Web Examples](../examples/web/)
- **C/C++ Integration**: [C FFI](api/c/) → [C Examples](../examples/c/)

### By Network Type
- **Hypergraph Networks**: [Connectivity Guide](user-guide/connectivity.md#hypergraph) → [Hypergraph Examples](../examples/hypergraph/)
- **Traditional SNNs**: [Graph Networks](user-guide/connectivity.md#graph) → [Graph Examples](../examples/graph/)
- **Dense Networks**: [Matrix Networks](user-guide/connectivity.md#matrix) → [Matrix Examples](../examples/matrix/)
- **Large Sparse Networks**: [Sparse Networks](user-guide/connectivity.md#sparse) → [Sparse Examples](../examples/sparse/)

## 📖 Key Documents

### Most Important
1. **[Getting Started](getting-started/README.md)** - Essential first read
2. **[Connectivity Types](user-guide/connectivity.md)** - Core feature documentation
3. **[Migration Guide](migration/README.md)** - For existing users
4. **[API Reference](api/)** - Complete function documentation

### For Developers
1. **[System Architecture](architecture/README.md)** - Understanding the design
2. **[Implementation Guide](implementation/HYPERGRAPH_MODULARIZATION_IMPLEMENTATION.md)** - Technical details
3. **[Performance Design](architecture/performance.md)** - Optimization internals
4. **[Custom Extensions](advanced/extensions.md)** - Extending the framework

### For Researchers
1. **[Advanced Topics](advanced/)** - Cutting-edge features
2. **[Research Examples](../examples/research/)** - Academic applications
3. **[Performance Benchmarks](../examples/benchmarks/)** - Empirical studies
4. **[Development Plan](development/HYPERGRAPH_MODULARIZATION_PLAN.md)** - Design rationale

## 🚀 Getting Help

### Documentation Issues
- **Missing Information**: [Open a documentation issue](https://github.com/hsnn-project/hsnn/issues/new?template=documentation.md)
- **Incorrect Content**: [Submit a documentation PR](https://github.com/hsnn-project/hsnn/pulls)
- **Unclear Explanations**: [Join the discussion](https://github.com/hsnn-project/hsnn/discussions)

### Community Support
- **GitHub Discussions**: Ask questions and share ideas
- **Stack Overflow**: Tag questions with `hsnn` and `spiking-neural-networks`
- **Research Collaboration**: Contact the development team

### Contributing to Documentation
See our [Contributing Guide](../CONTRIBUTING.md) for:
- Documentation standards and style guide
- How to build and test documentation locally
- Submission process for documentation improvements

---

**Note**: This documentation covers hSNN v0.1.0 and later. For older versions, see the [legacy documentation](legacy/).

**Last Updated**: December 2024 | **Version**: 0.1.0 | **Status**: Complete modular implementation
