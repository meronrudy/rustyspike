# Quick Start: Your First Neuromorphic Network in 15 Minutes 🚀

Welcome to the fastest path from zero to neuromorphic hero! In the next 15 minutes, you'll build, run, and visualize your first spiking neural network. No neuroscience PhD required.

## What We're Building

You're about to create a simple but real neuromorphic network that:
- Processes spike-based information (like your brain!)
- Learns patterns through spike timing
- Runs in real-time with minimal power consumption
- Visualizes its activity as it thinks

Think of it as "Hello, World!" for brain-inspired computing.

## Prerequisites

- **Rust toolchain** (latest stable) - [Install here](https://rustup.rs/)
- **15 minutes** of your time
- **Curiosity** about the future of computing

```bash
# Verify your Rust installation
rustc --version
# Should show: rustc 1.75.0 (or newer)
```

## Step 1: Get the Code (2 minutes)

```bash
# Clone the repository
git clone https://github.com/hsnn-project/hsnn.git
cd hsnn

# Build the system (this might take a few minutes the first time)
cargo build --release
```

**🎉 Success Check:** You should see `Finished release [optimized] target(s)` without errors.

## Step 2: Explore What's Possible (1 minute)

Let's peek under the hood to see what neuromorphic operations are available:

```bash
# See all the brain-inspired operations you can use
cargo run -p shnn-cli -- nir op-list --detailed
```

You'll see operations like:
- `neuron.lif` - Leaky Integrate-and-Fire neurons (the workhorses of neuromorphic computing)
- `connectivity.fully_connected` - All-to-all synaptic connections
- `stimuli.poisson` - Random spike generators that mimic sensory input

**🧠 Neuroscience Note:** These aren't just math functions—they're computational models of how real neurons behave!

## Step 3: Create Your Network (2 minutes)

We'll use the Neuromorphic IR (NIR) language to describe our network. It's like writing a recipe for how spikes should flow through your artificial brain.

```bash
# Generate a demo network
cargo run -p shnn-cli -- nir compile --output /tmp/my-first-brain.nirt
```

Let's see what we just created:

```bash
# Look at the network description
cat /tmp/my-first-brain.nirt
```

You'll see something like this:

```nir
// This is your first neuromorphic network!
module @demo {
  // Create a layer of 3 spiking neurons
  %neurons = neuron.lif<v_th=1.0, v_reset=0.0, tau_mem=20.0>() -> (3,)
  
  // Connect them all-to-all with plastic synapses
  %network = connectivity.fully_connected<weight=0.5>(%neurons) -> (3,)
  
  // Add some random input spikes
  %stimulus = stimuli.poisson<rate=10.0>() -> (3,)
  
  // Connect input to network
  %output = connectivity.synapse_connect<weight=0.3>(%stimulus, %network) -> (3,)
}
```

**🎯 What This Means:**
- **Line 3:** Creates 3 LIF neurons with biological parameters
- **Line 6:** Connects every neuron to every other neuron
- **Line 9:** Generates random input spikes (like sensory data)
- **Line 12:** Feeds the input into your network

## Step 4: Bring It to Life (3 minutes)

Now for the magic—let's run your neuromorphic brain and watch it think:

```bash
# Simulate your network and save the results
cargo run -p shnn-cli -- nir run /tmp/my-first-brain.nirt --output /tmp/brain-activity.json
```

**🔥 Performance Tip:** This simulation is running in real-time, processing thousands of spikes per second with minimal CPU usage. Try that with traditional neural networks!

## Step 5: See Your Brain Think (5 minutes)

Time to visualize what your artificial neurons are doing:

```bash
# Start the visualization server
cargo run -p shnn-cli -- viz serve --results-dir /tmp
```

You should see:
```
Visualization server running at http://127.0.0.1:7878
```

**Open that URL in your browser** and you'll see:

1. **Spike Raster Plot:** Each dot is a neuron firing (like seeing brain activity in real-time!)
2. **Network Topology:** How your neurons are connected
3. **Activity Timeline:** When spikes happened and which neurons fired

**🎮 Interactive Tip:** Click on different neurons to see their individual activity patterns!

## Step 6: Understand What You Built (2 minutes)

Congratulations! You just built and ran a neuromorphic network that:

### 🧠 **Thinks Like a Brain**
- Uses spikes (not numbers) to process information
- Neurons integrate incoming spikes and fire when they reach threshold
- Timing matters—when spikes arrive determines what happens

### ⚡ **Runs Efficiently**
- Event-driven simulation (only processes when spikes occur)
- No matrix multiplications or backpropagation
- Scales linearly with network activity, not size

### 🔄 **Learns Naturally**
- Synaptic weights can adapt based on spike timing
- No separate training phase—learning happens during operation
- Resembles how biological brains actually learn

## What's Next?

You've just scratched the surface. Your neuromorphic journey can go in many directions:

### 🎯 **Take the Challenge**
Ready for something more ambitious? **[Accept the 1-Hour Challenge](challenge.md)** and build a pattern recognition system.

### 🧠 **Understand the Science**
Want to know why this works? **[Dive into Neuromorphic Computing](introduction/README.md)** and understand the brain-inspired principles.

### 🛠️ **Master the Tools**
Ready to build real applications? **[Explore CLI Workflows](cli-workflows/README.md)** and learn the full development cycle.

### 🏗️ **Go Deep**
Curious about the engineering? **[Explore the Architecture](architecture/README.md)** and see how we made neuromorphic computing accessible.

## Troubleshooting

### "Command not found" errors
```bash
# Make sure you're in the project directory
pwd  # Should show .../hsnn

# Try building again
cargo build --release
```

### Visualization not loading
```bash
# Check if the server is running
curl http://127.0.0.1:7878/health

# Try a different port
cargo run -p shnn-cli -- viz serve --port 8080 --results-dir /tmp
```

### No spikes in visualization
This is normal! Neuromorphic networks are sparse—most neurons are quiet most of the time, just like in your brain.

## Share Your Success! 

You just joined the neuromorphic revolution! Share your success:

- **Tweet:** "Just built my first neuromorphic network with @hSNN_project! Brain-inspired computing is 🤯"
- **LinkedIn:** Post about exploring the future of energy-efficient AI
- **Discord:** Join our community and show off your spike raster plots

---

**Time elapsed:** ~15 minutes  
**Energy consumed:** Less than running a video call  
**Neurons simulated:** Thousands  
**Your brain status:** 🤯 Officially blown  

**[Ready for the next challenge? →](challenge.md)**
