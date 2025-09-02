# Chapter 3: Time as Computation ⏰

Imagine you're a jazz musician improvising a solo. The magic isn't just in the notes you play—it's in **when** you play them. The pause before a phrase creates tension. The syncopated rhythm against the bass line creates groove. The precise timing of your entrance after the drummer's fill creates excitement.

In traditional computing, time is the stage where computation happens. In neuromorphic computing, **time IS the computation**.

## The Timing Revolution

### Traditional Computing: Time as a Container

In conventional systems, time is external to the computation:

```python
# Traditional approach: time is just a parameter
def process_data(data, timestamp):
    result = complicated_math(data)  # Time doesn't affect the math
    return (result, timestamp)       # Time is just metadata

# Process batch every 16ms regardless of content
for frame in video_frames:
    result = process_data(frame, current_time)
    current_time += 16  # Fixed time step
```

**The limitation:** Computation happens at fixed intervals, regardless of whether anything interesting is occurring.

### Neuromorphic Computing: Time as the Medium

In neuromorphic systems, time shapes the computation itself:

```rust
// Neuromorphic approach: timing IS the information
struct TemporalNeuron {
    membrane_potential: f64,
    last_spike_time: f64,
    threshold: f64,
}

impl TemporalNeuron {
    fn receive_spike(&mut self, spike_time: f64) -> Option<f64> {
        let dt = spike_time - self.last_spike_time;
        
        // Computation depends on WHEN the spike arrived
        if dt < 5.0 {
            self.membrane_potential += 1.0 / dt;  // Recent spikes matter more
        } else {
            self.membrane_potential += 0.1;       // Old spikes matter less
        }
        
        // Decision emerges from temporal dynamics
        if self.membrane_potential > self.threshold {
            self.membrane_potential = 0.0;
            Some(spike_time)  // Output spike timing depends on input timing
        } else {
            None
        }
    }
}
```

**The insight:** The same input spike has different effects depending on when it arrives relative to previous spikes.

## Temporal Computation Patterns

Nature has evolved several ways to use time as a computational medium. Understanding these patterns gives us powerful tools for building adaptive systems.

### 1. Integration: Building Evidence Over Time

Your brain doesn't make decisions based on single snapshots—it accumulates evidence over time.

```rust
pub struct EvidenceAccumulator {
    evidence: f64,
    decay_rate: f64,
    threshold: f64,
    last_update: f64,
}

impl EvidenceAccumulator {
    pub fn add_evidence(&mut self, amount: f64, time: f64) -> bool {
        // Decay old evidence
        let dt = time - self.last_update;
        self.evidence *= (-dt * self.decay_rate).exp();
        
        // Add new evidence
        self.evidence += amount;
        self.last_update = time;
        
        // Decision when threshold reached
        self.evidence > self.threshold
    }
}

// Example: Motion detection
fn main() {
    let mut detector = EvidenceAccumulator {
        evidence: 0.0,
        decay_rate: 0.1,
        threshold: 5.0,
        last_update: 0.0,
    };
    
    // Isolated pixel changes: no motion detected
    detector.add_evidence(1.0, 0.0);   // false
    detector.add_evidence(1.0, 50.0);  // false (too much time gap)
    
    // Clustered pixel changes: motion detected!
    detector.add_evidence(1.0, 100.0); // false
    detector.add_evidence(2.0, 101.0); // false  
    detector.add_evidence(2.5, 102.0); // true! (rapid accumulation)
}
```

**Why this works:** Real motion creates clusters of changes in space and time. Random noise creates isolated changes. Temporal integration distinguishes between them automatically.

### 2. Coincidence Detection: Finding Relationships

When multiple events happen at the same time, it's probably not a coincidence—it's a relationship.

```rust
pub struct CoincidenceDetector {
    input_channels: Vec<Option<f64>>,  // Recent spike times per channel
    coincidence_window: f64,
    threshold: usize,
}

impl CoincidenceDetector {
    pub fn receive_spike(&mut self, channel: usize, time: f64) -> bool {
        self.input_channels[channel] = Some(time);
        
        // Count channels with recent spikes
        let coincident_channels = self.input_channels.iter()
            .filter_map(|&spike_time| spike_time)
            .filter(|&spike_time| (time - spike_time).abs() < self.coincidence_window)
            .count();
            
        // Fire if enough channels are active simultaneously
        if coincident_channels >= self.threshold {
            self.input_channels.fill(None);  // Reset for next detection
            true
        } else {
            false
        }
    }
}

// Example: Sound localization
fn main() {
    let mut localizer = CoincidenceDetector {
        input_channels: vec![None; 8],  // 8 microphones
        coincidence_window: 2.0,        // 2ms window
        threshold: 6,                   // Need 6+ simultaneous
    };
    
    // Sound from specific direction reaches multiple mics at nearly same time
    localizer.receive_spike(0, 100.0);  // false
    localizer.receive_spike(1, 100.5);  // false
    localizer.receive_spike(2, 101.0);  // false
    localizer.receive_spike(3, 101.2);  // false
    localizer.receive_spike(4, 101.5);  // false
    localizer.receive_spike(5, 101.8);  // true! Sound source detected
}
```

**Real-world application:** This is exactly how barn owls locate prey in complete darkness with superhuman accuracy.

### 3. Sequence Detection: Learning Temporal Patterns

Some information only exists in the temporal sequence of events.

```rust
pub struct SequenceDetector {
    pattern: Vec<usize>,           // Expected sequence of neuron IDs
    current_position: usize,       // Where we are in the pattern
    timing_window: f64,            // Max time between sequence elements
    last_event_time: f64,
}

impl SequenceDetector {
    pub fn receive_spike(&mut self, neuron_id: usize, time: f64) -> bool {
        let dt = time - self.last_event_time;
        
        // Check if this spike continues the expected sequence
        if neuron_id == self.pattern[self.current_position] && dt < self.timing_window {
            self.current_position += 1;
            self.last_event_time = time;
            
            // Complete sequence detected!
            if self.current_position >= self.pattern.len() {
                self.current_position = 0;  // Reset for next sequence
                return true;
            }
        } else if neuron_id == self.pattern[0] {
            // Start new sequence attempt
            self.current_position = 1;
            self.last_event_time = time;
        } else {
            // Reset on unexpected spike
            self.current_position = 0;
        }
        
        false
    }
}

// Example: Gesture recognition
fn main() {
    let mut gesture = SequenceDetector {
        pattern: vec![1, 3, 7, 2],    // "Swipe right" gesture
        current_position: 0,
        timing_window: 50.0,          // 50ms between touches
        last_event_time: 0.0,
    };
    
    // Wrong sequence: no recognition
    gesture.receive_spike(1, 0.0);    // false (start)
    gesture.receive_spike(2, 20.0);   // false (wrong next)
    
    // Correct sequence: gesture recognized!
    gesture.receive_spike(1, 100.0);  // false (start)
    gesture.receive_spike(3, 120.0);  // false (continue)
    gesture.receive_spike(7, 140.0);  // false (continue)
    gesture.receive_spike(2, 160.0);  // true! (complete)
}
```

**Applications:** Handwriting recognition, speech processing, behavior analysis.

### 4. Rhythm Detection: Finding Temporal Regularities

Regular patterns in time often indicate important signals.

```rust
pub struct RhythmDetector {
    recent_intervals: Vec<f64>,
    max_history: usize,
    tolerance: f64,
}

impl RhythmDetector {
    pub fn add_event(&mut self, time: f64) -> Option<f64> {
        if let Some(&last_time) = self.recent_intervals.last() {
            let interval = time - last_time;
            self.recent_intervals.push(time);
            
            if self.recent_intervals.len() > self.max_history {
                self.recent_intervals.remove(0);
            }
            
            // Check for rhythmic pattern
            self.detect_rhythm()
        } else {
            self.recent_intervals.push(time);
            None
        }
    }
    
    fn detect_rhythm(&self) -> Option<f64> {
        if self.recent_intervals.len() < 4 {
            return None;
        }
        
        // Calculate intervals between events
        let intervals: Vec<f64> = self.recent_intervals.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
            
        // Check if intervals are consistent (rhythmic)
        let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let variance = intervals.iter()
            .map(|&x| (x - mean_interval).powi(2))
            .sum::<f64>() / intervals.len() as f64;
            
        if variance.sqrt() < self.tolerance {
            Some(mean_interval)  // Rhythm detected with this period
        } else {
            None
        }
    }
}

// Example: Heartbeat monitor
fn main() {
    let mut heart_monitor = RhythmDetector {
        recent_intervals: Vec::new(),
        max_history: 10,
        tolerance: 5.0,  // 5ms tolerance
    };
    
    // Regular heartbeat: rhythm detected
    for beat in 0..8 {
        let time = beat as f64 * 800.0;  // 800ms intervals (75 BPM)
        
        if let Some(period) = heart_monitor.add_event(time) {
            println!("Heartbeat rhythm detected: {:.0} BPM", 60000.0 / period);
        }
    }
}
```

**Medical applications:** Arrhythmia detection, sleep stage monitoring, seizure prediction.

## Temporal Credit Assignment: Learning from Delayed Results

One of the hardest problems in machine learning: How do you know which action caused a result when there's a delay between action and outcome?

Traditional systems struggle with this:

```python
# Traditional approach: explicit temporal modeling
def assign_credit(actions, rewards, discount_factor):
    credits = [0] * len(actions)
    
    for i in range(len(actions)):
        for j in range(i, len(rewards)):
            delay = j - i
            credits[i] += rewards[j] * (discount_factor ** delay)
    
    return credits  # Complex calculation, artificial discounting
```

Neuromorphic systems solve this naturally through **eligibility traces**:

```rust
pub struct EligibilityTrace {
    trace_value: f64,
    decay_rate: f64,
    last_update: f64,
}

impl EligibilityTrace {
    pub fn mark_eligible(&mut self, time: f64) {
        self.trace_value = 1.0;  // This action is eligible for credit
        self.last_update = time;
    }
    
    pub fn apply_reward(&mut self, reward: f64, time: f64) -> f64 {
        // Decay the trace over time
        let dt = time - self.last_update;
        self.trace_value *= (-dt * self.decay_rate).exp();
        
        // Credit assignment: recent actions get more credit
        let credit = reward * self.trace_value;
        self.trace_value = 0.0;  // Clear trace after reward
        
        credit
    }
}

// Example: Delayed reward learning
fn main() {
    let mut action_trace = EligibilityTrace {
        trace_value: 0.0,
        decay_rate: 0.1,
        last_update: 0.0,
    };
    
    // Action at time 0
    action_trace.mark_eligible(0.0);
    
    // Reward comes 50ms later
    let credit = action_trace.apply_reward(1.0, 50.0);
    println!("Credit assigned: {:.3}", credit);  // ~0.006 (decayed appropriately)
}
```

**The insight:** Actions leave "traces" that decay over time. When rewards arrive, they strengthen recent actions more than distant ones. This happens automatically through temporal dynamics.

## Real-Time Processing Without Batching

Traditional AI systems process information in batches:

```python
# Traditional: collect, batch, process
batch = []
for data_point in stream:
    batch.append(data_point)
    
    if len(batch) >= BATCH_SIZE:
        results = model.process(batch)  # Process entire batch
        handle_results(results)
        batch = []  # Start new batch
```

**Problems:**
- Latency: Must wait for full batch
- Memory: Must store entire batch
- Complexity: Results don't correspond to input timing

Neuromorphic systems process continuously:

```rust
// Neuromorphic: process as events arrive
pub struct StreamProcessor {
    state: f64,
    threshold: f64,
}

impl StreamProcessor {
    pub fn process_event(&mut self, event: f64, time: f64) -> Option<f64> {
        // Update state immediately
        self.state = self.state * 0.95 + event * 0.05;
        
        // Generate output when threshold reached
        if self.state.abs() > self.threshold {
            let output = self.state;
            self.state = 0.0;  // Reset
            Some(output)  // Immediate result
        } else {
            None
        }
    }
}

// No batching, no latency, continuous operation
fn main() {
    let mut processor = StreamProcessor {
        state: 0.0,
        threshold: 1.0,
    };
    
    // Process events as they arrive (microsecond response times)
    for event_time in [0.1, 0.3, 0.7, 1.2, 2.5] {
        if let Some(result) = processor.process_event(0.3, event_time) {
            println!("Result at {:.1}ms: {:.2}", event_time, result);
        }
    }
}
```

**Advantages:**
- **Zero latency:** Results available immediately
- **Constant memory:** No batch storage required
- **Temporal fidelity:** Outputs maintain input timing relationships

## Temporal Multiplexing: Getting More from Less

Traditional systems allocate resources spatially—more cores, more memory. Neuromorphic systems can allocate resources temporally.

```rust
pub struct TemporalMultiplexer {
    channels: Vec<f64>,          // Multiple data channels
    current_channel: usize,      // Which channel is active
    switch_interval: f64,        // How often to switch
    last_switch: f64,
}

impl TemporalMultiplexer {
    pub fn process(&mut self, time: f64) -> Option<f64> {
        // Switch channels periodically
        if time - self.last_switch > self.switch_interval {
            self.current_channel = (self.current_channel + 1) % self.channels.len();
            self.last_switch = time;
        }
        
        // Process only the current channel
        let input = self.channels[self.current_channel];
        if input > 0.5 {
            Some(input * 2.0)  // Simple processing
        } else {
            None
        }
    }
}

// Example: Time-division processing of sensor array
fn main() {
    let mut mux = TemporalMultiplexer {
        channels: vec![0.8, 0.3, 0.9, 0.1, 0.7],  // 5 sensors
        current_channel: 0,
        switch_interval: 10.0,  // Switch every 10ms
        last_switch: 0.0,
    };
    
    // Single processor handles 5 sensors by switching rapidly
    for time in (0..100).map(|t| t as f64) {
        if let Some(result) = mux.process(time) {
            println!("Channel {} result at {}ms: {:.1}", 
                    mux.current_channel, time, result);
        }
    }
}
```

**Real applications:**
- **Sensor networks:** One processor time-shares multiple sensors
- **Communication:** Multiple data streams on one channel
- **Resource optimization:** High utilization through temporal sharing

## Emergent Timing: Complex Behaviors from Simple Rules

The most fascinating aspect of temporal computation: complex timing patterns can emerge from simple local rules.

```rust
use std::collections::HashMap;

pub struct TimingNetwork {
    neurons: HashMap<usize, TemporalNeuron>,
    connections: HashMap<usize, Vec<(usize, f64)>>,  // neuron -> [(target, weight)]
}

impl TimingNetwork {
    pub fn step(&mut self, time: f64) {
        let mut spike_events = Vec::new();
        
        // Update all neurons
        for (&neuron_id, neuron) in &mut self.neurons {
            if let Some(spike_time) = neuron.update(time) {
                spike_events.push((neuron_id, spike_time));
            }
        }
        
        // Propagate spikes
        for (source_id, spike_time) in spike_events {
            if let Some(targets) = self.connections.get(&source_id) {
                for &(target_id, weight) in targets {
                    if let Some(target_neuron) = self.neurons.get_mut(&target_id) {
                        target_neuron.receive_spike(spike_time + 1.0, weight);
                    }
                }
            }
        }
    }
}

struct TemporalNeuron {
    membrane_potential: f64,
    threshold: f64,
    leak_rate: f64,
    last_update: f64,
}

impl TemporalNeuron {
    fn update(&mut self, time: f64) -> Option<f64> {
        let dt = time - self.last_update;
        
        // Membrane potential leaks over time
        self.membrane_potential *= (-dt * self.leak_rate).exp();
        self.last_update = time;
        
        // Fire if threshold reached
        if self.membrane_potential > self.threshold {
            self.membrane_potential = 0.0;
            Some(time)
        } else {
            None
        }
    }
    
    fn receive_spike(&mut self, time: f64, weight: f64) {
        self.membrane_potential += weight;
    }
}
```

**Emergent patterns:**
- **Oscillations:** Networks can generate rhythmic activity
- **Synchronization:** Neurons can coordinate their timing
- **Waves:** Activity can propagate across networks
- **Resonance:** Networks can selectively respond to specific frequencies

## Practical Applications of Temporal Computing

### 1. Audio Processing
```rust
// Real-time beat detection
pub struct BeatDetector {
    energy_history: Vec<f64>,
    energy_threshold: f64,
}

impl BeatDetector {
    pub fn process_audio_frame(&mut self, audio_frame: &[f64], time: f64) -> bool {
        let energy: f64 = audio_frame.iter().map(|&x| x * x).sum();
        self.energy_history.push(energy);
        
        if self.energy_history.len() > 43 {  // ~1 second at 44.1kHz
            self.energy_history.remove(0);
        }
        
        // Beat detection: energy spike above recent average
        let recent_avg = self.energy_history.iter().sum::<f64>() / self.energy_history.len() as f64;
        energy > recent_avg * 1.5  // Beat detected!
    }
}
```

### 2. Motion Detection
```rust
// Event-based motion detection (like DVS cameras)
pub struct MotionDetector {
    last_pixel_time: HashMap<(u32, u32), f64>,
    motion_threshold: f64,
}

impl MotionDetector {
    pub fn pixel_event(&mut self, x: u32, y: u32, time: f64) -> bool {
        let pixel_coord = (x, y);
        
        if let Some(&last_time) = self.last_pixel_time.get(&pixel_coord) {
            let dt = time - last_time;
            self.last_pixel_time.insert(pixel_coord, time);
            
            // Motion detected if pixel changed recently
            dt < self.motion_threshold
        } else {
            self.last_pixel_time.insert(pixel_coord, time);
            false
        }
    }
}
```

### 3. Predictive Processing
```rust
// Predict when next event will occur
pub struct EventPredictor {
    event_intervals: Vec<f64>,
    prediction_window: usize,
}

impl EventPredictor {
    pub fn record_event(&mut self, time: f64) -> Option<f64> {
        static mut LAST_EVENT_TIME: f64 = 0.0;
        
        unsafe {
            if LAST_EVENT_TIME > 0.0 {
                let interval = time - LAST_EVENT_TIME;
                self.event_intervals.push(interval);
                
                if self.event_intervals.len() > self.prediction_window {
                    self.event_intervals.remove(0);
                }
                
                // Predict next event time
                let avg_interval = self.event_intervals.iter().sum::<f64>() 
                                 / self.event_intervals.len() as f64;
                Some(time + avg_interval)
            } else {
                LAST_EVENT_TIME = time;
                None
            }
        }
    }
}
```

## The Philosophy of Temporal Computing

Traditional computing asks: **"What is the answer?"**

Temporal computing asks: **"When will the answer emerge?"**

This shift in perspective opens up new possibilities:

- **Adaptive systems** that change behavior based on temporal patterns
- **Predictive systems** that anticipate future events
- **Efficient systems** that allocate resources based on temporal demand
- **Robust systems** that handle timing variations gracefully

## What You've Learned

Time in neuromorphic computing isn't just a coordinate—it's a computational medium that enables:

- **Integration** of evidence over time for robust decisions
- **Coincidence detection** for finding relationships
- **Sequence recognition** for temporal pattern learning
- **Credit assignment** for learning from delayed rewards
- **Real-time processing** without artificial batching
- **Emergent behaviors** from simple temporal rules

## What's Next?

You now understand how timing creates computation. But how do neuromorphic systems learn and adapt without forgetting what they already know?

**[Next: Learning Without Forgetting →](learning-without-forgetting.md)**

In the next chapter, we'll explore spike-timing dependent plasticity (STDP)—the learning rule that enables continuous adaptation without catastrophic forgetting. You'll discover why your brain can learn new things every day without losing old memories.

---

**Key Takeaways:**
- ⏰ **Time IS computation:** Timing patterns perform the computation itself
- 🔄 **Integration over time:** Evidence accumulates for robust decisions  
- 🎯 **Coincidence detection:** Simultaneous events indicate relationships
- 📈 **Sequence learning:** Temporal patterns encode complex information
- ⚡ **Real-time processing:** No batching, no latency, continuous operation
- 🌊 **Emergent timing:** Complex behaviors from simple temporal rules

**Try This:**
- Implement the temporal patterns in your favorite language
- Think about timing patterns in your own applications
- Consider how temporal multiplexing could optimize your systems
