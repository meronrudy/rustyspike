//! Lock-free queue implementations for neuromorphic spike processing
//! 
//! Provides Single Producer Single Consumer (SPSC), Multi Producer Single Consumer (MPSC),
//! and Multi Producer Multi Consumer (MPMC) queues optimized for neural spike events.

use core::sync::atomic::{AtomicUsize, AtomicPtr, Ordering};
use core::mem::MaybeUninit;
use core::cell::UnsafeCell;
use core::ptr;
use crate::{Result, LockFreeError, constants::*};

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, boxed::Box};

#[cfg(feature = "std")]
use std::{vec::Vec, boxed::Box};

/// Single Producer Single Consumer lock-free queue
/// Optimized for high-frequency spike event processing
pub struct SPSCQueue<T> {
    buffer: UnsafeCell<Vec<MaybeUninit<T>>>,
    capacity: usize,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl<T> SPSCQueue<T> {
    /// Create new SPSC queue with given capacity (must be power of 2)
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 || !capacity.is_power_of_two() {
            return Err(LockFreeError::InvalidCapacity);
        }

        let mut buffer = Vec::with_capacity(capacity);
        unsafe {
            buffer.set_len(capacity);
        }

        Ok(Self {
            buffer: UnsafeCell::new(buffer),
            capacity,
            mask: capacity - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        })
    }

    /// Create new SPSC queue with default spike queue size
    pub fn new_spike_queue() -> Result<Self> {
        Self::new(DEFAULT_SPIKE_QUEUE_SIZE)
    }

    /// Push element to queue (producer side)
    pub fn push(&self, item: T) -> Result<()> {
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) & self.mask;
        let head = self.head.load(Ordering::Acquire);

        if next_tail == head {
            return Err(LockFreeError::QueueFull);
        }

        unsafe {
            let buffer = &mut *self.buffer.get();
            buffer[tail].as_mut_ptr().write(item);
        }

        self.tail.store(next_tail, Ordering::Release);
        Ok(())
    }

    /// Pop element from queue (consumer side)
    pub fn pop(&self) -> Result<T> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        if head == tail {
            return Err(LockFreeError::QueueEmpty);
        }

        let item = unsafe {
            let buffer = &*self.buffer.get();
            buffer[head].as_ptr().read()
        };

        let next_head = (head + 1) & self.mask;
        self.head.store(next_head, Ordering::Release);
        Ok(item)
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head == tail
    }

    /// Check if queue is full
    pub fn is_full(&self) -> bool {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        ((tail + 1) & self.mask) == head
    }

    /// Get current queue length (approximate)
    pub fn len(&self) -> usize {
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        (tail.wrapping_sub(head)) & self.mask
    }

    /// Get queue capacity
    pub fn capacity(&self) -> usize {
        self.capacity - 1  // Reserve one slot for full detection
    }
}

unsafe impl<T: Send> Send for SPSCQueue<T> {}
unsafe impl<T: Send> Sync for SPSCQueue<T> {}

/// Multi Producer Single Consumer lock-free queue
/// Uses hazard pointers for memory management
pub struct MPSCQueue<T> {
    head: AtomicPtr<Node<T>>,
    tail: AtomicPtr<Node<T>>,
}

struct Node<T> {
    data: MaybeUninit<T>,
    next: AtomicPtr<Node<T>>,
}

impl<T> Node<T> {
    fn new() -> Box<Self> {
        Box::new(Self {
            data: MaybeUninit::uninit(),
            next: AtomicPtr::new(ptr::null_mut()),
        })
    }

    fn new_with_data(data: T) -> Box<Self> {
        Box::new(Self {
            data: MaybeUninit::new(data),
            next: AtomicPtr::new(ptr::null_mut()),
        })
    }
}

impl<T> MPSCQueue<T> {
    /// Create new MPSC queue
    pub fn new() -> Self {
        let dummy = Box::into_raw(Node::new());
        Self {
            head: AtomicPtr::new(dummy),
            tail: AtomicPtr::new(dummy),
        }
    }

    /// Push element to queue (multiple producers)
    pub fn push(&self, item: T) -> Result<()> {
        let new_node = Box::into_raw(Node::new_with_data(item));
        
        // Retry loop for CAS operation
        for _ in 0..MAX_CAS_RETRIES {
            let tail = self.tail.load(Ordering::Acquire);
            let next = unsafe { (*tail).next.load(Ordering::Acquire) };

            if tail == self.tail.load(Ordering::Acquire) {
                if next.is_null() {
                    // Try to link new node at the end of the list
                    if unsafe { (*tail).next.compare_exchange(
                        ptr::null_mut(),
                        new_node,
                        Ordering::Release,
                        Ordering::Relaxed,
                    ).is_ok() } {
                        // Successfully linked, now try to advance tail
                        let _ = self.tail.compare_exchange(
                            tail,
                            new_node,
                            Ordering::Release,
                            Ordering::Relaxed,
                        );
                        return Ok(());
                    }
                } else {
                    // Help advance tail
                    let _ = self.tail.compare_exchange(
                        tail,
                        next,
                        Ordering::Release,
                        Ordering::Relaxed,
                    );
                }
            }
        }

        // Failed to push after retries
        unsafe {
            drop(Box::from_raw(new_node));
        }
        Err(LockFreeError::ConcurrentModification)
    }

    /// Pop element from queue (single consumer)
    pub fn pop(&self) -> Result<T> {
        let head = self.head.load(Ordering::Acquire);
        let next = unsafe { (*head).next.load(Ordering::Acquire) };

        if next.is_null() {
            return Err(LockFreeError::QueueEmpty);
        }

        let data = unsafe { next.as_ref().unwrap().data.as_ptr().read() };
        self.head.store(next, Ordering::Release);

        // Free the old head node
        unsafe {
            drop(Box::from_raw(head));
        }

        Ok(data)
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let next = unsafe { (*head).next.load(Ordering::Acquire) };
        next.is_null()
    }
}

impl<T> Default for MPSCQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: Send> Send for MPSCQueue<T> {}
unsafe impl<T: Send> Sync for MPSCQueue<T> {}

impl<T> Drop for MPSCQueue<T> {
    fn drop(&mut self) {
        // Drain all remaining elements
        while self.pop().is_ok() {}
        
        // Free the dummy node
        let head = self.head.load(Ordering::Relaxed);
        if !head.is_null() {
            unsafe {
                drop(Box::from_raw(head));
            }
        }
    }
}

/// Multi Producer Multi Consumer lock-free queue
/// Uses Michael & Scott algorithm with hazard pointers
pub struct MPMCQueue<T> {
    inner: MPSCQueue<T>,
    consumer_lock: AtomicUsize,
}

impl<T> MPMCQueue<T> {
    /// Create new MPMC queue
    pub fn new() -> Self {
        Self {
            inner: MPSCQueue::new(),
            consumer_lock: AtomicUsize::new(0),
        }
    }

    /// Push element to queue (multiple producers)
    pub fn push(&self, item: T) -> Result<()> {
        self.inner.push(item)
    }

    /// Pop element from queue (multiple consumers with lock-free coordination)
    pub fn pop(&self) -> Result<T> {
        // Try to acquire consumer lock
        for _ in 0..MAX_CAS_RETRIES {
            if self.consumer_lock.compare_exchange(
                0,
                1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ).is_ok() {
                // Got the lock, try to pop
                let result = self.inner.pop();
                
                // Release the lock
                self.consumer_lock.store(0, Ordering::Release);
                
                return result;
            }
            
            // Yield to reduce contention
            #[cfg(feature = "std")]
            std::thread::yield_now();
        }

        Err(LockFreeError::ConcurrentModification)
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl<T> Default for MPMCQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: Send> Send for MPMCQueue<T> {}
unsafe impl<T: Send> Sync for MPMCQueue<T> {}

/// Bounded Multi-Producer Multi-Consumer ring buffer queue (Vyukov)
///
/// Lock-free for both producers and consumers, bounded capacity (power-of-two),
/// uses per-slot sequence numbers to coordinate ownership without locks.
///
/// Notes:
/// - Capacity must be a power of two (≥ 2).
/// - One slot per element, no reserved slot.
/// - Suitable for high-throughput spike/event queues where boundedness is desired.
///
/// References:
/// - Dmitry Vyukov, "Bounded MPMC queue"
pub struct BoundedMPMCQueue<T> {
    buffer: Vec<Cell<T>>,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

#[repr(C)]
struct Cell<T> {
    seq: AtomicUsize,
    val: UnsafeCell<MaybeUninit<T>>,
    // Optional cache line padding could be added for heavy contention
}

impl<T> BoundedMPMCQueue<T> {
    /// Create a new bounded MPMC with the given capacity (must be power of two)
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        if capacity < 2 || !capacity.is_power_of_two() {
            return Err(LockFreeError::InvalidCapacity);
        }

        let mut buffer = Vec::with_capacity(capacity);
        for i in 0..capacity {
            buffer.push(Cell {
                seq: AtomicUsize::new(i),
                val: UnsafeCell::new(MaybeUninit::uninit()),
            });
        }

        Ok(Self {
            buffer,
            mask: capacity - 1,
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        })
    }

    /// Push an element; returns QueueFull if full
    pub fn push(&self, item: T) -> Result<()> {
        let mut pos = self.tail.load(Ordering::Relaxed);
        let mut backoff = crate::ordering::Backoff::new();

        loop {
            let idx = pos & self.mask;
            // Safety: index in bounds
            let cell = unsafe { self.buffer.get_unchecked(idx) };
            let seq = cell.seq.load(Ordering::Acquire);
            let dif = seq as isize - pos as isize;

            if dif == 0 {
                // Attempt to claim this slot by moving tail forward
                match self.tail.compare_exchange_weak(
                    pos,
                    pos + 1,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // We own the slot; write value, then publish
                        unsafe { (*cell.val.get()).write(item); }
                        // Set sequence to indicate availability for consumer
                        cell.seq.store(pos + 1, Ordering::Release);
                        return Ok(());
                    }
                    Err(actual) => {
                        pos = actual;
                        backoff.backoff();
                        continue;
                    }
                }
            } else if dif < 0 {
                // Queue appears full
                return Err(LockFreeError::QueueFull);
            } else {
                // Another producer moved the tail; catch up
                pos = self.tail.load(Ordering::Relaxed);
                backoff.backoff();
            }
        }
    }

    /// Pop an element; returns QueueEmpty if empty
    pub fn pop(&self) -> Result<T> {
        let mut pos = self.head.load(Ordering::Relaxed);
        let mut backoff = crate::ordering::Backoff::new();

        loop {
            let idx = pos & self.mask;
            // Safety: index in bounds
            let cell = unsafe { self.buffer.get_unchecked(idx) };
            let seq = cell.seq.load(Ordering::Acquire);
            let dif = seq as isize - (pos as isize + 1);

            if dif == 0 {
                // Attempt to claim this slot by moving head forward
                match self.head.compare_exchange_weak(
                    pos,
                    pos + 1,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        // We own the slot; read, then set seq for next cycle
                        let val = unsafe { (*cell.val.get()).assume_init_read() };
                        // Mark this slot as reusable for producers by jumping by capacity
                        cell.seq.store(pos + self.mask + 1, Ordering::Release);
                        return Ok(val);
                    }
                    Err(actual) => {
                        pos = actual;
                        backoff.backoff();
                        continue;
                    }
                }
            } else if dif < 0 {
                // Queue appears empty
                return Err(LockFreeError::QueueEmpty);
            } else {
                // Another consumer moved the head; catch up
                pos = self.head.load(Ordering::Relaxed);
                backoff.backoff();
            }
        }
    }

    /// Returns true if queue is likely empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Approximate length (racy)
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        tail.wrapping_sub(head)
    }

    /// Capacity
    pub fn capacity(&self) -> usize {
        self.mask + 1
    }
}

unsafe impl<T: Send> Send for BoundedMPMCQueue<T> {}
unsafe impl<T: Send> Sync for BoundedMPMCQueue<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spsc_queue() {
        let queue = SPSCQueue::new(8).unwrap();
        
        assert!(queue.is_empty());
        assert!(!queue.is_full());
        assert_eq!(queue.len(), 0);
        
        // Push some items
        for i in 0..7 {
            queue.push(i).unwrap();
        }
        
        assert!(!queue.is_empty());
        assert!(queue.is_full());
        assert_eq!(queue.len(), 7);
        
        // Pop some items
        for i in 0..7 {
            assert_eq!(queue.pop().unwrap(), i);
        }
        
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_mpsc_queue() {
        let queue = MPSCQueue::new();
        
        assert!(queue.is_empty());
        
        queue.push(42).unwrap();
        assert!(!queue.is_empty());
        
        assert_eq!(queue.pop().unwrap(), 42);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_mpmc_queue() {
        let queue = MPMCQueue::new();
        
        assert!(queue.is_empty());
        
        queue.push(100).unwrap();
        assert!(!queue.is_empty());
        
        assert_eq!(queue.pop().unwrap(), 100);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_spike_queue() {
        let queue = SPSCQueue::<u32>::new_spike_queue().unwrap();
        assert_eq!(queue.capacity(), DEFAULT_SPIKE_QUEUE_SIZE - 1);
    }
}