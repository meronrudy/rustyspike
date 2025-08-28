//! Fixed-point arithmetic for deterministic neuromorphic computation
//!
//! This module provides ultra-fast, deterministic fixed-point arithmetic
//! optimized for embedded microcontrollers without floating-point units.

use core::{fmt, ops};

/// Q15.16 fixed-point number (15 integer bits, 16 fractional bits, 1 sign bit)
///
/// Range: [-32768.0, 32767.99998] with ~0.000015 precision
/// Optimized for common neural network values like weights (-1.0 to 1.0)
/// and membrane potentials (-100mV to 50mV).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Q15_16(i32);

impl Q15_16 {
    /// Number of fractional bits
    pub const FRAC_BITS: u32 = 16;
    /// Scale factor (2^16 = 65536)
    pub const SCALE: i32 = 1 << Self::FRAC_BITS;
    /// Maximum representable value
    pub const MAX: Self = Self(i32::MAX);
    /// Minimum representable value  
    pub const MIN: Self = Self(i32::MIN);
    /// Zero value
    pub const ZERO: Self = Self(0);
    /// One value
    pub const ONE: Self = Self(Self::SCALE);
    /// Negative one value
    pub const NEG_ONE: Self = Self(-Self::SCALE);
    
    /// Create from raw i32 value (for internal use)
    #[inline(always)]
    pub const fn from_raw(raw: i32) -> Self {
        Self(raw)
    }
    
    /// Get raw i32 value (for internal use)
    #[inline(always)]
    pub const fn to_raw(self) -> i32 {
        self.0
    }
    
    /// Create from integer value
    #[inline(always)]
    pub const fn from_int(value: i32) -> Self {
        // Saturating multiplication to prevent overflow
        if value > i32::MAX >> Self::FRAC_BITS {
            Self::MAX
        } else if value < i32::MIN >> Self::FRAC_BITS {
            Self::MIN
        } else {
            Self(value << Self::FRAC_BITS)
        }
    }
    
    /// Create from float (use sparingly, prefer compile-time constants)
    #[inline]
    pub fn from_float(value: f32) -> Self {
        if value >= 32767.0 {
            Self::MAX
        } else if value <= -32768.0 {
            Self::MIN
        } else {
            Self((value * Self::SCALE as f32) as i32)
        }
    }
    
    /// Convert to float (for interfacing with external systems)
    #[inline]
    pub fn to_float(self) -> f32 {
        self.0 as f32 / Self::SCALE as f32
    }
    
    /// Convert to integer (truncating fractional part)
    #[inline(always)]
    pub const fn to_int(self) -> i32 {
        self.0 >> Self::FRAC_BITS
    }
    
    /// Get fractional part as integer (0 to 65535)
    #[inline(always)]
    pub const fn frac_part(self) -> u16 {
        (self.0 & (Self::SCALE - 1)) as u16
    }
    
    /// Absolute value
    #[inline(always)]
    pub const fn abs(self) -> Self {
        if self.0 >= 0 {
            self
        } else if self.0 == i32::MIN {
            Self::MAX // Prevent overflow
        } else {
            Self(-self.0)
        }
    }
    
    /// Maximum of two values
    #[inline(always)]
    pub const fn max(self, other: Self) -> Self {
        if self.0 > other.0 { self } else { other }
    }
    
    /// Minimum of two values
    #[inline(always)]
    pub const fn min(self, other: Self) -> Self {
        if self.0 < other.0 { self } else { other }
    }
    
    /// Saturating addition (prevents overflow)
    #[inline(always)]
    pub const fn saturating_add(self, other: Self) -> Self {
        let result = self.0 as i64 + other.0 as i64;
        if result > i32::MAX as i64 {
            Self::MAX
        } else if result < i32::MIN as i64 {
            Self::MIN
        } else {
            Self(result as i32)
        }
    }
    
    /// Saturating subtraction (prevents overflow)
    #[inline(always)]
    pub const fn saturating_sub(self, other: Self) -> Self {
        let result = self.0 as i64 - other.0 as i64;
        if result > i32::MAX as i64 {
            Self::MAX
        } else if result < i32::MIN as i64 {
            Self::MIN
        } else {
            Self(result as i32)
        }
    }
    
    /// Saturating multiplication (prevents overflow)
    #[inline(always)]
    pub const fn saturating_mul(self, other: Self) -> Self {
        let result = (self.0 as i64 * other.0 as i64) >> Self::FRAC_BITS;
        if result > i32::MAX as i64 {
            Self::MAX
        } else if result < i32::MIN as i64 {
            Self::MIN
        } else {
            Self(result as i32)
        }
    }
    
    /// Fast multiplication without overflow checking (use with caution)
    #[inline(always)]
    pub const fn fast_mul(self, other: Self) -> Self {
        Self(((self.0 as i64 * other.0 as i64) >> Self::FRAC_BITS) as i32)
    }
    
    /// Division (may be slow on some embedded targets)
    #[inline]
    pub const fn div(self, other: Self) -> Self {
        if other.0 == 0 {
            if self.0 >= 0 { Self::MAX } else { Self::MIN }
        } else {
            let result = ((self.0 as i64) << Self::FRAC_BITS) / other.0 as i64;
            if result > i32::MAX as i64 {
                Self::MAX
            } else if result < i32::MIN as i64 {
                Self::MIN
            } else {
                Self(result as i32)
            }
        }
    }
    
    /// Fast reciprocal using Newton-Raphson iteration
    #[inline]
    pub fn fast_recip(self) -> Self {
        if self.0 == 0 {
            return Self::MAX;
        }
        
        // Initial guess using bit manipulation
        let x = self.abs();
        let mut r = if x.0 < Self::SCALE {
            Self::from_int(2) // 1/x where x < 1, start with 2
        } else {
            Self::ONE.div(x) // Rough division for initial guess
        };
        
        // One Newton-Raphson iteration: r = r * (2 - x * r)
        let two = Self::from_int(2);
        r = r.saturating_mul(two.saturating_sub(x.saturating_mul(r)));
        
        if self.0 < 0 { r.neg() } else { r }
    }
    
    /// Negate value
    #[inline(always)]
    pub const fn neg(self) -> Self {
        if self.0 == i32::MIN {
            Self::MAX // Prevent overflow
        } else {
            Self(-self.0)
        }
    }
    
    /// Shift left (multiply by power of 2)
    #[inline(always)]
    pub const fn shl(self, bits: u32) -> Self {
        if bits >= 31 {
            if self.0 >= 0 { Self::MAX } else { Self::MIN }
        } else {
            let result = (self.0 as i64) << bits;
            if result > i32::MAX as i64 {
                Self::MAX
            } else if result < i32::MIN as i64 {
                Self::MIN
            } else {
                Self(result as i32)
            }
        }
    }
    
    /// Shift right (divide by power of 2)
    #[inline(always)]
    pub const fn shr(self, bits: u32) -> Self {
        if bits >= 32 {
            Self::ZERO
        } else {
            Self(self.0 >> bits)
        }
    }
}

// Implement common traits
impl fmt::Display for Q15_16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.to_float())
    }
}

impl ops::Add for Q15_16 {
    type Output = Self;
    
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
}

impl ops::Sub for Q15_16 {
    type Output = Self;
    
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        self.saturating_sub(other)
    }
}

impl ops::Mul for Q15_16 {
    type Output = Self;
    
    #[inline(always)]
    fn mul(self, other: Self) -> Self {
        self.saturating_mul(other)
    }
}

impl ops::Div for Q15_16 {
    type Output = Self;
    
    #[inline]
    fn div(self, other: Self) -> Self {
        self.div(other)
    }
}

impl ops::Neg for Q15_16 {
    type Output = Self;
    
    #[inline(always)]
    fn neg(self) -> Self {
        self.neg()
    }
}

impl ops::AddAssign for Q15_16 {
    #[inline(always)]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl ops::SubAssign for Q15_16 {
    #[inline(always)]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl ops::MulAssign for Q15_16 {
    #[inline(always)]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl ops::DivAssign for Q15_16 {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl From<i32> for Q15_16 {
    #[inline(always)]
    fn from(value: i32) -> Self {
        Self::from_int(value)
    }
}

impl From<Q15_16> for f32 {
    #[inline]
    fn from(value: Q15_16) -> Self {
        value.to_float()
    }
}

impl Default for Q15_16 {
    #[inline(always)]
    fn default() -> Self {
        Self::ZERO
    }
}

/// Type alias for easier use
pub type FixedPoint = Q15_16;

/// Common neuromorphic constants in fixed-point format
pub mod constants {
    use super::Q15_16;
    
    /// Membrane potential threshold (typically -50mV)
    pub const THRESHOLD: Q15_16 = Q15_16::from_raw(-50 * Q15_16::SCALE / 1000);
    
    /// Resting potential (typically -70mV)
    pub const RESTING: Q15_16 = Q15_16::from_raw(-70 * Q15_16::SCALE / 1000);
    
    /// Reset potential (typically -80mV)
    pub const RESET: Q15_16 = Q15_16::from_raw(-80 * Q15_16::SCALE / 1000);
    
    /// Membrane time constant decay factor (exp(-dt/tau) for dt=1ms, tau=20ms)
    pub const MEMBRANE_DECAY: Q15_16 = Q15_16::from_raw(62259); // ≈ 0.951
    
    /// Synaptic time constant decay factor (exp(-dt/tau) for dt=1ms, tau=5ms)
    pub const SYNAPTIC_DECAY: Q15_16 = Q15_16::from_raw(53740); // ≈ 0.820
    
    /// Small epsilon for numerical stability
    pub const EPSILON: Q15_16 = Q15_16::from_raw(1); // Smallest positive value
    
    /// STDP learning rate
    pub const LEARNING_RATE: Q15_16 = Q15_16::from_raw(655); // ≈ 0.01
    
    /// Maximum weight value
    pub const MAX_WEIGHT: Q15_16 = Q15_16::ONE;
    
    /// Minimum weight value
    pub const MIN_WEIGHT: Q15_16 = Q15_16::ZERO;
}

/// Fast mathematical functions using lookup tables and approximations
impl Q15_16 {
    /// Fast exponential approximation using lookup table
    pub fn fast_exp(self) -> Self {
        // Clamp input to reasonable range
        if self.0 < -5 * Self::SCALE {
            return Self::ZERO;
        }
        if self.0 > 5 * Self::SCALE {
            return Self::from_int(148); // e^5 ≈ 148
        }
        
        // For very small values, use linear approximation: e^x ≈ 1 + x
        if self.abs().0 < Self::SCALE / 4 {
            return Self::ONE + self;
        }
        
        // Quadratic approximation for moderate values: e^x ≈ 1 + x + x²/2
        let x = self;
        let x_squared = x.fast_mul(x);
        Self::ONE + x + x_squared.shr(1)
    }
    
    /// Fast sigmoid approximation: σ(x) ≈ x / (1 + |x|)
    pub fn fast_sigmoid(self) -> Self {
        let abs_x = self.abs();
        let denominator = Self::ONE + abs_x;
        self.div(denominator)
    }
    
    /// Fast tanh approximation: tanh(x) ≈ x / (1 + |x|/2)
    pub fn fast_tanh(self) -> Self {
        let abs_x = self.abs();
        let denominator = Self::ONE + abs_x.shr(1);
        self.div(denominator)
    }
    
    /// ReLU activation function
    #[inline(always)]
    pub fn relu(self) -> Self {
        if self.0 > 0 { self } else { Self::ZERO }
    }
    
    /// Leaky ReLU with fixed alpha = 0.01
    #[inline(always)]
    pub fn leaky_relu(self) -> Self {
        if self.0 > 0 { 
            self 
        } else { 
            // Multiply by 0.01 (shift right by ~7 bits for approximation)
            Self(self.0 >> 7)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_arithmetic() {
        let a = Q15_16::from_float(1.5);
        let b = Q15_16::from_float(2.5);
        
        let sum = a + b;
        assert!((sum.to_float() - 4.0).abs() < 0.01);
        
        let product = a * b;
        assert!((product.to_float() - 3.75).abs() < 0.01);
    }
    
    #[test]
    fn test_constants() {
        assert!(constants::THRESHOLD.to_float() < 0.0);
        assert!(constants::RESTING.to_float() < constants::THRESHOLD.to_float());
        assert!(constants::MEMBRANE_DECAY.to_float() < 1.0);
        assert!(constants::MEMBRANE_DECAY.to_float() > 0.9);
    }
    
    #[test]
    fn test_fast_functions() {
        let x = Q15_16::from_float(0.5);
        let exp_x = x.fast_exp();
        assert!(exp_x.to_float() > 1.0);
        assert!(exp_x.to_float() < 2.0);
        
        let sigmoid_x = x.fast_sigmoid();
        assert!(sigmoid_x.to_float() > 0.0);
        assert!(sigmoid_x.to_float() < 1.0);
    }
    
    #[test]
    fn test_saturation() {
        let max_val = Q15_16::from_int(30000);
        let large_val = Q15_16::from_int(10000);
        
        let sum = max_val.saturating_add(large_val);
        assert_eq!(sum, Q15_16::MAX);
    }
    
    #[test]
    fn test_neuromorphic_range() {
        // Test typical membrane potential range
        let v_rest = constants::RESTING;
        let v_thresh = constants::THRESHOLD;
        let v_reset = constants::RESET;
        
        assert!(v_reset < v_rest);
        assert!(v_rest < v_thresh);
        assert!(v_thresh.to_float() > -0.1); // -100mV in volts
        assert!(v_thresh.to_float() < 0.1);   // +100mV in volts
    }
}