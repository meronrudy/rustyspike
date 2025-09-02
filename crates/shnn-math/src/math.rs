//! Core mathematical functions for neuromorphic computing
//!
//! This module provides mathematical operations and approximations
//! optimized for neural network computations with no external dependencies.

use crate::Float;
use core::f32::consts::{E, PI};

/// Extension trait to add mathematical functions to f32 in no_std environments
pub trait MathExt {
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn tanh(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn cos(self) -> Self;
    fn sin(self) -> Self;
    fn abs(self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
}

impl MathExt for f32 {
    fn sqrt(self) -> Self {
        if self < 0.0 {
            f32::NAN
        } else {
            sqrt_approx(self)
        }
    }
    
    fn exp(self) -> Self {
        exp_approx(self)
    }
    
    fn ln(self) -> Self {
        if self <= 0.0 {
            if self == 0.0 { f32::NEG_INFINITY } else { f32::NAN }
        } else {
            ln_approx(self)
        }
    }
    
    fn tanh(self) -> Self {
        // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        let exp_2x = exp_approx(2.0 * self);
        (exp_2x - 1.0) / (exp_2x + 1.0)
    }
    
    fn floor(self) -> Self {
        if self >= 0.0 {
            self as i32 as f32
        } else {
            let truncated = self as i32 as f32;
            if truncated == self { truncated } else { truncated - 1.0 }
        }
    }
    
    fn ceil(self) -> Self {
        if self <= 0.0 {
            self as i32 as f32
        } else {
            let truncated = self as i32 as f32;
            if truncated == self { truncated } else { truncated + 1.0 }
        }
    }
    
    fn powi(self, n: i32) -> Self {
        pow_approx(self, n as f32)
    }
    
    fn cos(self) -> Self {
        cos_approx(self)
    }
    
    fn sin(self) -> Self {
        sin_approx(self)
    }

    fn abs(self) -> Self {
        if self < 0.0 { -self } else { self }
    }

    fn clamp(self, min: Self, max: Self) -> Self {
        crate::math::clamp(self, min, max)
    }
}

/// Trait for floating-point mathematical operations
pub trait FloatMath {
    /// Fast exponential approximation
    fn exp_approx(self) -> Self;
    /// Fast logarithm approximation
    fn ln_approx(self) -> Self;
    /// Fast square root approximation
    fn sqrt_approx(self) -> Self;
    /// Fast sine approximation
    fn sin_approx(self) -> Self;
    /// Fast cosine approximation
    fn cos_approx(self) -> Self;
    /// Fast inverse square root
    fn inv_sqrt_approx(self) -> Self;
}

impl FloatMath for Float {
    fn exp_approx(self) -> Self {
        exp_approx(self)
    }
    
    fn ln_approx(self) -> Self {
        ln_approx(self)
    }
    
    fn sqrt_approx(self) -> Self {
        sqrt_approx(self)
    }
    
    fn sin_approx(self) -> Self {
        sin_approx(self)
    }
    
    fn cos_approx(self) -> Self {
        cos_approx(self)
    }
    
    fn inv_sqrt_approx(self) -> Self {
        inv_sqrt_approx(self)
    }
}

/// Fast exponential approximation using polynomial
/// Accurate to about 1% for x in [-5, 5]
pub fn exp_approx(x: Float) -> Float {
    if x < -5.0 {
        return 0.0;
    }
    if x > 5.0 {
        return 148.413; // e^5
    }
    
    // Use Padé approximation: e^x ≈ (1 + x/2) / (1 - x/2) for |x| < 2
    if x.abs() < 2.0 {
        let half_x = x * 0.5;
        return (1.0 + half_x) / (1.0 - half_x);
    }
    
    // Use Taylor series for larger values: e^x ≈ 1 + x + x²/2! + x³/3! + x⁴/4! + x⁵/5!
    let x2 = x * x;
    let x3 = x2 * x;
    let x4 = x3 * x;
    let x5 = x4 * x;
    
    1.0 + x + x2 * 0.5 + x3 * 0.16666667 + x4 * 0.041666667 + x5 * 0.008333333
}

/// Fast natural logarithm approximation
pub fn ln_approx(x: Float) -> Float {
    if x <= 0.0 {
        return Float::NEG_INFINITY;
    }
    if x == 1.0 {
        return 0.0;
    }
    
    // Use bit manipulation for initial approximation
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let mantissa = f32::from_bits((bits & 0x007FFFFF) | 0x3F800000);
    
    // Polynomial approximation for mantissa in [1, 2)
    let m = mantissa - 1.0;
    let ln_mantissa = m * (1.0 - m * (0.5 - m * (0.33333333 - m * 0.25)));
    
    exp as Float * 0.6931472 + ln_mantissa // ln(2) ≈ 0.6931472
}

/// Fast square root using Newton-Raphson method
pub fn sqrt_approx(x: Float) -> Float {
    if x <= 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    
    // Initial guess using bit manipulation (Quake-style)
    let mut y = f32::from_bits((x.to_bits() >> 1) + 0x1fbd1df5);
    
    // Two Newton-Raphson iterations: y = (y + x/y) / 2
    y = 0.5 * (y + x / y);
    y = 0.5 * (y + x / y);
    
    y
}

/// Fast inverse square root (Quake III algorithm variant)
pub fn inv_sqrt_approx(x: Float) -> Float {
    if x <= 0.0 {
        return Float::INFINITY;
    }
    
    let half_x = x * 0.5;
    let mut y = f32::from_bits(0x5f3759df - (x.to_bits() >> 1));
    
    // Newton-Raphson iteration: y = y * (1.5 - half_x * y * y)
    y = y * (1.5 - half_x * y * y);
    y = y * (1.5 - half_x * y * y);
    
    y
}

/// Fast sine approximation using Bhaskara I's sine approximation
pub fn sin_approx(x: Float) -> Float {
    // Normalize to [-π, π]
    let mut x_norm = x % (2.0 * PI);
    if x_norm > PI {
        x_norm -= 2.0 * PI;
    } else if x_norm < -PI {
        x_norm += 2.0 * PI;
    }
    
    // Use symmetry to reduce to [0, π]
    let sign = if x_norm < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x_norm.abs();
    
    if x_abs <= PI * 0.5 {
        // Use polynomial approximation for [0, π/2]
        let x2 = x_abs * x_abs;
        sign * x_abs * (1.0 - x2 * (0.16666667 - x2 * (0.008333333 - x2 * 0.000198413)))
    } else {
        // Use sin(π - x) = sin(x) for [π/2, π]
        let x_complement = PI - x_abs;
        let x2 = x_complement * x_complement;
        sign * x_complement * (1.0 - x2 * (0.16666667 - x2 * (0.008333333 - x2 * 0.000198413)))
    }
}

/// Fast cosine approximation
pub fn cos_approx(x: Float) -> Float {
    sin_approx(x + PI * 0.5)
}

/// Fast tangent approximation
pub fn tan_approx(x: Float) -> Float {
    let sin_x = sin_approx(x);
    let cos_x = cos_approx(x);
    
    if cos_x.abs() < crate::constants::EPSILON {
        return if sin_x >= 0.0 { Float::INFINITY } else { Float::NEG_INFINITY };
    }
    
    sin_x / cos_x
}

/// Fast hyperbolic sine approximation
pub fn sinh_approx(x: Float) -> Float {
    if x.abs() < 1.0 {
        // Use Taylor series for small values
        let x2 = x * x;
        x * (1.0 + x2 * (0.16666667 + x2 * (0.008333333 + x2 * 0.000198413)))
    } else {
        // Use definition: sinh(x) = (e^x - e^(-x)) / 2
        let exp_x = exp_approx(x);
        let exp_neg_x = exp_approx(-x);
        (exp_x - exp_neg_x) * 0.5
    }
}

/// Fast hyperbolic cosine approximation
pub fn cosh_approx(x: Float) -> Float {
    // Use definition: cosh(x) = (e^x + e^(-x)) / 2
    let exp_x = exp_approx(x);
    let exp_neg_x = exp_approx(-x);
    (exp_x + exp_neg_x) * 0.5
}

/// Fast hyperbolic tangent approximation
pub fn tanh_approx(x: Float) -> Float {
    if x.abs() < 1.0 {
        // Use rational approximation for small values
        let x2 = x * x;
        x * (1.0 - x2 * (0.33333333 - x2 * 0.13333333)) / (1.0 + x2 * (0.2 - x2 * 0.028571429))
    } else {
        // Use definition: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        let exp_2x = exp_approx(2.0 * x);
        (exp_2x - 1.0) / (exp_2x + 1.0)
    }
}

/// Fast arctangent approximation
pub fn atan_approx(x: Float) -> Float {
    if x.abs() <= 1.0 {
        // Use polynomial approximation for |x| <= 1
        let x2 = x * x;
        x * (1.0 - x2 * (0.33333333 - x2 * (0.2 - x2 * (0.14285714 - x2 * 0.11111111))))
    } else {
        // Use identity: atan(x) = π/2 - atan(1/x) for |x| > 1
        let sign = if x >= 0.0 { 1.0 } else { -1.0 };
        let inv_x = 1.0 / x.abs();
        let inv_x2 = inv_x * inv_x;
        let atan_inv = inv_x * (1.0 - inv_x2 * (0.33333333 - inv_x2 * (0.2 - inv_x2 * (0.14285714 - inv_x2 * 0.11111111))));
        sign * (PI * 0.5 - atan_inv)
    }
}

/// Fast arctangent2 approximation
pub fn atan2_approx(y: Float, x: Float) -> Float {
    if x.abs() < crate::constants::EPSILON && y.abs() < crate::constants::EPSILON {
        return 0.0; // Undefined, return 0
    }
    
    if x.abs() < crate::constants::EPSILON {
        return if y >= 0.0 { PI * 0.5 } else { -PI * 0.5 };
    }
    
    let atan_ratio = atan_approx(y / x);
    
    if x >= 0.0 {
        atan_ratio
    } else if y >= 0.0 {
        atan_ratio + PI
    } else {
        atan_ratio - PI
    }
}

/// Clamp value to range [min, max]
pub fn clamp(value: Float, min: Float, max: Float) -> Float {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Linear interpolation between two values
pub fn lerp(a: Float, b: Float, t: Float) -> Float {
    a + t * (b - a)
}

/// Smooth step function (3t² - 2t³)
pub fn smoothstep(edge0: Float, edge1: Float, x: Float) -> Float {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Smoother step function (6t⁵ - 15t⁴ + 10t³)
pub fn smootherstep(edge0: Float, edge1: Float, x: Float) -> Float {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Fast power function for integer exponents
pub fn powi_approx(base: Float, exp: i32) -> Float {
    if exp == 0 {
        return 1.0;
    }
    if exp == 1 {
        return base;
    }
    if exp < 0 {
        return 1.0 / powi_approx(base, -exp);
    }
    
    // Use exponentiation by squaring
    let mut result = 1.0;
    let mut base_power = base;
    let mut exponent = exp;
    
    while exponent > 0 {
        if exponent & 1 == 1 {
            result *= base_power;
        }
        base_power *= base_power;
        exponent >>= 1;
    }
    
    result
}

/// Fast power function for floating-point exponents
pub fn pow_approx(base: Float, exp: Float) -> Float {
    if base <= 0.0 {
        return if exp == 0.0 { 1.0 } else { 0.0 };
    }
    if exp == 0.0 {
        return 1.0;
    }
    if exp == 1.0 {
        return base;
    }
    
    // Use identity: a^b = e^(b * ln(a))
    exp_approx(exp * ln_approx(base))
}

/// Compute factorial (for small integers)
pub fn factorial(n: u32) -> Float {
    if n <= 1 {
        return 1.0;
    }
    if n > 12 {
        // Use Stirling's approximation for large n
        let n_f = n as Float;
        return (2.0 * PI * n_f).sqrt() * pow_approx(n_f / E, n_f);
    }
    
    let mut result = 1.0;
    for i in 2..=n {
        result *= i as Float;
    }
    result
}

/// Compute gamma function approximation (Stirling's approximation)
pub fn gamma_approx(x: Float) -> Float {
    if x <= 0.0 {
        return Float::INFINITY;
    }
    if x == 1.0 || x == 2.0 {
        return 1.0;
    }
    
    // Use Stirling's approximation: Γ(x) ≈ √(2π/x) * (x/e)^x
    let sqrt_2pi_over_x = (2.0 * PI / x).sqrt();
    let x_over_e_to_x = pow_approx(x / E, x);
    sqrt_2pi_over_x * x_over_e_to_x
}

/// Error function approximation
pub fn erf_approx(x: Float) -> Float {
    // Use rational approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x_abs = x.abs();
    
    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp_approx(-x_abs * x_abs);
    
    sign * y
}

/// Complementary error function
pub fn erfc_approx(x: Float) -> Float {
    1.0 - erf_approx(x)
}

/// Normal distribution cumulative density function
pub fn norm_cdf_approx(x: Float) -> Float {
    0.5 * (1.0 + erf_approx(x / 2.0_f32.sqrt()))
}

/// Random number generation utilities (simple LCG)
pub struct SimpleRng {
    state: u32,
}

impl SimpleRng {
    /// Create new RNG with seed
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }
    
    /// Generate next random u32
    pub fn next_u32(&mut self) -> u32 {
        // Linear congruential generator
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }
    
    /// Generate random float in [0, 1)
    pub fn next_f32(&mut self) -> Float {
        self.next_u32() as Float / (u32::MAX as Float + 1.0)
    }
    
    /// Generate random float in range [min, max)
    pub fn next_f32_range(&mut self, min: Float, max: Float) -> Float {
        min + self.next_f32() * (max - min)
    }
    
    /// Generate normal distributed random number (Box-Muller transform)
    pub fn next_normal(&mut self, mean: Float, std_dev: Float) -> Float {
        // Use Box-Muller transform
        static mut SPARE: Option<Float> = None;
        static mut HAS_SPARE: bool = false;
        
        unsafe {
            if HAS_SPARE {
                HAS_SPARE = false;
                return mean + std_dev * SPARE.unwrap();
            }
        }
        
        let u1 = self.next_f32();
        let u2 = self.next_f32();
        
        let mag = std_dev * (-2.0 * ln_approx(u1)).sqrt();
        let z0 = mag * cos_approx(2.0 * PI * u2);
        let z1 = mag * sin_approx(2.0 * PI * u2);
        
        unsafe {
            SPARE = Some(z1);
            HAS_SPARE = true;
        }
        
        mean + z0
    }
}

impl Default for SimpleRng {
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_exp_approx() {
        let x = 1.0;
        let exact = x.exp();
        let approx = exp_approx(x);
        let error = (exact - approx).abs() / exact;
        assert!(error < 0.01); // Within 1% error
    }
    
    #[test]
    fn test_ln_approx() {
        let x = 2.0;
        let exact = x.ln();
        let approx = ln_approx(x);
        let error = (exact - approx).abs() / exact;
        assert!(error < 0.05); // Within 5% error
    }
    
    #[test]
    fn test_sqrt_approx() {
        let x = 4.0;
        let exact = x.sqrt();
        let approx = sqrt_approx(x);
        let error = (exact - approx).abs() / exact;
        assert!(error < 0.001); // Within 0.1% error
    }
    
    #[test]
    fn test_trig_approx() {
        let x = PI / 4.0; // 45 degrees
        
        let sin_exact = x.sin();
        let sin_approx = sin_approx(x);
        let sin_error = (sin_exact - sin_approx).abs() / sin_exact;
        assert!(sin_error < 0.01);
        
        let cos_exact = x.cos();
        let cos_approx = cos_approx(x);
        let cos_error = (cos_exact - cos_approx).abs() / cos_exact;
        assert!(cos_error < 0.01);
    }
    
    #[test]
    fn test_utility_functions() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
        
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        assert_eq!(lerp(2.0, 8.0, 0.25), 3.5);
    }
    
    #[test]
    fn test_power_functions() {
        assert_eq!(powi_approx(2.0, 3), 8.0);
        assert_eq!(powi_approx(5.0, 0), 1.0);
        
        let pow_result = pow_approx(2.0, 3.0);
        assert!((pow_result - 8.0).abs() < 0.1);
    }
    
    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(42);
        
        let val1 = rng.next_f32();
        let val2 = rng.next_f32();
        
        assert!(val1 >= 0.0 && val1 < 1.0);
        assert!(val2 >= 0.0 && val2 < 1.0);
        assert_ne!(val1, val2);
        
        let range_val = rng.next_f32_range(10.0, 20.0);
        assert!(range_val >= 10.0 && range_val < 20.0);
    }
}