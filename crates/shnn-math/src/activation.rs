//! Neural activation functions for neuromorphic computing
//!
//! This module provides biologically-inspired activation functions
//! commonly used in spiking neural networks and neuromorphic systems.

use crate::Float;
use crate::math::MathExt;
use core::f32::consts::PI;

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Activation function trait for neural computations
pub trait ActivationFunction {
    /// Apply activation function to input
    fn activate(&self, x: Float) -> Float;
    
    /// Compute derivative of activation function
    fn derivative(&self, x: Float) -> Float;
    
    /// Apply activation to a slice of values in-place
    fn activate_slice(&self, values: &mut [Float]) {
        for x in values {
            *x = self.activate(*x);
        }
    }
    
    /// Apply activation to a vector, returning new vector
    fn activate_vec(&self, values: &[Float]) -> Vec<Float> {
        values.iter().map(|&x| self.activate(x)).collect()
    }
}

/// Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Sigmoid {
    /// Steepness parameter (default: 1.0)
    pub steepness: Float,
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self { steepness: 1.0 }
    }
}

impl Sigmoid {
    /// Create new sigmoid with custom steepness
    pub fn with_steepness(steepness: Float) -> Self {
        Self { steepness }
    }
}

impl ActivationFunction for Sigmoid {
    fn activate(&self, x: Float) -> Float {
        1.0 / (1.0 + (-self.steepness * x).exp())
    }
    
    fn derivative(&self, x: Float) -> Float {
        let sig = self.activate(x);
        self.steepness * sig * (1.0 - sig)
    }
}

/// Hyperbolic tangent activation: f(x) = tanh(x)
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Tanh {
    /// Steepness parameter (default: 1.0)
    pub steepness: Float,
}

impl Default for Tanh {
    fn default() -> Self {
        Self { steepness: 1.0 }
    }
}

impl Tanh {
    /// Create new tanh with custom steepness
    pub fn with_steepness(steepness: Float) -> Self {
        Self { steepness }
    }
}

impl ActivationFunction for Tanh {
    fn activate(&self, x: Float) -> Float {
        (self.steepness * x).tanh()
    }
    
    fn derivative(&self, x: Float) -> Float {
        let tanh_x = self.activate(x);
        self.steepness * (1.0 - tanh_x * tanh_x)
    }
}

/// Rectified Linear Unit: f(x) = max(0, x)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ReLU;

impl ActivationFunction for ReLU {
    fn activate(&self, x: Float) -> Float {
        x.max(0.0)
    }
    
    fn derivative(&self, x: Float) -> Float {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}

/// Leaky ReLU: f(x) = max(αx, x)
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LeakyReLU {
    /// Leak parameter (default: 0.01)
    pub alpha: Float,
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self { alpha: 0.01 }
    }
}

impl LeakyReLU {
    /// Create new Leaky ReLU with custom alpha
    pub fn with_alpha(alpha: Float) -> Self {
        Self { alpha }
    }
}

impl ActivationFunction for LeakyReLU {
    fn activate(&self, x: Float) -> Float {
        if x >= 0.0 { x } else { self.alpha * x }
    }
    
    fn derivative(&self, x: Float) -> Float {
        if x >= 0.0 { 1.0 } else { self.alpha }
    }
}

/// Exponential Linear Unit: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ELU {
    /// Alpha parameter (default: 1.0)
    pub alpha: Float,
}

impl Default for ELU {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl ELU {
    /// Create new ELU with custom alpha
    pub fn with_alpha(alpha: Float) -> Self {
        Self { alpha }
    }
}

impl ActivationFunction for ELU {
    fn activate(&self, x: Float) -> Float {
        if x >= 0.0 { x } else { self.alpha * (x.exp() - 1.0) }
    }
    
    fn derivative(&self, x: Float) -> Float {
        if x >= 0.0 { 1.0 } else { self.alpha * x.exp() }
    }
}

/// Swish activation: f(x) = x * sigmoid(x)
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Swish {
    /// Beta parameter (default: 1.0)
    pub beta: Float,
}

impl Default for Swish {
    fn default() -> Self {
        Self { beta: 1.0 }
    }
}

impl Swish {
    /// Create new Swish with custom beta
    pub fn with_beta(beta: Float) -> Self {
        Self { beta }
    }
}

impl ActivationFunction for Swish {
    fn activate(&self, x: Float) -> Float {
        let sigmoid = 1.0 / (1.0 + (-self.beta * x).exp());
        x * sigmoid
    }
    
    fn derivative(&self, x: Float) -> Float {
        let exp_neg_bx = (-self.beta * x).exp();
        let sigmoid = 1.0 / (1.0 + exp_neg_bx);
        sigmoid + x * self.beta * sigmoid * (1.0 - sigmoid)
    }
}

/// GELU (Gaussian Error Linear Unit): f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GELU;

impl ActivationFunction for GELU {
    fn activate(&self, x: Float) -> Float {
        let sqrt_2_over_pi = (2.0 / PI).sqrt();
        let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
        0.5 * x * (1.0 + inner.tanh())
    }
    
    fn derivative(&self, x: Float) -> Float {
        let sqrt_2_over_pi = (2.0 / PI).sqrt();
        let x3 = x.powi(3);
        let inner = sqrt_2_over_pi * (x + 0.044715 * x3);
        let tanh_inner = inner.tanh();
        let sech2_inner = 1.0 - tanh_inner * tanh_inner;
        
        0.5 * (1.0 + tanh_inner) + 
        0.5 * x * sech2_inner * sqrt_2_over_pi * (1.0 + 0.134145 * x.powi(2))
    }
}

/// Softplus activation: f(x) = ln(1 + e^x)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Softplus;

impl ActivationFunction for Softplus {
    fn activate(&self, x: Float) -> Float {
        // Use log1p for numerical stability when x is small
        if x > 20.0 {
            x // Avoid overflow
        } else if x < -20.0 {
            0.0 // Underflow protection
        } else {
            (1.0 + x.exp()).ln()
        }
    }
    
    fn derivative(&self, x: Float) -> Float {
        // Derivative is sigmoid
        1.0 / (1.0 + (-x).exp())
    }
}

/// Mish activation: f(x) = x * tanh(softplus(x))
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Mish;

impl ActivationFunction for Mish {
    fn activate(&self, x: Float) -> Float {
        let softplus = Softplus.activate(x);
        x * softplus.tanh()
    }
    
    fn derivative(&self, x: Float) -> Float {
        let softplus = Softplus.activate(x);
        let tanh_sp = softplus.tanh();
        let sech2_sp = 1.0 - tanh_sp * tanh_sp;
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        
        tanh_sp + x * sech2_sp * sigmoid
    }
}

/// Linear activation: f(x) = x (identity function)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Linear;

impl ActivationFunction for Linear {
    fn activate(&self, x: Float) -> Float {
        x
    }
    
    fn derivative(&self, _x: Float) -> Float {
        1.0
    }
}

/// Step function: f(x) = 1 if x > threshold, 0 otherwise
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Step {
    /// Threshold value (default: 0.0)
    pub threshold: Float,
}

impl Default for Step {
    fn default() -> Self {
        Self { threshold: 0.0 }
    }
}

impl Step {
    /// Create new step function with custom threshold
    pub fn with_threshold(threshold: Float) -> Self {
        Self { threshold }
    }
}

impl ActivationFunction for Step {
    fn activate(&self, x: Float) -> Float {
        if x > self.threshold { 1.0 } else { 0.0 }
    }
    
    fn derivative(&self, x: Float) -> Float {
        // Derivative is 0 everywhere except at threshold (where it's undefined)
        // We return 0 for practical purposes
        0.0
    }
}

/// Convenience functions for common activations
pub fn sigmoid(x: Float) -> Float {
    Sigmoid::default().activate(x)
}

pub fn tanh(x: Float) -> Float {
    Tanh::default().activate(x)
}

pub fn relu(x: Float) -> Float {
    ReLU.activate(x)
}

pub fn leaky_relu(x: Float) -> Float {
    LeakyReLU::default().activate(x)
}

pub fn leaky_relu_with_alpha(x: Float, alpha: Float) -> Float {
    LeakyReLU::with_alpha(alpha).activate(x)
}

pub fn elu(x: Float) -> Float {
    ELU::default().activate(x)
}

pub fn elu_with_alpha(x: Float, alpha: Float) -> Float {
    ELU::with_alpha(alpha).activate(x)
}

pub fn swish(x: Float) -> Float {
    Swish::default().activate(x)
}

pub fn gelu(x: Float) -> Float {
    GELU.activate(x)
}

pub fn softplus(x: Float) -> Float {
    Softplus.activate(x)
}

pub fn mish(x: Float) -> Float {
    Mish.activate(x)
}

/// Apply softmax to a slice of values
pub fn softmax(values: &mut [Float]) {
    if values.is_empty() {
        return;
    }
    
    // Find maximum for numerical stability
    let max_val = values.iter().copied().fold(Float::NEG_INFINITY, Float::max);
    
    // Compute exponentials
    for x in values.iter_mut() {
        *x = (*x - max_val).exp();
    }
    
    // Normalize
    let sum: Float = values.iter().sum();
    if sum > 0.0 {
        for x in values.iter_mut() {
            *x /= sum;
        }
    }
}

/// Apply log-softmax to a slice of values (numerically stable)
pub fn log_softmax(values: &mut [Float]) {
    if values.is_empty() {
        return;
    }
    
    let max_val = values.iter().copied().fold(Float::NEG_INFINITY, Float::max);
    
    // Compute log-sum-exp
    let log_sum_exp = {
        let sum_exp: Float = values.iter().map(|&x| (x - max_val).exp()).sum();
        max_val + sum_exp.ln()
    };
    
    // Compute log-softmax
    for x in values.iter_mut() {
        *x -= log_sum_exp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid::default();
        assert!((sigmoid.activate(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid.activate(10.0) > 0.99);
        assert!(sigmoid.activate(-10.0) < 0.01);
        
        // Test derivative
        let x = 0.0;
        let expected_derivative = 0.25; // sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5
        assert!((sigmoid.derivative(x) - expected_derivative).abs() < 1e-6);
    }
    
    #[test]
    fn test_tanh() {
        let tanh_fn = Tanh::default();
        assert!((tanh_fn.activate(0.0) - 0.0).abs() < 1e-6);
        assert!(tanh_fn.activate(10.0) > 0.99);
        assert!(tanh_fn.activate(-10.0) < -0.99);
        
        // Test derivative
        let x = 0.0;
        let expected_derivative = 1.0; // tanh'(0) = 1 - tanh²(0) = 1 - 0 = 1
        assert!((tanh_fn.derivative(x) - expected_derivative).abs() < 1e-6);
    }
    
    #[test]
    fn test_relu() {
        let relu = ReLU;
        assert_eq!(relu.activate(-1.0), 0.0);
        assert_eq!(relu.activate(0.0), 0.0);
        assert_eq!(relu.activate(1.0), 1.0);
        assert_eq!(relu.activate(5.0), 5.0);
        
        assert_eq!(relu.derivative(-1.0), 0.0);
        assert_eq!(relu.derivative(1.0), 1.0);
    }
    
    #[test]
    fn test_leaky_relu() {
        let leaky_relu = LeakyReLU::with_alpha(0.1);
        assert_eq!(leaky_relu.activate(-1.0), -0.1);
        assert_eq!(leaky_relu.activate(0.0), 0.0);
        assert_eq!(leaky_relu.activate(1.0), 1.0);
        
        assert_eq!(leaky_relu.derivative(-1.0), 0.1);
        assert_eq!(leaky_relu.derivative(1.0), 1.0);
    }
    
    #[test]
    fn test_elu() {
        let elu = ELU::default();
        assert_eq!(elu.activate(1.0), 1.0);
        assert!(elu.activate(-1.0) < 0.0 && elu.activate(-1.0) > -1.0);
        
        assert_eq!(elu.derivative(1.0), 1.0);
        assert!(elu.derivative(-1.0) > 0.0);
    }
    
    #[test]
    fn test_softmax() {
        let mut values = [1.0, 2.0, 3.0];
        softmax(&mut values);
        
        // Check that probabilities sum to 1
        let sum: Float = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that all values are positive
        assert!(values.iter().all(|&x| x > 0.0));
        
        // Check that largest input gives largest output
        assert!(values[2] > values[1] && values[1] > values[0]);
    }
    
    #[test]
    fn test_convenience_functions() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(1.0), 1.0);
        assert!((tanh(0.0) - 0.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_activation_trait() {
        let activations: Vec<Box<dyn ActivationFunction>> = vec![
            Box::new(Sigmoid::default()),
            Box::new(Tanh::default()),
            Box::new(ReLU),
            Box::new(LeakyReLU::default()),
        ];
        
        for activation in activations {
            let result = activation.activate(0.5);
            let derivative = activation.derivative(0.5);
            
            // Basic sanity checks
            assert!(result.is_finite());
            assert!(derivative.is_finite());
        }
    }
}