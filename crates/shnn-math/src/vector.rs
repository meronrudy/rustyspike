//! Vector operations for neuromorphic computing
//!
//! This module provides efficient vector operations optimized for
//! neural network computations with zero external dependencies.

use crate::{Float, Result, MathError};
use crate::math::MathExt;
use core::ops::{Add, Sub, Mul, Index, IndexMut};

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, vec};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Dense vector implementation for neural computations
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Vector {
    data: Vec<Float>,
}

impl Vector {
    /// Create a new vector with given length, initialized to zero
    pub fn zeros(length: usize) -> Self {
        Self {
            data: vec![0.0; length],
        }
    }
    
    /// Create a new vector with given length, initialized to ones
    pub fn ones(length: usize) -> Self {
        Self {
            data: vec![1.0; length],
        }
    }
    
    /// Create vector from data
    pub fn from_vec(data: Vec<Float>) -> Self {
        Self { data }
    }
    
    /// Create vector from slice
    pub fn from_slice(data: &[Float]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }
    
    /// Get vector length
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if vector is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Get reference to internal data
    pub fn data(&self) -> &[Float] {
        &self.data
    }
    
    /// Get mutable reference to internal data
    pub fn data_mut(&mut self) -> &mut [Float] {
        &mut self.data
    }
    
    /// Get element at index
    pub fn get(&self, index: usize) -> Result<Float> {
        if index >= self.data.len() {
            return Err(MathError::IndexOutOfBounds {
                index,
                len: self.data.len(),
            });
        }
        Ok(self.data[index])
    }
    
    /// Set element at index
    pub fn set(&mut self, index: usize, value: Float) -> Result<()> {
        if index >= self.data.len() {
            return Err(MathError::IndexOutOfBounds {
                index,
                len: self.data.len(),
            });
        }
        self.data[index] = value;
        Ok(())
    }
    
    /// Dot product with another vector
    pub fn dot(&self, other: &Vector) -> Result<Float> {
        if self.len() != other.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        
        Ok(self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a * b).sum())
    }
    
    /// L2 norm (Euclidean norm)
    pub fn norm(&self) -> Float {
        self.data.iter().map(|&x| x * x).sum::<Float>().sqrt()
    }
    
    /// L1 norm (Manhattan norm)
    pub fn norm_l1(&self) -> Float {
        self.data.iter().map(|&x| x.abs()).sum()
    }
    
    /// Normalize to unit length (L2 norm = 1)
    pub fn normalize(&mut self) -> Result<()> {
        let norm = self.norm();
        if norm < crate::constants::EPSILON {
            return Err(MathError::DivisionByZero);
        }
        
        for x in &mut self.data {
            *x /= norm;
        }
        
        Ok(())
    }
    
    /// Get normalized copy
    pub fn normalized(&self) -> Result<Vector> {
        let mut result = self.clone();
        result.normalize()?;
        Ok(result)
    }
    
    /// Element-wise addition
    pub fn add(&self, other: &Vector) -> Result<Vector> {
        if self.len() != other.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        
        let result_data: Vec<Float> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a + b)
            .collect();
        
        Ok(Vector::from_vec(result_data))
    }
    
    /// Element-wise subtraction
    pub fn sub(&self, other: &Vector) -> Result<Vector> {
        if self.len() != other.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        
        let result_data: Vec<Float> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        
        Ok(Vector::from_vec(result_data))
    }
    
    /// Element-wise multiplication (Hadamard product)
    pub fn hadamard(&self, other: &Vector) -> Result<Vector> {
        if self.len() != other.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        
        let result_data: Vec<Float> = self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| a * b)
            .collect();
        
        Ok(Vector::from_vec(result_data))
    }
    
    /// Scalar multiplication
    pub fn scale(&mut self, scalar: Float) {
        for x in &mut self.data {
            *x *= scalar;
        }
    }
    
    /// Get scaled copy
    pub fn scaled(&self, scalar: Float) -> Vector {
        let scaled_data: Vec<Float> = self.data.iter().map(|&x| x * scalar).collect();
        Vector::from_vec(scaled_data)
    }
    
    /// Add scalar to all elements
    pub fn add_scalar(&mut self, scalar: Float) {
        for x in &mut self.data {
            *x += scalar;
        }
    }
    
    /// Apply function to each element
    pub fn map<F>(&self, f: F) -> Vector
    where
        F: Fn(Float) -> Float,
    {
        let new_data: Vec<Float> = self.data.iter().map(|&x| f(x)).collect();
        Vector::from_vec(new_data)
    }
    
    /// Apply function to each element in-place
    pub fn map_inplace<F>(&mut self, f: F)
    where
        F: Fn(Float) -> Float,
    {
        for x in &mut self.data {
            *x = f(*x);
        }
    }
    
    /// Fill vector with given value
    pub fn fill(&mut self, value: Float) {
        for x in &mut self.data {
            *x = value;
        }
    }
    
    /// Sum all elements
    pub fn sum(&self) -> Float {
        self.data.iter().sum()
    }
    
    /// Get mean value
    pub fn mean(&self) -> Float {
        if self.is_empty() {
            return 0.0;
        }
        self.sum() / self.len() as Float
    }
    
    /// Get variance
    pub fn variance(&self) -> Float {
        if self.len() < 2 {
            return 0.0;
        }
        
        let mean = self.mean();
        let sum_sq_diff: Float = self.data.iter()
            .map(|&x| (x - mean) * (x - mean))
            .sum();
        
        sum_sq_diff / (self.len() - 1) as Float
    }
    
    /// Get standard deviation
    pub fn std_dev(&self) -> Float {
        self.variance().sqrt()
    }
    
    /// Get minimum value
    pub fn min(&self) -> Option<Float> {
        self.data.iter().copied().reduce(Float::min)
    }
    
    /// Get maximum value
    pub fn max(&self) -> Option<Float> {
        self.data.iter().copied().reduce(Float::max)
    }
    
    /// Get index of minimum value
    pub fn argmin(&self) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        
        let mut min_idx = 0;
        let mut min_val = self.data[0];
        
        for (i, &val) in self.data.iter().enumerate().skip(1) {
            if val < min_val {
                min_val = val;
                min_idx = i;
            }
        }
        
        Some(min_idx)
    }
    
    /// Get index of maximum value
    pub fn argmax(&self) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        
        let mut max_idx = 0;
        let mut max_val = self.data[0];
        
        for (i, &val) in self.data.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        
        Some(max_idx)
    }
    
    /// Cosine similarity with another vector
    pub fn cosine_similarity(&self, other: &Vector) -> Result<Float> {
        let dot = self.dot(other)?;
        let norm_self = self.norm();
        let norm_other = other.norm();
        
        if norm_self < crate::constants::EPSILON || norm_other < crate::constants::EPSILON {
            return Err(MathError::DivisionByZero);
        }
        
        Ok(dot / (norm_self * norm_other))
    }
    
    /// Euclidean distance to another vector
    pub fn distance(&self, other: &Vector) -> Result<Float> {
        let diff = self.sub(other)?;
        Ok(diff.norm())
    }
    
    /// Manhattan distance to another vector
    pub fn distance_l1(&self, other: &Vector) -> Result<Float> {
        let diff = self.sub(other)?;
        Ok(diff.norm_l1())
    }
    
    /// Resize vector to new length, filling with zeros if expanded
    pub fn resize(&mut self, new_len: usize) {
        self.data.resize(new_len, 0.0);
    }
    
    /// Get subvector from start to end (exclusive)
    pub fn slice(&self, start: usize, end: usize) -> Result<Vector> {
        if start >= self.len() || end > self.len() || start >= end {
            return Err(MathError::IndexOutOfBounds {
                index: start.max(end),
                len: self.len(),
            });
        }
        
        Ok(Vector::from_slice(&self.data[start..end]))
    }
    
    /// Concatenate with another vector
    pub fn concat(&self, other: &Vector) -> Vector {
        let mut result = self.clone();
        result.data.extend_from_slice(&other.data);
        result
    }
}

impl Index<usize> for Vector {
    type Output = Float;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Add for Vector {
    type Output = Vector;
    
    fn add(mut self, other: Vector) -> Self::Output {
        for i in 0..self.len().min(other.len()) {
            self.data[i] += other.data[i];
        }
        self
    }
}

impl Sub for Vector {
    type Output = Vector;
    
    fn sub(mut self, other: Vector) -> Self::Output {
        for i in 0..self.len().min(other.len()) {
            self.data[i] -= other.data[i];
        }
        self
    }
}

impl Mul<Float> for Vector {
    type Output = Vector;
    
    fn mul(self, scalar: Float) -> Self::Output {
        self.scaled(scalar)
    }
}

/// Vector operations trait for generic programming
pub trait VectorOps {
    /// Dot product
    fn dot_product(&self, other: &Self) -> Result<Float>;
    
    /// L2 norm
    fn l2_norm(&self) -> Float;
    
    /// Normalize to unit length
    fn normalize_inplace(&mut self) -> Result<()>;
}

impl VectorOps for Vector {
    fn dot_product(&self, other: &Self) -> Result<Float> {
        self.dot(other)
    }
    
    fn l2_norm(&self) -> Float {
        self.norm()
    }
    
    fn normalize_inplace(&mut self) -> Result<()> {
        self.normalize()
    }
}

// Implement VectorOps for slices for convenience
impl VectorOps for [Float] {
    fn dot_product(&self, other: &Self) -> Result<Float> {
        if self.len() != other.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        
        Ok(self.iter().zip(other.iter()).map(|(&a, &b)| a * b).sum())
    }
    
    fn l2_norm(&self) -> Float {
        self.iter().map(|&x| x * x).sum::<Float>().sqrt()
    }
    
    fn normalize_inplace(&mut self) -> Result<()> {
        let norm = self.l2_norm();
        if norm < crate::constants::EPSILON {
            return Err(MathError::DivisionByZero);
        }
        
        for x in self {
            *x /= norm;
        }
        
        Ok(())
    }
}

/// Convenience functions
pub fn dot_product(a: &[Float], b: &[Float]) -> Result<Float> {
    a.dot_product(b)
}

pub fn cosine_similarity(a: &[Float], b: &[Float]) -> Result<Float> {
    let dot = dot_product(a, b)?;
    let norm_a = a.l2_norm();
    let norm_b = b.l2_norm();
    
    if norm_a < crate::constants::EPSILON || norm_b < crate::constants::EPSILON {
        return Err(MathError::DivisionByZero);
    }
    
    Ok(dot / (norm_a * norm_b))
}

pub fn euclidean_distance(a: &[Float], b: &[Float]) -> Result<Float> {
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    
    let sum_sq_diff: Float = a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum();
    
    Ok(sum_sq_diff.sqrt())
}

pub fn manhattan_distance(a: &[Float], b: &[Float]) -> Result<Float> {
    if a.len() != b.len() {
        return Err(MathError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    
    let sum_abs_diff: Float = a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum();
    
    Ok(sum_abs_diff)
}

/// Element-wise vector operations
pub fn vector_add(a: &[Float], b: &[Float], result: &mut [Float]) -> Result<()> {
    if a.len() != b.len() || a.len() != result.len() {
        return Err(MathError::DimensionMismatch {
            expected: a.len(),
            got: b.len().min(result.len()),
        });
    }
    
    for ((r, &a_val), &b_val) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
        *r = a_val + b_val;
    }
    
    Ok(())
}

pub fn vector_sub(a: &[Float], b: &[Float], result: &mut [Float]) -> Result<()> {
    if a.len() != b.len() || a.len() != result.len() {
        return Err(MathError::DimensionMismatch {
            expected: a.len(),
            got: b.len().min(result.len()),
        });
    }
    
    for ((r, &a_val), &b_val) in result.iter_mut().zip(a.iter()).zip(b.iter()) {
        *r = a_val - b_val;
    }
    
    Ok(())
}

pub fn vector_scale(vector: &mut [Float], scalar: Float) {
    for x in vector {
        *x *= scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_creation() {
        let v = Vector::zeros(5);
        assert_eq!(v.len(), 5);
        assert_eq!(v[0], 0.0);
        
        let v = Vector::ones(3);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1.0);
        
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert_eq!(v[1], 2.0);
    }
    
    #[test]
    fn test_vector_operations() {
        let a = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Vector::from_vec(vec![4.0, 5.0, 6.0]);
        
        let dot = a.dot(&b).unwrap();
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6
        
        let sum = a.add(&b).unwrap();
        assert_eq!(sum.data(), &[5.0, 7.0, 9.0]);
        
        let diff = b.sub(&a).unwrap();
        assert_eq!(diff.data(), &[3.0, 3.0, 3.0]);
    }
    
    #[test]
    fn test_vector_norms() {
        let v = Vector::from_vec(vec![3.0, 4.0]);
        assert_eq!(v.norm(), 5.0);
        assert_eq!(v.norm_l1(), 7.0);
        
        let mut v = Vector::from_vec(vec![6.0, 8.0]);
        v.normalize().unwrap();
        assert!((v.norm() - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_vector_statistics() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(v.mean(), 3.0);
        assert_eq!(v.sum(), 15.0);
        assert_eq!(v.min(), Some(1.0));
        assert_eq!(v.max(), Some(5.0));
        assert_eq!(v.argmin(), Some(0));
        assert_eq!(v.argmax(), Some(4));
    }
    
    #[test]
    fn test_cosine_similarity() {
        let a = Vector::from_vec(vec![1.0, 0.0]);
        let b = Vector::from_vec(vec![0.0, 1.0]);
        let c = Vector::from_vec(vec![1.0, 0.0]);
        
        assert_eq!(a.cosine_similarity(&b).unwrap(), 0.0);
        assert_eq!(a.cosine_similarity(&c).unwrap(), 1.0);
    }
    
    #[test]
    fn test_slice_operations() {
        let data = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(data.l2_norm(), (1.0 + 4.0 + 9.0 + 16.0_f32).sqrt());
        
        let dot = dot_product(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert_eq!(dot, 11.0);
        
        let dist = euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]).unwrap();
        assert_eq!(dist, 5.0);
    }
}