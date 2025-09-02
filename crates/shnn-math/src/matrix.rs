//! Matrix operations for neuromorphic computing
//!
//! This module provides efficient matrix operations optimized for
//! neural network computations with zero external dependencies.

use crate::{Float, Result, MathError};
use core::ops::{Add, Sub, Mul, Index, IndexMut};
use crate::math::MathExt;

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, vec};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Dense matrix implementation for neural computations
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Matrix {
    data: Vec<Float>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// Create a new matrix with given dimensions, initialized to zero
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }
    
    /// Create a new matrix with given dimensions, initialized to ones
    pub fn ones(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![1.0; rows * cols],
            rows,
            cols,
        }
    }
    
    /// Create an identity matrix
    pub fn identity(size: usize) -> Self {
        let mut matrix = Self::zeros(size, size);
        for i in 0..size {
            matrix[(i, i)] = 1.0;
        }
        matrix
    }
    
    /// Create matrix from data vector (row-major order)
    pub fn from_vec(data: Vec<Float>, rows: usize, cols: usize) -> Result<Self> {
        if data.len() != rows * cols {
            return Err(MathError::DimensionMismatch {
                expected: rows * cols,
                got: data.len(),
            });
        }
        Ok(Self { data, rows, cols })
    }
    
    /// Create matrix from nested vector
    pub fn from_nested_vec(data: Vec<Vec<Float>>) -> Result<Self> {
        if data.is_empty() {
            return Ok(Self::zeros(0, 0));
        }
        
        let rows = data.len();
        let cols = data[0].len();
        
        // Check all rows have same length
        for row in &data {
            if row.len() != cols {
                return Err(MathError::DimensionMismatch {
                    expected: cols,
                    got: row.len(),
                });
            }
        }
        
        let flat_data: Vec<Float> = data.into_iter().flatten().collect();
        Self::from_vec(flat_data, rows, cols)
    }
    
    /// Get matrix dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    /// Get number of rows
    pub fn rows(&self) -> usize {
        self.rows
    }
    
    /// Get number of columns
    pub fn cols(&self) -> usize {
        self.cols
    }
    
    /// Get matrix dimensions as (rows, cols)
    pub fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    /// Get the underlying data as a slice
    pub fn as_slice(&self) -> &[Float] {
        &self.data
    }
    
    /// Get reference to internal data
    pub fn data(&self) -> &[Float] {
        &self.data
    }
    
    /// Get mutable reference to internal data
    pub fn data_mut(&mut self) -> &mut [Float] {
        &mut self.data
    }
    
    /// Get element at position (row, col)
    pub fn get(&self, row: usize, col: usize) -> Result<Float> {
        if row >= self.rows || col >= self.cols {
            return Err(MathError::IndexOutOfBounds {
                index: row * self.cols + col,
                len: self.data.len(),
            });
        }
        Ok(self.data[row * self.cols + col])
    }
    
    /// Set element at position (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: Float) -> Result<()> {
        if row >= self.rows || col >= self.cols {
            return Err(MathError::IndexOutOfBounds {
                index: row * self.cols + col,
                len: self.data.len(),
            });
        }
        self.data[row * self.cols + col] = value;
        Ok(())
    }
    
    /// Transpose the matrix
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(j, i)] = self[(i, j)];
            }
        }
        result
    }
    
    /// Matrix multiplication
    pub fn multiply(&self, other: &Matrix) -> Result<Matrix> {
        if self.cols != other.rows {
            return Err(MathError::DimensionMismatch {
                expected: self.cols,
                got: other.rows,
            });
        }
        
        let mut result = Matrix::zeros(self.rows, other.cols);
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self[(i, k)] * other[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        
        Ok(result)
    }
    
    /// Element-wise multiplication (Hadamard product)
    pub fn hadamard(&self, other: &Matrix) -> Result<Matrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MathError::DimensionMismatch {
                expected: self.data.len(),
                got: other.data.len(),
            });
        }
        
        let mut result = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] * other.data[i];
        }
        
        Ok(result)
    }
    
    /// Matrix-vector multiplication
    pub fn multiply_vector(&self, vec: &[Float]) -> Result<Vec<Float>> {
        if self.cols != vec.len() {
            return Err(MathError::DimensionMismatch {
                expected: self.cols,
                got: vec.len(),
            });
        }
        
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self[(i, j)] * vec[j];
            }
        }
        
        Ok(result)
    }
    
    /// Calculate determinant (for 2x2 matrices)
    pub fn determinant_2x2(&self) -> Result<Float> {
        if self.rows != 2 || self.cols != 2 {
            return Err(MathError::DimensionMismatch {
                expected: 4,
                got: self.data.len(),
            });
        }
        
        Ok(self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)])
    }
    
    /// Inverse for 2x2 matrix
    pub fn inverse_2x2(&self) -> Result<Matrix> {
        let det = self.determinant_2x2()?;
        if det.abs() < crate::constants::EPSILON {
            return Err(MathError::SingularMatrix);
        }
        
        let mut result = Matrix::zeros(2, 2);
        result[(0, 0)] = self[(1, 1)] / det;
        result[(0, 1)] = -self[(0, 1)] / det;
        result[(1, 0)] = -self[(1, 0)] / det;
        result[(1, 1)] = self[(0, 0)] / det;
        
        Ok(result)
    }
    
    /// Apply function to each element
    pub fn map<F>(&self, f: F) -> Matrix
    where
        F: Fn(Float) -> Float,
    {
        let new_data: Vec<Float> = self.data.iter().map(|&x| f(x)).collect();
        Matrix {
            data: new_data,
            rows: self.rows,
            cols: self.cols,
        }
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
    
    /// Fill matrix with given value
    pub fn fill(&mut self, value: Float) {
        for x in &mut self.data {
            *x = value;
        }
    }
    
    /// Get row as vector
    pub fn row(&self, row: usize) -> Result<Vec<Float>> {
        if row >= self.rows {
            return Err(MathError::IndexOutOfBounds {
                index: row,
                len: self.rows,
            });
        }
        
        let start = row * self.cols;
        let end = start + self.cols;
        Ok(self.data[start..end].to_vec())
    }
    
    /// Get column as vector
    pub fn col(&self, col: usize) -> Result<Vec<Float>> {
        if col >= self.cols {
            return Err(MathError::IndexOutOfBounds {
                index: col,
                len: self.cols,
            });
        }
        
        let mut result = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            result.push(self[(row, col)]);
        }
        Ok(result)
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = Float;
    
    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row * self.cols + col]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row * self.cols + col]
    }
}

impl Add for Matrix {
    type Output = Result<Matrix>;
    
    fn add(self, other: Matrix) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MathError::DimensionMismatch {
                expected: self.data.len(),
                got: other.data.len(),
            });
        }
        
        let mut result = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + other.data[i];
        }
        
        Ok(result)
    }
}

impl Sub for Matrix {
    type Output = Result<Matrix>;
    
    fn sub(self, other: Matrix) -> Self::Output {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(MathError::DimensionMismatch {
                expected: self.data.len(),
                got: other.data.len(),
            });
        }
        
        let mut result = Matrix::zeros(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] - other.data[i];
        }
        
        Ok(result)
    }
}

impl Mul<Float> for Matrix {
    type Output = Matrix;
    
    fn mul(self, scalar: Float) -> Self::Output {
        self.map(|x| x * scalar)
    }
}

/// Matrix operations trait for generic programming
pub trait MatrixOps {
    /// Matrix multiplication
    fn matrix_multiply(&self, other: &Self) -> Result<Self>
    where
        Self: Sized;
    
    /// Transpose operation
    fn transpose(&self) -> Self;
    
    /// Get matrix dimensions
    fn dimensions(&self) -> (usize, usize);
}

impl MatrixOps for Matrix {
    fn matrix_multiply(&self, other: &Self) -> Result<Self> {
        self.multiply(other)
    }
    
    fn transpose(&self) -> Self {
        self.transpose()
    }
    
    fn dimensions(&self) -> (usize, usize) {
        self.shape()
    }
}

/// Convenience functions
pub fn matrix_multiply(a: &Matrix, b: &Matrix) -> Result<Matrix> {
    a.multiply(b)
}

pub fn transpose(matrix: &Matrix) -> Matrix {
    matrix.transpose()
}

pub fn inverse_2x2(matrix: &Matrix) -> Result<Matrix> {
    matrix.inverse_2x2()
}

pub fn determinant(matrix: &Matrix) -> Result<Float> {
    matrix.determinant_2x2()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matrix_creation() {
        let m = Matrix::zeros(3, 2);
        assert_eq!(m.shape(), (3, 2));
        assert_eq!(m.data().len(), 6);
        
        let m = Matrix::ones(2, 2);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 1.0);
    }
    
    #[test]
    fn test_matrix_multiplication() {
        let a = Matrix::from_nested_vec(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]).unwrap();
        
        let b = Matrix::from_nested_vec(vec![
            vec![2.0, 0.0],
            vec![1.0, 2.0],
        ]).unwrap();
        
        let c = a.multiply(&b).unwrap();
        assert_eq!(c[(0, 0)], 4.0);
        assert_eq!(c[(0, 1)], 4.0);
        assert_eq!(c[(1, 0)], 10.0);
        assert_eq!(c[(1, 1)], 8.0);
    }
    
    #[test]
    fn test_transpose() {
        let m = Matrix::from_nested_vec(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]).unwrap();
        
        let t = m.transpose();
        assert_eq!(t.shape(), (3, 2));
        assert_eq!(t[(0, 0)], 1.0);
        assert_eq!(t[(2, 1)], 6.0);
    }
    
    #[test]
    fn test_determinant_2x2() {
        let m = Matrix::from_nested_vec(vec![
            vec![3.0, 8.0],
            vec![4.0, 6.0],
        ]).unwrap();
        
        let det = m.determinant_2x2().unwrap();
        assert_eq!(det, -14.0);
    }
    
    #[test]
    fn test_inverse_2x2() {
        let m = Matrix::from_nested_vec(vec![
            vec![4.0, 7.0],
            vec![2.0, 6.0],
        ]).unwrap();
        
        let inv = m.inverse_2x2().unwrap();
        let product = m.multiply(&inv).unwrap();
        
        // Should be close to identity matrix
        assert!((product[(0, 0)] - 1.0).abs() < 1e-6);
        assert!((product[(1, 1)] - 1.0).abs() < 1e-6);
        assert!(product[(0, 1)].abs() < 1e-6);
        assert!(product[(1, 0)].abs() < 1e-6);
    }
}