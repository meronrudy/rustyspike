//! Sparse matrix operations for neuromorphic computing
//!
//! This module provides memory-efficient sparse matrix implementations
//! optimized for neural networks with low connectivity density.

use crate::{Float, Result, MathError};
use core::ops::{Add, Sub, Mul};

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, vec};

#[cfg(feature = "std")]
use std::collections::HashMap;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Compressed Sparse Row (CSR) matrix format
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseMatrix {
    /// Non-zero values
    values: Vec<Float>,
    /// Column indices for each value
    column_indices: Vec<usize>,
    /// Row pointers (start index for each row)
    row_pointers: Vec<usize>,
    /// Matrix dimensions
    rows: usize,
    cols: usize,
}

impl SparseMatrix {
    /// Create empty sparse matrix with given dimensions
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let mut row_pointers = Vec::with_capacity(rows + 1);
        row_pointers.resize(rows + 1, 0);
        
        Self {
            values: Vec::new(),
            column_indices: Vec::new(),
            row_pointers,
            rows,
            cols,
        }
    }
    
    /// Create a new matrix (alias for zeros for compatibility)
    pub fn new(rows: usize, cols: usize) -> Self {
        Self::zeros(rows, cols)
    }
    
    /// Create a new matrix with capacity hint
    pub fn with_capacity(rows: usize, cols: usize, capacity: usize) -> Self {
        let mut row_pointers = Vec::with_capacity(rows + 1);
        row_pointers.resize(rows + 1, 0);
        
        Self {
            values: Vec::with_capacity(capacity),
            column_indices: Vec::with_capacity(capacity),
            row_pointers,
            rows,
            cols,
        }
    }
    
    /// Create sparse matrix from coordinate format (COO)
    pub fn from_coo(
        rows: usize,
        cols: usize,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<Float>,
    ) -> Result<Self> {
        if row_indices.len() != col_indices.len() || row_indices.len() != values.len() {
            return Err(MathError::DimensionMismatch {
                expected: row_indices.len(),
                got: values.len(),
            });
        }
        
        // Convert COO to CSR format
        let mut matrix = Self::zeros(rows, cols);
        
        // Count non-zeros per row
        let mut row_counts = vec![0; rows];
        for &row in &row_indices {
            if row >= rows {
                return Err(MathError::IndexOutOfBounds { index: row, len: rows });
            }
            row_counts[row] += 1;
        }
        
        // Set up row pointers
        let mut cumsum = 0;
        for i in 0..rows {
            matrix.row_pointers[i] = cumsum;
            cumsum += row_counts[i];
        }
        matrix.row_pointers[rows] = cumsum;
        
        // Fill values and column indices
        matrix.values.resize(values.len(), 0.0);
        matrix.column_indices.resize(values.len(), 0);
        
        let mut row_positions = matrix.row_pointers[..rows].to_vec();
        
        for ((row, col), value) in row_indices.into_iter().zip(col_indices).zip(values) {
            if col >= cols {
                return Err(MathError::IndexOutOfBounds { index: col, len: cols });
            }
            
            let pos = row_positions[row];
            matrix.values[pos] = value;
            matrix.column_indices[pos] = col;
            row_positions[row] += 1;
        }
        
        Ok(matrix)
    }
    
    /// Create sparse matrix from triplets (row, col, value)
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: Vec<(usize, usize, Float)>,
    ) -> Result<Self> {
        let (row_indices, col_indices, values): (Vec<_>, Vec<_>, Vec<_>) = 
            triplets.into_iter().unzip3();
        
        Self::from_coo(rows, cols, row_indices, col_indices, values)
    }
    
    /// Get matrix dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    /// Get number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    /// Get density (nnz / total_elements)
    pub fn density(&self) -> Float {
        if self.rows == 0 || self.cols == 0 {
            return 0.0;
        }
        self.nnz() as Float / (self.rows * self.cols) as Float
    }
    
    /// Get element at position (row, col)
    pub fn get(&self, row: usize, col: usize) -> Result<Float> {
        if row >= self.rows || col >= self.cols {
            return Err(MathError::IndexOutOfBounds {
                index: row * self.cols + col,
                len: self.rows * self.cols,
            });
        }
        
        let start = self.row_pointers[row];
        let end = self.row_pointers[row + 1];
        
        for i in start..end {
            if self.column_indices[i] == col {
                return Ok(self.values[i]);
            }
        }
        
        Ok(0.0) // Implicit zero
    }
    
    /// Set element at position (row, col) - this is expensive for sparse matrices
    pub fn set(&mut self, row: usize, col: usize, value: Float) -> Result<()> {
        if row >= self.rows || col >= self.cols {
            return Err(MathError::IndexOutOfBounds {
                index: row * self.cols + col,
                len: self.rows * self.cols,
            });
        }
        
        let start = self.row_pointers[row];
        let end = self.row_pointers[row + 1];
        
        // Try to find existing element
        for i in start..end {
            if self.column_indices[i] == col {
                self.values[i] = value;
                return Ok(());
            }
        }
        
        // Element doesn't exist - need to insert (expensive operation)
        if value != 0.0 {
            self.insert_element(row, col, value)?;
        }
        
        Ok(())
    }
    
    /// Matrix-vector multiplication (SpMV)
    pub fn multiply_vector(&self, x: &[Float]) -> Result<Vec<Float>> {
        if x.len() != self.cols {
            return Err(MathError::DimensionMismatch {
                expected: self.cols,
                got: x.len(),
            });
        }
        
        let mut result = vec![0.0; self.rows];
        
        for row in 0..self.rows {
            let start = self.row_pointers[row];
            let end = self.row_pointers[row + 1];
            
            for i in start..end {
                let col = self.column_indices[i];
                let value = self.values[i];
                result[row] += value * x[col];
            }
        }
        
        Ok(result)
    }
    
    /// Transpose the sparse matrix
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        
        // Count non-zeros per column (which becomes rows in transpose)
        let mut col_counts = vec![0; self.cols];
        for &col in &self.column_indices {
            col_counts[col] += 1;
        }
        
        // Set up row pointers for transpose
        let mut cumsum = 0;
        for i in 0..self.cols {
            result.row_pointers[i] = cumsum;
            cumsum += col_counts[i];
        }
        result.row_pointers[self.cols] = cumsum;
        
        // Fill values and column indices for transpose
        result.values.resize(self.nnz(), 0.0);
        result.column_indices.resize(self.nnz(), 0);
        
        let mut col_positions = result.row_pointers[..self.cols].to_vec();
        
        for row in 0..self.rows {
            let start = self.row_pointers[row];
            let end = self.row_pointers[row + 1];
            
            for i in start..end {
                let col = self.column_indices[i];
                let value = self.values[i];
                
                let pos = col_positions[col];
                result.values[pos] = value;
                result.column_indices[pos] = row;
                col_positions[col] += 1;
            }
        }
        
        result
    }
    
    /// Iterator over non-zero elements of a row
    pub fn row_iter(&self, row: usize) -> impl Iterator<Item = (usize, Float)> + '_ {
        let start = self.row_pointers[row];
        let end = self.row_pointers[row + 1];
        
        (start..end).map(move |i| (self.column_indices[i], self.values[i]))
    }
    
    /// Iterator over non-zero elements of a column (requires transpose for efficiency)
    pub fn col_iter(&self, col: usize) -> impl Iterator<Item = (usize, Float)> + '_ {
        (0..self.rows).filter_map(move |row| {
            self.get(row, col).ok().map(|value| (row, value))
        }).filter(|(_, value)| *value != 0.0)
    }
    
    /// Get dimensions of the matrix
    pub fn dims(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    
    /// Estimate memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        core::mem::size_of::<Self>() +
        self.values.len() * core::mem::size_of::<Float>() +
        self.column_indices.len() * core::mem::size_of::<usize>() +
        self.row_pointers.len() * core::mem::size_of::<usize>()
    }
    
    /// Apply function to all non-zero values
    pub fn map_values<F>(&mut self, f: F)
    where
        F: Fn(Float) -> Float,
    {
        for value in &mut self.values {
            *value = f(*value);
        }
    }
    
    /// Add scalar to all non-zero elements
    pub fn add_scalar(&mut self, scalar: Float) {
        for value in &mut self.values {
            *value += scalar;
        }
    }
    
    /// Scale all values by scalar
    pub fn scale(&mut self, scalar: Float) {
        for value in &mut self.values {
            *value *= scalar;
        }
    }
    
    /// Convert to dense matrix (expensive for large matrices)
    pub fn to_dense(&self) -> crate::matrix::Matrix {
        let mut dense = crate::matrix::Matrix::zeros(self.rows, self.cols);
        
        for row in 0..self.rows {
            let start = self.row_pointers[row];
            let end = self.row_pointers[row + 1];
            
            for i in start..end {
                let col = self.column_indices[i];
                let value = self.values[i];
                let _ = dense.set(row, col, value);
            }
        }
        
        dense
    }
    
    /// Private method to insert new element (expensive)
    fn insert_element(&mut self, row: usize, col: usize, value: Float) -> Result<()> {
        // Find insertion point
        let start = self.row_pointers[row];
        let end = self.row_pointers[row + 1];
        
        let mut insert_pos = end;
        for i in start..end {
            if self.column_indices[i] > col {
                insert_pos = i;
                break;
            }
        }
        
        // Insert value and column index
        self.values.insert(insert_pos, value);
        self.column_indices.insert(insert_pos, col);
        
        // Update row pointers
        for i in (row + 1)..=self.rows {
            self.row_pointers[i] += 1;
        }
        
        Ok(())
    }
}

/// Sparse vector implementation
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SparseVector {
    /// Non-zero values
    values: Vec<Float>,
    /// Indices of non-zero values
    indices: Vec<usize>,
    /// Vector length
    length: usize,
}

impl SparseVector {
    /// Create empty sparse vector
    pub fn zeros(length: usize) -> Self {
        Self {
            values: Vec::new(),
            indices: Vec::new(),
            length,
        }
    }
    
    /// Create from dense vector
    pub fn from_dense(dense: Vec<Float>) -> Self {
        let mut values = Vec::new();
        let mut indices = Vec::new();
        
        for (i, &value) in dense.iter().enumerate() {
            if value != 0.0 {
                values.push(value);
                indices.push(i);
            }
        }
        
        Self {
            values,
            indices,
            length: dense.len(),
        }
    }
    
    /// Get length
    pub fn len(&self) -> usize {
        self.length
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    
    /// Get number of non-zeros
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    /// Get element at index
    pub fn get(&self, index: usize) -> Result<Float> {
        if index >= self.length {
            return Err(MathError::IndexOutOfBounds {
                index,
                len: self.length,
            });
        }
        
        for (i, &idx) in self.indices.iter().enumerate() {
            if idx == index {
                return Ok(self.values[i]);
            }
        }
        
        Ok(0.0)
    }
    
    /// Dot product with dense vector
    pub fn dot(&self, other: &[Float]) -> Result<Float> {
        if other.len() != self.length {
            return Err(MathError::DimensionMismatch {
                expected: self.length,
                got: other.len(),
            });
        }
        
        let mut result = 0.0;
        for (i, &idx) in self.indices.iter().enumerate() {
            result += self.values[i] * other[idx];
        }
        
        Ok(result)
    }
    
    /// Convert to dense vector
    pub fn to_dense(&self) -> Vec<Float> {
        let mut dense = vec![0.0; self.length];
        for (i, &idx) in self.indices.iter().enumerate() {
            dense[idx] = self.values[i];
        }
        dense
    }
}

// Helper trait to unzip three iterators
trait Unzip3<A, B, C> {
    fn unzip3(self) -> (Vec<A>, Vec<B>, Vec<C>);
}

impl<I, A, B, C> Unzip3<A, B, C> for I
where
    I: Iterator<Item = (A, B, C)>,
{
    fn unzip3(self) -> (Vec<A>, Vec<B>, Vec<C>) {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        
        for (x, y, z) in self {
            a.push(x);
            b.push(y);
            c.push(z);
        }
        
        (a, b, c)
    }
}

/// Convenience functions
pub fn sparse_multiply(a: &SparseMatrix, b: &[Float]) -> Result<Vec<Float>> {
    a.multiply_vector(b)
}

pub fn sparse_add(a: &SparseMatrix, b: &SparseMatrix) -> Result<SparseMatrix> {
    if a.shape() != b.shape() {
        return Err(MathError::DimensionMismatch {
            expected: a.rows * a.cols,
            got: b.rows * b.cols,
        });
    }
    
    // Simple implementation - convert to triplets, merge, and reconstruct
    let mut triplets = Vec::new();
    
    // Add elements from matrix a
    for row in 0..a.rows {
        for (col, value) in a.row_iter(row) {
            triplets.push((row, col, value));
        }
    }
    
    // Add elements from matrix b
    for row in 0..b.rows {
        for (col, value) in b.row_iter(row) {
            // Check if element already exists from matrix a
            if let Some(existing) = triplets.iter_mut().find(|(r, c, _)| *r == row && *c == col) {
                existing.2 += value;
            } else {
                triplets.push((row, col, value));
            }
        }
    }
    
    SparseMatrix::from_triplets(a.rows, a.cols, triplets)
}

/// CSR Matrix type alias for compatibility
pub type CSRMatrix = SparseMatrix;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparse_matrix_creation() {
        let matrix = SparseMatrix::from_triplets(
            3, 3,
            vec![
                (0, 0, 1.0),
                (1, 1, 2.0),
                (2, 2, 3.0),
            ]
        ).unwrap();
        
        assert_eq!(matrix.shape(), (3, 3));
        assert_eq!(matrix.nnz(), 3);
        assert_eq!(matrix.get(0, 0).unwrap(), 1.0);
        assert_eq!(matrix.get(1, 1).unwrap(), 2.0);
        assert_eq!(matrix.get(2, 2).unwrap(), 3.0);
        assert_eq!(matrix.get(0, 1).unwrap(), 0.0);
    }
    
    #[test]
    fn test_sparse_matrix_vector_multiply() {
        let matrix = SparseMatrix::from_triplets(
            2, 3,
            vec![
                (0, 0, 1.0),
                (0, 2, 3.0),
                (1, 1, 2.0),
            ]
        ).unwrap();
        
        let x = vec![1.0, 2.0, 3.0];
        let result = matrix.multiply_vector(&x).unwrap();
        
        assert_eq!(result[0], 1.0 * 1.0 + 3.0 * 3.0); // 10.0
        assert_eq!(result[1], 2.0 * 2.0); // 4.0
    }
    
    #[test]
    fn test_sparse_transpose() {
        let matrix = SparseMatrix::from_triplets(
            2, 3,
            vec![
                (0, 1, 1.0),
                (1, 2, 2.0),
            ]
        ).unwrap();
        
        let transposed = matrix.transpose();
        assert_eq!(transposed.shape(), (3, 2));
        assert_eq!(transposed.get(1, 0).unwrap(), 1.0);
        assert_eq!(transposed.get(2, 1).unwrap(), 2.0);
    }
    
    #[test]
    fn test_sparse_vector() {
        let dense = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        let sparse = SparseVector::from_dense(dense.clone());
        
        assert_eq!(sparse.len(), 5);
        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.get(1).unwrap(), 1.0);
        assert_eq!(sparse.get(3).unwrap(), 2.0);
        assert_eq!(sparse.get(0).unwrap(), 0.0);
        
        let reconstructed = sparse.to_dense();
        assert_eq!(reconstructed, dense);
    }
}