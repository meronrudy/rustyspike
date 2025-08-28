//! Statistical functions for neuromorphic data analysis
//!
//! This module provides statistical operations for analyzing
//! neural network behavior and spike train statistics.

use crate::{Float, Result, MathError};
use crate::math::MathExt;
use core::cmp::Ordering;

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, vec};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Basic descriptive statistics
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DescriptiveStats {
    /// Number of samples
    pub count: usize,
    /// Mean value
    pub mean: Float,
    /// Variance
    pub variance: Float,
    /// Standard deviation
    pub std_dev: Float,
    /// Minimum value
    pub min: Float,
    /// Maximum value
    pub max: Float,
    /// Range (max - min)
    pub range: Float,
    /// Skewness (measure of asymmetry)
    pub skewness: Float,
    /// Kurtosis (measure of tail heaviness)
    pub kurtosis: Float,
}

impl DescriptiveStats {
    /// Compute descriptive statistics from data
    pub fn from_data(data: &[Float]) -> Result<Self> {
        if data.is_empty() {
            return Err(MathError::InvalidInput {
                reason: "Cannot compute statistics on empty data",
            });
        }
        
        let count = data.len();
        let mean = mean(data);
        let variance = variance(data);
        let std_dev = variance.sqrt();
        let min = data.iter().copied().fold(Float::INFINITY, Float::min);
        let max = data.iter().copied().fold(Float::NEG_INFINITY, Float::max);
        let range = max - min;
        let skewness = skewness(data);
        let kurtosis = kurtosis(data);
        
        Ok(Self {
            count,
            mean,
            variance,
            std_dev,
            min,
            max,
            range,
            skewness,
            kurtosis,
        })
    }
    
    /// Get coefficient of variation (std_dev / mean)
    pub fn coefficient_of_variation(&self) -> Float {
        if self.mean.abs() < crate::constants::EPSILON {
            0.0
        } else {
            self.std_dev / self.mean.abs()
        }
    }
    
    /// Check if data appears normally distributed (basic heuristic)
    pub fn is_approximately_normal(&self) -> bool {
        // Simple heuristic: skewness close to 0 and kurtosis close to 3
        self.skewness.abs() < 0.5 && (self.kurtosis - 3.0).abs() < 1.0
    }
}

/// Histogram for data distribution analysis
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Histogram {
    /// Bin edges
    pub bin_edges: Vec<Float>,
    /// Bin counts
    pub counts: Vec<usize>,
    /// Total number of samples
    pub total_count: usize,
}

impl Histogram {
    /// Create histogram from data with specified number of bins
    pub fn from_data(data: &[Float], num_bins: usize) -> Result<Self> {
        if data.is_empty() {
            return Err(MathError::InvalidInput {
                reason: "Cannot create histogram from empty data",
            });
        }
        
        if num_bins == 0 {
            return Err(MathError::InvalidInput {
                reason: "Number of bins must be positive",
            });
        }
        
        let min_val = data.iter().copied().fold(Float::INFINITY, Float::min);
        let max_val = data.iter().copied().fold(Float::NEG_INFINITY, Float::max);
        
        if (max_val - min_val).abs() < crate::constants::EPSILON {
            // All values are the same
            let mut bin_edges = vec![min_val];
            bin_edges.push(min_val + 1.0);
            let counts = vec![data.len()];
            
            return Ok(Self {
                bin_edges,
                counts,
                total_count: data.len(),
            });
        }
        
        // Create bin edges
        let bin_width = (max_val - min_val) / num_bins as Float;
        let mut bin_edges = Vec::with_capacity(num_bins + 1);
        for i in 0..=num_bins {
            bin_edges.push(min_val + i as Float * bin_width);
        }
        
        // Ensure the last edge covers the maximum value
        bin_edges[num_bins] = max_val + crate::constants::EPSILON;
        
        // Count data points in each bin
        let mut counts = vec![0; num_bins];
        for &value in data {
            let bin_index = ((value - min_val) / bin_width) as usize;
            let bin_index = bin_index.min(num_bins - 1);
            counts[bin_index] += 1;
        }
        
        Ok(Self {
            bin_edges,
            counts,
            total_count: data.len(),
        })
    }
    
    /// Get bin centers
    pub fn bin_centers(&self) -> Vec<Float> {
        let mut centers = Vec::with_capacity(self.counts.len());
        for i in 0..self.counts.len() {
            let center = (self.bin_edges[i] + self.bin_edges[i + 1]) / 2.0;
            centers.push(center);
        }
        centers
    }
    
    /// Get normalized frequencies (probabilities)
    pub fn frequencies(&self) -> Vec<Float> {
        if self.total_count == 0 {
            return vec![0.0; self.counts.len()];
        }
        
        self.counts.iter()
            .map(|&count| count as Float / self.total_count as Float)
            .collect()
    }
    
    /// Get probability density (frequency / bin_width)
    pub fn density(&self) -> Vec<Float> {
        let frequencies = self.frequencies();
        let mut densities = Vec::with_capacity(frequencies.len());
        
        for (i, freq) in frequencies.iter().enumerate() {
            let bin_width = self.bin_edges[i + 1] - self.bin_edges[i];
            densities.push(freq / bin_width);
        }
        
        densities
    }
}

/// Basic statistical functions

/// Calculate mean of a data slice
pub fn mean(data: &[Float]) -> Float {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<Float>() / data.len() as Float
}

/// Calculate sample variance (Bessel's correction)
pub fn variance(data: &[Float]) -> Float {
    if data.len() < 2 {
        return 0.0;
    }
    
    let mean_val = mean(data);
    let sum_sq_diff: Float = data.iter()
        .map(|&x| (x - mean_val) * (x - mean_val))
        .sum();
    
    sum_sq_diff / (data.len() - 1) as Float
}

/// Calculate population variance (no Bessel's correction)
pub fn variance_population(data: &[Float]) -> Float {
    if data.is_empty() {
        return 0.0;
    }
    
    let mean_val = mean(data);
    let sum_sq_diff: Float = data.iter()
        .map(|&x| (x - mean_val) * (x - mean_val))
        .sum();
    
    sum_sq_diff / data.len() as Float
}

/// Calculate standard deviation
pub fn standard_deviation(data: &[Float]) -> Float {
    variance(data).sqrt()
}

/// Calculate skewness (measure of asymmetry)
pub fn skewness(data: &[Float]) -> Float {
    if data.len() < 3 {
        return 0.0;
    }
    
    let mean_val = mean(data);
    let std_val = standard_deviation(data);
    
    if std_val < crate::constants::EPSILON {
        return 0.0;
    }
    
    let n = data.len() as Float;
    let sum_cubed_deviations: Float = data.iter()
        .map(|&x| ((x - mean_val) / std_val).powi(3))
        .sum();
    
    (n / ((n - 1.0) * (n - 2.0))) * sum_cubed_deviations
}

/// Calculate kurtosis (measure of tail heaviness)
pub fn kurtosis(data: &[Float]) -> Float {
    if data.len() < 4 {
        return 0.0;
    }
    
    let mean_val = mean(data);
    let std_val = standard_deviation(data);
    
    if std_val < crate::constants::EPSILON {
        return 0.0;
    }
    
    let n = data.len() as Float;
    let sum_fourth_deviations: Float = data.iter()
        .map(|&x| ((x - mean_val) / std_val).powi(4))
        .sum();
    
    let kurtosis_raw = sum_fourth_deviations / n;
    
    // Excess kurtosis (subtract 3 for normal distribution baseline)
    kurtosis_raw
}

/// Calculate Pearson correlation coefficient
pub fn correlation(x: &[Float], y: &[Float]) -> Result<Float> {
    if x.len() != y.len() {
        return Err(MathError::DimensionMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }
    
    if x.len() < 2 {
        return Err(MathError::InvalidInput {
            reason: "Need at least 2 data points for correlation",
        });
    }
    
    let mean_x = mean(x);
    let mean_y = mean(y);
    
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }
    
    let denominator = (sum_x2 * sum_y2).sqrt();
    if denominator < crate::constants::EPSILON {
        return Ok(0.0);
    }
    
    Ok(sum_xy / denominator)
}

/// Calculate covariance
pub fn covariance(x: &[Float], y: &[Float]) -> Result<Float> {
    if x.len() != y.len() {
        return Err(MathError::DimensionMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }
    
    if x.len() < 2 {
        return Err(MathError::InvalidInput {
            reason: "Need at least 2 data points for covariance",
        });
    }
    
    let mean_x = mean(x);
    let mean_y = mean(y);
    
    let sum_products: Float = x.iter().zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    
    Ok(sum_products / (x.len() - 1) as Float)
}

/// Calculate quantile (percentile)
pub fn quantile(data: &[Float], q: Float) -> Result<Float> {
    if data.is_empty() {
        return Err(MathError::InvalidInput {
            reason: "Cannot calculate quantile of empty data",
        });
    }
    
    if q < 0.0 || q > 1.0 {
        return Err(MathError::InvalidInput {
            reason: "Quantile must be between 0 and 1",
        });
    }
    
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    
    if q == 0.0 {
        return Ok(sorted_data[0]);
    }
    if q == 1.0 {
        return Ok(sorted_data[sorted_data.len() - 1]);
    }
    
    let index = q * (sorted_data.len() - 1) as Float;
    let lower_index = index.floor() as usize;
    let upper_index = index.ceil() as usize;
    
    if lower_index == upper_index {
        Ok(sorted_data[lower_index])
    } else {
        let weight = index - lower_index as Float;
        Ok(sorted_data[lower_index] * (1.0 - weight) + sorted_data[upper_index] * weight)
    }
}

/// Calculate median (50th percentile)
pub fn median(data: &[Float]) -> Result<Float> {
    quantile(data, 0.5)
}

/// Calculate interquartile range (IQR)
pub fn interquartile_range(data: &[Float]) -> Result<Float> {
    let q75 = quantile(data, 0.75)?;
    let q25 = quantile(data, 0.25)?;
    Ok(q75 - q25)
}

/// Normalize data to zero mean and unit variance (z-score)
pub fn normalize(data: &mut [Float]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    
    let mean_val = mean(data);
    let std_val = standard_deviation(data);
    
    if std_val < crate::constants::EPSILON {
        // All values are the same, set to zero
        for x in data {
            *x = 0.0;
        }
        return Ok(());
    }
    
    for x in data {
        *x = (*x - mean_val) / std_val;
    }
    
    Ok(())
}

/// Min-max normalization to range [0, 1]
pub fn min_max_normalize(data: &mut [Float]) -> Result<()> {
    if data.is_empty() {
        return Ok(());
    }
    
    let min_val = data.iter().copied().fold(Float::INFINITY, Float::min);
    let max_val = data.iter().copied().fold(Float::NEG_INFINITY, Float::max);
    
    let range = max_val - min_val;
    if range < crate::constants::EPSILON {
        // All values are the same, set to 0.5
        for x in data {
            *x = 0.5;
        }
        return Ok(());
    }
    
    for x in data {
        *x = (*x - min_val) / range;
    }
    
    Ok(())
}

/// Calculate entropy of a probability distribution
pub fn entropy(probabilities: &[Float]) -> Result<Float> {
    // Validate probabilities
    let sum: Float = probabilities.iter().sum();
    if (sum - 1.0).abs() > 1e-6 {
        return Err(MathError::InvalidInput {
            reason: "Probabilities must sum to 1",
        });
    }
    
    let entropy: Float = probabilities.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    
    Ok(entropy)
}

/// Calculate Kullback-Leibler divergence
pub fn kl_divergence(p: &[Float], q: &[Float]) -> Result<Float> {
    if p.len() != q.len() {
        return Err(MathError::DimensionMismatch {
            expected: p.len(),
            got: q.len(),
        });
    }
    
    let mut kl = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi > 0.0 {
            if qi <= 0.0 {
                return Ok(Float::INFINITY);
            }
            kl += pi * (pi / qi).ln();
        }
    }
    
    Ok(kl)
}

/// Calculate mutual information between two discrete distributions
pub fn mutual_information(joint: &[Float], marginal_x: &[Float], marginal_y: &[Float]) -> Result<Float> {
    if joint.len() != marginal_x.len() * marginal_y.len() {
        return Err(MathError::DimensionMismatch {
            expected: marginal_x.len() * marginal_y.len(),
            got: joint.len(),
        });
    }
    
    let mut mi = 0.0;
    let mut idx = 0;
    
    for &px in marginal_x {
        for &py in marginal_y {
            let pxy = joint[idx];
            if pxy > 0.0 && px > 0.0 && py > 0.0 {
                mi += pxy * (pxy / (px * py)).ln();
            }
            idx += 1;
        }
    }
    
    Ok(mi)
}

/// Simple linear regression
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LinearRegression {
    /// Slope
    pub slope: Float,
    /// Intercept
    pub intercept: Float,
    /// R-squared (coefficient of determination)
    pub r_squared: Float,
}

impl LinearRegression {
    /// Fit linear regression to data
    pub fn fit(x: &[Float], y: &[Float]) -> Result<Self> {
        if x.len() != y.len() {
            return Err(MathError::DimensionMismatch {
                expected: x.len(),
                got: y.len(),
            });
        }
        
        if x.len() < 2 {
            return Err(MathError::InvalidInput {
                reason: "Need at least 2 data points for regression",
            });
        }
        
        let mean_x = mean(x);
        let mean_y = mean(y);
        
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;
        
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }
        
        if sum_x2 < crate::constants::EPSILON {
            return Err(MathError::InvalidInput {
                reason: "X values have no variance",
            });
        }
        
        let slope = sum_xy / sum_x2;
        let intercept = mean_y - slope * mean_x;
        
        // Calculate R-squared
        let r_squared = if sum_y2 < crate::constants::EPSILON {
            1.0 // Perfect fit if Y has no variance
        } else {
            (sum_xy * sum_xy) / (sum_x2 * sum_y2)
        };
        
        Ok(Self {
            slope,
            intercept,
            r_squared,
        })
    }
    
    /// Predict Y value for given X
    pub fn predict(&self, x: Float) -> Float {
        self.slope * x + self.intercept
    }
    
    /// Predict multiple Y values
    pub fn predict_vec(&self, x: &[Float]) -> Vec<Float> {
        x.iter().map(|&xi| self.predict(xi)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_stats() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        
        assert_eq!(mean(&data), 3.0);
        assert_eq!(variance(&data), 2.5);
        assert!((standard_deviation(&data) - 2.5_f32.sqrt()).abs() < 1e-6);
    }
    
    #[test]
    fn test_descriptive_stats() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = DescriptiveStats::from_data(&data).unwrap();
        
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.range, 4.0);
        assert_eq!(stats.count, 5);
    }
    
    #[test]
    fn test_histogram() {
        let data = [1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0];
        let hist = Histogram::from_data(&data, 4).unwrap();
        
        assert_eq!(hist.counts.len(), 4);
        assert_eq!(hist.total_count, 7);
        
        let frequencies = hist.frequencies();
        let sum: Float = frequencies.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_correlation() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation
        
        let corr = correlation(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-6);
        
        let y_neg = [10.0, 8.0, 6.0, 4.0, 2.0]; // Perfect negative correlation
        let corr_neg = correlation(&x, &y_neg).unwrap();
        assert!((corr_neg + 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_quantiles() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        assert_eq!(quantile(&data, 0.0).unwrap(), 1.0);
        assert_eq!(quantile(&data, 1.0).unwrap(), 10.0);
        assert_eq!(median(&data).unwrap(), 5.5);
        
        let q25 = quantile(&data, 0.25).unwrap();
        let q75 = quantile(&data, 0.75).unwrap();
        let iqr = interquartile_range(&data).unwrap();
        assert!((iqr - (q75 - q25)).abs() < 1e-6);
    }
    
    #[test]
    fn test_normalization() {
        let mut data = [1.0, 2.0, 3.0, 4.0, 5.0];
        
        normalize(&mut data).unwrap();
        let normalized_mean = mean(&data);
        let normalized_std = standard_deviation(&data);
        
        assert!(normalized_mean.abs() < 1e-6);
        assert!((normalized_std - 1.0).abs() < 1e-6);
        
        let mut data2 = [1.0, 2.0, 3.0, 4.0, 5.0];
        min_max_normalize(&mut data2).unwrap();
        assert_eq!(data2[0], 0.0);
        assert_eq!(data2[4], 1.0);
    }
    
    #[test]
    fn test_linear_regression() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x
        
        let regression = LinearRegression::fit(&x, &y).unwrap();
        
        assert!((regression.slope - 2.0).abs() < 1e-6);
        assert!(regression.intercept.abs() < 1e-6);
        assert!((regression.r_squared - 1.0).abs() < 1e-6);
        
        assert_eq!(regression.predict(6.0), 12.0);
    }
    
    #[test]
    fn test_entropy() {
        let uniform = [0.25, 0.25, 0.25, 0.25];
        let entropy_uniform = entropy(&uniform).unwrap();
        
        let peaked = [0.97, 0.01, 0.01, 0.01];
        let entropy_peaked = entropy(&peaked).unwrap();
        
        // Uniform distribution should have higher entropy
        assert!(entropy_uniform > entropy_peaked);
    }
}