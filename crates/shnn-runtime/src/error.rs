//! Error types for the SNN runtime

use thiserror::Error;

/// Result type for runtime operations
pub type Result<T> = std::result::Result<T, RuntimeError>;

/// Errors that can occur in the SNN runtime
#[derive(Error, Debug)]
pub enum RuntimeError {
    /// Storage layer error
    #[error("Storage error: {source}")]
    Storage {
        #[from]
        /// Source storage error
        source: shnn_storage::StorageError,
    },

    /// Invalid network configuration
    #[error("Invalid network configuration: {reason}")]
    InvalidConfiguration { 
        /// Reason for invalid configuration
        reason: String 
    },

    /// Neuron not found
    #[error("Neuron {neuron_id} not found")]
    NeuronNotFound { 
        /// Neuron ID that was not found
        neuron_id: u32 
    },

    /// Invalid parameter value
    #[error("Invalid parameter {parameter}: {value} (expected {constraint})")]
    InvalidParameter { 
        /// Parameter name
        parameter: String, 
        /// Invalid value
        value: String,
        /// Constraint description
        constraint: String 
    },

    /// Simulation step failed
    #[error("Simulation step failed at time {time_ns}ns: {reason}")]
    SimulationStep { 
        /// Time when step failed
        time_ns: u64, 
        /// Reason for failure
        reason: String 
    },

    /// Network topology error
    #[error("Network topology error: {reason}")]
    NetworkTopology { 
        /// Reason for topology error
        reason: String 
    },

    /// Plasticity rule error
    #[error("Plasticity rule error: {reason}")]
    PlasticityError { 
        /// Reason for plasticity error
        reason: String 
    },

    /// Numerical computation error
    #[error("Numerical error: {reason}")]
    NumericalError { 
        /// Reason for numerical error
        reason: String 
    },

    /// Resource exhaustion
    #[error("Resource exhausted: {resource} (limit: {limit})")]
    ResourceExhausted { 
        /// Resource name
        resource: String, 
        /// Resource limit
        limit: String 
    },
}

impl RuntimeError {
    /// Create an invalid configuration error
    pub fn invalid_config(reason: impl Into<String>) -> Self {
        Self::InvalidConfiguration {
            reason: reason.into(),
        }
    }

    /// Create an invalid parameter error
    pub fn invalid_parameter(
        parameter: impl Into<String>,
        value: impl Into<String>,
        constraint: impl Into<String>,
    ) -> Self {
        Self::InvalidParameter {
            parameter: parameter.into(),
            value: value.into(),
            constraint: constraint.into(),
        }
    }

    /// Create a simulation step error
    pub fn simulation_step(time_ns: u64, reason: impl Into<String>) -> Self {
        Self::SimulationStep {
            time_ns,
            reason: reason.into(),
        }
    }

    /// Create a network topology error
    pub fn network_topology(reason: impl Into<String>) -> Self {
        Self::NetworkTopology {
            reason: reason.into(),
        }
    }

    /// Create a plasticity error
    pub fn plasticity_error(reason: impl Into<String>) -> Self {
        Self::PlasticityError {
            reason: reason.into(),
        }
    }

    /// Create a numerical error
    pub fn numerical_error(reason: impl Into<String>) -> Self {
        Self::NumericalError {
            reason: reason.into(),
        }
    }

    /// Create a resource exhausted error
    pub fn resource_exhausted(
        resource: impl Into<String>,
        limit: impl Into<String>,
    ) -> Self {
        Self::ResourceExhausted {
            resource: resource.into(),
            limit: limit.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = RuntimeError::invalid_config("missing neurons");
        assert!(matches!(err, RuntimeError::InvalidConfiguration { .. }));
        
        let err = RuntimeError::invalid_parameter("tau_m", "0.0", "> 0.0");
        assert!(matches!(err, RuntimeError::InvalidParameter { .. }));
    }

    #[test]
    fn test_error_display() {
        let err = RuntimeError::NeuronNotFound { neuron_id: 42 };
        let msg = format!("{}", err);
        assert!(msg.contains("Neuron 42 not found"));
    }
}