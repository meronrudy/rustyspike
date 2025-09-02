//! Sum-type connectivity that unifies plastic-capable backends with static dispatch
//!
//! This enum allows enabling plasticity updates without trait-object indirection,
//! by providing a single concrete type that implements both NetworkConnectivity and
//! PlasticConnectivity, dispatching to the selected inner connectivity at runtime.

use crate::{
    connectivity::{
        types::{ConnectivityStats, SpikeRoute, ConnectionId},
        graph::GraphNetwork,
        matrix::MatrixNetwork,
        sparse::SparseMatrixNetwork,
        NetworkConnectivity, PlasticConnectivity,
    },
    error::{Result, SHNNError},
    spike::{NeuronId, Spike},
    time::Time,
};

/// Unified plastic-capable connectivity wrapper
#[derive(Debug, Clone)]
pub enum PlasticConn {
    /// Traditional graph connectivity
    Graph(GraphNetwork),
    /// Dense matrix connectivity
    Matrix(MatrixNetwork),
    /// Sparse matrix connectivity
    Sparse(SparseMatrixNetwork),
    // Future: Hypergraph connectivity (placeholder for expansion)
    // Hypergraph(HypergraphNetwork),
}

impl PlasticConn {
    /// Construct from a GraphNetwork
    pub fn from_graph(inner: GraphNetwork) -> Self {
        Self::Graph(inner)
    }
    /// Construct from a MatrixNetwork
    pub fn from_matrix(inner: MatrixNetwork) -> Self {
        Self::Matrix(inner)
    }
    /// Construct from a SparseMatrixNetwork
    pub fn from_sparse(inner: SparseMatrixNetwork) -> Self {
        Self::Sparse(inner)
    }

    /// Return the number of neurons for the current variant
    fn neuron_len(&self) -> usize {
        match self {
            Self::Graph(g) => g.neuron_count(),
            Self::Matrix(m) => m.neuron_count(),
            Self::Sparse(s) => s.neuron_count(),
        }
    }

    /// Convert a (pre, post) tuple into a variant-specific connection id for Graph
    fn graph_id_from_tuple(conn: (NeuronId, NeuronId)) -> crate::connectivity::graph::GraphConnectionId {
        crate::connectivity::graph::GraphConnectionId {
            source: conn.0,
            target: conn.1,
        }
    }

    /// Ensure both neurons exist in Matrix variant and return (row, col) indices
    fn matrix_indices_for(
        matrix: &mut MatrixNetwork,
        conn: (NeuronId, NeuronId),
    ) -> Result<(usize, usize)> {
        let (pre, post) = conn;
        let row = match matrix.get_neuron_index(pre) {
            Some(i) => i,
            None => matrix.add_neuron(pre)?,
        };
        let col = match matrix.get_neuron_index(post) {
            Some(i) => i,
            None => matrix.add_neuron(post)?,
        };
        Ok((row, col))
    }

    /// Ensure both neurons exist in Sparse variant and return (row, col) indices
    fn sparse_indices_for(
        sparse: &mut SparseMatrixNetwork,
        conn: (NeuronId, NeuronId),
    ) -> Result<(usize, usize)> {
        let (pre, post) = conn;
        let row = match sparse.get_neuron_index(pre) {
            Some(i) => i,
            None => sparse.add_neuron(pre)?,
        };
        let col = match sparse.get_neuron_index(post) {
            Some(i) => i,
            None => sparse.add_neuron(post)?,
        };
        Ok((row, col))
    }
}

impl NetworkConnectivity<NeuronId> for PlasticConn {
    type ConnectionId = (NeuronId, NeuronId);
    type RouteInfo = SpikeRoute;
    type Error = SHNNError;

    fn route_spike(&self, spike: &Spike, current_time: Time) -> Result<Vec<SpikeRoute>> {
        match self {
            Self::Graph(g) => g.route_spike(spike, current_time),
            Self::Matrix(m) => m.route_spike(spike, current_time),
            Self::Sparse(s) => s.route_spike(spike, current_time),
        }
    }

    fn get_targets(&self, source: NeuronId) -> Result<Vec<NeuronId>> {
        match self {
            Self::Graph(g) => g.get_targets(source),
            Self::Matrix(m) => m.get_targets(source),
            Self::Sparse(s) => s.get_targets(source),
        }
    }

    fn get_sources(&self, target: NeuronId) -> Result<Vec<NeuronId>> {
        match self {
            Self::Graph(g) => g.get_sources(target),
            Self::Matrix(m) => m.get_sources(target),
            Self::Sparse(s) => s.get_sources(target),
        }
    }

    fn add_connection(&mut self, connection: Self::ConnectionId) -> Result<()> {
        match self {
            Self::Graph(g) => {
                let id = PlasticConn::graph_id_from_tuple(connection);
                g.add_connection(id)
            }
            Self::Matrix(m) => {
                let (row, col) = PlasticConn::matrix_indices_for(m, connection)?;
                let id = crate::connectivity::matrix::MatrixConnectionId { row, col };
                m.add_connection(id)
            }
            Self::Sparse(s) => {
                let (row, col) = PlasticConn::sparse_indices_for(s, connection)?;
                let id = crate::connectivity::sparse::SparseConnectionId { row, col };
                s.add_connection(id)
            }
        }
    }

    fn remove_connection(&mut self, connection: Self::ConnectionId) -> Result<Option<Self::RouteInfo>> {
        match self {
            Self::Graph(g) => {
                let id = PlasticConn::graph_id_from_tuple(connection);
                let opt = g.remove_connection(id)?;
                Ok(opt.map(|info| {
                    // Convert GraphRouteInfo -> SpikeRoute
                    let edge = info.edge;
                    SpikeRoute::new(
                        edge.id.to_raw(),
                        vec![edge.id.target],
                        vec![edge.weight],
                        Time::ZERO,
                    )
                    .expect("valid route from graph remove_connection")
                }))
            }
            Self::Matrix(m) => {
                let (row, col) = PlasticConn::matrix_indices_for(m, connection)?;
                let id = crate::connectivity::matrix::MatrixConnectionId { row, col };
                let opt = m.remove_connection(id)?;
                if let Some(info) = opt {
                    // Expect a single target entry for a single removed connection
                    let targets: Vec<NeuronId> = info
                        .target_indices
                        .iter()
                        .filter_map(|&idx| m.get_neuron_id(idx))
                        .collect();
                    if targets.is_empty() {
                        return Ok(None);
                    }
                    let weights = info.weights.clone();
                    let route = SpikeRoute::new(
                        crate::connectivity::matrix::MatrixConnectionId { row: info.source_index, col }
                            .to_raw(),
                        targets,
                        weights,
                        Time::ZERO,
                    )
                    .map_err(|_| SHNNError::generic("invalid route from removal"))?;
                    Ok(Some(route))
                } else {
                    Ok(None)
                }
            }
            Self::Sparse(s) => {
                let (row, col) = PlasticConn::sparse_indices_for(s, connection)?;
                let id = crate::connectivity::sparse::SparseConnectionId { row, col };
                let opt = s.remove_connection(id)?;
                if let Some(info) = opt {
                    let targets: Vec<NeuronId> = info
                        .target_indices
                        .iter()
                        .filter_map(|&idx| s.get_neuron_id(idx))
                        .collect();
                    if targets.is_empty() {
                        return Ok(None);
                    }
                    let weights = info.weights.clone();
                    let route = SpikeRoute::new(
                        crate::connectivity::sparse::SparseConnectionId { row: info.source_index, col }
                            .to_raw(),
                        targets,
                        weights,
                        Time::ZERO,
                    )
                    .map_err(|_| SHNNError::generic("invalid route from removal"))?;
                    Ok(Some(route))
                } else {
                    Ok(None)
                }
            }
        }
    }

    fn update_weight(&mut self, connection: Self::ConnectionId, new_weight: f32) -> Result<Option<f32>> {
        match self {
            Self::Graph(g) => {
                let id = PlasticConn::graph_id_from_tuple(connection);
                g.update_weight(id, new_weight)
            }
            Self::Matrix(m) => {
                let (row, col) = PlasticConn::matrix_indices_for(m, connection)?;
                let id = crate::connectivity::matrix::MatrixConnectionId { row, col };
                m.update_weight(id, new_weight)
            }
            Self::Sparse(s) => {
                let (row, col) = PlasticConn::sparse_indices_for(s, connection)?;
                let id = crate::connectivity::sparse::SparseConnectionId { row, col };
                s.update_weight(id, new_weight)
            }
        }
    }

    fn get_stats(&self) -> ConnectivityStats {
        match self {
            Self::Graph(g) => g.get_stats(),
            Self::Matrix(m) => m.get_stats(),
            Self::Sparse(s) => s.get_stats(),
        }
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::Graph(g) => g.validate(),
            Self::Matrix(m) => m.validate(),
            Self::Sparse(s) => s.validate(),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Graph(g) => g.reset(),
            Self::Matrix(m) => m.reset(),
            Self::Sparse(s) => s.reset(),
        }
    }

    fn connection_count(&self) -> usize {
        match self {
            Self::Graph(g) => g.connection_count(),
            Self::Matrix(m) => m.connection_count(),
            Self::Sparse(s) => s.connection_count(),
        }
    }

    fn neuron_count(&self) -> usize {
        self.neuron_len()
    }
}

impl PlasticConnectivity<NeuronId> for PlasticConn {
    fn apply_plasticity(
        &mut self,
        pre_neuron: NeuronId,
        post_neuron: NeuronId,
        weight_delta: f32,
    ) -> Result<Option<f32>> {
        match self {
            Self::Graph(g) => g.apply_plasticity(pre_neuron, post_neuron, weight_delta),
            Self::Matrix(m) => m.apply_plasticity(pre_neuron, post_neuron, weight_delta),
            Self::Sparse(s) => s.apply_plasticity(pre_neuron, post_neuron, weight_delta),
        }
    }

    fn get_weight(&self, pre_neuron: NeuronId, post_neuron: NeuronId) -> Result<Option<f32>> {
        match self {
            Self::Graph(g) => g.get_weight(pre_neuron, post_neuron),
            Self::Matrix(m) => <MatrixNetwork as PlasticConnectivity<NeuronId>>::get_weight(m, pre_neuron, post_neuron),
            Self::Sparse(s) => s.get_weight(pre_neuron, post_neuron),
        }
    }
}

// Phase 6: snapshot/apply delegation for PlasticConn
impl crate::connectivity::WeightSnapshotConnectivity<NeuronId> for PlasticConn {
    fn snapshot_weights(&self) -> Vec<(NeuronId, NeuronId, f32)> {
        match self {
            Self::Graph(g) => <_ as crate::connectivity::WeightSnapshotConnectivity<NeuronId>>::snapshot_weights(g),
            Self::Matrix(m) => <_ as crate::connectivity::WeightSnapshotConnectivity<NeuronId>>::snapshot_weights(m),
            Self::Sparse(s) => <_ as crate::connectivity::WeightSnapshotConnectivity<NeuronId>>::snapshot_weights(s),
        }
    }

    fn apply_weight_updates(&mut self, updates: &[(NeuronId, NeuronId, f32)]) -> Result<usize> {
        match self {
            Self::Graph(g) => <_ as crate::connectivity::WeightSnapshotConnectivity<NeuronId>>::apply_weight_updates(g, updates),
            Self::Matrix(m) => <_ as crate::connectivity::WeightSnapshotConnectivity<NeuronId>>::apply_weight_updates(m, updates),
            Self::Sparse(s) => <_ as crate::connectivity::WeightSnapshotConnectivity<NeuronId>>::apply_weight_updates(s, updates),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spike::NeuronId;
    use crate::connectivity::PlasticConnectivity;

    #[test]
    fn plastic_conn_dispatch_graph_only() {
        // Construct an empty graph variant and verify calls work
        let mut pc = PlasticConn::from_graph(GraphNetwork::new());
        assert_eq!(pc.neuron_count(), 0);

        // Update weight via plasticity path should be a no-op on empty graph
        let res = PlasticConnectivity::<NeuronId>::apply_plasticity(
            &mut pc, NeuronId::new(0), NeuronId::new(1), 0.1);
        assert!(res.is_ok());
    }
}