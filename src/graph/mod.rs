//! Compressed Sparse Row (CSR) graph for flow routing topology.

/// CSR representation of a directed flow graph.
pub struct CsrFlowGraph {
    pub offsets: Vec<usize>,
    pub neighbors: Vec<usize>,
    pub num_nodes: usize,
}

impl CsrFlowGraph {
    pub fn from_adjacency(adj: &[Vec<usize>]) -> Self {
        let num_nodes = adj.len();
        let total_edges: usize = adj.iter().map(|v| v.len()).sum();
        let mut offsets = Vec::with_capacity(num_nodes + 1);
        let mut neighbors = Vec::with_capacity(total_edges);
        let mut offset = 0;
        for node_neighbors in adj.iter() {
            offsets.push(offset);
            neighbors.extend_from_slice(node_neighbors);
            offset += node_neighbors.len();
        }
        offsets.push(offset);
        Self { offsets, neighbors, num_nodes }
    }

    pub fn from_receivers(receivers: &[usize]) -> Self {
        let num_nodes = receivers.len();
        let mut counts = vec![0usize; num_nodes];
        for &recv in receivers.iter() {
            if recv < num_nodes { counts[recv] += 1; }
        }
        let mut offsets = Vec::with_capacity(num_nodes + 1);
        offsets.push(0);
        for &c in counts.iter() {
            offsets.push(offsets.last().unwrap() + c);
        }
        let total_edges = *offsets.last().unwrap();
        let mut neighbors = vec![0usize; total_edges];
        let mut write_pos = offsets.clone();
        for (donor, &recv) in receivers.iter().enumerate() {
            if recv < num_nodes && recv != donor {
                let pos = write_pos[recv];
                neighbors[pos] = donor;
                write_pos[recv] += 1;
            }
        }
        Self { offsets, neighbors, num_nodes }
    }

    #[inline]
    pub fn neighbors_of(&self, node: usize) -> &[usize] {
        &self.neighbors[self.offsets[node]..self.offsets[node + 1]]
    }

    #[inline]
    pub fn num_edges(&self) -> usize { self.neighbors.len() }

    #[inline]
    pub fn degree(&self, node: usize) -> usize {
        self.offsets[node + 1] - self.offsets[node]
    }
}
