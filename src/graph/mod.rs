//! Compressed Sparse Row (CSR) graph for flow routing topology.
//!
//! Replaces the allocator-abusive Vec<Vec<usize>> donor graph with exactly 2 flat arrays.
//! For a 1024×1024 grid: reduces from ~1M heap allocations + 24MB metadata overhead
//! to exactly 2 allocations with perfect spatial locality for hardware prefetching.

/// CSR representation of a directed flow graph.
///
/// The donors of node `i` are stored in `neighbors[offsets[i]..offsets[i+1]]`.
pub struct CsrFlowGraph {
    /// Length = num_nodes + 1. offsets[i] is the start index in `neighbors` for node i.
    pub offsets: Vec<usize>,
    /// Densely packed donor/receiver IDs. Length = total number of edges.
    pub neighbors: Vec<usize>,
    /// Number of nodes in the graph.
    pub num_nodes: usize,
}

impl CsrFlowGraph {
    /// Build a CSR graph from a per-node adjacency list.
    /// This is called once per flow routing pass — O(V + E) construction.
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

        Self {
            offsets,
            neighbors,
            num_nodes,
        }
    }

    /// Build CSR directly from a flat receiver array (each node has exactly 1 receiver).
    /// This builds the *donor* graph: for each receiver r, collect all nodes that flow to r.
    /// Used for the primary D8/D∞ receiver → donor inversion.
    pub fn from_receivers(receivers: &[usize]) -> Self {
        let num_nodes = receivers.len();

        // Count donors per node
        let mut counts = vec![0usize; num_nodes];
        for &recv in receivers.iter() {
            if recv < num_nodes {
                counts[recv] += 1;
            }
        }

        // Build offsets via prefix sum
        let mut offsets = Vec::with_capacity(num_nodes + 1);
        offsets.push(0);
        for &c in counts.iter() {
            offsets.push(offsets.last().unwrap() + c);
        }
        let total_edges = *offsets.last().unwrap();

        // Fill neighbors
        let mut neighbors = vec![0usize; total_edges];
        let mut write_pos = offsets.clone(); // current write position per node
        for (donor, &recv) in receivers.iter().enumerate() {
            if recv < num_nodes && recv != donor {
                let pos = write_pos[recv];
                neighbors[pos] = donor;
                write_pos[recv] += 1;
            }
        }

        Self {
            offsets,
            neighbors,
            num_nodes,
        }
    }

    /// Get the donors/neighbors of node `i` as a slice.
    #[inline]
    pub fn neighbors_of(&self, node: usize) -> &[usize] {
        let start = self.offsets[node];
        let end = self.offsets[node + 1];
        &self.neighbors[start..end]
    }

    /// Number of edges in the graph.
    #[inline]
    pub fn num_edges(&self) -> usize {
        self.neighbors.len()
    }

    /// Degree (number of donors) of a node.
    #[inline]
    pub fn degree(&self, node: usize) -> usize {
        self.offsets[node + 1] - self.offsets[node]
    }
}
