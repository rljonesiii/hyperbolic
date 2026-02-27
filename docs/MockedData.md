# Contextualized Mock Data Generation

To properly test and evaluate the mathematical integrity of the Feature-Modulated Tangent Aggregation layer, our test scripts must generate and inject dense Euclidean attributes alongside our structural node definitions.

This document details how we mock Euclidean node and edge features within our primary execution scripts.

## The Raw Execution Script (`demo/demo.py`)

In `demo/demo.py`, the network is stressed continuously using high-dimensional random float tensors to ensure the math remains stable on the Lorentz manifold under heavy numerical load.

### 1. Generating Node Features

We define the node feature dimension to be 5 ($F_n=5$). We use a JAX pseudo-random number generator to blanket generate a massive, dense Euclidean tensor of shape `(num_nodes, F_n)` representing the node attributes.

```python
# From demo/demo.py (Lines 45-48)
print("   -> Generating Mock Euclidean Features (F_n=5, F_e=3)...")
F_n, F_e = 5, 3
key, k1, k2 = jax.random.split(key, 3)

# Global Node Features: Shape (400, 5)
global_node_features = jax.random.normal(k1, (num_nodes, F_n))
```

### 2. Generating Edge Features

Edges do not exist equally across all nodes, so they must be mapped cleanly against the generated Markov blankets. We determine the max neighborhood size (`max_pos`) across all blankets and create a padded, dense matrix representing the edge features, specifying $F_e=3$.

```python
# From demo/demo.py (Line 90)
max_pos = max([len(b) for b in markov_blankets.values()])

# Global Padded Edge Features: Shape (400, max_pos, 3)
pos_edge_padded = np.array(jax.random.normal(k2, (num_nodes, max_pos, F_e)))
```

### 3. Paging Host-to-Device Batches

Because passing the entire dense matrix to the GPU could lead to Out Of Memory (OOM) errors in billion-node graphs, the global mock attributes live on the CPU. During batch slicing, the specific rows corresponding to the target nodes and their Markov blanket neighbors are indexed and bundled into the `batch_indices` payload before the layer training step.

```python
# From demo/demo.py (Lines 133-141)
batch_indices = {
    # Structural identity...
    "targets": jnp.array(batch_targets),
    "positives": jnp.array(batch_pos),
    # ...
    # Active Attribute Slices!
    "target_features": jnp.array(global_node_features[batch_targets]),
    "pos_features": jnp.array(global_node_features[batch_pos]),
    "pos_edge_features": jnp.array(pos_edge_padded[batch_targets]),
}
```

## The NetworkX Engine (`demo/api_demo.py`)

Unlike `demo/demo.py` which violently stresses the raw math operations with random tensors, `demo/api_demo.py` demonstrates practical utility over categorical graphs.

In this demonstration, the mock data is intrinsically defined as strings when constructing the base `nx.MultiDiGraph`. For instance, when constructing relationships, specific string labels are given to the NetworkX edges.

```python
# From demo/api_demo.py
G.add_edge("Patient_John", "Symptom_Cough", label="exhibits")
G.add_edge("Disease_Flu", "Symptom_Cough", label="causes")
G.add_edge("Company_TechCorp", "Role_Engineer", label="employs")
```

Under the hood, `hyperbolic/api.py` parses these string attributes (e.g., `exhibits`, `causes`, `employs`) and one-hot encodes them automatically before passing them into the Host-to-Device paging system!
