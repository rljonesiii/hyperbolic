import jax
import jax.numpy as jnp
import numpy as np
from collections import defaultdict


def generate_mock_forest(num_trees=5, branching_factor=3, depth=4):
    """
    Generates a mock knowledge graph consisting of a forest of trees.
    Returns:
    - num_nodes: Total number of nodes
    - edges: List of tuples (parent, child)
    - node_depth: List mapping node_id -> depth
    - siblings: Dict mapping node_id -> list of sibling node_ids
    """
    edges = []
    node_depth = {}
    siblings = defaultdict(list)
    parents = {}

    current_node = 0

    for _ in range(num_trees):
        # Generate a tree using BFS to easily track depth
        queue = [(current_node, 0)]  # (node_id, current_depth)
        node_depth[current_node] = 0
        parents[current_node] = -1
        current_node += 1

        while queue:
            node, d = queue.pop(0)

            if d < depth:
                children = list(range(current_node, current_node + branching_factor))
                for child in children:
                    edges.append((node, child))
                    node_depth[child] = d + 1
                    parents[child] = node
                    queue.append((child, d + 1))

                # Register siblings
                for child in children:
                    siblings[child] = [c for c in children if c != child]

                current_node += branching_factor

    num_nodes = current_node

    return num_nodes, edges, node_depth, siblings, parents


def construct_markov_blankets(num_nodes, edges, siblings, parents):
    """
    Constructs the Markov blanket for each node.
    Markov Blanket = Parent + Children + Siblings
    Returns a dictionary mapping node_id -> list of node_ids in its Markov blanket.
    """
    adj_children = defaultdict(list)
    for p, c in edges:
        adj_children[p].append(c)

    markov_blankets = {}

    for i in range(num_nodes):
        blanket = set()
        # 1. Parent
        if i in parents and parents[i] != -1:
            blanket.add(parents[i])
        # 2. Children
        for c in adj_children[i]:
            blanket.add(c)
        # 3. Siblings
        if i in siblings:
            for s in siblings[i]:
                blanket.add(s)

        markov_blankets[i] = list(blanket)

    return markov_blankets


def build_negative_probability_matrix(num_nodes, node_depth, siblings, markov_blankets):
    """
    Builds a structured sampling matrix heavily weighting hard negatives:
    - Cousins (depth-matched nodes that are NOT in the true Markov blanket)
    - Other nodes at the exact same depth
    """
    probs = (
        np.ones((num_nodes, num_nodes)) * 0.1
    )  # Base probability for random negatives

    for i in range(num_nodes):
        # Zeros for self and actual Markov blanket (positives)
        probs[i, i] = 0.0
        for pos in markov_blankets[i]:
            probs[i, pos] = 0.0

        # Hard Negative: Depth matched
        d = node_depth[i]
        depth_matched = [
            n
            for n, d_n in node_depth.items()
            if d_n == d and n != i and n not in markov_blankets[i]
        ]

        for dn in depth_matched:
            probs[i, dn] = 5.0  # High probability

    # Normalize probabilities
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return jnp.array(probs)


def batch_sample_hard_negatives(key, target_indices, prob_matrix, num_samples=10):
    """
    Sample hard negatives using jax.random.choice based on probability matrix.
    """
    target_probs = prob_matrix[target_indices]
    keys = jax.random.split(key, target_indices.shape[0])

    def sample_single(k, probs):
        return jax.random.choice(
            k, probs.shape[0], shape=(num_samples,), p=probs, replace=False
        )

    sampled_negative_indices = jax.vmap(sample_single)(keys, target_probs)
    return sampled_negative_indices
