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


def parse_networkx_graph(G):
    """
    Parses a general NetworkX graph (like MultiDiGraph or MultiGraph).
    Returns basic graph structures mapped to integer indices.
    """

    # Map original node IDs to clean integers 0...N-1
    node_list = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}

    num_nodes = len(node_list)

    # Extract edges
    edges = []
    for u, v in G.edges():
        edges.append((node_to_idx[u], node_to_idx[v]))

    # Generalized Markov Blanket (Parents + Children + Siblings)
    # Since arbitrary graphs might not be strictly directed trees, we approximate
    # by taking immediate 1-hot neighbors, and then checking for shared parents via predecessors.
    markov_blankets = {}

    is_directed = G.is_directed()

    for node in node_list:
        idx = node_to_idx[node]
        blanket = set()

        # Immediate neighbors (Parents/Children in directed, adjacent in undirected)
        if is_directed:
            for pred in G.predecessors(node):
                blanket.add(node_to_idx[pred])
            for succ in G.successors(node):
                blanket.add(node_to_idx[succ])
        else:
            for neighbor in G.neighbors(node):
                blanket.add(node_to_idx[neighbor])

        # "Siblings" (Nodes sharing a parent in directed graphs)
        if is_directed:
            for pred in G.predecessors(node):
                for sibling in G.successors(pred):
                    if sibling != node:
                        blanket.add(node_to_idx[sibling])

        markov_blankets[idx] = list(blanket)

    return num_nodes, edges, node_to_idx, markov_blankets


def encode_graph_features(G, node_to_idx):
    """
    Extracts node and edge features from a NetworkX graph.
    Encodes categorical features as one-hot arrays and keeps numericals.
    Returns:
      node_feats: (num_nodes, F_n) array or None
      edge_feats_dict: dict of (u, v) -> array of shape (F_e,) or None
    """
    node_str_cats = defaultdict(set)
    for n in G.nodes():
        for k, v in G.nodes[n].items():
            if isinstance(v, str):
                node_str_cats[k].add(v)

    node_str_cats = {k: sorted(list(v)) for k, v in node_str_cats.items()}
    global_node_features = []
    num_nodes = len(node_to_idx)

    for n in G.nodes():
        idx = node_to_idx[n]
        vec = []
        for k, v in G.nodes[n].items():
            if k == "hyperbolic_embedding":
                continue
            if isinstance(v, str) and k in node_str_cats:
                onehot = [0.0] * len(node_str_cats[k])
                onehot[node_str_cats[k].index(v)] = 1.0
                vec.extend(onehot)
            elif isinstance(v, (int, float)):
                vec.append(float(v))
            elif isinstance(v, (list, tuple, np.ndarray)):
                vec.extend([float(x) for x in v])
        global_node_features.append((idx, vec))

    global_node_features.sort(key=lambda x: x[0])
    max_len = (
        max([len(v) for _, v in global_node_features]) if global_node_features else 0
    )
    if max_len == 0:
        node_feats = None
    else:
        node_feats = np.zeros((num_nodes, max_len), dtype=np.float32)
        for idx, vec in global_node_features:
            node_feats[idx, : len(vec)] = vec

    # Edge features
    edge_str_cats = defaultdict(set)
    is_multigraph = G.is_multigraph()

    edge_data_list = []
    if is_multigraph:
        for u, v, k, d in G.edges(keys=True, data=True):
            edge_data_list.append((u, v, d))
    else:
        for u, v, d in G.edges(data=True):
            edge_data_list.append((u, v, d))

    for u, v, d in edge_data_list:
        for k, val in d.items():
            if isinstance(val, str):
                edge_str_cats[k].add(val)

    edge_str_cats = {k: sorted(list(v)) for k, v in edge_str_cats.items()}

    edge_feats_dict = {}
    max_e_len = 0
    for u, v, d in edge_data_list:
        vec = []
        for k, val in d.items():
            if isinstance(val, str) and k in edge_str_cats:
                onehot = [0.0] * len(edge_str_cats[k])
                onehot[edge_str_cats[k].index(val)] = 1.0
                vec.extend(onehot)
            elif isinstance(val, (int, float)):
                vec.append(float(val))
            elif isinstance(val, (list, tuple, np.ndarray)):
                vec.extend([float(x) for x in val])

        # In a directed graph with bidrectional sharing or multiple edge features
        # we just overwrite with the latest edge if multigraph for simplicity in demo
        idx_u = node_to_idx[u]
        idx_v = node_to_idx[v]
        edge_feats_dict[(idx_u, idx_v)] = vec
        if len(vec) > max_e_len:
            max_e_len = len(vec)

    if max_e_len > 0:
        for k, vec in edge_feats_dict.items():
            padded = np.zeros(max_e_len, dtype=np.float32)
            padded[: len(vec)] = vec
            edge_feats_dict[k] = padded
    else:
        edge_feats_dict = None

    return node_feats, edge_feats_dict


def build_generalized_negative_matrix(
    num_nodes, edges, markov_blankets, is_directed=False
):
    """
    Builds a negative probability matrix without assuming strict tree 'depth'.
    Instead, it uses graph distance. Nodes exactly 2 hops away are considered
    hard negatives (structural cousins) compared to 1-hop (markov blanket).
    """
    import networkx as nx

    G_idx = nx.Graph() if not is_directed else nx.DiGraph()
    G_idx.add_nodes_from(range(num_nodes))
    G_idx.add_edges_from(edges)

    # We want undirected paths for structural similarity distances
    G_undir = G_idx.to_undirected()

    probs = np.ones((num_nodes, num_nodes)) * 0.1

    # Calculate all shortest paths up to length 2
    # This might be slow for massive graphs in a simple loop, but is mathematically correct
    path_lengths = dict(nx.all_pairs_shortest_path_length(G_undir, cutoff=2))

    for i in range(num_nodes):
        probs[i, i] = 0.0
        for pos in markov_blankets[i]:
            probs[i, pos] = 0.0

        # Hard Negative: Exactly 2 hops away structurally and NOT in the markov blanket
        if i in path_lengths:
            for target_node, dist in path_lengths[i].items():
                if dist == 2 and target_node not in markov_blankets[i]:
                    probs[i, target_node] = 5.0

    # Normalize probabilities
    row_sums = np.sum(probs, axis=1, keepdims=True)
    # Avoid division by zero for totally isolated nodes
    row_sums[row_sums == 0] = 1.0
    probs = probs / row_sums

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
