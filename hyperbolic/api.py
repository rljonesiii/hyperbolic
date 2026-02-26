import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx

from .data import (
    parse_networkx_graph,
    build_generalized_negative_matrix,
    batch_sample_hard_negatives,
)
from .optim import init_hyperbolic_weights, riemannian_adam_init
from .train import train_step_single_gpu
from .math import lorentz_distance
from .viz import plot_poincare_disk


class HyperbolicEngine:
    def __init__(self, spatial_dim=2, seed=42):
        self.spatial_dim = spatial_dim
        self.key = jax.random.PRNGKey(seed)

        self.master_embs = None
        self.node_to_idx = None
        self.idx_to_node = None
        self._is_fit = False

    def fit(self, G: nx.Graph, epochs=150, batch_size=64, num_negs=10):
        print(
            f"Parsing NetworkX Graph ({len(G.nodes())} nodes, {len(G.edges())} edges)..."
        )
        num_nodes, edges, self.node_to_idx, markov_blankets = parse_networkx_graph(G)
        self.idx_to_node = {idx: node for node, idx in self.node_to_idx.items()}
        self.edges = edges

        is_directed = G.is_directed()

        print("Building Generalized Negative Sampling Matrix...")
        neg_prob_matrix = build_generalized_negative_matrix(
            num_nodes, edges, markov_blankets, is_directed=is_directed
        )

        print("Initializing Hyperbolic Embeddings...")
        self.key, subkey = jax.random.split(self.key)
        self.master_embs = init_hyperbolic_weights(
            subkey, (num_nodes, self.spatial_dim), stddev=1e-3
        )

        opt_state = riemannian_adam_init(self.master_embs)
        m_state = opt_state.m
        v_state = opt_state.v

        # Init HGAT
        self.key, subkeyW, subkeyA = jax.random.split(self.key, 3)
        W = jax.random.normal(subkeyW, (self.spatial_dim, self.spatial_dim)) * 0.1
        a = jax.random.normal(subkeyA, (2 * self.spatial_dim,)) * 0.1

        hgat_params = (W, a)
        hgat_m = {"W": jnp.zeros_like(W), "a": jnp.zeros_like(a)}
        hgat_v = {"W": jnp.zeros_like(W), "a": jnp.zeros_like(a)}

        # Prep positive masks
        max_pos = max([len(b) for b in markov_blankets.values()])
        pos_padded = np.zeros((num_nodes, max_pos), dtype=np.int32)
        pos_mask = np.zeros((num_nodes, max_pos), dtype=np.float32)

        for i in range(num_nodes):
            b = markov_blankets[i]
            if len(b) > 0:
                pos_padded[i, : len(b)] = b
                pos_mask[i, : len(b)] = 1.0
            else:
                pos_padded[i, 0] = i  # fallback
                pos_mask[i, 0] = 1.0

        all_indices = np.arange(num_nodes)
        step_count = 0

        print("Starting Training Loop...")
        for epoch in range(epochs):
            np.random.shuffle(all_indices)
            epoch_loss = 0.0

            for i in range(0, num_nodes, batch_size):
                batch_targets = all_indices[i : i + batch_size]
                current_bs = len(batch_targets)

                self.key, subkey = jax.random.split(self.key)
                batch_negs = batch_sample_hard_negatives(
                    subkey, batch_targets, neg_prob_matrix, num_negs
                )

                batch_pos = pos_padded[batch_targets]
                b_pos_mask = pos_mask[batch_targets]

                batch_indices = {
                    "targets": jnp.array(batch_targets),
                    "positives": jnp.array(batch_pos),
                    "negatives": jnp.array(batch_negs),
                    "pos_mask": jnp.array(b_pos_mask),
                }

                step_count += 1
                (
                    self.master_embs,
                    m_state,
                    v_state,
                    hgat_params,
                    hgat_m,
                    hgat_v,
                    loss,
                ) = train_step_single_gpu(
                    self.master_embs,
                    m_state,
                    v_state,
                    step_count,
                    hgat_params,
                    hgat_m,
                    hgat_v,
                    batch_indices,
                )
                epoch_loss += loss * current_bs

            if (epoch + 1) % 25 == 0:
                print(
                    f"   [Epoch {epoch + 1}/{epochs}] Loss: {epoch_loss / num_nodes:.4f}"
                )

        # Persist back to NetworkX
        embeddings = np.array(self.master_embs)
        for node in G.nodes():
            idx = self.node_to_idx[node]
            G.nodes[node]["hyperbolic_embedding"] = embeddings[idx]

        self._is_fit = True
        print("Training Complete. Embeddings synced back to NetworkX graph attributes.")
        return G

    def get_embeddings(self):
        if not self._is_fit:
            raise ValueError("Must call fit() before retrieving embeddings.")
        return {
            node: np.array(self.master_embs[idx])
            for node, idx in self.node_to_idx.items()
        }

    def compute_dissimilarity(self, node_a, node_b):
        if not self._is_fit:
            raise ValueError("Must call fit() first.")
        if node_a not in self.node_to_idx or node_b not in self.node_to_idx:
            raise ValueError("Nodes not found in trained network.")

        idx_a = self.node_to_idx[node_a]
        idx_b = self.node_to_idx[node_b]

        emb_a = self.master_embs[idx_a]
        emb_b = self.master_embs[idx_b]

        dist = lorentz_distance(emb_a, emb_b)
        return float(dist)

    def find_similar_nodes(self, target_node, top_k=5):
        if not self._is_fit:
            raise ValueError("Must call fit() first.")

        idx_t = self.node_to_idx[target_node]
        emb_t = self.master_embs[idx_t]

        # We can vmap distance
        distance_fn_vmapped = jax.vmap(lambda emb: lorentz_distance(emb_t, emb))
        distances = distance_fn_vmapped(self.master_embs)

        # Sort indices
        sorted_indices = np.argsort(distances)

        results = []
        for rank, idx in enumerate(sorted_indices):
            node = self.idx_to_node[idx]
            if node != target_node:
                results.append({"node": node, "distance": float(distances[idx])})
            if len(results) >= top_k:
                break

        return results

    def visualize(
        self,
        output_path="poincare_viz.png",
        node_labels=None,
        show_edges=True,
        annotations=None,
    ):
        if not self._is_fit:
            raise ValueError("Must call fit() first.")

        edges_to_plot = self.edges if show_edges else None

        formatted_annotations = None
        if annotations is not None:
            if isinstance(annotations, list):
                formatted_annotations = {
                    self.node_to_idx[n]: n for n in annotations if n in self.node_to_idx
                }
            elif isinstance(annotations, dict):
                formatted_annotations = {
                    self.node_to_idx[n]: txt
                    for n, txt in annotations.items()
                    if n in self.node_to_idx
                }

        plot_poincare_disk(
            self.master_embs,
            save_path=output_path,
            node_labels=node_labels,
            edges=edges_to_plot,
            annotations=formatted_annotations,
        )
