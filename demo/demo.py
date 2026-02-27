import sys
import os

# Add parent directory to path to allow importing the hyperbolic package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np

from hyperbolic.data import (
    generate_mock_forest,
    construct_markov_blankets,
    build_negative_probability_matrix,
    batch_sample_hard_negatives,
)
from hyperbolic.optim import init_hyperbolic_weights, riemannian_adam_init
from hyperbolic.train import train_step_single_gpu
from hyperbolic.viz import plot_poincare_disk


def main():
    print("1. Generating Mock Hierarchical Forest Data...")
    num_nodes, edges, node_depth, siblings, parents = generate_mock_forest(
        num_trees=10, branching_factor=3, depth=3
    )
    print(f"   -> Graph generated with {num_nodes} nodes and {len(edges)} edges.")

    markov_blankets = construct_markov_blankets(num_nodes, edges, siblings, parents)
    neg_prob_matrix = build_negative_probability_matrix(
        num_nodes, node_depth, siblings, markov_blankets
    )

    print(
        "2. Initializing Hyperbolic Embeddings on the Lorentz Manifold (T_o -> H^n)..."
    )
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)

    # We use spatial_dim=2 so the resulting vectors are 3D (time, x, y).
    # This perfectly maps to our 2D Poincare disk projection for visualization.
    spatial_dim = 2
    master_embs = init_hyperbolic_weights(subkey, (num_nodes, spatial_dim), stddev=1e-3)

    print("   -> Generating Mock Euclidean Features (F_n=5, F_e=3)...")
    F_n, F_e = 5, 3
    key, k1, k2 = jax.random.split(key, 3)
    global_node_features = jax.random.normal(k1, (num_nodes, F_n))

    print("3. Initializing Riemannian Adam and HGAT Weights...")
    opt_state = riemannian_adam_init(master_embs)
    m_state = opt_state.m
    v_state = opt_state.v

    # Initialize HGAT Weights
    # W is (D-1) x (D-1) since it applies to spatial dimensions of tangent space
    key, subkeyW, subkeyA, subkeyWM = jax.random.split(key, 4)
    W = jnp.eye(spatial_dim)

    # a dimension: incorporates spatial dim plus features
    a_dim = 2 * spatial_dim + 2 * F_n + F_e
    a = jax.random.normal(subkeyA, (a_dim,)) * 0.1

    # W_message projects the concatenated [hyperbolic_msg, node_feat, edge_feat] -> ambient Lorentz space
    D = spatial_dim + 1
    msg_dim = D + F_n + F_e
    W_message = jax.random.normal(subkeyWM, (msg_dim, D)) * 0.1

    hgat_params = {"W": W, "a": a, "W_message": W_message}
    hgat_m = {
        "W": jnp.zeros_like(W),
        "a": jnp.zeros_like(a),
        "W_message": jnp.zeros_like(W_message),
    }
    hgat_v = {
        "W": jnp.zeros_like(W),
        "a": jnp.zeros_like(a),
        "W_message": jnp.zeros_like(W_message),
    }

    print("4. Training using Host-to-Device Paging Sparse Updates...")
    num_epochs = 500
    batch_size = 64
    num_negs = 10

    # Pre-pad markov blankets to uniform length for tensor sizing
    max_pos = max([len(b) for b in markov_blankets.values()])
    pos_padded = np.zeros((num_nodes, max_pos), dtype=np.int32)
    pos_mask = np.zeros((num_nodes, max_pos), dtype=np.float32)
    pos_edge_padded = np.array(jax.random.normal(k2, (num_nodes, max_pos, F_e)))

    for i in range(num_nodes):
        b = markov_blankets[i]
        pos_padded[i, : len(b)] = b
        # If blanket is empty (isolated node), we'll just ignore it or self-pad
        if len(b) > 0:
            pos_mask[i, : len(b)] = 1.0
        else:
            pos_padded[i, 0] = i  # fallback
            pos_mask[i, 0] = 1.0

    # For training, we'll iterate through nodes
    all_indices = np.arange(num_nodes)

    # Initial Visualization
    print("   -> Rendering initial random state...")
    plot_poincare_disk(
        master_embs,
        node_depth=node_depth,
        edges=edges,
        save_path="poincare_viz_initial.png",
    )

    step_count = 0
    for epoch in range(num_epochs):
        np.random.shuffle(all_indices)
        epoch_loss = 0.0

        for i in range(0, num_nodes, batch_size):
            batch_targets = all_indices[i : i + batch_size]
            current_bs = len(batch_targets)

            # Sample Negatives
            key, subkey = jax.random.split(key)
            batch_negs = batch_sample_hard_negatives(
                subkey, batch_targets, neg_prob_matrix, num_negs
            )

            # Extract Positives
            batch_pos = pos_padded[batch_targets]
            b_pos_mask = pos_mask[batch_targets]

            batch_indices = {
                "targets": jnp.array(batch_targets),
                "positives": jnp.array(batch_pos),
                "negatives": jnp.array(batch_negs),
                "pos_mask": jnp.array(b_pos_mask),
                "target_features": jnp.array(global_node_features[batch_targets]),
                "pos_features": jnp.array(global_node_features[batch_pos]),
                "pos_edge_features": jnp.array(pos_edge_padded[batch_targets]),
            }

            # Step
            step_count += 1
            master_embs, m_state, v_state, hgat_params, hgat_m, hgat_v, loss = (
                train_step_single_gpu(
                    master_embs,
                    m_state,
                    v_state,
                    step_count,
                    hgat_params,
                    hgat_m,
                    hgat_v,
                    batch_indices,
                )
            )
            epoch_loss += loss * current_bs

        if (epoch + 1) % 25 == 0:
            print(
                f"   [Epoch {epoch + 1}/{num_epochs}] Loss: {epoch_loss / num_nodes:.4f}"
            )

    print("5. Evaluating trained geometry...")
    # Render final
    plot_poincare_disk(
        master_embs,
        node_depth=node_depth,
        edges=edges,
        save_path="poincare_viz_final.png",
    )
    print(
        "   -> Success! Open 'poincare_viz_final.png' to see the hierarchical clusters."
    )


if __name__ == "__main__":
    main()
