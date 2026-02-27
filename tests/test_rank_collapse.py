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


def test_rank_collapse_prevention():
    """
    Verifies that the spatial embeddings do not collapse onto a 1D line during training.
    This runs a short training loop and asserts that both singular values of the spatial dimensions
    of the Lorentz embeddings remain comfortably large.
    """
    key = jax.random.PRNGKey(42)

    # 1. Generate Mock Data
    num_nodes, edges, node_depth, siblings, parents = generate_mock_forest(
        num_trees=10, branching_factor=3, depth=3
    )
    markov_blankets = construct_markov_blankets(num_nodes, edges, siblings, parents)
    neg_prob_matrix = build_negative_probability_matrix(
        num_nodes, node_depth, siblings, markov_blankets
    )

    # 2. Init Embeddings
    spatial_dim = 2
    key, subkey = jax.random.split(key)
    master_embs = init_hyperbolic_weights(subkey, (num_nodes, spatial_dim), stddev=1e-3)
    opt_state = riemannian_adam_init(master_embs)
    m_state = opt_state.m
    v_state = opt_state.v

    # 3. Init HGAT Weights (W must be Identity to prevent rank collapse)
    key, subkeyA = jax.random.split(key)
    W = jnp.eye(spatial_dim)
    a = jax.random.normal(subkeyA, (2 * spatial_dim,)) * 0.1

    hgat_params = {"W": W, "a": a}
    hgat_m = {"W": jnp.zeros_like(W), "a": jnp.zeros_like(a)}
    hgat_v = {"W": jnp.zeros_like(W), "a": jnp.zeros_like(a)}

    # 4. Padding Setup
    max_pos = max([len(b) for b in markov_blankets.values()])
    pos_padded = np.zeros((num_nodes, max_pos), dtype=np.int32)
    pos_mask = np.zeros((num_nodes, max_pos), dtype=np.float32)

    for i in range(num_nodes):
        b = markov_blankets[i]
        if len(b) > 0:
            pos_padded[i, : len(b)] = b
            pos_mask[i, : len(b)] = 1.0
        else:
            pos_padded[i, 0] = i
            pos_mask[i, 0] = 1.0

    # 5. Training Loop
    all_indices = np.arange(num_nodes)
    step_count = 0

    # Run for 150 epochs to give it enough time to potentially collapse if buggy
    for epoch in range(150):
        np.random.shuffle(all_indices)
        for i in range(0, num_nodes, 64):
            batch_targets = all_indices[i : i + 64]
            key, subkey = jax.random.split(key)
            batch_negs = batch_sample_hard_negatives(
                subkey, batch_targets, neg_prob_matrix, 10
            )
            batch_indices = {
                "targets": jnp.array(batch_targets),
                "positives": jnp.array(pos_padded[batch_targets]),
                "negatives": jnp.array(batch_negs),
                "pos_mask": jnp.array(pos_mask[batch_targets]),
            }
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

    # 6. Evaluation and Assertion
    # Calculate Singular Values of the spatial dimensions (index 1 and 2)
    u, s, vh = np.linalg.svd(np.array(master_embs[:, 1:]))

    # Assert that both spatial dimensions possess significant variance
    # Before the fix, s[1] would drop to ~0.2
    assert s[0] > 1.0, f"Primary spatial singular value too small: {s[0]}"
    assert s[1] > 1.0, (
        f"Secondary spatial singular value dropped to: {s[1]} (Rank Collapse!)"
    )
