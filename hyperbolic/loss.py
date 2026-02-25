import jax
import jax.numpy as jnp

from hyperbolic.math import lorentz_distance


def hyperbolic_infonce_loss(u, v_pos, v_negs, temperature=0.1):
    """
    Computes Hyperbolic InfoNCE loss given target u, positive v_pos, and array of negatives v_negs.
    u: Target node embedding (shape: ..., D)
    v_pos: Positive pair embedding (shape: ..., D)
    v_negs: Array of negative pair embeddings (shape: ..., K, D)
    """
    # 1. Calculate distances
    dist_pos = lorentz_distance(u, v_pos)  # Shape: (...)

    # We vmap over the negatives axis (axis -2) assuming u is broadcastable
    # JAX vmap can map over the K dimension. Alternatively we can just expand u.
    # U shape: (..., D) -> (..., 1, D)
    u_expanded = jnp.expand_dims(u, axis=-2)
    dist_negs = lorentz_distance(u_expanded, v_negs)  # Shape: (..., K)

    # 2. Apply temperature scaling and negate (closer = higher score)
    score_pos = -dist_pos / temperature
    scores_negs = -dist_negs / temperature

    # 3. Concatenate all scores for the denominator
    # [score_pos, score_neg_1, score_neg_2, ..., score_neg_K]
    score_pos_expanded = jnp.expand_dims(score_pos, axis=-1)
    all_scores = jnp.concatenate([score_pos_expanded, scores_negs], axis=-1)

    # 4. Compute the log-softmax loss
    # We use logsumexp for numerical stability
    lse = jax.scipy.special.logsumexp(all_scores, axis=-1)
    loss = -(score_pos - lse)

    # Mean across batch
    return jnp.mean(loss)
