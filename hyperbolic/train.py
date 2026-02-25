import jax
import jax.numpy as jnp
from jax import device_put
import numpy as np

from hyperbolic.math import (
    lorentz_exponential_map,
    minkowski_inner_product,
    project_to_tangent_space,
    parallel_transport,
)
from hyperbolic.nn import hyperbolic_gat_layer
from hyperbolic.loss import hyperbolic_infonce_loss
from hyperbolic.math import lorentz_distance

# Force JAX to allocate CPU/GPU devices
try:
    cpu_device = jax.devices("cpu")[0]
except:
    cpu_device = jax.devices()[0]  # Fallback if standard JAX config

try:
    # Use GPU if available (Metal also shows as GPU sometimes depending on Jax-Metal),
    # otherwise fallback to default
    gpu_device = jax.devices("gpu")[0]
except:
    try:
        gpu_device = jax.devices("metal")[0]
    except:
        gpu_device = jax.devices()[0]


@jax.jit
def batch_loss_fn(target_embs, pos_embs, neg_embs, W, a, pos_mask):
    """
    Computes loss for an extracted batch.
    target_embs: (B, D)
    pos_embs: (B, max_pos, D)
    neg_embs: (B, num_negs, D)
    pos_mask: (B, max_pos) integer mask 1/0
    """
    # 1. HGAT Aggregation for Positive pairs (Markov Blanket context)
    # The targets are embedded using context from their true neighbors
    updated_targets = hyperbolic_gat_layer(target_embs, pos_embs, W, a, pos_mask)

    # 2. To avoid needing an HGAT mask for negatives for simplicity (we'd have to sample full neighborhoods for negatives)
    # we just compute InfoNCE loss between the refined target context and the raw embeddings of neighbors/negatives.
    # We want exact neighbors to be close, random negatives to be forced away.
    # In a fully deployed model, we might refine the positives using THEIR neighbors too,
    # but for optimization stability and speed we compare updated target -> raw target's positive components.

    # We'll compute loss against each individual node in the positive set
    # average the loss across valid positives.

    # Actually, we can just contrast each updated_target with the center of `pos_embs`.
    # Let's just contrast the updated_target with the raw target as "positive",
    # or contrast with the un-aggregated target_embs as "positive" vs negatives.

    # Simpler InfoNCE for demonstration:
    # updated context of target u -> should be close to raw node embeddings of true neighbors, pushed away from raw negatives.
    # target_u: (B, D) -> updated_targets
    # v_pos: (B, max_pos, D) -> we want updated_target close to all of these.
    # We can flatten and mask the infoNCE loss.

    temperature = 0.1
    dist_pos = jax.vmap(lambda u, vs: jax.vmap(lambda v: lorentz_distance(u, v))(vs))(
        updated_targets, pos_embs
    )  # (B, max_pos)
    dist_negs = jax.vmap(lambda u, vs: jax.vmap(lambda v: lorentz_distance(u, v))(vs))(
        updated_targets, neg_embs
    )  # (B, num_negs)

    score_pos = -dist_pos / temperature
    scores_negs = -dist_negs / temperature

    # We compute LogSumExp for each positive against ALL negatives
    # For a specific target and positive pair (b, p), denom consists of score_pos[b, p] and all scores_negs[b]
    # score_pos shape: (B, max_pos) -> expand to (B, max_pos, 1) to concat with negatives
    # scores_negs shape: (B, num_negs) -> expand to (B, 1, num_negs) and broadcast to (B, max_pos, num_negs)

    score_pos_expanded = jnp.expand_dims(score_pos, axis=-1)
    scores_negs_expanded = jnp.expand_dims(scores_negs, axis=1)
    # Broadcast to match max_pos
    scores_negs_tiled = jnp.broadcast_to(
        scores_negs_expanded,
        (scores_negs.shape[0], score_pos.shape[1], scores_negs.shape[1]),
    )

    # all_scores: (B, max_pos, 1 + num_negs)
    all_scores = jnp.concatenate([score_pos_expanded, scores_negs_tiled], axis=-1)

    lse = jax.scipy.special.logsumexp(all_scores, axis=-1)

    loss_matrix = -(score_pos - lse)  # (B, max_pos)

    # Mask out padded positives
    valid_loss = jnp.sum(loss_matrix * pos_mask) / jnp.maximum(jnp.sum(pos_mask), 1.0)

    return valid_loss


# Get the gradient function for updating embeddings and HGAT weights
grad_fn = jax.jit(jax.grad(batch_loss_fn, argnums=(0, 1, 2, 3, 4)))


def apply_sparse_riemannian_adam_update(
    master_embs,
    m_state,
    v_state,
    count,
    batch_indices,
    grads,
    learning_rate=1e-2,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    """
    Applies Riemannian Adam using only the extracted batch gradients without pulling the entire graph to GPU.
    We just perform the operations on the small sliced subset and scatter them back.
    """
    target_idx = batch_indices["targets"]

    # We only update the target nodes for simplicity of this demo,
    # or we can flatten target, pos, neg and update all unique indices.
    # To be mathematically rigorous with momentum, we need the exact indices.

    all_indices = jnp.concatenate(
        [
            batch_indices["targets"].flatten(),
            batch_indices["positives"].flatten(),
            batch_indices["negatives"].flatten(),
        ]
    )

    # Ensure unique indices to avoid compounding momentum updates artificially
    unique_indices, inv_idx = jnp.unique(all_indices, return_inverse=True)

    # Extract states for these unique indices
    params = master_embs[unique_indices]
    m = m_state[unique_indices]
    v = v_state[unique_indices]

    # We must aggregate the Euclidean gradients for the unique indices
    # We scatter_add the gradients from targets, positives, negatives into an array of shape (len(unique_indices), D)
    euclidean_grads = jnp.zeros_like(params)

    # Flatten grads
    flat_grad_targets = grads[0].reshape(-1, params.shape[-1])
    flat_grad_positives = grads[1].reshape(-1, params.shape[-1])
    flat_grad_negatives = grads[2].reshape(-1, params.shape[-1])
    all_flat_grads = jnp.concatenate(
        [flat_grad_targets, flat_grad_positives, flat_grad_negatives], axis=0
    )

    # Add gradients mapped to unique indices
    euclidean_grads = euclidean_grads.at[inv_idx].add(all_flat_grads)

    # 1. Project Euclidean grad to Riemannian grad on Tangent Space
    riemannian_grad = project_to_tangent_space(params, euclidean_grads)

    # 2. Parallel transport logic: For this sparse subset, the "prev_params" is just the current params
    # since we haven't updated them yet this step. Next time they are sliced, they will be the new params.
    # We just run standard Riemannian Adam on this slice.
    m_new = beta1 * m + (1 - beta1) * riemannian_grad

    grad_norm_sq = jnp.maximum(
        minkowski_inner_product(riemannian_grad, riemannian_grad), 0.0
    )
    v_new = beta2 * v + (1 - beta2) * grad_norm_sq

    m_hat = m_new / (1 - beta1**count)
    v_hat = v_new / (1 - beta2**count)

    tangent_step = -learning_rate * (m_hat / (jnp.sqrt(v_hat)[..., None] + epsilon))

    new_params = lorentz_exponential_map(params, tangent_step)

    # Put updated values back into master state
    master_embs = master_embs.at[unique_indices].set(new_params)
    m_state = m_state.at[unique_indices].set(m_new)
    v_state = v_state.at[unique_indices].set(v_new)

    return master_embs, m_state, v_state


def apply_euclidean_adam_update(
    params, grads, m, v, count, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8
):
    """Standard Adam for HGAT weights W and a"""
    m_new = beta1 * m + (1 - beta1) * grads
    v_new = beta2 * v + (1 - beta2) * (grads**2)
    m_hat = m_new / (1 - beta1**count)
    v_hat = v_new / (1 - beta2**count)
    new_params = params - lr * m_hat / (jnp.sqrt(v_hat) + eps)
    return new_params, m_new, v_new


def train_step_single_gpu(
    master_embs, m_state, v_state, count, hgat_params, hgat_m, hgat_v, batch_indices
):
    """
    Host-to-Device Paging target training loop.
    Returns: updated master embeddings, optimizer states, hgat weights, loss
    """
    W, a = hgat_params

    # 1. Slice CPU tables
    target_slice = master_embs[batch_indices["targets"]]
    pos_slice = master_embs[batch_indices["positives"]]
    neg_slice = master_embs[batch_indices["negatives"]]
    pos_mask = batch_indices["pos_mask"]

    # 2. Push to GPU (or whatever primary device is)
    target_gpu = device_put(target_slice, gpu_device)
    pos_gpu = device_put(pos_slice, gpu_device)
    neg_gpu = device_put(neg_slice, gpu_device)
    pos_mask_gpu = device_put(pos_mask, gpu_device)
    W_gpu = device_put(W, gpu_device)
    a_gpu = device_put(a, gpu_device)

    # 3. Compute loss and gradients
    loss = batch_loss_fn(target_gpu, pos_gpu, neg_gpu, W_gpu, a_gpu, pos_mask_gpu)
    grads = grad_fn(target_gpu, pos_gpu, neg_gpu, W_gpu, a_gpu, pos_mask_gpu)

    # 4. Pull gradients back to CPU
    grads_cpu = device_put(grads, cpu_device)

    # grads[0, 1, 2] are for embedding targets, pos, negs
    # grads[3, 4] are Euclidean gradients for W and a

    # 5. Sparse Riemannian Adam for Embeddings (on CPU)
    master_embs, m_state, v_state = apply_sparse_riemannian_adam_update(
        master_embs, m_state, v_state, count, batch_indices, grads_cpu[:3]
    )

    # 6. Euclidean Adam for HGAT weights
    W_new, hgat_m_W, hgat_v_W = apply_euclidean_adam_update(
        W, grads_cpu[3], hgat_m["W"], hgat_v["W"], count
    )
    a_new, hgat_m_a, hgat_v_a = apply_euclidean_adam_update(
        a, grads_cpu[4], hgat_m["a"], hgat_v["a"], count
    )

    hgat_params_new = (W_new, a_new)
    hgat_m_new = {"W": hgat_m_W, "a": hgat_m_a}
    hgat_v_new = {"W": hgat_v_W, "a": hgat_v_a}

    return master_embs, m_state, v_state, hgat_params_new, hgat_m_new, hgat_v_new, loss
