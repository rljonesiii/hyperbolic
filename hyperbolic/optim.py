import jax
import jax.numpy as jnp
from typing import NamedTuple

from hyperbolic.math import (
    lorentz_exponential_map_origin,
    project_to_tangent_space,
    parallel_transport,
    minkowski_inner_product,
    lorentz_exponential_map,
)


def init_hyperbolic_weights(key, shape, stddev=1e-3):
    """
    Initializes weights on the Lorentz manifold using Tangent Space Initialization.
    Shape should be (num_nodes, spatial_dim).
    """
    # 1. Sample the spatial dimensions using standard normal
    spatial_weights = stddev * jax.random.normal(key, shape)

    # 2. Prepend the time-like coordinate with zeros to place it in T_o H^n
    zeros = jnp.zeros((shape[0], 1))
    tangent_vectors = jnp.concatenate([zeros, spatial_weights], axis=-1)

    # 3. Project the tangent vectors onto the Lorentz manifold using the origin exp map
    manifold_weights = lorentz_exponential_map_origin(tangent_vectors)

    return manifold_weights


class RiemannianAdamState(NamedTuple):
    m: jax.Array
    v: jax.Array
    count: int
    prev_params: jax.Array


def riemannian_adam_init(params):
    """Initializes the state for the Riemannian Adam optimizer."""
    return RiemannianAdamState(
        m=jnp.zeros_like(params),
        v=jnp.zeros((params.shape[0],)),
        count=0,
        prev_params=params,
    )


def riemannian_adam_step(
    params,
    euclidean_grads,
    state: RiemannianAdamState,
    learning_rate=1e-2,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    """
    Performs one step of Riemannian Adam.
    """
    count = state.count + 1

    # 1. Project Euclidean grad to Riemannian grad on the Tangent Space
    riemannian_grad = project_to_tangent_space(params, euclidean_grads)

    # 2. Parallel Transport the old momentum to the new tangent space
    # (Since the parameters moved in the last step, from prev_params to params)
    m_transported = parallel_transport(state.prev_params, params, state.m)

    # 3. Update biased first moment estimate (in the tangent space)
    m_new = beta1 * m_transported + (1 - beta1) * riemannian_grad

    # 4. Update biased second raw moment estimate
    # (using the Minkowski inner product norm of the gradient)
    grad_norm_sq = jnp.maximum(
        minkowski_inner_product(riemannian_grad, riemannian_grad), 0.0
    )
    v_new = beta2 * state.v + (1 - beta2) * grad_norm_sq

    # 5. Compute the tangent step vector (with bias corrections)
    m_hat = m_new / (1 - beta1**count)
    v_hat = v_new / (1 - beta2**count)
    tangent_step = -learning_rate * (m_hat / (jnp.sqrt(v_hat)[..., None] + epsilon))

    # 6. Apply the Exponential Map to update the parameters
    new_params = lorentz_exponential_map(params, tangent_step)

    # 7. Create new state
    new_state = RiemannianAdamState(
        m=m_new, v=v_new, count=count, prev_params=new_params
    )

    return new_params, new_state
