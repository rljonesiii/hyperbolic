import jax
import jax.numpy as jnp
import pytest

from hyperbolic.nn import hyperbolic_gat_layer
from hyperbolic.optim import init_hyperbolic_weights


def test_hyperbolic_gat_layer():
    """
    Verifies that the Hyperbolic Graph Attention Layer can run a forward pass
    without producing NaNs or violating the Lorentz manifold constraint.
    """
    key = jax.random.PRNGKey(42)
    key, subkey_x, subkey_n = jax.random.split(key, 3)

    num_nodes = 10
    spatial_dim = 2
    num_neighbors = 3

    # Generate random Lorentz embeddings
    x = init_hyperbolic_weights(subkey_x, (num_nodes, spatial_dim))

    # Generate random neighbor embeddings (num_nodes, num_neighbors, spatial_dim)
    # Using a loop for simplicity to guarantee all are on-manifold
    neighbors = []
    for _ in range(num_neighbors):
        key, sub_k = jax.random.split(key)
        neighbors.append(init_hyperbolic_weights(sub_k, (num_nodes, spatial_dim)))

    # Stack along the neighbor dimension: (num_nodes, num_neighbors, spatial_dim+1)
    neighbors = jnp.stack(neighbors, axis=1)

    # Initialize W to Identity to prevent rank collapse, and a to random noise
    key, subkeyA = jax.random.split(key)
    W = jnp.eye(spatial_dim)
    a = jax.random.normal(subkeyA, (2 * spatial_dim,)) * 0.1

    # Run Forward Pass
    updated_x = hyperbolic_gat_layer(x, neighbors, W, a)

    # Assert shape is maintained
    assert updated_x.shape == x.shape, f"Shape mismatch: {updated_x.shape} != {x.shape}"

    # Assert no NaNs during geometry computation
    assert not jnp.isnan(updated_x).any(), "HGAT forward pass produced NaNs."

    # Assert outputs are on the Lorentz manifold: <x, x>_L = -1
    from hyperbolic.math import minkowski_inner_product

    inner_prods = minkowski_inner_product(updated_x, updated_x)

    # Check that all inner products are close to -1.0
    assert jnp.allclose(inner_prods, -1.0, atol=1e-5), (
        "HGAT output points fell off the Lorentz manifold."
    )
