import jax
import jax.numpy as jnp
import pytest

from hyperbolic.math import (
    lorentz_exponential_map_origin,
    minkowski_inner_product,
    lorentz_distance,
)


def test_lorentz_exponential_map_origin():
    """
    Verifies that the exponential map successfully translates a flat
    tangent vector to the correct point on the Lorentz manifold.
    Particularly checks that the zero-vector maps to the origin pole.
    """
    # Test zero-vector (should map to the base origin (1, 0, 0))
    zero_vec = jnp.array([0.0, 0.0, 0.0])
    origin_point = lorentz_exponential_map_origin(zero_vec)

    # Assert origin pole properties
    assert jnp.allclose(origin_point, jnp.array([1.0, 0.0, 0.0])), (
        f"Zero vector mapped to: {origin_point}"
    )

    # Test valid non-zero vector mapping
    tangent_vec = jnp.array([0.0, 1.0, 2.0])
    point = lorentz_exponential_map_origin(tangent_vec)

    # Validate the resulting point lies on the manifold: <x, x>_L = -1
    inner_prod = minkowski_inner_product(point, point)
    assert jnp.allclose(inner_prod, -1.0, atol=1e-5), (
        f"Resulting point off-manifold: <x, x>_L = {inner_prod}"
    )


def test_minkowski_inner_product():
    """
    Verifies the Minkowski inner product `<u,v>_L = -u_0v_0 + u_1v_1 + ...`
    """
    u = jnp.array([1.0, 2.0, 3.0])
    v = jnp.array([0.5, 1.0, -1.0])

    # Expected: - (1.0 * 0.5) + (2.0 * 1.0) + (3.0 * -1.0) = -0.5 + 2.0 - 3.0 = -1.5
    inner_prod = minkowski_inner_product(u, v)
    assert jnp.allclose(inner_prod, -1.5)


def test_lorentz_distance():
    """
    Verifies the Riemannian distance calculation on the Lorentz manifold.
    """
    # Let's map two tangent vectors to the manifold
    u_tangent = jnp.array([0.0, 0.5, 0.0])
    v_tangent = jnp.array([0.0, 0.0, -0.5])

    u = lorentz_exponential_map_origin(u_tangent)
    v = lorentz_exponential_map_origin(v_tangent)

    # Distance from node to itself is 0
    dist_self = lorentz_distance(u, u)
    assert jnp.allclose(dist_self, 0.0, atol=1e-5)

    # Distance is symmetric
    dist_uv = lorentz_distance(u, v)
    dist_vu = lorentz_distance(v, u)
    assert jnp.allclose(dist_uv, dist_vu)

    # Distance should be positive
    assert dist_uv > 0.0
