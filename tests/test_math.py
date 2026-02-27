import jax.numpy as jnp
import pytest

from hyperbolic.math import (
    lorentz_exponential_map_origin,
    minkowski_inner_product,
    lorentz_distance,
    safe_arccosh,
)


@pytest.mark.parametrize(
    "tangent_vec, expected_norm",
    [
        (jnp.array([0.0, 0.0, 0.0]), -1.0),
        (jnp.array([0.0, 1.0, 2.0]), -1.0),
        (jnp.array([0.0, -1.5, 0.5]), -1.0),
        (jnp.array([0.0, 3.0, -3.0]), -1.0),
        (jnp.array([0.0, 1e-5, 1e-5]), -1.0),
    ],
)
def test_lorentz_exponential_map_origin(tangent_vec, expected_norm):
    """
    Verifies that the exponential map successfully translates a flat
    tangent vector to the correct point on the Lorentz manifold.
    Particularly checks that the zero-vector maps to the origin pole.
    """
    point = lorentz_exponential_map_origin(tangent_vec)

    if jnp.allclose(tangent_vec, 0.0):
        # Assert origin pole properties
        assert jnp.allclose(point, jnp.array([1.0, 0.0, 0.0])), (
            f"Zero vector mapped to: {point}"
        )

    # Validate the resulting point lies on the manifold: <x, x>_L = -1
    inner_prod = minkowski_inner_product(point, point)
    assert jnp.allclose(inner_prod, expected_norm, atol=1e-3), (
        f"Resulting point off-manifold: <x, x>_L = {inner_prod}"
    )


@pytest.mark.parametrize(
    "u, v, expected",
    [
        (jnp.array([1.0, 2.0, 3.0]), jnp.array([0.5, 1.0, -1.0]), -1.5),
        (jnp.array([1.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0]), -1.0),
        (jnp.array([2.0, 1.0, 0.0]), jnp.array([3.0, 0.0, 1.0]), -6.0),
    ],
)
def test_minkowski_inner_product(u, v, expected):
    """
    Verifies the Minkowski inner product `<u,v>_L = -u_0v_0 + u_1v_1 + ...`
    """
    inner_prod = minkowski_inner_product(u, v)
    assert jnp.allclose(inner_prod, expected)


@pytest.mark.parametrize(
    "u_tangent, v_tangent",
    [
        (jnp.array([0.0, 0.5, 0.0]), jnp.array([0.0, 0.0, -0.5])),
        (jnp.array([0.0, 1.0, 1.0]), jnp.array([0.0, 1.0, 1.0])),
        (jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 2.0, -1.0])),
        (jnp.array([0.0, -0.5, 0.5]), jnp.array([0.0, 0.5, -0.5])),
    ],
)
def test_lorentz_distance(u_tangent, v_tangent):
    """
    Verifies the Riemannian distance calculation on the Lorentz manifold.
    """
    u = lorentz_exponential_map_origin(u_tangent)
    v = lorentz_exponential_map_origin(v_tangent)

    # Distance from node to itself is 0
    dist_u_u = lorentz_distance(u, u)
    assert jnp.allclose(dist_u_u, 0.0, atol=1e-2)

    dist_v_v = lorentz_distance(v, v)
    assert jnp.allclose(dist_v_v, 0.0, atol=1e-2)

    # Distance is symmetric
    dist_uv = lorentz_distance(u, v)
    dist_vu = lorentz_distance(v, u)
    assert jnp.allclose(dist_uv, dist_vu, atol=1e-2)

    # Distance should be positive
    assert dist_uv >= 0.0


@pytest.mark.parametrize(
    "x, expected",
    [
        (1.0, 0.0),
        (1.0 + 1e-10, 0.0),  # near 1
        (2.0, jnp.arccosh(2.0)),
        (10.0, jnp.arccosh(10.0)),
        (0.5, 0.0),  # Values < 1 should be clipped to 1 by safe_arccosh
    ],
)
def test_safe_arccosh(x, expected):
    """
    Verifies safe_arccosh behaves correctly, especially near x=1.
    """
    val = safe_arccosh(jnp.array([x]))
    assert jnp.allclose(val, jnp.array([expected]), atol=1e-5)
