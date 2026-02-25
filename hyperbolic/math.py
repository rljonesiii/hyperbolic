import jax
import jax.numpy as jnp


def minkowski_inner_product(u, v):
    """
    Computes the Minkowski inner product between u and v.
    u and v have shape (..., spatial_dim + 1)
    Time-like coordinate is at index 0
    """
    time_product = -u[..., 0] * v[..., 0]
    space_product = jnp.sum(u[..., 1:] * v[..., 1:], axis=-1)
    return time_product + space_product


@jax.custom_jvp
def safe_arccosh(x):
    """
    Computes arccosh(x) but provides a safe custom gradient when x approaches 1.
    Standard derivative is 1 / sqrt(x^2 - 1). When x=1, this is infinity.
    Here we clip the denominator gradient safely.
    """
    return jnp.arccosh(jnp.maximum(x, 1.0))


@safe_arccosh.defjvp
def safe_arccosh_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents

    # Primal pass
    primal_out = safe_arccosh(x)

    # Tangent pass (Gradient)
    # The true derivative of arccosh(x) is 1 / sqrt(x^2 - 1)
    # To prevent division by zero, we clip the term inside the square root
    denom = jnp.sqrt(jnp.maximum(x**2 - 1.0, 1e-5))
    tangent_out = x_dot / denom

    return primal_out, tangent_out


def lorentz_distance(u, v, eps=1e-5):
    """
    Computes the Riemannian distance on the Lorentz manifold.
    """
    inner_prod = minkowski_inner_product(u, v)
    # The negative inner product is x.
    # Because <u, v>_L <= -1, we have x >= 1.
    val = -inner_prod

    return safe_arccosh(val)


def project_to_tangent_space(x, u):
    """
    Projects a Euclidean vector u to the tangent space of x on the Lorentz manifold.
    x is a point on the manifold.
    u is a vector in ambient space.
    The tangent space condition: <x, v>_L = 0, where v is the projected vector.
    v = u + <x, u>_L * x
    """
    inner_prod = minkowski_inner_product(x, u)
    return u + inner_prod[..., None] * x


def lorentz_exponential_map_origin(v):
    """
    Maps a tangent vector v = (0, x_1, ..., x_n) from the origin's
    tangent space onto the Lorentz manifold.
    """
    # The Minkowski norm of a vector with a 0 time-coord is just its Euclidean norm
    # To prevent NaN in gradients at v=0 (where derivative of sqrt is inf),
    # we add eps inside the sqrt, but only for the gradient pass, similar to safe_arccosh.
    # Alternatively simply clip inside the sqrt:
    sq_norm_v = jnp.sum(v**2, axis=-1, keepdims=True)
    norm_v = jnp.sqrt(jnp.maximum(sq_norm_v, 1e-10))

    # Calculate the exponential map at origin o = (1, 0, ..., 0)
    time_coord = jnp.cosh(norm_v)
    spatial_coords = jnp.sinh(norm_v) * (v[..., 1:] / norm_v)

    # Concatenate the new time coordinate with the projected spatial coordinates
    return jnp.concatenate([time_coord, spatial_coords], axis=-1)


def lorentz_exponential_map(x, v, eps=1e-5):
    """
    Computes the exponential map of tangent vector v at point x.
    Maps v from T_x H^n to H^n.
    y = cosh(|v|_L) * x + sinh(|v|_L) * (v / |v|_L)
    """
    norm_v = jnp.sqrt(jnp.maximum(minkowski_inner_product(v, v), eps))
    norm_v_expanded = norm_v[..., None]

    return jnp.cosh(norm_v_expanded) * x + jnp.sinh(norm_v_expanded) * (
        v / norm_v_expanded
    )


def lorentz_logarithmic_map(x, y, eps=1e-5):
    """
    Computes the logarithmic map of point y from the base point x.
    Maps y from H^n to T_x H^n.
    v = arc_cosh(-<x, y>_L) * (y + <x, y>_L * x) / sqrt(<x, y>_L^2 - 1)
    """
    xy = minkowski_inner_product(x, y)
    val = -xy

    dist = safe_arccosh(val)

    # sqrt(val^2 - 1)
    denom = jnp.sqrt(jnp.maximum(val**2 - 1.0, eps))
    factor = dist / denom

    return factor[..., None] * (y + xy[..., None] * x)


def parallel_transport(x, y, v, eps=1e-5):
    """
    Parallel transports a tangent vector v from T_x H^n to T_y H^n.
    PT_{x->y}(v) = v - <log_x(y), v>_L / d_L(x,y)^2 * (log_x(y) + log_y(x))
    """
    log_xy = lorentz_logarithmic_map(x, y, eps)
    log_yx = lorentz_logarithmic_map(y, x, eps)

    inner_v_log = minkowski_inner_product(v, log_xy)
    dist_sq = jnp.maximum(lorentz_distance(x, y, eps) ** 2, eps)

    return v - (inner_v_log / dist_sq)[..., None] * (log_xy + log_yx)


def lorentz_to_poincare_2d(lorentz_embeddings):
    """
    Projects Lorentz embeddings (time, x, y, ...) to the Poincaré ball.
    lorentz_embeddings shape: (N, D), where index 0 is time-like.
    Stereographic projection: u = x_spatial / (x_0 + 1)
    """
    x_0 = lorentz_embeddings[..., 0]
    x_spatial = lorentz_embeddings[..., 1:]

    return x_spatial / (x_0[..., None] + 1.0)


def poincare_to_lorentz_2d(poincare_embeddings):
    """
    Maps points from the Poincaré ball to the Lorentz hyperboloid.
    poincare_embeddings shape: (N, D-1)
    x_0 = (1 + |u|^2) / (1 - |u|^2)
    x_spatial = 2u / (1 - |u|^2)
    """
    norm_sq = jnp.sum(poincare_embeddings**2, axis=-1)
    denom = 1.0 - norm_sq

    x_0 = (1.0 + norm_sq) / denom
    x_spatial = (2.0 * poincare_embeddings) / denom[..., None]

    return jnp.concatenate([x_0[..., None], x_spatial], axis=-1)
