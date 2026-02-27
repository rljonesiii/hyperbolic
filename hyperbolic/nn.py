import jax
import jax.numpy as jnp

from hyperbolic.math import (
    lorentz_exponential_map_origin,
    lorentz_logarithmic_map,
    lorentz_exponential_map,
)


def origin():
    """Returns the origin point (pole) of the Lorentz manifold."""
    # Note: We'll construct it dynamically based on the dimension of the embeddings
    pass


def mobius_matvec(W, x):
    """
    Applies a linear transformation W to a hyperbolic point x.
    This is equivalent to mapping x to the origin's tangent space, applying W,
    and mapping back to the manifold.
    W: (D_out-1, D_in-1) standard linear weight matrix for spatial dims
    x: (..., D_in) Lorentzian embeddings
    """
    # 1. Map node to origin's tangent space: h = log_o(x)
    # Origin o = (1, 0, ..., 0)
    D_in = x.shape[-1]
    o = jnp.zeros(D_in)
    o = o.at[0].set(1.0)

    # Vector in T_o H^n
    h = lorentz_logarithmic_map(o, x)

    # 2. Apply linear weight matrix: h' = W h
    # Since time-like coordinate of h is 0, we can just apply W to the spatial part.
    h_spatial = h[..., 1:]
    h_spatial_transformed = jnp.dot(h_spatial, W.T)

    # Zero for the time-like coordinate of the new vector
    zeros = jnp.zeros(x.shape[:-1] + (1,))
    h_prime = jnp.concatenate([zeros, h_spatial_transformed], axis=-1)

    # 3. Project back to the manifold: x' = exp_o(h')
    return lorentz_exponential_map_origin(h_prime)


def compute_attention_scores(
    a, x, y, node_features_x=None, node_features_y=None, edge_features=None
):
    """
    Computes raw attention scores between node x and neighbor y, optionally modulated by features.
    a: (...,) attention weight vector. Size depends on whether features are provided.
    x: (..., D) target nodes
    y: (..., K, D) neighbors
    node_features_x: (..., F_n) optional node features for targets
    node_features_y: (..., K, F_n) optional node features for neighbors
    edge_features: (..., K, F_e) optional edge features
    Returns: (..., K) raw attention scores
    """
    D = x.shape[-1]
    o = jnp.zeros(D)
    o = o.at[0].set(1.0)

    # 1. Map to Origin's Tangent Space
    h_x = lorentz_logarithmic_map(o, x)[..., 1:]  # (..., D-1)
    h_y = lorentz_logarithmic_map(o, y)[..., 1:]  # (..., K, D-1)

    # h_x_expanded: (..., 1, D-1)
    h_x_expanded = jnp.expand_dims(h_x, axis=-2)
    h_x_tiled = jnp.broadcast_to(h_x_expanded, h_y.shape)

    # Base spatial coordinates
    components = [h_x_tiled, h_y]

    # Optional feature concatenation for attention routing
    if node_features_x is not None and node_features_y is not None:
        nx_expanded = jnp.expand_dims(node_features_x, axis=-2)
        nx_tiled = jnp.broadcast_to(nx_expanded, node_features_y.shape)
        components.extend([nx_tiled, node_features_y])

    if edge_features is not None:
        components.append(edge_features)
    # 2. Concatenation and Scoring
    # concat: (..., K, total_dim)
    concat_features = jnp.concatenate(components, axis=-1)

    # score: (..., K)
    raw_scores = jax.nn.leaky_relu(jnp.dot(concat_features, a))
    return raw_scores


def hyperbolic_gat_layer(
    x,
    neighbors,
    W,
    a,
    mask=None,
    node_features_x=None,
    node_features_y=None,
    edge_features=None,
    W_message=None,
):
    """
    A single Hyperbolic Graph Attention mechanism layer for Markov Blankets.
    x: (N, D) Target nodes
    neighbors: (N, K, D) Neighbors in the Markov blanket
    W: (D_out-1, D_in-1) Linear transformation weights
    a: Attention weights
    mask: (N, K) Optional mask for variable-sized neighborhoods (1 for valid, 0 for padded)
    node_features_x: Target node features
    node_features_y: Neighbor node features
    edge_features: Edge features connecting target to neighbors
    W_message: Linear weights to project enriched messages back to D_out-1
    """
    # 1. Feature Transformation
    # x_transformed: (N, D_out)
    x_transformed = mobius_matvec(W, x)
    # neighbors_transformed: (N, K, D_out)
    neighbors_transformed = mobius_matvec(W, neighbors)

    # 2. Compute Feature-Modulated Attention
    raw_scores = compute_attention_scores(
        a,
        x_transformed,
        neighbors_transformed,
        node_features_x,
        node_features_y,
        edge_features,
    )

    if mask is not None:
        # Mask out padded neighbors by setting raw scores to -inf
        raw_scores = jnp.where(mask > 0, raw_scores, -1e9)

    # Softmax normalization over the K neighbors
    attention_weights = jax.nn.softmax(raw_scores, axis=-1)  # (N, K)

    # 3. Tangent-Space Aggregation
    # Lift neighbors to the target's tangent space
    # v_i: (..., K, D_out)
    # Target node: broadcast to (..., K, D_out)
    x_transformed_expanded = jnp.expand_dims(x_transformed, axis=-2)
    x_transformed_tiled = jnp.broadcast_to(
        x_transformed_expanded, neighbors_transformed.shape
    )
    v_i = lorentz_logarithmic_map(x_transformed_tiled, neighbors_transformed)

    # Enrichen the message with features if provided
    if W_message is not None:
        message_components = [v_i]
        if node_features_y is not None:
            message_components.append(node_features_y)
        if edge_features is not None:
            message_components.append(edge_features)

        # message_uv: (..., K, total_msg_dim)
        enriched_msg = jnp.concatenate(message_components, axis=-1)
        # Apply projection
        v_i = jnp.dot(enriched_msg, W_message)
        # We must forcibly project this Euclidean vector back onto the Lorentz Tangent Space of x
        from hyperbolic.math import project_to_tangent_space

        v_i = project_to_tangent_space(x_transformed_tiled, v_i)

    # Multiply by attention weights
    # attention_weights: (..., K, 1) to broadcast with v_i
    v_agg = jnp.sum(
        jnp.expand_dims(attention_weights, axis=-1) * v_i, axis=-2
    )  # (..., D_out)

    # 4. Retraction to the Lorentz Manifold
    updated_x = lorentz_exponential_map(x_transformed, v_agg)

    return updated_x
