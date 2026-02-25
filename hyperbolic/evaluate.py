import jax
import jax.numpy as jnp

from hyperbolic.math import lorentz_distance


def compute_mrr(target_embeddings, all_embeddings, true_neighbor_indices):
    """
    Calculates Mean Reciprocal Rank for neighborhood retrieval.
    target_embeddings: (Batch, D)
    all_embeddings: (Total_Nodes, D)
    true_neighbor_indices: (Batch, num_true_neighbors)
    """
    # 1. Compute Lorentzian distance from all targets to ALL nodes in the graph
    distances = jax.vmap(
        lambda target: jax.vmap(lambda node: lorentz_distance(target, node))(
            all_embeddings
        )
    )(target_embeddings)

    # 2. Sort the distances to get the ranking of each node (closest first)
    ranked_indices = jnp.argsort(distances, axis=-1)

    # 3. Find where the true neighbors are in this ranked list
    def get_ranks(ranked_list, true_neighbors):
        # Creates a boolean mask of where the true neighbors appear in the ranking
        matches = jnp.isin(ranked_list, true_neighbors)
        # Find the actual rank positions (1-indexed)
        ranks = jnp.where(matches)[0] + 1
        return ranks

    # vmap get_ranks over the batch
    ranks = jax.vmap(get_ranks)(ranked_indices, true_neighbor_indices)

    # 4. Calculate Reciprocal Ranks and take the Mean
    # Note: ranks can return variable lengths depending on true_neighbor_indices if padded.
    # We assume padding is handled by setting padding to negative index which will not map
    reciprocal_ranks = 1.0 / ranks
    mrr = jnp.mean(reciprocal_ranks)

    return float(mrr)
