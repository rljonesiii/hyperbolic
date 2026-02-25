# Evaluation

Once our JAX model has finished training and the Riemannian Adam optimizer has beautifully arranged our nodes on the Lorentz manifold, we need to prove that the geometry actually captures our forest's hierarchy and Markov blankets. We evaluate this in two ways: **Quantitatively** (using ranking metrics) and **Qualitatively** (by projecting the high-dimensional manifold down to a 2D visualizable disk).

## Quantitative Evaluation: Mean Reciprocal Rank (MRR)

Because we optimized the network using a contrastive loss based on Lorentzian distance, the ultimate test is a "Link Prediction" or "Neighborhood Retrieval" task.  
If we pick a target node, calculate its distance to *every other node in the entire forest*, and sort them from closest to furthest, where do its true Markov blanket neighbors end up?

* **Rank:** If a true neighbor is the absolute closest node, its rank is 1. If it's the 50th closest, its rank is 50.  
* **Reciprocal Rank (RR):** We take 1 / \\text{Rank}. A rank of 1 gives a score of 1.0. A rank of 50 gives a score of 0.02. This heavily penalizes the model if true neighbors aren't exactly at the top of the list.  
* **Mean Reciprocal Rank (MRR):** We average this score across all true neighbors for all nodes in our test set.

Because hyperbolic space has exponential volume, a well-trained model will often achieve near-perfect MRR (e.g., > 0.95), because there is so much "room" to push unrelated nodes infinitely far away.

## Qualitative Visualization: Projecting to the Poincaré Disk

While the Lorentz manifold is mathematically superior for gradient descent, it is terrible for human visualization. It's an (n+1)-dimensional hyperboloid.  
To actually *see* the branches of our forest and the clustered Markov blankets, we must map the embeddings back to the Poincaré disk. As we discussed earlier, we use stereographic projection to smoothly map the (n+1)-dimensional Lorentz coordinates onto an n-dimensional unit ball.  
If we set our spatial dimension to n=2 during training (meaning our Lorentz vectors are 3D), we can project them perfectly onto a 2D circle for plotting in matplotlib.

## JAX Implementation: Evaluation & Projection

Here is how we execute both the MRR calculation and the visual projection efficiently in JAX:  

```python
import jax  
import jax.numpy as jnp

# --- 1. MRR Calculation ---`

def compute_mrr(target_embeddings, all_embeddings, true_neighbor_indices):
    """
    Calculates Mean Reciprocal Rank for neighborhood retrieval.
    target_embeddings: (Batch, D)
    all_embeddings: (Total_Nodes, D)
    true_neighbor_indices: (Batch, num_true_neighbors)
    """  
    # 1. Compute Lorentzian distance from all targets to ALL nodes in the graph
    # jax.vmap allows us to broadcast the distance function elegantly
    distances = jax.vmap(
        lambda target: jax.vmap(
            lambda node: lorentz_distance(target, node)
        )(all_embeddings)
    )(target_embeddings)
      
    # 2. Sort the distances to get the ranking of each node (closest first)
    # argsort gives us the node IDs ordered by distance
    ranked_indices = jnp.argsort(distances, axis=-1)
      
    # 3. Find where the true neighbors are in this ranked list
    def get_ranks(ranked_list, true_neighbors):
        # Creates a boolean mask of where the true neighbors appear in the ranking
        matches = jnp.isin(ranked_list, true_neighbors)
        # Find the actual rank positions (1-indexed)
        ranks = jnp.where(matches)[0] + 1
        return ranks
          
    ranks = jax.vmap(get_ranks)(ranked_indices, true_neighbor_indices)
      
    # 4. Calculate Reciprocal Ranks and take the Mean
    reciprocal_ranks = 1.0 / ranks
    mrr = jnp.mean(reciprocal_ranks)
      
    return mrr

# --- 2. Poincaré Projection for Plotting ---

def lorentz_to_poincare_2d(lorentz_embeddings):
    """
    Projects 3D Lorentz embeddings (time, x, y) to a 2D Poincaré disk (x, y).
    lorentz_embeddings shape: (N, 3) where index 0 is the time-like coordinate.
    """  
    # Extract coordinates
    x_0 = lorentz_embeddings[:, 0] # Time-like
    x_spatial = lorentz_embeddings[:, 1:] # Space-like (x, y)
      
    # Apply the stereographic projection: x_spatial / (x_0 + 1)
    # The denominator is always >= 2, so it is perfectly stable
    poincare_2d = x_spatial / (x_0[:, None] + 1.0)
      
    return poincare_2d

# Example usage for plotting:
# projected_points = lorentz_to_poincare_2d(trained_lorentz_nodes)
# plt.scatter(projected_points[:, 0], projected_points[:, 1])
```
When we plot those projected\_points, we should see the root of our forest near the center (0,0), with independent trees branching outward toward the rim. The individual Markov blankets will appear as tight, localized clusters along those branches.  

Evaluation is performed in two ways: **Quantitatively** (using ranking metrics) and **Qualitatively** (by projecting the high-dimensional manifold down to a 2D visualizable disk).

## Quantitative Evaluation: Mean Reciprocal Rank (MRR)

Because the network is optimized using a contrastive loss based on Lorentzian distance, the ultimate test is a "Link Prediction" or "Neighborhood Retrieval" task.

If a target node is picked, and its distance is calculated to *every other node in the entire forest*, and the nodes are sorted from closest to furthest, the rank of its true Markov blanket neighbors is determined.

* **Rank:** If a true neighbor is the absolute closest node, its rank is 1\. If it is the 50th closest, its rank is 50\.  
* **Reciprocal Rank (RR):** This is calculated as 1 / Rank. A rank of 1 gives a score of 1.0. A rank of 50 gives a score of 0.02. This heavily penalizes the model if true neighbors are not exactly at the top of the list.  
* **Mean Reciprocal Rank (MRR):** This score is averaged across all true neighbors for all nodes in the test set.

Because hyperbolic space has exponential volume, a well-trained model will often achieve near-perfect MRR (e.g., \> 0.95), because there is so much "room" to push unrelated nodes infinitely far away.

## Qualitative Visualization: Projecting to the Poincaré Disk

While the Lorentz manifold is mathematically superior for gradient descent, it is terrible for human visualization. It is an (n+1)-dimensional hyperboloid.

To actually *see* the branches of the forest and the clustered Markov blankets, the embeddings must be mapped back to the Poincaré disk. Stereographic projection is used to smoothly map the (n+1)-dimensional Lorentz coordinates onto an n-dimensional unit ball.

If the spatial dimension is set to n=2 during training (meaning the Lorentz vectors are 3D), they can be projected perfectly onto a 2D circle for plotting in matplotlib.

## JAX Implementation: Evaluation & Projection

Here is how both the MRR calculation and the visual projection are executed efficiently in JAX:  

```python
import jax    
import jax.numpy as jnp    
    
# 1. MRR Calculation
    
def compute_mrr(target_embeddings, all_embeddings, true_neighbor_indices):    
    """    
    Calculates Mean Reciprocal Rank for neighborhood retrieval.    
    target_embeddings: (Batch, D)    
    all_embeddings: (Total_Nodes, D)    
    true\_neighbor\_indices: (Batch, num\_true\_neighbors)    
    """    
    # 1. Compute Lorentzian distance from all targets to ALL nodes in the graph
    # jax.vmap allows broadcasting the distance function elegantly
    distances = jax.vmap(
        lambda target: jax.vmap(
            lambda node: lorentz_distance(target, node)
        )(all_embeddings)
    )(target_embeddings)
        
    # 2. Sort the distances to get the ranking of each node (closest first)
    # argsort gives the node IDs ordered by distance
    ranked_indices = jnp.argsort(distances, axis=-1)
        
    # 3. Find where the true neighbors are in this ranked list
    def get_ranks(ranked_list, true_neighbors):
        # Creates a boolean mask of where the true neighbors appear in the ranking
        matches = jnp.isin(ranked_list, true_neighbors)
        # Find the actual rank positions (1-indexed)
        ranks = jnp.where(matches)[0] + 1
        return ranks
            
    ranks = jax.vmap(get_ranks)(ranked_indices, true_neighbor_indices)
        
    # 4. Calculate Reciprocal Ranks and take the Mean
    reciprocal_ranks = 1.0 / ranks
    mrr = jnp.mean(reciprocal_ranks)
        
    return mrr    

# 2. Poincaré Projection for Plotting
    
def lorentz_to_poincare_2d(lorentz_embeddings):    
    """    
    Projects 3D Lorentz embeddings (time, x, y) to a 2D Poincaré disk (x, y).    
    lorentz\_embeddings shape: (N, 3\) where index 0 is the time-like coordinate.    
    """    
    # Extract coordinates
    x_0 = lorentz_embeddings[:, 0] # Time-like
    x_spatial = lorentz_embeddings[:, 1:] # Space-like (x, y)
        
    # Apply the stereographic projection: x\_spatial / (x\_0 \+ 1\)    
    # The denominator is always \>= 2, so it is perfectly stable    
    poincare_2d = x_spatial / (x_0[:, None] + 1.0)
        
    return poincare_2d    
    
# Example usage for plotting:    
# projected_points = lorentz_to_poincare_2d(trained_lorentz_nodes)    
# plt.scatter(projected_points[:, 0], projected_points[:, 1])   

```

When those projected points are plotted, the root of the forest should be seen near the center (0,0), with independent trees branching outward toward the rim. The individual Markov blankets will appear as tight, localized clusters along those branches.  
