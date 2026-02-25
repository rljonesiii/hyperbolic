# Negative Sampling in Hyperbolic Space

If we just use standard, uniform random negative sampling in hyperbolic space, our network will quickly stop learning.

Here is why: Hyperbolic space expands exponentially. If two nodes are picked at random from a large forest, the mathematical probability is that they are already incredibly far apart. Pushing them even further apart generates almost zero gradient signal. The network falls into a "lazy" local minimum where it just shoves everything toward the boundary of the space and stops organizing the hierarchy.

To force the network to learn the nuanced topology of our Markov blankets, we need **Hard Negative Sampling strategies** specifically tailored for trees and forests.

Here are the three most effective strategies for hierarchical knowledge graphs:

## Sibling and Cousin Contrast (Local Hard Negatives)

When two nodes are deep in the same tree, their Markov blankets look very similar. They might share the same parent and have similar children.

* **The Strategy:** Use siblings or close cousins as negative samples.  
* **Why it works:** Because their predictive contexts are so similar, the network will naturally try to embed them right on top of each other. By explicitly using a sibling as a negative, we force the model to carve out distinct angular coordinates for them, effectively fanning out the branches of the tree rather than collapsing them into a single line.

## Depth-Matched Negatives (The "Altitude" Problem)

In hyperbolic space, a node's distance from the origin (its norm) perfectly correlates with its depth in the hierarchy. Root nodes are near the center; leaf nodes are near the edge.

* **The Strategy:** If our target node is at depth d=4 in Tree A, we pick a negative node that is also at depth d=4 but in Tree B (or a distant branch of Tree A).  
* **Why it works:** If we do not do this, the network might just learn to cluster nodes solely by their depth, completely ignoring which specific branch or tree they belong to. Depth-matched negatives force the model to learn that "norm" is not the only important feature—the specific angular direction matters just as much.

## Markov Blanket Perturbation (Contextual Hard Negatives)

Since our explicit goal is to find nodes with similar Markov blankets, the hardest possible negative is a node whose Markov blanket is *almost* identical, but synthetically broken.

* **The Strategy:** We take the target node's actual Markov blanket, randomly drop one critical edge (e.g., sever the connection to its parent), and feed this "broken" structural neighborhood through our Graph Attention Network (HGAT) to generate a negative embedding.  
* **Why it works:** This is the ultimate test for our attention mechanism. It forces the network to pay extreme attention to the exact composition of the local neighborhood, learning that missing a specific parent fundamentally changes the semantic meaning of the node.

## How to Implement This in JAX efficiently

We do not want to calculate tree-depths and sibling relationships on the fly during training—that destroys GPU throughput. Instead, we pre-compute these relationships into a structured sampling matrix before training begins.

In JAX, we can use `jax.random.choice` paired with a pre-computed probability mask.

```python
import jax    
import jax.numpy as jnp    
    
def sample_hard_negatives(key, target_indices, negative_probability_matrix, num_samples):    
    """    
    target_indices: Array of node IDs we are training on.    
    negative_probability_matrix: A pre-computed (N x N) matrix where the value at (i, j)     
                                 is the probability of picking node j as a negative for node i.    
                                 (e.g., high probability for cousins, zero for actual positives).    
    """    
    # Fetch the probability distributions for our specific target nodes    
    # Shape: (batch_size, num_nodes_in_graph)    
    target_probs = negative_probability_matrix[target_indices]    
        
    # Use JAX's vectorized choice to sample 'num_samples' negatives per target    
    # categorically based on the biased probabilities.    
    keys = jax.random.split(key, target_indices.shape[0])    
        
    # vmap the random choice across the batch    
    def sample_single(k, probs):    
        return jax.random.choice(k, probs.shape[0], shape=(num_samples,), p=probs, replace=False)    
            
    sampled_negative_indices = jax.vmap(sample_single)(keys, target_probs)    
        
    return sampled_negative_indices 
```

By heavily weighting the `negative_probability_matrix` toward cousins, depth-matched nodes, and perturbed neighbors, our JAX training loop will constantly feed the optimizer the exact geometric collisions it needs to perfectly structure the Lorentz manifold.