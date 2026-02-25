# Training

In standard Euclidean contrastive learning (like SimCLR or standard GraphSAGE), we often use the dot product or cosine similarity to measure how close two vectors are. In the Lorentz model, we cannot do this. Instead, we directly use the **Lorentzian distance** to define our contrastive push-and-pull dynamics.

Here is the mathematical and programmatic breakdown of how to build a Hyperbolic InfoNCE (Noise-Contrastive Estimation) loss function.

## The Hyperbolic Distance (The Logit)

Everything starts with how far apart two node embeddings are on the hyperboloid. For a target node u and another node v, the distance is defined by the Minkowski inner product:

In a contrastive setting, we want this distance to be exactly our "score." A smaller distance means higher similarity. To use this in a softmax-style loss, we simply negate the distance: $-d_{\mathcal{L}}(u, v)$.

## Defining Positive and Negative Pairs

To train the network, we need to sample pairs of nodes:

* **Positive Pairs $(u, v^+)$:** Two nodes that share highly similar Markov blankets. (e.g., they share the same parents, or we generated $v^+$ by heavily masking/augmenting the structural neighborhood of u). We want to **pull** these together.  
* **Negative Pairs $(u, v^-_k)$:** A set of K random nodes from the forest that have completely different topological contexts. We want to **push** these apart.

## The Hyperbolic InfoNCE Loss

We feed our negative hyperbolic distances into a standard cross-entropy framework. The loss for a single target node u looks like this:

Here, \tau is the **temperature parameter**. In hyperbolic space, distance grows exponentially, meaning the raw distance values can become very large very quickly. The temperature parameter is absolutely critical here to scale the distances down before they hit the exp() function; otherwise, our softmax will immediately overflow into NaNs.

## JAX Implementation

Because JAX is heavily optimized for matrix operations, we can compute the distance between the target node u and *all* samples (both positive and negative) simultaneously using jax.vmap or standard broadcasting.

Here is how **we** write this hyperbolic contrastive loss cleanly in JAX:  

```python
import jax  
import jax.numpy as jnp

def minkowski_inner_product(u, v):  
    """Computes the Minkowski inner product between u and v."""  
    # u and v have shape (..., spatial\_dim \+ 1\)  
    # Time-like coordinate is at index 0  
    time_product = -u[..., 0] * v[..., 0]  
    space_product = jnp.sum(u[..., 1:] * v[..., 1:], axis=-1)  
    return time_product + space_product

def lorentz_distance(u, v, eps=1e-15):  
    """Computes the Riemannian distance on the Lorentz manifold."""  
    inner_prod = minkowski_inner_product(u, v)  
    # Clip to avoid domain errors in arcosh (inner_prod must be \<= \-1)  
    # We use \-1 \- eps to ensure numerical stability during gradients  
    inner_prod = jnp.minimum(inner_prod, -1.0 - eps)  
    return jnp.arccosh(-inner_prod)

def hyperbolic_infonce_loss(u, v_pos, v_negs, temperature=0.1):  
    """  
    u: Target node embedding (shape: D)  
    v_pos: Positive pair embedding (shape: D)  
    v_negs: Array of negative pair embeddings (shape: K, D)  
    """  
    # 1. Calculate distances  
    dist_pos = lorentz_distance(u, v_pos)          # Shape: scalar  
    dist_negs = jax.vmap(lorentz_distance, in_axes=(None, 0))(u, v_negs) # Shape: (K,)  
        
    # 2. Apply temperature scaling and negate (closer = higher score)  
    score_pos = -dist_pos / temperature  
    scores_negs = -dist_negs / temperature  
        
    # 3. Concatenate all scores for the denominator  
    # [score_pos, score_neg_1, score_neg_2, ..., score_neg_K]  
    all_scores = jnp.concatenate([jnp.array([score_pos]), scores_negs])  
        
    # 4. Compute the log-softmax loss  
    # We use logsumexp for numerical stability  
    lse = jax.scipy.special.logsumexp(all_scores)  
    loss = -(score_pos - lse)  
        
    return loss
```

## Why This Works Perfectly for Forests

By optimizing this loss function using our JAX-based Riemannian Adam optimizer, the network forces the geometric structure of the hyperboloid to mirror our Markov blanket similarities. Nodes with similar predictive contexts will clump into tight semantic neighborhoods, and the exponential volume of the hyperbolic space ensures that even massive numbers of unrelated negative nodes have plenty of room to be pushed away without crowding each other.

