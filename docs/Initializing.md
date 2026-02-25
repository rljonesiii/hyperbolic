# Initializing

If standard PyTorch or JAX initializers (like Xavier/Glorot or Kaiming/He) are blindly applied to node embeddings or weight matrices, the values will be scattered randomly throughout ambient $\mathbb{R}^{n+1}$ space.

Because they will not satisfy the Minkowski inner product condition $\langle x, x \rangle_\mathcal{L} = -1$, the very first forward pass will immediately return NaNs when the network attempts to compute hyperbolic distances.

To safely initialize parameters on the Lorentz manifold, a technique called **Tangent Space Initialization** is used.

## The Geometry of Initialization

Direct sampling on the hyperboloid is not easily achieved, but sampling is straightforward in flat, Euclidean space. The trick is to treat the tangent space at the "origin" of the manifold as the flat canvas, initialize the weights there, and then wrap them onto the hyperboloid.

The exact geometric sequence is as follows:

1. **Start at the Origin:** In the Lorentz model, the origin (or base point) is the bottom pole of the upper hyperboloid sheet.  
2. **Sample in Euclidean Space:** The weights for the $n$ spatial dimensions are sampled using a favorite Euclidean initializer (such as a standard normal or uniform distribution scaled by Xavier bounds).  
3. **Embed in the Tangent Space:** To make the sampled vector a valid tangent vector at the origin ($T_o \mathcal{H}^n$), its time-like coordinate must be precisely zero. A zero is prepended to the sampled weights, yielding a velocity vector $v$.  
4. **Project to the Manifold:** Finally, the exponential map at the origin is used to "shoot" the origin along this velocity vector, landing exactly on the Lorentz manifold.

## Writing the JAX Initializer

JAX's purely functional random number generation (`jax.random.PRNGKey`) makes this pipeline incredibly clean to write. Because JAX requires explicit key splitting, there is total deterministic control over this manifold mapping.

Here is how a custom hyperbolic initializer function is written in JAX:

```python
import jax    
import jax.numpy as jnp    
    
def lorentz_exponential_map_origin(v):    
    """    
    Maps a tangent vector v = (0, x_1, ..., x_n) from the origin's     
    tangent space onto the Lorentz manifold.    
    """    
    # The Minkowski norm of a vector with a 0 time-coord is just its Euclidean norm    
    norm_v = jnp.linalg.norm(v, axis=-1, keepdims=True)    
        
    # Avoid division by zero for the zero vector    
    norm_v = jnp.where(norm_v == 0, 1e-15, norm_v)    
        
    # Calculate the exponential map at origin o = (1, 0, ..., 0)    
    time_coord = jnp.cosh(norm_v)    
    spatial_coords = jnp.sinh(norm_v) * (v[..., 1:] / norm_v)    
        
    # Concatenate the new time coordinate with the projected spatial coordinates    
    return jnp.concatenate([time_coord, spatial_coords], axis=-1)    
    
def init_hyperbolic_weights(key, shape, stddev=1e-3):    
    """    
    Initializes weights on the Lorentz manifold using Tangent Space Initialization.    
    Shape should be (num_nodes, spatial_dim).    
    """    
    # 1. Sample the spatial dimensions using standard normal (or Xavier)    
    spatial_shape = (shape[0], shape[1])    
    spatial_weights = stddev * jax.random.normal(key, spatial_shape)    
        
    # 2. Prepend the time-like coordinate with zeros to place it in T\_o H^n    
    zeros = jnp.zeros((shape[0], 1))    
    tangent_vectors = jnp.concatenate([zeros, spatial_weights], axis=-1)    
        
    # 3. Project the tangent vectors onto the Lorentz manifold    
    manifold_weights = lorentz_exponential_map_origin(tangent_vectors)    
        
    return manifold_weights    
    
# Example Usage:    
key = jax.random.PRNGKey(42)    
num_nodes = 1000    
spatial_dim = 64 # This means the Lorentz vectors will be 65-dimensional    
    
initial_embeddings = init_hyperbolic_weights(key, (num_nodes, spatial_dim))
```

## Why Scale Matters (The stddev Parameter)

Note the `stddev=1e-3` in the initialization code. In hyperbolic space, distance grows exponentially. If the weights are initialized with too large of a variance in the tangent space, the exponential map will shoot them incredibly far out toward the boundary of the Poincaré disk (or extremely high up the Lorentz hyperboloid).

When embeddings start too far from the origin, the gradients become vanishingly small, and the optimizer will struggle to move them. It is best practice in hyperbolic neural networks to initialize the embeddings very tightly clustered around the origin, allowing the network to slowly push them outward as it learns the hierarchical structure of the forest.
