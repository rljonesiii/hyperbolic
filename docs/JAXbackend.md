# JAX Backend

JAX is arguably the best possible backend for this task right now. In fact, the machine learning community has been actively migrating Riemannian geometry and hyperbolic neural networks to JAX because its functional paradigm—specifically jit (Just-In-Time compilation), vmap (auto-vectorization), and grad (automatic differentiation)—is a perfect match for the heavy mathematical operations required by manifold geometry.

Here is a breakdown of how one approaches Riemannian optimization (like RSGD or Riemannian Adam) in JAX, the libraries available, and how the underlying math shifts.

## The JAX Paradigm for Manifolds

In standard Euclidean JAX (using a library like DeepMind's optax), an optimizer simply takes the gradients, calculates some statistics (like momentum), and returns an update vector that is subtracted from the parameters.

This operation is not possible on a manifold. If an Adam-calculated update vector is simply subtracted from a Lorentz node embedding, the embedding will be pulled off the hyperboloid.

To accomplish this in JAX, a Riemannian optimizer must perform three distinct geometric operations:

1. **Riemannian Gradient Conversion:** Take the standard Euclidean gradient computed by jax.grad and project it onto the tangent space using the metric of the manifold.  
2. **Tangent Space Momentum (Parallel Transport):** If an optimizer with "memory" is used (like Adam, which tracks running averages of past gradients), adding yesterday's gradient to today's gradient directly is not feasible. Yesterday's gradient lives on a *different* tangent plane (the tangent plane of the previous network weights). A geometric operation called **Parallel Transport** must be used to "slide" the old momentum vector along the curved surface of the manifold until it rests on the current tangent plane.  
3. **Retraction (The Exponential Map):** Finally, the exponential map is used to apply the computed step vector, ensuring the updated weights land perfectly back on the manifold.

## The JAX Ecosystem for Hyperbolic Deep Learning

It is not necessary to write all of this from scratch. There are emerging libraries built specifically to bridge Riemannian geometry and JAX:

* **Rieoptax & RiemannAX:** These are dedicated open-source libraries for Riemannian optimization in JAX. Rieoptax is explicitly designed to mimic the API of standard optax, allowing the use of Riemannian Stochastic Gradient Descent (RSGD), Riemannian Adam, and Riemannian AdamW seamlessly behind the scenes.  
* **Geomstats (with JAX backend):** geomstats is an incredibly robust differential geometry library that allows setting jax as its computational backend. It can be used to fetch the exact, heavily tested formulas for the Lorentz manifold's exponential maps, logarithmic maps, and parallel transport, and then jit-compile them.  
* **dm-haiku implementations:** There are several open-source repositories (like hyperbolic-nn-haiku) that implement Stereographic and Lorentz layers directly in DeepMind's Haiku/Optax framework.

**How Riemannian Adam Works under the Hood**

If the update\_step function for a Riemannian Adam optimizer were written in pure JAX, it would look something like this mathematically:

```python
import jax    
import jax.numpy as jnp    
    
@jax.jit    
def riemannian_adam_step(params, euclidean_grads, state, learning_rate):    
    # state contains [m (first moment), v (second moment)]    
        
    # 1. Project Euclidean grad to Riemannian grad on the Tangent Space    
    riemannian_grad = project_to_tangent_space(params, euclidean_grads)    
        
    # 2. Parallel Transport the old momentum to the new tangent space    
    # (Because the parameters moved in the last step)    
    m_transported = parallel_transport(state.prev_params, params, state.m)    
        
    # 3. Update biased first moment estimate (in the tangent space)    
    m_new = beta1 * m_transported + (1 - beta1) * riemannian_grad    
        
    # 4. Update biased second raw moment estimate     
    # (using the Minkowski inner product norm of the gradient)    
    v_new = beta2 * state.v + (1 - beta2) * minkowski_norm(riemannian_grad)**2    
        
    # 5. Compute the tangent step vector (with bias corrections)    
    m_hat = m_new / (1 - beta1**t)    
    v_hat = v_new / (1 - beta2**t)    
    tangent_step = -learning_rate * (m_hat / (jnp.sqrt(v_hat) + epsilon))    
        
    # 6. Apply the Exponential Map to update the parameters    
    new_params = exponential_map(params, tangent_step)    
        
    return new_params, new_state
```

## To Summarize

The beauty of writing this in JAX is `jax.vmap`.

In an HGAT, the logarithmic map between a target node and *every single node in its Markov blanket* will have to be computed. In PyTorch, doing this efficiently requires writing highly complex, custom CUDA kernels to batch the operations without boundary errors.

In JAX, one simply writes the math for a *single* node-to-node logarithmic map, wraps the function in `jax.vmap`, and JAX's XLA compiler automatically generates an optimal, heavily parallelized hardware routine for the entire graph.

The next topic for exploration is how to initialize the weights of a hyperbolic neural network layer in JAX, considering that standard Euclidean initializations (like Xavier or He) will place weights completely off the manifold.