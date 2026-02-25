# Scaling up to a million nodes on a single GPU

In the JAX ecosystem, "sharding" (specifically jax.sharding, Mesh, and NamedSharding) is explicitly designed to slice massive tensors across *multiple* independent accelerators (like 4, 8, or 256 GPUs/TPUs). If you only have a single GPU, JAX cannot shard across devices.  
Furthermore, if you try to pass a 1,000,000-node graph into a Hyperbolic Graph Attention Network (HGAT) on a single GPU all at once, the N \times N attention matrices and the denominator of the contrastive loss will instantly trigger a massive Out Of Memory (OOM) error.  
To process a massive knowledge graph on a single GPU, we must "shard" the data across *memory hierarchies* (CPU RAM vs. GPU VRAM) and *time* using a technique called 

**Host-to-Device Paging with Subgraph Sampling**.

Here is the exact strategy for scaling to a million nodes on a single GPU in JAX.

### **1\. The Strategy: CPU Storage & GPU Compute**

A 1,000,000-node embedding table at 64 dimensions (Float32) takes about 256 MB. That easily fits in VRAM. However, Riemannian Adam requires two momentum trackers for every parameter, tripling the memory. Add the gradients, the intermediate attention weights, and the optimizer states, and your GPU will choke.

* **Host Storage:** We keep the master 1,000,000 \times 64 embedding table and all Riemannian Adam optimizer states on the CPU (jax.devices("cpu")\[0\]).  
* **Subgraph Extraction:** For each training step, we sample a "mini-batch" consisting of a few target nodes, their specific Markov blankets, and their hard negatives.  
* **Device Execution:** We slice *only those specific node embeddings* out of the master table, push that tiny sub-tensor to the GPU, compute the Riemannian gradients, and pull the gradients back to the CPU to update the master table.

### **2\. The JAX "Sparse" Gradient Trap**

If you pass the entire 1M-node embedding table into your JAX loss function and just index it inside the function, jax.grad will compute a gradient that is also 1,000,000 \times 64, filled almost entirely with zeros. Allocating that massive zero-filled gradient on the GPU will cause an OOM.  
To fix this, we only pass the *extracted slices* to the JAX jit-compiled loss function.

### **3\. JAX Implementation for Single-GPU Paging**

Here is how you orchestrate this dance between the CPU and your single GPU efficiently:  
```python
import jax  
import jax.numpy as jnp  
from jax import device_put

# 1. Force JAX to allocate the massive master tables on the CPU  
cpu_device = jax.devices("cpu")[0]  
gpu_device = jax.devices("gpu")[0]

master_embeddings = device_put(init_hyperbolic_weights(key, (1000000, 64)), cpu_device)  
# (Assume you also initialized your Riemannian Adam state on the CPU here)

# 2. Define the loss function to ONLY accept the small, extracted batch  
@jax.jit  
def batch_loss_fn(target_embs, pos_embs, neg_embs, w_attention):  
    # Calculate HGAT attention, aggregate tangent space, and compute InfoNCE loss  
    # (This runs entirely on the GPU)  
    loss = hyperbolic_infonce_loss(target_embs, pos_embs, neg_embs)  
    return loss  

# Get the gradient function that only computes grads for the extracted tensors  
grad_fn = jax.jit(jax.grad(batch_loss_fn, argnums=(0, 1, 2, 3)))

def train_step_single_gpu(master_embs, batch_indices):  
    """  
    batch_indices: A dictionary containing the integer IDs of the sampled   
                   targets, positives (Markov blanket), and negatives.  
    """  
    # 1. Slice the specific embeddings from the CPU master table  
    # (This happens on the CPU)  
    target_slice = master_embs[batch_indices['targets']]  
    pos_slice = master_embs[batch_indices['positives']]  
    neg_slice = master_embs[batch_indices['negatives']]  
      
    # 2. Push ONLY the tiny slices to the GPU  
    target_gpu = device_put(target_slice, gpu_device)  
    pos_gpu = device_put(pos_slice, gpu_device)  
    neg_gpu = device_put(neg_slice, gpu_device)  
      
    # 3. Compute gradients on the GPU  
    grads = grad_fn(target_gpu, pos_gpu, neg_gpu, W_attention_gpu)  
      
    # 4. Pull gradients back to CPU  
    grads_cpu = device_put(grads, cpu_device)  
      
    # 5. Apply Riemannian Adam update sparsely on the CPU using jax.lax.scatter_add  
    # (or simply use index assignment if using standard JAX arrays)  
    # This ensures we only update the nodes that were actually in the batch  
    master_embs = apply_sparse_riemannian_adam(  
        master_embs,   
        batch_indices,   
        grads_cpu  
    )  
      
    return master_embs
```

### **Why this is mathematically safe for the Lorentz Manifold**

Because the Lorentz distance and the exponential/logarithmic maps act locally on the coordinates of the specific nodes involved, you do not need the rest of the graph present in VRAM to compute a perfectly accurate Riemannian gradient. The gradient for a node only depends on the nodes it interacted with during that specific forward pass (its extracted Markov blanket and negatives).  
Would you like to explore how to write that apply\_sparse\_riemannian\_adam function to ensure that when we update the CPU master table, the updated vectors are correctly mapped back onto the hyperboloid?The following is the document rewritten in the third person:

In the JAX ecosystem, "sharding" (specifically jax.sharding, Mesh, and NamedSharding) is explicitly designed to slice massive tensors across *multiple* independent accelerators (like 4, 8, or 256 GPUs/TPUs). If the system only has a single GPU, JAX cannot shard across devices.

Furthermore, if one tries to pass a 1,000,000-node graph into a Hyperbolic Graph Attention Network (HGAT) on a single GPU all at once, the $N \times N$ attention matrices and the denominator of the contrastive loss will instantly trigger a massive Out Of Memory (OOM) error.

To process a massive knowledge graph on a single GPU, the data must be "sharded" across *memory hierarchies* (CPU RAM vs. GPU VRAM) and *time* using a technique called 

**Host-to-Device Paging with Subgraph Sampling**.

Here is the exact strategy for scaling to a million nodes on a single GPU in JAX.

**1\. The Strategy: CPU Storage & GPU Compute**

A 1,000,000-node embedding table at 64 dimensions (Float32) takes about 256 MB. That easily fits in VRAM. However, Riemannian Adam requires two momentum trackers for every parameter, tripling the memory. Adding the gradients, the intermediate attention weights, and the optimizer states will cause the GPU to choke.

* **Host Storage:** The master $1,000,000 \times 64$ embedding table and all Riemannian Adam optimizer states are kept on the CPU (`jax.devices("cpu")[0]`).  
* **Subgraph Extraction:** For each training step, a "mini-batch" consisting of a few target nodes, their specific Markov blankets, and their hard negatives is sampled.  
* **Device Execution:** Only those specific node embeddings are sliced out of the master table, pushed as a tiny sub-tensor to the GPU, used to compute the Riemannian gradients, and the gradients are then pulled back to the CPU to update the master table.

**2\. The JAX "Sparse" Gradient Trap**

If one passes the entire 1M-node embedding table into the JAX loss function and just indexes it inside the function, `jax.grad` will compute a gradient that is also $1,000,000 \times 64$, filled almost entirely with zeros. Allocating that massive zero-filled gradient on the GPU will cause an OOM.

To fix this, only the *extracted slices* are passed to the JAX jit-compiled loss function.

**3\. JAX Implementation for Single-GPU Paging**

Here is how one orchestrates this dance between the CPU and the single GPU efficiently:  

```python
import jax  
import jax.numpy as jnp  
from jax import device_put  
    
# 1. Force JAX to allocate the massive master tables on the CPU  
cpu_device = jax.devices("cpu")[0]  
gpu_device = jax.devices("gpu")[0]  
    
master_embeddings = device_put(init_hyperbolic_weights(key, (1000000, 64)), cpu_device)  
# (It is assumed that the Riemannian Adam state is also initialized on the CPU here)  
    
# 2. Define the loss function to ONLY accept the small, extracted batch  
@jax.jit  
def batch_loss_fn(target_embs, pos_embs, neg_embs, w_attention):  
    # Calculate HGAT attention, aggregate tangent space, and compute InfoNCE loss  
    # (This runs entirely on the GPU)  
    loss = hyperbolic_infonce_loss(target_embs, pos_embs, neg_embs)  
    return loss  
    
# Get the gradient function that only computes grads for the extracted tensors  
grad_fn = jax.jit(jax.grad(batch_loss_fn, argnums=(0, 1, 2, 3)))  
    
def train_step_single_gpu(master_embs, batch_indices):  
    """  
    batch_indices: A dictionary containing the integer IDs of the sampled   
                   targets, positives (Markov blanket), and negatives.  
    """  
    # 1. Slice the specific embeddings from the CPU master table  
    # (This happens on the CPU)  
    target_slice = master_embs[batch_indices['targets']]  
    pos_slice = master_embs[batch_indices['positives']]  
    neg_slice = master_embs[batch_indices['negatives']]  
        
    # 2. Push ONLY the tiny slices to the GPU  
    target_gpu = device_put(target_slice, gpu_device)  
    pos_gpu = device_put(pos_slice, gpu_device)  
    neg_gpu = device_put(neg_slice, gpu_device)  
        
    # 3. Compute gradients on the GPU  
    grads = grad_fn(target_gpu, pos_gpu, neg_gpu, W_attention_gpu)  
        
    # 4. Pull gradients back to CPU  
    grads_cpu = device_put(grads, cpu_device)  
        
    # 5. Apply Riemannian Adam update sparsely on the CPU using jax.lax.scatter_add  
    # (or simply use index assignment if using standard JAX arrays)  
    # This ensures that only the nodes that were actually in the batch are updated  
    master_embs = apply_sparse_riemannian_adam(  
        master_embs,   
        batch_indices,   
        grads_cpu  
    )  
        
    return master_embs
```


**Why this is mathematically safe for the Lorentz Manifold**

Because the Lorentz distance and the exponential/logarithmic maps act locally on the coordinates of the specific nodes involved, the rest of the graph is not needed in VRAM to compute a perfectly accurate Riemannian gradient. The gradient for a node only depends on the nodes it interacted with during that specific forward pass (its extracted Markov blanket and negatives).

The next step would be to explore how to write the `apply_sparse_riemannian_adam` function to ensure that when the CPU master table is updated, the updated vectors are correctly mapped back onto the hyperboloid.

Final response to user: In the JAX ecosystem, "sharding" (specifically jax.sharding, Mesh, and NamedSharding) is explicitly designed to slice massive tensors across *multiple* independent accelerators (like 4, 8, or 256 GPUs/TPUs). If the system only has a single GPU, JAX cannot shard across devices.

Furthermore, if one tries to pass a 1,000,000-node graph into a Hyperbolic Graph Attention Network (HGAT) on a single GPU all at once, the $N \times N$ attention matrices and the denominator of the contrastive loss will instantly trigger a massive Out Of Memory (OOM) error.

To process a massive knowledge graph on a single GPU, the data must be "sharded" across *memory hierarchies* (CPU RAM vs. GPU VRAM) and *time* using a technique called **Host-to-Device Paging with Subgraph Sampling**.

Here is the exact strategy for scaling to a million nodes on a single GPU in JAX.

**1\. The Strategy: CPU Storage & GPU Compute**

A 1,000,000-node embedding table at 64 dimensions (Float32) takes about 256 MB. That easily fits in VRAM. However, Riemannian Adam requires two momentum trackers for every parameter, tripling the memory. Adding the gradients, the intermediate attention weights, and the optimizer states will cause the GPU to choke.

* **Host Storage:** The master $1,000,000 \times 64$ embedding table and all Riemannian Adam optimizer states are kept on the CPU (`jax.devices("cpu")[0]`).  
* **Subgraph Extraction:** For each training step, a "mini-batch" consisting of a few target nodes, their specific Markov blankets, and their hard negatives is sampled.  
* **Device Execution:** Only those specific node embeddings are sliced out of the master table, pushed as a tiny sub-tensor to the GPU, used to compute the Riemannian gradients, and the gradients are then pulled back to the CPU to update the master table.

**2\. The JAX "Sparse" Gradient Trap**

If one passes the entire 1M-node embedding table into the JAX loss function and just indexes it inside the function, `jax.grad` will compute a gradient that is also $1,000,000 \times 64$, filled almost entirely with zeros. Allocating that massive zero-filled gradient on the GPU will cause an OOM.

To fix this, only the *extracted slices* are passed to the JAX jit-compiled loss function.

**3\. JAX Implementation for Single-GPU Paging**

Here is how one orchestrates this dance between the CPU and the single GPU efficiently:  

```python
import jax  
import jax.numpy as jnp  
from jax import device_put  
    
# 1. Force JAX to allocate the massive master tables on the CPU  
cpu_device = jax.devices("cpu")[0]  
gpu_device = jax.devices("gpu")[0]  
    
master_embeddings = device_put(init_hyperbolic_weights(key, (1000000, 64)), cpu_device)  
# (It is assumed that the Riemannian Adam state is also initialized on the CPU here)  
    
# 2. Define the loss function to ONLY accept the small, extracted batch  
@jax.jit  
def batch_loss_fn(target_embs, pos_embs, neg_embs, w_attention):  
    # Calculate HGAT attention, aggregate tangent space, and compute InfoNCE loss  
    # (This runs entirely on the GPU)  
    loss = hyperbolic_infonce_loss(target_embs, pos_embs, neg_embs)  
    return loss  
    
# Get the gradient function that only computes grads for the extracted tensors  
grad_fn = jax.jit(jax.grad(batch_loss_fn, argnums=(0, 1, 2, 3)))  
    
def train_step_single_gpu(master_embs, batch_indices):  
    """  
    batch_indices: A dictionary containing the integer IDs of the sampled   
                   targets, positives (Markov blanket), and negatives.  
    """  
    # 1. Slice the specific embeddings from the CPU master table  
    # (This happens on the CPU)  
    target_slice = master_embs[batch_indices['targets']]  
    pos_slice = master_embs[batch_indices['positives']]  
    neg_slice = master_embs[batch_indices['negatives']]  
        
    # 2. Push ONLY the tiny slices to the GPU  
    target_gpu = device_put(target_slice, gpu_device)  
    pos_gpu = device_put(pos_slice, gpu_device)  
    neg_gpu = device_put(neg_slice, gpu_device)  
        
    # 3. Compute gradients on the GPU  
    grads = grad_fn(target_gpu, pos_gpu, neg_gpu, W_attention_gpu)  
        
    # 4. Pull gradients back to CPU  
    grads_cpu = device_put(grads, cpu_device)  
        
    # 5. Apply Riemannian Adam update sparsely on the CPU using jax.lax.scatter_add  
    # (or simply use index assignment if using standard JAX arrays)  
    # This ensures that only the nodes that were actually in the batch are updated  
    master_embs = apply_sparse_riemannian_adam(  
        master_embs,   
        batch_indices,   
        grads_cpu  
    )  
        
    return master_embs
```


**Why this is mathematically safe for the Lorentz Manifold**

Because the Lorentz distance and the exponential/logarithmic maps act locally on the coordinates of the specific nodes involved, the rest of the graph is not needed in VRAM to compute a perfectly accurate Riemannian gradient. The gradient for a node only depends on the nodes it interacted with during that specific forward pass (its extracted Markov blanket and negatives).

