# Feature-Modulated Tangent Aggregation

We advanced our network from a purely structural topological embedding model to a fully contextualized Knowledge Graph representation. By natively fusing flat (Euclidean) node and edge attributes with curved (Hyperbolic) geometric representations, our model now creates rich embeddings that deeply account for both *where* a node is in the hierarchy and *what* properties it and its relationships possess.

**Why are features "Euclidean"?** 
Standard node and edge attributes—such as real-valued continuous measurements, one-hot encoded categorical variables, or dense representation vectors—naturally exist in a flat $\mathbb{R}^n$ space. In this Euclidean space, straight-line distance, vector addition, and scaling follow standard classical geometry. In contrast, our structural node embeddings live on a curved Hyperbolic manifold where space expands exponentially, meaning operations like simple vector addition or concatenation with Euclidean vectors are mathematically invalid and break the manifold's curvature constraints.

**What is the Tangent Space?**
To resolve this incompatibility, we use the **Tangent Space** as a "mixing desk." Geometrically, the Tangent Space at a specific point on the manifold (such as the origin) is a flat, Euclidean hyperplane that grazes the curved surface exactly at that point—similar to how a flat, rigid sheet of paper can touch the curved surface of a sphere or saddle at a single spot. Because this local tangent space is fully Euclidean ($\mathbb{R}^n$), it allows us to safely pull our curved Hyperbolic embeddings into it (using the *logarithmic map*), concatenate them seamlessly with our flat Euclidean features, apply standard neural network layers (like Linear layers), and finally project the enriched result perfectly back onto the curved manifold (using the *exponential map*).

This document details the realization of the Feature-Modulated Tangent Aggregation logic in our JAX codebase.

## 1. Attention Modulation (Routing via Attributes)

Implemented in `hyperbolic/nn.py` inside the `compute_attention_scores` function.

### Concept
The HGAT layer learns to route information based on a combination of where nodes are structurally (geometry) and what they represent (features). The attention mechanism can thus prioritize connections that share high-weight features (e.g., `is_malignant=True`) even if structurally distant.

### Realization
Instead of deriving attention weights purely from the hyperbolic representations $x$ and $y$, we map both to the tangent space of the origin $o$ to get $h_x$ and $h_y$. We then extend this base with the node features ($F_x, F_y$) and the edge attributes ($E_{xy}$):

```python
# From hyperbolic/nn.py: compute_attention_scores
components = [h_x_tiled, h_y]

if node_features_x is not None and node_features_y is not None:
    nx_expanded = jnp.expand_dims(node_features_x, axis=-2)
    nx_tiled = jnp.broadcast_to(nx_expanded, node_features_y.shape)
    components.extend([nx_tiled, node_features_y])

if edge_features is not None:
    components.append(edge_features)

concat_features = jnp.concatenate(components, axis=-1)
raw_scores = jax.nn.leaky_relu(jnp.dot(concat_features, a))
```

## 2. Message Enrichment (Fusing Features into the Embedding)

Implemented in `hyperbolic/nn.py` inside the `hyperbolic_gat_layer` function.

### Concept
Rather than simply pulling a neighbor's hyperbolic coordinate into the target node's tangent space and treating it as a raw coordinate point, we allow the neighbor's attributes and connecting edge labels to "color" the message. 

### Realization
We concatenate the tangent-space vector of the neighbor (`v_i`) with its attributes (`node_features_y`) and edge labels (`edge_features`). A linear projection weight `W_message` acts as an MLP in the Tangent space to compress the enriched message back down to the target dimension. To maintain manifold constraints, the resulting message is explicitly projected back onto the active tangent space before aggregation and retraction.

```python
# From hyperbolic/nn.py: hyperbolic_gat_layer
if W_message is not None:
    message_components = [v_i]
    if node_features_y is not None:
        message_components.append(node_features_y)
    if edge_features is not None:
        message_components.append(edge_features)

    # 1. Concatenate Geometry and Attributes
    enriched_msg = jnp.concatenate(message_components, axis=-1)
    
    # 2. Project via W_message 
    v_i = jnp.dot(enriched_msg, W_message)
    
    # 3. Project firmly back onto the target's tangent space!
    v_i = project_to_tangent_space(x_transformed_tiled, v_i)

# ...
# Multiply by attention weights, sum, and retract!
```

## 3. Paging Strategy & Context Delivery

To supply these features natively without massive rewrites to the core mathematical framework, a focused paging strategy was introduced.

Implemented across `hyperbolic/api.py`, `hyperbolic/data.py`, and `hyperbolic/train.py`:
1. **Extraction**: While parsing the `NetworkX MultiDiGraph`, node attributes and edge identifiers are extracted into dense, mapped arrays (`global_node_features` and an `edge_features_dict`).
2. **Batch Paging**: Inside the main evaluation and `train_step_single_gpu` batches, `b_target_feats`, `b_pos_feats`, and `b_pos_edge_feats` are actively sliced out alongside the pure structure nodes.
3. **Execution**: The sliced feature arrays are pushed immediately into the generic GPU layer step as `node_features_x`, `node_features_y`, and `edge_features`.

## Summary
By using the flat tangent space of the Lorentz manifold as our interaction layer, these modifications create a perfectly sound geometric foundation for analyzing deeply rich relational Knowledge Graphs—without breaking Hyperbolic principles.
