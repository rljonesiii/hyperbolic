# Attention

Implementing an attention mechanism in hyperbolic space is a brilliant way to handle Markov blankets. Not all neighbors in a Markov blanket are equally important; for a given predictive task, a parent might be highly informative, while a spouse might be irrelevant noise. Standard Hyperbolic Graph Convolutional Networks (HGCNs) treat all neighbors equally (or weight them purely by node degree). A Hyperbolic Graph Attention Network (HGAT) solves this by dynamically learning the importance of each node. Here is how we can weave the attention mechanism into the Lorentz model pipeline we just built.

## Step 1: Computing Raw Attention Scores (At the Origin)

To decide how much attention node $x$ should pay to its neighbor $y \in \mathcal{M}(x)$, we need to compare their features. Since standard neural network layers (like linear transformations and non-linearities) operate in Euclidean space, we project the node embeddings to the tangent space of the origin, $T_o\mathcal{H}^n$, to compute the attention score.

1. **Map to Origin:** Project both the target node x and the neighbor y to the origin's tangent space:  
2. **Concatenation & Scoring:** We concatenate these two Euclidean feature vectors, multiply them by a learnable weight vector a, and apply a LeakyReLU activation to get the raw attention score $e_{xy}$.

*Note: Some advanced HGAT implementations also explicitly include the Lorentzian distance $d_{\mathcal{L}}(x, y)$ as an extra input feature here, letting the network explicitly weigh geometric proximity alongside feature similarity.*

## Step 2: Softmax Normalization Over the Markov Blanket

The raw score $e_{xy}$ is unconstrained. To make it a valid, probabilistic weight, we normalize it across the entire Markov blanket $\mathcal{M}(x)$ using a softmax function.  
This ensures that the total attention node x pays to its parents, children, and spouses sums to exactly 1:

## Step 3: Weighted Tangent-Space Aggregation

Now that we have our learned attention weight $w_{xy}$, we return to the tangent space of the target node x ($T_x\mathcal{H}^n$) to pull the messages together, just as we did before.  
Instead of a simple average, we multiply each neighbor's tangent velocity vector by its specific attention weight:

## Step 4: Retraction to the Lorentz Manifold

Finally, we take this attention-weighted velocity vector $v_{agg}$ and shoot the node back onto the hyperboloid using the exponential map:

## Why This is Powerful for Knowledge Graphs

By using HGAT on a Markov blanket, your network can dynamically shift its focus as it traverses the hierarchy. Deep in the leaves of the forest, it might learn to heavily weight sibling/spouse relationships (lateral attention), while near the roots, it might focus almost entirely on parent-child structural edges (vertical attention)—all while remaining mathematically locked to the stable Lorentz manifold.  

