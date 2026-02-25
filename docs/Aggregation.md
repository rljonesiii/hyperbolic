# Aggregation

Aggregating a Markov blanket in hyperbolic space is the beating heart of a Hyperbolic Graph Convolutional Network (HGCN). Because the Lorentz manifold is curved, adding the coordinate vectors of a node's parents, children, and spouses together is not a viable operation—doing so would instantly pull the resulting vector off the hyperboloid and into the ambient Euclidean space, destroying the geometric properties.

To solve this, the tangent space at the target node is used as a temporary, flat "workstation" to perform message passing. The following is the step-by-step implementation of a single neighborhood aggregation phase for a target node $x$.

## The Setup: Defining the Neighborhood

Let the target node embedding be $x \in \mathcal{H}^n$. Its Markov blanket is defined as the set of neighboring nodes $\mathcal{M}(x)$, which consists of its parents $\mathcal{P}(x)$, children $\mathcal{C}(x)$, and spouses $\mathcal{S}(x)$. Let $y_i \in \mathcal{M}(x)$ represent the embedding of a specific node within this blanket.

## Step 1: Feature Transformation (At the Origin)

Before messages are passed, a learnable weight matrix $W$ is typically applied to transform the node features. Because standard matrix multiplication is a Euclidean operation, this transformation is performed in the tangent space of the "origin" (the pole of the hyperboloid, typically $o = (1, 0, \dots, 0)$).

1. Map the node to the origin's tangent space: $h_i = \log_o(y_i)$  
2. Apply the linear weight matrix: $h_i' = W h_i$  
3. Project back to the manifold: $y_i' = \exp_o(h_i')$

*(Note: In practice, PyTorch libraries often optimize this sequence into a single $mobius\_matvec$ operation).*

## Step 2: Lifting to the Target's Tangent Space

The transformed neighbor embeddings, $y_i'$, must be aggregated into the target node $x$. This requires pulling all of these neighbors onto the flat tangent plane resting precisely at $x$, denoted as $T_x \mathcal{H}^n$.

This is achieved using the logarithmic map. For every node in the Markov blanket, its tangent vector is calculated:

## Step 3: Tangent Space Aggregation

Now that all the messages $v_i$ are represented as vectors on a flat, Euclidean-like tangent plane, standard linear combinations can be safely performed.

The messages are aggregated using a weighted sum:

The weight $w_i$ can be a standard GCN normalization constant (like $1 / \sqrt{\text{deg}(x)\text{deg}(y_i)})$, or it can be a learned attention coefficient.

## Step 4: Retraction Back to the Manifold

Finally, $v_{agg}$ is a velocity vector sitting on the tangent plane $T_x \mathcal{H}^n$. It represents the combined "pull" of the entire Markov blanket. The exponential map is used to shoot the target node $x$ along this vector, landing safely back on the Lorentz manifold to get the updated embedding:

By routing the message passing through the tangent space, the mathematically rigorous geometry of the Lorentz model is perfectly preserved while successfully pooling the predictive context of the Markov blanket.

An implementation walkthrough of a Hyperbolic Graph Attention (HGAT) mechanism can dynamically learn those $w_i$ weights, allowing the network to decide whether parents, children, or spouses are more important for a given node.