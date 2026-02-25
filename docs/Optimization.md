# Optimization

When a neural network is trained in standard Euclidean space, the gradient is calculated and a fraction of it is simply subtracted from the weights. However, if this procedure is attempted on a hyperboloid, the updated point will instantly "fall off" the curved manifold and into empty space.

To remain on the hyperbolic surface, gradient descent must be performed using the **tangent space**, the **exponential map**, and the **logarithmic map**.

Here is the step-by-step mathematical pipeline for a single gradient update on the Lorentz manifold ($\mathcal{H}^n$).

## 1. Finding the Riemannian Gradient

The automatic differentiation framework (such as PyTorch) will blindly compute standard Euclidean gradients, $\nabla_E L$. To utilize this, it must first be converted into a Minkowski gradient by flipping the sign of the time-like coordinate:

Next, this gradient must be projected so that it lies perfectly flat on the **tangent space** ($T_x \mathcal{H}^n$) touching the current node embedding x. This is done by removing any component of the gradient that points away from the surface:

Now, $\nabla_R L$ is the true Riemannian gradient. It indicates the steepest direction to walk *while remaining on the hyperboloid*.

## 2. The Exponential Map (Taking the Step)

This vector cannot be simply added to $x$. Instead, the gradient is scaled by the learning rate $\eta$ to obtain the step vector $v = -\eta \nabla_R L$.

To apply this step, the flat tangent vector $v$ is "wrapped" back around the curvature of the hyperboloid using the **exponential map** ($\exp_x$):

*(Note: $\Vert v\Vert_{\mathcal{L}} = \sqrt{\langle v, v \rangle_{\mathcal{L}}}$ is the Minkowski norm of the vector).*

This process cleanly deposits the updated embedding exactly on the hyperbolic manifold, completely avoiding the boundary singularities of the Poincaré ball.

## 3. The Logarithmic Map (The Inverse)

The **logarithmic map** ($\log_x$) is the exact inverse of the exponential map. If the exponential map takes a flat velocity vector and curves it into a destination point $y$ on the manifold, the logarithmic map takes a destination point $y$ and flattens it into a velocity vector on the tangent space of $x$.

For two points $x, y \in \mathcal{H}^n$, the logarithmic map is defined as:

## Why the Logarithmic Map Matters for Markov Blankets

While the exponential map is used for *updating* embeddings during training, the logarithmic map is incredibly useful for *aggregating* Markov blankets.

When aggregating the features of a node's Markov blanket (its parents, children, and spouses), the coordinates cannot be simply averaged in hyperbolic space. Instead, the logarithmic map is used to project all those neighboring nodes onto the flat tangent space of the target node, a standard Euclidean aggregation (like a weighted sum in a Graph Convolutional Network) is performed, and then the exponential map is used to project the result back onto the manifold.