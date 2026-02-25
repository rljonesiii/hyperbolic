# The Lorentz Model

While the Poincaré ball is visually intuitive for understanding how hyperbolic space works (because it can be easily drawn as a 2D disk), the **Lorentz model** (also known as the Hyperboloid model) is the industry standard for actual deep learning implementations and optimization. 

## The Poincaré Singularity Problem

In the Poincaré ball model, the distance formula and the Riemannian metric both rely on the term $1 - |x|^2$ in the denominator. As a node is pushed deeper into the hierarchy (meaning its embedding moves toward the edge of the ball, so $|x| \to 1$), this denominator approaches zero.

**The Result:** Gradients explode, floating-point precision collapses, and NaN (Not a Number) errors occur during backpropagation. This is exactly why the authors of the paper discussed earlier had to invent complex "floating point expansions" just to keep Poincaré calculations stable on GPUs.

## The Lorentz Solution

The Lorentz model avoids the hard boundary of a ball. Instead, it embeds an $n$-dimensional hyperbolic space onto the upper sheet of an $(n+1)$-dimensional hyperboloid.

This is done by replacing the standard Euclidean dot product with the **Minkowski inner product**:
$$
<x, y>_L = -x_0y_0 + \sum_{i=1}^n x_iy_i
$$

The hyperbolic manifold is simply the set of points where this inner product with itself equals \-1:
$$
<x, x>_L = -1
$$

## Why It Is Mathematically Superior for Optimization

* **No Fractional Boundaries:** The distance between two points in the Lorentz model is elegantly simple: Because there is no $1 - |x|^2$ denominator, there are no catastrophic boundary singularities. The distance computation is numerically stable everywhere on the manifold.  

* **Efficient Optimization:** Operations in the Lorentz model look much closer to standard linear algebra. Researchers can perform standard stochastic gradient descent steps in the ambient Euclidean space and then cleanly project the points back onto the hyperboloid using a simple, stable operation.

