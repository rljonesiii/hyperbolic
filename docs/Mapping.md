# Mapping

The relationship between the Poincaré ball ($\mathbb{D}^n$) and the Lorentz hyperboloid ($\mathbb{H}^n$) is elegantly defined by a stereographic projection. This can be viewed as projecting points from the bottom pole of the hyperboloid, located at $(-1, 0, \dots, 0)$, up through the origin and onto the unit disk.

Because these two models are isometric (they represent the exact same hyperbolic geometry), there is a smooth, reversible mapping—a diffeomorphism—between them. The exact mathematical bridge is as follows.

## From Lorentz to Poincaré

Let a point in the $(n+1)$-dimensional Lorentz model be defined as $x = (x_0, x_1, \dots, x_n) \in \mathbb{H}^n$, where $x_0$ is the time-like coordinate and the rest are space-like coordinates.

To map this point into the n-dimensional Poincaré ball, the projection $f: \mathbb{H}^n \rightarrow \mathbb{D}^n$ is applied:

It is noteworthy that this is clean. Since $x_0 \ge 1$ for all points on the upper sheet of the hyperboloid, the denominator $x_0 + 1$ is always strictly positive and well-behaved.

## From Poincaré to Lorentz

To reverse this, a point in the Poincaré ball is defined as $u \in \mathbb{D}^n$.

To map this point back onto the Lorentz hyperboloid, the inverse projection $f^{-1}: \mathbb{D}^n \rightarrow \mathbb{H}^n$ is used:

### The Proof of the Intuition

Examining the inverse mapping formula $f^{-1}(u)$ reveals that the denominator is exactly $1 - |u|^2$.

This explicitly proves the earlier point about numerical stability. As a point $u$ approaches the boundary of the Poincaré ball ($|u| \to 1$), that denominator approaches zero. In the Lorentz model, this simply means the time-like coordinate $x_0$ approaches infinity.

When optimizing directly in the Lorentz model, one completely bypasses dividing by $1 - |u|^2$. The coordinates are allowed to grow naturally in the ambient Euclidean space, which enables standard gradient descent to work without collapsing floating-point precision.

One may then consider how the actual gradient updates are performed directly on the Lorentz manifold using the exponential map and logarithmic map.