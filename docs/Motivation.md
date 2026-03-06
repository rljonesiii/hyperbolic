# Motivating Hyperbolic Embeddings

Before diving into the mathematics of the Lorentz model, it's crucial to understand **why** we use hyperbolic geometry to model hierarchical or tree-like data in the first place, and what we're actually looking at when we see a visualization of these embeddings.

## The Problem with Euclidean Space

Standard deep learning heavily relies on Euclidean space (flat space). However, embedding hierarchical data (like a tree) into flat space creates significant distortion. 

Consider how a tree grows:
*   At depth 1, the root has $b$ children.
*   At depth 2, there are $b^2$ children.
*   At depth $d$, there are $b^d$ nodes.

The number of nodes grows **exponentially** with the depth of the tree.

Now, consider the volume of a sphere in Euclidean space:
*   In $n$-dimensional Euclidean space, the volume of a sphere of radius $r$ grows **polynomially** proportional to $r^n$.

If you try to pack an exponentially growing number of nodes into a space that only expands polynomially, you quickly run out of room. Nodes that are structurally distinct and far apart in the tree end up being squished tightly together in the continuous space, destroying the hierarchy and confusing the neural network.

## The Hyperbolic Solution

Hyperbolic space is a space with constant negative curvature. One of its defining properties is that its volume grows **exponentially** with respect to its radius:
$$ \text{Volume} \propto e^{(n-1)r} $$

This means hyperbolic space naturally expands at the exact same rate that a tree grows out. Because of this property, hyperbolic space can be thought of as a **continuous analogue to discrete trees**. You can embed massive, highly-branched trees in a hyperbolic space with nearly zero distortion because there is always enough "room" at the boundaries to accommodate the exponentially growing number of leaf nodes.

When we model datasets that are inherently hierarchical—like social graphs, knowledge graphs, or organizational charts—using a hyperbolic manifold instead of a Euclidean one allows the network to capture the underlying structure perfectly without forcing artificial crowding.

## Visualizing the Unvisualizable: The Poincaré Ball

Hyperbolic space has constant negative curvature. It is infinite, meaning it goes on forever just like a flat 2D plane does, but its geometry makes it impossible to draw perfectly on a flat sheet of paper (or a flat computer monitor) without some form of distortion.

When we look at a visualization of our embeddings (often outputted as a circle), we are typically looking at the **Poincaré Ball** (or Disk in 2D). But what are we actually seeing?

You can think of the Poincaré Disk not as a sphere you are looking down upon, but rather as **an infinite plane that has been squashed into a finite circle** using a mathematical lens (a stereographic projection).

*   **The Center is "Flat":** Near the origin (the center of the disk), the space looks very similar to standard flat Euclidean space. Distances aren't heavily distorted. We usually place the "root" of our trees here.
*   **The Edge is Infinite:** As you move towards the edge of the circle (the boundary), the space is increasingly compressed by the visual projection. In the actual hyperbolic space, the boundary is infinitely far away. 
*   **The "Ant" Analogy:** Imagine you are an ant walking from the center of the disk toward the edge. In the visual representation, the ant's steps appear to get shorter and shorter as it gets closer to the boundary. But from the ant's perspective inside the hyperbolic space, its steps are always the exact same size. It will walk forever and never actually reach the drawing's outer edge, because the space expands infinitely to accommodate it.

This distortion is necessary so we can visualize the *entire, infinite* space at once. The "squishing" effect visually demonstrates the exponential volume—all that space near the boundary contains the exponentially vast amount of "room" needed for the leaves of our deeply nested trees.

Understanding this allows you to interpret the embeddings: nodes clustered near the center are high up in the hierarchy (roots), while nodes pushed out towards the dense, compressed edge are lower down (leaves).
