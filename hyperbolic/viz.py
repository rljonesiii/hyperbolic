import matplotlib.pyplot as plt
import numpy as np
from hyperbolic.math import lorentz_to_poincare_2d


def plot_poincare_disk(
    lorentz_embeddings, node_depth=None, edges=None, save_path="poincare_viz.png"
):
    """
    Visualizes Lorentz embeddings by projecting them down to a 2D Poincare disk.
    lorentz_embeddings shape: (N, 3)
    node_depth: optional dictionary or array of node depths for coloring.
    edges: optional list of (u, v) tuples to draw connecting lines between nodes.
    """
    # Project to 2D
    poincare_2d = lorentz_to_poincare_2d(lorentz_embeddings)
    poincare_2d = np.array(poincare_2d)  # Convert to numpy for matplotlib

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the boundary of the Poincare disk
    circle = plt.Circle(
        (0, 0), 1.0, color="black", fill=False, linestyle="--", alpha=0.5
    )
    ax.add_patch(circle)

    # Draw edges
    if edges is not None:
        for u, v in edges:
            x_vals = [poincare_2d[u, 0], poincare_2d[v, 0]]
            y_vals = [poincare_2d[u, 1], poincare_2d[v, 1]]

            # NOTE: Drawing proper hyperbolic geodesics (arcs) is complex,
            # standard straight lines are an approximation for visual clustering.
            ax.plot(x_vals, y_vals, color="gray", alpha=0.3, linewidth=0.5)

    # Scatter points
    colors = "blue"
    if node_depth is not None:
        if isinstance(node_depth, dict):
            colors = [node_depth.get(i, 0) for i in range(len(poincare_2d))]
        else:
            colors = node_depth

    sc = ax.scatter(
        poincare_2d[:, 0], poincare_2d[:, 1], c=colors, cmap="viridis", s=15, zorder=5
    )

    if node_depth is not None:
        plt.colorbar(sc, label="Node Depth")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title("Poincaré Disk Visualization")
    plt.axis("off")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {save_path}")
