import matplotlib.pyplot as plt
import numpy as np
from hyperbolic.math import lorentz_to_poincare_2d
from adjustText import adjust_text


def plot_poincare_disk(
    lorentz_embeddings,
    node_depth=None,
    node_labels=None,
    edges=None,
    annotations=None,
    save_path="poincare_viz.png",
):
    """
    Visualizes Lorentz embeddings by projecting them down to a 2D Poincare disk.
    lorentz_embeddings shape: (N, 3)
    node_depth: optional dictionary or array of numerical node depths for continuous coloring.
    node_labels: optional dictionary or array of string category labels for categorical coloring & legend.
    edges: optional list of (u, v) tuples to draw connecting lines between nodes.
    annotations: optional dictionary mapping node indices to string labels to annotate.
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

    # Determine colors and plot
    if node_labels is not None:
        # Categorical coloring with legend
        if isinstance(node_labels, dict):
            labels = [node_labels.get(i, "Unknown") for i in range(len(poincare_2d))]
        else:
            labels = node_labels

        unique_labels = list(set(labels))
        cmap = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")

        for i, label in enumerate(unique_labels):
            idx = [j for j, lbl in enumerate(labels) if lbl == label]
            ax.scatter(
                poincare_2d[idx, 0],
                poincare_2d[idx, 1],
                color=cmap(i % cmap.N),
                label=str(label),
                s=25,
                zorder=5,
            )
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Categories")

    elif node_depth is not None:
        # Continuous coloring via depth
        if isinstance(node_depth, dict):
            colors = [node_depth.get(i, 0) for i in range(len(poincare_2d))]
        else:
            colors = node_depth

        sc = ax.scatter(
            poincare_2d[:, 0],
            poincare_2d[:, 1],
            c=colors,
            cmap="viridis",
            s=15,
            zorder=5,
        )
        plt.colorbar(sc, label="Node Depth")

    else:
        # No coloring
        ax.scatter(poincare_2d[:, 0], poincare_2d[:, 1], color="blue", s=15, zorder=5)

    # Add text and arrow annotations
    if annotations is not None:
        texts = []
        for idx, text in annotations.items():
            px, py = poincare_2d[idx, 0], poincare_2d[idx, 1]
            texts.append(
                ax.text(
                    px,
                    py,
                    text,
                    fontsize=9,
                    zorder=10,
                    bbox=dict(pad=0.2, facecolor="white", edgecolor="none", alpha=0.8),
                )
            )

        # Use adjustText to naturally repel the labels away from data points and each other
        if texts:
            # Set axes limits BEFORE adjust_text so it bounds the layout engine
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

            adjust_text(
                texts,
                x=poincare_2d[:, 0],
                y=poincare_2d[:, 1],
                ax=ax,
                expand=(1.5, 1.5),
                force_text=(1.5, 2.0),
                force_points=(0.2, 0.2),
                ensure_inside_axes=True,
            )

            # Draw arrows connecting the closest edge (left or right) of the text box to the target
            renderer = fig.canvas.get_renderer()
            for idx, text_obj in zip(annotations.keys(), texts):
                target_x, target_y = poincare_2d[idx, 0], poincare_2d[idx, 1]

                # Get the bounding box of the text in data coordinates
                bbox = text_obj.get_window_extent(renderer).transformed(
                    ax.transData.inverted()
                )

                # Coordinates of the left and right edges horizontally centered
                left_x, right_x = bbox.x0, bbox.x1
                mid_y = (bbox.y0 + bbox.y1) / 2.0

                # Pick the edge that is closest to the target point
                if abs(target_x - left_x) < abs(target_x - right_x):
                    start_x = left_x
                else:
                    start_x = right_x

                # Enforce minimum arrow length
                arrow_len = np.hypot(start_x - target_x, mid_y - target_y)
                min_arrow_len = 0.15

                if arrow_len < min_arrow_len:
                    vec_x, vec_y = start_x - target_x, mid_y - target_y
                    if arrow_len == 0:
                        vec_x, vec_y = 0.1, 0.1
                        arrow_len = np.hypot(vec_x, vec_y)

                    shift_magnitude = min_arrow_len - arrow_len
                    shift_x = (vec_x / arrow_len) * shift_magnitude
                    shift_y = (vec_y / arrow_len) * shift_magnitude

                    tx, ty = text_obj.get_position()
                    text_obj.set_position((tx + shift_x, ty + shift_y))

                    # Recompute bounding box after shift
                    bbox = text_obj.get_window_extent(renderer).transformed(
                        ax.transData.inverted()
                    )
                    left_x, right_x = bbox.x0, bbox.x1
                    mid_y = (bbox.y0 + bbox.y1) / 2.0

                    if abs(target_x - left_x) < abs(target_x - right_x):
                        start_x = left_x
                    else:
                        start_x = right_x

                ax.annotate(
                    "",
                    xy=(target_x, target_y),
                    xytext=(start_x, mid_y),
                    textcoords="data",
                    arrowprops=dict(arrowstyle="->", color="black", alpha=0.7, lw=0.8),
                    zorder=9,
                )

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title("Poincaré Disk Visualization")
    plt.axis("off")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {save_path}")
