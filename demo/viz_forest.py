import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
import matplotlib.pyplot as plt
from hyperbolic.data import generate_mock_forest


def main():
    print("Generating forest...")
    num_nodes, edges, node_depth, siblings, parents = generate_mock_forest(
        num_trees=10, branching_factor=3, depth=3
    )

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    plt.figure(figsize=(12, 8))
    # Use spring layout but adjust slightly for tree structures
    pos = nx.spring_layout(G, k=0.1, iterations=50, seed=42)

    depth_colors = [node_depth.get(n, 0) for n in G.nodes()]

    nx.draw(
        G,
        pos,
        node_size=30,
        node_color=depth_colors,
        cmap=plt.cm.viridis,
        edge_color="lightgray",
        width=0.5,
        alpha=0.9,
        arrows=False,
    )

    plt.title(
        "Mock Hierarchical Forest Structure\n(10 Trees, Depth 3, Branching Factor 3)"
    )

    # Create the demo directory if it doesn't exist just in case
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

    save_path = os.path.join(os.path.dirname(__file__), "forest_viz.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {save_path}")


if __name__ == "__main__":
    main()
