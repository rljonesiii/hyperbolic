import os

import matplotlib.pyplot as plt
import networkx as nx

from hyperbolic.interfaces.yaml_graph import load_yaml_to_graph

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(DIR_PATH, "data")


def load_and_plot(filename: str, title: str, output_path: str):
    filepath = os.path.join(DATA_DIR, filename)
    G = load_yaml_to_graph(filepath)

    # Use graphviz layout (dot) for a nice tree structure if pygraphviz is installed
    # Use PyGraphviz for structural tree layout
    try:
        from networkx.drawing.nx_agraph import graphviz_layout

        # 'dot' algorithm is specifically designed for directed graphs to look like hierarchical trees
        pos = graphviz_layout(G, prog="dot")
    except ImportError:
        print("Error: PyGraphviz not available. Cannot draw hierarchical tree.")
        return

    plt.figure(figsize=(14, 10))

    # Calculate depths from root
    depths = nx.single_source_shortest_path_length(G, "root")
    max_depth = max(depths.values()) if depths else 0

    # Generate a lighter color palette based on depth using a matplotlib colormap
    import matplotlib as mpl

    cmap = mpl.colormaps.get_cmap("Pastel1").resampled(max(1, max_depth + 1))

    node_sizes = [1500 if n == "root" else 1000 for n in G.nodes()]

    node_colors = []
    for n in G.nodes():
        # Get RGBA tuple and enforce 0.6 opacity
        r, g, b, _ = cmap(depths.get(n, 0))
        node_colors.append((r, g, b, 0.6))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.5,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        edge_color="#454545",
        width=2.0,
        node_size=node_sizes,  # This prevents arrows from occluding the nodes
    )

    # Format labels strictly
    labels = {}
    for node, data in G.nodes(data=True):
        if node == "root":
            labels[node] = "ROOT"
        else:
            # Fallbacks: fulllevel -> title -> node ID
            label_text = (
                data.get("fulllevel")
                or data.get("name")
                or data.get("title", str(node)[:6])
            )

            # Insert newlines to make long names fit nicely inside the circles
            if len(label_text) > 10 and " " in label_text:
                label_text = label_text.replace(" ", "\n", 1)

            labels[node] = label_text

    nx.draw_networkx_labels(
        G, pos, labels=labels, font_size=9, font_weight="bold", font_color="#333333"
    )

    plt.title(
        f"{title}\n{G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    load_and_plot(
        "test_file.yaml", "YAML Tree Structure", os.path.join(DATA_DIR, "yaml_tree.png")
    )
    load_and_plot(
        "test_generic_tree.yaml",
        "Generic Technical Architecture Graph",
        os.path.join(DATA_DIR, "generic_tree.png"),
    )


if __name__ == "__main__":
    main()
