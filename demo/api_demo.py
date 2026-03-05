import sys
import os

# Ensure we can import the hyperbolic module from the root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperbolic.api import HyperbolicEngine


from hyperbolic.interfaces.yaml_graph import load_yaml_to_graph


def main():
    """
    Demonstrates the end-to-end usage of the HyperbolicEngine API on a standard NetworkX graph.

    This script proves that the engine can:
    1. Parse a dynamic YAML-defined architectural topography (like an org chart or tech stack).
    2. Automatically extract node relationships (Markov Blankets) without explicit hierarchical depth tags.
    3. Initialize, train, and map these nodes onto the Lorentz Manifold using JAX.
    4. Persist the generated 3D hyperbolic coordinates back into the original NetworkX node attributes.
    5. Query the continuous space using Riemannian `lorentz_distance` to find structurally homologous nodes.
    """
    print("1. Parsing YAML Taxonomy into NetworkX Graph...")
    filepath = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests",
        "data",
        "test_generic_tree.yaml",
    )
    G = load_yaml_to_graph(filepath)
    print(f"   Loaded: {G.graph.get('title', 'Unknown Architecture')}")
    print(f"   Nodes Identified: {G.number_of_nodes()}")

    print("\n2. Initializing the Hyperbolic Engine...")
    # Initialize the engine. `spatial_dim=3` provides a richer manifold for complex embeddings.
    engine = HyperbolicEngine(spatial_dim=3, seed=42)

    print("\n3. Fitting NetworkX Graph to the Lorentz Manifold...")
    # The `fit` function executes the JAX training loop (Host-to-Device paging).
    # It dynamically calculates hard-negatives based on a 2-hop structural distance algorithm.
    G_embedded = engine.fit(G, epochs=600, batch_size=64, num_negs=5)

    print("\n4. Retrieving Persistent Embeddings from NetworkX...")
    # The output of `fit` is the identical NetworkX graph, but with a new `hyperbolic_embedding` attribute!
    example_node = "1.0"  # API Gateway
    print(f"   Original Node: {example_node} (API Gateway)")
    print(
        f"   Embedded Coordinate: {G_embedded.nodes[example_node]['hyperbolic_embedding']}"
    )

    print("\n5. Querying Structural Similarity via Markov Blankets...")
    # We can now query the space. `find_similar_nodes` calculates the true `lorentz_distance`
    # between the target and all other nodes to find the closest structural elements.

    target_1 = "1.0"  # API Gateway
    print(f"\n   Target Node: {target_1} (API Gateway)")
    similar_nodes_1 = engine.find_similar_nodes(target_1, top_k=3)
    for rank, result in enumerate(similar_nodes_1):
        print(
            f"   [{rank + 1}] {result['node']} (Lorentz Distance: {result['distance']:.4f})"
        )

    print(
        f"\n   Observe how '{target_1}' is structurally homologous to other top-level core components like Core Services (2.0)."
    )

    # Generate labels for visual legend dynamically based on their YAML components
    labels = []

    # Engine tracks index order in `idx_to_node`
    for i in range(len(engine.idx_to_node)):
        node_id = engine.idx_to_node[i]
        tag = G.nodes[node_id].get("component_type", "Standard Resource")
        if node_id == "root":
            labels.append("Root Meta")
        else:
            labels.append(tag.capitalize())

    # Annotate central overlapping nodes near the root/origin
    important_nodes_to_annotate = [
        "root",
        "1.0",
        "2.0",
        "3.0",
    ]

    # Visualize the projection of the Lorentz embeddings onto the unit disk
    save_path = os.path.join(os.path.dirname(__file__), "api_poincare_viz.png")
    engine.visualize(
        output_path=save_path,
        node_labels=labels,
        annotations=important_nodes_to_annotate,
    )
    print(f"\n6. Success! Visualization saved to {save_path}")
    print("\n   What to look for in the 'api_poincare_viz.png' image:")
    print(
        "   - You should see the Corporate Architectural Graph projected onto the Poincaré disk."
    )
    print(
        "   - The Root node sits near the stable center of the hyperbolic geometric space."
    )
    print("   - The children nodes radiantly branch off into dense structural pockets.")


if __name__ == "__main__":
    main()
