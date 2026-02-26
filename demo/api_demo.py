import sys
import os

# Ensure we can import the hyperbolic module from the root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
from hyperbolic.api import HyperbolicEngine


def main():
    """
    Demonstrates the end-to-end usage of the HyperbolicEngine API on a standard NetworkX graph.

    This script proves that the engine can:
    1. Consume an arbitrary string-labeled NetworkX MultiDiGraph.
    2. Automatically extract node relationships (Markov Blankets) without explicit hierarchical depth tags.
    3. Initialize, train, and map these nodes onto the Lorentz Manifold using JAX.
    4. Persist the generated 3D hyperbolic coordinates back into the original NetworkX node attributes.
    5. Query the continuous space using Riemannian `lorentz_distance` to find structurally homologous nodes.
    """
    print("1. Constructing a mock knowledge graph in NetworkX...")
    # Initialize a standard NetworkX Multi-Directed Graph. The engine handles any string labels.
    G = nx.MultiDiGraph()

    # We create two distinct topological "clusters" to prove the engine can separate them purely
    # through contrastive learning on their connection topology (their Generalized Markov Blankets).

    # --- Topic A: Medical Entities ---
    G.add_edge("Patient_John", "Symptom_Cough", label="exhibits")
    G.add_edge("Patient_John", "Symptom_Fever", label="exhibits")
    G.add_edge("Patient_John", "Diagnosis_Flu", label="diagnosed_with")

    G.add_edge("Patient_Jane", "Symptom_Cough", label="exhibits")
    G.add_edge("Patient_Jane", "Symptom_Fatigue", label="exhibits")
    G.add_edge("Patient_Jane", "Diagnosis_Flu", label="diagnosed_with")

    G.add_edge("Patient_Bob", "Symptom_Fever", label="exhibits")
    G.add_edge("Patient_Bob", "Symptom_Rash", label="exhibits")
    G.add_edge("Patient_Bob", "Diagnosis_Measles", label="diagnosed_with")

    G.add_edge("Disease_Flu", "Symptom_Cough", label="causes")
    G.add_edge("Disease_Flu", "Symptom_Fever", label="causes")
    G.add_edge("Disease_Flu", "Symptom_Fatigue", label="causes")

    G.add_edge("Disease_Measles", "Symptom_Fever", label="causes")
    G.add_edge("Disease_Measles", "Symptom_Rash", label="causes")

    # --- Topic B: Corporate Entities ---
    G.add_edge("Company_TechCorp", "Role_Engineer", label="employs")
    G.add_edge("Company_TechCorp", "Role_DataScientist", label="employs")
    G.add_edge("Company_TechCorp", "CEO_Alice", label="employs")

    G.add_edge("Company_HealthInc", "Role_Doctor", label="employs")
    G.add_edge("Company_HealthInc", "Role_DataScientist", label="employs")
    G.add_edge("Company_HealthInc", "CEO_Charlie", label="employs")

    G.add_edge("Person_Eve", "Role_Engineer", label="has_role")
    G.add_edge("Person_Eve", "Company_TechCorp", label="works_at")

    G.add_edge("Person_Mallory", "Role_DataScientist", label="has_role")
    G.add_edge("Person_Mallory", "Company_TechCorp", label="works_at")

    G.add_edge("Person_Trent", "Role_DataScientist", label="has_role")
    G.add_edge("Person_Trent", "Company_HealthInc", label="works_at")

    print("\n2. Initializing the Hyperbolic Engine...")
    # Initialize the engine. `spatial_dim=3` provides a richer manifold for complex embeddings.
    engine = HyperbolicEngine(spatial_dim=3, seed=42)

    print("\n3. Fitting NetworkX Graph to the Lorentz Manifold...")
    # The `fit` function executes the JAX training loop (Host-to-Device paging).
    # It dynamically calculates hard-negatives based on a 2-hop structural distance algorithm.
    G_embedded = engine.fit(G, epochs=600, batch_size=64, num_negs=5)

    print("\n4. Retrieving Persistent Embeddings from NetworkX...")
    # The output of `fit` is the identical NetworkX graph, but with a new `hyperbolic_embedding` attribute!
    example_node = "Patient_John"
    print(f"   Original Node: {example_node}")
    print(
        f"   Embedded Coordinate: {G_embedded.nodes[example_node]['hyperbolic_embedding']}"
    )

    print("\n5. Querying Structural Similarity via Markov Blankets...")
    # We can now query the space. `find_similar_nodes` calculates the true `lorentz_distance`
    # between the target and all other nodes to find the closest structural elements.

    target_1 = "Patient_John"
    print(f"\n   Target Node: {target_1}")
    similar_nodes_1 = engine.find_similar_nodes(target_1, top_k=3)
    for rank, result in enumerate(similar_nodes_1):
        print(
            f"   [{rank + 1}] {result['node']} (Lorentz Distance: {result['distance']:.4f})"
        )

    print(
        f"\n   Observe how '{target_1}' is similar to 'Disease_Flu' and 'Patient_Jane'."
    )
    print("   They share overlapping symptom nodes (Cough, Fever, Fatigue).")

    target_2 = "Role_DataScientist"
    print(f"\n   Target Node: {target_2}")
    similar_nodes_2 = engine.find_similar_nodes(target_2, top_k=4)
    for rank, result in enumerate(similar_nodes_2):
        print(
            f"   [{rank + 1}] {result['node']} (Lorentz Distance: {result['distance']:.4f})"
        )

    print(
        f"\n   Observe how '{target_2}' is structurally linked to TechCorp, HealthInc, Mallory, and Trent."
    )

    # Generate labels for visual legend
    labels = []
    medical_nodes = [
        "Patient_John",
        "Patient_Jane",
        "Patient_Bob",
        "Symptom_Cough",
        "Symptom_Fever",
        "Symptom_Fatigue",
        "Symptom_Rash",
        "Diagnosis_Flu",
        "Diagnosis_Measles",
        "Disease_Flu",
        "Disease_Measles",
    ]
    corporate_nodes = [
        "Company_TechCorp",
        "Company_HealthInc",
        "Role_Engineer",
        "Role_DataScientist",
        "Role_Doctor",
        "CEO_Alice",
        "CEO_Charlie",
        "Person_Eve",
        "Person_Mallory",
        "Person_Trent",
    ]

    # Engine tracks index order in `idx_to_node`
    for i in range(len(engine.idx_to_node)):
        node = engine.idx_to_node[i]
        if node in medical_nodes:
            labels.append("Medical Topic")
        elif node in corporate_nodes:
            labels.append("Corporate Topic")
        else:
            labels.append("Unknown")

    # Annotate central overlapping nodes near the root/origin
    important_nodes_to_annotate = [
        "Patient_John",
        "Patient_Jane",
        "Disease_Flu",
        "Role_DataScientist",
        "Company_TechCorp",
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
        "   - You should see two distinct regional clusters of points pushed toward the edges of the disk."
    )
    print(
        "   - The Topic A (Medical) and Topic B (Corporate) clusters are repelled from each other because they belong to disjoint graphs."
    )
    print(
        "   - Within the Medical cluster, 'Patient_1' and 'Disease_Flu' should be located extremely close together (or perfectly overlapping) due to their identical structural context."
    )
    print(
        "   - This visually proves that hyperbolic space naturally organizes hierarchical and disjoint tabular data simultaneously."
    )


if __name__ == "__main__":
    main()
