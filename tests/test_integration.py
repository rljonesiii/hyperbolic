import os
import networkx as nx
import numpy as np

from hyperbolic.api import HyperbolicEngine


def test_api_end_to_end():
    """
    Tests the HyperbolicEngine API end-to-end to ensure
    no breaking changes are introduced to the core user flow.
    Combines fit, visualization, and querying.
    """
    # 1. Create a minimal mock graph (a simple tree to encourage hierarchy)
    G = nx.Graph()
    G.add_edges_from(
        [
            ("root", "child1"),
            ("root", "child2"),
            ("child1", "leaf1"),
            ("child1", "leaf2"),
            ("child2", "leaf3"),
        ]
    )

    # 2. Add some dummy node features to trigger F_n path
    for node in G.nodes():
        G.nodes[node]["feature"] = np.random.rand(4)

    # 3. Initialize engine
    engine = HyperbolicEngine(spatial_dim=2, seed=42)

    # 4. Run fit for 2 epochs (dry run)
    G_trained = engine.fit(G, epochs=2, batch_size=2, num_negs=2)

    # 5. Verify embeddings were attached
    for node in G_trained.nodes():
        assert "hyperbolic_embedding" in G_trained.nodes[node]
        emb = G_trained.nodes[node]["hyperbolic_embedding"]
        assert emb.shape == (3,)  # 1 time + 2 spatial dims

        # Time-like coordinate should be >= 1.0
        assert emb[0] >= 1.0

    # 6. Test similarity querying
    similar = engine.find_similar_nodes("root", top_k=2)
    assert len(similar) == 2
    assert "node" in similar[0]
    assert "distance" in similar[0]

    # 7. Test visualization
    tmp_path = "test_hyperbolic_viz.png"
    engine.visualize(
        output_path=tmp_path,
        node_labels=["root", "child1", "child2", "leaf1", "leaf2", "leaf3"],
        show_edges=True,
        annotations=["root"],
    )
    assert os.path.exists(tmp_path)
    os.remove(tmp_path)


def test_compute_dissimilarity():
    """
    Verifies compute_dissimilarity API functionality.
    """
    G = nx.cycle_graph(4)
    engine = HyperbolicEngine(spatial_dim=2, seed=42)
    engine.fit(G, epochs=1, batch_size=4, num_negs=1)

    dist = engine.compute_dissimilarity(0, 1)
    assert isinstance(dist, float)
    assert dist >= 0.0
