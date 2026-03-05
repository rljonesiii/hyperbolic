"""
Module for converting nested YAML table data into a NetworkX directed graph.

This preserves the hierarchical structure (e.g., subrows) found in the YAML representation.
"""

import typing
import uuid

import networkx as nx
import yaml


def load_yaml_to_graph(filepath: str) -> nx.DiGraph:
    """
    Reads a nested YAML file and converts it to a purely generic NetworkX directed graph.
    The graph maintains the hierarchical structure via directed edges from parent to child.

    Args:
        filepath: Path to the YAML file.

    Returns:
        A NetworkX DiGraph representing the hierarchical data.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Root YAML object must be a dictionary.")

    G = nx.DiGraph()

    # We assume 'metadata' holds top-level graph attributes
    metadata = data.get("metadata", {})
    G.graph.update(metadata)

    # Use a root node to tie the top-level rows together.
    # If the metadata has a title, we use it, else generic 'root'
    root_id = "root"
    G.add_node(root_id, **metadata)

    def _traverse_and_add(nodes: typing.Any, parent_id: str) -> None:
        if not isinstance(nodes, list):
            return

        for node_data in nodes:
            if not isinstance(node_data, dict):
                continue

            # Extract features excluding children ('contains')
            attrs: typing.Dict[str, typing.Any] = {
                k: v for k, v in node_data.items() if k != "contains"
            }

            # Determine a unique ID for the node.
            # Try to use 'fulllevel', fallback to 'brlevel', or generate UUID.
            node_id_val = attrs.get("fulllevel") or attrs.get("brlevel")
            base_node_id = str(node_id_val) if node_id_val else str(uuid.uuid4())

            node_id = base_node_id

            # Append a random salt if the node already exists to avoid collisions.
            if G.has_node(node_id):
                node_id = f"{node_id}_{uuid.uuid4().hex[:8]}"

            G.add_node(node_id, **attrs)
            G.add_edge(parent_id, node_id)

            # Recurse and link children to this node
            children = node_data.get("contains", [])
            _traverse_and_add(children, node_id)

    # Begin traversal typically at 'rows' key
    rows = data.get("rows", [])
    _traverse_and_add(rows, root_id)

    return G
