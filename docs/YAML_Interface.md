# The YAML Graph Interface

The `hyperbolic.interfaces.yaml_graph` module provides utilities for parsing nested YAML architectures into NetworkX directed graphs. This is particularly useful for modeling hierarchical taxonomies, organizational structures, and static component architectures as topological graphs.

## Core Features

- **Nested Key Preservation**: Reads unbounded layers of YAML arrays via the `contains` attribute.
- **Dynamic Identification**: If an explicit ID identifier is not provided (via `fulllevel` or `brlevel`), the parser automatically mints a UUID for the node.
- **Salt Collision Avoidance**: If multiple children provide the same identity parameters (e.g. duplicated levels in source data), the parser automatically appends an 8-hexadecimal salt to the duplicate node IDs to ensure NetworkX can map the distinct edges correctly instead of collapsing them.
- **Metadata Merging**: Top-level `metadata:` attributes are intercepted and merged directly into `G.graph` properties.

## Usage

```python
import networkx as nx
from hyperbolic.interfaces.yaml_graph import load_yaml_to_graph

# Parse a complex organizational YAML file
G = load_yaml_to_graph("data/my_architecture.yaml")

print(f"Parsed {G.number_of_nodes()} Nodes")
print(f"Graph Metadata: {G.graph}")

# Root children
root_connections = list(G.successors("root"))
```

## Example File Structure
The interface expects the following format:
```yaml
metadata:
  title: "Example Title"
rows:
  - name: "Parent Node"
    brlevel: "1.0"
    contains:
      - name: "Child Node"
        brlevel: "1.1"
```
