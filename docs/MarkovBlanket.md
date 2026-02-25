
# Markov Blanket

In a knowledge graph structured as a forest of trees (a collection of separate, hierarchical networks of entities and their relationships), 
the **Markov Blanket** of a specific target node represents its immediate structural neighborhood.
This blanket is composed of three distinct sets of nodes:
    1. The immediate parent node (if one exists).
    2. All direct child nodes.
    3. All sibling nodes (nodes that share the same parent).

Conceptually, these three sets capture all the information in the graph that is directly relevant to the target node. The parent provides the hierarchical context (the "what kind of thing" it is), the children represent its specific instances or subcategories (the "what it contains"), and the siblings represent alternative entities at the same level of abstraction (the "what else is like it").

From a probabilistic perspective, the Markov Blanket is the minimal set of nodes that renders the target node conditionally independent of all other nodes in the graph. In simpler terms, once you know the state of the nodes in the Markov Blanket, learning anything about the rest of the graph provides no additional information about the target node.