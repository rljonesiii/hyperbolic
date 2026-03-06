# Hyperbolic Geometry Implementation Guide

This folder contains 17 documents, all focused on the advanced topic of using **Hyperbolic Embedding Spaces** (specifically the Lorentz/Hyperboloid model) for modeling hierarchical data, with a strong emphasis on scalable implementation using the **JAX** deep learning framework.

The documents, primarily authored or co-authored by Robert Jones, provide a comprehensive deep dive transitioning from theoretical mathematical justification to practical code optimization for building Hyperbolic Graph Attention Networks (HGATs).

## Reading Guide & Syllabus

For the best reading experience, it is highly recommended to read the documents in the following structured sequence, which builds from foundational geometry to advanced neural network architecture and optimization:

### Chapter 1: The Foundations of Hyperbolic Geometry
*Grasping the core mathematical concepts and understanding why we chose the Lorentz model.*

1. [**Motivating Hyperbolic Space**](./Motivation.md): Why standard Euclidean space fails for hierarchical data and how the exponential volume of hyperbolic space acts as a continuous analogue to discrete trees. Includes a conceptual guide to interpreting the Poincaré Ball visualization.

2. [**Lorentz**](./Lorentz.md): The primary justification for preferring the Lorentz (Hyperboloid) Model over the Poincaré Ball for deep learning optimization due to its superior numerical stability.

3. [**Mapping**](./Mapping.md): How to project points between the Lorentz and Poincaré spaces.

### Chapter 2: The JAX Backend & Optimization
*Explaining how these non-Euclidean geometries are translated into differentiable, numerically stable code.*

4. [**JAX Backend**](./JAXbackend.md): Leveraging JAX-specific operations (like `jit`, `vmap`, and custom gradients) to handle manifold operations efficiently.

5. [**Initializing**](./Initializing.md): The mathematics of Tangent Space Initialization for network weights.

6. [**Optimization**](./Optimization.md): Upgrading standard optimizer loops (like Adam) into Riemannian Adam by utilizing parallel transport to track momentum equations on curved surfaces.

### Chapter 3: Graph Attention Architecture (HGAT)
*Detailing the neural network layers built on top of the established geometric math.*

7. [**Attention**](./Attention.md): The mechanism for calculating attention alignment scores between target nodes and their neighbors by pulling them into the target's origin tangent space.

8. [**Aggregation**](./Aggregation.md): The process of applying attention weights dynamically in flat space and using the Exponential Map to retract the new embedding back onto the manifold.

9. [**Feature-Modulated Tangent Aggregation**](./FeatureModulatedTangentAggregation.md): Natively fusing flat Euclidean node and edge attributes with curved Hyperbolic geometry by utilizing the Tangent Space as a mixing desk to produce rich, contextualized embeddings.

### Chapter 4: Training, Scaling, and Evaluation
*Handling the logistics of actually training this model on large-scale mock or real datasets.*

10. [**Markov Blankets**](./MarkovBlanket.md): Subgraph extraction theory for defining node context and neighborhoods cleanly.

11. [**Negative Sampling**](./NegSampling.md): Hard negative sampling techniques (like picking structural siblings) to prevent training collapse and force fine-grained angular learning.

12. [**Training**](./Training.md): The implementation of Hyperbolic InfoNCE Loss.

13. [**Scaling**](./Scaling.md): Utilizing a Host-to-Device paging strategy to scale massive, billion-node graph networks on limited single-GPU memory.

14. [**Mocked Data Validation**](./MockedData.md): How dense Euclidean feature attributes are organically generated and injected over the Markov Blankets during our demo evaluation scripts to aggressively validate the underlying mathematics.

15. [**Evaluation**](./Evaluation.md): Analyzing outputs quantitatively (MRR) and qualitatively.

### Chapter 5: Code Integration
*Linking theory directly to the Python applications built.*

16. [**Implementation Walkthrough**](./Implementation_Walkthrough.md): A practical guide bridging theory with the final code in `demo/demo.py` and the `hyperbolic/` library, featuring run visualizations showcasing structural clustering.

17. [**YAML Interface**](./YAML_Interface.md): Documentation covering the `yaml_graph.py` parsing utilities mapping deeply nested string architectures into topological NetworkX graphs for easy manifold loading.
