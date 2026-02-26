# Hyperbolic Graph Neural Networks in JAX

This repository contains a full implementation of Hyperbolic Graph Attention Networks (HGATs) using the **Lorentz (Hyperboloid) Model** in JAX. It's designed for modeling hierarchical data structures (like knowledge graphs) and features a Host-to-Device paging strategy for scaling on single-GPU hardware.

- **Docs:** Read the comprehensive [Documentation Directory](./docs/README.md) for deep dives into Riemannian optimization, the Lorentz model, hyperbolic InfoNCE loss, and attention aggregation.
- **Walkthrough:** See the [Implementation Walkthrough](./docs/Implementation_Walkthrough.md) for a guide bridging the math with the Python codebase, along with training visualizations showcasing the clustered hierarchy.

## Installation

This project utilizes [`uv`](https://docs.astral.sh/uv/) for dependency management and requires **Python 3.12+**. 

The main dependencies are:
- `jax` and `jaxlib` for hardware-accelerated, differentiable tensor operations.
- `matplotlib` for stereographic projection visualizations to the Poincaré disk.
- `scipy` for logging and stable metric computations.

### Setup Instructions

1. Ensure you have `uv` installed. You can install it via the official standalone script:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   **macOS Alternative (Homebrew):** If you prefer using Homebrew for package management on Mac, you can install it via:
   ```bash
   brew install uv
   ```
2. Clone the repository and navigate into it:
   ```bash
   git clone <repository_url>
   cd hyperbolic
   ```
3. Sync the environment and install dependencies using `uv`:
   ```bash
   uv sync
   ```

## Running the Demo

The repository includes an end-to-end demonstration script that synthesizes hierarchical data, initializes the network, trains on a single GPU using host-to-device paging, and renders the before/after visualizations.

To run the full pipeline, navigate to the `demo` directory and run the script:

```bash
cd demo
uv run demo.py
```

This will output the training logs and save the final clustered embedding representation to `demo/poincare_viz_final.png`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

* **Robert Jones** - [rljonesiii](https://github.com/rljonesiii)

## Citation

If you use this library in your research, please cite:

```bibtex
@software{jones2026hyperbolic,
  author = {Robert Jones},
  title = {Hyperbolic Graph Neural Networks in JAX},
  url = {https://github.com/rljonesiii/hyperbolic},
  year = {2026},
}
```
