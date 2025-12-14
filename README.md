# CMFO: Geometric Compute Engine (Experimental)

![Status: Experimental](https://img.shields.io/badge/Status-Experimental-orange)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE.txt)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](native/)

**CMFO** (Continuous Modal Fractal Oscillation) is an experimental computational framework designed to explore deterministic geometric operations on high-dimensional manifolds ($T^7_\varphi$).

Unlike probabilistic models (e.g., neural networks), CMFO attempts to model logical relationships as fixed geometric paths within a specific metric space.

> **Research Note:** This is an active research project. For the long-term theoretical vision (including aerospace/HFT applications), please see [VISION.md](VISION.md).

## Project Goal
To engineer a compute substrate that prioritizes **determinism** and **traceability** over statistical approximation.

## Core Features (v1.1.0)

*   **Fractal Operators:** Implementation of classic and modified fractal iterations in C.
*   **7D Vector Machine:** A custom data structure (`Tensor7`) optimized for operations in 7-dimensional space.
*   **Hybrid Runtime:** High-performance C kernels with Python bindings.
*   **Deterministic Output:** Same input + Same parameters = Identical output (Bit-exact).

## Repository Structure

The project uses a Monorepo structure:

```text
cmfo-universe/
├── core/                # Native C/C++ Kernels (The Engine)
│   ├── native/src/      # Reference implementation
│   └── native/cuda/     # GPU Acceleration (Experimental)
├── bindings/            # Language interfaces
│   └── python/          # `pip install cmfo` package
├── docs/                # Documentation
│   └── theory/          # Mathematical formalizations
├── experiments/         # Benchmarks and Notebooks
└── web/                 # Visualization tools
```

## Getting Started

### Prerequisites
*   Python 3.8+
*   GCC / Clang (for native extensions)
*   CUDA Toolkit (Optional, for GPU support)

### Installation (Python)

```bash
cd bindings/python
pip install .
```

### Basic Usage

```python
from cmfo.core import Tensor7

# Initialize a vector space
v1 = Tensor7([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Apply the Gamma operator (Single iteration)
v2 = v1.apply_gamma()

print(f"Input Norm: {v1.norm()}")
print(f"Output Norm: {v2.norm()}") # Should verify conservation
```

## Mathematical Foundations

The physics and math behind the engine are detailed in [docs/theory/mathematical_foundation.md](docs/theory/mathematical_foundation.md).
Key concepts:
*   **Space**: $\mathbb{R}^7$ referenced as $T^7_\varphi$.
*   **Metric**: Weighted Golden Ratio distance.
*   **Dynamics**: Discrete time evolution of state vectors.

## Contributing

We welcome contributions on:
1.  **Code Optimization**: SIMD/AVX implementation of kernels.
2.  **Mathematical Review**: Verification of theorems in `docs/theory`.
3.  **Language Bindings**: Rust or Go interfaces.

Please review our [CONTRIBUTING.md](CONTRIBUTING.md) guide.

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

---
**Disclaimer**: This software is currently **Concept / Pre-Alpha**. It is not yet certified for use in safety-critical systems.
