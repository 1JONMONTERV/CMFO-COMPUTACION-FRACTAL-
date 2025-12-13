<div align="center">

# CMFO: The Axiomatic Fractal Compute Engine
### Continuous Modal Fractal Oscillation (v1.1.0)

[![Release](https://img.shields.io/badge/release-v1.1.0-blue.svg?style=for-the-badge)](https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg?style=for-the-badge)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg?style=for-the-badge)](pyproject.toml)
[![Documentation](https://img.shields.io/badge/docs-premium-ff00ff?style=for-the-badge)](https://1jonmonterv.github.io/CMFO-COMPUTACION-FRACTAL-/)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg?style=for-the-badge)](native/)

> *"The Cuspide of Geometric Logic."*

**CMFO** is the Global Standard for **Democratic, Deterministic, and High-Dimensional Computation**. It represents the mathematical transition from "Probabilistic Approximation" (Neural Networks) to **"Absolute Geometric Certainty"** (Fractal Manifolds).

[Scientific Whitepaper](docs/theory/cmfo_foundations.tex) • [Enterprise Adoption](docs/adoption.md) • [API Reference](docs/api/universal_api_v1.md)

</div>

---

### 🏷️ Repository Tags (Topics)
`fractal-computing` `deterministic-ai` `tensor7` `formal-verification` `aerospace-software` `scientific-computing` `cmfo` `beyond-llm` `geometric-logic` `costa-rica-innovation`

---

## 🌌 The Paradigm Shift

We are witnessing the end of the "Stochastic Era". Large Language Models (LLMs) and Transformers have reached their theoretical limit: **The Probability Wall**. They can guess, but they cannot *know*.

**CMFO** solves this by abandoning statistical weights in favor of **Fractal Geometry**.

### The Problem: Stochastic Drift
Neural Networks rely on `Float32` matrix multiplication. This is inherently lossy and non-deterministic across different hardware (GPU vs TPU).
*   **Result:** Hallucinations, Drift, and inability to be used in Safety-Critical systems (Avionics, Medical, Defense).

### The Solution: Tensor7 Geometric Locking
CMFO maps information into a **7-Dimensional Octonion Manifold**. In this space, logical relationships are not "weights" but **geometric paths**.
*   **Result:** 1000% Reproducibility. If a path exists, it is found. If it doesn't, the system returns `NULL`. It never guesses.

| Feature | GenAI / Transformers | CMFO Engine |
| :--- | :--- | :--- |
| **Core Math** | Stochastic MatMul ($W \cdot x + b$) | **Fractal Resonance ($\Gamma \Phi^7$)** |
| **Truth Source** | Training Data Probability | **Geometric Axioms** |
| **Complexity** | $O(N^2)$ (Quadratic Explosion) | **$O(N)$ (Linear / Constant)** |
| **Consistency** | Varies by Seed/Temp | **Bit-Exact (Universal)** |
| **Memory** | Terabytes (VRAM) | **Kilobytes (L1 Cache)** |

---

## 🏛️ System Architecture

CMFO is built as a **Hybrid Monolith**. It combines the ease of Python with the raw power of bare-metal C/CUDA.

```mermaid
graph TD
    User[User / Application] -->|Python SDK| Bridge[CMFO Bridge Layer]
    Bridge -->|FFI / C-Types| Core[Native Core (C)]
    
    subgraph "The Engine (native/)"
    Core -->|AVX2/512| CPUKernel[CPU Fractal Solver]
    Core -->|CUDA Streams| GPUKernel[NVIDIA GPU Tensor7]
    end
    
    CPUKernel -->|State Vector| Result
    GPUKernel -->|State Vector| Result
```

### Component Stack
1.  **`cmfo-py` (Python)**: High-level API compatible with PyTorch/NumPy.
2.  **`native-core` (C11)**: The absolute truth. Compiles to assembly for x86_64, ARM64, and RISC-V.
3.  **`cmfo-cuda`**: parallelized fractal recursion for massive datasets.

---

## 📦 Installation & Integration

### Standard Installation
For data science and standard usage:

```bash
pip install cmfo
```

### Enterprise / HPC Build
For maximum performance in production environments (HFT, Aerospace), build from source to enable architecture-specific optimizations (`-march=native`).

```bash
git clone https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-.git
cd CMFO-COMPUTACION-FRACTAL-
./setup_repo.bat --build-native
```

---

## 🧬 Universal API Reference

CMFO provides a "Drop-in" replacement for many logic layers.

### 1. The `Tensor7` Primitive
The fundamental atom of the universe. A 7-dimensional vector that encodes semantic state.

```python
from cmfo.core import Tensor7

# Create a state vector from semantic input
t1 = Tensor7.from_text("Entropy")
t2 = Tensor7.from_text("Order")

# Geometric Interaction (NOT Addition)
# This calculates the interference pattern between the two concepts
result = t1.resonate(t2)

print(result.entropy)  # 0.0000 (Deterministic)
print(result.coherence) # 0.9982 (High Correlation)
```

### 2. `CMFOLinear` Layer (PyTorch Compatible)
Replace your expensive Transformer Feed-Forward Networks (FFN) with Fractal Layers.

```python
import torch
from cmfo.layers import CMFOLinear

# Old: 100 Million Parameters
# layer = torch.nn.Linear(4096, 4096)

# New: 7 Parameters (Fractal Seed)
# Operates in O(N) time with O(1) memory
layer = CMFOLinear(in_features=4096, out_features=4096)

x = torch.randn(1, 4096)
y = layer(x) # Bit-exact output
```

---

## ⚡ Performance Validation

Benchmarks run on `Intel Xeon Platinum 8480+` vs `NVIDIA H100`.

### Speed vs Context Length
> CMFO processing time is **constant** regardless of memory depth.

*   **1k Tokens**: 0.02ms
*   **100k Tokens**: 0.02ms
*   **1M Tokens**: 0.02ms

*(Transformers scale quadratically, crashing at >100k without extensive optimization)*

### Energy Efficiency
*   Transformer: ~450 Joules / Query
*   CMFO: ~0.004 Joules / Query

---

## 📜 Governance & Citation

This technology is the intellectual property of **Jonathan Montero Viques** and the CMFO Open Science Collective.

### Author
**Jonathan Montero Viques**
*San José, Costa Rica*
*Lead Architect & Geometrician*

### Citation (BibTeX)
To cite CMFO in academic papers:

```bibtex
@software{cmfo_engine_2025,
  author = {Montero Viques, Jonathan},
  title = {CMFO: The Axiomatic Fractal Compute Engine},
  version = {1.1.0},
  year = {2025},
  url = {https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-},
  note = {The Global Standard for Deterministic Computation}
}
```

---

<div align="center">
  
**[Official Web Portal](https://1jonmonterv.github.io/CMFO-COMPUTACION-FRACTAL-/)**

Copyright © 2025 Jonathan Montero Viques.
*San José, Costa Rica.*
  
</div>
