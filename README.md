<div align="center">
  
# CMFO: Computational Matter Fractal Ontology

[![Release](https://img.shields.io/badge/release-v1.1.0-blue.svg)](https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![Documentation](https://img.shields.io/badge/docs-premium-ff00ff)](https://1jonmonterv.github.io/CMFO-COMPUTACION-FRACTAL-/)

### The First Deterministic Fractal Compute Engine.
*Beyond Probabilistic AI. Beyond Chaos. Pure Geometric Truth.*

[Quick Start](#-quick-start) • [Features](#-key-features) • [Benchmarks](#-performance-benchmarks) • [Documentation](#-documentation) • [Citation](#-citation)

</div>
  
### 🏷️ Repository Tags (Topics)
`fractal-computing` `deterministic-ai` `tensor7` `formal-verification` `aerospace-software` `scientific-computing` `cmfo` `beyond-llm` `geometric-logic` `costa-rica-innovation`

---

## ⚡ What is CMFO?

**CMFO** (Continuous Modal Fractal Oscillation) is the **Global Standard for Deterministic High-Dimensional Computation**. 

> *"The Cuspide of Geometric Logic."*

Developed by **Jonathan Montero Viques**, CMFO represents a paradigm shift from probabilistic approximation to **absolute geometric certainty**. It is designed to be the backbone of **Aerospace, Financial, and Scientific** systems where error is unacceptable. 

Unlike Nueral Networks which rely on probabilistic matrix multiplications (MatMul) and "best guess" token prediction, CMFO uses **7-Dimensional Fractal Geometry** to map information into a deterministic state space.

| Feature | Large Language Models (LLMs) | CMFO Engine |
| :--- | :--- | :--- |
| **Core Math** | Float32 Matrix Multiplication | **7D Fractal Resonance (Tensor7)** |
| **Determinism** | Statistical / Probabilistic | **Absolute / Bit-Exact** |
| **Complexity** | $O(N^2)$ (Quadratic) | **$O(N)$ (Linear)** |
| **Drift** | Subject to Hallucination | **Converges to Null Attractor** |
| **Target** | Creative Generation | **Critical Systems & Logic** |

---

## 🚀 Key Features

### 1. Drop-In Efficiency
Replace heavy `torch.nn.Linear` layers with `CMFOLinear` to instantly reduce parameter count by **99%** while retaining geometric structural memory.

### 2. Universal Core
A unified **C/C++ Kernel** (`native/`) ensures that math calculated in Python, C, Rust, or on GPU is identical down to the last bit.

### 3. Safety Critical
Designed for **Aerospace and Defense**. When CMFO doesn't know an answer, it doesn't "hallucinate"; it geometrically proves that no path exists (Null Return).

---

## 📦 Installation

```bash
# Install via pip
pip install cmfo
```

---

## 🛠️ Quick Start

### Python SDK
Calculate the unique **Fractal Checksum** of any text. This is an $O(N)$ operation that is collision-resistant and semantic.

```python
import cmfo
from cmfo.bridge import text_to_tensor
from cmfo.core.api import tensor7

# 1. Semantic Hashing
# Convert text to a unique 7D geometric signature
text = "The universe is built on geometric truth."
signature = text_to_tensor(text)

print(f"Text Signature: {signature}")
# Output: [0.124 0.982 -0.331 ...] (Deterministic)

# 2. Fractal Resonance
# Combine two states geometrically (not simple addition)
state_a = tensor7(1.0, 0.5)
state_b = tensor7(0.5, 0.2)
# Result is the geometric interference pattern
print(f"Resonance: {state_a + state_b}")
```

### CLI Tools
CMFO comes with a power suite of command-line tools.

```bash
# 1. Visualize the fractal attractor of a phrase
cmfo visualize "Entropy is not inevitable"

# 2. Start the HTTP API Microservice
cmfo serve --port 8000
```

### Fractal Compression (`cmfo-zip`)
Store semantic meaning in 56 bytes.
```python
import cmfo.compression

data = "A very long repeated string..." * 1000
compressed = cmfo.compression.compress_text(data)

# You can read the semantic header WITHOUT decompressing the payload!
header = cmfo.compression.get_semantic_header(compressed)
```

---

## 📊 Performance Benchmarks

CMFO scales **linearly** $O(N)$ with context length, whereas Transformers scale quadratically $O(N^2)$.

> **Benchmark Result (N=5000 tokens):**
> *   Transformer Attention: ~25.0s
> *   CMFO Resonance: **0.6s (41.3x Faster)**

*See `benchmarks/cmfo_vs_transformer.py` for reproduction.*

---

## ❓ FAQ

**Q: Is this an LLM?**
**A:** No. It is a Geometric Kernel. It complements LLMs by providing deterministic logic and memory, but it does not "predict tokens" probabilistically.

**Q: Does it hallucinate?**
**A:** No. If a geometric path does not exist, the system returns a Null State. It cannot invent information.

**Q: Why 7 Dimensions?**
**A:** Based on Octonion Fano Plane structures, 7 is the minimal dimensionality required for self-referential closure without singularity.

*Read the full [FAQ.md](FAQ.md).*

---

## 📚 Documentation

The official documentation is hosted on our **[Premium Web Portal](https://1jonmonterv.github.io/CMFO-COMPUTACION-FRACTAL-/)**.

*   [Scientific Foundations (LaTeX)](docs/theory/cmfo_foundations.tex)
*   [Business Case & Adoption](docs/adoption.md)
*   [Monorepo Architecture](SCOPE.md)

---

## 🔖 Citation

If you use CMFO in your research, please cite:

```bibtex
@software{cmfo2025,
  author = {Jonnathan Montero},
  title = {CMFO: continuous Modal Fractal Oscillation Engine},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-},
  version = {1.1.0}
}
```

---

<div align="center">
  <sub>MIT License • Built with Geometric Precision.</sub>
</div>
