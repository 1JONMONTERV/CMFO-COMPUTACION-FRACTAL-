# CMFO: Fractal Computation Framework

> **Experimental Framework for Deterministic Fractal Computation**  
> _Note: This package is currently in Alpha (Experimental). Not intended for production use._

## 📌 Overview

**CMFO** (Computación Matriz Fractal Oscilada) is a computational framework that explores **deterministic operations over 7-dimensional fractal manifolds**. Unlike statistical models (like ML/AI) or binary logic, CMFO operates on structured geometric spaces where state evolution is continuous and analytically reversible.

This package provides the core **Engine** and **Operators** to simulate physics-consistent fractal logic.

### Key Capabilities
* **7D Manifold State**: Represent data as vectors on the $\varphi$-surface ($T^7_\varphi$).
* **Geometric Logic**: Reversible Boolean gates (AND, OR, XOR) via unitary rotations.
* **Deterministic Evolution**: Physics-based state evolution using Matrix-Gamma coupling ($v_{t+1} = \sin(M \cdot v_t)$).
* **High-Performance**: C++ Native backend for simulation loops.

---

## 🚀 Installation

```bash
pip install cmfo
```

### Requirements
* Python 3.9+
* Numpy >= 1.20
* **Visual C++ Build Tools** (Windows) or **GCC** (Linux/Mac) for native acceleration.

---

## 💻 Minimal API Usage

### 1. Basic State & Evolution
Create a state and evolve it using the Gamma Operator.

```python
from cmfo import T7Tensor, T7Matrix
import numpy as np

# Initialize a 7D State (Normalized to Phi)
initial_state = [1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0]
matrix = T7Matrix() # Identity Engine

# Evolve for 100 steps (Deterministic)
final_state = matrix.evolve_state(initial_state, steps=100)

print("Final State:", final_state)
```

### 2. Geometric Logic Gates
Perform reversible logic operations on fractal vectors.

```python
from cmfo import phi_and, phi_xor

# Inputs (Binary mapped to Fractal Phase)
a = 1  # Logic True
b = 0  # Logic False

# Compute geometric interference
res_and = phi_and(a, b)
res_xor = phi_xor(a, b)

print(f"AND(1,0) = {res_and}") # Close to -1.618 (Logic 0)
print(f"XOR(1,0) = {res_xor}") # Close to +1.618 (Logic 1)
```

---

## ⚠️ Stability & Roadmap

This project is semantically versioned:
* **0.x.y**: Experimental / Research (Current)
* **1.0.0**: Stable API

Detailed documentation and mathematical proofs can be found in the [GitHub Repository](https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-).

## 📄 License
**Apache 2.0** (Academic/Personal Use).  
For commercial licensing, please contact the author.
