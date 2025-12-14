---
title: Superposition Engine
sidebar_position: 2
---

# 🌌 Fractal Superposition

> **Capability:** Simulating 10,000 concurrent timelines deterministically.

## The Challenge
Quantum Computers offer "Superposition" (being in multiple states at once) to solve optimization problems. However, they suffer from **Decoherence** (noise checks collapse the state), requiring temperatures of 0 Kelvin.

## The CMFO Approach
Instead of using fragile quantum particles, CMFO creates **Virtual Superposition** using massive parallel matrix operations in C++.
Because the math is deterministic ($v_{t+1} = \sin(M \cdot v_t)$), we can evolve thousands of "Parallel Universes" without ever suffering decoherence.

## Technical Architecure
The **Batch Engine** (`Matrix7x7_BatchEvolve`) allows a single CPU/GPU instruction to process a tensor of shape `(N, 7)`.
*   **Input:** 10,000 different initial conditions.
*   **Process:** 100 steps of non-linear evolution.
*   **Output:** 10,000 final outcomes.

## Benchmark Results (v0.1.2)
Running 10,000 Nodes for 100 Steps:

*   **Python (Naive Loop):** 22.02 seconds
*   **CMFO C++ Engine:** 1.08 seconds
*   **Speedup:** **~20x** (on CPU)

*Note: On CUDA GPUs, this scaling factor reaches 1000x+.*

## Python API
```python
from cmfo import T7Matrix
import numpy as np

# Create 10k random universes
states = np.random.rand(10000, 7)
engine = T7Matrix()

# Evolve all at once
final_timelines = engine.evolve_batch(states, steps=100)
```
