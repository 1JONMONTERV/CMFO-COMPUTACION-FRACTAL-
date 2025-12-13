# Theory: CMFO CUDA Implementation

## 1. Overview

The GPU acceleration layer of CMFO-UNIVERSE is designed to handle the massive parallelism required for high-dimensional fractal dynamics. The current implementation targets **NVIDIA Ampere (SM86)** architectures.

## 2. Kernel: `cmfo_dynamics_gpu`

The primary kernel located in `cuda/theta_cmfo_kernel.cu` implements a **Phase 1** dynamics model: **Independent Mode Evolution**.

### 2.1 Mathematical Model

Each thread $t$ corresponds to a spectral component (or dimension) of the T7 vector. The evolution is governed by a frequency $\omega_t$ derived from the Golden Ratio $\phi$:

$$
\omega_t = \phi \cdot (t + 1) \quad \text{where } t \in \{0, ..., 6\}
$$

The state variable $\theta_t$ evolves as:

$$
\frac{d\theta_t}{dt} = \omega_t \implies \theta_t(\tau) = \theta_t(0) + \omega_t \cdot \tau
$$

Periodic boundary conditions are applied: $\theta_t \in [0, 2\pi)$.

### 2.2 Implementation Details

```cpp
// cuda/theta_cmfo_kernel.cu
double omega = 1.61803398875 * (tid + 1);
theta += omega * dt;
if (theta > 2 * M_PI) theta -= 2 * M_PI;
```

### 2.3 Limitations & Roadmap

This generic evolution serves as a **throughput test** for the memory coherence of the T7 structure on the GPU.

**Future Phase 2 (Planned):**
- Implement **Cross-Coupling**: The evolution of $\theta_i$ should depend on $\theta_j$ (Interaction Tensor).
- $$ \dot{\theta}_i = \omega_i + \sum_{j,k} \Gamma_{ijk} \sin(\theta_j - \theta_k) $$
- This requires shared memory optimization to load the full state vector for all threads in a block.

## 3. Performance Considerations

- **Coalesced Access**: The array `d_theta_in` is accessed linearly `tid`, ensuring optimal memory bandwidth.
- **Precision**: Uses `double` precision (FP64), which has a performance cost on consumer GPUs but is required for T7 numerical stability.
