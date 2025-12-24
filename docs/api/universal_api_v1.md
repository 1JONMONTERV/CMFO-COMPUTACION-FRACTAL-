# CMFO Universal API Standard v1.0

**Status:** Official Specification
**Target:** ALL Host Languages (Python, Rust, C, Go, Julia)

## 1. Core Principles
1.  **Immutability:** All functions must be pure. Input tensors are never modified in place unless explicitly suffixed with `_inplace`.
2.  **Determinism:** Identical floating-point inputs must yield bit-exact identical outputs across all architectures (IEEE 754 compliance required).
3.  **No Exceptions:** Errors are handled via return codes or Null Tensors (all-zero), never by throwing exceptions.

## 2. The Golden Functions
Every CMFO implementation MUST expose these exact symbols.

### 2.1 `tensor7_reduce`
Projects a high-dimensional vector into the fundamental 7D manifold.
- **Signature:** `tensor7_reduce(input: Vector[N]) -> Vector[7]`
- **Logic:** $\mathcal{R}(v)_i = \sum_{k=0}^{\lfloor N/7 \rfloor} v_{7k+i}$

### 2.2 `tensor7_expand`
Projects a 7D seed vector back into high-dimensional space via fractal recursion.
- **Signature:** `tensor7_expand(seed: Vector[7], target_dim: int) -> Vector[target_dim]`
- **Logic:** $v_j = \text{seed}_{j \pmod 7} \cdot \varphi^{\lfloor j/7 \rfloor}$

### 2.3 `tensor7_rotate`
Rotates the vector within the Fano plane structure.
- **Signature:** `tensor7_rotate(v: Vector[7]) -> Vector[7]`
- **Logic:** Permutation $\sigma = (1, 2, 4)(3, 6, 5)(0)$

### 2.4 `cmfo_forward`
The primary inference step. Absorbs input state into the fractal attractor.
- **Signature:** `cmfo_forward(state: Tensor, input: Tensor) -> Tensor`
- **Equation:** $S_{t+1} = \frac{S_t \otimes_7 I_t + \varphi}{1 + \varphi}$

## 3. Data Types
- **Float32 (Single):** Standard for inference.
- **Float64 (Double):** Standard for calibration and gold-standard verification.
