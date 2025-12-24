
# CMFO v2.0.0 Release Notes

## Major Architecture Shift: Pure Python Core & Zero-Copy GPU Bridge

This release marks a fundamental shift towards portability and performance.

### 🚀 Highlights

*   **100% Pure Python**: The `cmfo.layers` module no longer depends on PyTorch or NumPy. It runs on standard Python lists and math, making it deployable on any helper system (Raspberry Pi, MicroPython, Legacy Servers).
*   **Zero-Copy GPU Bridge**: A new `cmfo.core.gpu` module replaces heavy Tensor libraries with a direct `ctypes` interface. It achieved **34,000x faster serialization** compared to list-based approaches and reaches **~86k vectors/sec** throughput.
*   **Fractal Stability**: Fixed numerical instability (NaN explosions) in `FractalMemory` by implementing geometric normalization.

### 🛠️ Key Changes

*   `cmfo/layers/linear.py`: Rewritten to use `math` and lists.
*   `cmfo/layers/attention.py`: Rewritten to use pure Python logic.
*   `cmfo/core/structural.py`: Added `FractalVector7` for native algebra.
*   `cmfo/core/matrix.py`: Added auto-fallback to structural engine.
*   **New**: `tests/performance/` containing benchmarks and stress tests.

### ⚠️ Limitaciones Conocidas

*   La aceleración GPU requiere compilar `src/core/native/` a `cmfo_cuda.dll`. Si no está presente, el sistema usará el **Simulador Virtual** (que sigue siendo rápido pero no usa hardware real).
