# CMFO v3.1-beta Release Notes
**Codename:** "The Sniper" (Auto-JIT Edition)
**Date:** 2025-12-14

## 🌟 Major Features

### 1. Native JIT Engine (Stateful)
- **Persistent CUDA Context**: Eliminates initialization overhead.
- **Auto-Caching**: MD5 hashing of generated kernels prevents redundant compilation.
- **Performance**: >1000 iter/sec validated throughput on consumer GPU.
- **Backend**: C++ Bridge (`nvrtc_bridge.cpp`) + NVRTC + CUDA Driver API.

### 2. Auto-JIT Core (Phase 4)
- **Lazy Evaluation**: `FractalVector7` now supports symbolic graph building.
- **Operator Overloading**: Natural Python syntax (`v * 0.5 + h`) generates CUDA kernels.
- **Dual Mode**: Seamless switching between CPU (Eager) and GPU (Lazy) execution.

### 3. Stability "Combat Proven"
- **Stress Tested**: 1000 consecutive execution cycles with 0 errors.
- **Installer**: `setup_alpha.py` provides tactical environment detection and compilation.

## 🛠️ Technical Details
- **Architecture**: Fused Kernel ("The Sniper") vs Multi-Kernel.
- **IR**: Semantic Graph for 7D Geometric Algebra.
- **Requirements**: NVIDIA GPU (CC 6.0+), CUDA Toolkit 11.0+.

## ⚠️ Known Issues
- `FractalVector7.compute()` requires manual context binding (Planned for v3.2).
- Native Bridge compiled for `compute_60` (Maxwell) for compatibility; recompile for Turing/Ampere optimization.

## 📦 Installation
```bash
python setup_alpha.py
python demo_v3_native.py
```
