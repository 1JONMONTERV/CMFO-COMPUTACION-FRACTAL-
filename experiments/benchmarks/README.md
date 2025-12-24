# CMFO Benchmarks

This directory contains reproducible scripts to compare CMFO computational characteristics against standard architectures.

## `cmfo_vs_transformer.py`

This script simulates the algorithmic complexity scaling of the core mechanisms:
- **Transformer:** Self-Attention Mechanism ($O(N^2)$)
- **CMFO:** Fractal State Absorption ($O(N)$)

### Usage
```bash
python cmfo_vs_transformer.py
```

### Interpretation
You will observe that as Sequence Length ($N$) increases, the Transformer cost explodes quadratically, while CMFO scales linearly. This demonstrates the fundamental efficiency advantage of the fractal approach for long contexts.
