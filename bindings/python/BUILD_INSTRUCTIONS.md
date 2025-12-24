# Building CMFO Native Extension

To unlock the High-Performance C++ Engine ($>100x$ speedup), you must compile the native extension.

## Prerequisites (Windows)
1. **Visual Studio Build Tools**:
   - Install "Desktop development with C++" workload.
   - Ensure `cl.exe` is in your PATH (e.g., launch "Developer Command Prompt for VS").

## Build & Install
Run the following command from the repository root:

```bash
cd bindings/python
pip install .
```

## Verification
Run the benchmark script to confirm the speedup:

```bash
python benchmarks/benchmark_native.py
```

## Troubleshooting
If you see `CommandNotFoundException` for `cl`, you are likely not in the Developer Command Prompt.
