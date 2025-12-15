
import sys
sys.path.insert(0, 'bindings/python')
from cmfo.compiler.jit import FractalJIT

print("--- Verifying Native JIT ---")
if FractalJIT.is_available():
    print("[OK] SUCCESS: Native JIT (cmfo_jit.dll) Loaded!")
    print("CMFO v3.0 is running in NATIVE ACCELERATED MODE.")
else:
    print("[FAIL] FAILURE: Could not load native library.")
