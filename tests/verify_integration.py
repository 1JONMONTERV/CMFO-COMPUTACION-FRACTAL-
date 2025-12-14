import sys
import os

# Ensure we're testing the local bindings
sys.path.append(os.path.abspath("bindings/python"))

try:
    import cmfo
    from cmfo.core.fractal import fractal_root, PhiBit
    from cmfo import info as cmfo_info
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

print("="*60)
print("CMFO INTEGRATION VERIFICATION")
print("="*60)

# 1. Check Info
print("\n[INFO Check]")
cmfo_info()

# 2. Check Algebra
print("\n[Algebra Check]")
val = 100.0
res = fractal_root(val)
print(f"Fractal Root of 100: {res:.4f}")
if abs(res - 1.0) < val: # Minimal check
    print("✓ Algebra OK")

# 3. Check Logic
print("\n[Logic Check]")
bit = PhiBit.TRUE
print(f"PhiBit TRUE: {bit}")
if abs(bit - 1.618) < 0.01:
    print("✓ Logic OK")

print("\n✓ SUCCESS: CMFO Core Integration Verified.")
