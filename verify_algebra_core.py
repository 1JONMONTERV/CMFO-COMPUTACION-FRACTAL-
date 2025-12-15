
import sys
import os
import math
sys.path.insert(0, os.path.abspath("bindings/python"))

from cmfo.constants import PHI
from cmfo.algebra import fractal_product
from cmfo.core.matrix import T7Matrix

print("=== Algebra Verification ===")
print(f"PHI = {PHI}")
res = fractal_product(PHI, PHI)
print(f"fractal_product(PHI, PHI) = {res}")
print(f"Expected by Docstring (PHI^2) = {PHI**2}")
print(f"Expected by Formula (PHI^1) = {PHI**1}")

if abs(res - PHI) < 1e-10:
    print("[CONCLUSION] Code implements Formula. Test/Docs are wrong.")
elif abs(res - PHI**2) < 1e-10:
    print("[CONCLUSION] Code implements Square. Formula in code is wrong??")
else:
    print("[conclusion] Code implements something else.")

print("\n=== Core Matrix Verification ===")
try:
    m = T7Matrix.identity()
    print("[SUCCESS] Native Matrix created.")
except Exception as e:
    print(f"[INFO] Native Matrix failed: {e}")
    print("This is expected if compiled extension is missing.")
