
import sys
import os

print(f"Python executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")

try:
    import cmfo
    print(f"CMFO version: {cmfo.__version__}")
    print(f"CMFO file: {cmfo.__file__}")
except ImportError as e:
    print(f"FAIL: Could not import cmfo: {e}")
    sys.exit(1)

try:
    from cmfo.algebra import fractal_product
    res = fractal_product(2.0, 1.618)
    print(f"Fractal product result: {res}")
except Exception as e:
    print(f"FAIL: Algebra error: {e}")

try:
    from cmfo.core.matrix import T7Matrix
    print("Imported T7Matrix class")
    m = T7Matrix()
    print("Instantiated T7Matrix (Native check)")
except Exception as e:
    print(f"FAIL: Matrix/Native error: {e}")
