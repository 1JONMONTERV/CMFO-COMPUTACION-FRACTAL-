
import sys
import traceback
import numpy as np
from numba import cuda, uint32

print("Numba version:", sys.modules.get('numba').__version__)

K_np = np.array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5
], dtype=np.uint32)

@cuda.jit(device=True)
def rotr(x, n):
    return (x >> n) | (x << (32 - n))

@cuda.jit
def test_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        # Simple ops
        val = arr[idx]
        res = rotr(val, 5)
        arr[idx] = res

def run_test():
    try:
        if not cuda.is_available():
            print("CUDA not available")
            return
            
        print("Compiling kernel...")
        arr = cuda.device_array(10, dtype=np.uint32)
        test_kernel[1, 10](arr)
        cuda.synchronize()
        print("Kernel OK")
        
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
