
from numba import jit, prange
import numpy as np
import time

@jit(nopython=True, parallel=True)
def test_cpu():
    acc = 0
    for i in prange(100):
        acc += i
    return acc

try:
    print("Testing CPU JIT...")
    res = test_cpu()
    print(f"CPU JIT Result: {res}")
except Exception as e:
    print(f"CPU JIT Failed: {e}")
