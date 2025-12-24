
from numba import cuda
import numpy as np

@cuda.jit
def test_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2

try:
    data = np.ones(256)
    threadsperblock = 32
    blockspergrid = (data.size + (threadsperblock - 1)) // threadsperblock
    test_kernel[blockspergrid, threadsperblock](data)
    print("Kernel run successful")
    print(data[:10])
except Exception as e:
    print(f"Kernel failed: {e}")
