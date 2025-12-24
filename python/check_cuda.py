
from numba import cuda
import sys

try:
    print("Checking CUDA...")
    cuda.select_device(0)
    print("Device selected.")
    print("Detecting...")
    print(cuda.detect())
except Exception as e:
    print(e)
