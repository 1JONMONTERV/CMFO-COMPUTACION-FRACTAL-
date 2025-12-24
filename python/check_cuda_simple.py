
import os
import sys

cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
os.environ["CUDA_HOME"] = cuda_path
os.environ["NUMBAPRO_NVVM"] = os.path.join(cuda_path, "nvvm", "bin", "nvvm64.dll")
os.environ["NUMBAPRO_LIBDEVICE"] = os.path.join(cuda_path, "nvvm", "libdevice")
os.environ["PATH"] += ";" + os.path.join(cuda_path, "bin")

from numba import cuda
import numpy as np

print("Compiling simple kernel...")
@cuda.jit
def add(a, b, c):
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]

N = 1000
a = cuda.to_device(np.ones(N))
b = cuda.to_device(np.ones(N))
c = cuda.device_array_like(a)

print("Running...")
add[1, 1024](a, b, c)
print("Done.")
print(c.copy_to_host()[0])
