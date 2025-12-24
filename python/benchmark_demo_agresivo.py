
import os
import sys

cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
os.environ["CUDA_HOME"] = cuda_path
os.environ["NUMBAPRO_NVVM"] = os.path.join(cuda_path, "nvvm", "bin", "nvvm64.dll")
os.environ["NUMBAPRO_LIBDEVICE"] = os.path.join(cuda_path, "nvvm", "libdevice")
os.environ["PATH"] += ";" + os.path.join(cuda_path, "bin")

try:
    from numba import cuda
    import numpy as np
    import math
    import time
except ImportError as e:
    print(f"CRITICAL: Failed to import numba: {e}")
    sys.exit(1)

# === CONFIGURACION AGRESIVA PERO COMPILABLE ===
N_PARTICLES = 2_000_000 
BLOCK_SIZE = 256 
GRID_SIZE = (N_PARTICLES + BLOCK_SIZE - 1) // BLOCK_SIZE

print(f"\n=== INICIANDO BENCHMARK DE DEMO AGRESIVO (CMFO GPU) V3 ===")
print(f"Configuracion: {N_PARTICLES:,} particulas 7D")

# Use fastmath to speed up compilation and execution
@cuda.jit(fastmath=True)
def cmfo_kernel_7d_agresivo(states_out, states_in, phases, t_offset):
    idx = cuda.grid(1)
    if idx < states_in.shape[0]:
        v0 = states_in[idx, 0]
        v1 = states_in[idx, 1]
        v2 = states_in[idx, 2]
        v3 = states_in[idx, 3]
        v4 = states_in[idx, 4]
        v5 = states_in[idx, 5]
        v6 = states_in[idx, 6]
        
        phi = 1.61803398875
        phase = phases[idx]
        
        # 50 Iterations
        for k in range(50):
            # Optimizable Rotation
            c_p = math.cos(phase)
            s_p = math.sin(phase)
            
            temp = v0
            v0 = (v0 * c_p - v1 * s_p) * phi
            v1 = (temp * s_p + v1 * c_p) / phi
            
            # Simple Trig Coupling
            v2 += math.sin(v0 * v1)
            v3 += math.cos(v2 * t_offset)
            
            # Heavy Ops
            v5 = math.sqrt(abs(v0 * v5 + 1.0))
            
            phase += 0.1
            
        states_out[idx, 0] = v0
        states_out[idx, 1] = v1
        states_out[idx, 2] = v2
        states_out[idx, 3] = v3
        states_out[idx, 4] = v4
        states_out[idx, 5] = v5
        states_out[idx, 6] = v6

def run_benchmark():
    try:
        cuda.select_device(0)
    except:
        pass # Already selected in simple test

    print("\n[FASE 1] Allocating...")
    # Generate data
    host_states = np.random.rand(N_PARTICLES, 7).astype(np.float32)
    host_phases = np.random.rand(N_PARTICLES).astype(np.float32)

    d_states_in = cuda.to_device(host_states)
    d_states_out = cuda.device_array_like(host_states)
    d_phases = cuda.to_device(host_phases)
    
    # Warmup
    print("[FASE 2] Compiling (FastMath ON)...")
    t0 = time.time()
    cmfo_kernel_7d_agresivo[GRID_SIZE, BLOCK_SIZE](d_states_out, d_states_in, d_phases, 0.0)
    cuda.synchronize()
    t1 = time.time()
    print(f"  -> Compiled in {t1-t0:.2f} s")
    
    # Benchmark
    print("[FASE 3] Executing Stress Test (20 Iters)...")
    t_start = time.time()
    
    ITERATIONS = 20
    for i in range(ITERATIONS):
        cmfo_kernel_7d_agresivo[GRID_SIZE, BLOCK_SIZE](d_states_out, d_states_in, d_phases, float(i))
    
    cuda.synchronize()
    t_end = time.time()
    
    duration = t_end - t_start
    total_steps = N_PARTICLES * ITERATIONS * 50
    mops = (total_steps) / duration / 1e6
    
    # FLOPs estimate: ~20 ops per step
    flops = total_steps * 20
    tflops = flops / duration / 1e12

    print(f"\n=== RESULTADOS SUCCESS (GPU) ===")
    print(f"  Tiempo:       {duration:.4f} s")
    print(f"  Speed:        {mops:,.2f} M Steps/s")
    print(f"  Fractal Ops:  {tflops:.4f} TFLOPS")

if __name__ == "__main__":
    run_benchmark()
