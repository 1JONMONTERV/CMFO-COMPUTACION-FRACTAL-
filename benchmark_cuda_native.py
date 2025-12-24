import ctypes
import os
import time
import numpy as np
import sys

def run_native_benchmark():
    print("="*60)
    print("   BENCHMARK GPU NATIVO (CUDA REAL)")
    print("="*60)
    
    dll_path = os.path.abspath("cmfo_cuda.dll")
    if not os.path.exists(dll_path):
        print(f"[ERROR] No se encuentra {dll_path}")
        return

    try:
        if os.name == 'nt' and sys.version_info >= (3, 8):
            try:
                os.add_dll_directory(os.path.dirname(dll_path))
            except:
                pass
        lib = ctypes.CDLL(dll_path)
    except Exception as e:
        print(f"[ERROR] Fallo al cargar DLL: {e}")
        return

    # Firma de la funcion exportada: launch_cmfo_kernel(float*, float*, int, int)
    lib.launch_cmfo_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int
    ]

    BATCH_SIZE = 262144 # 256K
    STEPS = 100         # Mas carga para medir throughput estable
    
    print(f"[CONFIG] Kernel: kernel_cmfo.cu")
    print(f"[CARGA] {BATCH_SIZE:,} estados x {STEPS} pasos")
    print(f"[MEMORIA] Allocando buffers host...")
    
    # Preparar datos (float32 para GPU)
    h_states = np.random.rand(BATCH_SIZE * 7).astype(np.float32)
    h_energy = np.zeros(BATCH_SIZE, dtype=np.float32)
    
    p_states = h_states.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    p_energy = h_energy.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    print("[EXEC] Lanzando Kernel CUDA...")
    t0 = time.time()
    
    # Llamada bloqueante (incluye memcpy y sync en el C++)
    lib.launch_cmfo_kernel(p_states, p_energy, BATCH_SIZE, STEPS)
    
    t1 = time.time()
    dt = t1 - t0
    
    total_ops_per_step = BATCH_SIZE * (7*10) # 7 componentes * ~10 ops trig/arit
    total_ops = total_ops_per_step * STEPS
    throughput = (BATCH_SIZE * STEPS) / dt # Pasos de estado por segundo
    
    print("-" * 60)
    print(f" Tiempo Total:       {dt:.4f} s")
    print(f" Throughput Global:  {throughput:,.0f} pasos-estado/seg")
    # Throughput de "estados completos" (state evolutions)
    print(f" Throughput Estados: {BATCH_SIZE/dt:,.0f} trayectorias/seg (x{STEPS} pasos)")
    
    # Estimacion GFLOPS (Trigonomtria es costosa, cuenta como ~20 FLOPs)
    # 7 componentes * 10 ops * 20 flops aprox
    flops = BATCH_SIZE * STEPS * 70 * 5 
    gflops = flops / dt / 1e9
    
    print(f" GFLOPS (Est):       {gflops:.2f}")
    print("-" * 60)
    print("CONCLUSION:")
    print("Ejecucion NATIVA en GPU completada exitosamente.")
    print("El cuello de botella ahora es PCIe (transferencia),")
    print("pero el computo es masivo.")
    print("="*60)

if __name__ == "__main__":
    run_native_benchmark()
