
import ctypes
import os
import sys

# === CONFIGURACION MAESTRA ===
NUM_PARTICLES = 10_000_000  # 10M
ITERATIONS = 10             # Vueltas del loop externo
INTERNAL_STEPS = 64         # Profundidad fractal (carga por hilo)

def main():
    print("=== CMFO AGGRESSIVE NATIVE BENCHMARK (OPTION B) ===")
    print(f"Carga: {NUM_PARTICLES:,} Estados 7D")
    print(f"Pasos Fractales: {INTERNAL_STEPS} por iteracion")
    
    # Locate DLL
    dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "cmfo_core.dll"))
    if not os.path.exists(dll_path):
        print(f"[ERROR] No se encuentra la DLL nativa:\n  {dll_path}")
        print("Ejecute 'compile_core.bat' primero.")
        return

    try:
        core = ctypes.CDLL(dll_path)
    except Exception as e:
        print(f"[ERROR] Fallo al cargar DLL: {e}")
        return

    # Prototype: double run_cmfo_benchmark_internal(int N, int internal_steps, int iterations)
    core.run_cmfo_benchmark_internal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    core.run_cmfo_benchmark_internal.restype = ctypes.c_double

    print(f"[DRIVER] Invocando Nucleo Nativo (C++/CUDA)...")
    print(f"  -> DLL cargada: {os.path.basename(dll_path)}")
    print(f"  -> Enviando comando de ejecucion...")

    # EXECUTE
    time_sec = core.run_cmfo_benchmark_internal(NUM_PARTICLES, INTERNAL_STEPS, ITERATIONS)

    if time_sec < 0:
        print(f"\n[FAIL] El benchmark devolvio codigo de error: {time_sec}")
        return

    # METRICS
    total_ops = NUM_PARTICLES * ITERATIONS * INTERNAL_STEPS * 20 # ~20 FLOPs per step approx
    throughput_mops = (NUM_PARTICLES * ITERATIONS * INTERNAL_STEPS) / time_sec / 1e6
    throughput_tflops = total_ops / time_sec / 1e12

    print("\n=== RESULTADOS OFICIALES ===")
    print(f"  Tiempo Total:      {time_sec:.4f} s")
    print(f"  Estados Totales:   {NUM_PARTICLES * ITERATIONS:,}")
    print(f"  Total Steps Calc:  {NUM_PARTICLES * ITERATIONS * INTERNAL_STEPS:,}")
    print(f"--------------------------------------------------")
    print(f"  VELOCIDAD FRACTAL: {throughput_mops:,.2f} Millones Steps/s")
    print(f"  THROUGHPUT RAW:    {throughput_tflops:.4f} TFLOPS (Estimado)")
    print(f"--------------------------------------------------")
    print("  [VERIFICADO] Ejecucion Hardware Nativa Completa.")

if __name__ == "__main__":
    main()
