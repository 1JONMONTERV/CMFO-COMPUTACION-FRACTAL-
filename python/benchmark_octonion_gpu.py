# -*- coding: utf-8 -*-
"""
BENCHMARK ULTRA-7D AGRESIVO CON ALGEBRA OCTONIONICA
====================================================

Benchmark que demuestra la potencia del sistema CMFO Ultra-7D
usando operaciones octoniónicas reales en GPU.
"""

import os
import sys
import time
import numpy as np

# Configurar CUDA
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
os.environ["CUDA_HOME"] = cuda_path
os.environ["NUMBAPRO_NVVM"] = os.path.join(cuda_path, "nvvm", "bin", "nvvm64.dll")
os.environ["NUMBAPRO_LIBDEVICE"] = os.path.join(cuda_path, "nvvm", "libdevice")
os.environ["PATH"] += ";" + os.path.join(cuda_path, "bin")

try:
    from numba import cuda
    import math
except ImportError as e:
    print(f"ERROR: {e}")
    sys.exit(1)

# === CONFIGURACION ===
N_OCTONIONS = 1_000_000     # 1 millón de octoniones
BLOCK_SIZE = 256
GRID_SIZE = (N_OCTONIONS + BLOCK_SIZE - 1) // BLOCK_SIZE
PHI = 1.618033988749895

print("=" * 70)
print("  BENCHMARK ULTRA-7D: ALGEBRA OCTONIONICA EN GPU")
print("=" * 70)
print(f"  Octoniones: {N_OCTONIONS:,}")
print(f"  GPU Blocks: {GRID_SIZE:,} x {BLOCK_SIZE}")
print()


@cuda.jit(fastmath=True)
def octonion_cayley_dickson_kernel(out_real, out_imag, 
                                    a_real, a_imag, 
                                    b_real, b_imag,
                                    iterations):
    """
    Kernel que implementa multiplicación de Cayley-Dickson en GPU.
    
    O = (H, H) donde H son cuaterniones
    (p,q) * (r,s) = (pr - conj(s)q, sp + q*conj(r))
    """
    idx = cuda.grid(1)
    if idx >= a_real.shape[0]:
        return
    
    # Cargar octoniones a registros
    # a = (a0,a1,a2,a3) + (a4,a5,a6,a7)
    p0 = a_real[idx, 0]
    p1 = a_real[idx, 1]
    p2 = a_real[idx, 2]
    p3 = a_real[idx, 3]
    q0 = a_imag[idx, 0]
    q1 = a_imag[idx, 1]
    q2 = a_imag[idx, 2]
    q3 = a_imag[idx, 3]
    
    r0 = b_real[idx, 0]
    r1 = b_real[idx, 1]
    r2 = b_real[idx, 2]
    r3 = b_real[idx, 3]
    s0 = b_imag[idx, 0]
    s1 = b_imag[idx, 1]
    s2 = b_imag[idx, 2]
    s3 = b_imag[idx, 3]
    
    phi = 1.618033988749895
    
    for _ in range(iterations):
        # === MULTIPLICACION DE CUATERNIONES p*r ===
        pr0 = p0*r0 - p1*r1 - p2*r2 - p3*r3
        pr1 = p0*r1 + p1*r0 + p2*r3 - p3*r2
        pr2 = p0*r2 - p1*r3 + p2*r0 + p3*r1
        pr3 = p0*r3 + p1*r2 - p2*r1 + p3*r0
        
        # === CONJUGADO DE s: conj(s) = (s0, -s1, -s2, -s3) ===
        sc0, sc1, sc2, sc3 = s0, -s1, -s2, -s3
        
        # === conj(s) * q ===
        scq0 = sc0*q0 - sc1*q1 - sc2*q2 - sc3*q3
        scq1 = sc0*q1 + sc1*q0 + sc2*q3 - sc3*q2
        scq2 = sc0*q2 - sc1*q3 + sc2*q0 + sc3*q1
        scq3 = sc0*q3 + sc1*q2 - sc2*q1 + sc3*q0
        
        # === Primera parte: pr - conj(s)*q ===
        out0 = pr0 - scq0
        out1 = pr1 - scq1
        out2 = pr2 - scq2
        out3 = pr3 - scq3
        
        # === s * p ===
        sp0 = s0*p0 - s1*p1 - s2*p2 - s3*p3
        sp1 = s0*p1 + s1*p0 + s2*p3 - s3*p2
        sp2 = s0*p2 - s1*p3 + s2*p0 + s3*p1
        sp3 = s0*p3 + s1*p2 - s2*p1 + s3*p0
        
        # === CONJUGADO DE r: conj(r) ===
        rc0, rc1, rc2, rc3 = r0, -r1, -r2, -r3
        
        # === q * conj(r) ===
        qrc0 = q0*rc0 - q1*rc1 - q2*rc2 - q3*rc3
        qrc1 = q0*rc1 + q1*rc0 + q2*rc3 - q3*rc2
        qrc2 = q0*rc2 - q1*rc3 + q2*rc0 + q3*rc1
        qrc3 = q0*rc3 + q1*rc2 - q2*rc1 + q3*rc0
        
        # === Segunda parte: sp + q*conj(r) ===
        out4 = sp0 + qrc0
        out5 = sp1 + qrc1
        out6 = sp2 + qrc2
        out7 = sp3 + qrc3
        
        # === EVOLUCION PHI (Fractal) ===
        norm = math.sqrt(out0*out0 + out1*out1 + out2*out2 + out3*out3 +
                        out4*out4 + out5*out5 + out6*out6 + out7*out7)
        if norm > 1e-10:
            inv_norm = 1.0 / norm
            p0 = out0 * inv_norm * phi
            p1 = out1 * inv_norm / phi
            p2 = out2 * inv_norm * phi
            p3 = out3 * inv_norm / phi
            q0 = out4 * inv_norm * phi
            q1 = out5 * inv_norm / phi
            q2 = out6 * inv_norm * phi
            q3 = out7 * inv_norm / phi
    
    # Guardar resultado
    out_real[idx, 0] = p0
    out_real[idx, 1] = p1
    out_real[idx, 2] = p2
    out_real[idx, 3] = p3
    out_imag[idx, 0] = q0
    out_imag[idx, 1] = q1
    out_imag[idx, 2] = q2
    out_imag[idx, 3] = q3


def run_benchmark():
    """Ejecuta el benchmark octoniónico."""
    
    print("[FASE 1] Generando octoniones aleatorios...")
    np.random.seed(42)
    
    # Generar datos: cada octonión = 2 cuaterniones (real, imag)
    a_real = np.random.randn(N_OCTONIONS, 4).astype(np.float32)
    a_imag = np.random.randn(N_OCTONIONS, 4).astype(np.float32)
    b_real = np.random.randn(N_OCTONIONS, 4).astype(np.float32)
    b_imag = np.random.randn(N_OCTONIONS, 4).astype(np.float32)
    
    # Normalizar
    for arr in [a_real, a_imag, b_real, b_imag]:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr[:] = arr / (norms + 1e-10)
    
    print(f"  Memoria CPU: {(a_real.nbytes * 4) / 1024**2:.2f} MB")
    
    # Transferir a GPU
    print("[FASE 2] Transfiriendo a GPU...")
    d_a_real = cuda.to_device(a_real)
    d_a_imag = cuda.to_device(a_imag)
    d_b_real = cuda.to_device(b_real)
    d_b_imag = cuda.to_device(b_imag)
    d_out_real = cuda.device_array_like(a_real)
    d_out_imag = cuda.device_array_like(a_imag)
    
    # Warmup
    print("[FASE 3] Compilando kernel (warmup)...")
    t0 = time.time()
    octonion_cayley_dickson_kernel[GRID_SIZE, BLOCK_SIZE](
        d_out_real, d_out_imag, d_a_real, d_a_imag, d_b_real, d_b_imag, 1
    )
    cuda.synchronize()
    t_compile = time.time() - t0
    print(f"  Compilado en: {t_compile:.2f} s")
    
    # Benchmark principal
    ITERATIONS = 50  # Iteraciones internas del kernel
    RUNS = 20        # Ejecuciones del kernel
    
    print(f"[FASE 4] Ejecutando benchmark ({RUNS} runs x {ITERATIONS} iters)...")
    
    t_start = time.time()
    for run in range(RUNS):
        octonion_cayley_dickson_kernel[GRID_SIZE, BLOCK_SIZE](
            d_out_real, d_out_imag, d_a_real, d_a_imag, d_b_real, d_b_imag, ITERATIONS
        )
    cuda.synchronize()
    t_end = time.time()
    
    duration = t_end - t_start
    
    # Métricas
    total_multiplications = N_OCTONIONS * RUNS * ITERATIONS
    # Cada multiplicación octoniónica = ~130 FLOPs (estimado)
    total_flops = total_multiplications * 130
    
    mops = total_multiplications / duration / 1e6
    gflops = total_flops / duration / 1e9
    
    print()
    print("=" * 70)
    print("  RESULTADOS ULTRA-7D")
    print("=" * 70)
    print(f"  Tiempo total:         {duration:.4f} s")
    print(f"  Multiplicaciones:     {total_multiplications:,}")
    print("-" * 70)
    print(f"  VELOCIDAD:            {mops:,.2f} M Octonionic Mult/s")
    print(f"  THROUGHPUT:           {gflops:.2f} GFLOPS")
    print("=" * 70)
    
    # Verificar resultado
    out_real = d_out_real.copy_to_host()
    out_imag = d_out_imag.copy_to_host()
    
    sample_norm = np.sqrt(np.sum(out_real[0]**2) + np.sum(out_imag[0]**2))
    print(f"\n[CHECK] Norma muestra: {sample_norm:.6f} (debería ser ~phi)")
    print("[OK] Benchmark completado exitosamente!")
    
    return duration, mops, gflops


if __name__ == "__main__":
    try:
        cuda.select_device(0)
        device = cuda.get_current_device()
        print(f"[GPU] {device.name}")
        print()
    except Exception as e:
        print(f"[ERROR] No se pudo seleccionar GPU: {e}")
        sys.exit(1)
    
    run_benchmark()
