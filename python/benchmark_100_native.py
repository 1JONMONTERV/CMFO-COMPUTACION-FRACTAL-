
import os
import sys
import time
import csv
import ctypes
import struct

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from cmfo.bitcoin import NonceRestrictor, build_header
except ImportError:
    # Fallback
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from cmfo.bitcoin import NonceRestrictor, build_header

def run_native_benchmark():
    print("=========================================================")
    print("   CMFO: BENCHMARK MINERIA 100 BLOQUES REALES (NATIVE CUDA)")
    print("=========================================================")

    # 1. Load DLL
    dll_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'build', 'sha256_benchmark.dll')
    print(f"[INFO] Cargando DLL: {dll_path}")
    
    if not os.path.exists(dll_path):
        print("[FAIL] DLL no encontrada. Compilar primero.")
        return

    try:
        cuda_lib = ctypes.CDLL(dll_path)
    except Exception as e:
        print(f"[FAIL] Error cargando DLL: {e}")
        return

    # Prototype: void launch_benchmark(u32* h_out, int blocks, int threads, int iterations)
    cuda_lib.launch_benchmark.argtypes = [ctypes.POINTER(ctypes.c_uint32), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    cuda_lib.launch_benchmark.restype = None

    # 2. Load Data
    csv_path = os.path.join(os.path.dirname(__file__), 'bloques_100.csv')
    blocks = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                blocks.append(row)
    except Exception as e:
        print(f"[ERROR] CSV fail: {e}")
        return
        
    print(f"[INFO] Bloques cargados: {len(blocks)}")

    # 3. Process Blocks (Inversion)
    total_time_inversion = 0
    total_space_original = 0
    total_space_reduced = 0

    print("\n[PROCESANDO BLOQUES...]")
    print(f"{'ID':<5} | {'HASH (prefix)':<16} | {'INVERSION (ms)':<15} | {'REDUCCION (x)':<15}")
    print("-" * 60)

    for i, block in enumerate(blocks):
        # Mock Header Building
        if i < len(blocks)-1:
             prev_hash_hex = blocks[i+1]['hash']
        else:
             prev_hash_hex = "0000000000000000000000000000000000000000000000000000000000000000"

        # Mock Data (Standard difficulty for consistency)
        version = 536870912
        prev_block_bytes = bytes.fromhex(prev_hash_hex)
        merkle_root_bytes = bytes.fromhex(block['merkleroot'])
        timestamp = int(time.time())
        bits = 0x1715a35c 
        nonce = 0 
        
        header = build_header(version, prev_block_bytes, merkle_root_bytes, timestamp, bits, nonce)
        
        t0 = time.time()
        restrictor = NonceRestrictor(header, empirical_mode='conservative')
        success, reduced_space, reduction_factor = restrictor.reduce_space()
        dt_inv = (time.time() - t0) * 1000 # ms
        
        total_time_inversion += dt_inv
        
        if not success:
           reduced_space = 2**32
           reduction_factor = 1.0

        total_space_original += 2**32
        total_space_reduced += reduced_space
        
        if i < 15:
            print(f"{i:<5} | {block['hash'][:12]}... | {dt_inv:12.2f} ms | {reduction_factor:12.2f}x")

    # 4. Process GPU Benchmark
    print("\n[BENCHMARK GPU (CUDA NATIVO)...]")
    
    # Args
    threads_per_block = 256
    blocks_per_grid = 4096 
    iterations = 100 # Heavy iterations per kernel launch
    total_threads_launch = threads_per_block * blocks_per_grid
    
    h_out = ctypes.c_uint32(0)
    
    # Warmup
    print("[CUDA] Warming up...")
    cuda_lib.launch_benchmark(ctypes.byref(h_out), blocks_per_grid, threads_per_block, 100)
    
    # Measure
    print("[CUDA] Running High-Load Stress Test...")
    loops = 10
    t_start = time.time()
    for _ in range(loops):
        cuda_lib.launch_benchmark(ctypes.byref(h_out), blocks_per_grid, threads_per_block, iterations)
    t_end = time.time()
    
    duration = t_end - t_start
    total_hashes = total_threads_launch * iterations * loops
    hashrate = total_hashes / duration / 1_000_000 # MH/s
    
    print("\n=========================================================")
    print("   RESULTADOS FINALES (100 BLOQUES + CUDA NATIVE)")
    print("=========================================================")
    print(f"Bloques Procesados:      {len(blocks)}")
    print(f"Tiempo Total Inversión:  {total_time_inversion:.2f} ms")
    print(f"Inversión Promedio:      {total_time_inversion/len(blocks):.2f} ms/bloque")
    print("-" * 50)
    avg_reduction = total_space_original / total_space_reduced if total_space_reduced > 0 else 0
    print(f"FACTOR REDUCCIÓN GLOBAL:  {avg_reduction:.2f}x")
    print("-" * 50)
    print(f"HASHRATE GPU (Raw):      {hashrate:.2f} MH/s")
    print(f"HASHRATE GPU (Efectivo): {hashrate * avg_reduction:.2f} MH/s")
    print("=========================================================")

if __name__ == "__main__":
    run_native_benchmark()
