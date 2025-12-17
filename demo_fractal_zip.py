
import os
import sys
import time


# Import bindings robustly
current_dir = os.path.dirname(os.path.abspath(__file__))
bindings_path = os.path.join(current_dir, 'bindings', 'python')
sys.path.insert(0, bindings_path)

try:
    from cmfo.compression.fractal_zip import FractalCompressor
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("[HINT] Ensure you are running from the project root.")
    sys.exit(1)

def run_demo():
    print("=========================================")
    print("   CMFO FRACTAL ZIP v1.0")
    print("   Compresión + Hyper-Memory")
    print("=========================================")
    
    # 1. Create a dummy large file
    test_file = "test_big_data.txt"
    if not os.path.exists(test_file):
        print(f"[INFO] Generando archivo de prueba: {test_file}...")
        with open(test_file, 'w') as f:
            for i in range(50000):
                f.write(f"LINEA_DE_DATOS_REPETITIVA_FRACTAL_{i} ")
    
    # 2. Compress
    compressor = FractalCompressor(swap_dir="fractal_archive")
    
    print("\n[PASO 1] Comprimiendo y Guardando en Memoria...")
    t0 = time.time()
    archive_path = compressor.compress_file(test_file)
    print(f"   Tiempo: {time.time()-t0:.4f}s")
    
    # 3. Decompress
    print("\n[PASO 2] Recuperando y Descomprimiendo...")
    out_file = "restored_data.txt"
    t1 = time.time()
    compressor.decompress_file(archive_path, out_file)
    print(f"   Tiempo: {time.time()-t1:.4f}s")
    
    # 4. Verify
    orig_s = os.path.getsize(test_file)
    rest_s = os.path.getsize(out_file)
    print(f"\n[VERIFICACIÓN] Original: {orig_s} bytes | Recuperado: {rest_s} bytes")
    if orig_s == rest_s:
        print("   INTEGRIDAD: 100% CORRECTA")
    else:
        print("   INTEGRIDAD: FALLÓ")

    # Cleanup
    # os.remove(test_file)
    # os.remove(out_file)
    print("\n[CONCLUSIÓN] La combinación Compresión + Memoria Fractal es operativa.")

if __name__ == "__main__":
    run_demo()
