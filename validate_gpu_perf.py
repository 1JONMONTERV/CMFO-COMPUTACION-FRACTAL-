
import sys
import time
import argparse
sys.path.insert(0, 'bindings/python')
import cmfo # Initialize env

from cmfo.compiler.jit import FractalJIT
from cmfo.compiler.ir import *

def stress_test(runs=1000):
    print("\n" + "="*60)
    print(f"  COMBAT VERIFICATION: {runs} RUNS STRESS TEST")
    print("="*60)
    
    # 1. Setup Graph
    v = symbol('v')
    term1 = fractal_mul(v, v)
    expr = fractal_add(term1, constant(0.618034)) # v^2 + PHI
    
    # 2. Setup Data
    N_VECTORS = 1000 
    input_v = [1.0] * (N_VECTORS * 7)
    input_h = [0.0] * (N_VECTORS * 7)
    
    print(f"[+] Compiled Graph loaded.")
    print(f"[+] Payload: {N_VECTORS} vectors/run")
    
    # 3. Warmup
    print("[+] Warming up JIT...")
    try:
        if not FractalJIT.is_available():
            print("[FAIL] NATIVE JIT NOT DETECTED. Aborting Stress Test.")
            return False
        _ = FractalJIT.compile_and_run(expr, input_v, input_h)
    except Exception as e:
        print(f"[FAIL] Warmup Failed: {e}")
        return False
        
    # 4. Loop
    print(f"[+] Engaging Loop ({runs} iterations)...")
    start_time = time.time()
    errors = 0
    
    for i in range(runs):
        if i % 100 == 0:
            print(f"    Iter {i}/{runs}...", end='\r')
        try:
            # We use the same graph/data repeatedly
            # The JIT Manager should handle caching internally to avoid re-compiling every time?
            # Actually our current simple JIT re-compiles every call in this prototype version.
            # This makes it a great TORTURE TEST for the definition of "JIT".
            res = FractalJIT.compile_and_run(expr, input_v, input_h)
            
            # Simple check on first element
            if abs(res[0][0] - 1.618034) > 1e-4:
                errors += 1
        except Exception as e:
            print(f"\n    [FAIL] Crash at iter {i}: {e}")
            errors += 1
            break
            
    total_time = time.time() - start_time
    avg_ips = runs / total_time
    
    print(f"\n\n[RESULTS]")
    print(f"  - Total Time:   {total_time:.4f}s")
    print(f"  - Throughput:   {avg_ips:.2f} runs/sec")
    print(f"  - Errors:       {errors}")
    
    if errors == 0:
        print("[OK] STABILITY CONFIRMED: 0 Casualties.")
        return True
    else:
        print("[FAIL] STABILITY FAILED.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1000)
    args = parser.parse_args()
    
    success = stress_test(args.runs)
    sys.exit(0 if success else 1)
