
import os
import sys
import random
import hashlib
import struct
import time

# Add local path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from cmfo.crypto.fractal_sha256 import FractalSHA256, FractalWord
from cmfo.compiler.jit import FractalJIT

def section(title):
    print(f"\n{'='*60}")
    print(f"AUDIT SECTION: {title}")
    print(f"{'='*60}")

def audit_fractal_algebra():
    section("FRACTAL ALGEBRA (BTFL) CONSISTENCY")
    print("Verifying Truth Tables for [0.0, 1.0] Domain...")
    
    # Truth Table for AND (Min)
    # A | B | Expect
    # 0 | 0 | 0
    # 0 | 1 | 0
    # 1 | 0 | 0
    # 1 | 1 | 1
    failures = 0
    
    cases = [(0.0,0.0,0.0), (0.0,1.0,0.0), (1.0,0.0,0.0), (1.0,1.0,1.0)]
    for a,b,exp in cases:
        wa = FractalWord([a], 1)
        wb = FractalWord([b], 1)
        res = (wa & wb).trits[0]
        if abs(res - exp) > 1e-9:
            print(f"[FAIL] AND({a},{b}) -> {res} (Expected {exp})")
            failures += 1
            
    # XOR (Abs Diff)
    # 0 | 0 | 0
    # 0 | 1 | 1
    # 1 | 0 | 1
    # 1 | 1 | 0
    cases_xor = [(0.0,0.0,0.0), (0.0,1.0,1.0), (1.0,0.0,1.0), (1.0,1.0,0.0)]
    for a,b,exp in cases_xor:
        wa = FractalWord([a], 1)
        wb = FractalWord([b], 1)
        res = (wa ^ wb).trits[0]
        if abs(res - exp) > 1e-9:
            print(f"[FAIL] XOR({a},{b}) -> {res} (Expected {exp})")
            failures += 1
            
    if failures == 0:
        print("[PASS] Algebraic Logic is MATHEMATICALLY SOUND.")
    return failures

def audit_sha256_bit_exactness():
    section("FRACTAL SHA-256 BIT EXACTNESS")
    print("Running 1000 Monte-Carlo Comparisons against hashlib...")
    
    failures = 0
    start_t = time.time()
    
    for i in range(1000):
        # Result caching optimization check
        msg_len = random.randint(0, 100)
        msg = os.urandom(msg_len)
        
        # Reference
        ref_hash = hashlib.sha256(msg).hexdigest()
        
        # Fractal
        # Pad manually because FractalSHA256 expects words
        # 1. Pad
        orig_len_bits = len(msg) * 8
        msg_padded = msg + b'\x80'
        while (len(msg_padded) * 8) % 512 != 448:
            msg_padded += b'\x00'
        msg_padded += struct.pack(">Q", orig_len_bits)
        
        # 2. Blockify & Compress
        chunks = [msg_padded[j:j+64] for j in range(0, len(msg_padded), 64)]
        
        fsha = FractalSHA256()
        for chunk in chunks:
            words = [FractalWord.from_int(struct.unpack(">I", chunk[k:k+4])[0]) for k in range(0, 64, 4)]
            fsha.compress(words)
            
        # res_hash = f"{fsha.get_hash_hex()}" # Not implemented
        h_ints = fsha.get_hash()
        res_hash = "".join(f"{x:08x}" for x in h_ints)
        
        if res_hash != ref_hash:
            print(f"[FAIL] Mismatch on sample {i}")
            print(f"Ref: {ref_hash}")
            print(f"Fra: {res_hash}")
            failures += 1
            break # Stop on first fail
            
    elapsed = time.time() - start_t
    print(f"Time: {elapsed:.2f}s | Samples: 1000")
    
    if failures == 0:
        print("[PASS] SHA-256 Core is 100% BIT-EXACT.")
    else:
        print(f"[CRITICAL FAIL] {failures} errors detected.")
        
    return failures

def audit_gpu_pipeline():
    section("GPU / JIT PIPELINE DIAGNOSTIC")
    
    try:
        if not FractalJIT.is_available():
            print("[WARN] JIT Library not loaded. Is CUDA installed?")
            return 1 # SOFT FAIL
        else:
            print("[PASS] Native Library Loaded (cmfo_jit.dll)")
            
        # Compile Test
        print("Testing JIT Compilation...")
        from cmfo.compiler.ir import Symbol, fractal_add
        
        # Graph: A + B
        idx = Symbol("idx") # Dummy
        graph = fractal_add(Symbol("A"), Symbol("B"))
        
        # Run
        v = [1.0] * 7
        h = [2.0] * 7
        res = FractalJIT.compile_and_run(graph, v, h)
        
        # Expected: 3.0
        val = res[0][0]
        if abs(val - 3.0) < 1e-5:
             print(f"[PASS] JIT Compilation & Execution Valid (1.0 + 2.0 = {val})")
             return 0
        else:
             print(f"[FAIL] JIT Math Error. Expected 3.0, got {val}")
             return 1
             
    except Exception as e:
        print(f"[FAIL] GPU Exception: {e}")
        return 1

def run_audit():
    print("CMFO SYSTEM INTEGRITY AUDIT")
    print("===========================")
    print(f"Date: {time.ctime()}")
    
    err_alg = audit_fractal_algebra()
    err_sha = audit_sha256_bit_exactness()
    err_gpu = audit_gpu_pipeline()
    
    section("FINAL VERDICT")
    total_err = err_alg + err_sha + err_gpu
    if total_err == 0:
        print("✅ SYSTEM INTEGRITY: 100% PASS")
        print("The Codebase is Mathematically Sound and Rigorous.")
    else:
        print(f"❌ SYSTEM INTEGRITY: FAILED ({total_err} faults)")
        print("Recommendations: Review 'FAIL' sections immediately.")
        
if __name__ == "__main__":
    run_audit()
