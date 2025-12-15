
import subprocess
import sys
import time
import os

def run_step(name, command):
    print(f"\n[{name}] Running...")
    start = time.time()
    try:
        # Run command and capture output
        # Using shell=True for simple command chaining compatibility
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed = time.time() - start
        print(f"[PASS] ({elapsed:.2f}s)")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"[FAIL] ({elapsed:.2f}s)")
        print(f"   Error: {e.stderr}")
        return False, e.stderr

def master_validation():
    print("==================================================")
    print("      CMFO v3.1 MASTER VALIDATION PROTOCOL        ")
    print("==================================================")
    print(f"Time: {time.ctime()}")
    
    steps = [
        ("Unit Tests (Accuracy)", "python -m unittest discover tests_v3"),
        ("Certification (Deep Stress)", "python tests_certification/test_deep_stress.py"),
        ("Genesis Verification", "python verify_genesis.py"),
        ("Equation Compilation", "python verify_base_equations.py"),
        ("Sci: Convergence", "python experiments/exp_convergence.py"),
        ("Sci: Stability", "python experiments/exp_stability.py"),
        ("Sci: Correspondence", "python experiments/exp_correspondence.py"),
        ("Info: Compression", "python experiments/exp_compression.py")
    ]
    
    failures = []
    
    for name, cmd in steps:
        success, output = run_step(name, cmd)
        if not success:
            failures.append(name)
            
    print("\n==================================================")
    print("               FINAL VERDICT                      ")
    print("==================================================")
    
    if not failures:
        print("🟢 ALL SYSTEMS GO. RIGOROUS VALIDATION PASSED.")
        print("Ready for GitHub Upload.")
        sys.exit(0)
    else:
        print(f"🔴 CRITICAL FAILURES FOUND in: {failures}")
        sys.exit(1)

if __name__ == "__main__":
    master_validation()
