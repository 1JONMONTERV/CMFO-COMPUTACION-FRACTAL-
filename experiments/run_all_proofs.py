import os
import subprocess
import sys

def run_script(script_name):
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    
    path = os.path.join("experiments", "reproducibility", script_name)
    
    try:
        # Use sys.executable to ensure we use the same python
        result = subprocess.run([sys.executable, path], check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"!!! FAIL: {script_name} crashed !!!")
        return False
    except Exception as e:
        print(f"!!! ERROR: Could not run {script_name}: {e}")
        return False

def main():
    print("CMFO CERTIFICATION SUITE")
    print("Verifying Physics, Logic, and Computation Claims...\n")
    
    scripts = [
        "verify_physics.py",
        "verify_full_logic_suite.py",
        "invert_mini_sha.py",
        "simulate_fractal_mining.py"
    ]
    
    passed = 0
    for s in scripts:
        if run_script(s):
            passed += 1
            
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{len(scripts)} Scripts Passed Verified")
    print(f"{'='*60}")
    
    if passed == len(scripts):
        print("ALL SYSTEMS GREEN. The Theory and Code are consistent.")
    else:
        print("WARNING: Some verifications failed. Check output above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
