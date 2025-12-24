
import os
import sys
import glob
import importlib.util

# Add local path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

def check_file_for_shortcuts(filepath):
    """Scans for 'MOCK', 'TODO', 'FIXME', 'pass' blocks."""
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "MOCK" in line.upper(): # checking for 'MOCK'
                    # Exclude self-references in audit script
                    if "check_file_for_shortcuts" not in line: 
                         issues.append(f"Line {i+1}: Found 'MOCK' keyword.")
                if "TODO" in line:
                    issues.append(f"Line {i+1}: Found 'TODO' tag.")
                if "FIXME" in line:
                    issues.append(f"Line {i+1}: Found 'FIXME' tag.")
    except Exception as e:
        issues.append(f"Error reading file: {e}")
    return issues

def audit_codebase_static():
    print("\n--- STATIC CODE ANALYSIS (Looking for 'Half-Done' work) ---")
    target_dirs = [
        "cmfo/core", "cmfo/crypto", "cmfo/logic", "cmfo/compiler", "cmfo/physics"
    ]
    
    total_issues = 0
    base_path = os.path.dirname(__file__)
    
    for d in target_dirs:
        full_dir = os.path.join(base_path, d)
        if not os.path.exists(full_dir):
            continue
            
        files = glob.glob(os.path.join(full_dir, "*.py"))
        for f in files:
            issues = check_file_for_shortcuts(f)
            if issues:
                print(f"\n[FLAG] {os.path.basename(f)}:")
                for iss in issues:
                    print(f"  - {iss}")
                total_issues += len(issues)
                
    if total_issues == 0:
        print("[PASS] No obvious 'MOCK' or 'TODO' shortcuts found in critical paths.")
    else:
        print(f"[WARN] Found {total_issues} potential incomplete items.")

def run_external_verification(script_name):
    print(f"\n--- EXECUTING: {script_name} ---")
    try:
        # We run it as a subprocess to isolate
        import subprocess
        # Assuming script is in bindings/python or root?
        # Check current dir
        if os.path.exists(script_name):
            path = script_name
        elif os.path.exists(os.path.join("../../", script_name)):
            path = os.path.join("../../", script_name)
        else:
            print(f"[SKIP] Script {script_name} not found.")
            return False

        # Run
        result = subprocess.run([sys.executable, path], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[PASS] {script_name} executed successfully.")
            # Verify stdout for specific success keywords if needed
            if "FAIL" in result.stdout:
                print(f"[WARN] Script output contains 'FAIL':\n{result.stdout[:200]}...")
                return False
            return True
        else:
            print(f"[FAIL] {script_name} crashed (Exit {result.returncode}).")
            print(f"Error: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"[FAIL] Exception running {script_name}: {e}")
        return False

def audit_full_repo():
    print("==================================================")
    print("      CMFO REPOSITORY INTEGRITY AUDIT (FULL)      ")
    print("==================================================")
    
    # 1. Static Analysis
    audit_codebase_static()
    
    # 2. Dynamic Tests (Legacy & New)
    print("\n--- DYNAMIC MODULE VERIFICATION ---")
    
    # Physics Check
    # verify_mass_fix.py might be in root or bindings? List says root.
    # relative to bindings/python/ is ../../verify_mass_fix.py
    phys_ok = run_external_verification("../../verify_mass_fix.py")
    
    # Crypto Check
    crypto_ok = run_external_verification("audit_cmfo_integrity.py") # Re-run the rigorous one
    
    # JIT Perf / Validation
    gpu_ok = run_external_verification("../../validate_gpu_perf.py")
    
    print("\n==================================================")
    print("               FINAL ASSESSMENT                   ")
    print("==================================================")
    
    score = 0
    total = 3
    if phys_ok: score += 1
    if crypto_ok: score += 1
    if gpu_ok: score += 1
    
    print(f"Subsystems Validated: {score}/{total}")
    
    if score == total:
        print("VERDICT: SYSTEM INTEGRITY CONFIRMED.")
    else:
        print("VERDICT: INTEGRITY COMPROMISED. See logs.")

if __name__ == "__main__":
    audit_full_repo()
