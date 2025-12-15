
import os
import sys
import subprocess
import shutil

def print_header():
    print("="*60)
    print("  CMFO v3.0-alpha TACTICAL INSTALLER")
    print("  Codename: 'The Sniper'")
    print("="*60)

def check_cuda():
    print("\n[1] Checking CUDA Toolkit...")
    cuda_path = os.environ.get('CUDA_PATH')
    if cuda_path:
        print(f"    ✅ CUDA_PATH detected: {cuda_path}")
        return True
    
    # Check default path
    default_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
    if os.path.exists(default_path):
        print(f"    ✅ CUDA Found at default: {default_path}")
        os.environ['CUDA_PATH'] = default_path
        return True
        
    print("    ❌ CUDA Toolkit NOT FOUND. Please install CUDA 11.0+")
    return False

def find_vcvars():
    print("\n[2] Search for MSVC Compiler Environment...")
    # Known locations for Enterprise, Community, Professional 2022/2019
    possible_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
    ]
    
    for p in possible_paths:
        if os.path.exists(p):
            print(f"    ✅ Found vcvars64.bat: {p}")
            return p
            
    print("    ⚠️  Visual Studio compiler not found automatically.")
    print("        Compilation might fail unless 'cl.exe' is in PATH.")
    return None

def compile_jit(vcvars_path):
    print("\n[3] Compiling 'The Sniper' Native Bridge (cmfo_jit.dll)...")
    
    build_cmd = "nvcc -shared -o cmfo_jit.dll src/jit/nvrtc_bridge.cpp -lnvrtc -lcuda"
    
    if vcvars_path:
        # Chain setup + compile
        final_cmd = f'call "{vcvars_path}" && {build_cmd}'
        print(f"    Executing: {build_cmd} (via vcvars environment)")
        ret = subprocess.call(f'cmd /c "{final_cmd}"', shell=True)
    else:
        # Try direct
        print(f"    Executing: {build_cmd}")
        ret = subprocess.call(build_cmd, shell=True)
        
    if ret == 0 and os.path.exists("cmfo_jit.dll"):
        print("    ✅ SUCCESS: cmfo_jit.dll generated.")
        return True
    else:
        print("    ❌ FAILURE: Compilation failed.")
        return False

def main():
    print_header()
    
    if not check_cuda():
        sys.exit(1)
        
    vcvars = find_vcvars()
    
    # Check if already compiled
    if os.path.exists("cmfo_jit.dll"):
        print("\n[!] 'cmfo_jit.dll' already exists.")
        choice = input("    Recompile? [y/N]: ").lower()
        if choice != 'y':
            print("    Skipping compilation.")
            sys.exit(0)
    
    success = compile_jit(vcvars)
    
    if success:
        print("\n" + "="*60)
        print("  INSTALLATION COMPLETE: SYSTEM COMBAT READY")
        print("  Run 'python validate_gpu_perf.py' to verify.")
        print("="*60)
    else:
        print("\n  INSTALLATION FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
