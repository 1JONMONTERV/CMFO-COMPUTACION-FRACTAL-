import os

def generate():
    if not os.path.exists("compiler_path.txt"):
        print("ERROR: compiler_path.txt not found")
        return

    with open("compiler_path.txt", "r") as f:
        vcvars_path = f.read().strip()
    
    print(f"Generating build script using: {vcvars_path}")
    
    bat_content = f"""@echo off
echo [AUTO] Setting up environment...
call "{vcvars_path}"

echo [AUTO] Compiling CUDA Kernel...
nvcc -shared -o cmfo_cuda.dll kernel_cmfo.cu
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Compilation failed.
    exit /b 1
)

echo [AUTO] Running Native Benchmark...
python benchmark_cuda_native.py
"""

    with open("AUTO_COMPILE.bat", "w") as f:
        f.write(bat_content)
    
    print("SUCCESS: AUTO_COMPILE.bat created.")

if __name__ == "__main__":
    generate()
