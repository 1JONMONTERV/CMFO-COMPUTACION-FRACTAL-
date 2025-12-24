# Build script for CMFO native libraries (Windows PowerShell)
# This script compiles the C/C++ core libraries needed by Python and Node.js bindings

Write-Host "=== CMFO Native Library Build Script (Windows) ===" -ForegroundColor Cyan
Write-Host ""

# Configuration
$BUILD_DIR = "core\native\build"
$INSTALL_DIR = "core\native\lib"
$PYTHON_BINDING_DIR = "bindings\python\cmfo"
$NODE_BINDING_DIR = "bindings\node\lib"

# Create directories
New-Item -ItemType Directory -Force -Path $BUILD_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $INSTALL_DIR | Out-Null
New-Item -ItemType Directory -Force -Path "$PYTHON_BINDING_DIR\core" | Out-Null
New-Item -ItemType Directory -Force -Path $NODE_BINDING_DIR | Out-Null

Write-Host "Platform: Windows" -ForegroundColor Green
Write-Host ""

# Build with CMake (if CMakeLists.txt exists)
if (Test-Path "core\native\CMakeLists.txt") {
    Write-Host "Building with CMake..." -ForegroundColor Yellow
    
    Push-Location $BUILD_DIR
    cmake .. -G "Visual Studio 17 2022" -A x64
    cmake --build . --config Release
    Pop-Location
    
    Write-Host "CMake build complete" -ForegroundColor Green
}
else {
    Write-Host "No CMakeLists.txt found, skipping CMake build" -ForegroundColor Yellow
}

# Copy libraries to binding directories
Write-Host ""
Write-Host "Copying libraries to binding directories..." -ForegroundColor Yellow

$dllPath = "$BUILD_DIR\Release\cmfo_core.dll"
if (Test-Path $dllPath) {
    Copy-Item $dllPath "$PYTHON_BINDING_DIR\core\" -Force
    Copy-Item $dllPath $NODE_BINDING_DIR -Force
    Write-Host "Copied cmfo_core.dll" -ForegroundColor Green
}
else {
    Write-Host "Warning: cmfo_core.dll not found at $dllPath" -ForegroundColor Yellow
}

# Build CUDA libraries if available
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    Write-Host ""
    Write-Host "CUDA detected, building CUDA kernels..." -ForegroundColor Yellow
    
    if (Test-Path "core\native\cuda\cmfo_cuda.cu") {
        # One line command to avoid backtick issues
        nvcc -shared -o "$BUILD_DIR\cmfo_cuda.dll" core\native\cuda\cmfo_cuda.cu -O3 -arch=sm_70
        
        Copy-Item "$BUILD_DIR\cmfo_cuda.dll" "$PYTHON_BINDING_DIR\core\" -Force
        Write-Host "Built and copied CUDA library" -ForegroundColor Green
    }
    else {
        Write-Host "CUDA source files not found" -ForegroundColor Yellow
    }
}
else {
    Write-Host ""
    Write-Host "CUDA not available, skipping GPU build" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Build Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Libraries installed to:" -ForegroundColor White
Write-Host "  - Python: $PYTHON_BINDING_DIR\core\" -ForegroundColor Gray
Write-Host "  - Node.js: $NODE_BINDING_DIR\" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "  1. Test Python: cd bindings\python; python -c 'import cmfo; cmfo.info()'" -ForegroundColor Gray
Write-Host "  2. Test Node.js: cd bindings\node; npm test" -ForegroundColor Gray
