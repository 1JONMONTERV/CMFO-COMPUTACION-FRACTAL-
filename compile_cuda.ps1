# Script de compilación CUDA para CMFO
# Configura el entorno y compila el kernel nativo

$ErrorActionPreference = "Continue"

Write-Host "=== COMPILANDO CMFO CUDA KERNEL ===" -ForegroundColor Cyan

# Rutas
$msvcPath = "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.44.35207"
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
$srcFile = "src\cmfo_kernel.cu"
$outFile = "build\cmfo_core.dll"

# Verificar que existen
if (!(Test-Path $msvcPath)) {
    Write-Host "[ERROR] MSVC no encontrado en: $msvcPath" -ForegroundColor Red
    exit 1
}

# Configurar entorno
$env:PATH = "$msvcPath\bin\Hostx64\x64;$cudaPath\bin;$env:PATH"
$env:INCLUDE = "$msvcPath\include;$cudaPath\include;C:\Program Files (x86)\Windows Kits\10\Include\10.0.22621.0\ucrt"
$env:LIB = "$msvcPath\lib\x64;$cudaPath\lib\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64"

# Crear directorio build
if (!(Test-Path "build")) { 
    New-Item -ItemType Directory -Path "build" | Out-Null 
}

Write-Host "[INFO] MSVC: $msvcPath" -ForegroundColor Gray
Write-Host "[INFO] CUDA: $cudaPath" -ForegroundColor Gray
Write-Host "[INFO] Compilando: $srcFile" -ForegroundColor Yellow

# Compilar
$nvccArgs = @(
    $srcFile,
    "-o", $outFile,
    "--shared",
    "-arch=sm_86",
    "-O3",
    "--use_fast_math",
    "-Xcompiler", "/MD"
)

Write-Host "[CMD] nvcc $($nvccArgs -join ' ')" -ForegroundColor DarkGray

& nvcc @nvccArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[SUCCESS] DLL compilada exitosamente!" -ForegroundColor Green
    Get-Item $outFile | Format-List Name, Length, LastWriteTime
} else {
    Write-Host ""
    Write-Host "[FAIL] Error de compilación (código: $LASTEXITCODE)" -ForegroundColor Red
    exit 1
}
