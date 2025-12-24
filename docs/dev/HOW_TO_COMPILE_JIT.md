# Cómo Compilar el Motor JIT Nativo (NVRTC)

CMFO v3.0 incluye un puente C++ ("The Sniper Bridge") que permite ejecutar código CUDA compilado dinámicamente.

## Requisitos Previos
1. **NVIDIA CUDA Toolkit** (v11 o superior).
2. **Visual Studio C++ Compiler (MSVC)** instalado y en el PATH (`cl.exe`).

## Pasos de Compilación

### 1. Configurar Entorno
Abre una terminal "x64 Native Tools Command Prompt for VS 2019/2022". Esto asegura que `cl.exe` esté disponible.

### 2. Ejecutar Comando NVCC
Navega a la raíz del repositorio y ejecuta:

```powershell
nvcc -shared -o cmfo_jit.dll src/jit/nvrtc_bridge.cpp -lnvrtc -lcuda
```

*Nota: En Linux, usa `-o libcmfo_jit.so`.*

### 3. Verificar
Si la compilación es exitosa, verás el archivo `cmfo_jit.dll` (o `.so`) en la raíz.

### 4. Ejecución
CMFO detectará automáticamente la librería y activará el modo "Native JIT".
Si no la encuentra, usará el modo "Python Simulation" (lento).
