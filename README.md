# CMFO-UNIVERSE v1.0
> **Motor de cálculo fractal φ-lógico • Álgebra T⁷ • Acelerado CUDA**

![License](https://img.shields.io/badge/License-Commercial%20Restricted-red)
![Build Status](https://img.shields.io/github/actions/workflow/status/1JONMONTERV/cmfo-universe/ci.yml)
[![codecov](https://codecov.io/gh/1JONMONTERV/cmfo-universe/branch/main/graph/badge.svg)](https://codecov.io/gh/1JONMONTERV/cmfo-universe)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![CUDA](https://img.shields.io/badge/CUDA-SM86-green)

**CMFO-UNIVERSE** es el primer SDK público que implementa computación fractal determinista basada en geometría T⁷, reemplazando completamente la lógica booleana tradicional por **φ-logic**, una lógica operativa continua y geométrica.

## Características Principales

*   **Núcleo matemático fractal (C)**: Álgebra vectorial y matricial exacta en dimensión 7.
*   **Precisión Exacta**: Determinante e inversa 7×7 exacta (sin errores de punto flotante).
*   **Tensores**: Tensores fractales de segundo orden.
*   **Alto Rendimiento**: Kernels CUDA altamente optimizados para SM86.
*   **Fiabilidad**: Pruebas automáticas de CPU/GPU.
*   **Estabilidad**: API pública estable, portátil y reproducible.

Diseñado para investigación avanzada, ingeniería industrial, física computacional, IA fractal y minería criptográfica estructural.

## Quality Gates

CMFO-UNIVERSE mantiene estándares rigurosos de calidad de código para garantizar confiabilidad y precisión numérica:

| Métrica | Requisito | Estado |
|---------|-----------|--------|
| **Cobertura de Código** | ≥ 80% (proyecto), ≥ 70% (patch) | [![codecov](https://codecov.io/gh/1JONMONTERV/cmfo-universe/branch/main/graph/badge.svg)](https://codecov.io/gh/1JONMONTERV/cmfo-universe) |
| **Tests Unitarios** | 100% pass rate | ![Build Status](https://img.shields.io/github/actions/workflow/status/1JONMONTERV/cmfo-universe/ci.yml) |
| **Estabilidad Numérica** | Tolerancia < 1e-12 | ✅ Verificado |
| **Compilación Multi-Plataforma** | Linux, Windows, macOS | ✅ CI Activo |

**Dashboard de Cobertura**: [codecov.io/gh/1JONMONTERV/cmfo-universe](https://codecov.io/gh/1JONMONTERV/cmfo-universe)

### Estándares de Calidad

- **Cobertura C/C++**: Medida con `lcov` y `gcov`, incluye todos los archivos en `src/`
- **Cobertura Python**: Medida con `pytest-cov`, incluye módulo `cmfo`
- **Tolerancias Numéricas**: Documentadas en [Docs/NUMERICAL_TOLERANCES.md](Docs/NUMERICAL_TOLERANCES.md)
- **CI/CD**: Ejecuta automáticamente en cada push y pull request
- **Revisión de Código**: Todas las PRs requieren aprobación y pasar quality gates

---

## Supported Environments

CMFO-UNIVERSE is tested and supported across multiple platforms and Python versions.

### Operating Systems

| OS | Versions | Architectures | Status |
|----|----------|---------------|--------|
| **Linux** | Ubuntu 20.04+, Debian 11+, RHEL 8+, Fedora 35+ | x86_64 | ✅ Fully Tested |
| **Windows** | Windows 10/11 | x64 | ✅ Fully Tested |
| **macOS** | macOS 11+ (Big Sur and later) | x86_64, ARM64 (M1/M2/M3) | ✅ Fully Tested |

### Python Versions

![Python 3.8](https://img.shields.io/badge/Python-3.8-blue)
![Python 3.9](https://img.shields.io/badge/Python-3.9-blue)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)

All Python versions from 3.8 to 3.12 are actively tested in CI.

### Compilers

| Platform | Compiler | Minimum Version |
|----------|----------|-----------------|
| Linux | GCC | 9.0+ |
| Linux | Clang | 10.0+ |
| Windows | MSVC | 2019+ (v142) |
| Windows | MinGW-w64 | 8.0+ |
| macOS | Clang (Xcode) | 12.0+ |

### Optional: CUDA Support

- **CUDA Toolkit**: 11.8+ or 12.x
- **GPU**: NVIDIA with Compute Capability SM86 (RTX 30/40 series)
- **Driver**: NVIDIA driver 520+ (Linux), 527+ (Windows)

For detailed installation instructions, see [Docs/INSTALLATION.md](Docs/INSTALLATION.md).

---

## Arquitectura del Sistema

```
cmfo-universe/
├── include/        # Cabeceras (.h)
├── src/            # Código fuente C (Núcleo)
├── cuda/           # Kernels CUDA (.cu)
├── python/         # SDK de Python
├── tests/          # Pruebas unitarias
├── CMakeLists.txt  # Configuración de construcción
└── README.md       # Documentación e información
```

## CI/CD Matrix

CMFO-UNIVERSE is continuously tested across multiple platforms and Python versions:

| Platform | Python 3.8 | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 |
|----------|------------|------------|-------------|-------------|-------------|
| **Ubuntu Latest** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Windows Latest** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **macOS Latest** | ✅ | ✅ | ✅ | ✅ | ✅ |

**Build Status**: ![CI](https://img.shields.io/github/actions/workflow/status/1JONMONTERV/cmfo-universe/ci.yml?branch=main)  
**Coverage**: [![codecov](https://codecov.io/gh/1JONMONTERV/cmfo-universe/branch/main/graph/badge.svg)](https://codecov.io/gh/1JONMONTERV/cmfo-universe)

View detailed test results: [GitHub Actions](https://github.com/1JONMONTERV/cmfo-universe/actions)

---

## Instalación y Compilación

### Instalación Rápida (Recomendada)

```bash
pip install cmfo-universe
```

### Requisitos previos (Compilación desde Fuente)
*   CMake 3.18+
*   Compilador C++ (GCC 9+/Clang 10+/MSVC 2019+)
*   Python 3.8+ (para bindings de Python)
*   **Opcional (para GPU)**: NVIDIA CUDA Toolkit 12.x+ y GPU SM86 (Serie RTX 30/40)

**Para instrucciones detalladas de instalación por plataforma, consulte [Docs/INSTALLATION.md](Docs/INSTALLATION.md).**

### Compilación (CPU)

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Ejecutar Pruebas
```bash
cd build
ctest --output-on-failure
```

### Verificación de Solitones (Sine-Gordon)

Para validar la conservación de energía y las propiedades topológicas:

1.  Ejecutar la simulación C (genera CSV):
    ```bash
    ./test_soliton
    ```
2.  **Visualizar Resultados (Nuevo)**:
    Generar un GIF animado de la colisión:
    ```bash
    python python/visualize_soliton.py
    ```
3.  Ejecutar script de verificación numérica:
    ```bash
    python python/verify_soliton.py
    ```
    Consulte [Docs/THEORY_SOLITON.md](Docs/THEORY_SOLITON.md) para detalles teóricos.

### Verificación de Estabilidad Matricial T7

Para validar que la inversión de matrices 7x7 es numéricamente estable:

```bash
# Ejecutar test de estrés en C
./test_mat7_stab
```

### Interoperabilidad C++ (Nuevo)
Ejemplo de API moderna (`tests/demo_cpp.cpp`):
```cpp
cmfo::Vector7 a = {1,1,1,1,1,1,1};
cmfo::Vector7 b = {2,2,2,2,2,2,2};
cmfo::Vector7 c = a ^ b; // Operador sobrecargado llamando al núcleo C
```

### Verificación de Tensores T7

Para validar propiedades algebraicas (Conmutatividad, Distributividad, Antisimetría):

```bash
./test_tensor_props
```

### Lógica Phi (Generalización Booleana)

CMFO demuestra que la lógica booleana es un caso degenerado ($\phi \to 1$) de la lógica fractal:

- **Teoría**: [Docs/THEORY_PHI_LOGIC.md](Docs/THEORY_PHI_LOGIC.md)
- **Comparación Teórica**: [Docs/THEORY_COMPARISON.md](Docs/THEORY_COMPARISON.md) - Análisis riguroso de CMFO vs lógica booleana
- **Demo**: Ejecutar `python python/demo_phi_logic.py` para ver la convergencia.

### Aceleración CUDA (GPU)

El núcleo CUDA (`cuda/theta_cmfo_kernel.cu`) implementa actualmente la **Fase 1: Evolución Independiente**.
Cada modo T7 evoluciona con su propia frecuencia fractal $\omega_i \propto \phi \cdot i$.

- **Teoría**: [Docs/THEORY_CUDA.md](Docs/THEORY_CUDA.md)

### Python Bindings (Nuevo)
El SDK ahora incluye visualización y wrappers directos:
- `python/visualize_soliton.py`: Generador de GIFs.
- `python/cmfo_wrapper.py`: Interfaz directa con C.

### Tests de Estrés (Máximo Nivel)
Para validar la robustez del sistema bajo carga extrema:
```bash
python python/stress_test_benchmark.py
```
Esto ejecutará:
1.  **Flood Test**: 100k operaciones tensoriales.
2.  **Stability Test**: Inversión de 1k matrices aleatorias.
3.  **Benchmark**: Medición de rendimiento (ops/sec).

### Interoperabilidad Node.js (Nuevo)
Ejemplo de uso desde JavaScript (requiere `npm install`):
```bash
cd node
node index.js
```
Esto carga `cmfo_core.dll` usando FFI y ejecuta operaciones tensoriales nativas.

---

## Ejemplo de Uso

### Python (Recomendado)

```python
from cmfo import CMFOCore

# Inicializar núcleo CMFO
core = CMFOCore()

# Definir vectores T⁷
a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
b = [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

# Producto tensorial
result = core.tensor7(a, b)
print(f"Tensor product: {result}")
```

**Salida esperada:**
```
Tensor product: [7.0, 12.0, 15.0, 16.0, 15.0, 12.0, 7.0]
```

**Más ejemplos**: Ver directorio [`examples/`](examples/) para demostraciones completas.

### C++ (CPU)
```c
#include "cmfo_core.h"

int main() {
    double a[7] = {1,2,3,4,5,6,7};
    double b[7] = {6,5,4,3,2,1,0};
    double out[7];
    
    cmfo_tensor7(out, a, b);
    cmfo_print_vec7(out);
    return 0;
}
```

**Salida esperada:**
```
[6.000, 10.000, 12.000, 12.000, 10.000, 6.000, 0.000]
```

### CUDA
```cpp
#include "cmfo_kernels.cuh"

__global__ void demo(double* out) {
    double a[7] = {1,2,3,4,5,6,7};
    double b[7] = {6,5,4,3,2,1,0};
    cmfo_dynamics_gpu(out, a, b); // Nuevo kernel dinámico
}
```

---

## Releases

Las versiones estables están disponibles en [GitHub Releases](https://github.com/1JONMONTERV/cmfo-universe/releases).

Cada release incluye:
- ✅ Ruedas Python multi-plataforma (Linux, Windows, macOS)
- ✅ Checksums SHA256 para verificación de integridad
- ✅ SBOM (Software Bill of Materials) para transparencia de dependencias
- ✅ Notas de versión detalladas

**Última versión**: [![GitHub release](https://img.shields.io/github/v/release/1JONMONTERV/cmfo-universe)](https://github.com/1JONMONTERV/cmfo-universe/releases/latest)

---

## Licencia

**CMFO-UNIVERSE Licencia v1.1 – Uso comercial restringido**
Copyright © 2025 Jonnathan Montero Víquez

*   **Permitido**: Uso personal, académico e investigación no comercial.
*   **Requiere Licencia**: Integración en productos, uso empresarial o distribución comercial.

Para solicitudes de licencia, contactar a: **jmvlavacar@hotmail.com** (San José, Costa Rica).
