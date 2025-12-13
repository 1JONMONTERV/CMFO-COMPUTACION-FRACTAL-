# CMFO-UNIVERSE v1.0
> **Motor de cálculo fractal φ-lógico • Álgebra T⁷ • Acelerado CUDA**

![License](https://img.shields.io/badge/License-Commercial%20Restricted-red)
![Build Status](https://img.shields.io/github/actions/workflow/status/1JONMONTERV/cmfo-universe/ci.yml)
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

## Instalación y Compilación

### Requisitos previos
*   CMake 3.10+
*   Compilador C++ (GCC/Clang/MSVC)
*   **Opcional (para GPU)**: NVIDIA CUDA Toolkit 12.x+ y GPU SM86 (Serie RTX 30/40)

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
2.  Ejecutar script de verificación Python:
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

### Verificación de Tensores T7

Para validar propiedades algebraicas (Conmutatividad, Distributividad, Antisimetría):

```bash
./test_tensor_props
```

### Lógica Phi (Generalización Booleana)

CMFO demuestra que la lógica booleana es un caso degenerado ($\phi \to 1$) de la lógica fractal:

- **Teoría**: [Docs/THEORY_PHI_LOGIC.md](Docs/THEORY_PHI_LOGIC.md)
- **Demo**: Ejecutar `python python/demo_phi_logic.py` para ver la convergencia.

### Aceleración CUDA (GPU)

El núcleo CUDA (`cuda/theta_cmfo_kernel.cu`) implementa actualmente la **Fase 1: Evolución Independiente**.
Cada modo T7 evoluciona con su propia frecuencia fractal $\omega_i \propto \phi \cdot i$.

- **Teoría**: [Docs/THEORY_CUDA.md](Docs/THEORY_CUDA.md)

---

## Ejemplo de Uso

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

## Licencia

**CMFO-UNIVERSE Licencia v1.1 – Uso comercial restringido**
Copyright © 2025 Jonnathan Montero Víquez

*   **Permitido**: Uso personal, académico e investigación no comercial.
*   **Requiere Licencia**: Integración en productos, uso empresarial o distribución comercial.

Para solicitudes de licencia, contactar a: **jmvlavacar@hotmail.com** (San José, Costa Rica).
