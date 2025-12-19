# CMFO: Computación Multidimensional Fractal Orientada

**Sistema de Computación Geométrica en Toro de 7 Dimensiones con Métrica Fractal del Ratio Áureo**

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]() [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]() [![C++](https://img.shields.io/badge/C%2B%2B-17-blue)]() [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]() [![ISO Compliant](https://img.shields.io/badge/ISO%2025010-compliant-blue)]()

---

## 📋 Tabla de Contenidos

- [Visión General](#-visión-general)
- [Inicio Rápido](#-inicio-rápido)
- [Estructura del Repositorio](#-estructura-del-repositorio)
- [Documentación](#-documentación)
- [Componentes Principales](#-componentes-principales)
- [Aplicaciones](#-aplicaciones)
- [Desarrollo](#-desarrollo)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)

---

## 🌟 Visión General

CMFO es un framework de computación geométrica rigurosamente formalizado que opera en un toro de 7 dimensiones (T⁷) equipado con una métrica fractal basada en el ratio áureo (φ). A diferencia de enfoques estadísticos o semánticos, CMFO proporciona:

### Características Clave

- **🔬 Geometría Pura**: Sin interpretación semántica, solo estructura matemática
- **✅ Verificación Formal**: Todos los teoremas probados y testeados
- **🎯 Determinista**: Sin aleatoriedad, completamente reproducible
- **🔐 Post-Quantum Secure**: Seguridad geométrica, no criptográfica
- **🌍 Auditable Internacionalmente**: Cumple con estándares ISO, IEEE, FAIR

### Innovación Principal

Métrica fractal con pesos del ratio áureo (φ) que permite **compresión >100x** manteniendo reconstrucción exacta.

### Fundamento Matemático

```
Toro 7D:        T⁷ = (S¹)⁷ ≅ ℝ⁷/(2πℤ)⁷
Métrica Fractal: g_φ = Σᵢ₌₁⁷ λᵢ dθᵢ²  donde λᵢ = φ^(i-1)
Distancia:      d_φ(θ, η) = √(Σᵢ₌₁⁷ λᵢ Δᵢ²)
```

### 🧠 Diccionario Técnico Fundamental

Para entender CMFO, es vital distinguir sus términos de la computación clásica:

#### 1. Raíz Fractal (Fractal Root `ℛφ`) vs Raíz Cuadrada (`√`)
- **Clásico**: `√x` solo sirve para áreas cuadradas.
- **CMFO**: `ℛφ(x)` encuentra la "semilla" geométrica de cualquier estructura jerárquica. Converge asintóticamente a la unidad, lo que permite estabilizar sistemas caóticos.

#### 2. Lógica Phi (`∧φ`) vs Lógica Booleana (`AND`)
- **Clásico**: `1 AND 0 = 0` (Pérdida de información).
- **CMFO**: Mantiene grados de coherencia. Es reversible. Un "Falso" (0.0) es distinto de un "Casi Falso" (0.1).
- **Analogía**: Interruptor ON/OFF vs Regulador de Intensidad (Dimmer).

#### 3. Tensor7 (`T⁷`) vs Tensor Clásico
- **Clásico**: Matriz pasiva de números.
- **CMFO**: Objeto geométrico activo en un toro 7D. Al interactuar, "evoluciona" siguiendo reglas de fase, no solo suma algébrica.

#### 4. Computación Reversible (Landauer Zero)
- **Clásico**: Borrar un bit genera calor (`kT ln(2)`).
- **CMFO**: Al usar operadores reversibles, no se destruye información, el costo energético teórico es **cero**.

#### 5. Fractal NPU vs CPU/GPU
- **Clásico**: Procesa bits lineales.
- **CMFO**: Procesa ondas y geometrías. Una instrucción `F_ROOT` equivale a cientos de operaciones de punto flotante clásicas.

*(Ver el [Diccionario Técnico Completo](docs/manual/CMFO_DICCIONARIO_TECNICO.md) para más detalles)*

---

## 🚀 Inicio Rápido

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-.git
cd CMFO-COMPUTACION-FRACTAL-

# Instalar dependencias Python
pip install -r requirements.txt

# Compilar componentes nativos (opcional)
cd src/jit
cmake . && make
```

### Primer Uso

```python
import cmfo

# Crear punto en T⁷
punto = cmfo.phi_encode(42.0)

# Operación fractal
resultado = cmfo.phi_add(punto, cmfo.phi_encode(13.0))

# Distancia geométrica
distancia = cmfo.phi_distance(punto, resultado)
```

### Ejecutar Tests

```bash
# Todos los tests
python -m pytest tests/ -v

# Tests específicos
python -m pytest tests/test_geometric_foundation.py -v

# Suite de verificación completa
python experiments/run_all_proofs.py
```

---

## 📁 Estructura del Repositorio

### Directorios Principales

```
CMFO-COMPUTACION-FRACTAL-/
│
├── 📚 docs/                    # Documentación completa
│   ├── theory/                 # Documentación teórica (10+ archivos)
│   ├── reports/                # Reportes de investigación
│   ├── api/                    # Documentación de API
│   └── guide/                  # Guías de usuario
│
├── 🔬 bindings/                # Bindings de lenguajes
│   ├── python/                 # Package Python (cmfo)
│   └── node/                   # Package Node.js
│
├── 🧪 experiments/             # Experimentos de investigación (60+ archivos)
│   ├── reproducibility/        # Scripts de verificación
│   └── benchmarks/             # Benchmarks de rendimiento
│
├── 🏆 products/                # 🆕 Catálogo de Productos
│   ├── CATALOGO_PRODUCTOS.md
│   └── CERTIFICADO_AUDITORIA_FINAL.md
│
├── ✅ tests/                   # Suite de tests (34 archivos)
│   ├── test_geometric_foundation.py
│   ├── test_boolean_proof.py
│   └── performance/            # Tests de rendimiento
│
├── 💻 src/                     # Código fuente C++
│   └── jit/                    # JIT compiler
│
├── 🎯 examples/                # Ejemplos de uso (34 archivos)
│
├── 🛠️ cmfo/                    # Core Python package
│   ├── core/                   # Operadores core
│   ├── crypto/                 # SHA-256d reversible
│   ├── topology/               # Generador procedural 2^512
│   ├── logic/                  # Circuitos lógicos
│   └── physics/                # Física computacional
│
├── 📊 data/                    # Datasets
│   ├── FRACTAL_OMNIVERSE.csv          # 136 KB
│   └── FRACTAL_OMNIVERSE_RECURSIVE.csv # 637 KB (20k relaciones)
│
└── 🌐 web/                     # Interfaz web
```

### Archivos de Configuración

| Archivo | Propósito |
|---------|-----------|
| `pyproject.toml` | Configuración Python package |
| `setup.py` | Setup Python |
| `requirements.txt` | Dependencias Python |
| `CONTRIBUTING.md` | Guía de contribución |
| `LICENSE` | Licencia MIT |

---

## 📚 Documentación

### Documentación Teórica (`docs/theory/`)

#### Especificaciones Principales

1. **[CMFO_MASTER.tex](docs/theory/CMFO_MASTER.tex)** - Documento maestro LaTeX
   - Framework algebraico completo
   - φ-logic y interpretaciones físicas
   - Aspectos computacionales

2. **[CMFO_COMPLETE_ALGEBRA.md](docs/theory/CMFO_COMPLETE_ALGEBRA.md)** - Álgebra completa
   - Definiciones formales
   - Teoremas y pruebas
   - Operadores fundamentales

3. **[SPANISH_ALGEBRA_SPEC.md](docs/theory/SPANISH_ALGEBRA_SPEC.md)** - Álgebra de Español
   - Interfaz de lenguaje natural
   - Compilación español → operadores CMFO
   - Procesamiento determinista de lenguaje natural

4. **[BOOLEAN_LOGIC_COMPLETE.md](docs/theory/BOOLEAN_LOGIC_COMPLETE.md)** - Lógica Booleana
   - Absorción de lógica booleana clásica
   - Pruebas de completitud funcional
   - Extensión continua a lógica difusa

5. **[DETERMINISTIC_AI_SPEC.md](docs/theory/DETERMINISTIC_AI_SPEC.md)** - IA Determinista
   - Garantías de reproducibilidad bit-exacta
   - Aplicaciones en sistemas críticos
   - Capacidades de verificación formal

#### Fuentes LaTeX (`docs/theory/latex_source/`)

29 archivos LaTeX organizados por tema:
- `01-fundamentals/` - Fundamentos (torus, Hopf algebra, teoremas)
- `02-physics/` - Física (validación, estructura fina, masas hadrónicas)
- `03-biology/` - Biología (código genético fractal)
- `04-computation/` - Computación (computación fractal)

### Reportes de Investigación (`docs/reports/`)

- **Mining & Optimization**
  - `MINING_OPTIMIZATION_REPORT.md` - Optimización de minería
  - `MINING_TOPOLOGY_REPORT.md` - Topología de minería
  - `HYPER_RESOLUTION_REPORT.md` - Hiper-resolución
  - `SYNTHESIS_NON_BRUTE_FORCE.md` - Síntesis no-brute-force

- **System Reports**
  - `AUTONOMOUS_MINING_SYSTEM.md` - Sistema autónomo
  - `GPU_MINING_ARCHITECTURE.md` - Arquitectura GPU
  - `GEOMETRIC_MINING_SCHEDULER.md` - Scheduler geométrico

### Especificaciones Técnicas (`docs/`)

- `SHA256D_FRACTAL_SPEC.md` - Especificación SHA-256d fractal
- `COMPLETE_SYSTEM_SPECIFICATION.md` - Especificación completa del sistema
- `FRACTAL_TORUS_REPORT.md` - Reporte del toro fractal

### Guías de Usuario

- `MANUAL_USUARIO.md` - Manual de usuario
- `FAQ.md` - Preguntas frecuentes
- `REPRODUCIBILITY.md` - Guía de reproducibilidad
- `BUILD.md` - Guía de compilación

---

## 🔧 Componentes Principales

### 1. Core CMFO (`cmfo/`)

#### Operadores Fundamentales

```python
# Operadores φ (phi)
cmfo.phi_add(a, b)      # Suma con ratio áureo
cmfo.phi_sub(a, b)      # Resta con ratio áureo
cmfo.phi_mul(a, b)      # Multiplicación
cmfo.phi_distance(a, b) # Distancia geométrica

# Operadores tensoriales
cmfo.tensor_mul(a, b)   # Multiplicación tensorial
cmfo.tensor_div(a, b)   # División tensorial

# Operadores lógicos
cmfo.f_and(a, b)        # AND continuo
cmfo.f_or(a, b)         # OR continuo
cmfo.f_not(a)           # NOT continuo
cmfo.f_xor(a, b)        # XOR continuo
```

### 2. GPU Bridge (`bindings/python/cmfo/bridge.py`)

Interfaz Python ↔ C++ GPU para aceleración:

```python
from cmfo import bridge

# Operación acelerada por GPU
resultado = bridge.gpu_compute(data)
```

### 3. Procedural Space Generator (`bindings/python/cmfo/topology/procedural_512.py`)

Generador procedural para espacio 2^512:

```python
from cmfo.topology import ProceduralSpace512

space = ProceduralSpace512()

# Generar bloque desde coordenadas
block = space.coords_to_block(x=1000, y=2000)

# Mapeo inverso
x, y = space.block_to_coords(block)

# Muestrear región
blocks = space.sample_region(center_x=500, center_y=500, radius=10, count=100)
```

### 4. SHA-256d Reversible (`bindings/python/cmfo/crypto/sha256d_reversible.py`)

Implementación reversible de SHA-256d:

```python
from cmfo.crypto import sha256d_reversible

# Hash reversible
hash_result = sha256d_reversible.hash(data)

# Verificación
is_valid = sha256d_reversible.verify(data, hash_result)
```

### 5. Circuit Physics (`bindings/python/cmfo/logic/circuits.py`)

Análisis de propiedades físicas de circuitos:

```python
from cmfo.logic import circuits

# Crear circuito
circuit = circuits.LogicCircuit()

# Analizar métricas
metrics = circuit.analyze_physics()
```

---

## 🎯 Aplicaciones

### 1. Mining Intelligence System

Sistema de IA para optimización de minería:

```bash
python cmfo_mining_ai.py
```

**Características**:
- Optimización geométrica de búsqueda
- Scheduler inteligente
- Reducción de espacio de búsqueda

### 2. Álgebra de Español

Interfaz de lenguaje natural en español:

```bash
python experiments/demo_spanish_algebra.py
```

**Ejemplos**:
- "suma cinco más tres" → 8.0
- "el doble de diez" → 20.0
- "raíz cuadrada de dieciséis" → 4.0

### 3. IA Determinista

Sistema de IA con reproducibilidad bit-exacta:

```bash
python experiments/demo_deterministic_ai.py
```

**Aplicaciones**:
- Aviación (DO-178C)
- Medicina (FDA Class III)
- Finanzas (regulación)

### 4. Knowledge Library

Biblioteca de 20,000 relaciones semánticas recursivas:

```python
import pandas as pd

# Cargar biblioteca
df = pd.read_csv('FRACTAL_OMNIVERSE_RECURSIVE.csv')

# Explorar relaciones
print(df.head())
```

Ver: `THE_LIBRARY_REPORT.md`

---

## 🛠️ Desarrollo

### Estructura de Desarrollo

```
Development Workflow:
1. Fork & Clone
2. Create feature branch
3. Implement changes
4. Run tests
5. Submit PR
```

### Ejecutar Tests

```bash
# Tests unitarios
python -m pytest tests/ -v

# Tests de integración
python -m pytest tests/test_integration.py -v

# Tests de rendimiento
python -m pytest tests/performance/ -v

# Suite completa de verificación
python experiments/run_all_proofs.py
```

### Verificación Triple

Sistema de verificación triple para máxima confiabilidad:

```bash
# Verificación Python
python experiments/reproducibility/verify_fractal_memory.py

# Verificación JavaScript
node bindings/node/tests/verify_memory.js

# Verificación completa
python experiments/reproducibility/verify_full_logic_suite.py
```

### Compilar Componentes Nativos

```bash
cd src/jit
cmake .
make
```

Genera: `cmfo_jit.dll` (Windows) o `cmfo_jit.so` (Linux)

---

## 👥 Contribuir

### Proceso de Contribución

1. **Leer** [`CONTRIBUTING.md`](CONTRIBUTING.md)
2. **Fork** el repositorio
3. **Crear** branch: `git checkout -b feature/mi-feature`
4. **Implementar** cambios con tests
5. **Verificar**: `python -m pytest tests/ -v`
6. **Commit**: `git commit -m "feat: descripción"`
7. **Push**: `git push origin feature/mi-feature`
8. **Crear** Pull Request

### Estándares

- ✅ **Commits firmados** (GPG)
- ✅ **Tests passing** (100%)
- ✅ **Documentación** actualizada
- ✅ **Código formateado** (black, isort)
- ✅ **Sin randomness** en core

### Áreas de Contribución

- 🔬 **Matemáticas**: Extensiones teóricas
- 💻 **Código**: Optimizaciones, nuevas features
- 📚 **Documentación**: Guías, tutoriales
- 🧪 **Tests**: Cobertura, casos edge
- 🌍 **Traducciones**: Internacionalización

---

## 📊 Estadísticas del Repositorio

| Métrica | Valor |
|---------|-------|
| **Archivos de código** | ~200+ |
| **Documentación** | 134 archivos .md |
| **Tests** | 34 archivos |
| **Experimentos** | 60+ scripts |
| **Líneas de código** | ~50,000+ |
| **Idiomas** | Python, C++, JavaScript, LaTeX |

---

## 🔗 Enlaces Importantes

### Documentación

- [Visión del Proyecto](VISION.md)
- [Roadmap](ROADMAP.md)
- [Changelog](CHANGELOG.md)
- [Limitaciones Conocidas](KNOWN_LIMITATIONS.md)

### Reportes

- [Reporte de Auditoría](AUDIT_REPORT.md)
- [Certificado de Verificación](VERIFICATION_CERTIFICATE.md)
- [Reporte de Reproducibilidad](REPRODUCIBILITY.md)

### Guías

- [Manual de Usuario](MANUAL_USUARIO.md)
- [Guía de Compilación](BUILD.md)
- [FAQ](FAQ.md)

---

## 📜 Licencia

Este proyecto está licenciado bajo la **Licencia MIT**.

```
Copyright (c) 2025 Jonathan Montero Viques

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

Ver [LICENSE](LICENSE) para el texto completo.

---

## 🙏 Agradecimientos

### Fundamentos Matemáticos

- M. Spivak: *Comprehensive Introduction to Differential Geometry*
- M. P. do Carmo: *Riemannian Geometry*
- J. M. Lee: *Introduction to Riemannian Manifolds*

### Inspiración

- B. B. Mandelbrot: *The Fractal Geometry of Nature*
- K. Falconer: *Fractal Geometry*

---

## 📞 Contacto

- **Issues**: [GitHub Issues](https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-/issues)
- **Discussions**: [GitHub Discussions](https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-/discussions)
- **Website**: [GitHub Pages](https://1jonmonterv.github.io/CMFO-COMPUTACION-FRACTAL-/)

---

## 🎓 Citación

Si usas CMFO en tu investigación, por favor cita:

```bibtex
@software{cmfo2025,
  title={CMFO: Computación Multidimensional Fractal Orientada},
  author={Montero Viques, Jonathan},
  year={2025},
  url={https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-},
  note={Sistema de computación geométrica en toro 7D con métrica fractal}
}
```

---

<div align="center">

**Estado**: Production Ready | **Tests**: Passing | **Standards**: ISO/IEEE Compliant

**Última Actualización**: 2025-12-18

Made with ❤️ and φ (golden ratio)

</div>
