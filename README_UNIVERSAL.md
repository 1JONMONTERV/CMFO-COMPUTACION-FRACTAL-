# CMFO - Computación Fractal Orientada a Objetos
## Universal Platform for Fractal Computation

[![License](https://img.shields.io/badge/license-Dual-blue.svg)](CMFO_LICENSING_POLICY.md)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](CHANGELOG.md)
[![Certification](https://img.shields.io/badge/certification-53%25-yellow.svg)](CMFO_UNIVERSAL_CERTIFICATION_MATRIX.md)

**CMFO** is a foundational computational system based on 7-dimensional fractal geometry. It provides:

- ✅ **100% CMFO** - No external AI dependencies
- ✅ **Mathematically Certified** - Formal axioms and proofs
- ✅ **Universal Platform** - Runs everywhere (C, Python, Rust, WASM, etc.)
- ✅ **Reversible Memory** - Fractal memory with audit trails
- ✅ **Open Knowledge** - Specifications public, use contractual

---

## Quick Start

### Python

```python
from cmfo import CMFO

with CMFO() as cmfo:
    # Parse text to semantic vector
    vec = cmfo.parse("verdad")
    print(f"verdad = {vec}")
    
    # Solve equation
    solution = cmfo.solve("2x + 3 = 7")
    print(solution)
```

### Rust

```rust
use cmfo::CMFO;

fn main() {
    let cmfo = CMFO::new().unwrap();
    let vec = cmfo.parse("verdad").unwrap();
    println!("verdad = {:?}", vec);
}
```

### JavaScript/WASM

```javascript
import { CMFO } from '@cmfo/wasm';

const cmfo = await CMFO.init();
const vec = cmfo.parse("verdad");
console.log("verdad =", vec);
```

---

## Features

### Mathematical Foundation
- 7D torus geometry (T^7_φ)
- Fractal metric with golden ratio weights
- Formal axioms (6 core + derived properties)
- 82% mathematically certified

### Computational Core
- C ABI (stable forever)
- WebAssembly (universal deployment)
- Multi-language SDKs (Python, Rust, JS, Java, C#, Swift)
- O(1) operations (rotations)

### Applications
- **Education**: Math tutor (10th grade)
- **Enterprise**: Governance and compliance
- **Research**: Formal verification
- **Security**: Audit-locked memory

---

## Installation

### From Source

```bash
# Clone repository
git clone https://github.com/cmfo/cmfo.git
cd cmfo

# Build core
cd core && mkdir build && cd build
cmake .. && make

# Install Python SDK
cd ../../sdks/python
pip install -e .

# Install Rust SDK
cd ../rust
cargo build --release
```

### Pre-built Packages

```bash
# Python
pip install cmfo

# Rust
cargo add cmfo

# JavaScript
npm install @cmfo/wasm
```

---

## Documentation

- **[Getting Started](docs/getting-started/)** - Quick tutorials
- **[API Reference](docs/api/)** - Complete API docs
- **[Specifications](docs/specification/)** - Formal specifications
- **[Certification](CMFO_UNIVERSAL_CERTIFICATION_MATRIX.md)** - Certification status
- **[Licensing](CMFO_LICENSING_POLICY.md)** - License terms

---

## Architecture

```
CORE (C/Rust)
    ↓
ABI (C-compatible)
    ↓
┌─────────┬─────────┬─────────┬─────────┐
│ Python  │  Rust   │   JS    │  Java   │ SDKs
└─────────┴─────────┴─────────┴─────────┘
    ↓
┌─────────┬─────────┬─────────┐
│  REST   │  gRPC   │ GraphQL │ APIs
└─────────┴─────────┴─────────┘
    ↓
┌─────────┬─────────┬─────────┬─────────┐
│ Desktop │ Mobile  │   Web   │   CLI   │ Apps
└─────────┴─────────┴─────────┴─────────┘
```

---

## Certification Status

| Layer | Completion |
|-------|-----------|
| Mathematical | 82% |
| Computational | 81% |
| Semantic | 67% |
| Physical | 8% |
| Cognitive | 0% |
| Metatheoretical | 33% |
| **Overall** | **53%** |

See [CMFO_UNIVERSAL_CERTIFICATION_MATRIX.md](CMFO_UNIVERSAL_CERTIFICATION_MATRIX.md) for details.

---

## Licensing

CMFO uses a **dual licensing model**:

- **Study/Research**: Open (free for academic use)
- **Commercial/Enterprise**: Closed (requires license)

See [CMFO_LICENSING_POLICY.md](CMFO_LICENSING_POLICY.md) for complete terms.

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Key areas**:
- Core C implementation
- SDK improvements
- Documentation
- Tests and benchmarks

---

## Citation

If you use CMFO in academic work, please cite:

```bibtex
@software{cmfo2025,
  title={CMFO: Computación Fractal Orientada a Objetos},
  author={CMFO Development Team},
  year={2025},
  url={https://github.com/cmfo/cmfo},
  version={1.0.0}
}
```

---

## Contact

- **Website**: https://cmfo.org (coming soon)
- **Email**: info@cmfo.org
- **GitHub**: https://github.com/cmfo/cmfo
- **Discussions**: https://github.com/cmfo/cmfo/discussions

---

## Status

**Current Phase**: Core Implementation  
**Next Milestone**: 75% Certification (Semantic + Metatheory)  
**Target**: Production Release Q2 2025

---

**Built with ❤️ using pure fractal geometry**
