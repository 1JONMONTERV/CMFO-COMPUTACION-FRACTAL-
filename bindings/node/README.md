# CMFO Node.js Bindings (Experimental)

Use the Fractal Compute Engine directly from JavaScript/TypeScript.

## Installation
```bash
npm install @cmfo/core
```

## Usage

```javascript
import { T7Matrix, phi_xor } from '@cmfo/core';

// 1. Initialize Engine
const engine = new T7Matrix();

// 2. Logic
console.log(phi_xor(1, 0)); // 1.618...

// 3. Evolve State
const state = [1.0, 0, 0, 0, 0, 0, 0];
const next = engine.evolve(state);
```

See [Main Documentation](../../README.md) for theory.
