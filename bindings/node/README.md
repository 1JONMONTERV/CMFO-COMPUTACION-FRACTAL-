# @cmfo/core

[![npm version](https://badge.fury.io/js/%40cmfo%2Fcore.svg)](https://badge.fury.io/js/%40cmfo%2Fcore)
[![Node.js 14+](https://img.shields.io/badge/node-14%2B-brightgreen.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Node.js bindings for **CMFO** (Fractal Universal Computation Engine) - a deterministic geometric computing platform operating on a 7-dimensional φ-manifold.

## Installation

```bash
npm install @cmfo/core
```

## Prerequisites

The native CMFO library must be built before using these bindings. See the [main repository](https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-) for build instructions.

## Quick Start

```javascript
const cmfo = require('@cmfo/core');

// Display CMFO information
cmfo.info();

// Get the phi constant (golden ratio)
const phi = cmfo.phi();
console.log(`Phi: ${phi}`); // ≈ 1.618033988749895

// Compute T7 tensor product
const a = [1, 1, 1, 1, 1, 1, 1];
const b = [2, 2, 2, 2, 2, 2, 2];
const result = cmfo.tensor7(a, b);
console.log(`Result: [${result}]`);

// Use gamma step function
const gamma = cmfo.gammaStep(0.5);
console.log(`Gamma: ${gamma}`);

// Object-oriented interface
const t1 = new cmfo.T7Tensor([1, 2, 3, 4, 5, 6, 7]);
const t2 = new cmfo.T7Tensor([1, 1, 1, 1, 1, 1, 1]);
const t3 = t1.multiply(t2);
console.log(t3.toString());
```

## TypeScript Support

This package includes TypeScript definitions:

```typescript
import { phi, tensor7, gammaStep, T7Tensor, info } from '@cmfo/core';

const phiValue: number = phi();
const result: number[] = tensor7([1,1,1,1,1,1,1], [2,2,2,2,2,2,2]);
const tensor: T7Tensor = new T7Tensor([1,2,3,4,5,6,7]);
```

## API Reference

### Functions

#### `phi(): number`
Returns the phi constant (golden ratio, ≈ 1.618033988749895).

#### `tensor7(a: number[], b: number[]): number[]`
Computes the T7 tensor product of two 7-element vectors.

**Parameters:**
- `a` - First 7-element array
- `b` - Second 7-element array

**Returns:** 7-element array result

#### `gammaStep(x: number): number`
Computes the gamma step function for fractal iterations.

**Parameters:**
- `x` - Input value

**Returns:** Gamma step result

#### `info(): void`
Displays CMFO information to the console.

### Classes

#### `T7Tensor`

Object-oriented interface for 7-dimensional tensor operations.

**Constructor:**
```javascript
new T7Tensor(data: number[])
```

**Methods:**
- `multiply(other: T7Tensor | number[]): T7Tensor` - Multiply with another tensor
- `toArray(): number[]` - Get tensor as array
- `toString(): string` - String representation

**Example:**
```javascript
const t1 = new T7Tensor([1, 2, 3, 4, 5, 6, 7]);
const t2 = new T7Tensor([1, 1, 1, 1, 1, 1, 1]);
const result = t1.multiply(t2);
console.log(result.toArray());
```

## Architecture

CMFO Node.js bindings use FFI (Foreign Function Interface) to call native C/C++ libraries:

- **FFI Layer**: `ffi-napi`, `ref-napi`, `ref-array-napi`
- **Native Core**: C/C++/CUDA implementation
- **Platform Support**: Windows, macOS, Linux

## Building Native Library

Before using these bindings, you need to build the native CMFO library:

```bash
# Clone the repository
git clone https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-.git
cd CMFO-COMPUTACION-FRACTAL-

# Build native library (see main README for detailed instructions)
# The library should be built to: core/native/build/Release/
```

## Testing

Run the test suite:

```bash
npm test
```

## Platform Support

- **Windows**: `cmfo_core.dll`
- **macOS**: `libcmfo_core.dylib`
- **Linux**: `libcmfo_core.so`

## Documentation

- [Full Documentation](https://1jonmonterv.github.io/CMFO-COMPUTACION-FRACTAL-/)
- [GitHub Repository](https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-)
- [Python Package](https://pypi.org/project/cmfo/)

## Features

- ✅ T7 Tensor operations on φ-manifold
- ✅ Gamma-phi fractal functions
- ✅ TypeScript definitions included
- ✅ Cross-platform support (Windows, macOS, Linux)
- ✅ Object-oriented and functional APIs
- ✅ Native C/C++ performance via FFI

## License

MIT License with commercial restrictions for enterprise modules.

## Author

**Jonnathan Montero Viques**  
Email: jmvlavacar@hotmail.com  
Location: San José, Costa Rica

## Contributing

See [CONTRIBUTING.md](https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-/blob/main/CONTRIBUTING.md) in the main repository.

## Support

For issues, questions, or commercial licensing:
- GitHub Issues: https://github.com/1JONMONTERV/CMFO-COMPUTACION-FRACTAL-/issues
- Email: jmvlavacar@hotmail.com

## Related Packages

- **Python**: `pip install cmfo`
- **C/C++**: See main repository for native library usage
