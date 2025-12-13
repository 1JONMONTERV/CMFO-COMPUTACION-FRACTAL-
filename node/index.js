
const ffi = require('ffi-napi');
const ref = require('ref-napi');
const path = require('path');
const ArrayType = require('ref-array-napi');

// Tipos C
const double = ref.types.double;
const Vector7 = ArrayType(double, 7);
const Matrix7 = ArrayType(ArrayType(double, 7), 7);

// Buscar librería dinámica
const libPath = path.resolve(__dirname, '../build/Release/cmfo_core');
// Nota: ffi-napi añade .dll/.so automáticamente

// Bindings
const cmfo = ffi.Library(libPath, {
    'cmfo_phi': [double, []],
    'cmfo_tensor7': ['void', [Vector7, Vector7, Vector7]]
});

console.log("=== CMFO Node.js Interop Demo ===");
console.log(`Phi Constant (from C): ${cmfo.cmfo_phi()}`);

// Tensor Test
const a = new Vector7([1, 1, 1, 1, 1, 1, 1]);
const b = new Vector7([2, 2, 2, 2, 2, 2, 2]);
const out = new Vector7(new Array(7).fill(0));

cmfo.cmfo_tensor7(out, a, b);

console.log("Tensor Product [1..] x [2..]:");
for (let i = 0; i < 7; i++) {
    process.stdout.write(out[i] + " ");
}
console.log("\n[SUCCESS] Node.js called C Core via FFI.");
