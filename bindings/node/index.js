/**
 * CMFO Node.js Bindings
 * FFI bindings to CMFO native C/C++ library
 */

const ffi = require('ffi-napi');
const ref = require('ref-napi');
const path = require('path');
const ArrayType = require('ref-array-napi');
const os = require('os');

// Type definitions
const double = ref.types.double;
const Vector7 = ArrayType(double, 7);
const Matrix7 = ArrayType(ArrayType(double, 7), 7);

/**
 * Locate the native CMFO library
 * @returns {string} Path to the native library
 */
function findNativeLibrary() {
    const platform = os.platform();
    let libName;

    if (platform === 'win32') {
        libName = 'cmfo_core.dll';
    } else if (platform === 'darwin') {
        libName = 'libcmfo_core.dylib';
    } else {
        libName = 'libcmfo_core.so';
    }

    // Try multiple possible locations
    const possiblePaths = [
        path.resolve(__dirname, '..', '..', 'core', 'native', 'build', 'Release', libName),
        path.resolve(__dirname, '..', '..', 'build', 'Release', libName),
        path.resolve(__dirname, 'lib', libName),
    ];

    // For now, return the first path (user will need to build the library)
    return possiblePaths[0];
}

const libPath = findNativeLibrary();

// FFI bindings to native library
let cmfo;
try {
    cmfo = ffi.Library(libPath, {
        'cmfo_phi': [double, []],
        'cmfo_tensor7': ['void', [Vector7, Vector7, Vector7]],
        'cmfo_gamma_step': [double, [double]],
        // Matrix API
        'Matrix7x7_Create': ['pointer', []],
        'Matrix7x7_Destroy': ['void', ['pointer']],
        'Matrix7x7_SetIdentity': ['void', ['pointer']],
        'Matrix7x7_BatchEvolve': ['void', ['pointer', 'pointer', 'pointer', 'int', 'int']]
    });
} catch (error) {
    console.error('Failed to load CMFO native library.');
    console.error('Make sure the native library is built and located at:', libPath);
    console.error('Error:', error.message);
    throw error;
}

/**
 * Get the phi constant (golden ratio)
 * @returns {number} The phi constant
 */
function phi() {
    return cmfo.cmfo_phi();
}

/**
 * Compute T7 tensor product
 * @param {number[]} a - First 7-element vector
 * @param {number[]} b - Second 7-element vector
 * @returns {number[]} Result 7-element vector
 */
function tensor7(a, b) {
    if (!Array.isArray(a) || a.length !== 7) {
        throw new Error('First argument must be an array of 7 numbers');
    }
    if (!Array.isArray(b) || b.length !== 7) {
        throw new Error('Second argument must be an array of 7 numbers');
    }

    const vecA = new Vector7(a);
    const vecB = new Vector7(b);
    const result = new Vector7(new Array(7).fill(0));

    cmfo.cmfo_tensor7(result, vecA, vecB);

    return Array.from(result);
}

/**
 * Compute gamma step function
 * @param {number} x - Input value
 * @returns {number} Gamma step result
 */
function gammaStep(x) {
    if (typeof x !== 'number') {
        throw new Error('Argument must be a number');
    }
    return cmfo.cmfo_gamma_step(x);
}

/**
 * T7Tensor class for object-oriented interface
 */
class T7Tensor {
    /**
     * Create a T7 tensor
     * @param {number[]} data - 7-element array
     */
    constructor(data) {
        if (!Array.isArray(data) || data.length !== 7) {
            throw new Error('T7Tensor requires an array of 7 numbers');
        }
        this.data = [...data];
    }

    /**
     * Multiply with another T7 tensor
     * @param {T7Tensor|number[]} other - Another tensor or array
     * @returns {T7Tensor} Result tensor
     */
    multiply(other) {
        const otherData = other instanceof T7Tensor ? other.data : other;
        const result = tensor7(this.data, otherData);
        return new T7Tensor(result);
    }

    /**
     * Get tensor as array
     * @returns {number[]} The tensor data
     */
    toArray() {
        return [...this.data];
    }

    /**
     * String representation
     * @returns {string}
     */
    toString() {
        return `T7Tensor[${this.data.join(', ')}]`;
    }
}

/**
 * T7Matrix class for Batch Operations
 */
class T7Matrix {
    constructor() {
        this.ptr = cmfo.Matrix7x7_Create();
    }

    static identity() {
        const m = new T7Matrix();
        cmfo.Matrix7x7_SetIdentity(m.ptr);
        return m;
    }

    destroy() {
        if (this.ptr) {
            cmfo.Matrix7x7_Destroy(this.ptr);
            this.ptr = null;
        }
    }

    /**
     * Evolve a batch of states
     * @param {Array<Object>} batch - Array of {real: [], imag: []}
     * @param {number} steps
     * @returns {Array<Object>} Evolved batch
     */
    evolveBatch(batch, steps) {
        const N = batch.length;
        const realData = new Float64Array(N * 7);
        const imagData = new Float64Array(N * 7);

        // Flatten data
        for (let i = 0; i < N; i++) {
            for (let j = 0; j < 7; j++) {
                realData[i * 7 + j] = batch[i].real[j];
                imagData[i * 7 + j] = batch[i].imag[j];
            }
        }

        // Pointers
        const pReal = Buffer.from(realData.buffer);
        const pImag = Buffer.from(imagData.buffer);

        cmfo.Matrix7x7_BatchEvolve(this.ptr, pReal, pImag, N, steps);

        // Reconstruct
        const outReal = new Float64Array(pReal.buffer, pReal.byteOffset, N * 7);
        const outImag = new Float64Array(pImag.buffer, pImag.byteOffset, N * 7);

        const result = [];
        for (let i = 0; i < N; i++) {
            result.push({
                real: Array.from(outReal.slice(i * 7, (i + 1) * 7)),
                imag: Array.from(outImag.slice(i * 7, (i + 1) * 7))
            });
        }
        return result;
    }
}

/**
 * Display CMFO information
 */
function info() {
    console.log('=== CMFO Node.js Bindings ===');
    console.log(`Phi Constant: ${phi()}`);
    console.log('Status: LOADED');
    console.log('Core: T7 Phi-Manifold');
    console.log('Extensions: Matrix/Batch API Enabled');
    console.log('Author: Jonnathan Montero Viques');
    console.log('=============================');
}

// Exports
module.exports = {
    phi,
    tensor7,
    gammaStep,
    T7Tensor,
    T7Matrix,
    info
};
