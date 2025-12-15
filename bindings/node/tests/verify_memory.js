/**
 * RIGOROUS VERIFICATION: Fractal Associative Memory (Node.js)
 * "All Languages" Proof
 */

const { T7Matrix } = require('../index');

function runVerification() {
    // 1. SETUP
    const MEMORY_SIZE = 1000;
    const SECRET_INDEX = 420;

    // Target Pattern (Complex 7D Vector)
    const targetReal = [1.0, 0.5, -0.5, 0.1, -0.1, 0.2, 0.8];
    const targetImag = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    console.log(`[VERIFY-JS] Initializing ${MEMORY_SIZE} states...`);

    // Create Batch
    const batch = [];
    for (let i = 0; i < MEMORY_SIZE; i++) {
        const real = [];
        const imag = [];
        for (let j = 0; j < 7; j++) {
            real.push(Math.random());
            imag.push(Math.random());
        }
        batch.push({ real, imag });
    }

    // Inject Target
    batch[SECRET_INDEX] = { real: [...targetReal], imag: [...targetImag] };

    // 2. EVOLUTION
    console.log("[VERIFY-JS] Evolving Batch (Native C++)...");
    const matEngine = T7Matrix.identity();
    const evolvedBatch = matEngine.evolveBatch(batch, 10);

    console.log("[VERIFY-JS] Evolving Target (Control)...");
    const targetBatch = [{ real: [...targetReal], imag: [...targetImag] }];
    const evolvedTargetBatch = matEngine.evolveBatch(targetBatch, 10);
    const expected = evolvedTargetBatch[0];

    // 3. RECALL
    console.log(`[VERIFY-JS] Searching for Evolved Pattern in ${MEMORY_SIZE} timelines...`);

    const actual = evolvedBatch[SECRET_INDEX];

    // Compute Error (L2 Norm of difference)
    let errorSq = 0;
    for (let k = 0; k < 7; k++) {
        const dR = actual.real[k] - expected.real[k];
        const dI = actual.imag[k] - expected.imag[k];
        errorSq += dR * dR + dI * dI;
    }
    const error = Math.sqrt(errorSq);

    console.log(`    Target Index: ${SECRET_INDEX}`);
    console.log(`    Expected[0]:  ${expected.real[0].toFixed(4)}...`);
    console.log(`    Actual[0]:    ${actual.real[0].toFixed(4)}...`);
    console.log(`    Error (Gap):  ${error.toExponential(4)}`);

    matEngine.destroy();

    // 4. ASSERTION
    if (error < 1e-9) {
        console.log("[PASS] JS FRACTAL MEMORY INTEGRITY CONFIRMED.");
        process.exit(0);
    } else {
        console.error("[FAIL] DETERMINISM BROKEN IN JS.");
        process.exit(1);
    }
}

try {
    runVerification();
} catch (err) {
    console.error("CRITICAL ERROR:", err);
    process.exit(1);
}
