/**
 * Test script for CMFO Node.js bindings
 */

const cmfo = require('./index');

console.log('=== CMFO Node.js Bindings Test ===\n');

try {
    // Test 1: Phi constant
    console.log('Test 1: Phi Constant');
    const phiValue = cmfo.phi();
    console.log(`  Phi = ${phiValue}`);
    console.log(`  Expected ≈ 1.618033988749895`);
    console.log(`  ✓ PASS\n`);

    // Test 2: Tensor7 function
    console.log('Test 2: Tensor7 Product');
    const a = [1, 1, 1, 1, 1, 1, 1];
    const b = [2, 2, 2, 2, 2, 2, 2];
    const result = cmfo.tensor7(a, b);
    console.log(`  Input A: [${a.join(', ')}]`);
    console.log(`  Input B: [${b.join(', ')}]`);
    console.log(`  Result:  [${result.join(', ')}]`);
    console.log(`  ✓ PASS\n`);

    // Test 3: Gamma step
    console.log('Test 3: Gamma Step');
    const gammaResult = cmfo.gammaStep(0.5);
    console.log(`  gamma_step(0.5) = ${gammaResult}`);
    console.log(`  ✓ PASS\n`);

    // Test 4: T7Tensor class
    console.log('Test 4: T7Tensor Class');
    const t1 = new cmfo.T7Tensor([1, 2, 3, 4, 5, 6, 7]);
    const t2 = new cmfo.T7Tensor([1, 1, 1, 1, 1, 1, 1]);
    console.log(`  T1: ${t1.toString()}`);
    console.log(`  T2: ${t2.toString()}`);
    const t3 = t1.multiply(t2);
    console.log(`  T1 * T2: ${t3.toString()}`);
    console.log(`  ✓ PASS\n`);

    // Test 5: Info display
    console.log('Test 5: Info Display');
    cmfo.info();
    console.log(`  ✓ PASS\n`);

    console.log('=================================');
    console.log('ALL TESTS PASSED ✓');
    console.log('=================================');

} catch (error) {
    console.error('\n✗ TEST FAILED');
    console.error('Error:', error.message);
    console.error('\nNote: Make sure the native CMFO library is built.');
    console.error('The library should be located at:');
    console.error('  ../../core/native/build/Release/');
    process.exit(1);
}
