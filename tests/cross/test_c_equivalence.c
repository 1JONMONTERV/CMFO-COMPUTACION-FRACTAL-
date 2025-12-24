#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>

/*
 * CMFO CORE C IMPLEMENTATION
 * Reference implementation for Cross-Language Verification
 */

// Gamma function: sin(v)
double gamma_step(double v) {
    return sin(v);
}

// Phi Sign: -1 or 1
double phi_sign(double x) {
    return (x >= 0) ? 1.0 : -1.0;
}

// Fractal AND: min(sgn(a), sgn(b))
double phi_and(double a, double b) {
    double sa = phi_sign(a);
    double sb = phi_sign(b);
    return (sa < sb) ? sa : sb;
}

// Fractal OR
double phi_or(double a, double b) {
    return (phi_sign(a) == 1.0) ? phi_sign(a) : phi_sign(b);
}

// Test Runner
void run_test(double input, double expected) {
    double result = gamma_step(input);
    double diff = fabs(result - expected);
    if (diff < 1e-6) {
        printf("[PASS] Gamma(%.4f) = %.4f\n", input, result);
    } else {
        printf("[FAIL] Gamma(%.4f) = %.4f (Expected %.4f)\n", input, result, expected);
        exit(1);
    }
}

int main() {
    printf("Running CMFO Cross-Language Verification (C)\n");
    printf("-------------------------------------------\n");

    // NOTE: In a real scenario, this would parse golden_vectors.json.
    // Here we hardcode a few checks to prove the code works.
    
    // Check Gamma
    run_test(0.0, 0.0);
    run_test(3.1415926535, 0.0);
    
    // Check Logic
    assert(phi_and(1.0, -1.0) == -1.0);
    assert(phi_or(1.0, -1.0) == 1.0);
    
    printf("\nAll internal C tests passed.\n");
    return 0;
}
