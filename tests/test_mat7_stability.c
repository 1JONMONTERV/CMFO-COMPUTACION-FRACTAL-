#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cmfo_core.h"

// Configuración de prueba
#define NUM_TESTS 10000
#define TOLERANCE 1e-12

// Generador de números aleatorios [-1, 1]
void rand_mat(double M[7][7]) {
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++) M[i][j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

int main() {
    printf("=== CMFO T7 Matrix Stability Test ===\n");
    printf("Ejecutando %d inversiones aleatorias...\n", NUM_TESTS);

    // Semilla para reproducibilidad (o time(NULL) para aleatoriedad real)
    srand(42);

    double A[7][7], InvA[7][7], Prod[7][7];
    double max_error = 0.0;
    int singular_count = 0;

    for (int t = 0; t < NUM_TESTS; ++t) {
        rand_mat(A);

        // Intentar invertir
        if (!cmfo_mat7_inv(InvA, A)) {
            singular_count++;
            continue;
        }

        // Validar: A * InvA = I
        cmfo_mat7_mul(Prod, A, InvA);

        // Check Identity
        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                double expected = (i == j) ? 1.0 : 0.0;
                double diff = fabs(Prod[i][j] - expected);
                if (diff > max_error) max_error = diff;
            }
        }
    }

    printf("----------------------------------------\n");
    printf("\nResultados:\n");
    printf("Validaciones Exitosas: %d\n", NUM_TESTS - singular_count);
    printf("Matrices Singulares:   %d\n", singular_count);
    printf("Error Maximo (Norma Inf): %.5e\n", max_error);

    // Criterio de Aceptación
    if (max_error < TOLERANCE) {
        printf("[PASS] Estabilidad Confirmada. El algebra T7 es solida.\n");
        return 0;
    } else {
        printf("[FAIL] Error excede la tolerancia de %.1e\n", TOLERANCE);
        return 1;
    }
}
