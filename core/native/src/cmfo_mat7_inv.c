/*
 * CMFO-UNIVERSE Core
 * Inversa de matriz 7×7 (Gauss–Jordan con pivoteo parcial)
 * Copyright (c) 2025
 * Autor:   Jonnathan Montero Víquez
 * Ciudad:  San José, Costa Rica
 * Email:   jmvlavacar@hotmail.com
 *
 * Licencia:
 *   CMFO-UNIVERSE LICENSE v1.1 — Commercial Restricted.
 */

#include "cmfo_core.h"
#include <math.h>
#include <stdio.h>

int cmfo_mat7_inv(double out[7][7], const double M[7][7]) {

    double a[7][14];

    // Construir matriz aumentada [M | I]
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++)
            a[i][j] = M[i][j];

        for (int j = 0; j < 7; j++)
            a[i][j + 7] = (i == j ? 1.0 : 0.0);
    }

    // Gauss–Jordan
    for (int i = 0; i < 7; i++) {

        // 1. Seleccionar pivote máximo por estabilidad
        double maxA = fabs(a[i][i]);
        int pivotRow = i;

        for (int r = i + 1; r < 7; r++) {
            if (fabs(a[r][i]) > maxA) {
                maxA = fabs(a[r][i]);
                pivotRow = r;
            }
        }

        // Si el pivote es cero -> singular
        if (maxA < 1e-15)
            return 0;

        // 2. Intercambiar filas si es necesario
        if (pivotRow != i) {
            for (int c = 0; c < 14; c++) {
                double tmp = a[i][c];
                a[i][c] = a[pivotRow][c];
                a[pivotRow][c] = tmp;
            }
        }

        // 3. Normalizar fila del pivote
        double piv = a[i][i];
        for (int c = 0; c < 14; c++)
            a[i][c] /= piv;

        // 4. Eliminar todas las otras filas
        for (int r = 0; r < 7; r++) {
            if (r == i) continue;
            double factor = a[r][i];
            for (int c = 0; c < 14; c++)
                a[r][c] -= factor * a[i][c];
        }
    }

    // Extraer la inversa (parte derecha)
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++)
            out[i][j] = a[i][j + 7];

    return 1;
}


