/*
 * CMFO-UNIVERSE Core
 * Copyright (c) 2025
 * Autor:   Jonnathan Montero Víquez
 * Ciudad:  San José, Costa Rica
 * Email:   jmvlavacar@hotmail.com
 *
 * Licencia:
 *   CMFO-UNIVERSE LICENSE v1.1 — Commercial Restricted.
 */

#include "cmfo_core.h"

// ===============================================================
// 1. Multiplicación fractal reducida CMFO (original del autómata)
// ===============================================================
void cmfo_mat7_mul(double out[7][7], const double A[7][7], const double B[7][7]) {
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            out[i][j] = A[i][j] * B[j][j];   // proyección fractal
        }
    }
}

// ===============================================================
// 2. Multiplicación REAL estándar 7×7 (para validación matemática)
// ===============================================================
void cmfo_mat7_mul_std(double out[7][7], const double A[7][7], const double B[7][7]) 
{
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            double sum = 0.0;
            for (int k = 0; k < 7; k++) {
                sum += A[i][k] * B[k][j];
            }
            out[i][j] = sum;
        }
    }
}
