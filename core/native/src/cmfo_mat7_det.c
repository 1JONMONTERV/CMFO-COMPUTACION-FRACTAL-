/*
 * CMFO-UNIVERSE Core - Determinante 7x7 (expansión por Gauss)
 * Copyright (c) 2025
 * Autor:   Jonnathan Montero Víquez
 * Ciudad:  San José, Costa Rica
 * Email:   jmvlavacar@hotmail.com
 */

#include "cmfo_core.h"

double cmfo_mat7_det(const double M[7][7]) {
    double A[7][7];
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++)
            A[i][j] = M[i][j];

    double det = 1.0;

    for (int i = 0; i < 7; i++) {
        if (A[i][i] == 0.0)
            return 0.0;

        for (int j = i + 1; j < 7; j++) {
            double factor = A[j][i] / A[i][i];
            for (int k = i; k < 7; k++)
                A[j][k] -= factor * A[i][k];
        }
        det *= A[i][i];
    }

    return det;
}

