/*
 * CMFO-UNIVERSE Core - Matrices 7x7
 * Copyright (c) 2025
 * Autor:   Jonnathan Montero Víquez
 * Ciudad:  San José, Costa Rica
 * Email:   jmvlavacar@hotmail.com
 *
 * Licencia: CMFO-UNIVERSE LICENSE v1.1 — Commercial Restricted
 */

#include "cmfo_core.h"

void cmfo_mat7_identity(double M[7][7]) {
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++)
            M[i][j] = (i == j ? 1.0 : 0.0);
}

void cmfo_mat7_zero(double M[7][7]) {
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++)
            M[i][j] = 0.0;
}

void cmfo_mat7_transpose(double out[7][7], const double M[7][7]) {
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++)
            out[i][j] = M[j][i];
}

void cmfo_mat7_vec(double out[7], const double M[7][7], const double v[7]) {
    for (int i = 0; i < 7; i++) {
        out[i] = 0.0;
        for (int j = 0; j < 7; j++)
            out[i] += M[i][j] * v[j];
    }
}

