/*
 * CMFO-UNIVERSE Core
 * Copyright (c) 2025
 * Autor:   Jonnathan Montero Víquez
 * Ciudad:  San José, Costa Rica
 * Email:   jmvlavacar@hotmail.com
 *
 * Licencia:
 *   Este archivo forma parte de CMFO-UNIVERSE y se distribuye bajo la
 *   "CMFO-UNIVERSE LICENSE v1.1 — Commercial Restricted".
 *
 *   Uso personal, académico y de investigación no comercial: PERMITIDO.
 *   Uso comercial: REQUIERE licencia comercial explícita del autor.
 *
 *   Ver archivo LICENSE en la raíz del repositorio para los detalles
 *   completos de permisos, restricciones y condiciones.
 */

#include "cmfo_core.h"

// Producto tensorial reducido (CMFO): 
void cmfo_tensor7(double out[7], const double a[7], const double b[7]) {
    for (int i = 0; i < 7; i++)
        out[i] = a[i] * b[i];
}

void cmfo_tensor7_sym(double out[7], const double a[7], const double b[7]) {
    for (int i = 0; i < 7; i++)
        out[i] = 0.5 * (a[i] * b[i] + b[i] * a[i]);
}

void cmfo_tensor7_antisym(double out[7], const double a[7], const double b[7]) {
    for (int i = 0; i < 7; i++)
        out[i] = 0.5 * (a[i] * b[i] - b[i] * a[i]);
}

