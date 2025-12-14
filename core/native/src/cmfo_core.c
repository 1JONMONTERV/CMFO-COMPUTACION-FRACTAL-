/*
 * CMFO-UNIVERSE Core Implementación
 */

#include "cmfo_core.h"
#include <math.h>

// =======================
// Constantes fractales
// =======================
const double CMFO_PHI = 1.6180339887498948482;
const double CMFO_TAU = 0.6180339887498948482;

double cmfo_phi(void) { return CMFO_PHI; }
double cmfo_tau(void) { return CMFO_TAU; }

// =======================
// Funciones vectoriales T7
// =======================
double cmfo_norm7(const double a[7]) {
    double s = 0.0;
    for (int i = 0; i < 7; i++) s += a[i] * a[i];
    return sqrt(s);
}

double cmfo_dot7(const double a[7], const double b[7]) {
    double s = 0.0;
    for (int i = 0; i < 7; i++) s += a[i] * b[i];
    return s;
}

void cmfo_add7(double out[7], const double a[7], const double b[7]) {
    for (int i = 0; i < 7; i++) out[i] = a[i] + b[i];
}

void cmfo_scale7(double out[7], const double a[7], double k) {
    for (int i = 0; i < 7; i++) out[i] = a[i] * k;
}


