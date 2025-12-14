#ifndef CMFO_SOLITON_H
#define CMFO_SOLITON_H

#include <math.h>
#include <stddef.h>

// Definiciones del modelo Sine-Gordon
// Ecuación: phi_xx - phi_tt = sin(phi)

// Estructura de Simulación 1D
typedef struct {
    double* phi;    // Campo
    double* phi_t;  // Derivada temporal (momento)
    double dx;      // Espaciado espacial
    double dt;      // Paso de tiempo
    int N;          // Puntos de grilla
    double x_min;
    double x_max;
} SineGordonSystem;

// Inicialización de Solitones (Kink / Anti-Kink)
// v: velocidad (-1 < v < 1)
// x0: posición inicial
// gamma = 1 / sqrt(1 - v^2)
// Kink (Q=+1): phi(x,t) = 4 * arctan(exp(gamma * (x - vt)))
// Anti-Kink (Q=-1): phi(x,t) = 4 * arctan(exp(-gamma * (x - vt)))
void cmfo_init_kink(SineGordonSystem* sys, double v, double x0, int polarity);

// Evolución Temporal (Método Leapfrog o RK4 simplificado para la Demo)
void cmfo_step_sg(SineGordonSystem* sys);

// Observables Conservados
// Energía Total (Hamiltoniano)
double cmfo_energy_sg(SineGordonSystem* sys);

// Carga Topológica: Q = (1/2pi) * [phi(+inf) - phi(-inf)]
// Debe ser entero (+1, -1, 0, etc.) conservado.
double cmfo_topo_charge_sg(SineGordonSystem* sys);

// Utilidades
SineGordonSystem* cmfo_create_system(int N, double x_min, double x_max, double dt);
void cmfo_free_system(SineGordonSystem* sys);

#endif // CMFO_SOLITON_H
