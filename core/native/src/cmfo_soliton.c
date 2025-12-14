#include "cmfo_soliton.h"
#include <stdlib.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

SineGordonSystem* cmfo_create_system(int N, double x_min, double x_max, double dt) {
    SineGordonSystem* sys = (SineGordonSystem*)malloc(sizeof(SineGordonSystem));
    sys->N = N;
    sys->x_min = x_min;
    sys->x_max = x_max;
    sys->dx = (x_max - x_min) / (N - 1);
    sys->dt = dt;
    sys->phi = (double*)calloc(N, sizeof(double));
    sys->phi_t = (double*)calloc(N, sizeof(double));
    return sys;
}

void cmfo_free_system(SineGordonSystem* sys) {
    if (sys) {
        free(sys->phi);
        free(sys->phi_t);
        free(sys);
    }
}

// Inicializa o añade un Kink analítico a la grilla
void cmfo_init_kink(SineGordonSystem* sys, double v, double x0, int polarity) {
    double gamma = 1.0 / sqrt(1.0 - v*v);
    
    for (int i = 0; i < sys->N; ++i) {
        double x = sys->x_min + i * sys->dx;
        // Solución exacta de Kink móvil en t=0
        double arg = gamma * (x - x0);
        double val = 4.0 * atan(exp(arg));
        
        // Derivada temporal exacta en t=0: phi_t = -v * phi_x
        // phi_x = 2 * gamma / cosh(arg)
        double sech = 1.0 / cosh(arg);
        double phi_x = 2.0 * gamma * sech;
        double vel = -v * phi_x;

        if (polarity < 0) {
            // Anti-Kink: 4 * atan(exp(-arg)) -> equivalente a 2PI - val (aprox) o negativo
            // Formalmente: phi_antikink = -phi_kink
            val = -val; 
            vel = -vel;
        }

        // Superposición lineal simple (válida para solitones bien separados)
        sys->phi[i] += val;
        sys->phi_t[i] += vel;
    }
}

// Hamiltoniano Discreto (Energia Total)
// Referencia: Docs/THEORY_SOLITON.md Section 4.1
// H = Integral [ 0.5*phi_t^2 + 0.5*phi_x^2 + (1 - cos(phi)) ] dx
double cmfo_energy_sg(SineGordonSystem* sys) {
    double H = 0.0;
    for (int i = 0; i < sys->N - 1; ++i) {
        // [Termino Cinetico]: 0.5 * (dphi/dt)^2
        // Representa la inercia del campo.
        double kin = 0.5 * sys->phi_t[i] * sys->phi_t[i];
        
        // [Termino Elastico]: 0.5 * (dphi/dx)^2
        // Representa la tension elastica espacial.
        double dphi = (sys->phi[i+1] - sys->phi[i]) / sys->dx;
        double elas = 0.5 * dphi * dphi;
        
        // [Termino Potencial]: 1 - cos(phi)
        // Potencial periodico no lineal que genera la masa del soliton.
        double pot = 1.0 - cos(sys->phi[i]);

        H += (kin + elas + pot) * sys->dx;
    }
    return H;
}

// Carga Topologica (Indice de Winding)
// Referencia: Docs/THEORY_SOLITON.md Section 4.2
// Q = (1/2pi) * [phi(+inf) - phi(-inf)]
// Es un invariante topologico robusto contra perturbaciones continuas.
double cmfo_topo_charge_sg(SineGordonSystem* sys) {
    // Aproximacion numerica: phi(+L) - phi(-L)
    double delta = sys->phi[sys->N - 1] - sys->phi[0];
    return delta / (2.0 * M_PI);
}


// Evolución Temporal: Diferencias Finitas (Ec. de Onda con fuente sin(phi))
// phi_tt - phi_xx + sin(phi) = 0
// Discretización explícita simple (inestable para dt grande, usar Courant < 1)
void cmfo_step_sg(SineGordonSystem* sys) {
    // Necesitamos phi_new, phi_current, phi_old. 
    // Para simplificar esta demo sin un esquema Leapfrog de 3 niveles,
    // usaremos un esquema Euler Simpléctico o Velocity Verlet que es mejorconservativo.
    
    // 1. Actualizar phi a medio paso usando phi_t
    // (Omitido por simplicidad de implementación directa de "test de conservación")
    
    // Esquema FDTD clásico:
    // phi(t+dt) = 2phi(t) - phi(t-dt) + dt^2 * (phi_xx - sin(phi))
    
    // Implementación Reversibilidad: Euler Simpléctico
    // phi_t(t+dt) = phi_t(t) + dt * a(t)
    // phi(t+dt)   = phi(t)   + dt * phi_t(t+dt)
    
    double* acc = (double*)malloc(sys->N * sizeof(double));
    
    // Calcular Aceleración: phi_xx - sin(phi)
    for (int i = 1; i < sys->N - 1; ++i) {
        double d2phi = (sys->phi[i+1] - 2.0*sys->phi[i] + sys->phi[i-1]) / (sys->dx * sys->dx);
        acc[i] = d2phi - sin(sys->phi[i]);
    }
    // Condiciones de frontera libre (neumann aprox o fixed) -> Fixed at boundary
    acc[0] = 0; 
    acc[sys->N-1] = 0;

    // Integración
    for (int i = 0; i < sys->N; ++i) {
        sys->phi_t[i] += acc[i] * sys->dt;
        sys->phi[i]   += sys->phi_t[i] * sys->dt;
    }
    
    free(acc);
}
