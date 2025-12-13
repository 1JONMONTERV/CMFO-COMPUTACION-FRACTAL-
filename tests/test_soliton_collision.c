#include <math.h>
#include <stdio.h>

#include "cmfo_soliton.h"


#define ENERGY_TOL 0.05  // FDTD es aproximado, tolerancia moderada
#define CHARGE_TOL 1e-4

void save_to_csv(SineGordonSystem* sys, const char* filename);

int main() {
    printf("=== CMFO Hito 4: Puertas Logicas Solitonicas ===\n");
    printf("Objetivo: Verificar Reversibilidad (Conservacion de E y Q)\n");

    // 1. Configuración del Universo
    int N = 500;
    double L = 40.0;
    double dt = 0.05;  // Paso temporal pequeño para estabilidad
    SineGordonSystem* sys = cmfo_create_system(N, -L / 2, L / 2, dt);

    // 2. Inicialización: Colisión Elástica
    // Kink en x=-10, v=0.5
    // Anti-Kink en x=+10, v=-0.5
    double v = 0.5;
    cmfo_init_kink(sys, v, -10.0, 1);  // Kink (+1)
    cmfo_init_kink(sys, v, 10.0,
                   -1);  // Anti-Kink (-1) moving left (v is speed, direction handled by init?)
    // Correction: my init_kink uses 'v' directly. So for left moving, use -v.
    // Wait, the previous logic was: arg = gamma * (x - vt).
    // If we want collision, one moves right (v>0), one left (v<0).
    // Let's re-initialize correctly to be sure.

    // Reset manual
    cmfo_free_system(sys);
    sys = cmfo_create_system(N, -L / 2, L / 2, dt);

    cmfo_init_kink(sys, 0.4, -8.0, 1);   // Kink derecho (+1) viajando a +0.4
    cmfo_init_kink(sys, -0.4, 8.0, -1);  // Anti-Kink izquierdo (-1) viajando a -0.4

    // 3. Medición Inicial
    double E_init = cmfo_energy_sg(sys);
    double Q_init = cmfo_topo_charge_sg(sys);

    printf("[T=0] Energia: %.5f | Carga Topologica: %.5f\n", E_init, Q_init);
    save_to_csv(sys, "soliton_init.csv");  // Save initial state

    // 4. Simulación (Colisión)
    // Tiempo suficiente para chocar y separarse
    // Distancia ~16, Vel Relativa ~0.8 -> Tiempo ~ 20. Simulemos 3000 pasos * 0.01 = 30.0
    int steps = 2000;
    int save_interval = 20;  // Save every 20 steps (100 frames total)
    char filename[64];

    // Create output directory if needed (system command usually, but here we just write to current)

    for (int t = 0; t < steps; ++t) {
        if (t % save_interval == 0) {
            sprintf(filename, "soliton_step_%04d.csv", t);
            save_to_csv(sys, filename);
        }
        cmfo_step_sg(sys);
    }
    // Save final frame
    sprintf(filename, "soliton_step_%04d.csv", steps);
    save_to_csv(sys, filename);

    // 5. Medición Final
    double E_final = cmfo_energy_sg(sys);
    double Q_final = cmfo_topo_charge_sg(sys);
    // save_to_csv(sys, "soliton_final.csv"); // Redundant now, but keep for compatibility if
    // needed. Actually, let's keep it for verify_soliton.py which expects it.
    save_to_csv(sys, "soliton_final.csv");

    printf("[T=%d] Energia: %.5f | Carga Topologica: %.5f\n", steps, E_final, Q_final);

    // 6. Verificación "Dura"
    double delta_E = fabs(E_final - E_init);
    double delta_Q = fabs(Q_final - Q_init);

    printf("Delta Energia: %.5e (Tol: %.1e)\n", delta_E, ENERGY_TOL);
    printf("Delta Carga:   %.5e (Tol: %.1e)\n", delta_Q, CHARGE_TOL);

    int passed = 1;
    if (delta_E > ENERGY_TOL) {
        printf("[FALLO] La energia no se conservo suficiente (Disipacion numerica alta).\n");
        passed = 0;
    }
    if (delta_Q > CHARGE_TOL) {
        printf("[FALLO] La carga topologica cambio (Rompimiento de simetria).\n");
        passed = 0;
    }

    if (passed) {
        printf("[EXITO] Reversibilidad verificada. Colision Elastica Confirmada.\n");
    }

    cmfo_free_system(sys);
    return passed ? 0 : 1;
}

void save_to_csv(SineGordonSystem* sys, const char* filename) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;
    fprintf(fp, "x,phi,phi_t\n");
    for (int i = 0; i < sys->N; ++i) {
        double x = sys->x_min + i * sys->dx;
        fprintf(fp, "%.5f,%.5f,%.5f\n", x, sys->phi[i], sys->phi_t[i]);
    }
    fclose(fp);
}
