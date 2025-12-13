#include <cuda_runtime.h>
#include <stdio.h>

// Definición de constantes si no están definidas
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Kernel: cmfo_dynamics_gpu
// Objetivo: Simular N=7 modos simultáneamente.
// Mapping: 1 Bloque, 7 Hilos.
// Input: h_theta[7] (Host -> Device)

__global__ void cmfo_dynamics_gpu(double* d_theta_out, const double* d_theta_in, int steps,
                                  double dt) {
    // Identificador de hilo: mapea a uno de los 7 modos (0 a 6)
    int tid = threadIdx.x;

    if (tid < 7) {
        // Estado local del modo
        double theta = d_theta_in[tid];

        // Simulación simple de dinámica (ejemplo conceptual, reemplazar con lógica fractal real)
        // Por ejemplo: d(theta)/dt = omega * cos(theta) (Dinámica no lineal simple)
        // Aquí usamos una evolución dummy para probar la concurrencia

        for (int s = 0; s < steps; ++s) {
            // Evolución placeholder basada en el índice del modo (fractalidad simulada)
            double omega = 1.61803398875 * (tid + 1);  // phi factor
            theta += omega * dt;

            // Normalización periódica (Toro T^7 implica periodicidad)
            if (theta > 2 * M_PI) theta -= 2 * M_PI;
        }

        // Escribir resultado
        d_theta_out[tid] = theta;
    }
}

// Wrapper para lanzar el kernel fácilmente desde C++
extern "C" void launch_cmfo_dynamics(double* h_out, const double* h_in, int steps, double dt) {
    double *d_in, *d_out;
    size_t size = 7 * sizeof(double);

    // Allocar memoria en GPU
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);

    // Copiar entrada
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Lanzar Kernel: 1 bloque, 7 hilos
    cmfo_dynamics_gpu<<<1, 7>>>(d_out, d_in, steps, dt);

    // Sincronizar y copiar salida
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Liberar memoria
    cudaFree(d_in);
    cudaFree(d_out);
}

// --- PHASE 2: COUPLED DYNAMICS ---
// Implementacion "Real" con acople N-Cuerpos (Kuramoto-like)
// d(theta_i)/dt = omega_i + K * sum_j(sin(theta_j - theta_i))

__global__ void cmfo_dynamics_gpu_v2(double* d_theta_out, const double* d_theta_in, int steps,
                                     double dt) {
    int tid = threadIdx.x;

    // Shared Memory para interaccion N-cuerpos eficiente
    __shared__ double s_theta[7];

    if (tid < 7) {
        double theta = d_theta_in[tid];
        double omega = 1.61803398875 * (tid + 1);
        double K = 0.1;  // Constante de acople

        for (int s = 0; s < steps; ++s) {
            // Cargar estado actual a memoria compartida
            s_theta[tid] = theta;
            __syncthreads();  // Esperar a que todos carguen

            // Calcular interaccion con todos los otros modos
            double interaction = 0.0;
            for (int j = 0; j < 7; j++) {
                interaction += sin(s_theta[j] - theta);
            }

            // Evolucion
            theta += (omega + K * interaction) * dt;

            // Periodicidad
            // if (theta > 2 * M_PI) theta -= 2 * M_PI; // Opcional, mantener unbound para ver
            // deriva
            __syncthreads();  // Sincronizar antes del siguiente paso
        }
        d_theta_out[tid] = theta;
    }
}

extern "C" void launch_cmfo_dynamics_v2(double* h_out, const double* h_in, int steps, double dt) {
    double *d_in, *d_out;
    size_t size = 7 * sizeof(double);
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    cmfo_dynamics_gpu_v2<<<1, 7>>>(d_out, d_in, steps, dt);

    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
}
