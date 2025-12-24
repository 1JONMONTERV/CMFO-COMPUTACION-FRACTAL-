#include <cuda_runtime.h>
#include <math.h>

#define PHI 1.61803398875f
#define PHI_INV 0.61803398875f

extern "C" {

// Kernel CMFO 7D Real
// Cada thread evoluciona 1 estado independiente
__global__ void cmfo_kernel(float *states, float *out_energy, int n_states,
                            int steps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_states)
    return;

  // Cargar estado 7D a registros (rapido)
  float v[7];
  for (int k = 0; k < 7; k++) {
    v[k] = states[idx * 7 + k];
  }

  float energy_acc = 0.0f;

  // Evolucion Temporal Fractal (Operador T7 simplificado para benchmark)
  for (int t = 0; t < steps; t++) {
    // Rotacion no-lineal phi-modulada
    float next[7];

    // v_i+1 = R_phi(v_i + phi * v_j)
    // Usamos sin/cos como proxy de carga ALU pesada no-lineal
    next[0] = sinf(v[0] + PHI * v[1]);
    next[1] = cosf(v[1] + PHI * v[2]);
    next[2] = sinf(v[2] + PHI * v[3]);
    next[3] = cosf(v[3] + PHI * v[4]);
    next[4] = sinf(v[4] + PHI * v[5]);
    next[5] = cosf(v[5] + PHI * v[6]);
    next[6] = sinf(v[6] + PHI * v[0]);

    // Acumular energia (metrica)
    float step_e = 0.0f;
    for (int k = 0; k < 7; k++) {
      v[k] = next[k];
      step_e += v[k] * v[k];
    }
    energy_acc += sqrtf(step_e);
  }

  // Guardar resultado
  out_energy[idx] = energy_acc;

  // Guardar estado final (opcional, para verificar memoria)
  for (int k = 0; k < 7; k++) {
    states[idx * 7 + k] = v[k];
  }
}

// Wrapper para llamar desde Host
__declspec(dllexport) void launch_cmfo_kernel(float *h_states, float *h_energy,
                                              int n, int steps) {
  float *d_states, *d_energy;

  // 1. Allocate GPU Memory
  cudaMalloc(&d_states, n * 7 * sizeof(float));
  cudaMalloc(&d_energy, n * sizeof(float));

  // 2. Copy Host -> Device
  cudaMemcpy(d_states, h_states, n * 7 * sizeof(float), cudaMemcpyHostToDevice);

  // 3. Launch
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  cmfo_kernel<<<blocks, threads>>>(d_states, d_energy, n, steps);

  // Sincronizar para medir tiempo real de kernel
  cudaDeviceSynchronize();

  // 4. Copy Device -> Host
  cudaMemcpy(h_states, d_states, n * 7 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_energy, d_energy, n * sizeof(float), cudaMemcpyDeviceToHost);

  // 5. Free
  cudaFree(d_states);
  cudaFree(d_energy);
}
}
