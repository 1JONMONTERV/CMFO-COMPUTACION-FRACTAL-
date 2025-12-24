
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PHI 1.618033988749895f

// ==========================================
// 7D FRACTAL KERNEL (AGRESIVO)
// ==========================================
__global__ void cmfo_7d_kernel_agresivo(float* states, int N, int steps, float time_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load 7D state locally (Registers)
    // Structure: Struct of Arrays (SoA) or Array of Structs (AoS)?
    // For simplicity without complex indexing, we treat 'states' as N * 7 array.
    // idx*7 + 0..6
    
    float v[7];
    #pragma unroll
    for (int k = 0; k < 7; k++) {
        v[k] = states[idx * 7 + k];
    }

    // Heavy Math Loop (Fractal Evolution)
    for (int s = 0; s < steps; s++) {
        float phase = time_offset + (float)s * 0.1f;
        
        // 1. Nonlinear Phi Rotation
        float t0 = v[0];
        v[0] = (v[0] * cosf(phase) - v[1] * sinf(phase)) * PHI;
        v[1] = (t0 * sinf(phase) + v[1] * cosf(phase)) / PHI;

        // 2. Torsional Coupling (Fractal Charge)
        float coupling = sinf(v[0] * v[1]);
        v[2] += coupling;
        v[3] += cosf(v[2] * phase);

        // 3. High-Order Resonance
        v[4] = fmodf(v[4] + v[2]*v[3], 7.0f);
        v[5] = sqrtf(fabsf(v[0] * v[5] + 1.0f));
        
        // 4. Decay / bounding
        v[6] = expf(-0.01f * fabsf(v[6])) * v[0];
    }

    // Store back
    #pragma unroll
    for (int k = 0; k < 7; k++) {
        states[idx * 7 + k] = v[k];
    }
}

// ==========================================
// C-COMPATIBLE EXPORTS (DRIVER)
// ==========================================
extern "C" {

    // Simple error checking wrapper
    #define CUDA_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                printf("CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
                return -1.0; \
            } \
        } while(0)

    // Runs the full benchmark loop on GPU and returns execution time in seconds.
    // Returns negative value on error.
    __declspec(dllexport) double run_cmfo_benchmark_internal(int N, int internal_steps, int iterations) {
        
        size_t size = N * 7 * sizeof(float);
        
        // Host Alloc
        float* h_states = (float*)malloc(size);
        if (!h_states) return -2.0;

        // Initialize (Deterministic random-ish)
        for (int i = 0; i < N * 7; i++) {
            h_states[i] = (float)((i % 100) / 100.0);
        }

        // Device Alloc
        float* d_states;
        if (cudaMalloc(&d_states, size) != cudaSuccess) {
            free(h_states);
            return -3.0; // VRAM alloc fail
        }

        // Host -> Device
        if (cudaMemcpy(d_states, h_states, size, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_states);
            free(h_states);
            return -4.0;
        }

        // Setup Execution
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        printf("[CORE] Kernel Launch Config: Grid=%d, Block=%d. Total Threads=%d\n", gridSize, blockSize, N);
        printf("[CORE] Running %d iterations with %d internal fracture steps...\n", iterations, internal_steps);

        // Warmup
        cmfo_7d_kernel_agresivo<<<gridSize, blockSize>>>(d_states, N, 10, 0.0f);
        cudaDeviceSynchronize();

        // BENCHMARK RECORD
        cudaEventRecord(start);

        for (int i = 0; i < iterations; i++) {
            cmfo_7d_kernel_agresivo<<<gridSize, blockSize>>>(d_states, N, internal_steps, (float)i);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Device -> Host (Verify success)
        cudaMemcpy(h_states, d_states, size, cudaMemcpyDeviceToHost);

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_states);
        free(h_states);

        return (double)(milliseconds / 1000.0f);
    }
}
