/*
 * CMFO-UNIVERSE CUDA Kernels
 * Copyright (c) 2025
 * Autor:   Jonnathan Montero Víquez
 * Ciudad:  San José, Costa Rica
 * Email:   jmvlavacar@hotmail.com
 *
 * Licencia:
 *   Este archivo es parte de CMFO-UNIVERSE bajo licencia
 *   "CMFO-UNIVERSE LICENSE v1.1 — Commercial Restricted".
 */

extern "C" {

__global__ void tensor7_gpu(double *out, const double *a, const double *b) {
    int i = threadIdx.x;
    if (i < 7)
        out[i] = a[i] * b[i];
}

__global__ void mat7_mul_gpu(double *out, const double *A, const double *B) {
    int row = threadIdx.x;
    if (row >= 7) return;

    for (int col = 0; col < 7; col++) {
        double acc = 0.0;
        for (int k = 0; k < 7; k++)
            acc += A[row * 7 + k] * B[k * 7 + col];
        out[row * 7 + col] = acc;
    }
}

__global__ void dot7_gpu(double *out, const double *a, const double *b) {
    __shared__ double temp[7];
    int i = threadIdx.x;

    if (i < 7)
        temp[i] = a[i] * b[i];

    __syncthreads();

    // Warp reduction (7 values)
    if (i == 0) {
        double sum = 0.0;
        for (int j = 0; j < 7; j++) sum += temp[j];
        *out = sum;
    }
}

__global__ void norm7_gpu(double *out, const double *a) {
    __shared__ double temp[7];
    int i = threadIdx.x;

    if (i < 7)
        temp[i] = a[i] * a[i];

    __syncthreads();

    if (i == 0) {
        double sum = 0.0;
        for (int j = 0; j < 7; j++) sum += temp[j];
        *out = sqrt(sum);
    }
}

} // extern "C"
