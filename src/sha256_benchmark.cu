
#include <cuda_runtime.h>
#include <stdio.h>

__constant__ unsigned int K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

__device__ __forceinline__ unsigned int rotr(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ unsigned int choice(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (~x & z);
}

__device__ __forceinline__ unsigned int majority(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ unsigned int sigma0(unsigned int x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ unsigned int sigma1(unsigned int x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

__device__ __forceinline__ unsigned int Sigma0(unsigned int x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ unsigned int Sigma1(unsigned int x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__global__ void benchmark_sha256_kernel(unsigned int* out_found, unsigned int start_nonce,
                                        int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                             0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    unsigned int block[16];
#pragma unroll
    for (int i = 0; i < 15; i++) block[i] = i * 0x01010101;

    unsigned int nonce = start_nonce + idx;

    for (int k = 0; k < iterations; k++) {
        block[15] = nonce + k;

        unsigned int w[64];
        unsigned int a, b, c, d, e, f, g, h;
        unsigned int t1, t2;

#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = block[i];

#pragma unroll
        for (int i = 16; i < 64; i++) {
            w[i] = sigma1(w[i - 2]) + w[i - 7] + sigma0(w[i - 15]) + w[i - 16];
        }

        a = state[0];
        b = state[1];
        c = state[2];
        d = state[3];
        e = state[4];
        f = state[5];
        g = state[6];
        h = state[7];

#pragma unroll
        for (int i = 0; i < 64; i++) {
            t1 = h + Sigma1(e) + choice(e, f, g) + K[i] + w[i];
            t2 = Sigma0(a) + majority(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        if (a == 0xFFFFFFFF) {
            *out_found = a;
        }
    }
}

extern "C" {
__declspec(dllexport) void launch_benchmark(unsigned int* h_out, int blocks, int threads,
                                            int iterations) {
    unsigned int* d_out;
    if (cudaMalloc(&d_out, sizeof(unsigned int)) != cudaSuccess) return;
    cudaMemset(d_out, 0, sizeof(unsigned int));

    benchmark_sha256_kernel<<<blocks, threads>>>(d_out, 0, iterations);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
}
}
