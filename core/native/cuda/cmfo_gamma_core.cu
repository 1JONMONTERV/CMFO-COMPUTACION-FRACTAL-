#include <cuda.h>
#include <cuda_runtime.h>
#include "cmfo_gamma_symbols.h"

__constant__ double PHI_INV[7];
__constant__ double PHI_POW[7];
__constant__ double W0[7];
__constant__ double W1[7];
__constant__ double W2[7];

extern "C" __global__
void cmfo_gamma_step(const double* __restrict__ X_in,
                     double* __restrict__ X_out,
                     int N, int steps)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;

    double x[7];

    #pragma unroll
    for(int i = 0; i < 7; i++){
        x[i] = X_in[idx*7 + i];
    }

    for(int s = 0; s < steps; s++){
        double y[7];

        #pragma unroll
        for(int i = 0; i < 7; i++){
            int ip1 = (i + 1) % 7;
            int im1 = (i + 6) % 7;
            int ip2 = (i + 2) % 7;
            int im2 = (i + 5) % 7;

            double acc = W0[i]*x[i];
            acc += W1[i]*(x[ip1] + x[im1]);
            acc += W2[i]*(x[ip2] + x[im2]);

            y[i] = acc;
        }

        double norm = 0.0;

        #pragma unroll
        for(int k = 0; k < 7; k++){
            norm += y[k] * PHI_POW[k];
        }

        double inv_norm = 1.0 / norm;

        #pragma unroll
        for(int k = 0; k < 7; k++){
            x[k] = y[k] * inv_norm;
        }
    }

    #pragma unroll
    for(int i = 0; i < 7; i++){
        X_out[idx*7 + i] = x[i];
    }
}
