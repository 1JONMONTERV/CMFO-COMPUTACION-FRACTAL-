#ifndef CMFO_CORE_H
#define CMFO_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

/**
 * CMFO Universal Core - Layer A (Reference)
 * Standard: C99
 * Precision: Float32
 */

// Constants
#define CMFO_PHI 1.618033988749895f
#define CMFO_DIM 7

// Error Codes
typedef enum {
    CMFO_OK = 0,
    CMFO_ERR_NULL_PTR = -1,
    CMFO_ERR_INVALID_DIM = -2
} cmfo_status_t;

/**
 * Reduces an N-dimensional vector to 7 dimensions.
 * @param input: Pointer to input array
 * @param dim: Dimension of input (must be >= 7)
 * @param output: Pointer to pre-allocated float[7]
 */
cmfo_status_t cmfo_tensor7_reduce(const float* input, size_t dim, float* output);

/**
 * Core T7 Operator: (a * b + PHI) / (1 + PHI)
 * @param a, b: Scalar inputs
 * @return: Scalar result
 */
float cmfo_tensor7_op_scalar(float a, float b);

#ifdef __cplusplus
}
#endif

#endif // CMFO_CORE_H
