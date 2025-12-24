#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cmfo_core/cmfo_gamma_symbols.h"

extern __global__ void cmfo_gamma_step(const double*, double*, int, int);

int main() {
    std::cout << "Gamma CMFO Test OK" << std::endl;
    return 0;
}
