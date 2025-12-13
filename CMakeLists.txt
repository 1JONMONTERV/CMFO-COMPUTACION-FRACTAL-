cmake_minimum_required(VERSION 3.10)
project(CMFO_Universe LANGUAGES C CXX CUDA)

# ===========================================================
# CONFIGURACIÓN GENERAL
# ===========================================================
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
include_directories(include)

# ===========================================================
# CMFO CORE LIB (CPU)
# ===========================================================
add_library(cmfo_core STATIC
    src/cmfo_core.c
    src/cmfo_mat7.c
    src/cmfo_mat7_mul.c
    src/cmfo_mat7_det.c
    src/cmfo_mat7_inv.c
    src/cmfo_tensor7.c
    src/cmfo_soliton.c 
)

# ===========================================================
# CMFO CUDA LIB (GPU)
# ===========================================================
# Configuración de Arquitectura (SM86 por defecto, ajustable)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

# Solo compilar CUDA si se encuentran los archivos (por seguridad)
if(EXISTS "${CMAKE_SOURCE_DIR}/cuda/cmfo_kernels.cu")
    add_library(cmfo_cuda STATIC
        cuda/cmfo_kernels.cu
        cuda/theta_cmfo_kernel.cu
    )
    set_target_properties(cmfo_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# ===========================================================
# TESTS
# ===========================================================
enable_testing()

# Test Core (Existente)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_core.c")
    add_executable(test_core tests/test_core.c)
    target_link_libraries(test_core PRIVATE cmfo_core)
    add_test(NAME CoreTest COMMAND test_core)
endif()

# Test CUDA Original
if(TARGET cmfo_cuda AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_cuda.cu")
    add_executable(test_cuda tests/test_cuda.cu)
    target_link_libraries(test_cuda PRIVATE cmfo_cuda cmfo_core cudart)
    set_target_properties(test_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME CudaTest COMMAND test_cuda)
endif()

# Test Theta Kernel
if(TARGET cmfo_cuda AND EXISTS "${CMAKE_SOURCE_DIR}/tests/test_theta_kernel.cu")
    add_executable(test_theta_kernel tests/test_theta_kernel.cu)
    target_link_libraries(test_theta_kernel PRIVATE cmfo_cuda cmfo_core cudart)
    set_target_properties(test_theta_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    add_test(NAME ThetaKernelTest COMMAND test_theta_kernel)
endif()

# Test Soliton Collision (Nuevo Hito 4) - Pure CPU C
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_soliton_collision.c")
    add_executable(test_soliton tests/test_soliton_collision.c)
    target_link_libraries(test_soliton PRIVATE cmfo_core m) # Link math lib
    add_test(NAME SolitonCollisionTest COMMAND test_soliton)
endif()

# Test Matrix Stability (Nuevo Hito Extra)
if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_mat7_stability.c")
    add_executable(test_mat7_stab tests/test_mat7_stability.c)
    target_link_libraries(test_mat7_stab PRIVATE cmfo_core m)
    add_test(NAME Mat7StabilityTest COMMAND test_mat7_stab)
endif()
