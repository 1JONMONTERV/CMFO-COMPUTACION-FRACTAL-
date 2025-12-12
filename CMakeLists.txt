
# Top-level CMake for CMFO-UNIVERSE

cmake_minimum_required(VERSION 3.18)
project(cmfo_universe LANGUAGES C)

# Source layout
set(CMFO_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(CMFO_SRC_DIR "${CMAKE_SOURCE_DIR}/src")

file(GLOB CMFO_SOURCES
    "${CMFO_SRC_DIR}/cmfo_core.c"
    "${CMFO_SRC_DIR}/cmfo_mat7.c"
    "${CMFO_SRC_DIR}/cmfo_mat7_mul.c"
    "${CMFO_SRC_DIR}/cmfo_mat7_det.c"
    "${CMFO_SRC_DIR}/cmfo_mat7_inv.c"
    "${CMFO_SRC_DIR}/cmfo_tensor7.c"
)

# Library target (core C implementation)
add_library(cmfo_core STATIC ${CMFO_SOURCES})
target_include_directories(cmfo_core PUBLIC ${CMFO_INCLUDE_DIR})
target_compile_features(cmfo_core PUBLIC c_std_99)

# Optionally configure CUDA pieces if CUDA sources present and CUDA is available
if(EXISTS "${CMAKE_SOURCE_DIR}/cuda")
  enable_language(CUDA)
  cmake_policy(SET CMP0104 NEW)
  # Default to a reasonable architecture if not set by user
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()

  file(GLOB CUDA_SOURCES "${CMAKE_SOURCE_DIR}/cuda/*.cu")
  if(CUDA_SOURCES)
    add_library(cmfo_cuda STATIC ${CUDA_SOURCES})
    target_include_directories(cmfo_cuda PUBLIC ${CMFO_INCLUDE_DIR})
    set_target_properties(cmfo_cuda PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(cmfo_cuda PUBLIC cmfo_core)

    # Add CUDA test executable if test exists
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_cuda.cu")
      add_executable(test_cuda tests/test_cuda.cu)
      set_target_properties(test_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
      )
      target_link_libraries(test_cuda PRIVATE cmfo_cuda cmfo_core cudart)
    endif()
  endif()
endif()

# Enable tests and add tests folder
enable_testing()
add_subdirectory(tests)

