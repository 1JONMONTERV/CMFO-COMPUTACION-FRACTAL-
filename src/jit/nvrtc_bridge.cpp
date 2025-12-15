

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <nvrtc.h>
#include <string>
#include <vector>

#define NVRTC_SAFE_CALL(x)                                                     \
  do {                                                                         \
    nvrtcResult result = x;                                                    \
    if (result != NVRTC_SUCCESS) {                                             \
      printf("\nerror: " #x " failed with error %s\n",                         \
             nvrtcGetErrorString(result));                                     \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define CUDA_SAFE_CALL(x)                                                      \
  do {                                                                         \
    CUresult result = x;                                                       \
    if (result != CUDA_SUCCESS) {                                              \
      const char *msg;                                                         \
      cuGetErrorName(result, &msg);                                            \
      printf("\nerror: " #x " failed with error %s\n", msg);                   \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

// Kernel Cache Structure
struct CachedKernel {
  CUmodule module;
  CUfunction function;
};

// Global Cache State
static std::map<int, CachedKernel> kernel_cache;
static int next_kernel_id = 1;

// Global Context State
static CUcontext global_context = NULL;
static bool is_initialized = false;

extern "C" {

// Initialize CUDA Driver API and Context
DLLEXPORT int cmfo_jit_init() {
  if (is_initialized)
    return 0;

  CUresult res = cuInit(0);
  if (res != CUDA_SUCCESS) {
    printf("cmfo_jit_init: cuInit failed\n");
    return 1;
  }

  CUdevice cuDevice;
  res = cuDeviceGet(&cuDevice, 0);
  if (res != CUDA_SUCCESS) {
    printf("cmfo_jit_init: cuDeviceGet failed\n");
    return 1;
  }

  res = cuCtxCreate(&global_context, 0, cuDevice);
  if (res != CUDA_SUCCESS) {
    const char *msg;
    cuGetErrorName(res, &msg);
    printf("cmfo_jit_init: cuCtxCreate failed: %s\n", msg);
    return 1;
  }

  is_initialized = true;
  printf("[NATIVE] CUDA Context Initialized.\n");
  return 0;
}

// [NEW] Load Kernel into Cache
// Returns: ID > 0 on success, -1 on failure
DLLEXPORT int cmfo_jit_load_cache(const char *kernel_source,
                                  const char *kernel_name) {
  if (!kernel_source)
    return -1;

  // 1. Create NVRTC Program
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernel_source, "cmfo_kernel.cu", 0,
                                     NULL, NULL));

  // 2. Compile
  const char *opts[] = {"--gpu-architecture=compute_60"}; // Safer
  nvrtcResult compileResult = nvrtcCompileProgram(prog, 1, opts);

  if (compileResult != NVRTC_SUCCESS) {
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    std::vector<char> log(logSize);
    nvrtcGetProgramLog(prog, log.data());
    printf("COMPILATION ERROR:\n%s\n", log.data());
    return -1;
  }

  // 3. Get PTX
  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  std::vector<char> ptx(ptxSize);
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx.data()));
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  // 4. Load Module
  CUmodule module;
  CUDA_SAFE_CALL(cuModuleLoadData(&module, ptx.data()));

  // 5. Get Function
  CUfunction kernel;
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, kernel_name));

  // 6. Store in Cache
  int id = next_kernel_id++;
  CachedKernel ck;
  ck.module = module;
  ck.function = kernel;
  kernel_cache[id] = ck;

  return id;
}

// [NEW] Launch Cached Kernel
DLLEXPORT int cmfo_jit_launch_cache(int kernel_id, float *v_ptr, float *h_ptr,
                                    float *out_ptr, int N) {
  // 1. Lookup
  if (kernel_cache.find(kernel_id) == kernel_cache.end()) {
    printf("Error: Kernel ID %d not found.\n", kernel_id);
    return -1;
  }

  CachedKernel &ck = kernel_cache[kernel_id];

  // 2. Allocate Device Buffers (Still doing per-run transfer for now)
  CUdeviceptr d_v, d_h, d_out;
  size_t bytes = N * 7 * sizeof(float);

  CUDA_SAFE_CALL(cuMemAlloc(&d_v, bytes));
  CUDA_SAFE_CALL(cuMemAlloc(&d_h, bytes));
  CUDA_SAFE_CALL(cuMemAlloc(&d_out, bytes));

  CUDA_SAFE_CALL(cuMemcpyHtoD(d_v, v_ptr, bytes));
  CUDA_SAFE_CALL(cuMemcpyHtoD(d_h, h_ptr, bytes));

  // 3. Launch
  void *args[] = {&d_v, &d_h, &d_out, &N};

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  CUDA_SAFE_CALL(cuLaunchKernel(ck.function, blocksPerGrid, 1, 1,
                                threadsPerBlock, 1, 1, 0, 0, args, 0));

  CUDA_SAFE_CALL(cuCtxSynchronize());

  CUDA_SAFE_CALL(cuMemcpyDtoH(out_ptr, d_out, bytes));

  // 4. Cleanup Memory
  CUDA_SAFE_CALL(cuMemFree(d_v));
  CUDA_SAFE_CALL(cuMemFree(d_h));
  CUDA_SAFE_CALL(cuMemFree(d_out));

  return 0;
}

// Legacy One-Shot Wrapper (Backward Compatibility)
DLLEXPORT int cmfo_jit_compile_and_run(const char *kernel_source,
                                       const char *kernel_name, float *v_ptr,
                                       float *h_ptr, float *out_ptr, int N) {
  int id = cmfo_jit_load_cache(kernel_source, kernel_name);
  if (id < 0)
    return id;
  return cmfo_jit_launch_cache(id, v_ptr, h_ptr, out_ptr, N);
}

} // extern "C"
