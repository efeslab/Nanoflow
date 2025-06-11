#include <cuda.h>
#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define CUDA_RT(call)                                                          \
  do {                                                                         \
    cudaError_t _status = (call);                                              \
    if (_status != cudaSuccess) {                                              \
      std::cerr << "ERROR: CUDA RT call \"" << #call << "\" in line "          \
                << __LINE__ << " of file " << __FILE__ << " failed with "      \
                << cudaGetErrorString(_status) << std::endl;                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define CUDA_DRV(call)                                                         \
  do {                                                                         \
    CUresult _status = (call);                                                 \
    if (_status != CUDA_SUCCESS) {                                             \
      const char *err_str;                                                     \
      cuGetErrorString(_status, &err_str);                                     \
      std::cerr << "ERROR: CUDA DRV call \"" << #call << "\" in line "         \
                << __LINE__ << " of file " << __FILE__ << " failed with "      \
                << err_str << std::endl;                                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define ASSERT(condition, message)                                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "ERROR: Assertion failed in line " << __LINE__              \
                << " of file " << __FILE__ << ": " << message << std::endl;    \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)


std::vector<int64_t> CreateGreenCtxStreamByPercent(float smA, float smB,
                                                   int device) {
  CUgreenCtx gctx[3];
  CUdevResourceDesc desc[3];
  CUdevResource input;
  CUdevResource resources[4];
  CUstream streamA;
  CUstream streamB;

  unsigned int nbGroups = 1;

  if (smA + smB > 1.0) {
    ASSERT(false, "Sum of SM percentages cannot exceed 1.0");
  }

  if (smA <= 0.0 || smB <= 0.0) {
    ASSERT(false, "SM percentages must be greater than 0.0");
  }

  // Initialize device
  CUDA_RT(cudaInitDevice(device, 0, 0));

  // Query input SMs
  CUDA_DRV(cuDeviceGetDevResource((CUdevice)device, &input,
                                  CU_DEV_RESOURCE_TYPE_SM));
  // We want 3/4 the device for our green context
  unsigned int minCount = (unsigned int)((float)input.sm.smCount * (smA + smB));
  unsigned int minCountA = (unsigned int)((float)input.sm.smCount * smA);

  // Split resources
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[2], &nbGroups, &input,
                                       &resources[3], 0, minCount));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[2], &resources[2], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[2], desc[2], (CUdevice)device,
                            CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuGreenCtxGetDevResource(gctx[2], &input, CU_DEV_RESOURCE_TYPE_SM));
  CUDA_DRV(cuDevSmResourceSplitByCount(&resources[0], &nbGroups, &input,
                                       &resources[1], 0, minCountA));

  CUDA_DRV(cuDevResourceGenerateDesc(&desc[0], &resources[0], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[0], desc[0], (CUdevice)device,
                            CU_GREEN_CTX_DEFAULT_STREAM));
  CUDA_DRV(cuDevResourceGenerateDesc(&desc[1], &resources[1], 1));
  CUDA_DRV(cuGreenCtxCreate(&gctx[1], desc[1], (CUdevice)device,
                            CU_GREEN_CTX_DEFAULT_STREAM));

  CUDA_DRV(
      cuGreenCtxStreamCreate(&streamA, gctx[0], CU_STREAM_NON_BLOCKING, 0));
  CUDA_DRV(
      cuGreenCtxStreamCreate(&streamB, gctx[1], CU_STREAM_NON_BLOCKING, 0));

  int smCountA = resources[0].sm.smCount;
  int smCountB = resources[1].sm.smCount;

  std::vector<int64_t> vec = {(int64_t)streamA, (int64_t)streamB, smCountA,
                              smCountB};
  return vec;
}

PYBIND11_MODULE(bind_green_ctx, m) {
  m.def("create_greenctx_stream_by_percent", &CreateGreenCtxStreamByPercent,
                    "Create stream with green context");
}
