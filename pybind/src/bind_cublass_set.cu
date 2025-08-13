#include <cuda.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define CUBLAS_RT(call)                                                                            \
  do {                                                                                             \
    cublasStatus_t _status = (call);                                                               \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                                        \
      std::cerr << "ERROR: CUBLAS RT call \"" << #call << "\" in line " << __LINE__ << " of file " \
                << __FILE__ << " failed with " << cublasGetStatusString(_status) << std::endl;     \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

void SetCublasSMCountTarget(int sm_target) {
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  CUBLAS_RT(cublasSetSmCountTarget(handle, sm_target));
}

PYBIND11_MODULE(bind_cublass_set, m) {
  m.def("set_cublas_sm_count_target", &SetCublasSMCountTarget,
        "Set cublas sm count target");
}