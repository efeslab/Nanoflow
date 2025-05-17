#include <pybind11/pybind11.h>
#include <torch/torch.h>     // LibTorch
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include <iostream>
#include <unordered_map>
#include <cutlass/cutlass.h>
#include "cutlassGemmWrapper.cuh"
#include "cutlassH100Wrapper.cuh"
#include "gemmFactory.cuh"

namespace py = pybind11;
using ElementInputA = typename BaseGEMMWrapper::ElementInputA;
using ElementInputB = typename BaseGEMMWrapper::ElementInputB;
using ElementOutput = typename BaseGEMMWrapper::ElementOutput;

std::unordered_map<std::string, BaseGEMMWrapper*> gemm_map;
// BaseGEMMWrapper* gemm_wrapper = nullptr;
ElementInputA* gemm_input = nullptr;
ElementInputB* gemm_weight = nullptr;
ElementOutput* gemm_bias = nullptr;
ElementOutput* gemm_output = nullptr;

void configGEMM(const std::string& gemm_tag, const std::string& gemm_name, int M, int N, int K, float alpha, float beta) {

  BaseGEMMWrapper* gemm_wrapper = generateGEMM(gemm_tag);
  gemm_wrapper->set_shape(M, N, K);
  gemm_wrapper->set_alpha(alpha);
  gemm_wrapper->set_beta(beta);

  gemm_wrapper->init(beta);

  gemm_map[gemm_name] = gemm_wrapper;
}

void runGEMM(const std::string& gemm_name, torch::Tensor Input_A, torch::Tensor Input_B, torch::Tensor Input_C, torch::Tensor Output_D, intptr_t stream_handle = 0ULL) {
  // Ensure that tensors are on CUDA and contiguous.
  // TORCH_CHECK(Input_B.is_cuda(), "Input_B must be a CUDA tensor");
  // Input_B = Input_B.contiguous();
  cudaStream_t stream = nullptr;
  if (stream_handle == 0ULL) {
      stream = at::cuda::getCurrentCUDAStream();
  } else {
      stream = reinterpret_cast<cudaStream_t>(stream_handle);
  }
  // continuous
  TORCH_CHECK(Input_A.is_cuda(), "Input_A must be a CUDA tensor");
  TORCH_CHECK(Input_B.is_cuda(), "Input_B must be a CUDA tensor");
  TORCH_CHECK(Input_C.is_cuda(), "Input_C must be a CUDA tensor");
  TORCH_CHECK(Output_D.is_cuda(), "Output_D must be a CUDA tensor");

  Input_A = Input_A.contiguous();
  Input_B = Input_B.contiguous();
  Input_C = Input_C.contiguous();
  Output_D = Output_D.contiguous();

  gemm_input = reinterpret_cast<ElementInputA*>(Input_A.data_ptr());
  gemm_weight = reinterpret_cast<ElementInputB*>(Input_B.data_ptr());
  gemm_bias = reinterpret_cast<ElementOutput*>(Input_C.data_ptr());
  gemm_output = reinterpret_cast<ElementOutput*>(Output_D.data_ptr());
  BaseGEMMWrapper* gemm_wrapper = gemm_map[gemm_name];

    gemm_wrapper->setA(gemm_input);
    gemm_wrapper->set_weight(gemm_weight);
    gemm_wrapper->setC(gemm_bias);
    gemm_wrapper->setD(gemm_output);

    gemm_wrapper->updateArgument();
    gemm_wrapper->setStream(stream);
    gemm_wrapper->work(); // Launch the GEMM operation

  // // // Synchronize the stream to ensure the operation is complete.
  // cudaStreamSynchronize(stream);
}

PYBIND11_MODULE(bind_gemm, m) {
    m.doc() = "Pybind11 bindings for GEMM";
    m.def("configGEMM", &configGEMM, "GEMM Configuration");
    m.def("gemmLauncher", &runGEMM, "GEMM Launcher");
}