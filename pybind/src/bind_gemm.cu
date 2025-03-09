#include <pybind11/pybind11.h>
#include <torch/torch.h>     // LibTorch
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

#include <cutlass/cutlass.h>
#include "cutlassH100Wrapper.cuh"

namespace py = pybind11;

// A sample launcher function for a batched GEMM using the CUTLASS H100 wrapper.
void cutlassH100Launcher(torch::Tensor Input_A, torch::Tensor Input_B, 
                           torch::Tensor Input_C, torch::Tensor Output_D, 
                           int M, int N, int K,
                           float alpha, float beta) {
  // Use the CUTLASS H100 wrapper with chosen tile and cluster parameters.
  using MyGEMMWrapper = CutlassH100GEMMWrapper<128 ,256 ,64  ,2 ,1 ,1 ,  1, cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor,cutlass::epilogue::TmaWarpSpecializedCooperative, cutlass::gemm::KernelTmaWarpSpecializedCooperative, cutlass::gemm::PersistentScheduler>;

  // Get the current CUDA stream from PyTorch.
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  // Get element types from the wrapper.
  using ElementInputA = typename MyGEMMWrapper::ElementInputA;
  using ElementInputB = typename MyGEMMWrapper::ElementInputB;
  using ElementOutput = typename MyGEMMWrapper::ElementOutput;

  // Ensure that tensors are on CUDA and contiguous.
  TORCH_CHECK(Input_A.is_cuda(), "Input_A must be a CUDA tensor");
  TORCH_CHECK(Input_B.is_cuda(), "Input_B must be a CUDA tensor");
  TORCH_CHECK(Output_D.is_cuda(), "Output_C must be a CUDA tensor");
  TORCH_CHECK(Input_C.is_cuda(), "Bias must be a CUDA tensor when is_bias is true");
  
  Input_A = Input_A.contiguous();
  Input_B = Input_B.contiguous();
  Input_C = Input_C.contiguous();
  Output_D = Output_D.contiguous();

ElementInputA* ptr_A = reinterpret_cast<ElementInputA*>(Input_A.data_ptr());
ElementInputB* ptr_B = reinterpret_cast<ElementInputB*>(Input_B.data_ptr());
ElementOutput* ptr_C = reinterpret_cast<ElementOutput*>(Input_C.data_ptr());
ElementOutput* ptr_D = reinterpret_cast<ElementOutput*>(Output_D.data_ptr());



  
  // Create an instance of the wrapper.
  MyGEMMWrapper gemm_wrapper;
    gemm_wrapper.set_shape(M, N, K);
    gemm_wrapper.setStream(stream);
    gemm_wrapper.set_alpha(alpha);
    gemm_wrapper.set_beta(beta);
    gemm_wrapper.setA(ptr_A);
    gemm_wrapper.set_weight(ptr_B);
    gemm_wrapper.setC(ptr_C);
    
    gemm_wrapper.setD(ptr_D);
    
    // Initialize the GEMM operator with the current arguments.
    gemm_wrapper.init(beta);
    
    // Launch the GEMM kernel.
    gemm_wrapper.work();
  
  // // Synchronize the stream to ensure the operation is complete.
  // cudaStreamSynchronize(stream);
}

PYBIND11_MODULE(bind_gemm, m) {
    m.doc() = "Pybind11 bindings for CUTLASS H100 GEMM";
    m.def("gemmLauncher", &cutlassH100Launcher, "CUTLASS H100 GEMM Launcher");
}