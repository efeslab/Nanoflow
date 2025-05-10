#include <pybind11/pybind11.h>
// #include <pybind11/torch.h>  // PyTorch tensor support
#include <torch/torch.h>     // LibTorch
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <rms_norm.cuh>
#include <stdio.h>


namespace py = pybind11;

template <typename T>
void rms_norm_wrapper(torch::Tensor output, torch::Tensor input, torch::Tensor weight, float epsilon, intptr_t stream_handle) {
  // Check that tensors are CUDA tensors and contiguous.
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

  output = output.contiguous();
  input = input.contiguous();
  weight = weight.contiguous();

  // Get dimensions; assume input is 2D: [rows, columns]
  int rows = input.size(0);
  int columns = input.size(1);

  cudaStream_t stream = nullptr;
  if (stream_handle == 0ULL) {
    stream = at::cuda::getCurrentCUDAStream();
  } else {
    stream = reinterpret_cast<cudaStream_t>(stream_handle);
  }

  // Call the templated rms_norm function.
  bool ret = rms_norm<T>(
      reinterpret_cast<T*>(output.data_ptr()),
      reinterpret_cast<const T*>(input.data_ptr()),
      reinterpret_cast<const T*>(weight.data_ptr()),
      rows,
      columns,
      epsilon,
      stream);
  TORCH_CHECK(ret, "rms_norm kernel launch failed (columns must be a multiple of 8?)");
}

void rms_norm_dispatch(torch::Tensor output, torch::Tensor input, torch::Tensor weight, float epsilon, intptr_t stream_handle = 0ULL) {
  // Dispatch based on tensor dtype.
  if (output.dtype() == torch::kHalf) {
    rms_norm_wrapper<nv_half>(output, input, weight, epsilon, stream_handle);
  } else if (output.dtype() == torch::kBFloat16) {
    rms_norm_wrapper<nv_bfloat16>(output, input, weight, epsilon, stream_handle);
  } else {
    TORCH_CHECK(false, "Unsupported tensor dtype for rms_norm; use torch.kHalf or torch.kBFloat16.");
  }
}

PYBIND11_MODULE(bind_rms_norm, m) {
    m.doc() = "Pybind11 bindings for the RMS Norm kernel";
    m.def("rms_norm", &rms_norm_dispatch,
          "Apply RMS Norm kernel to the input tensor",
          py::arg("output"), py::arg("input"), py::arg("weight"), py::arg("epsilon"), py::arg("stream_handle") = 0ULL);
  }
