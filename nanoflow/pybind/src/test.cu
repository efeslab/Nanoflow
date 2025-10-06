#include <pybind11/pybind11.h>
// #include <pybind11/torch.h>  // PyTorch tensor support
#include <torch/torch.h>     // LibTorch
#include <torch/extension.h>

namespace py = pybind11;

// Function to multiply all elements of the tensor by 2
torch::Tensor multiply_tensor(torch::Tensor input_tensor) {
    // Ensure the tensor is on CPU and contiguous
    input_tensor = input_tensor.to(torch::kCPU).contiguous();

    // Multiply the tensor by 2
    torch::Tensor result = input_tensor * 2;

    return result;
}

PYBIND11_MODULE(example_module, m) {
    m.doc() = "Pybind11 example with PyTorch tensor support"; // Module docstring
    m.def("multiply_tensor", &multiply_tensor, "Multiply all elements of a tensor by 2",
          py::arg("input_tensor"));
}
