#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <string>

#include "gemm_lib.hpp"
#include "gemm_lib_universal.hpp"

namespace py = pybind11;

/**
 * \brief A simple helper that wraps gemm_fp16 for Python Tensors.
 *        We do no searching for the device op here because gemm_lib already does it.
 */
bool run_gemm_half(torch::Tensor A,
                   torch::Tensor B,
                   torch::Tensor C,
                   const std::string& type_string)
{
    // Basic checks
    TORCH_CHECK(A.dtype() == torch::kHalf && B.dtype() == torch::kHalf && C.dtype() == torch::kHalf,
                "All inputs must be half precision (torch.float16).");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2,
                "All inputs must be 2D.");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t K2 = B.size(0);
    int64_t N = B.size(1);

    TORCH_CHECK(K == K2,
                "Inner dimension mismatch, A is [M, K], B is [K, N].");
    TORCH_CHECK(C.size(0) == M && C.size(1) == N,
                "C must be [M, N].");

    // Pointers
    auto A_ptr = reinterpret_cast<const void*>(A.data_ptr<at::Half>());
    auto B_ptr = reinterpret_cast<const void*>(B.data_ptr<at::Half>());
    auto C_ptr = reinterpret_cast<void*>(C.data_ptr<at::Half>());

    // Now we call gemm_fp16 from the library, passing type_string
    return gemm_fp16(static_cast<int>(M),
                     static_cast<int>(N),
                     static_cast<int>(K),
                     A_ptr,
                     B_ptr,
                     C_ptr,
                     type_string);
}

std::vector<std::string> get_all(){
      auto vector_gemm = get_all_typestrings();
      auto vector_gemm_universal = get_all_typestrings_universal();
      std::vector<std::string> all;
      all.insert(all.end(), vector_gemm.begin(), vector_gemm.end());
      all.insert(all.end(), vector_gemm_universal.begin(), vector_gemm_universal.end());

      std::cout << "All gemm type strings: " << std::endl;
      for (auto &s : vector_gemm) {
          std::cout << s << std::endl;
      }
      std::cout << "All gemm universal type strings: " << std::endl;
      for (auto &s : vector_gemm_universal) {
          std::cout << s << std::endl;
      }

      std::cout << "All type strings: " << std::endl;
      for (auto &s : all) {
          std::cout << s << std::endl;
      }
      return all;
}

PYBIND11_MODULE(bind_ck, m)
{
    m.doc() = "Composable Kernel (CK) half-precision GEMM Pybind module";

    // Expose the existing library function that enumerates all type strings
    m.def("get_all_typestrings", &get_all,
          "Return a list of available CK gemm type strings for half precision.");

    // Expose a Python function that calls gemm_fp16 with the given type_string
    m.def("run_gemm_half", &run_gemm_half,
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("type_string"),
          "Run half-precision GEMM on CK, specifying a particular kernel by type_string.");
}
