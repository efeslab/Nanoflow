#include <c10/cuda/CUDAStream.h>
#include <cassert>
#include <nccl.h>
#include <torch/extension.h>

typedef std::vector<uint8_t> ncclIdWrapper;
auto NcclIdToWrapper(ncclUniqueId id) -> ncclIdWrapper {
  return ncclIdWrapper(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES);
}
auto WrapperToNcclId(ncclIdWrapper wrapper) -> ncclUniqueId {
  ncclUniqueId id;
  std::copy(wrapper.begin(), wrapper.end(), id.internal);
  return id;
}


class NCCLWrapper {
private:
  ncclComm_t comm_;
  int rank_;
  int world_size_;
  bool initialized_;

public:
  NCCLWrapper(int rank, int world_size, ncclIdWrapper unique_id) {
    rank_ = rank;
    world_size_ = world_size;
    ncclCommInitRank(&comm_, world_size, WrapperToNcclId(unique_id), rank);
    initialized_ = true;
  }

  ~NCCLWrapper() {
    if (initialized_) {
      ncclCommDestroy(comm_);
    }
  }

  static auto get_nccl_unique_id() -> ncclIdWrapper {
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    return NcclIdToWrapper(id);
  }

  auto all_reduce(torch::Tensor &input, const std::string &op_str = "sum")
      -> torch::Tensor {
    assert(initialized_ && "NCCLWrapper not initialized");
    assert(input.is_cuda() && "Input tensor must be on GPU");
    assert(input.is_contiguous() && "Input tensor must be contiguous");

    ncclRedOp_t op;
    if (op_str == "sum") {
      op = ncclSum;
    } else if (op_str == "prod" || op_str == "product") {
      op = ncclProd;
    } else if (op_str == "max") {
      op = ncclMax;
    } else if (op_str == "min") {
      op = ncclMin;
    } else {
      throw std::runtime_error("Unsupported reduction operation: " + op_str);
    }

    auto input_ptr = input.data_ptr();
    auto num_elements = input.numel();
    auto dtype = input.scalar_type();

    ncclDataType_t nccl_dtype;
    switch (dtype) {
    case torch::kFloat32:
      nccl_dtype = ncclFloat32;
      break;
    case torch::kFloat16:
      nccl_dtype = ncclFloat16;
      break;
    case torch::kInt32:
      nccl_dtype = ncclInt32;
      break;
    case torch::kInt64:
      nccl_dtype = ncclInt64;
      break;
    default:
      assert(false && "Unsupported data type for all-reduce");
    }

    ncclAllReduce(input_ptr, input_ptr, num_elements, nccl_dtype, op, comm_,
                  c10::cuda::getCurrentCUDAStream());

    return input;
  }

  void barrier() {
    auto tensor =
        torch::ones({}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    ncclAllReduce(tensor.data_ptr(), tensor.data_ptr(), 1, ncclInt32, ncclSum,
                  comm_, c10::cuda::getCurrentCUDAStream());
  }

  void send(torch::Tensor &input, int dst) {
    assert(initialized_ && "NCCLWrapper not initialized");
    assert(input.is_cuda() && "Input tensor must be on GPU");
    assert(input.is_contiguous() && "Input tensor must be contiguous");

    ncclSend(input.data_ptr(), input.numel(), ncclFloat32, dst, comm_,
             c10::cuda::getCurrentCUDAStream());
  }

  void recv(torch::Tensor &input, int src) {
    assert(initialized_ && "NCCLWrapper not initialized");
    assert(input.is_cuda() && "Input tensor must be on GPU");
    assert(input.is_contiguous() && "Input tensor must be contiguous");

    auto dtype = input.scalar_type();
    ncclDataType_t nccl_dtype;
    switch (dtype) {
    case torch::kFloat32:
      nccl_dtype = ncclFloat32;
      break;
    case torch::kFloat16:
      nccl_dtype = ncclFloat16;
      break;
    default:
      assert(false && "Unsupported data type for recv");
    }
    ncclRecv(input.data_ptr(), input.numel(), nccl_dtype, src, comm_,
             c10::cuda::getCurrentCUDAStream());
  }
};

PYBIND11_MODULE(bind_all_reduce, m) {
  py::class_<NCCLWrapper>(m, "NCCLWrapper")
      .def(py::init<int, int, ncclIdWrapper>())
      .def_static("get_nccl_unique_id", &NCCLWrapper::get_nccl_unique_id,
                  "Get the unique ID for NCCL initialization")
      .def("all_reduce", &NCCLWrapper::all_reduce,
           "Perform all-reduce operation")
      .def("barrier", &NCCLWrapper::barrier, "Synchronize all processes")
      .def("send", &NCCLWrapper::send, "Send a tensor to a destination process")
      .def("recv", &NCCLWrapper::recv, "Receive a tensor from a source process");
}
