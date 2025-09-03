#include <c10/cuda/CUDAStream.h>
#include <cassert>
#include <nccl.h>
#include <torch/extension.h>
#include <memory>
#include <cuda_runtime.h>

typedef std::vector<uint8_t> ncclIdWrapper;
auto NcclIdToWrapper(ncclUniqueId id) -> ncclIdWrapper {
  return ncclIdWrapper(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES);
}
auto WrapperToNcclId(ncclIdWrapper wrapper) -> ncclUniqueId {
  ncclUniqueId id;
  std::copy(wrapper.begin(), wrapper.end(), id.internal);
  return id;
}

// Handle class for async operations
class NCCLHandle {
private:
  cudaEvent_t event_;
  torch::Tensor tensor_;  // Keep reference to tensor to prevent deallocation
  bool completed_;
  
public:
  NCCLHandle(torch::Tensor tensor, cudaStream_t stream) : tensor_(tensor), completed_(false) {
    // Create CUDA event and record it on the stream
    cudaEventCreate(&event_);
    cudaEventRecord(event_, stream);
  }
  
  ~NCCLHandle() {
    if (!completed_) {
      // If handle is destroyed without calling wait(), synchronize automatically
      cudaEventSynchronize(event_);
    }
    cudaEventDestroy(event_);
  }
  
  void wait() {
    if (!completed_) {
      cudaEventSynchronize(event_);
      completed_ = true;
    }
  }
  
  bool is_completed() {
    if (completed_) {
      return true;
    }
    
    cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) {
      completed_ = true;
      return true;
    } else if (status == cudaErrorNotReady) {
      return false;
    } else {
      throw std::runtime_error("CUDA event query failed");
    }
  }
  
  torch::Tensor& get_tensor() {
    return tensor_;
  }
};

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

  // Synchronous all-reduce (original method)
  auto all_reduce(torch::Tensor &input, const std::string &op_str = "sum")
      -> torch::Tensor {
    auto handle = all_reduce_async(input, op_str);
    handle->wait();
    return handle->get_tensor();
  }

  // Asynchronous all-reduce that returns a handle
  auto all_reduce_async(torch::Tensor &input, const std::string &op_str = "sum")
      -> std::shared_ptr<NCCLHandle> {
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

    auto cuda_stream = c10::cuda::getCurrentCUDAStream();
    
    // Launch the NCCL operation
    ncclAllReduce(input_ptr, input_ptr, num_elements, nccl_dtype, op, comm_,
                  cuda_stream);

    // Create and return handle
    return std::make_shared<NCCLHandle>(input, cuda_stream);
  }

  // Asynchronous all-reduce with separate output tensor
  auto all_reduce_async(torch::Tensor &input, torch::Tensor &output, 
                       const std::string &op_str = "sum")
      -> std::shared_ptr<NCCLHandle> {
    assert(initialized_ && "NCCLWrapper not initialized");
    assert(input.is_cuda() && "Input tensor must be on GPU");
    assert(output.is_cuda() && "Output tensor must be on GPU");
    assert(input.is_contiguous() && "Input tensor must be contiguous");
    assert(output.is_contiguous() && "Output tensor must be contiguous");
    assert(input.numel() == output.numel() && "Input and output tensors must have same number of elements");

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
    auto output_ptr = output.data_ptr();
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

    auto cuda_stream = c10::cuda::getCurrentCUDAStream();
    
    // Launch the NCCL operation
    ncclAllReduce(input_ptr, output_ptr, num_elements, nccl_dtype, op, comm_,
                  cuda_stream);

    // Create and return handle with output tensor
    return std::make_shared<NCCLHandle>(output, cuda_stream);
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

auto get_nccl_unique_id() -> ncclIdWrapper {
  ncclUniqueId id;
  ncclGetUniqueId(&id);
  return NcclIdToWrapper(id);
}

PYBIND11_MODULE(bind_all_reduce, m) {
  py::class_<NCCLHandle>(m, "NCCLHandle")
      .def("wait", &NCCLHandle::wait, "Wait for the operation to complete")
      .def("is_completed", &NCCLHandle::is_completed, "Check if operation is completed")
      .def("get_tensor", &NCCLHandle::get_tensor, py::return_value_policy::reference,
           "Get the tensor associated with this handle");
  
  py::class_<NCCLWrapper>(m, "NCCLWrapper")
      .def(py::init<int, int, ncclIdWrapper>())
      .def("all_reduce", &NCCLWrapper::all_reduce, "Perform synchronous all-reduce operation")
      .def("all_reduce_async", 
           py::overload_cast<torch::Tensor&, const std::string&>(&NCCLWrapper::all_reduce_async),
           "Perform asynchronous all-reduce operation (in-place)")
      .def("all_reduce_async", 
           py::overload_cast<torch::Tensor&, torch::Tensor&, const std::string&>(&NCCLWrapper::all_reduce_async),
           "Perform asynchronous all-reduce operation with separate output tensor")
      .def("barrier", &NCCLWrapper::barrier, "Synchronize all processes")
      .def("send", &NCCLWrapper::send, "Send tensor to destination rank")
      .def("recv", &NCCLWrapper::recv, "Receive tensor from source rank");
  
  m.def("get_nccl_unique_id", &get_nccl_unique_id,
        "Get the unique ID for NCCL initialization");
}