#include "gemm_lib.hpp"
#include <iostream>
#include <vector>
#include <string>

// CK includes
#include "common.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v2.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"
// #include "ck/library/tensor_operation_instance/gpu/gemm_universal.hpp"
// #include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"

// #include "ck/ck.hpp"
// #include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
// #include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp"
// #include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

// #include "ck/library/tensor_operation_instance/gpu/gemm_universal.hpp"

// #include "ck/library/utility/check_err.hpp"
// #include "ck/library/utility/device_memory.hpp"
// #include "ck/library/utility/host_tensor.hpp"
// #include "ck/library/utility/host_tensor_generator.hpp"
// #include "ck/library/utility/literals.hpp"
// #include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

/**
 * \brief CK Type definitions, matching half-precision (fp16).
 */
using ADataType   = ck::half_t;
using BDataType   = ck::half_t;
using CDataType   = ck::half_t;
using AccDataType = float;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

// The CK device GEMM concept for half-precision
using DeviceOp = ck::tensor_operation::device::DeviceGemm<
    ALayout,
    BLayout,
    CLayout,
    ADataType,
    BDataType,
    CDataType,
    AElementOp,
    BElementOp,
    CElementOp
>;

//-----------------------------------------------------
// Return all type strings
std::vector<std::string> get_all_typestrings()
{
    auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOp>::GetInstances();
    std::vector<std::string> results;
    results.reserve(op_ptrs.size());

    for(const auto& op : op_ptrs)
        results.push_back(op->GetTypeString());

    return results;
}

//-----------------------------------------------------
// gemm_fp16 that does its own search for the correct instance
bool gemm_fp16(int M, int N, int K,
               const void* d_A_,
               const void* d_B_,
               void*       d_C_,
               const std::string& type_string)
{
    // Cast to CK's half type pointers
    auto d_A = static_cast<const ADataType*>(d_A_);
    auto d_B = static_cast<const BDataType*>(d_B_);
    auto d_C = static_cast<CDataType*>(d_C_);

    // Find the correct instance that matches the provided type_string
    auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<DeviceOp>::GetInstances();
    DeviceOp* found_op = nullptr;

    for(auto& op : op_ptrs)
    {
        if(op->GetTypeString() == type_string)
        {
            found_op = op.get();
            break;
        }
    }
    if(!found_op)
    {
        std::cerr << "[CK GEMM] No instance found for type_string = " 
                  << type_string << std::endl;
        return false;
    }

    // Create the pass-through operators
    AElementOp a_op{};
    BElementOp b_op{};
    CElementOp c_op{};

    // Build argument & invoker
    auto argument_ptr = found_op->MakeArgumentPointer(
        const_cast<ADataType*>(d_A),
        const_cast<BDataType*>(d_B),
        d_C,
        M, N, K,
        /*stride_A=*/ K,  // row-major
        /*stride_B=*/ N,  // row-major
        /*stride_C=*/ N,  // row-major
        a_op, b_op, c_op
    );
    auto invoker_ptr = found_op->MakeInvokerPointer();

    // Check if supported
    if(!found_op->IsSupportedArgument(argument_ptr.get()))
    {
        std::cerr << "[CK GEMM] Unsupported config for type_string=" 
                  << type_string << std::endl;
        return false;
    }

    // Launch kernel
    float t = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true, 0, 5, 10});
    std::cout << "[CK GEMM] type_string=" << type_string 
              << ", time=" << t << " ms" << std::endl;

    return true;
}
