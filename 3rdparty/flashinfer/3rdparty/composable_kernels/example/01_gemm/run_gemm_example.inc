// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tensor_operation/gpu/device/device_gemm_streamk.hpp"

template <typename DataType>
inline __host__ __device__ constexpr double get_rtol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 1e-1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 1.5e-1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <typename DataType>
inline __host__ __device__ constexpr double get_atol()
{
    if constexpr(std::is_same_v<DataType, float>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, double>)
    {
        return 1e-6;
    }
    else if constexpr(std::is_same_v<DataType, ck::half_t>)
    {
        return 1e-3;
    }
    else if constexpr(std::is_same_v<DataType, ck::bhalf_t>)
    {
        return 5e-2;
    }
    else if constexpr(std::is_same_v<DataType, int32_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, int8_t>)
    {
        return 1e-1;
    }
    else if constexpr(std::is_same_v<DataType, ck::f8_t>)
    {
        return 16.1; // 240 and 224 are acceptable
    }
    else if constexpr(std::is_same_v<DataType, ck::bf8_t>)
    {
        return 8192.1; // 57344 and 49152 are acceptable
    }
    else
    {
        return 1e-3;
    }
}

template <typename ProblemType>
bool run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
{
#if defined(BUILD_INT4_EXAMPLE) && defined(CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4)
    static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
#endif

    using namespace ck::literals;

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    auto f_get_default_stride =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(stride == 0)
            {
                // give a chance if stride is zero, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return col;
                }
                else
                {
                    return row;
                }
            }
            else
                return stride;
        };

    StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
    StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
    StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    switch(config.init_method)
    {
    case 0:
        ck::utils::FillConstant<ADataType>{static_cast<ADataType>(1.f)}(a_m_k);
        ck::utils::FillConstant<BDataType>{static_cast<BDataType>(1.f)}(b_k_n);
        break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n);
        break;
    case 2:
        ck::utils::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
        break;
    case 3:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n);
        break;
    case 4:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{1.f, 1.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{1.f, 1.f}(b_k_n);
        break;
    case 5:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-2.f, 2.f}(a_m_k);
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-2.f, 2.f}(b_k_n);
        break;
    default:
        ck::utils::FillUniformDistribution<ADataType>{-0.1f, 0.1f}(a_m_k);
        ck::utils::FillUniformDistribution<BDataType>{-0.1f, 0.1f}(b_k_n);
    }

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

#ifdef BUILD_INT4_EXAMPLE
    DeviceMem a_m_k_device_buf(sizeof(KernelADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(KernelBDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(KernelCDataType) *
                               c_m_n_device_result.mDesc.GetElementSpaceSize());

    const Tensor<KernelADataType> a_m_k_converted(a_m_k);
    const Tensor<KernelBDataType> b_k_n_converted(b_k_n);

    a_m_k_device_buf.ToDevice(a_m_k_converted.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n_converted.mData.data());
#else
    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    b_k_n_device_buf.ToDevice(b_k_n.mData.data());
#endif
    DeviceMem workspace;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    using BaseStreamK = ck::tensor_operation::device::DeviceGemmStreamK<ALayout,
                                                                        BLayout,
                                                                        CLayout,
                                                                        ADataType,
                                                                        BDataType,
                                                                        CDataType,
                                                                        AElementOp,
                                                                        BElementOp,
                                                                        CElementOp>;

    // do GEMM
    auto gemm      = DeviceGemmInstance{};
    auto invoker   = gemm.MakeInvoker();
    float ave_time = 0;

    if constexpr(std::is_same<ProblemType, ProblemSize>::value &&
                 !std::is_base_of<BaseStreamK, DeviceGemmInstance>::value)
    {
        auto argument = gemm.MakeArgument(
#ifdef BUILD_INT4_EXAMPLE
            static_cast<KernelADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<KernelBDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
            static_cast<KernelCDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
#else
            static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
#endif
            M,
            N,
            K,
            StrideA,
            StrideB,
            StrideC,
            a_element_op,
            b_element_op,
            c_element_op);

        if(!gemm.IsSupportedArgument(argument))
        {
            std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

            return true;
        }

        ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});
    }
    else if constexpr(std::is_same<ProblemType, ProblemSizeStreamK>::value &&
                      std::is_base_of<BaseStreamK, DeviceGemmInstance>::value)
    {
        auto argument = gemm.MakeArgument(
#ifdef BUILD_INT4_EXAMPLE
            static_cast<KernelADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<KernelBDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
            static_cast<KernelCDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
#else
            static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
#endif
            M,
            N,
            K,
            StrideA,
            StrideB,
            StrideC,
            a_element_op,
            b_element_op,
            c_element_op,
            problem_size.NumSKBlocks);

        if(!gemm.IsSupportedArgument(argument))
        {
            std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;

            return true;
        }

        std::size_t workspace_size = gemm.GetWorkSpaceSize(&argument);
        if(workspace_size != 0)
        {
            workspace.Realloc(workspace_size);
            gemm.SetWorkSpacePointer(&argument, workspace.GetDeviceBuffer());
        }

        ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

#if 0
        // TODO!!!!!
        if(workspace_size != 0){
            float * ws_ptr = reinterpret_cast<float*>(malloc(workspace_size));
            size_t ws_dwords = workspace_size / sizeof(float);
            workspace.FromDevice(ws_ptr);

            for(size_t i = 0; i < ws_dwords; i++) {
                uint32_t rere = reinterpret_cast<uint32_t*>(ws_ptr)[i];
                printf("%4lu : %f(0x%08x)\n", i, ws_ptr[i], rere);
            }
            free(ws_ptr);
        }
#endif
    }

    std::size_t flop = 2_uz * M * N * K;
    std::size_t num_btype =
        sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << gemm.GetTypeString() << std::endl;

    if(config.do_verification)
    {
        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

#ifdef BUILD_INT4_EXAMPLE
        Tensor<CDataType> c_m_n_device_result_converted(c_m_n_host_result.mDesc);

        c_m_n_device_buf.FromDevice(c_m_n_device_result_converted.mData.data());

        c_m_n_device_result = c_m_n_device_result_converted.CopyAsType<CDataType>();

        return ck::utils::check_err(c_m_n_device_result_converted, c_m_n_host_result);
#else
        c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

        return ck::utils::check_err(c_m_n_device_result,
                                    c_m_n_host_result,
                                    "Error: Incorrect results!",
                                    get_rtol<CDataType>(),
                                    get_atol<CDataType>());
#endif
    }

    return true;
}

bool run_gemm_example(int argc, char* argv[])
{
    ProblemSize problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_gemm(problem_size, config);
}

bool run_gemm_streamk_example(int argc, char* argv[])
{
    ProblemSizeStreamK problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) || run_gemm(problem_size, config);
}
