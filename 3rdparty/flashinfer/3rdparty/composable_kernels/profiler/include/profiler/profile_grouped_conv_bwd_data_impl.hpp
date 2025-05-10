// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/library/utility/convolution_host_tensor_descriptor_helper.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_bwd_data.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data.hpp"

namespace ck {
namespace profiler {

template <ck::index_t NDimSpatial,
          typename OutLayout,
          typename WeiLayout,
          typename InLayout,
          typename OutDataType,
          typename WeiDataType,
          typename InDataType>
bool profile_grouped_conv_bwd_data_impl(int do_verification,
                                        int init_method,
                                        bool do_log,
                                        bool time_kernel,
                                        const ck::utils::conv::ConvParam& conv_param)
{
    using OutElementOp = ck::tensor_operation::element_wise::PassThrough;
    using WeiElementOp = ck::tensor_operation::element_wise::PassThrough;
    using InElementOp  = ck::tensor_operation::element_wise::PassThrough;

    const auto out_element_op = OutElementOp{};
    const auto wei_element_op = WeiElementOp{};
    const auto in_element_op  = InElementOp{};

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<OutLayout>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<WeiLayout>(conv_param);

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<InLayout>(conv_param);

    Tensor<OutDataType> out(out_g_n_k_wos_desc);
    Tensor<WeiDataType> wei(wei_g_k_c_xs_desc);
    Tensor<InDataType> in_host(in_g_n_c_wis_desc);
    Tensor<InDataType> in_device(in_g_n_c_wis_desc);

    std::cout << "out: " << out.mDesc << std::endl;
    std::cout << "wei: " << wei.mDesc << std::endl;
    std::cout << "in: " << in_host.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        wei.GenerateTensorValue(GeneratorTensor_2<WeiDataType>{-5, 5});
        break;
    case 2:
        out.GenerateTensorValue(GeneratorTensor_3<OutDataType>{0.0, 1.0});
        wei.GenerateTensorValue(GeneratorTensor_3<WeiDataType>{-0.5, 0.5});
        break;
    default:
        out.GenerateTensorValue(GeneratorTensor_1<OutDataType>{1});
        wei.GenerateTensorValue(GeneratorTensor_1<WeiDataType>{1});
    }

    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei.mDesc.GetElementSpaceSize());
    DeviceMem in_device_buf(sizeof(InDataType) * in_device.mDesc.GetElementSpaceSize());

    out_device_buf.ToDevice(out.mData.data());
    wei_device_buf.ToDevice(wei.mData.data());

    // reset input to zero
    in_device_buf.SetZero();

    if(do_verification)
    {
        auto ref_conv = ck::tensor_operation::host::ReferenceConvBwdData<NDimSpatial,
                                                                         InDataType,
                                                                         WeiDataType,
                                                                         OutDataType,
                                                                         InElementOp,
                                                                         WeiElementOp,
                                                                         OutElementOp>();

        auto ref_invoker = ref_conv.MakeInvoker();

        in_host.SetZero();

        auto ref_argument = ref_conv.MakeArgument(in_host,
                                                  wei,
                                                  out,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  out_element_op,
                                                  wei_element_op,
                                                  in_element_op);

        ref_invoker.Run(ref_argument);
    }

    std::string best_op_name;
    float best_avg_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device op instances
    bool pass = true;

    auto run_impl = [&](auto& op_ptr, auto& argument_ptr) {
        if(op_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            // re-init output to zero before profiling next kernel
            in_device_buf.SetZero();

            std::string op_name = op_ptr->GetTypeString();

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            float avg_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop      = conv_param.GetFlops();
            std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

            float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

            float gb_per_sec = num_btype / 1.E6 / avg_time;

            std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << op_name << std::endl;

            if(tflops > best_tflops)
            {
                best_op_name    = op_name;
                best_tflops     = tflops;
                best_avg_time   = avg_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                in_device_buf.FromDevice(in_device.mData.data());

                pass = pass & ck::utils::check_err(in_device, in_host);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "output : ", out.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "weight: ", wei.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "in_host  : ", in_host.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "in_device: ", in_device.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            std::cout << op_ptr->GetTypeString() << " does not support this problem" << std::endl;
        }
    };

    // do GEMM
    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD<NDimSpatial,
                                                                                     OutLayout,
                                                                                     WeiLayout,
                                                                                     ck::Tuple<>,
                                                                                     InLayout,
                                                                                     OutDataType,
                                                                                     WeiDataType,
                                                                                     ck::Tuple<>,
                                                                                     InDataType,
                                                                                     OutElementOp,
                                                                                     WeiElementOp,
                                                                                     InElementOp>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::array<ck::index_t, NDimSpatial + 3> out_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> out_strides{};
    std::array<ck::index_t, NDimSpatial + 3> wei_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> wei_strides{};
    std::array<ck::index_t, NDimSpatial + 3> in_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> in_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto copy = [](const auto& x, auto& y) { ck::ranges::copy(x, y.begin()); };

    copy(out_g_n_k_wos_desc.GetLengths(), out_lengths);
    copy(out_g_n_k_wos_desc.GetStrides(), out_strides);
    copy(wei_g_k_c_xs_desc.GetLengths(), wei_lengths);
    copy(wei_g_k_c_xs_desc.GetStrides(), wei_strides);
    copy(in_g_n_c_wis_desc.GetLengths(), in_lengths);
    copy(in_g_n_c_wis_desc.GetStrides(), in_strides);
    copy(conv_param.conv_filter_strides_, conv_filter_strides);
    copy(conv_param.conv_filter_dilations_, conv_filter_dilations);
    copy(conv_param.input_left_pads_, input_left_pads);
    copy(conv_param.input_right_pads_, input_right_pads);

    for(auto& op_ptr : op_ptrs)
    {
        auto argument_ptr =
            op_ptr->MakeArgumentPointer(static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                        static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                        {},
                                        static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                        out_lengths,
                                        out_strides,
                                        wei_lengths,
                                        wei_strides,
                                        {},
                                        {},
                                        in_lengths,
                                        in_strides,
                                        conv_filter_strides,
                                        conv_filter_dilations,
                                        input_left_pads,
                                        input_right_pads,
                                        out_element_op,
                                        wei_element_op,
                                        in_element_op);

        run_impl(op_ptr, argument_ptr);
    }

    std::cout << "Best configuration parameters:"
              << "\nname: " << best_op_name << "\navg_time: " << best_avg_time
              << "\ntflops: " << best_tflops << "\nGB/s: " << best_gb_per_sec << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
