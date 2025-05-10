// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

template <ck::index_t NDimSpatial>
bool run_grouped_conv_bwd_weight(const ExecutionConfig& config,
                                 const ck::utils::conv::ConvParam& conv_param)
{
    // Dl and WMMA ops don't support split_k > 1
    constexpr ck::index_t split_k = 1;

    const auto in_g_n_c_wis_desc =
        ck::utils::conv::make_input_host_tensor_descriptor_g_n_c_wis_packed<
            InputLayout<NDimSpatial>>(conv_param);

    const auto wei_g_k_c_xs_desc =
        ck::utils::conv::make_weight_host_tensor_descriptor_g_k_c_xs_packed<
            WeightLayout<NDimSpatial>>(conv_param);

    const auto out_g_n_k_wos_desc =
        ck::utils::conv::make_output_host_tensor_descriptor_g_n_k_wos_packed<
            OutputLayout<NDimSpatial>>(conv_param);

    Tensor<InDataType> in(in_g_n_c_wis_desc);
    Tensor<WeiDataType> wei_host_result(wei_g_k_c_xs_desc);
    Tensor<WeiDataType> wei_device_result(wei_g_k_c_xs_desc);
    Tensor<OutDataType> out(out_g_n_k_wos_desc);

    std::cout << "in: " << in.mDesc << std::endl;
    std::cout << "wei: " << wei_host_result.mDesc << std::endl;
    std::cout << "out: " << out.mDesc << std::endl;

    switch(config.init_method)
    {
    case 0: break;
    case 1:
        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});
        break;
    default:
        in.GenerateTensorValue(GeneratorTensor_3<InDataType>{0.0, 0.2});
        out.GenerateTensorValue(GeneratorTensor_3<OutDataType>{-0.1, 0.1});
    }

    DeviceMem in_device_buf(sizeof(InDataType) * in.mDesc.GetElementSpaceSize());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_device_result.mDesc.GetElementSpaceSize());
    DeviceMem out_device_buf(sizeof(OutDataType) * out.mDesc.GetElementSpaceSize());

    in_device_buf.ToDevice(in.mData.data());
    out_device_buf.ToDevice(out.mData.data());

    // init to 0
    wei_device_buf.SetZero();

    std::array<ck::index_t, NDimSpatial + 3> input_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> input_strides{};
    std::array<ck::index_t, NDimSpatial + 3> filter_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> weights_strides{};
    std::array<ck::index_t, NDimSpatial + 3> output_lengths{};
    std::array<ck::index_t, NDimSpatial + 3> output_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_strides{};
    std::array<ck::index_t, NDimSpatial> conv_filter_dilations{};
    std::array<ck::index_t, NDimSpatial> input_left_pads{};
    std::array<ck::index_t, NDimSpatial> input_right_pads{};

    auto range_copy = [](const auto& from, auto to) { std::copy(begin(from), end(from), to); };

    range_copy(in_g_n_c_wis_desc.GetLengths(), begin(input_lengths));
    range_copy(in_g_n_c_wis_desc.GetStrides(), begin(input_strides));
    range_copy(wei_g_k_c_xs_desc.GetLengths(), begin(filter_lengths));
    range_copy(wei_g_k_c_xs_desc.GetStrides(), begin(weights_strides));
    range_copy(out_g_n_k_wos_desc.GetLengths(), begin(output_lengths));
    range_copy(out_g_n_k_wos_desc.GetStrides(), begin(output_strides));
    range_copy(conv_param.conv_filter_strides_, begin(conv_filter_strides));
    range_copy(conv_param.conv_filter_dilations_, begin(conv_filter_dilations));
    range_copy(conv_param.input_left_pads_, begin(input_left_pads));
    range_copy(conv_param.input_right_pads_, begin(input_right_pads));

    // do GEMM
    auto conv     = DeviceConvBwdWeightInstance<NDimSpatial>{};
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                      static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                      static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                      input_lengths,
                                      input_strides,
                                      filter_lengths,
                                      weights_strides,
                                      output_lengths,
                                      output_strides,
                                      conv_filter_strides,
                                      conv_filter_dilations,
                                      input_left_pads,
                                      input_right_pads,
                                      InElementOp{},
                                      WeiElementOp{},
                                      OutElementOp{},
                                      split_k);

    if(!conv.IsSupportedArgument(argument))
    {
        std::cerr << "wrong! device_conv with the specified compilation parameters does "
                     "not support this Conv problem"
                  << std::endl;
        return true;
    }

    invoker.Run(argument, StreamConfig{nullptr, false});

    if(config.do_verification)
    {
        auto ref_conv     = HostConvBwdWeightInstance<NDimSpatial>{};
        auto ref_invoker  = ref_conv.MakeInvoker();
        auto ref_argument = ref_conv.MakeArgument(in,
                                                  wei_host_result,
                                                  out,
                                                  conv_param.conv_filter_strides_,
                                                  conv_param.conv_filter_dilations_,
                                                  conv_param.input_left_pads_,
                                                  conv_param.input_right_pads_,
                                                  InElementOp{},
                                                  WeiElementOp{},
                                                  OutElementOp{},
                                                  {},
                                                  {},
                                                  {});

        ref_invoker.Run(ref_argument);

        wei_device_buf.FromDevice(wei_device_result.mData.data());

        return ck::utils::check_err(wei_device_result.mData, wei_host_result.mData);
    }

    float avg_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    std::size_t flop      = conv_param.GetFlops();
    std::size_t num_btype = conv_param.GetByte<InDataType, WeiDataType, OutDataType>();

    float tflops = static_cast<float>(flop) / 1.E9 / avg_time;

    float gb_per_sec = num_btype / 1.E6 / avg_time;

    std::cerr << "Perf: " << avg_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl
              << "DeviceOp: " << conv.GetTypeString() << std::endl;

    return true;
}
