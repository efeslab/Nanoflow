// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_fwd.hpp"
#include "ck_tile/host.hpp"
#include "mask.hpp"
#include "utils.hpp"

#include <array>
#include <cstring>
#include <functional>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    using size_type = typename std::vector<T>::size_type;

    os << "[";
    for(size_type idx = 0; idx < v.size(); ++idx)
    {
        if(0 < idx)
        {
            os << ", ";
        }
        os << v[idx];
    }
    return os << "]";
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "weather do CPU validation or not")
        .insert("mode", "0", "kernel mode. 0:batch, 1:group")
        .insert("b", "2", "batch size")
        .insert("h", "8", "num of head, for q")
        .insert("h_k",
                "-1",
                "num of head, for k/v, -1 means equal to h\n"
                "if not equal to h, then this is GQA/MQA case")
        .insert(
            "s",
            "3328",
            "seqlen_q. if group-mode, means the average value of seqlen_q\n"
            "total_seqlen_q = seqlen_q * batch, and seqlen_q per batch may vary\n"
            "also with \"-s=s0,s1,s2...\" comma seperated int to set per batch seqlen(group-mode)")
        .insert("s_k", "-1", "seqlen_k, -1 means equal to s")
        .insert("s_kpad",
                "-1",
                "seqlen_k stride between 2 tokens, currently used in group-mode only\n"
                "for kv-cache case, each batch [1,s,h,d]/[1,h,s,d] can have a stride\n"
                "along seqlen, instead of packed. same as xformer kv_padding")
        .insert("d", "128", "head dim for q, k")
        .insert("d_v", "-1", "head dim for v, -1 means equal to d")
        .insert("scale_s",
                "0",
                "scale factor of S. 0 means equal to 1/sqrt(hdim).\n"
                "note when squant=1, this value will be modified by range_q/k")
        .insert("range_q", "16", "per-tensor quantization range of q. used if squant=1.")
        .insert("range_k", "16", "per-tensor quantization range of k. used if squant=1.")
        .insert("range_v", "16", "per-tensor quantization range of v. used if squant=1.")
        .insert("range_p", "1", "per-tensor quantization range of p [e^(s-m)]. used if squant=1.")
        .insert("range_o", "16", "per-tensor quantization range of o (p*v). used if squant=1.")
        .insert("squant",
                "auto",
                "if using static quantization fusion or not. auto: fp8 will default use squant, "
                "other will not\n"
                "0: no static quant(not implemented) 1: apply scale_p and scale_o with respect to "
                "P and O.\n"
                "calculate scale_s, scale_p, scale_o according to range_q, range_k, range_v, "
                "range_p, range_o")
        .insert("iperm",
                "1",
                "permute input\n"
                "if true, will be b*h*s*d, else b*s*h*d")
        .insert("operm", "1", "permute output")
        .insert("bias",
                "n",
                "n or 0, no bias\n"
                "e(lementwise) or 1, elementwise bias with 1*1*s*s. e:1, 1*h*s*s. e:2, b*h*s*s\n"
                "a(libi) or 2, alibi with 1*h. a:1, b*h")
        .insert("prec", "fp16", "data type. fp16/bf16/fp8/bf8")
        .insert("mask",
                "0",
                "0: no mask, 1: top-left(same as 't'), 2:bottom-right(same as 'b')\n"
                "'t', top-left causal mask, 'b', bottom-r causal mask\n"
                "'t:l,r', top-left sliding window attn(swa) with FA style left right size\n"
                "'b:l,r', bottom-r sliding window attn(swa) with FA style left right size\n"
                "'xt:window_size', xformer style masking from top-left, window_size negative is "
                "causal, positive is swa\n"
                "'xb:window_size', xformer style masking from bottom-r, window_size negative is "
                "causal, positive is swa\n"
                "'g:y,x', generic attention mask coordinate with y/x size (only debug purpose for "
                "now)")
        .insert("vlayout", "r", "r for row-major(seqlen*hdim), c for col-major(hdim*seqlen)")
        .insert("lse", "0", "0 not store lse, 1 store lse")
        .insert("kname", "0", "if set to 1 will print kernel name")
        .insert("init",
                "uf",
                "init method. ui, uniform random int, ni, normalized random int\n"
                "uf, uniform random float, nf, normalized random float, tf, trig float, uf:q, "
                "quantization")
        .insert("seed",
                "11939",
                "random seed used for initializing input tensors. 0 for "
                "non-deterministic seed")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "20", "number of iterations to benchmark the kernel");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// different threshold for different dtype
template <typename DataType>
auto get_elimit(std::string /*init_method*/)
{
    double rtol = 1e-3;
    double atol = 1e-3;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>(std::string init_method)
{
    if(init_method == "ui" || init_method == "ni")
    {
        double rtol = 1e-2;
        double atol = 1e-2;
        return ck_tile::make_tuple(rtol, atol);
    }
    else if(init_method == "nf")
    {
        double rtol = 1e-2;
        double atol = 1e-2;
        return ck_tile::make_tuple(rtol, atol);
    }
    else
    {
        double rtol = 3e-3;
        double atol = 3e-3;
        return ck_tile::make_tuple(rtol, atol);
    }
}

template <>
auto get_elimit<ck_tile::fp8_t>(std::string init_method)
{
    if(init_method == "ui" || init_method == "ni")
    {
        unsigned max_rounding_point_distance = 0;
        double atol                          = 2e-3;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
    else
    {
        unsigned max_rounding_point_distance = 1;
        double atol                          = 0.0625;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    std::string data_type    = arg_parser.get_str("prec");
    int do_validation        = arg_parser.get_int("v");
    auto mode                = static_cast<mode_enum>(arg_parser.get_uint32("mode"));
    ck_tile::index_t batch   = arg_parser.get_int("b");
    ck_tile::index_t nhead   = arg_parser.get_int("h");
    ck_tile::index_t nhead_k = arg_parser.get_int("h_k");
    if(nhead_k < 0)
        nhead_k = nhead;

    if(nhead % nhead_k != 0)
    {
        std::cerr << "nhead:" << nhead << " must be multiple of nhead_k:" << nhead_k << std::endl;
        return false;
    }

    auto [seqlen_qs, seqlen_ks, seqlen_kpads] = decode_seqlen(mode,
                                                              batch,
                                                              arg_parser.get_str("s"),
                                                              arg_parser.get_str("s_k"),
                                                              arg_parser.get_str("s_kpad"));

#if 0
    // clang-format off
    std::cout << "seqlen_qs:"; for(auto xx : seqlen_qs) { std::cout << xx << ","; } std::cout << std::endl;
    std::cout << "seqlen_ks:"; for(auto xx : seqlen_ks) { std::cout << xx << ","; } std::cout << std::endl;
    std::cout << "seqlen_kpads:"; for(auto xx : seqlen_kpads) { std::cout << xx << ","; } std::cout << std::endl;
    // clang-format on
#endif

    ck_tile::index_t hdim_q = arg_parser.get_int("d");
    ck_tile::index_t hdim_v = arg_parser.get_int("d_v");
    if(hdim_v < 0)
        hdim_v = hdim_q;

    bool i_perm = arg_parser.get_bool("iperm"); // if true, will be batch * nhead * seqlen * hdim
    bool o_perm = arg_parser.get_bool("operm"); // if false, will be batch * seqlen * nhead * hdim

    float scale_s = arg_parser.get_float("scale_s");
    if(scale_s == .0f)
        scale_s = 1.0 / ck_tile::sqrt(static_cast<float>(hdim_q)); // TODO: q ? v ?

    std::string squant_str = arg_parser.get_str("squant");
    bool squant            = [&]() {
        if(squant_str == "auto")
        {
            if(data_type == "fp8")
                return true;
            else
                return false;
        }
        else
            return atoi(squant_str.c_str()) != 0 ? true : false;
    }();

    float range_q = arg_parser.get_float("range_q");
    float range_k = arg_parser.get_float("range_k");
    float range_v = arg_parser.get_float("range_v");
    float range_p = arg_parser.get_float("range_p");
    float range_o = arg_parser.get_float("range_o");

    float dtype_max = ck_tile::type_convert<float>(ck_tile::numeric<DataType>::max());

    float scale_p = 1.f;
    float scale_o = 1.f;

    if(squant)
    {
        scale_s = scale_s * (range_q / dtype_max) * (range_k / dtype_max);
        scale_p = dtype_max / range_p;
        // scale_p = [max(fp8_t)/range_o] * [range_p/max(fp8_t)] * [range_v/max(fp8_t)]
        scale_o = range_p * range_v / range_o / dtype_max;
    }

    std::string vlayout = arg_parser.get_str("vlayout");
    bool lse            = arg_parser.get_bool("lse");

    bias_info bias = bias_info::decode(arg_parser.get_str("bias"));
    mask_info mask = mask_info::decode(
        arg_parser.get_str("mask"), seqlen_qs[0], seqlen_ks[0]); // TODO: we don't need x/y anymore

    std::string init_method      = arg_parser.get_str("init");
    std::optional<uint32_t> seed = arg_parser.get_uint32("seed");
    if(*seed == 0)
    {
        seed.reset();
    }

    int stream_warmup = arg_parser.get_int("warmup");
    int stream_repeat = arg_parser.get_int("repeat");
    bool kname        = arg_parser.get_bool("kname");

    ck_tile::stream_config stream_config{nullptr,
                                         true,
                                         /* log_level = */ (kname ? 1 : 0),
                                         stream_warmup,
                                         stream_repeat,
                                         arg_parser.get_str("timer") == std::string("gpu")};

    const auto seqstart_q_host              = to_seqstarts(seqlen_qs);
    const auto seqstart_k_host              = to_seqstarts(seqlen_ks);
    const auto seqstart_k_with_padding_host = to_seqstarts(seqlen_kpads);

    using TypeConfig = FmhaFwdTypeConfig<DataType>;

    using QDataType           = typename TypeConfig::QDataType;
    using KDataType           = typename TypeConfig::KDataType;
    using VDataType           = typename TypeConfig::VDataType;
    using BiasDataType        = typename TypeConfig::BiasDataType;
    using LSEDataType         = typename TypeConfig::LSEDataType;
    using SaccDataType        = typename TypeConfig::SaccDataType;
    using SMPLComputeDataType = typename TypeConfig::SMPLComputeDataType;
    using PDataType           = typename TypeConfig::PDataType;
    using OaccDataType        = typename TypeConfig::OaccDataType;
    using ODataType           = typename TypeConfig::ODataType;

    // accumulation numbers for performance evaluation
    std::size_t flop = 0, num_byte = 0;
    auto max_seqlen_q =
        std::numeric_limits<int32_t>::min(); // we will use max seqlen to decide grid size
    {
        for(ck_tile::index_t wb = 0; wb < batch; ++wb)
        {
            const int32_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
            const int32_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

            if(max_seqlen_q < real_seqlen_q)
            {
                max_seqlen_q = real_seqlen_q;
            }

            flop += nhead * (static_cast<std::size_t>(2) * real_seqlen_q * real_seqlen_k * hdim_q +
                             static_cast<std::size_t>(2) * real_seqlen_q * hdim_v * real_seqlen_k);

            num_byte += nhead * (sizeof(QDataType) * real_seqlen_q * hdim_q +
                                 sizeof(KDataType) * real_seqlen_k * hdim_q +
                                 sizeof(VDataType) * hdim_v * real_seqlen_k +
                                 sizeof(ODataType) * real_seqlen_q * hdim_v);
        }
    }

    auto get_lengths = [&](bool permute,
                           ck_tile::index_t b /*batch*/,
                           ck_tile::index_t h /*nhead*/,
                           ck_tile::index_t s /*seqlen*/,
                           ck_tile::index_t d /*hdim*/) {
        if(permute)
            return std::array<ck_tile::index_t, 4>{b, h, s, d};
        else
            return std::array<ck_tile::index_t, 4>{b, s, h, d};
    };

    bool is_v_rowmajor = vlayout == std::string("r");

    // host memory for storing all the tensor elements
    const ck_tile::index_t shape_batch = (mode == mode_enum::batch ? batch : 1);
    const ck_tile::index_t shape_seqlen_q =
        (mode == mode_enum::batch ? seqlen_qs[0] : seqstart_q_host.back());
    const ck_tile::index_t shape_seqlen_k =
        (mode == mode_enum::batch ? seqlen_ks[0]
                                  : (seqlen_kpads[0] < 0 ? seqstart_k_host.back()
                                                         : seqstart_k_with_padding_host.back()));

    ck_tile::HostTensor<QDataType> q_host(
        get_lengths(i_perm, shape_batch, nhead, shape_seqlen_q, hdim_q));
    ck_tile::HostTensor<KDataType> k_host(
        get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_q));
    ck_tile::HostTensor<VDataType> v_host(
        is_v_rowmajor ? get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_v)
                      : get_lengths(i_perm, shape_batch, nhead_k, hdim_v, shape_seqlen_k));

    ck_tile::HostTensor<BiasDataType> bias_host(
        bias.type == bias_enum::elementwise_bias
            ? get_lengths(i_perm, 1, 1, shape_seqlen_q, shape_seqlen_k)
            : std::array<ck_tile::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);

    ck_tile::HostTensor<SaccDataType> alibi_slope_host(
        bias.type == bias_enum::alibi
            ? (bias.rank_info == 0 ? std::array<ck_tile::index_t, 2>{1, nhead}
                                   : std::array<ck_tile::index_t, 2>{batch, nhead})
            : std::array<ck_tile::index_t, 2>{1, 1});

    // self define lse data layout as [shape_batch, nhead, shape_seqlen_q]
    ck_tile::HostTensor<LSEDataType> lse_host(
        lse ? std::array<ck_tile::index_t, 3>{shape_batch, nhead, shape_seqlen_q}
            : std::array<ck_tile::index_t, 3>{1, 1, 1} /* dummy shape for simplifying code */);

    ck_tile::HostTensor<ODataType> o_host(
        get_lengths(o_perm, shape_batch, nhead, shape_seqlen_q, hdim_v));

    if(init_method == "ui" || init_method == "0")
    {
        ck_tile::FillUniformDistributionIntegerValue<QDataType>{-3.f, 3.f, seed}(q_host);
        ck_tile::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f, seed}(k_host);
        ck_tile::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f, seed}(v_host);
        ck_tile::FillUniformDistributionIntegerValue<BiasDataType>{-3.f, 3.f, seed}(bias_host);
    }
    else if(init_method == "ni")
    {
        ck_tile::FillNormalDistributionIntegerValue<QDataType>{-3.f, 3.f, seed}(q_host);
        ck_tile::FillNormalDistributionIntegerValue<KDataType>{-3.f, 3.f, seed}(k_host);
        ck_tile::FillNormalDistributionIntegerValue<VDataType>{-3.f, 3.f, seed}(v_host);
        ck_tile::FillNormalDistributionIntegerValue<BiasDataType>{-3.f, 3.f, seed}(bias_host);
    }
    else if(init_method == "uf" || init_method == "1")
    {
        ck_tile::FillUniformDistribution<QDataType>{0.f, 1.f, seed}(q_host);
        ck_tile::FillUniformDistribution<KDataType>{0.f, 1.f, seed}(k_host);
        ck_tile::FillUniformDistribution<VDataType>{0.f, 1.f, seed}(v_host);
        ck_tile::FillUniformDistribution<BiasDataType>{0.f, 1.f, seed}(bias_host);
    }
    else if(init_method == "nf")
    {
        ck_tile::FillNormalDistribution<QDataType>{0.f, 3.f, seed}(q_host);
        ck_tile::FillNormalDistribution<KDataType>{0.f, 3.f, seed}(k_host);
        ck_tile::FillNormalDistribution<VDataType>{0.f, 3.f, seed}(v_host);
        ck_tile::FillNormalDistribution<BiasDataType>{0.f, 3.f, seed}(bias_host);
    }
    else if(init_method == "tf" || init_method == "2")
    {
        ck_tile::FillTrigValue<QDataType>{}(q_host);
        ck_tile::FillTrigValue<KDataType>{}(k_host);
        ck_tile::FillTrigValue<VDataType>{}(v_host);
        ck_tile::FillTrigValue<BiasDataType>{}(bias_host);
    }
    else if(init_method == "ufq" || init_method == "uf:q" ||
            init_method == "3") // suitable for fp8 quantization
    {
        ck_tile::FillUniformDistribution<QDataType>{-dtype_max, dtype_max, seed}(q_host);
        ck_tile::FillUniformDistribution<KDataType>{-dtype_max, dtype_max, seed}(k_host);
        ck_tile::FillUniformDistribution<VDataType>{-dtype_max, dtype_max, seed}(v_host);

        // bias_fp8 = qscale_bias * bias_fp32
        float qscale_bias = (dtype_max / range_q) * (dtype_max / range_k);
        // Assume bias is in [-1.f, 1.f] in original fp32
        ck_tile::FillUniformDistribution<BiasDataType>{-qscale_bias, qscale_bias, seed}(bias_host);
    }
    if(bias.type == bias_enum::alibi)
    {
        auto slopes = ck_tile::get_alibi_slopes<SaccDataType>(nhead);
        assert(slopes.size() == nhead);
        if(bias.rank_info == 0)
        {
            // alibi in 1*h
            std::copy(slopes.begin(), slopes.end(), alibi_slope_host.begin());
        }
        else
        {
            // alibi in b*h
            for(auto i_b = 0; i_b < batch; i_b++)
            {
                std::copy(slopes.begin(), slopes.end(), alibi_slope_host.begin() + i_b * nhead);
            }
        }
    }

    ck_tile::DeviceMem q_buf(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem bias_buf(bias_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lse_buf(lse_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(o_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqlen_k_buf(seqlen_kpads[0] < 0 ? 0 : seqlen_ks.size() * sizeof(int32_t));
    ck_tile::DeviceMem alibi_slope_buf(alibi_slope_host.get_element_space_size_in_bytes());

    q_buf.ToDevice(q_host.data());
    k_buf.ToDevice(k_host.data());
    v_buf.ToDevice(v_host.data());
    bias_buf.ToDevice(bias_host.data());
    seqstart_q.ToDevice(seqstart_q_host.data());
    seqstart_k.ToDevice(seqlen_kpads[0] < 0 ? seqstart_k_host.data()
                                            : seqstart_k_with_padding_host.data());
    seqlen_k_buf.ToDevice(seqlen_kpads[0] < 0 ? nullptr : seqlen_ks.data());
    alibi_slope_buf.ToDevice(alibi_slope_host.data());

    // clang-format off
    auto layout_str = [&](bool permute){
        if (permute) return std::string("bhsd");
        else return std::string("bshd");
    };
    auto io_layout = [&](bool iperm_, bool operm_) {
        if (iperm_ == operm_) return layout_str(iperm_);
        else return layout_str(iperm_) + std::string("-") + layout_str(operm_);
    };
    // clang-format on
    const std::string prec = arg_parser.get_str("prec");

    std::cout << "[" << prec << "|" << mode << "|" << io_layout(i_perm, o_perm) << "] b:" << batch
              << ", h:" << nhead << "/" << nhead_k << ", s:" << seqlen_qs[0] << "/" << seqlen_ks[0]
              << (seqlen_kpads[0] < 0 ? ""
                                      : (std::string("(") + std::to_string(seqlen_kpads[0]) + ")"))
              << ", d:" << hdim_q << "/" << hdim_v << ", scale_s:" << scale_s << ", bias:" << bias
              << ", lse:" << lse << ", squant:" << squant << ", mask:" << mask << ", v:" << vlayout
              << std::flush;

    auto fmha_traits = fmha_fwd_traits{hdim_q,
                                       hdim_v,
                                       data_type,
                                       mode == mode_enum::group,
                                       is_v_rowmajor,
                                       mask.type,
                                       bias.type,
                                       lse,
                                       squant};

    auto p_compute_element_func = [&]() {
        if constexpr(std::is_same_v<DataType, ck_tile::fp8_t>)
            return ck_tile::scales{scale_p};
        else
            return ck_tile::identity{};
    }();

    auto oacc_element_func = [&]() {
        if constexpr(std::is_same_v<DataType, ck_tile::fp8_t>)
            return ck_tile::composes(ck_tile::saturates<ck_tile::fp8_t>{},
                                     ck_tile::scales{scale_o});
        else
            return ck_tile::identity{};
    }();

    auto fmha_args = [&, k_paddings_ = seqlen_kpads]() {
        assert(nhead % nhead_k == 0);
        /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
        ///       seqlen_k] in this example, hence both the 'batch_stride_bias' &
        ///       'nhead_stride_bias' are 0.
        // setup stride_* arguments
        const ck_tile::index_t stride_q = (i_perm ? hdim_q : nhead * hdim_q);
        const ck_tile::index_t stride_k = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_v = [&]() {
            if(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return i_perm ? shape_seqlen_k : nhead_k * shape_seqlen_k;
        }();
        const ck_tile::index_t stride_bias = (i_perm ? shape_seqlen_k : 1 * shape_seqlen_k);
        const ck_tile::index_t stride_o    = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k = (i_perm ? shape_seqlen_k * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_v = [&]() {
            if(is_v_rowmajor)
                return i_perm ? shape_seqlen_k * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k;
        }();
        const ck_tile::index_t nhead_stride_bias =
            (i_perm ? 0 * shape_seqlen_q * shape_seqlen_k : 0 * shape_seqlen_k);
        const ck_tile::index_t nhead_stride_lse = (shape_seqlen_q * 1);
        const ck_tile::index_t nhead_stride_o   = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q    = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k    = (nhead_k * shape_seqlen_k * hdim_q);
        const ck_tile::index_t batch_stride_v    = (nhead_k * hdim_v * shape_seqlen_k);
        const ck_tile::index_t batch_stride_bias = (0 * nhead * shape_seqlen_q * shape_seqlen_k);
        const ck_tile::index_t batch_stride_lse  = (nhead * shape_seqlen_q * 1);
        const ck_tile::index_t batch_stride_o    = (nhead * shape_seqlen_q * hdim_v);

        return fmha_fwd_args{q_buf.GetDeviceBuffer(),
                             k_buf.GetDeviceBuffer(),
                             v_buf.GetDeviceBuffer(),
                             bias.type == bias_enum::alibi ? alibi_slope_buf.GetDeviceBuffer()
                                                           : bias_buf.GetDeviceBuffer(),
                             lse_buf.GetDeviceBuffer(),
                             o_buf.GetDeviceBuffer(),
                             seqstart_q.GetDeviceBuffer(),
                             seqstart_k.GetDeviceBuffer(),
                             k_paddings_[0] < 0 ? nullptr : seqlen_k_buf.GetDeviceBuffer(),
                             shape_seqlen_q,
                             shape_seqlen_k,
                             batch,
                             max_seqlen_q,
                             hdim_q,
                             hdim_v,
                             nhead,
                             nhead_k,
                             scale_s,
                             scale_p,
                             scale_o,
                             stride_q,
                             stride_k,
                             stride_v,
                             bias.type == bias_enum::alibi ? (bias.rank_info == 0 ? 0 : nhead)
                                                           : stride_bias,
                             stride_o,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_lse,
                             nhead_stride_o,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_lse,
                             batch_stride_o,
                             mask.left,
                             mask.right,
                             static_cast<ck_tile::index_t>(mask.type)};
    }();

    float ave_time = fmha_fwd(fmha_traits, fmha_args, stream_config);

    if(ave_time < 0)
    {
        std::cout << ", not supported yet" << std::flush << std::endl;
        return false;
    }

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << std::fixed << ", " << std::setprecision(3) << ave_time << " ms, "
              << std::setprecision(2) << tflops << " TFlops, " << std::setprecision(2) << gb_per_sec
              << " GB/s" << std::flush;

    if(!do_validation)
    {
        std::cout << std::flush << std::endl;
        return true;
    }

    o_buf.FromDevice(o_host.data());
    lse_buf.FromDevice(lse_host.data());

    bool pass = true;

    for(ck_tile::index_t wb = 0; wb < batch; ++wb)
    {
        const ck_tile::index_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
        const ck_tile::index_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

        // adjust matrix index according to the mode
        const ck_tile::index_t b            = (mode == mode_enum::batch ? wb : 0);
        const ck_tile::index_t query_offset = (mode == mode_enum::batch ? 0 : seqstart_q_host[wb]);
        const ck_tile::index_t key_offset =
            (mode == mode_enum::batch
                 ? 0
                 : (seqlen_kpads[0] < 0 ? seqstart_k_host[wb] : seqstart_k_with_padding_host[wb]));

        const auto v_host_ref_lengths =
            std::array<ck_tile::index_t, 3>{nhead, hdim_v, real_seqlen_k};
        const auto v_host_ref_strides =
            is_v_rowmajor
                ? std::array<ck_tile::index_t, 3>{hdim_v * real_seqlen_k, 1, hdim_v}
                : std::array<ck_tile::index_t, 3>{hdim_v * real_seqlen_k, real_seqlen_k, 1};

        ck_tile::HostTensor<QDataType> q_host_ref({nhead, real_seqlen_q, hdim_q});
        ck_tile::HostTensor<KDataType> k_host_ref({nhead, real_seqlen_k, hdim_q});
        ck_tile::HostTensor<VDataType> v_host_ref(v_host_ref_lengths, v_host_ref_strides);
        ck_tile::HostTensor<ODataType> o_host_ref({nhead, real_seqlen_q, hdim_v});

        ck_tile::HostTensor<SMPLComputeDataType> s_host_ref({nhead, real_seqlen_q, real_seqlen_k});
        ck_tile::HostTensor<PDataType> p_host_ref({nhead, real_seqlen_q, real_seqlen_k});
        ck_tile::HostTensor<SMPLComputeDataType> lse_host_ref({nhead, real_seqlen_q});

        ck_tile::index_t nr = nhead / nhead_k;

        // clang-format off
        // permute
        if(i_perm) q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b, i[0], i[1] + query_offset, i[2]); });
        else       q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b, i[1] + query_offset, i[0], i[2]); });

        if(i_perm) k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(b, i[0] / nr, i[1] + key_offset, i[2]); });
        else       k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(b, i[1] + key_offset, i[0] / nr, i[2]); });

        if (is_v_rowmajor) {
            //                                                             v_host_ref: [nhead, hdim, seq], v_host: [b, h_k, s, d]
            if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[0] / nr, i[2] + key_offset, i[1]); });
            //                                                             v_host_ref: [nhead, hdim, seq], v_host: [b, s, h_k, d]
            else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[2] + key_offset, i[0] / nr, i[1]); });
        }
        else {
            if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[0] / nr, i[1], i[2] + key_offset); });
            else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[1], i[0] / nr, i[2] + key_offset); });
        }
        // clang-format on

        // reference
        ck_tile::reference_batched_gemm<QDataType, KDataType, SaccDataType, SMPLComputeDataType>(
            q_host_ref,
            k_host_ref,
            s_host_ref,
            ck_tile::identity{},
            ck_tile::identity{},
            ck_tile::scales(scale_s));

        if(bias.type == bias_enum::elementwise_bias)
        {
            // elementwise bias
            ck_tile::HostTensor<BiasDataType> bias_host_ref({1, real_seqlen_q, real_seqlen_k});
            // clang-format off
            if(i_perm)
                bias_host_ref.ForEach([&](auto& self, auto i) { self(i) = bias_host(0, 0, i[1] + query_offset, i[2] + key_offset); });
            else
                bias_host_ref.ForEach([&](auto& self, auto i) { self(i) = bias_host(0, i[1] + query_offset, 0, i[2] + key_offset); });
            // clang-format on

            // broadcast from [1, real_seqlen_q, real_seqlen_k] to [nhead, real_seqlen_q,
            // real_seqlen_k]
            ck_tile::reference_batched_elementwise<SMPLComputeDataType,
                                                   BiasDataType,
                                                   SMPLComputeDataType,
                                                   SMPLComputeDataType>(
                s_host_ref, bias_host_ref, s_host_ref);
        }
        else if(bias.type == bias_enum::alibi)
        {
            // alibi construct elementwise bias to verify
            auto alibi_host = [&]() {
                if(mask.type != mask_enum::no_mask)
                {
                    return ck_tile::make_alibi_from_lr_mask<SaccDataType, true>(
                        0,
                        mask.left,
                        mask.right,
                        real_seqlen_q,
                        real_seqlen_k,
                        static_cast<ck_tile::GenericAttentionMaskEnum>(mask.type));
                }
                else
                {
                    return ck_tile::Alibi<SaccDataType, true>{
                        0, real_seqlen_q, real_seqlen_k, ck_tile::AlibiMode::FROM_BOTTOM_RIGHT};
                }
            }();

            ck_tile::HostTensor<SaccDataType> alibi_bias_host_ref(
                {nhead, real_seqlen_q, real_seqlen_k});
            auto i_b_slope = bias.rank_info == 0 ? 0 : wb;
            for(auto i_h = 0; i_h < nhead; i_h++)
            {
                SaccDataType current_slope = alibi_slope_host(i_b_slope, i_h);
                alibi_host.slope = alibi_host.mode == ck_tile::AlibiMode::VERTICAL ? current_slope
                                                                                   : -current_slope;
                for(auto i_r = 0; i_r < real_seqlen_q; i_r++)
                {
                    for(auto i_c = 0; i_c < real_seqlen_k; i_c++)
                    {
                        SaccDataType pixel = 0;
                        alibi_host.update(pixel, i_r, i_c);
                        alibi_bias_host_ref(i_h, i_r, i_c) = pixel;
                    }
                }
            }
            // [nhead, real_seqlen_q, real_seqlen_k]
            ck_tile::reference_batched_elementwise<SMPLComputeDataType,
                                                   SaccDataType,
                                                   SMPLComputeDataType,
                                                   SMPLComputeDataType>(
                s_host_ref, alibi_bias_host_ref, s_host_ref);
        }

        if(mask.type == mask_enum::no_mask)
        {
            ck_tile::reference_batched_masking<SaccDataType>(
                s_host_ref, FmhaMasks::NoMask{real_seqlen_q, real_seqlen_k});
        }
        else if(mask.type == mask_enum::window_generic)
        {
            ck_tile::reference_batched_masking<SaccDataType>(
                s_host_ref,
                ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::GenericMask>(
                    mask.left, mask.right, real_seqlen_q, real_seqlen_k));
        }
        else
        {
            // if left window size is negative, means causal
            // else means generic (for current batch)
            if(mask.left < 0)
                ck_tile::reference_batched_masking<SaccDataType>(
                    s_host_ref,
                    ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::CausalMask>(
                        mask.left,
                        mask.right,
                        real_seqlen_q,
                        real_seqlen_k,
                        mask.type == mask_enum::mask_top_left));
            else
                ck_tile::reference_batched_masking<SaccDataType>(
                    s_host_ref,
                    ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::GenericMask>(
                        mask.left,
                        mask.right,
                        real_seqlen_q,
                        real_seqlen_k,
                        mask.type == mask_enum::mask_top_left));
        }
        if(lse)
        {
            ck_tile::reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(
                s_host_ref, p_host_ref, p_compute_element_func, lse_host_ref);
        }
        else
        {
            ck_tile::reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(
                s_host_ref, p_host_ref, p_compute_element_func);
        }

        ck_tile::reference_batched_gemm<PDataType, VDataType, OaccDataType, ODataType>(
            p_host_ref,
            v_host_ref,
            o_host_ref,
            ck_tile::identity{},
            ck_tile::identity{},
            oacc_element_func);

        ck_tile::HostTensor<ODataType> o_host_result({nhead, real_seqlen_q, hdim_v});
        // clang-format off
        // permute
        if(o_perm) o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b, idx[0], idx[1] + query_offset, idx[2]); });
        else       o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b, idx[1] + query_offset, idx[0], idx[2]); });
        // clang-format on

        auto [rtol, atol] = get_elimit<DataType>(init_method);
        bool cur_pass     = ck_tile::check_err(
            o_host_result, o_host_ref, std::string("OUT Error: Incorrect results!"), rtol, atol);
        pass &= cur_pass;
        if(!cur_pass)
        {
            std::cerr << "OUT mismatch found at batch: " << wb << std::endl
                      << "\tseqlen_q: " << real_seqlen_q << std::endl
                      << "\tseqlen_k: " << real_seqlen_k << std::endl
                      << "\tseqstart_q: " << seqstart_q_host << std::endl
                      << "\tseqstart_k: " << seqstart_k_host << std::endl;

            break;
        }

        if(lse)
        {
            ck_tile::HostTensor<SMPLComputeDataType> lse_host_result({nhead, real_seqlen_q});
            lse_host_result.ForEach([&](auto& self, auto idx) {
                self(idx) = lse_host(b, idx[0], idx[1] + query_offset);
            });

            bool lse_pass = ck_tile::check_err(lse_host_result,
                                               lse_host_ref,
                                               "LSE Error: Incorrect results!",
                                               rtol,
                                               atol,
                                               /* allow_infinity_ref = */ true);

            pass &= lse_pass;
            if(!cur_pass)
            {
                std::cerr << "LSE mismatch found at batch: " << wb << std::endl
                          << "\tseqlen_q: " << real_seqlen_q << std::endl
                          << "\tseqlen_k: " << real_seqlen_k << std::endl
                          << "\tseqstart_q: " << seqstart_q_host << std::endl
                          << "\tseqstart_k: " << seqstart_k_host << std::endl;

                break;
            }
        }
    }

    std::cout << ", valid:" << (pass ? "y" : "n") << std::flush << std::endl;

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");
    if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "bf16")
    {
        return run<ck_tile::bf16_t>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "fp8")
    {
        return run<ck_tile::fp8_t>(arg_parser) ? 0 : -2;
    }

    return -3;
}
