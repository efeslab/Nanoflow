// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_bwd.hpp"
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
        .insert("s",
                "3328",
                "seqlen_q. if group-mode, means the average value of seqlen_q\n"
                "total_seqlen_q = seqlen_q * batch, and seqlen_q per batch may vary")
        .insert("s_k", "-1", "seqlen_k, -1 means equal to s")
        .insert("d", "128", "head dim for q, k")
        .insert("d_v", "-1", "head dim for v, -1 means equal to d")
        .insert("scale", "0", "scale factor. 0 means equal to 1/sqrt(hdim)")
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
        .insert("dbias", "0", "output bias gradient or not")
        .insert("prec", "fp16", "data type. fp16 or bf16")
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
        .insert("kname", "0", "if set to 1 will print kernel name")
        .insert("init", "1", "init method. 0:random int, 1:random float, 2:trig float")
        .insert("seed",
                "11939",
                "random seed used for initializing input tensors. 0 for "
                "non-deterministic seed")
        .insert("p_drop", "0", "0~1 probability of dropout")
        .insert("drop_seed", "1", "seed for random number generator")
        .insert("drop_offset", "0", "offset for random number generator")
        .insert("drop_prefs",
                "0",
                "seed and offset values are present on GPU; 0 - host, 1 - device/GPU")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "20", "number of iterations to benchmark the kernel")
        .insert("deterministic",
                "0",
                "if set to 1 will use multi-buffer reduction strategy for dq, atomic opeartion "
                "will not be used");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// different threshold for different dtype
template <typename DataTypeConfig>
auto get_elimit(ck_tile::index_t /*hdim_q*/, ck_tile::index_t /*hdim_v*/)
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<FmhaBwdBf16>(ck_tile::index_t hdim_q, ck_tile::index_t hdim_v)
{
    double rtol = 1e-2;
    double atol = 1e-2;
    if(hdim_q > 128 && hdim_v > 128) // 3.2 for RTZ/1.5 for RTN
    {
        rtol = 3.2e-2;
        atol = 3.2e-2;
    }
    return ck_tile::make_tuple(rtol, atol);
}

template <typename DataTypeConfig>
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

    ck_tile::index_t seqlen_q = arg_parser.get_int("s");
    ck_tile::index_t seqlen_k = arg_parser.get_int("s_k");
    if(seqlen_k < 0)
        seqlen_k = seqlen_q;
    ck_tile::index_t hdim_q = arg_parser.get_int("d");
    ck_tile::index_t hdim_v = arg_parser.get_int("d_v");
    if(hdim_v < 0)
        hdim_v = hdim_q;

    bool i_perm = arg_parser.get_bool("iperm"); // if true, will be batch * nhead * seqlen * hdim
    bool o_perm = arg_parser.get_bool("operm"); // if false, will be batch * seqlen * nhead * hdim

    float scale = arg_parser.get_float("scale");
    if(scale == .0f)
        scale = 1.0 / ck_tile::sqrt(static_cast<float>(hdim_q));

    bias_info bias       = bias_info::decode(arg_parser.get_str("bias"));
    bool use_dbias       = arg_parser.get_bool("dbias");
    float p_drop         = arg_parser.get_float("p_drop");
    uint64_t drop_seed   = arg_parser.get_uint64("drop_seed");
    uint64_t drop_offset = arg_parser.get_uint64("drop_offset");
    bool drop_prefs      = arg_parser.get_bool("drop_prefs");

    if(use_dbias && bias.type != bias_enum::elementwise_bias)
    {
        std::cerr << "dbias only exists when bias type is elementwise" << std::endl;
        return false;
    }

    if(p_drop < 0.0f || p_drop > 1.0f)
    {
        std::cerr << "The value of p_drop should be 0~1" << std::endl;
        return false;
    }
    float p_undrop = 1.0 - p_drop;
    uint8_t p_undrop_in_uint8_t =
        uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
    float rp_undrop = 1.0 / p_undrop;

    bool s_randval = false;
    if(p_drop > 0.0f && do_validation)
    {
        s_randval = true;
    }

    mask_info mask = mask_info::decode(arg_parser.get_str("mask"), seqlen_q, seqlen_k);

    int init_method              = arg_parser.get_int("init");
    std::optional<uint32_t> seed = arg_parser.get_uint32("seed");
    if(*seed == 0)
    {
        seed.reset();
    }

    int stream_warmup  = arg_parser.get_int("warmup");
    int stream_repeat  = arg_parser.get_int("repeat");
    bool kname         = arg_parser.get_bool("kname");
    bool deterministic = arg_parser.get_bool("deterministic");

    ck_tile::stream_config stream_config{nullptr,
                                         true,
                                         /* log_level = */ (kname ? 1 : 0),
                                         stream_warmup,
                                         stream_repeat,
                                         arg_parser.get_str("timer") == std::string("gpu")};

    const auto seqstart_q_host = generate_seqstarts(mode, batch, seqlen_q);
    const auto seqstart_k_host = generate_seqstarts(mode, batch, seqlen_k);

    using TypeConfig = FmhaBwdTypeConfig<DataTypeConfig>;

    using QDataType             = typename TypeConfig::QDataType;
    using KDataType             = typename TypeConfig::KDataType;
    using VDataType             = typename TypeConfig::VDataType;
    using GemmDataType          = typename TypeConfig::GemmDataType;
    using BiasDataType          = typename TypeConfig::BiasDataType;
    using LSEDataType           = typename TypeConfig::LSEDataType;
    using AccDataType           = typename TypeConfig::AccDataType;
    using DDataType             = typename TypeConfig::DDataType;
    using RandValOutputDataType = typename TypeConfig::RandValOutputDataType;
    using ODataType             = typename TypeConfig::ODataType;
    using OGradDataType         = typename TypeConfig::OGradDataType;
    using QGradDataType         = typename TypeConfig::QGradDataType;
    using KGradDataType         = typename TypeConfig::KGradDataType;
    using VGradDataType         = typename TypeConfig::VGradDataType;
    using BiasGradDataType      = typename TypeConfig::BiasGradDataType;

    // accumulation numbers for performance evaluation
    std::size_t flop = 0, num_byte = 0;
    auto max_seqlen_q =
        std::numeric_limits<int32_t>::min(); // we will use max seqlen to decide grid size
    auto max_seqlen_k =
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

            if(max_seqlen_k < real_seqlen_k)
            {
                max_seqlen_k = real_seqlen_k;
            }

            flop += nhead * (static_cast<std::size_t>(3) * static_cast<std::size_t>(2) *
                                 real_seqlen_q * real_seqlen_k * hdim_q + // Q@K/dS^T@Q^T/dS@K^T
                             static_cast<std::size_t>(2) * static_cast<std::size_t>(2) *
                                 real_seqlen_q * real_seqlen_k * hdim_v); // dO@V/P^T@dO^T

            num_byte += nhead * (sizeof(QDataType) * real_seqlen_q * hdim_q +
                                 sizeof(KDataType) * real_seqlen_k * hdim_q +
                                 sizeof(VDataType) * real_seqlen_k * hdim_v +
                                 sizeof(ODataType) * real_seqlen_q * hdim_v +
                                 sizeof(OGradDataType) * real_seqlen_q * hdim_v +
                                 sizeof(QGradDataType) * real_seqlen_q * hdim_q +
                                 sizeof(KGradDataType) * real_seqlen_k * hdim_q +
                                 sizeof(VGradDataType) * real_seqlen_k * hdim_v +
                                 sizeof(LSEDataType) * real_seqlen_q);
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

    // host memory for storing all the tensor elements
    const ck_tile::index_t shape_batch = (mode == mode_enum::batch ? batch : 1);
    const ck_tile::index_t shape_seqlen_q =
        (mode == mode_enum::batch ? seqlen_q : seqstart_q_host.back());
    const ck_tile::index_t shape_seqlen_k =
        (mode == mode_enum::batch ? seqlen_k : seqstart_k_host.back());
    const ck_tile::index_t kN0 = (hdim_q <= 128) ? 128 : 64;
    const ck_tile::index_t nsplits =
        deterministic ? ck_tile::integer_divide_ceil(max_seqlen_k, kN0) : 1;

    ck_tile::HostTensor<QDataType> q_host(
        get_lengths(i_perm, shape_batch, nhead, shape_seqlen_q, hdim_q));
    ck_tile::HostTensor<KDataType> k_host(
        get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_q));
    ck_tile::HostTensor<VDataType> v_host(
        get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_v));
    ck_tile::HostTensor<BiasDataType> bias_host(
        bias.type == bias_enum::elementwise_bias
            ? get_lengths(i_perm, 1, 1, shape_seqlen_q, max_seqlen_k)
            : std::array<ck_tile::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);
    ck_tile::HostTensor<AccDataType> alibi_slope_host(
        bias.type == bias_enum::alibi
            ? (bias.rank_info == 0 ? std::array<ck_tile::index_t, 2>{1, nhead}
                                   : std::array<ck_tile::index_t, 2>{batch, nhead})
            : std::array<ck_tile::index_t, 2>{1, 1});
    ck_tile::HostTensor<ODataType> o_host(
        get_lengths(o_perm, shape_batch, nhead, shape_seqlen_q, hdim_v));
    ck_tile::HostTensor<LSEDataType> lse_host(
        std::array<ck_tile::index_t, 3>{shape_batch, nhead, shape_seqlen_q});
    ck_tile::HostTensor<DDataType> d_host(
        std::array<ck_tile::index_t, 3>{shape_batch, nhead, shape_seqlen_q});
    ck_tile::HostTensor<RandValOutputDataType> randval_host(
        p_drop > 0 ? get_lengths(true, shape_batch, nhead, shape_seqlen_q, max_seqlen_k)
                   : std::array<ck_tile::index_t, 4>{1, 1, 1, 1});
    ck_tile::HostTensor<QGradDataType> dq_host(
        get_lengths(i_perm, shape_batch, nhead, shape_seqlen_q, hdim_q));
    ck_tile::HostTensor<KGradDataType> dk_host(
        get_lengths(i_perm, shape_batch, nhead, shape_seqlen_k, hdim_q));
    ck_tile::HostTensor<VGradDataType> dv_host(
        get_lengths(i_perm, shape_batch, nhead, shape_seqlen_k, hdim_v));
    ck_tile::HostTensor<OGradDataType> do_host(
        get_lengths(o_perm, shape_batch, nhead, shape_seqlen_q, hdim_v));
    ck_tile::HostTensor<BiasGradDataType> dbias_host(
        use_dbias
            ? get_lengths(i_perm, shape_batch, nhead, shape_seqlen_q, max_seqlen_k)
            : std::array<ck_tile::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);
    ck_tile::HostTensor<AccDataType> dq_acc_host(
        i_perm
            ? std::array<ck_tile::index_t, 5>{nsplits, shape_batch, nhead, shape_seqlen_q, hdim_q}
            : std::array<ck_tile::index_t, 5>{nsplits, shape_batch, shape_seqlen_q, nhead, hdim_q});

    if(init_method == 0)
    {
        ck_tile::FillUniformDistributionIntegerValue<QDataType>{-2.f, 2.f, seed}(q_host);
        ck_tile::FillUniformDistributionIntegerValue<KDataType>{-2.f, 2.f, seed}(k_host);
        ck_tile::FillUniformDistributionIntegerValue<VDataType>{-2.f, 2.f, seed}(v_host);
        ck_tile::FillUniformDistributionIntegerValue<BiasDataType>{-2.f, 2.f, seed}(bias_host);
        ck_tile::FillUniformDistributionIntegerValue<OGradDataType>{-2.f, 2.f, seed}(do_host);
    }
    else if(init_method == 1)
    {
        ck_tile::FillUniformDistribution<QDataType>{0.f, 1.f, seed}(q_host);
        ck_tile::FillUniformDistribution<KDataType>{0.f, 1.f, seed}(k_host);
        ck_tile::FillUniformDistribution<VDataType>{0.f, 1.f, seed}(v_host);
        ck_tile::FillUniformDistribution<BiasDataType>{0.f, 1.f, seed}(bias_host);
        ck_tile::FillUniformDistribution<OGradDataType>{0.f, 1.f, seed}(do_host);
    }
    else if(init_method == 2)
    {
        ck_tile::FillTrigValue<QDataType>{}(q_host);
        ck_tile::FillTrigValue<KDataType>{}(k_host);
        ck_tile::FillTrigValue<VDataType>{}(v_host);
        ck_tile::FillTrigValue<BiasDataType>{}(bias_host);
        ck_tile::FillTrigValue<OGradDataType>{}(do_host);
    }
    if(bias.type == bias_enum::alibi)
    {
        auto slopes = ck_tile::get_alibi_slopes<AccDataType>(nhead);
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
    ck_tile::DeviceMem o_buf(o_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lse_buf(lse_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d_buf(d_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem randval_buf(randval_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dq_buf(dq_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dk_buf(dk_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dv_buf(dv_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem do_buf(do_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dbias_buf(dbias_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem drop_seed_buf(drop_prefs ? sizeof(uint64_t) : 0);
    ck_tile::DeviceMem drop_offset_buf(drop_prefs ? sizeof(uint64_t) : 0);
    ck_tile::DeviceMem alibi_slope_buf(alibi_slope_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem dq_acc_buf(dq_acc_host.get_element_space_size_in_bytes());

    q_buf.ToDevice(q_host.data());
    k_buf.ToDevice(k_host.data());
    v_buf.ToDevice(v_host.data());
    bias_buf.ToDevice(bias_host.data());
    do_buf.ToDevice(do_host.data());
    seqstart_q.ToDevice(seqstart_q_host.data());
    seqstart_k.ToDevice(seqstart_k_host.data());
    drop_seed_buf.ToDevice(drop_prefs ? &drop_seed : nullptr);
    drop_offset_buf.ToDevice(drop_prefs ? &drop_offset : nullptr);
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
              << ", h:" << nhead << "/" << nhead_k << ", s:" << seqlen_q << "/" << seqlen_k
              << ", d:" << hdim_q << "/" << hdim_v << ", scale:" << scale << ", bias:" << bias
              << ", dbias:" << use_dbias << ", p_drop:" << p_drop << ", s_randval:" << s_randval
              << ", deterministic:" << deterministic << ", mask:" << mask << std::flush;

    std::size_t workspace_size =
        dq_acc_host.get_element_space_size_in_bytes() * sizeof(AccDataType) / (1024 * 1024);

    if(deterministic == 1)
    {
        std::cout << "\nDeterministic mode ON: " << workspace_size
                  << " MByte memory workspace allocated" << std::endl;
    }

    auto fmha_traits = fmha_bwd_traits{hdim_q,
                                       hdim_v,
                                       data_type,
                                       mode == mode_enum::group,
                                       mask.type,
                                       bias.type,
                                       use_dbias,
                                       p_drop > 0.0f,
                                       s_randval,
                                       deterministic};
    auto fmha_args   = [&]() {
        assert(nhead % nhead_k == 0);
        /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
        ///       seqlen_k] in this example, hence both the 'batch_stride_bias' &
        ///       'nhead_stride_bias' are 0.
        // setup stride_* arguments
        const ck_tile::index_t stride_q       = (i_perm ? hdim_q : nhead * hdim_q);
        const ck_tile::index_t stride_k       = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_v       = (i_perm ? hdim_v : nhead_k * hdim_v);
        const ck_tile::index_t stride_bias    = (max_seqlen_k);
        const ck_tile::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
        const ck_tile::index_t stride_randval = (max_seqlen_k);
        const ck_tile::index_t stride_do      = (o_perm ? hdim_v : nhead * hdim_v);
        const ck_tile::index_t stride_dk      = (i_perm ? hdim_q : nhead * hdim_q);
        const ck_tile::index_t stride_dv      = (i_perm ? hdim_v : nhead * hdim_v);
        const ck_tile::index_t stride_dbias   = (i_perm ? max_seqlen_k : nhead * max_seqlen_k);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q       = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k       = (i_perm ? shape_seqlen_k * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_v       = (i_perm ? shape_seqlen_k * hdim_v : hdim_v);
        const ck_tile::index_t nhead_stride_bias    = 0;
        const ck_tile::index_t nhead_stride_o       = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        const ck_tile::index_t nhead_stride_randval = (shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t nhead_stride_do      = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        const ck_tile::index_t nhead_stride_lsed    = shape_seqlen_q;
        const ck_tile::index_t nhead_stride_dbias =
            (i_perm ? shape_seqlen_q * max_seqlen_k : max_seqlen_k);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q       = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k       = (nhead_k * shape_seqlen_k * hdim_q);
        const ck_tile::index_t batch_stride_v       = (nhead_k * shape_seqlen_k * hdim_v);
        const ck_tile::index_t batch_stride_bias    = 0;
        const ck_tile::index_t batch_stride_o       = (nhead * shape_seqlen_q * hdim_v);
        const ck_tile::index_t batch_stride_randval = (nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_do      = (nhead * shape_seqlen_q * hdim_v);
        const ck_tile::index_t batch_stride_lsed    = (nhead * shape_seqlen_q);
        const ck_tile::index_t batch_stride_dk      = (nhead * shape_seqlen_k * hdim_q);
        const ck_tile::index_t batch_stride_dv      = (nhead * shape_seqlen_k * hdim_v);
        const ck_tile::index_t batch_stride_dbias   = (nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t split_stride_dq_acc =
            (shape_batch * nhead * shape_seqlen_q * hdim_q);

        const auto drop_seed_offset = [&]() -> decltype(fmha_bwd_args::drop_seed_offset) {
            if(drop_prefs)
            {
                return std::make_pair(drop_seed_buf.GetDeviceBuffer(),
                                      drop_offset_buf.GetDeviceBuffer());
            }
            else
            {
                return std::make_pair(drop_seed, drop_offset);
            }
        }();

        return fmha_bwd_args{q_buf.GetDeviceBuffer(),
                             k_buf.GetDeviceBuffer(),
                             v_buf.GetDeviceBuffer(),
                             bias.type == bias_enum::alibi ? alibi_slope_buf.GetDeviceBuffer()
                                                             : bias_buf.GetDeviceBuffer(),
                             o_buf.GetDeviceBuffer(),
                             lse_buf.GetDeviceBuffer(),
                             do_buf.GetDeviceBuffer(),
                             d_buf.GetDeviceBuffer(),
                             randval_buf.GetDeviceBuffer(),
                             dq_buf.GetDeviceBuffer(),
                             dk_buf.GetDeviceBuffer(),
                             dv_buf.GetDeviceBuffer(),
                             dbias_buf.GetDeviceBuffer(),
                             dq_acc_buf.GetDeviceBuffer(),
                             seqstart_q.GetDeviceBuffer(),
                             seqstart_k.GetDeviceBuffer(),
                             nullptr,
                             shape_seqlen_q,
                             shape_seqlen_k,
                             batch,
                             max_seqlen_q,
                             max_seqlen_k,
                             hdim_q,
                             hdim_v,
                             nhead,
                             nhead_k,
                             scale,
                             stride_q,
                             stride_k,
                             stride_v,
                             bias.type == bias_enum::alibi ? (bias.rank_info == 0 ? 0 : nhead)
                                                             : stride_bias,
                             stride_o,
                             stride_randval,
                             stride_do,
                             stride_q, // stride_dq_acc
                             stride_q, // stride_dq
                             stride_dk,
                             stride_dv,
                             stride_dbias,
                             nhead_stride_q,
                             nhead_stride_k,
                             nhead_stride_v,
                             nhead_stride_bias,
                             nhead_stride_o,
                             nhead_stride_randval,
                             nhead_stride_do,
                             nhead_stride_lsed,
                             nhead_stride_q, // nhead_stride_dq_acc
                             nhead_stride_q, // nhead_stride_dq
                             nhead_stride_k, // nhead_stride_dk
                             nhead_stride_v, // nhead_stride_dv
                             nhead_stride_dbias,
                             batch_stride_q,
                             batch_stride_k,
                             batch_stride_v,
                             batch_stride_bias,
                             batch_stride_o,
                             batch_stride_randval,
                             batch_stride_do,
                             batch_stride_lsed,
                             batch_stride_q, // batch_stride_dq_acc
                             batch_stride_q, // batch_stride_dq
                             batch_stride_dk,
                             batch_stride_dv,
                             batch_stride_dbias,
                             split_stride_dq_acc,
                             mask.left,
                             mask.right,
                             static_cast<ck_tile::index_t>(mask.type),
                             p_drop,
                             p_undrop,
                             drop_seed_offset};
    }();

    float ave_time = fmha_bwd(fmha_traits, fmha_args, stream_config);
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

    bool pass = true;

    std::vector<ck_tile::HostTensor<QDataType>> q_host_refs;
    std::vector<ck_tile::HostTensor<KDataType>> k_host_refs;
    std::vector<ck_tile::HostTensor<VDataType>> v_host_refs;
    std::vector<ck_tile::HostTensor<ODataType>> o_host_refs;
    std::vector<ck_tile::HostTensor<RandValOutputDataType>> randval_host_refs;
    std::vector<ck_tile::HostTensor<AccDataType>> p_hp_host_refs;
    std::vector<ck_tile::HostTensor<GemmDataType>> p_lp_host_refs;

    randval_buf.FromDevice(randval_host.data());

    for(ck_tile::index_t wb = 0; wb < batch; ++wb)
    {
        const ck_tile::index_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
        const ck_tile::index_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

        // adjust matrix index according to the mode
        const ck_tile::index_t b            = (mode == mode_enum::batch ? wb : 0);
        const ck_tile::index_t query_offset = (mode == mode_enum::batch ? 0 : seqstart_q_host[wb]);
        const ck_tile::index_t key_offset   = (mode == mode_enum::batch ? 0 : seqstart_k_host[wb]);

        ck_tile::HostTensor<QDataType> q_host_ref({nhead, real_seqlen_q, hdim_q}); // q_g_m_k
        ck_tile::HostTensor<KDataType> k_host_ref({nhead, real_seqlen_k, hdim_q}); // k_g_n_k
        ck_tile::HostTensor<VDataType> v_host_ref({nhead, hdim_v, real_seqlen_k}); // v_g_o_n
        ck_tile::HostTensor<ODataType> o_host_ref({nhead, real_seqlen_q, hdim_v}); // o_g_m_o
        ck_tile::HostTensor<LSEDataType> lse_host_ref({nhead, real_seqlen_q});     // lse_g_m
        ck_tile::HostTensor<RandValOutputDataType> randval_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // randval_g_m_n
        ck_tile::HostTensor<AccDataType> s_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // s_g_m_n
        ck_tile::HostTensor<AccDataType> p_hp_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // p_hp_g_m_n high precision
        ck_tile::HostTensor<AccDataType> p_dropped_hp_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // p_dropped_hp_g_m_n high precision
        ck_tile::HostTensor<GemmDataType> p_lp_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // p_lp_g_m_n low precision

        ck_tile::index_t nr = nhead / nhead_k;

        // clang-format off
        // permute
        if(i_perm) q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b, i[0], i[1] + query_offset, i[2]); });
        else       q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b, i[1] + query_offset, i[0], i[2]); });

        if(i_perm) k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(b, i[0] / nr, i[1] + key_offset, i[2]); });
        else       k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(b, i[1] + key_offset, i[0] / nr, i[2]); });

        // v_host_ref: [nhead, hdim, seq], v_host: [b, h_k, s, d]
        if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[0] / nr, i[2] + key_offset, i[1]); });
        // v_host_ref: [nhead, hdim, seq], v_host: [b, s, h_k, d]
        else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[2] + key_offset, i[0] / nr, i[1]); });
        // clang-format on

        // reference
        // S = scale * Q * K^T
        ck_tile::reference_batched_gemm<QDataType, KDataType, AccDataType, AccDataType>(
            q_host_ref,
            k_host_ref,
            s_host_ref,
            ck_tile::identity{},
            ck_tile::identity{},
            ck_tile::scales(scale)); // s_g_m_n = scale * q_g_m_k@k_g_n_k

        if(bias.type == bias_enum::elementwise_bias)
        {
            // elementwise bias
            ck_tile::HostTensor<BiasDataType> bias_host_ref({1, real_seqlen_q, real_seqlen_k});
            // clang-format off
            if(i_perm)
                bias_host_ref.ForEach([&](auto& self, auto i) { self(i) = bias_host(0, 0, i[1] + query_offset, i[2]); });
            else
                bias_host_ref.ForEach([&](auto& self, auto i) { self(i) = bias_host(0, i[1] + query_offset, 0, i[2]); });
            // clang-format on

            // broadcast from [1, real_seqlen_q, real_seqlen_k] to [nhead, real_seqlen_q,
            // real_seqlen_k]
            ck_tile::
                reference_batched_elementwise<AccDataType, BiasDataType, AccDataType, AccDataType>(
                    s_host_ref, bias_host_ref, s_host_ref);
        }
        else if(bias.type == bias_enum::alibi)
        {
            // alibi construct elementwise bias to verify
            auto alibi_host = [&]() {
                if(mask.type != mask_enum::no_mask)
                {
                    return ck_tile::make_alibi_from_lr_mask<AccDataType, false>(
                        0,
                        mask.left,
                        mask.right,
                        real_seqlen_q,
                        real_seqlen_k,
                        static_cast<ck_tile::GenericAttentionMaskEnum>(mask.type));
                }
                else
                {
                    return ck_tile::Alibi<AccDataType, false>{
                        0, real_seqlen_q, real_seqlen_k, ck_tile::AlibiMode::FROM_BOTTOM_RIGHT};
                }
            }();

            ck_tile::HostTensor<AccDataType> alibi_bias_host_ref(
                {nhead, real_seqlen_q, real_seqlen_k});
            auto i_b_slope = bias.rank_info == 0 ? 0 : wb;
            for(auto i_h = 0; i_h < nhead; i_h++)
            {
                AccDataType current_slope = alibi_slope_host(i_b_slope, i_h);
                alibi_host.slope = alibi_host.mode == ck_tile::AlibiMode::VERTICAL ? current_slope
                                                                                   : -current_slope;
                for(auto i_r = 0; i_r < real_seqlen_q; i_r++)
                {
                    for(auto i_c = 0; i_c < real_seqlen_k; i_c++)
                    {
                        AccDataType pixel = 0;
                        alibi_host.update(pixel, i_r, i_c);
                        alibi_bias_host_ref(i_h, i_r, i_c) = pixel;
                    }
                }
            }
            // [nhead, real_seqlen_q, real_seqlen_k]
            ck_tile::
                reference_batched_elementwise<AccDataType, AccDataType, AccDataType, AccDataType>(
                    s_host_ref, alibi_bias_host_ref, s_host_ref);
        }

        if(mask.type == mask_enum::no_mask)
        {
            ck_tile::reference_batched_masking<AccDataType>(
                s_host_ref, FmhaMasks::NoMask{real_seqlen_q, real_seqlen_k});
        }
        else if(mask.type == mask_enum::window_generic)
        {
            ck_tile::reference_batched_masking<AccDataType>(
                s_host_ref,
                ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::GenericMask>(
                    mask.left, mask.right, real_seqlen_q, real_seqlen_k));
        }
        else
        {
            // if left window size is negative, means causal
            // else means generic (for current batch)
            if(mask.left < 0)
                ck_tile::reference_batched_masking<AccDataType>(
                    s_host_ref,
                    ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::CausalMask>(
                        mask.left,
                        mask.right,
                        real_seqlen_q,
                        real_seqlen_k,
                        mask.type == mask_enum::mask_top_left));
            else
                ck_tile::reference_batched_masking<AccDataType>(
                    s_host_ref,
                    ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::GenericMask>(
                        mask.left,
                        mask.right,
                        real_seqlen_q,
                        real_seqlen_k,
                        mask.type == mask_enum::mask_top_left));
        }
        ck_tile::reference_batched_softmax<AccDataType, LSEDataType, AccDataType>(
            s_host_ref, p_hp_host_ref, ck_tile::identity{}, lse_host_ref);

        if(p_drop > 0)
        {
            p_hp_host_ref.ForEach(
                [&](auto& self, auto idx) { p_dropped_hp_host_ref(idx) = self(idx); });
            randval_host_ref.ForEach([&](auto& self, auto idx) {
                self(idx) = randval_host(b, idx[0], idx[1] + query_offset, idx[2]);
            });
            ck_tile::reference_batched_dropout(
                p_dropped_hp_host_ref, randval_host_ref, p_undrop_in_uint8_t, rp_undrop);
            p_dropped_hp_host_ref.ForEach([&](auto& self, auto idx) {
                p_lp_host_ref(idx) = ck_tile::type_convert<GemmDataType>(self(idx));
            });
        }
        else
        {
            p_hp_host_ref.ForEach([&](auto& self, auto idx) {
                p_lp_host_ref(idx) = ck_tile::type_convert<GemmDataType>(self(idx));
            });
        }

        // O = P * V
        ck_tile::reference_batched_gemm<GemmDataType, VDataType, AccDataType, ODataType>(
            p_lp_host_ref, v_host_ref, o_host_ref); // o_g_m_o = p_lp_g_m_n@v_g_o_n

        // clang-format off
        // permute
        if(o_perm) o_host_ref.ForEach([&](auto& self, auto idx) { o_host(b, idx[0], idx[1] + query_offset, idx[2]) = self(idx); });
        else       o_host_ref.ForEach([&](auto& self, auto idx) { o_host(b, idx[1] + query_offset, idx[0], idx[2]) = self(idx); });

        lse_host_ref.ForEach([&](auto& self, auto idx) { lse_host(b, idx[0], idx[1] + query_offset) = self(idx); });
        // clang-format on

        q_host_refs.push_back(q_host_ref);
        k_host_refs.push_back(k_host_ref);
        v_host_refs.push_back(v_host_ref);
        o_host_refs.push_back(o_host_ref);
        p_hp_host_refs.push_back(p_hp_host_ref);
        p_lp_host_refs.push_back(p_lp_host_ref);
        if(p_drop > 0)
        {
            randval_host_refs.push_back(randval_host_ref);
        }
    }

    o_buf.ToDevice(o_host.data());
    lse_buf.ToDevice(lse_host.data());
    dq_buf.SetZero();
    dbias_buf.SetZero();
    dq_acc_buf.SetZero();

    ck_tile::stream_config stream_config_v{
        nullptr, true, 0, 0, 1, arg_parser.get_str("timer") == std::string("gpu")};
    fmha_bwd(fmha_traits, fmha_args, stream_config_v);

    dq_buf.FromDevice(dq_host.data());
    dk_buf.FromDevice(dk_host.data());
    dv_buf.FromDevice(dv_host.data());
    dbias_buf.FromDevice(dbias_host.data());

    for(ck_tile::index_t wb = 0; wb < batch; ++wb)
    {
        const ck_tile::index_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
        const ck_tile::index_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

        // adjust matrix index according to the mode
        const ck_tile::index_t b            = (mode == mode_enum::batch ? wb : 0);
        const ck_tile::index_t query_offset = (mode == mode_enum::batch ? 0 : seqstart_q_host[wb]);
        const ck_tile::index_t key_offset   = (mode == mode_enum::batch ? 0 : seqstart_k_host[wb]);

        ck_tile::HostTensor<OGradDataType> do_host_ref({nhead, real_seqlen_q, hdim_v}); // do_g_m_o
        ck_tile::HostTensor<AccDataType> ds_hp_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // ds_g_m_n high precision
        ck_tile::HostTensor<GemmDataType> ds_lp_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // ds_g_m_n low precision
        ck_tile::HostTensor<AccDataType> dp_hp_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // dp_g_m_n high precision
        ck_tile::HostTensor<BiasGradDataType> dbias_host_ref(
            {nhead, real_seqlen_q, real_seqlen_k}); // dbias_g_m_n
        ck_tile::HostTensor<QGradDataType> dq_host_ref({nhead, real_seqlen_q, hdim_q}); // dq_g_m_k
        ck_tile::HostTensor<KGradDataType> dk_host_ref({nhead, real_seqlen_k, hdim_q}); // dk_g_n_k
        ck_tile::HostTensor<VGradDataType> dv_host_ref({nhead, real_seqlen_k, hdim_v}); // dv_g_n_o

        // clang-format off
        if(o_perm) do_host_ref.ForEach([&](auto& self, auto i) { self(i) = do_host(b, i[0], i[1] + query_offset, i[2]); });
        else       do_host_ref.ForEach([&](auto& self, auto i) { self(i) = do_host(b, i[1] + query_offset, i[0], i[2]); });
        // clang-format on

        // dP = dO@V x Z w/  dropout
        // dP = dO@V     w/o dropout
        auto v_t_host_ref = v_host_refs[wb].transpose({0, 2, 1}); // v_g_o_n -> v_g_n_o
        ck_tile::reference_batched_gemm<OGradDataType, VDataType, AccDataType, AccDataType>(
            do_host_ref, v_t_host_ref, dp_hp_host_ref); // dp_g_m_n = do_g_m_o@v_g_n_o

        if(p_drop > 0)
        {
            ck_tile::reference_batched_dropout(
                dp_hp_host_ref, randval_host_refs[wb], p_undrop_in_uint8_t, rp_undrop);
        }

        // dS_i_j = P_i_j .* (dP_i_j - dO_i dot O_i)
        ds_hp_host_ref.ForEach([&](auto& self, auto idx_gmn) {
            AccDataType do_dot_o = 0;
            for(int o = 0; o < hdim_v; o++)
            {
                auto idx_gmo = idx_gmn;
                idx_gmo[2]   = o;
                do_dot_o += ck_tile::type_convert<AccDataType>(do_host_ref(idx_gmo)) *
                            ck_tile::type_convert<AccDataType>(o_host_refs[wb](idx_gmo));
            }
            self(idx_gmn) = ck_tile::type_convert<AccDataType>(
                p_hp_host_refs[wb](idx_gmn) * (dp_hp_host_ref(idx_gmn) - do_dot_o));
        });

        if(use_dbias)
        {
            ds_hp_host_ref.ForEach([&](auto& self, auto idx) {
                dbias_host_ref(idx) = ck_tile::type_convert<BiasGradDataType>(self(idx));
            });
        }

        ds_hp_host_ref.ForEach([&](auto& self, auto idx) {
            ds_lp_host_ref(idx) = ck_tile::type_convert<GemmDataType>(self(idx));
        });

        // dV = P_drop^T@dO^T
        // dV = P^T@dO^T w/o dropout
        auto p_t_lp_host_ref = p_lp_host_refs[wb].transpose({0, 2, 1}); // p_lp_g_m_n -> p_lp_g_n_m
        auto do_t_host_ref   = do_host_ref.transpose({0, 2, 1});        // do_g_m_o -> do_g_o_m
        ck_tile::reference_batched_gemm<GemmDataType, OGradDataType, AccDataType, VGradDataType>(
            p_t_lp_host_ref, do_t_host_ref, dv_host_ref); // dv_g_n_o = p_lp_g_n_m@do_g_o_m

        // dQ = scale * dS@K^T
        auto k_t_host_ref = k_host_refs[wb].transpose({0, 2, 1}); // k_g_n_k -> k_g_k_n
        ck_tile::reference_batched_gemm<GemmDataType, KDataType, AccDataType, QGradDataType>(
            ds_lp_host_ref,
            k_t_host_ref,
            dq_host_ref,
            ck_tile::identity{},
            ck_tile::identity{},
            ck_tile::scales(scale)); // dq_g_m_k = ds_g_m_n@k_g_k_n

        // dK = scale * dS^T@Q^T
        auto ds_t_lp_host_ref = ds_lp_host_ref.transpose({0, 2, 1});  // ds_g_m_n -> ds_g_n_m
        auto q_t_host_ref     = q_host_refs[wb].transpose({0, 2, 1}); // q_g_m_k -> q_g_k_m
        ck_tile::reference_batched_gemm<GemmDataType, QDataType, AccDataType, KGradDataType>(
            ds_t_lp_host_ref,
            q_t_host_ref,
            dk_host_ref,
            ck_tile::identity{},
            ck_tile::identity{},
            ck_tile::scales(scale)); // dk_g_n_k = ds_g_n_m@q_g_k_m

        ck_tile::HostTensor<QGradDataType> dq_host_result(
            {nhead, real_seqlen_q, hdim_q}); // dq_g_m_k
        ck_tile::HostTensor<KGradDataType> dk_host_result(
            {nhead, real_seqlen_k, hdim_q}); // dk_g_n_k
        ck_tile::HostTensor<VGradDataType> dv_host_result(
            {nhead, real_seqlen_k, hdim_v}); // dv_g_n_o
        ck_tile::HostTensor<BiasGradDataType> dbias_host_result(
            {nhead, real_seqlen_q, real_seqlen_k}); // dbias_g_m_n

        // clang-format off
        // permute
        if(i_perm) dq_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dq_host(b, idx[0], idx[1] + query_offset, idx[2]); });
        else       dq_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dq_host(b, idx[1] + query_offset, idx[0], idx[2]); });

        if(i_perm) dk_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dk_host(b, idx[0], idx[1] + key_offset, idx[2]); });
        else       dk_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dk_host(b, idx[1] + key_offset, idx[0], idx[2]); });

        if(i_perm) dv_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dv_host(b, idx[0], idx[1] + key_offset, idx[2]); });
        else       dv_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dv_host(b, idx[1] + key_offset, idx[0], idx[2]); });

        if(use_dbias)
        {
            if(i_perm) dbias_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dbias_host(b, idx[0], idx[1] + query_offset, idx[2]); });
            else       dbias_host_result.ForEach([&](auto& self, auto idx) {self(idx) = dbias_host(b, idx[1] + query_offset, idx[0], idx[2]); });
        }
        // clang-format on

        auto [rtol, atol] = get_elimit<DataTypeConfig>(hdim_q, hdim_v);
        bool dq_cur_pass  = ck_tile::check_err(dq_host_result,
                                              dq_host_ref,
                                              std::string("Error: QGrad Incorrect results!"),
                                              rtol,
                                              atol);
        bool dk_cur_pass  = ck_tile::check_err(dk_host_result,
                                              dk_host_ref,
                                              std::string("Error: KGrad Incorrect results!"),
                                              rtol,
                                              atol);
        bool dv_cur_pass  = ck_tile::check_err(dv_host_result,
                                              dv_host_ref,
                                              std::string("Error: VGrad Incorrect results!"),
                                              rtol,
                                              atol);

        bool dbias_cur_pass = true;
        if(use_dbias)
        {
            dbias_cur_pass = ck_tile::check_err(dbias_host_result,
                                                dbias_host_ref,
                                                std::string("Error: BiasGrad Incorrect results!"),
                                                rtol,
                                                atol);
        }
        pass &= (dq_cur_pass & dk_cur_pass & dv_cur_pass & dbias_cur_pass);
        if(!(dq_cur_pass & dk_cur_pass & dv_cur_pass & dbias_cur_pass))
        {
            std::cerr << "mismatch found at batch: " << wb << std::endl
                      << "\tseqlen_q: " << real_seqlen_q << std::endl
                      << "\tseqlen_k: " << real_seqlen_k << std::endl
                      << "\tseqstart_q: " << seqstart_q_host << std::endl
                      << "\tseqstart_k: " << seqstart_k_host << std::endl;

            break;
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
        return run<FmhaBwdFp16>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "bf16")
    {
        return run<FmhaBwdBf16>(arg_parser) ? 0 : -2;
    }

    return -3;
}
