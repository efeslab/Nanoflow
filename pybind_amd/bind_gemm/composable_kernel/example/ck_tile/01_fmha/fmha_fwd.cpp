// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_fwd.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ref/naive_attention.hpp"
#include "mask.hpp"
#include "rotary.hpp"
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

#if CK_TILE_FMHA_FWD_APPENDKV_API && !CK_TILE_FMHA_FWD_SPLITKV_API
#error "we should enable fmha_fwd_splitkv() api in order to cooperate with fmha_fwd_appendkv()"
#endif

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
    arg_parser.insert("v", "1", "0:no validation, 2:cpu validation, 2:gpu validation(experimental)")
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
        .insert("s_k", "-1", "seqlen_k (including new key/value), -1 means equal to s")
        .insert("s_knew",
                "0",
                "seqlen_k for new key/value, 0 means not to use this at all; "
                "-1 to choose s_knew in [1, s] randomly.")
        .insert("s_kpad",
                "-1",
                "seqlen_k stride between 2 batches, currently used in group-mode only\n"
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
        .insert("p_drop", "0", "0~1 probability of dropout")
        .insert("drop_seed", "1", "seed for random number generator")
        .insert("drop_offset", "0", "offset for random number generator")
        .insert("drop_prefs",
                "0",
                "seed and offset values are present on GPU; 0 - host, 1 - device/GPU")
        .insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
        .insert(
            "rotary_dim", "0", "RoPE rotary dimension. rotary_dim <= 0 means not apply RoPE at all")
        .insert("rotary_interleaved", "1", "whether to apply interleaved RoPE")
        .insert("num_splits",
                "1",
                "# of splits for key/value. 0 to determine actual number by heuristic")
        .insert("page_block_size", "0", "paged-kvcache block size. 0 means not use paged-kvcahe")
        .insert("cache_batch_idx", "0", "whether to use index map to the kvcache")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "20", "number of iterations to benchmark the kernel");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// different threshold for different dtype
template <typename DataTypeConfig>
auto get_elimit(std::string /*init_method*/)
{
    double rtol = 1e-3;
    double atol = 1e-3;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<FmhaFwdBf16>(std::string /*init_method*/)
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<FmhaFwdFp8>(std::string init_method)
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

int num_splits_heuristic(int batch_nhead_mblocks, int num_SMs, int num_n_blocks, int max_splits)
{
    // If we have enough to almost fill the SMs, then just use 1 split
    if(batch_nhead_mblocks >= 0.8f * num_SMs)
    {
        return 1;
    }
    max_splits           = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 ||
               ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for(int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        if(!is_split_eligible(num_splits))
        {
            efficiency.push_back(0.f);
        }
        else
        {
            float n_waves = float(batch_nhead_mblocks * num_splits) / num_SMs;
            float eff     = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if(eff > max_efficiency)
            {
                max_efficiency = eff;
            }
            efficiency.push_back(eff);
        }
    }
    for(int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        if(!is_split_eligible(num_splits))
        {
            continue;
        }
        if(efficiency[num_splits - 1] >= 0.85 * max_efficiency)
        {
            // printf("num_splits chosen = %d\n", num_splits);
            return num_splits;
        }
    }
    return 1;
}

int override_num_splits_if_necessary(
    int batch, int nhead, int max_seqlen_q, int hdim_v, float p_drop, int num_splits)
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return num_splits;
    }

    hipDeviceProp_t props{};
    status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
    {
        return num_splits;
    }

    // tile size should match the generate.py
    const int kM0 = 64;
    const int kN1 = hdim_v;

    const int num_m_blocks = ck_tile::integer_divide_ceil(max_seqlen_q, kM0);
    const int num_n_blocks = ck_tile::integer_divide_ceil(hdim_v, kN1);

    if(num_splits < 1 && p_drop == 0.0f)
    {
        return num_splits_heuristic(
            batch * nhead * num_m_blocks, props.multiProcessorCount * 2, num_n_blocks, 128);
    }

    return num_splits;
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

    std::optional<uint32_t> seed = arg_parser.get_uint32("seed");
    if(*seed == 0)
    {
        seed.reset();
    }

    ck_tile::index_t hdim_q = arg_parser.get_int("d");
    ck_tile::index_t hdim_v = arg_parser.get_int("d_v");
    if(hdim_v < 0)
        hdim_v = hdim_q;

    ck_tile::index_t seqlen_knew = arg_parser.get_int("s_knew");
#if !CK_TILE_FMHA_FWD_APPENDKV_API
    if(seqlen_knew != 0)
    {
        std::cerr << "fmha_fwd_appendkv() is not enabled. ignoring the 's_knew' option"
                  << std::endl;
        seqlen_knew = 0;
    }
#endif
    if(seqlen_knew < 0)
    {
        seqlen_knew = randint<ck_tile::index_t>(1, arg_parser.get_int("s"), seed);
    }

    ck_tile::index_t rotary_dim = arg_parser.get_int("rotary_dim");
    if constexpr(!(std::is_same_v<DataTypeConfig, FmhaFwdFp16> ||
                   std::is_same_v<DataTypeConfig, FmhaFwdBf16>))
    {
        if(0 < rotary_dim)
        {
            std::cerr << "rotary embedding is only available for data type=fp16|bf16" << std::endl;
            return false;
        }
    }
#if !CK_TILE_FMHA_FWD_APPENDKV_API
    else if(0 < rotary_dim)
    {
        std::cerr << "rotary embedding is not supported. ignoring the 'rotary_dim' option"
                  << std::endl;
        rotary_dim = 0;
    }
#endif
    // to use fmha_fwd_appendkv(), make sure it's in batch mode
    const bool need_append_kvcache = (0 < seqlen_knew || 0 < rotary_dim);
    if(need_append_kvcache && mode == mode_enum::group)
    {
        std::cerr << "fmha_fwd_appendkv() will be invoked. ignoring the 'mode' option" << std::endl;
        mode = mode_enum::batch;
    }
    if(!(rotary_dim <= hdim_q))
    {
        std::cerr << "rotary_dim should be less than or equal to head dim for q" << std::endl;
        return false;
    }
    else if(!(rotary_dim % 16 == 0))
    {
        std::cerr << "only rotary dimensions divisible by 16 are currently supported" << std::endl;
        return false;
    }

    ck_tile::index_t page_block_size = arg_parser.get_int("page_block_size");
#if !CK_TILE_FMHA_FWD_APPENDKV_API && !CK_TILE_FMHA_FWD_SPLITKV_API
    if(0 < page_block_size)
    {
        std::cerr << "paged-kvcache is not supported. ignoring the 'page_block_size' option"
                  << std::endl;
        page_block_size = 0;
    }
#endif
    if(!(page_block_size % 128 == 0))
    {
        std::cerr << "only paged-kvcache block size divisible by 128 are currently supported"
                  << std::endl;
        return false;
    }

    bool use_cache_batch_idx = arg_parser.get_bool("cache_batch_idx");
#if !CK_TILE_FMHA_FWD_APPENDKV_API && !CK_TILE_FMHA_FWD_SPLITKV_API
    if(use_cache_batch_idx)
    {
        std::cerr << "split-kv is not supported. ignoring the 'cache_batch_idx' option"
                  << std::endl;
        use_cache_batch_idx = false;
    }
#else
    if(use_cache_batch_idx)
    {
        if(0 < page_block_size)
        {
            std::cerr << "paged-kvcache does not support cache_batch_idx. ignoring the "
                         "'cache_batch_idx' option"
                      << std::endl;
            use_cache_batch_idx = false;
        }
        else if(mode == mode_enum::group)
        {
            std::cerr << "group mode will not use cache_batch_idx. ignoring the "
                         "'cache_batch_idx' option"
                      << std::endl;
            use_cache_batch_idx = false;
        }
    }
#endif
    const bool use_kvcache = (need_append_kvcache || use_cache_batch_idx || 0 < page_block_size);

    auto [seqlen_qs, seqlen_ks, seqlen_kpads] =
        decode_seqlen(mode,
                      batch,
                      arg_parser.get_str("s"),
                      arg_parser.get_str("s_k"),
                      arg_parser.get_str("s_kpad"),
                      /*seqlen_k_min=*/0 < seqlen_knew ? seqlen_knew : 0,
                      need_append_kvcache);
    // compute kvcache seqlen_k (before appending knew/vnew)
    auto cache_seqlen_ks = seqlen_ks;
    std::transform(cache_seqlen_ks.begin(),
                   cache_seqlen_ks.end(),
                   cache_seqlen_ks.begin(),
                   [&](auto seqlen_k) { return seqlen_k - seqlen_knew; });

#if 0
    // clang-format off
    std::cout << "seqlen_qs:"; for(auto xx : seqlen_qs) { std::cout << xx << ","; } std::cout << std::endl;
    std::cout << "seqlen_ks:"; for(auto xx : seqlen_ks) { std::cout << xx << ","; } std::cout << std::endl;
    std::cout << "seqlen_kpads:"; for(auto xx : seqlen_kpads) { std::cout << xx << ","; } std::cout << std::endl;
    // clang-format on
#endif

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

    std::string vlayout = arg_parser.get_str("vlayout");
    bool lse            = arg_parser.get_bool("lse");

    bias_info bias = bias_info::decode(arg_parser.get_str("bias"));
    mask_info mask = mask_info::decode(
        arg_parser.get_str("mask"), seqlen_qs[0], seqlen_ks[0]); // TODO: we don't need x/y anymore

    float p_drop         = arg_parser.get_float("p_drop");
    uint64_t drop_seed   = arg_parser.get_uint64("drop_seed");
    uint64_t drop_offset = arg_parser.get_uint64("drop_offset");
    bool drop_prefs      = arg_parser.get_bool("drop_prefs");

    if(p_drop < 0.0f || p_drop > 1.0f)
    {
        std::cerr << "The value of p_drop should be 0~1" << std::endl;
        return false;
    }

    bool s_randval = false;
    if(p_drop > 0.0f && do_validation != 0)
    {
        s_randval = true;
    }

    std::string init_method = arg_parser.get_str("init");

    const bool is_rotary_interleaved = arg_parser.get_bool("rotary_interleaved");

    ck_tile::index_t num_splits = arg_parser.get_int("num_splits");
#if !CK_TILE_FMHA_FWD_SPLITKV_API
    if(num_splits != 1)
    {
        std::cerr << "split-kv is not supported. ignoring the 'num_splits' option" << std::endl;
        num_splits = 1;
    }
#endif

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

    using TypeConfig = FmhaFwdTypeConfig<DataTypeConfig>;

    using QDataType             = typename TypeConfig::QDataType;
    using KDataType             = typename TypeConfig::KDataType;
    using VDataType             = typename TypeConfig::VDataType;
    using BiasDataType          = typename TypeConfig::BiasDataType;
    using RandValOutputDataType = typename TypeConfig::RandValOutputDataType;
    using LSEDataType           = typename TypeConfig::LSEDataType;
    using SaccDataType          = typename TypeConfig::SaccDataType;
    using SMPLComputeDataType   = typename TypeConfig::SMPLComputeDataType;
    using PDataType             = typename TypeConfig::PDataType;
    using OaccDataType          = typename TypeConfig::OaccDataType;
    using ODataType             = typename TypeConfig::ODataType;

    float range_q = arg_parser.get_float("range_q");
    float range_k = arg_parser.get_float("range_k");
    float range_v = arg_parser.get_float("range_v");
    float range_p = arg_parser.get_float("range_p");
    float range_o = arg_parser.get_float("range_o");

    float q_dtype_max = ck_tile::type_convert<float>(ck_tile::numeric<QDataType>::max());
    float k_dtype_max = ck_tile::type_convert<float>(ck_tile::numeric<KDataType>::max());
    float v_dtype_max = ck_tile::type_convert<float>(ck_tile::numeric<VDataType>::max());
    float p_dtype_max = v_dtype_max; // assume p and v is the same type
    float o_dtype_max = ck_tile::type_convert<float>(ck_tile::numeric<ODataType>::max());

    float scale_p = 1.f;
    float scale_o = 1.f;

    if(squant)
    {
        scale_s = scale_s * (range_q / q_dtype_max) * (range_k / k_dtype_max);
        scale_p = p_dtype_max / range_p;
        scale_o = (o_dtype_max / range_o) * (range_p / p_dtype_max) * (range_v / v_dtype_max);
    }

    // accumulation numbers for performance evaluation
    std::size_t flop = 0, num_byte = 0;
    auto max_seqlen_q =
        std::numeric_limits<int32_t>::min(); // we will use max seqlen to decide grid size
    auto max_seqlen_k = std::numeric_limits<int32_t>::min();
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

            flop += nhead * (static_cast<std::size_t>(2) * real_seqlen_q * real_seqlen_k * hdim_q +
                             static_cast<std::size_t>(2) * real_seqlen_q * hdim_v * real_seqlen_k);

            num_byte += nhead * (sizeof(QDataType) * real_seqlen_q * hdim_q +
                                 sizeof(KDataType) * real_seqlen_k * hdim_q +
                                 sizeof(VDataType) * hdim_v * real_seqlen_k +
                                 sizeof(ODataType) * real_seqlen_q * hdim_v);
        }
    }

    const ck_tile::index_t max_num_page_blocks =
        (0 < page_block_size
             ? batch * std::max(1, ck_tile::integer_divide_ceil(max_seqlen_k, page_block_size))
             : 0);

    // legalize num_splits according to other options
    if(num_splits < 1)
    {
        num_splits = override_num_splits_if_necessary(
            batch, nhead, max_seqlen_q, hdim_v, p_drop, num_splits);
    }
    if(128 < num_splits)
    {
        std::cerr << "num_splits greater than 128 is not supported" << std::endl;
        return false;
    }
#if CK_TILE_FMHA_FWD_SPLITKV_API
    if(0 < p_drop && (1 < num_splits || use_kvcache))
    {
        std::cerr << "dropout is not supoprted by split-kv kernels. ignoring the 'p_drop' option"
                  << std::endl;
        p_drop = 0.0f;
    }
#endif

    static const auto get_lengths = [](bool permute,
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
        0 < page_block_size
            ? get_lengths(i_perm, max_num_page_blocks, nhead_k, page_block_size, hdim_q)
            : get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_q));
    /// NOTICE: always use same shape for knew_host & vnew_host in batch/group mode
    ck_tile::HostTensor<KDataType> knew_host(
        0 < seqlen_knew
            ? get_lengths(i_perm, batch, nhead_k, seqlen_knew, hdim_q)
            : std::array<ck_tile::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);
    ck_tile::HostTensor<VDataType> v_host(
        0 < page_block_size
            ? (is_v_rowmajor
                   ? get_lengths(i_perm, max_num_page_blocks, nhead_k, page_block_size, hdim_v)
                   : get_lengths(i_perm, max_num_page_blocks, nhead_k, hdim_v, page_block_size))
            : (is_v_rowmajor ? get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_v)
                             : get_lengths(i_perm, shape_batch, nhead_k, hdim_v, shape_seqlen_k)));
    ck_tile::HostTensor<VDataType> vnew_host(
        0 < seqlen_knew
            ? (is_v_rowmajor ? get_lengths(i_perm, batch, nhead_k, seqlen_knew, hdim_v)
                             : get_lengths(i_perm, batch, nhead_k, hdim_v, seqlen_knew))
            : std::array<ck_tile::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);
    ck_tile::HostTensor<BiasDataType> bias_host(
        bias.type == bias_enum::elementwise_bias
            ? get_lengths(i_perm, 1, 1, shape_seqlen_q, shape_seqlen_k)
            : std::array<ck_tile::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);

    ck_tile::HostTensor<SaccDataType> alibi_slope_host(
        bias.type == bias_enum::alibi
            ? (bias.rank_info == 0 ? std::array<ck_tile::index_t, 2>{1, nhead}
                                   : std::array<ck_tile::index_t, 2>{batch, nhead})
            : std::array<ck_tile::index_t, 2>{1, 1});

    auto [rotary_cos_host, rotary_sin_host] = generate_rotary_cos_sin<KDataType>(
        std::max(shape_seqlen_q, shape_seqlen_k), rotary_dim, seed);

    ck_tile::HostTensor<LSEDataType> lse_acc_host(
        1 < num_splits || use_kvcache
            ? std::array<ck_tile::index_t, 4>{shape_batch, nhead, num_splits, shape_seqlen_q}
            : std::array<ck_tile::index_t, 4>{1, 1, 1, 1});
    ck_tile::HostTensor<OaccDataType> o_acc_host(
        1 < num_splits || use_kvcache ? std::array<ck_tile::index_t, 5>{shape_batch,
                                                                        nhead,
                                                                        num_splits,
                                                                        shape_seqlen_q,
                                                                        hdim_v}
                                      : std::array<ck_tile::index_t, 5>{1, 1, 1, 1, 1});

    // batch mode of lse data layout is [batch, nhead, seqlen_q]
    // group mode of lse data layout is [nhead, total_seqlen_q]
    ck_tile::HostTensor<LSEDataType> lse_host(
        lse ? std::array<ck_tile::index_t, 3>{shape_batch, nhead, shape_seqlen_q}
            : std::array<ck_tile::index_t, 3>{1, 1, 1} /* dummy shape for simplifying code */);

    ck_tile::HostTensor<ODataType> o_host(
        get_lengths(o_perm, shape_batch, nhead, shape_seqlen_q, hdim_v));

    ck_tile::HostTensor<RandValOutputDataType> randval_host(
        p_drop > 0 ? get_lengths(true, shape_batch, nhead, shape_seqlen_q, max_seqlen_k)
                   : std::array<ck_tile::index_t, 4>{1, 1, 1, 1});

    ck_tile::HostTensor<int32_t> block_table_host(
        0 < page_block_size ? std::array<ck_tile::index_t, 2>{batch, max_num_page_blocks / batch}
                            : std::array<ck_tile::index_t, 2>{1, 1});

    ck_tile::HostTensor<int32_t> cache_batch_idx_host(use_cache_batch_idx
                                                          ? std::array<ck_tile::index_t, 1>{batch}
                                                          : std::array<ck_tile::index_t, 1>{1});

    if(init_method == "ui" || init_method == "0")
    {
        ck_tile::FillUniformDistributionIntegerValue<QDataType>{-3.f, 3.f, seed}(q_host);
        ck_tile::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f, seed}(k_host);
        ck_tile::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f, seed}(knew_host);
        ck_tile::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f, seed}(v_host);
        ck_tile::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f, seed}(vnew_host);
        ck_tile::FillUniformDistributionIntegerValue<BiasDataType>{-3.f, 3.f, seed}(bias_host);
    }
    else if(init_method == "ni")
    {
        ck_tile::FillNormalDistributionIntegerValue<QDataType>{-3.f, 3.f, seed}(q_host);
        ck_tile::FillNormalDistributionIntegerValue<KDataType>{-3.f, 3.f, seed}(k_host);
        ck_tile::FillNormalDistributionIntegerValue<KDataType>{-3.f, 3.f, seed}(knew_host);
        ck_tile::FillNormalDistributionIntegerValue<VDataType>{-3.f, 3.f, seed}(v_host);
        ck_tile::FillNormalDistributionIntegerValue<VDataType>{-3.f, 3.f, seed}(vnew_host);
        ck_tile::FillNormalDistributionIntegerValue<BiasDataType>{-3.f, 3.f, seed}(bias_host);
    }
    else if(init_method == "uf" || init_method == "1")
    {
        ck_tile::FillUniformDistribution<QDataType>{0.f, 1.f, seed}(q_host);
        ck_tile::FillUniformDistribution<KDataType>{0.f, 1.f, seed}(k_host);
        ck_tile::FillUniformDistribution<KDataType>{0.f, 1.f, seed}(knew_host);
        ck_tile::FillUniformDistribution<VDataType>{0.f, 1.f, seed}(v_host);
        ck_tile::FillUniformDistribution<VDataType>{0.f, 1.f, seed}(vnew_host);
        ck_tile::FillUniformDistribution<BiasDataType>{0.f, 1.f, seed}(bias_host);
    }
    else if(init_method == "nf")
    {
        ck_tile::FillNormalDistribution<QDataType>{0.f, 3.f, seed}(q_host);
        ck_tile::FillNormalDistribution<KDataType>{0.f, 3.f, seed}(k_host);
        ck_tile::FillNormalDistribution<KDataType>{0.f, 3.f, seed}(knew_host);
        ck_tile::FillNormalDistribution<VDataType>{0.f, 3.f, seed}(v_host);
        ck_tile::FillNormalDistribution<VDataType>{0.f, 3.f, seed}(vnew_host);
        ck_tile::FillNormalDistribution<BiasDataType>{0.f, 3.f, seed}(bias_host);
    }
    else if(init_method == "tf" || init_method == "2")
    {
        ck_tile::FillTrigValue<QDataType>{}(q_host);
        ck_tile::FillTrigValue<KDataType>{}(k_host);
        ck_tile::FillTrigValue<KDataType>{}(knew_host);
        ck_tile::FillTrigValue<VDataType>{}(v_host);
        ck_tile::FillTrigValue<VDataType>{}(vnew_host);
        ck_tile::FillTrigValue<BiasDataType>{}(bias_host);
    }
    else if(init_method == "ufq" || init_method == "uf:q" ||
            init_method == "3") // suitable for fp8 quantization
    {
        ck_tile::FillUniformDistribution<QDataType>{-q_dtype_max, q_dtype_max, seed}(q_host);
        ck_tile::FillUniformDistribution<KDataType>{-k_dtype_max, k_dtype_max, seed}(k_host);
        ck_tile::FillUniformDistribution<KDataType>{-k_dtype_max, k_dtype_max, seed}(knew_host);
        ck_tile::FillUniformDistribution<VDataType>{-v_dtype_max, v_dtype_max, seed}(v_host);
        ck_tile::FillUniformDistribution<VDataType>{-v_dtype_max, v_dtype_max, seed}(vnew_host);

        // bias_fp8 = qscale_bias * bias_fp32
        float qscale_bias = (q_dtype_max / range_q) * (k_dtype_max / range_k);
        // Assume bias is in [-1.f, 1.f] in original fp32
        ck_tile::FillUniformDistribution<BiasDataType>{-qscale_bias, qscale_bias, seed}(bias_host);
    }
    if(bias.type == bias_enum::alibi)
    {
        auto slopes = ck_tile::get_alibi_slopes<SaccDataType>(nhead);
        assert(slopes.size() == static_cast<std::size_t>(nhead));
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
    iota_shuffle(block_table_host.begin(), block_table_host.end(), 0);
    iota_shuffle(cache_batch_idx_host.begin(), cache_batch_idx_host.end(), 0);

    ck_tile::DeviceMem q_buf(q_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(k_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem knew_buf(knew_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(v_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem vnew_buf(vnew_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem bias_buf(bias_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lse_acc_buf(lse_acc_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_acc_buf(o_acc_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lse_buf(lse_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(o_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(int32_t));
    ck_tile::DeviceMem seqlen_k_buf((mode == mode_enum::batch && use_kvcache) ||
                                            0 <= seqlen_kpads[0]
                                        ? seqlen_ks.size() * sizeof(int32_t)
                                        : 0);
    ck_tile::DeviceMem cache_seqlen_k_buf(
        need_append_kvcache ? cache_seqlen_ks.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem rotary_cos_buf(rotary_cos_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem rotary_sin_buf(rotary_sin_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem drop_seed_buf(drop_prefs ? sizeof(uint64_t) : 0);
    ck_tile::DeviceMem drop_offset_buf(drop_prefs ? sizeof(uint64_t) : 0);
    ck_tile::DeviceMem randval_buf(randval_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem alibi_slope_buf(alibi_slope_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem block_table_buf(block_table_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem cache_batch_idx_buf(cache_batch_idx_host.get_element_space_size_in_bytes());

    q_buf.ToDevice(q_host.data());
    k_buf.ToDevice(k_host.data());
    knew_buf.ToDevice(knew_host.data());
    v_buf.ToDevice(v_host.data());
    vnew_buf.ToDevice(vnew_host.data());
    bias_buf.ToDevice(bias_host.data());
    seqstart_q.ToDevice(seqstart_q_host.data());
    seqstart_k.ToDevice(seqlen_kpads[0] < 0 ? seqstart_k_host.data()
                                            : seqstart_k_with_padding_host.data());
    seqlen_k_buf.ToDevice((mode == mode_enum::batch && use_kvcache) || 0 <= seqlen_kpads[0]
                              ? seqlen_ks.data()
                              : nullptr);
    cache_seqlen_k_buf.ToDevice(need_append_kvcache ? cache_seqlen_ks.data() : nullptr);
    rotary_cos_buf.ToDevice(rotary_cos_host.data());
    rotary_sin_buf.ToDevice(rotary_sin_host.data());
    drop_seed_buf.ToDevice(drop_prefs ? &drop_seed : nullptr);
    drop_offset_buf.ToDevice(drop_prefs ? &drop_offset : nullptr);
    alibi_slope_buf.ToDevice(alibi_slope_host.data());
    block_table_buf.ToDevice(block_table_host.data());
    cache_batch_idx_buf.ToDevice(cache_batch_idx_host.data());

    // clang-format off
    auto layout_str = [&](bool permute){
        if(permute) return std::string("bhsd");
        else return std::string("bshd");
    };
    auto io_layout = [&](bool iperm_, bool operm_) {
        if(iperm_ == operm_) return layout_str(iperm_);
        else return layout_str(iperm_) + std::string("-") + layout_str(operm_);
    };
    // clang-format on
    const std::string prec = arg_parser.get_str("prec");

    std::cout << "[" << prec << "|" << mode << "|" << io_layout(i_perm, o_perm) << "] b:" << batch
              << ", h:" << nhead << "/" << nhead_k << ", s:" << seqlen_qs[0] << "/" << seqlen_ks[0]
              << (seqlen_kpads[0] < 0 ? ""
                                      : (std::string("(") + std::to_string(seqlen_kpads[0]) + ")"))
              << ", d:" << hdim_q << "/" << hdim_v << ", scale_s:" << scale_s << ", bias:" << bias
              << ", p_drop:" << p_drop << ", lse:" << lse << ", squant:" << squant
              << ", mask:" << mask << ", v:" << vlayout;
#if CK_TILE_FMHA_FWD_APPENDKV_API
    if(0 < rotary_dim)
    {
        std::cout << ", rotary_dim:" << rotary_dim << "("
                  << (is_rotary_interleaved ? "inter" : "half") << ")";
    }
#endif
#if CK_TILE_FMHA_FWD_SPLITKV_API
    if(1 < num_splits)
    {
        std::cout << ", num_splits:" << num_splits;
    }
    if(0 < page_block_size)
    {
        std::cout << ", page_block_size:" << page_block_size;
    }
    if(use_cache_batch_idx)
    {
        std::cout << ", cache_batch_idx:" << use_cache_batch_idx;
    }
#endif
    std::cout << std::flush;

    const auto init_traits = [&](auto& traits) {
        traits.hdim_q        = hdim_q;
        traits.hdim_v        = hdim_v;
        traits.data_type     = data_type;
        traits.is_v_rowmajor = is_v_rowmajor;

        if constexpr(std::is_same_v<fmha_fwd_appendkv_traits, std::decay_t<decltype(traits)>>)
        {
            traits.rope_type = (0 < rotary_dim ? (is_rotary_interleaved ? rope_enum::interleaved
                                                                        : rope_enum::half_rotated)
                                               : rope_enum::none);
        }
        else // fmha_fwd_traits or fmha_splitkv_traits
        {
            traits.is_group_mode       = (mode == mode_enum::group);
            traits.mask_type           = mask.type;
            traits.bias_type           = bias.type;
            traits.has_lse             = lse;
            traits.do_fp8_static_quant = squant;

            if constexpr(std::is_same_v<fmha_fwd_traits, std::decay_t<decltype(traits)>>)
            {
                traits.has_dropout = (p_drop > 0.0f);
            }
        }
    };

    const auto init_args = [&, k_paddings_ = seqlen_kpads](auto& args) {
        assert(nhead % nhead_k == 0);
        /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
        ///       seqlen_k] in this example, hence both the 'batch_stride_bias' &
        ///       'nhead_stride_bias' are 0.
        // setup stride_* arguments
        const ck_tile::index_t stride_q    = (i_perm ? hdim_q : nhead * hdim_q);
        const ck_tile::index_t stride_k    = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_knew = (i_perm ? hdim_q : nhead_k * hdim_q);
        const ck_tile::index_t stride_v    = [&]() {
            if(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return 0 < page_block_size ? (i_perm ? page_block_size : nhead_k * page_block_size)
                                              : (i_perm ? shape_seqlen_k : nhead_k * shape_seqlen_k);
        }();
        const ck_tile::index_t stride_vnew = [&]() {
            if(is_v_rowmajor)
                return i_perm ? hdim_v : nhead_k * hdim_v;
            else
                return i_perm ? seqlen_knew : nhead_k * seqlen_knew;
        }();
        const ck_tile::index_t stride_bias    = (i_perm ? shape_seqlen_k : 1 * shape_seqlen_k);
        const ck_tile::index_t stride_randval = (max_seqlen_k);
        const ck_tile::index_t stride_o_acc   = (hdim_v);
        const ck_tile::index_t stride_o       = (o_perm ? hdim_v : nhead * hdim_v);
        // setup nhead_stride_* arguments
        const ck_tile::index_t nhead_stride_q = (i_perm ? shape_seqlen_q * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_k =
            (0 < page_block_size ? (i_perm ? page_block_size * hdim_q : hdim_q)
                                 : (i_perm ? shape_seqlen_k * hdim_q : hdim_q));
        const ck_tile::index_t nhead_stride_knew = (i_perm ? seqlen_knew * hdim_q : hdim_q);
        const ck_tile::index_t nhead_stride_v    = [&]() {
            if(is_v_rowmajor)
                return 0 < page_block_size ? (i_perm ? page_block_size * hdim_v : hdim_v)
                                              : (i_perm ? shape_seqlen_k * hdim_v : hdim_v);
            else
                return 0 < page_block_size ? (i_perm ? hdim_v * page_block_size : page_block_size)
                                              : (i_perm ? hdim_v * shape_seqlen_k : shape_seqlen_k);
        }();
        const ck_tile::index_t nhead_stride_vnew = [&]() {
            if(is_v_rowmajor)
                return i_perm ? seqlen_knew * hdim_v : hdim_v;
            else
                return i_perm ? hdim_v * seqlen_knew : seqlen_knew;
        }();
        const ck_tile::index_t nhead_stride_bias =
            (i_perm ? 0 * shape_seqlen_q * shape_seqlen_k : 0 * shape_seqlen_k);
        const ck_tile::index_t nhead_stride_randval = (shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t nhead_stride_lse     = shape_seqlen_q;
        const ck_tile::index_t nhead_stride_lse_acc = (num_splits * shape_seqlen_q);
        const ck_tile::index_t nhead_stride_o_acc   = (num_splits * shape_seqlen_q * hdim_v);
        const ck_tile::index_t nhead_stride_o       = (o_perm ? shape_seqlen_q * hdim_v : hdim_v);
        // setup batch_stride_* arguments
        const ck_tile::index_t batch_stride_q = (nhead * shape_seqlen_q * hdim_q);
        const ck_tile::index_t batch_stride_k =
            (0 < page_block_size ? (nhead_k * page_block_size * hdim_q)
                                 : (nhead_k * shape_seqlen_k * hdim_q));
        const ck_tile::index_t batch_stride_knew = (nhead_k * seqlen_knew * hdim_q);
        const ck_tile::index_t batch_stride_v =
            (0 < page_block_size ? (nhead_k * hdim_v * page_block_size)
                                 : (nhead_k * hdim_v * shape_seqlen_k));
        const ck_tile::index_t batch_stride_vnew    = (nhead_k * hdim_v * seqlen_knew);
        const ck_tile::index_t batch_stride_bias    = (0 * nhead * shape_seqlen_q * shape_seqlen_k);
        const ck_tile::index_t batch_stride_randval = (nhead * shape_seqlen_q * max_seqlen_k);
        const ck_tile::index_t batch_stride_lse     = (nhead * shape_seqlen_q);
        const ck_tile::index_t batch_stride_lse_acc = (nhead * num_splits * shape_seqlen_q);
        const ck_tile::index_t batch_stride_o_acc = (nhead * num_splits * shape_seqlen_q * hdim_v);
        const ck_tile::index_t batch_stride_o     = (nhead * shape_seqlen_q * hdim_v);
        const ck_tile::index_t batch_stride_block_table = (max_num_page_blocks / batch);
        // setup split_stride_* arguments (only used in split-kv kernel)
        const ck_tile::index_t split_stride_lse_acc = (shape_seqlen_q);
        const ck_tile::index_t split_stride_o_acc   = (shape_seqlen_q * hdim_v);

        args.q_ptr = q_buf.GetDeviceBuffer();
        args.k_ptr = k_buf.GetDeviceBuffer();
        args.v_ptr = v_buf.GetDeviceBuffer();

        args.batch    = batch;
        args.seqlen_q = shape_seqlen_q; // unused in group mode
        args.hdim_q   = hdim_q;
        args.hdim_v   = hdim_v;
        args.nhead_q  = nhead;
        args.nhead_k  = nhead_k;

        args.stride_q       = stride_q;
        args.stride_k       = stride_k;
        args.stride_v       = stride_v;
        args.nhead_stride_q = nhead_stride_q;
        args.nhead_stride_k = nhead_stride_k;
        args.nhead_stride_v = nhead_stride_v;
        args.batch_stride_q = batch_stride_q;
        args.batch_stride_k = batch_stride_k;
        args.batch_stride_v = batch_stride_v;

        if constexpr(std::is_same_v<fmha_fwd_appendkv_args, std::decay_t<decltype(args)>>)
        {
            args.knew_ptr    = knew_buf.GetDeviceBuffer();
            args.vnew_ptr    = vnew_buf.GetDeviceBuffer();
            args.seqlen_knew = seqlen_knew;

            args.seqlen_k_ptr = cache_seqlen_k_buf.GetDeviceBuffer();

            args.rotary_cos_ptr = (0 < rotary_dim ? rotary_cos_buf.GetDeviceBuffer() : nullptr);
            args.rotary_sin_ptr = (0 < rotary_dim ? rotary_sin_buf.GetDeviceBuffer() : nullptr);
            args.rotary_dim     = rotary_dim;
            args.has_mask       = (mask.type != mask_enum::no_mask);

            args.block_table_ptr =
                (0 < page_block_size ? block_table_buf.GetDeviceBuffer() : nullptr);
            args.batch_stride_block_table = batch_stride_block_table;
            args.page_block_size          = page_block_size;

            args.cache_batch_idx =
                (use_cache_batch_idx ? cache_batch_idx_buf.GetDeviceBuffer() : nullptr);

            args.stride_knew       = stride_knew;
            args.stride_vnew       = stride_vnew;
            args.nhead_stride_knew = nhead_stride_knew;
            args.nhead_stride_vnew = nhead_stride_vnew;
            args.batch_stride_knew = batch_stride_knew;
            args.batch_stride_vnew = batch_stride_vnew;
        }
        else // fmha_fwd_args or fmha_fwd_splitkv_args
        {
            args.bias_ptr = bias.type == bias_enum::alibi ? alibi_slope_buf.GetDeviceBuffer()
                                                          : bias_buf.GetDeviceBuffer();
            args.lse_ptr  = lse_buf.GetDeviceBuffer();
            args.o_ptr    = o_buf.GetDeviceBuffer();

            args.seqstart_q_ptr =
                (mode == mode_enum::group ? seqstart_q.GetDeviceBuffer() : nullptr);
            args.seqstart_k_ptr =
                (mode == mode_enum::group ? seqstart_k.GetDeviceBuffer() : nullptr);
            args.seqlen_k_ptr = ((mode == mode_enum::batch && use_kvcache) || 0 <= k_paddings_[0]
                                     ? seqlen_k_buf.GetDeviceBuffer()
                                     : nullptr);

            args.seqlen_k     = shape_seqlen_k; // unused in group mode (or kvcache enabled)
            args.max_seqlen_q = max_seqlen_q;

            args.scale_s = scale_s;
            args.scale_p = scale_p;
            args.scale_o = scale_o;

            args.stride_bias =
                (bias.type == bias_enum::alibi ? (bias.rank_info == 0 ? 0 : nhead) : stride_bias);
            args.stride_o          = stride_o;
            args.nhead_stride_bias = nhead_stride_bias;
            args.nhead_stride_lse  = nhead_stride_lse;
            args.nhead_stride_o    = nhead_stride_o;
            args.batch_stride_bias = batch_stride_bias;
            args.batch_stride_lse  = batch_stride_lse;
            args.batch_stride_o    = batch_stride_o;

            args.window_size_left  = mask.left;
            args.window_size_right = mask.right;
            args.mask_type         = static_cast<ck_tile::index_t>(mask.type);

            if constexpr(std::is_same_v<fmha_fwd_args, std::decay_t<decltype(args)>>)
            {
                args.rand_val_ptr = randval_buf.GetDeviceBuffer();

                args.stride_randval       = stride_randval;
                args.nhead_stride_randval = nhead_stride_randval;
                args.batch_stride_randval = batch_stride_randval;

                args.p_drop    = p_drop;
                args.s_randval = s_randval;
                if(drop_prefs)
                {
                    args.drop_seed_offset = std::make_pair(drop_seed_buf.GetDeviceBuffer(),
                                                           drop_offset_buf.GetDeviceBuffer());
                }
                else
                {
                    args.drop_seed_offset = std::make_pair(drop_seed, drop_offset);
                }
            }
            else if constexpr(std::is_same_v<fmha_fwd_splitkv_args, std::decay_t<decltype(args)>>)
            {
                args.lse_acc_ptr = lse_acc_buf.GetDeviceBuffer();
                args.o_acc_ptr   = o_acc_buf.GetDeviceBuffer();

                args.block_table_ptr =
                    (0 < page_block_size ? block_table_buf.GetDeviceBuffer() : nullptr);
                args.batch_stride_block_table = batch_stride_block_table;
                args.page_block_size          = page_block_size;
                args.is_gappy = false; // use 'false' for flash-attention integration

                args.cache_batch_idx =
                    (use_cache_batch_idx ? cache_batch_idx_buf.GetDeviceBuffer() : nullptr);

                args.num_splits = num_splits;

                args.stride_o_acc         = stride_o_acc;
                args.nhead_stride_lse_acc = nhead_stride_lse_acc;
                args.nhead_stride_o_acc   = nhead_stride_o_acc;
                args.batch_stride_lse_acc = batch_stride_lse_acc;
                args.batch_stride_o_acc   = batch_stride_o_acc;
                args.split_stride_lse_acc = split_stride_lse_acc;
                args.split_stride_o_acc   = split_stride_o_acc;
            }
        }
    };

    const float appendkv_ave_time = [&] {
#if CK_TILE_FMHA_FWD_APPENDKV_API
        if(need_append_kvcache)
        {
            fmha_fwd_appendkv_traits fwd_appendkv_traits;
            init_traits(fwd_appendkv_traits);

            fmha_fwd_appendkv_args fwd_appendkv_args;
            init_args(fwd_appendkv_args);

            return fmha_fwd_appendkv(fwd_appendkv_traits, fwd_appendkv_args, stream_config);
        }
#endif
        return 0.0f;
    }();

    const float fwd_ave_time = [&] {
#if CK_TILE_FMHA_FWD_SPLITKV_API
        if(1 < num_splits || use_kvcache)
        {
            fmha_fwd_splitkv_traits fmha_splitkv_traits;
            init_traits(fmha_splitkv_traits);

            fmha_fwd_splitkv_args fmha_splitkv_args;
            init_args(fmha_splitkv_args);

            return fmha_fwd_splitkv(fmha_splitkv_traits, fmha_splitkv_args, stream_config);
        }
#endif
        fmha_fwd_traits fmha_traits;
        init_traits(fmha_traits);

        fmha_fwd_args fmha_args;
        init_args(fmha_args);

        return fmha_fwd(fmha_traits, fmha_args, stream_config);
    }();

    if(appendkv_ave_time < 0.0f || fwd_ave_time < 0.0f)
    {
        std::cout << ", not supported yet" << std::flush << std::endl;
        return false;
    }

    const float ave_time = (appendkv_ave_time + fwd_ave_time);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << std::fixed << ", " << std::setprecision(3) << ave_time << " ms, "
              << std::setprecision(2) << tflops << " TFlops, " << std::setprecision(2) << gb_per_sec
              << " GB/s" << std::flush;

    if(do_validation == 0)
    {
        std::cout << std::flush << std::endl;
        return true;
    }
    if(do_validation == 2)
    {
        // NOTE: use gpu to do validation
        ck_tile::naive_attention_fwd_traits naive_t;
        naive_t.q_type     = data_type;
        naive_t.k_type     = data_type;
        naive_t.v_type     = data_type;
        naive_t.o_type     = data_type;
        naive_t.q_layout   = i_perm == 1 ? "bhsd" : "bshd";
        naive_t.k_layout   = i_perm == 1 ? "bhsd" : "bshd";
        naive_t.v_layout   = i_perm == 1 ? "bhsd" : "bshd";
        naive_t.o_layout   = o_perm == 1 ? "bhsd" : "bshd";
        naive_t.variation  = 0; // TODO?
        naive_t.quant_algo = 0;

        ck_tile::DeviceMem o_naive_buf(o_host.get_element_space_size_in_bytes());

        ck_tile::naive_attention_fwd_args naive_a;
        naive_a.q_ptr           = q_buf.GetDeviceBuffer();
        naive_a.k_ptr           = k_buf.GetDeviceBuffer();
        naive_a.v_ptr           = v_buf.GetDeviceBuffer();
        naive_a.o_ptr           = o_naive_buf.GetDeviceBuffer();
        naive_a.scale_s         = scale_s;
        naive_a.context_len_ptr = nullptr; // used when seqlen kv come from a pointer
        naive_a.page_table_ptr =
            nullptr; // [batch, num_blocks] seqlen_kv is in different block(paged attn)
        naive_a.hdim           = hdim_q;
        naive_a.hdim_v         = hdim_v; // could be cross-attn, where V and Q/K hdim are different
        naive_a.batch_q        = batch;
        naive_a.batch_kv       = batch;
        naive_a.batch_ratio_kv = 1; // batch_q / batch_kv
        naive_a.seqlen_q       = seqlen_qs[0];
        naive_a.seqlen_kv = seqlen_ks[0]; // if context_len_ptr is not nullptr, ignore this field
        naive_a.nhead_q   = nhead;
        naive_a.nhead_kv  = nhead_k;
        naive_a.nhead_ratio_kv = naive_a.nhead_q / naive_a.nhead_kv; // nhead_q / nhead_kv
        naive_a.page_size      = 0; // if paged, the seqlen-kv for each block

        ck_tile::stream_config naive_s{};

        naive_attention_fwd(naive_t, naive_a, naive_s);

        auto o_naive_ref = o_naive_buf.ToHost<ODataType>();
        o_buf.FromDevice(o_host.data()); // TODO: ugly

        auto [rtol_, atol_] = get_elimit<DataTypeConfig>(init_method);
        bool pass_          = ck_tile::check_err(
            o_host, o_naive_ref, std::string("OUT Error: Incorrect results!"), rtol_, atol_);
        std::cout << ", valid:" << (pass_ ? "y" : "n") << std::flush << std::endl;
        return pass_;
    }

    o_buf.FromDevice(o_host.data());
    lse_buf.FromDevice(lse_host.data());
    randval_buf.FromDevice(randval_host.data());

    auto p_compute_element_func = [&]() {
        if constexpr(std::is_same_v<DataTypeConfig, ck_tile::fp8_t>)
            return ck_tile::scales{scale_p};
        else
            return ck_tile::identity{};
    }();

    auto oacc_element_func = [&]() {
        if constexpr(std::is_same_v<DataTypeConfig, ck_tile::fp8_t>)
            return ck_tile::composes(ck_tile::saturates<ck_tile::fp8_t>{},
                                     ck_tile::scales{scale_o});
        else
            return ck_tile::identity{};
    }();

    float p_undrop = 1.0 - p_drop;
    uint8_t p_undrop_in_uint8_t =
        uint8_t(std::floor(p_undrop * std::numeric_limits<uint8_t>::max()));
    float rp_undrop = 1.0 / p_undrop;

    bool pass = true;
    for(ck_tile::index_t wb = 0; wb < batch; ++wb)
    {
        const ck_tile::index_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
        const ck_tile::index_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

        // adjust matrix index according to the mode
        const ck_tile::index_t b_idx = (mode == mode_enum::batch ? wb : 0);
        const ck_tile::index_t cache_b_idx =
            (use_cache_batch_idx ? cache_batch_idx_host(b_idx) : b_idx);
        const ck_tile::index_t query_offset = (mode == mode_enum::batch ? 0 : seqstart_q_host[wb]);
        const ck_tile::index_t key_offset =
            (mode == mode_enum::batch
                 ? 0
                 : (seqlen_kpads[0] < 0 ? seqstart_k_host[wb] : seqstart_k_with_padding_host[wb]));

        ck_tile::HostTensor<QDataType> q_host_ref({nhead, real_seqlen_q, hdim_q});
        ck_tile::HostTensor<KDataType> k_host_ref({nhead, real_seqlen_k, hdim_q});
        ck_tile::HostTensor<VDataType> v_host_ref({nhead, hdim_v, real_seqlen_k});
        ck_tile::HostTensor<ODataType> o_host_ref({nhead, real_seqlen_q, hdim_v});

        ck_tile::HostTensor<SMPLComputeDataType> s_host_ref({nhead, real_seqlen_q, real_seqlen_k});
        ck_tile::HostTensor<PDataType> p_host_ref({nhead, real_seqlen_q, real_seqlen_k});
        ck_tile::HostTensor<SMPLComputeDataType> lse_host_ref({nhead, real_seqlen_q});

        ck_tile::index_t nr = nhead / nhead_k;

        // clang-format off
        // permute
        if(i_perm) q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b_idx, i[0], i[1] + query_offset, i[2]); });
        else       q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b_idx, i[1] + query_offset, i[0], i[2]); });

#if CK_TILE_FMHA_FWD_APPENDKV_API
        // optionally apply RoPE to the q_host_ref
        if(0 < rotary_dim)
        {
            decltype(q_host_ref) q_host_ref_ro(q_host_ref.get_lengths());

            auto [rotary_cos_slice, rotary_sin_slice] =
                slice_rotary_cos_sin(rotary_cos_host, rotary_sin_host, cache_seqlen_ks[wb], real_seqlen_q);

            ck_tile::reference_batched_rotary_position_embedding(
                q_host_ref, rotary_cos_slice, rotary_sin_slice, is_rotary_interleaved, q_host_ref_ro,
                /*use_1_row_sin_cos=*/mask.type == mask_enum::no_mask);

            q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host_ref_ro(i); });
        }
#endif
#if CK_TILE_FMHA_FWD_SPLITKV_API
        if(0 < page_block_size) {
            if(i_perm) {
                k_host_ref.ForEach([&](auto& self, auto i) {
                    self(i) = k_host(block_table_host(wb, i[1] / page_block_size), i[0] / nr, i[1] % page_block_size, i[2]);
                });
            } else {
                k_host_ref.ForEach([&](auto& self, auto i) {
                    self(i) = k_host(block_table_host(wb, i[1] / page_block_size), i[1] % page_block_size, i[0] / nr, i[2]);
                });
            }
        } else
#endif
        {
            if(i_perm) k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(cache_b_idx, i[0] / nr, i[1] + key_offset, i[2]); });
            else       k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(cache_b_idx, i[1] + key_offset, i[0] / nr, i[2]); });
        }

#if CK_TILE_FMHA_FWD_APPENDKV_API
        // copy Knew to the end of K
        if(0 < seqlen_knew)
        {
            ck_tile::HostTensor<KDataType> knew_host_ref({nhead, seqlen_knew, hdim_q});
            if(i_perm) knew_host_ref.ForEach([&](auto& self, auto i) { self(i) = knew_host(wb, i[0] / nr, i[1], i[2]); });
            else       knew_host_ref.ForEach([&](auto& self, auto i) { self(i) = knew_host(wb, i[1], i[0] / nr, i[2]); });

            // optionally apply RoPE to the knew_host_ref
            auto* real_knew_host_ref = &knew_host_ref;
            std::optional<decltype(knew_host_ref)> knew_host_ref_ro;
            if(0 < rotary_dim)
            {
                knew_host_ref_ro.emplace(knew_host_ref.get_lengths());

                auto [rotary_cos_slice, rotary_sin_slice] =
                    slice_rotary_cos_sin(rotary_cos_host, rotary_sin_host, cache_seqlen_ks[wb], seqlen_knew);

                ck_tile::reference_batched_rotary_position_embedding(
                    knew_host_ref,
                    rotary_cos_slice,
                    rotary_sin_slice,
                    is_rotary_interleaved,
                    knew_host_ref_ro.value());

                real_knew_host_ref = &knew_host_ref_ro.value();
            }

            (*real_knew_host_ref).ForEach([&](auto& self, auto i) {
                k_host_ref(i[0], i[1] + cache_seqlen_ks[wb], i[2]) = self(i);
            });
        }
#endif
#if CK_TILE_FMHA_FWD_SPLITKV_API
        if(0 < page_block_size) {
            if(is_v_rowmajor) {
                if(i_perm) {
                    v_host_ref.ForEach([&](auto& self, auto i) {
                        self(i) = v_host(block_table_host(wb, i[2] / page_block_size), i[0] / nr, i[2] % page_block_size, i[1]);
                    });
                } else {
                    v_host_ref.ForEach([&](auto& self, auto i) {
                        self(i) = v_host(block_table_host(wb, i[2] / page_block_size), i[2] % page_block_size, i[0] / nr, i[1]);
                    });
                }
            }
            else
            {
                if(i_perm) {
                    v_host_ref.ForEach([&](auto& self, auto i) {
                        self(i) = v_host(block_table_host(wb, i[2] / page_block_size), i[0] / nr, i[1], i[2] % page_block_size);
                    });
                } else {
                    v_host_ref.ForEach([&](auto& self, auto i) {
                        self(i) = v_host(block_table_host(wb, i[2] / page_block_size), i[1], i[0] / nr, i[2] % page_block_size);
                    });
                }
            }
        } else
#endif
        {
            if(is_v_rowmajor) {
                //                                                             v_host_ref: [nhead, hdim, seq], v_host: [b, h_k, s, d]
                if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(cache_b_idx, i[0] / nr, i[2] + key_offset, i[1]); });
                //                                                             v_host_ref: [nhead, hdim, seq], v_host: [b, s, h_k, d]
                else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(cache_b_idx, i[2] + key_offset, i[0] / nr, i[1]); });
            }
            else
            {
                if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(cache_b_idx, i[0] / nr, i[1], i[2] + key_offset); });
                else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(cache_b_idx, i[1], i[0] / nr, i[2] + key_offset); });
            }
        }

#if CK_TILE_FMHA_FWD_APPENDKV_API
        // copy Vnew to the end of V
        if(0 < seqlen_knew)
        {
            ck_tile::HostTensor<VDataType> vnew_host_ref({nhead, hdim_v, seqlen_knew});
            if(is_v_rowmajor)
            {
                if(i_perm) vnew_host_ref.ForEach([&](auto& self, auto i) { self(i) = vnew_host(wb, i[0] / nr, i[2], i[1]); });
                else       vnew_host_ref.ForEach([&](auto& self, auto i) { self(i) = vnew_host(wb, i[2], i[0] / nr, i[1]); });
            }
            else
            {
                if(i_perm) vnew_host_ref.ForEach([&](auto& self, auto i) { self(i) = vnew_host(wb, i[0] / nr, i[1], i[2]); });
                else       vnew_host_ref.ForEach([&](auto& self, auto i) { self(i) = vnew_host(wb, i[1], i[0] / nr, i[2]); });
            }

            vnew_host_ref.ForEach([&](auto& self, auto i) {
                v_host_ref(i[0], i[1], i[2] + cache_seqlen_ks[wb]) = self(i);
            });
        }
#endif
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

        if(p_drop > 0)
        {
            ck_tile::HostTensor<RandValOutputDataType> randval_host_ref(
                {nhead, real_seqlen_q, real_seqlen_k});
            randval_host_ref.ForEach([&](auto& self, auto idx) {
                self(idx) = randval_host(b_idx, idx[0], idx[1] + query_offset, idx[2]);
            });
            ck_tile::reference_batched_dropout(
                p_host_ref, randval_host_ref, p_undrop_in_uint8_t, rp_undrop);
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
        if(o_perm) o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b_idx, idx[0], idx[1] + query_offset, idx[2]); });
        else       o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b_idx, idx[1] + query_offset, idx[0], idx[2]); });
        // clang-format on

        auto [rtol, atol] = get_elimit<DataTypeConfig>(init_method);
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
                self(idx) = lse_host(b_idx, idx[0], idx[1] + query_offset);
            });

            cur_pass = ck_tile::check_err(lse_host_result,
                                          lse_host_ref,
                                          "LSE Error: Incorrect results!",
                                          rtol,
                                          atol,
                                          /* allow_infinity_ref = */ true);

            pass &= cur_pass;
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
        return run<FmhaFwdFp16>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "bf16")
    {
        return run<FmhaFwdBf16>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "fp8")
    {
        return run<FmhaFwdFp8>(arg_parser) ? 0 : -2;
    }

    return -3;
}
