// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>
#include <functional>
#include <string>

#include "ck_tile/core/container/span.hpp"

enum class mode_enum
{
    batch = 0,
    group
};

std::ostream& operator<<(std::ostream& stream, mode_enum mode)
{
    return stream << (mode == mode_enum::batch ? "batch" : "group");
}

std::vector<int32_t> to_seqstarts(ck_tile::span<const int32_t> seqlens)
{
    std::vector<int32_t> seqstarts = {0};
    for(int32_t seqlen : seqlens)
    {
        seqstarts.push_back(seqstarts.back() + seqlen);
    }
    assert(seqstarts.size() == seqlens.size() + 1);
    return seqstarts;
}

std::vector<int32_t> generate_seqlens(mode_enum mode,
                                      unsigned count,
                                      int32_t seqlen_avg,
                                      int32_t seqlen_max = -1, // if not negative, clamp max
                                      std::optional<unsigned> seed = std::nullopt)
{
    assert(0 < count);

    std::vector<int32_t> seqlens(
        count, seqlen_max > 0 ? (seqlen_avg < seqlen_max ? seqlen_avg : seqlen_max) : seqlen_avg);

    if(mode == mode_enum::group && 1 < count)
    {
        using size_type = std::vector<int32_t>::size_type;

        std::mt19937 random_engine(seed.has_value() ? *seed : std::random_device{}());
        std::uniform_int_distribution<size_type> idx_dist(0, count - 1);
        auto next_idx = std::bind(idx_dist, std::ref(random_engine));

        std::uniform_int_distribution<size_type> step_dist(1, count - 1);
        auto next_step = std::bind(step_dist, std::ref(random_engine));

        for(unsigned repeat = seqlen_avg * (count / 2); 0 < repeat; --repeat)
        {
            const size_type to_decrease = next_idx();
            // make sure each elements of seqlens is always greater than 0
            if(seqlens[to_decrease] == 1)
            {
                continue;
            }

            const size_type to_increase = (to_decrease + next_step()) % count;

            if(seqlen_max > 0 && seqlens[to_increase] >= seqlen_max)
            {
                continue;
            }

            --seqlens[to_decrease];
            ++seqlens[to_increase];
        }
    }

    return seqlens;
}

std::vector<int32_t> generate_seqstarts(mode_enum mode,
                                        unsigned count,
                                        int32_t seqlen_avg,
                                        int32_t seqlen_max           = -1,
                                        std::optional<unsigned> seed = std::nullopt)
{
    return to_seqstarts(generate_seqlens(mode, count, seqlen_avg, seqlen_max, seed));
}

/*
 * decode the seqlen string from cmdline
 * example (assume batch=3)
 *   q_val=1,2,3 k_val=4,5,6 -> OK
 *   q_val=1,2,3             -> OK, k same as q
 *   q_val=1,2               -> OK, q will rand remaining 1 element, k same as q
 *   q_val=1,2   k_val=4,5   -> OK, q/k will rand remaining 1 element
 *   q_val=1,2,3,4           -> OK, but ignore exceed one
 *
 *   q_val=1,2   k_val=4,5,6 -> not OK, k must have same splits with q
 *   q_val=1,2   k_val=4     -> not OK, k must have same splits with q
 */
std::tuple<std::vector<ck_tile::index_t>,
           std::vector<ck_tile::index_t>,
           std::vector<ck_tile::index_t>>
decode_seqlen(mode_enum mode,
              ck_tile::index_t batch,
              std::string q_val,
              std::string k_val,
              std::string k_pad_val,
              std::optional<unsigned> seed = std::nullopt)
{
#define _S2I_(str_) static_cast<ck_tile::index_t>(std::atoi((str_).c_str()))
    if(mode == mode_enum::batch)
    {
        ck_tile::index_t q = _S2I_(q_val);
        ck_tile::index_t k = _S2I_(k_val);
        auto s_q           = std::vector<ck_tile::index_t>(batch, q);
        auto s_k           = std::vector<ck_tile::index_t>(batch, k < 0 ? q : k);
        auto s_kpad = std::vector<ck_tile::index_t>(batch, -1); // TODO: batch not support k_padding
        return std::make_tuple(s_q, s_k, s_kpad);
    }
    else
    {
        ck_tile::index_t idx          = 0;
        std::string::size_type pos_q  = 0;
        std::string::size_type pos_k  = 0;
        std::string::size_type pos_kp = 0;
        std::vector<ck_tile::index_t> s_q;
        std::vector<ck_tile::index_t> s_k;
        std::vector<ck_tile::index_t> s_kpad;
        while(true)
        {
            auto found_q  = q_val.find(',', pos_q);
            auto found_k  = k_val.find(',', pos_k);
            auto found_kp = k_pad_val.find(',', pos_kp);

            ck_tile::index_t q = _S2I_(
                q_val.substr(pos_q, found_q == std::string::npos ? found_q : found_q - pos_q));
            ck_tile::index_t k = _S2I_(
                k_val.substr(pos_k, found_k == std::string::npos ? found_k : found_k - pos_k));
            ck_tile::index_t kp = _S2I_(k_pad_val.substr(
                pos_kp, found_kp == std::string::npos ? found_kp : found_kp - pos_kp));

            s_q.push_back(q);
            s_k.push_back(k < 0 ? q : k);
            s_kpad.push_back(kp);
            idx++;
            if(found_q == std::string::npos || idx >= batch)
            {
                break;
            }
            pos_q  = found_q + 1;
            pos_k  = found_k == std::string::npos ? pos_k : found_k + 1;
            pos_kp = found_kp == std::string::npos ? pos_kp : found_kp + 1;
        }
        if(idx < batch)
        {
            auto rem_q = generate_seqlens(mode, batch - idx, s_q.back(), s_kpad.back(), seed);
            auto rem_k = generate_seqlens(mode, batch - idx, s_k.back(), s_kpad.back(), seed);

            s_q.insert(s_q.end(), rem_q.begin(), rem_q.end());
            s_k.insert(s_k.end(), rem_k.begin(), rem_k.end());
            s_kpad.insert(s_kpad.end(), batch - idx, s_kpad.back());
        }
        return std::make_tuple(s_q, s_k, s_kpad);
    }
#undef _S2I_
}

int env_get_int(const char* var_name, int default_int)
{
    char* v = getenv(var_name);
    int r   = default_int;
    if(v)
        r = std::atoi(v);
    return r;
}
