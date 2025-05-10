// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#ifndef CK_CODE_GEN_RTC
#include <string>
#include <sstream>
#include <regex>
#include <optional>
#include "ck/stream_config.hpp"
#endif

namespace ck {
namespace tensor_operation {
namespace device {

#ifndef CK_CODE_GEN_RTC
#define GET_OBJECT_NAME_IMLP                                                  \
    std::optional<std::string> GetObjectName() const override                 \
    {                                                                         \
        std::string str = __PRETTY_FUNCTION__;                                \
        static std::regex obj_name_expr{"<std::string> (.*)::GetObjectName"}; \
        std::smatch match;                                                    \
        if(!std::regex_search(str, match, obj_name_expr))                     \
        {                                                                     \
            return str;                                                       \
        }                                                                     \
        return std::string(match[1]) + ';';                                   \
    }

#define GET_TEMPLATE_INFO_IMPL                                  \
    std::optional<std::string> GetTemplateInfo() const override \
    {                                                           \
        std::string str = __PRETTY_FUNCTION__;                  \
        static std::regex template_expr{"\\[(.*)\\]"};          \
        std::smatch match;                                      \
        if(!std::regex_search(str, match, template_expr))       \
        {                                                       \
            return std::nullopt;                                \
        }                                                       \
        return std::string(match[1]);                           \
    }

#define REGISTER_EXTRA_PRINTING_METHODS GET_OBJECT_NAME_IMLP GET_TEMPLATE_INFO_IMPL
#endif

#ifndef CK_CODE_GEN_RTC
struct BaseArgument
{
    BaseArgument()                    = default;
    BaseArgument(const BaseArgument&) = default;
    BaseArgument& operator=(const BaseArgument&) = default;

    virtual ~BaseArgument() {}

    void* p_workspace_ = nullptr;
};

struct BaseInvoker
{
    BaseInvoker()                   = default;
    BaseInvoker(const BaseInvoker&) = default;
    BaseInvoker& operator=(const BaseInvoker&) = default;

    virtual float Run(const BaseArgument*, const StreamConfig& = StreamConfig{})
    {
        return float{0};
    }

    virtual ~BaseInvoker() {}
};
#endif

struct BaseOperator
{
    BaseOperator()                    = default;
    BaseOperator(const BaseOperator&) = default;
    BaseOperator& operator=(const BaseOperator&) = default;
#ifndef CK_CODE_GEN_RTC
    virtual bool IsSupportedArgument(const BaseArgument*) { return false; }
    virtual std::string GetTypeString() const { return ""; }

    virtual std::string GetTypeIdName() const { return typeid(*this).name(); }

    virtual std::optional<std::string> GetObjectName() const { return std::nullopt; }

    virtual std::optional<std::string> GetTemplateInfo() const { return std::nullopt; }

    virtual std::string GetTypeIdHashCode() const
    {
        std::ostringstream oss;

        oss << std::hex << typeid(*this).hash_code();

        return oss.str();
    };

    virtual size_t GetWorkSpaceSize(const BaseArgument*) const { return 0; }

    virtual void SetWorkSpacePointer(BaseArgument* p_arg,
                                     void* p_workspace,
                                     const StreamConfig& = StreamConfig{}) const
    {
        assert(p_arg);
        p_arg->p_workspace_ = p_workspace;
    }
#endif
    virtual ~BaseOperator() {}
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
