#include "ck/host/types.hpp"
#include "ck/host/stringutils.hpp"
#include <algorithm>
#include <stdexcept>

namespace ck {
namespace host {

Solution::Solution(std::string str, std::unordered_map<std::string, std::string> values)
    : template_str(std::move(str)), template_values(std::move(values))
{
}

std::string Solution::ToTemplateString() const { return this->template_str; }
std::string Solution::GetTemplateParameter(const std::string& name) const
{
    return this->template_values.at(name);
}

std::string ToString(DataType dt)
{
    switch(dt)
    {
    case DataType::Float: return "float";
    case DataType::Half: return "ck::half_t";
    case DataType::Int8: return "int8_t";
    case DataType::Int32: return "int32_t";
    }
    throw std::runtime_error("Incorrect data type");
}

std::string ToString(Layout dl)
{
    switch(dl)
    {
    case Layout::Row: return "ck::tensor_layout::gemm::RowMajor";
    case Layout::Column: return "ck::tensor_layout::gemm::ColumnMajor";
    }
    throw std::runtime_error("Incorrect layout");
}

std::string ToString(GemmType gt)
{
    switch(gt)
    {
    case GemmType::Default: return "ck::tensor_operation::device::GemmSpecialization::Default";
    }
    throw std::runtime_error("Incorrect gemm type");
}

std::string SequenceStr(const std::vector<int>& v)
{
    return "ck::Sequence<" +
           JoinStrings(Transform(v, [](int x) { return std::to_string(x); }), ", ") + ">";
}

std::string MakeTuple(const std::vector<std::string>& v)
{
    return "ck::Tuple<" + JoinStrings(v, ", ") + ">";
}

} // namespace host
} // namespace ck
