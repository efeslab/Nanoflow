#include "ck/host/headers.hpp"
#include "ck_headers.hpp"

namespace ck {
namespace host {

const std::string config_header = "";

std::unordered_map<std::string_view, std::string_view> GetHeaders()
{
    auto headers = ck_headers();
    headers.insert(std::make_pair("ck/config.h", config_header));
    return headers;
}

} // namespace host
} // namespace ck