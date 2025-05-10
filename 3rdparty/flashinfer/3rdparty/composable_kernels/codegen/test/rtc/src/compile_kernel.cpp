#include "rtc/hip.hpp"
#include <rtc/compile_kernel.hpp>
#include <rtc/tmp_dir.hpp>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cassert>

namespace rtc {

template <class T>
T generic_read_file(const std::string& filename, size_t offset = 0, size_t nbytes = 0)
{
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    if(nbytes == 0)
    {
        // if there is a non-zero offset and nbytes is not set,
        // calculate size of remaining bytes to read
        nbytes = is.tellg();
        if(offset > nbytes)
            throw std::runtime_error("offset is larger than file size");
        nbytes -= offset;
    }
    if(nbytes < 1)
        throw std::runtime_error("Invalid size for: " + filename);
    is.seekg(offset, std::ios::beg);

    T buffer(nbytes, 0);
    if(not is.read(&buffer[0], nbytes))
        throw std::runtime_error("Error reading file: " + filename);
    return buffer;
}

std::vector<char> read_buffer(const std::string& filename, size_t offset = 0, size_t nbytes = 0)
{
    return generic_read_file<std::vector<char>>(filename, offset, nbytes);
}

std::string read_string(const std::string& filename)
{
    return generic_read_file<std::string>(filename);
}

void write_buffer(const std::string& filename, const char* buffer, std::size_t size)
{
    std::ofstream os(filename);
    os.write(buffer, size);
}
void write_buffer(const std::string& filename, const std::vector<char>& buffer)
{
    write_buffer(filename, buffer.data(), buffer.size());
}
void write_string(const std::string& filename, const std::string_view& buffer)
{
    write_buffer(filename, buffer.data(), buffer.size());
}

std::string compiler() { return "/opt/rocm/llvm/bin/clang++ -x hip --cuda-device-only"; }

kernel compile_kernel(const std::vector<src_file>& srcs, compile_options options)
{
    assert(not srcs.empty());
    tmp_dir td{"compile"};
    options.flags += " -I. -O3";
    options.flags += " -std=c++17";
    options.flags += " --offload-arch=" + get_device_name();
    std::string out;

    for(const auto& src : srcs)
    {
        std::filesystem::path full_path   = td.path / src.path;
        std::filesystem::path parent_path = full_path.parent_path();
        std::filesystem::create_directories(parent_path);
        write_string(full_path.string(), src.content);
        if(src.path.extension().string() == ".cpp")
        {
            options.flags += " -c " + src.path.filename().string();
            if(out.empty())
                out = src.path.stem().string() + ".o";
        }
    }

    options.flags += " -o " + out;
    td.execute(compiler() + options.flags);

    auto out_path = td.path / out;
    if(not std::filesystem::exists(out_path))
        throw std::runtime_error("Output file missing: " + out);

    auto obj = read_buffer(out_path.string());

    return kernel{obj.data(), options.kernel_name};
}

} // namespace rtc
