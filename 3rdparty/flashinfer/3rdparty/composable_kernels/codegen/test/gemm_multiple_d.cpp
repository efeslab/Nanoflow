#include "ck/host/device_gemm_multiple_d/problem.hpp"
#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/headers.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>
#include <test.hpp>
#include <rtc/compile_kernel.hpp>
#include <rtc/hip.hpp>

using half = _Float16;
// using half = __fp16;

std::vector<rtc::src_file> get_headers_for_test()
{
    std::vector<rtc::src_file> result;
    auto hs = ck::host::GetHeaders();
    std::transform(
        hs.begin(), hs.end(), std::back_inserter(result), [&](const auto& p) -> rtc::src_file {
            return {p.first, p.second};
        });
    return result;
}

template <class T>
rtc::buffer<T> generate_buffer(std::size_t n, std::size_t seed = 0)
{
    rtc::buffer<T> result(n);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(-1.0);
    std::generate(result.begin(), result.end(), [&] { return dis(gen); });
    return result;
}

template <class T, class U>
bool allclose(const T& a, const U& b, double atol = 0.01, double rtol = 0.01)
{
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [&](double x, double y) {
        return fabs(x - y) < atol + rtol * fabs(y);
    });
}

std::string classify(double x)
{
    switch(std::fpclassify(x))
    {
    case FP_INFINITE: return "inf";
    case FP_NAN: return "nan";
    case FP_NORMAL: return "normal";
    case FP_SUBNORMAL: return "subnormal";
    case FP_ZERO: return "zero";
    default: return "unknown";
    }
}

template <class Buffer>
void print_classification(const Buffer& x)
{
    std::unordered_set<std::string> result;
    for(const auto& i : x)
        result.insert(classify(i));
    for(const auto& c : result)
        std::cout << c << ", ";
    std::cout << std::endl;
}

template <class Buffer>
void print_statistics(const Buffer& x)
{
    std::cout << "Min value: " << *std::min_element(x.begin(), x.end()) << ", ";
    std::cout << "Max value: " << *std::max_element(x.begin(), x.end()) << ", ";
    double num_elements = x.size();
    auto mean =
        std::accumulate(x.begin(), x.end(), double{0.0}, std::plus<double>{}) / num_elements;
    auto stddev = std::sqrt(
        std::accumulate(x.begin(),
                        x.end(),
                        double{0.0},
                        [&](double r, double v) { return r + std::pow((v - mean), 2.0); }) /
        num_elements);
    std::cout << "Mean: " << mean << ", ";
    std::cout << "StdDev: " << stddev << "\n";
}

template <class Buffer>
void print_preview(const Buffer& x)
{
    if(x.size() <= 10)
    {
        std::for_each(x.begin(), x.end(), [&](double i) { std::cout << i << ", "; });
    }
    else
    {
        std::for_each(x.begin(), x.begin() + 5, [&](double i) { std::cout << i << ", "; });
        std::cout << "..., ";
        std::for_each(x.end() - 5, x.end(), [&](double i) { std::cout << i << ", "; });
    }
    std::cout << std::endl;
}

template <class T>
struct check_all
{
    rtc::buffer<T> data{};
    bool operator()(const rtc::buffer<T>& x)
    {
        if(data.empty())
        {
            data = x;
            return true;
        }
        if(std::any_of(x.begin(), x.end(), [](double y) { return std::isnan(y); }))
            return false;
        return allclose(data, x);
    }
};

template <class Solution>
auto report(const Solution& solution, bool pass)
{
    return test::make_predicate(solution.ToTemplateString(), [=] { return pass; });
}

const std::string gemm_compile_check = R"__ck__(
#include <${include}>

extern "C" __global__ void f(const ck::half_t* a, const ck::half_t* b, ck::half_t* c) {
    using G = ${template};
    constexpr auto desc = ${template}::make_descriptor(ck::make_naive_tensor_descriptor_packed(ck::make_tuple(${m}, ${k})),
                                             ck::make_naive_tensor_descriptor(ck::make_tuple(${n}, ${k}), ck::make_tuple(1, ${n})),
                                             ck::make_tuple(),
                                             ck::make_naive_tensor_descriptor_packed(ck::make_tuple(${m}, ${n})));

    static_assert(desc.IsValid(), "Invalid ck gemm.");

    if constexpr(desc.IsValid())
    {
        ${template}::Run(desc,
               a,
               b,
               ck::make_tuple(),
               c);
    }
}

)__ck__";

TEST_CASE(test_problem_kernel)
{
    ck::host::device_gemm_multiple_d::Problem prob;
    prob.M = 1024;
    prob.N = 1024;
    prob.K = 1024;
    check_all<half> check;
    auto a = to_gpu(generate_buffer<half>(1024 * 1024, 0));
    auto b = to_gpu(generate_buffer<half>(1024 * 1024, 1));
    auto c = to_gpu(generate_buffer<half>(1024 * 1024, 2));

    for(auto solution : prob.GetSolutions("gfx90a"))
    {
        auto src  = ck::host::InterpolateString(gemm_compile_check,
                                               {{"include", prob.GetIncludeHeader()},
                                                {"template", solution.ToTemplateString()},
                                                {"m", std::to_string(prob.M)},
                                                {"n", std::to_string(prob.N)},
                                                {"k", std::to_string(prob.K)}});
        auto srcs = get_headers_for_test();
        srcs.push_back({"main.cpp", src});
        rtc::compile_options options;
        options.kernel_name = "f";
        auto k              = rtc::compile_kernel(srcs, options);
        auto block_size     = solution.GetTemplateParameter<std::size_t>("BlockSize");
        auto m_per_block    = solution.GetTemplateParameter<std::size_t>("MPerBlock");
        auto n_per_block    = solution.GetTemplateParameter<std::size_t>("NPerBlock");
        auto grid_size      = ck::host::integer_divide_ceil(prob.M, m_per_block) *
                         ck::host::integer_divide_ceil(prob.N, n_per_block);
        k.launch(nullptr, grid_size * block_size, block_size)(a.data(), b.data(), c.data());
        CHECK(report(solution, check(rtc::from_gpu(c))));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
