/*
 *  Copyright 2021 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 with the LLVM exception
 *  (the "License"); you may not use this file except in compliance with
 *  the License.
 *
 *  You may obtain a copy of the License at
 *
 *      http://llvm.org/foundation/relicensing/LICENSE.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <nvbench/benchmark_base.cuh>
#include <nvbench/benchmark_manager.cuh>
#include <nvbench/config.cuh>
#include <nvbench/cuda_call.cuh>
#include <nvbench/option_parser.cuh>
#include <nvbench/printer_base.cuh>

#include <cstdlib>
#include <iostream>

// Advanced users can rebuild NVBench's `main` function using the macros in this file, or replace
// them with customized implementations.

// Customization point, called before NVBench initialization.
#ifndef NVBENCH_MAIN_INITIALIZE_CUSTOM_PRE
#define NVBENCH_MAIN_INITIALIZE_CUSTOM_PRE(argc, argv) []() {}()
#endif

// Customization point, called after NVBench initialization.
#ifndef NVBENCH_MAIN_INITIALIZE_CUSTOM_POST
#define NVBENCH_MAIN_INITIALIZE_CUSTOM_POST(argc, argv) []() {}()
#endif

// Customization point, called before NVBench parsing. Update argc/argv if needed.
// argc/argv are the usual command line arguments types. The ARGS version of this
// macro is a bit more convenient.
#ifndef NVBENCH_MAIN_CUSTOM_ARGC_ARGV_HANDLER
#define NVBENCH_MAIN_CUSTOM_ARGC_ARGV_HANDLER(argc, argv) []() {}()
#endif

// Customization point, called before NVBench parsing. Update args if needed.
// Args is a vector of strings, each element is an argument.
#ifndef NVBENCH_MAIN_CUSTOM_ARGS_HANDLER
#define NVBENCH_MAIN_CUSTOM_ARGS_HANDLER(args) []() {}()
#endif

// Customization point, called before NVBench parsing.
#ifndef NVBENCH_MAIN_PARSE_CUSTOM_PRE
#define NVBENCH_MAIN_PARSE_CUSTOM_PRE(parser, args) []() {}()
#endif

// Customization point, called after NVBench parsing.
#ifndef NVBENCH_MAIN_PARSE_CUSTOM_POST
#define NVBENCH_MAIN_PARSE_CUSTOM_POST(parser) []() {}()
#endif

// Customization point, called before NVBench finalization.
#ifndef NVBENCH_MAIN_FINALIZE_CUSTOM_PRE
#define NVBENCH_MAIN_FINALIZE_CUSTOM_PRE() []() {}()
#endif

// Customization point, called after NVBench finalization.
#ifndef NVBENCH_MAIN_FINALIZE_CUSTOM_POST
#define NVBENCH_MAIN_FINALIZE_CUSTOM_POST() []() {}()
#endif

// Customization point, use to catch addition exceptions.
#ifndef NVBENCH_MAIN_CATCH_EXCEPTIONS_CUSTOM
#define NVBENCH_MAIN_CATCH_EXCEPTIONS_CUSTOM
#endif

/************************************ Default implementation **************************************/

#ifndef NVBENCH_MAIN
#define NVBENCH_MAIN                                                                               \
  int main(int argc, char **argv)                                                                  \
  try                                                                                              \
  {                                                                                                \
    NVBENCH_MAIN_BODY(argc, argv);                                                                 \
    return 0;                                                                                      \
  }                                                                                                \
  NVBENCH_MAIN_CATCH_EXCEPTIONS_CUSTOM                                                             \
  NVBENCH_MAIN_CATCH_EXCEPTIONS
#endif

#ifndef NVBENCH_MAIN_BODY
#define NVBENCH_MAIN_BODY(argc, argv)                                                              \
  NVBENCH_MAIN_INITIALIZE(argc, argv);                                                             \
  {                                                                                                \
    NVBENCH_MAIN_PARSE(argc, argv);                                                                \
                                                                                                   \
    NVBENCH_MAIN_PRINT_PREAMBLE(parser);                                                           \
    NVBENCH_MAIN_RUN_BENCHMARKS(parser);                                                           \
    NVBENCH_MAIN_PRINT_EPILOGUE(parser);                                                           \
                                                                                                   \
    NVBENCH_MAIN_PRINT_RESULTS(parser);                                                            \
  } /* Tear down parser before finalization */                                                     \
  NVBENCH_MAIN_FINALIZE();                                                                         \
  return 0;
#endif

#ifndef NVBENCH_MAIN_INITIALIZE
#define NVBENCH_MAIN_INITIALIZE(argc, argv)                                                        \
  { /* Open a scope to ensure that the inner initialize/finalize hooks clean up in order. */       \
    NVBENCH_MAIN_INITIALIZE_CUSTOM_PRE(argc, argv);                                                \
    nvbench::detail::main_initialize(argc, argv);                                                  \
    { /* Open a scope to ensure that the inner initialize/finalize hooks clean up in order. */     \
      NVBENCH_MAIN_INITIALIZE_CUSTOM_POST(argc, argv)
#endif

#ifndef NVBENCH_MAIN_PARSE
#define NVBENCH_MAIN_PARSE(argc, argv)                                                             \
  NVBENCH_MAIN_CUSTOM_ARGC_ARGV_HANDLER(argc, argv);                                               \
  std::vector<std::string> args = nvbench::detail::main_convert_args(argc, argv);                  \
  NVBENCH_MAIN_CUSTOM_ARGS_HANDLER(args);                                                          \
  nvbench::option_parser parser;                                                                   \
  NVBENCH_MAIN_PARSE_CUSTOM_PRE(parser, args);                                                     \
  parser.parse(args);                                                                              \
  NVBENCH_MAIN_PARSE_CUSTOM_POST(parser)
#endif

#ifndef NVBENCH_MAIN_PRINT_PREAMBLE
#define NVBENCH_MAIN_PRINT_PREAMBLE(parser) nvbench::detail::main_print_preamble(parser)
#endif

#ifndef NVBENCH_MAIN_RUN_BENCHMARKS
#define NVBENCH_MAIN_RUN_BENCHMARKS(parser) nvbench::detail::main_run_benchmarks(parser)
#endif

#ifndef NVBENCH_MAIN_PRINT_EPILOGUE
#define NVBENCH_MAIN_PRINT_EPILOGUE(parser) nvbench::detail::main_print_epilogue(parser)
#endif

#ifndef NVBENCH_MAIN_PRINT_RESULTS
#define NVBENCH_MAIN_PRINT_RESULTS(parser) nvbench::detail::main_print_results(parser)
#endif

#ifndef NVBENCH_MAIN_FINALIZE
#define NVBENCH_MAIN_FINALIZE()                                                                    \
  NVBENCH_MAIN_FINALIZE_CUSTOM_PRE();                                                              \
  } /* Close a scope to ensure that the inner initialize/finalize hooks clean up in order. */      \
  nvbench::detail::main_finalize();                                                                \
  NVBENCH_MAIN_FINALIZE_CUSTOM_POST();                                                             \
  } /* Close a scope to ensure that the inner initialize/finalize hooks clean up in order. */      \
  []() {}()
#endif

#ifndef NVBENCH_MAIN_CATCH_EXCEPTIONS
#define NVBENCH_MAIN_CATCH_EXCEPTIONS                                                              \
  catch (std::exception & e)                                                                       \
  {                                                                                                \
    std::cerr << "\nNVBench encountered an error:\n\n" << e.what() << "\n";                        \
    return 1;                                                                                      \
  }                                                                                                \
  catch (...)                                                                                      \
  {                                                                                                \
    std::cerr << "\nNVBench encountered an unknown error.\n";                                      \
    return 1;                                                                                      \
  }
#endif

namespace nvbench::detail
{

inline void set_env(const char *name, const char *value)
{
#ifdef _MSC_VER
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

inline void main_initialize(int, char **)
{
  // See NVIDIA/NVBench#136 for CUDA_MODULE_LOADING
  set_env("CUDA_MODULE_LOADING", "EAGER");

  // Initialize CUDA driver API if needed:
#ifdef NVBENCH_HAS_CUPTI
  NVBENCH_DRIVER_API_CALL(cuInit(0));
#endif

  // Initialize the benchmarks *after* setting up the CUDA environment:
  nvbench::benchmark_manager::get().initialize();
}

inline std::vector<std::string> main_convert_args(int argc, char const *const *argv)
{
  std::vector<std::string> args;
  for (int i = 0; i < argc; ++i)
  {
    args.push_back(argv[i]);
  }
  return args;
}

inline void main_print_preamble(option_parser &parser)
{
  auto &printer = parser.get_printer();

  printer.print_device_info();
  printer.print_log_preamble();
}

inline void main_run_benchmarks(option_parser &parser)
{
  auto &printer    = parser.get_printer();
  auto &benchmarks = parser.get_benchmarks();

  std::size_t total_states = 0;
  for (auto &bench_ptr : benchmarks)
  {
    total_states += bench_ptr->get_config_count();
  }

  printer.set_completed_state_count(0);
  printer.set_total_state_count(total_states);

  for (auto &bench_ptr : benchmarks)
  {
    bench_ptr->set_printer(printer);
    bench_ptr->run();
    bench_ptr->clear_printer();
  }
}

inline void main_print_epilogue(option_parser &parser)
{
  auto &printer = parser.get_printer();
  printer.print_log_epilogue();
}

inline void main_print_results(option_parser &parser)
{
  auto &printer    = parser.get_printer();
  auto &benchmarks = parser.get_benchmarks();
  printer.print_benchmark_results(benchmarks);
}

inline void main_finalize() { NVBENCH_CUDA_CALL(cudaDeviceReset()); }

} // namespace nvbench::detail
