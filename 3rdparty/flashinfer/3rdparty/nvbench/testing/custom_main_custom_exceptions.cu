/*
 *  Copyright 2022 NVIDIA Corporation
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

#include <nvbench/nvbench.cuh>

#include <stdexcept>

/******************************************************************************
 * Install exception handler around the NVBench main body. This is used
 * to print helpful information when a user exception is thrown before exiting.
 *
 * Note that this will **NOT** be used when a benchmark throws an exception.
 * That will fail the benchmark and note the exception, and continue
 * execution.
 *
 * This is used to catch exceptions in user extensions of NVBench, things like
 * customized initialization, command line parsing, finalization, etc. See
 * <nvbench/main.cuh> for more details.
 ******************************************************************************/

struct user_exception : public std::runtime_error
{
  user_exception()
      : std::runtime_error("Expected exception thrown.")
  {}
};

// User code to handle user exception:
void handle_my_exception(user_exception &e)
{
  std::cerr << "Custom error detected: " << e.what() << std::endl;
  std::exit(1);
}

// Install the exception handler around the NVBench main body.
// NVBench will have sensible defaults for common exceptions following this if no terminating catch
// block is defined.
// Either define this before any NVBench headers are included, or undefine and redefine.
#undef NVBENCH_MAIN_CATCH_EXCEPTIONS_CUSTOM
#define NVBENCH_MAIN_CATCH_EXCEPTIONS_CUSTOM                                                       \
  catch (user_exception & e) { handle_my_exception(e); }

// For testing purposes, install a argument parser that throws:
void really_robust_argument_parser(std::vector<std::string> &) { throw user_exception(); }
#undef NVBENCH_MAIN_CUSTOM_ARGS_HANDLER
#define NVBENCH_MAIN_CUSTOM_ARGS_HANDLER(args) really_robust_argument_parser(args);

// Define the customized main function:
NVBENCH_MAIN
