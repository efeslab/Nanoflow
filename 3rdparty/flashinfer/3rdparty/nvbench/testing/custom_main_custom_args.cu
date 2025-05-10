/*
 *  Copyright 2024 NVIDIA Corporation
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
#include "nvbench/cuda_call.cuh"

/******************************************************************************
 * Install custom parser.
 * sSee <nvbench/main.cuh> for more details.
 ******************************************************************************/

//
// Step 1: Define a custom argument handler that accepts a vector of strings.
//          - This handler should modify the vector in place to remove any custom
//            arguments it handles. NVbench will then parse the remaining arguments.
//          - The handler should also update any application state needed to handle
//            the custom arguments.
//

// User code to handle a specific argument:
void handle_my_custom_arg();

// NVBench hook for modiifying the command line arguments before parsing:
void custom_arg_handler(std::vector<std::string> &args)
{
  // Handle and remove "--my-custom-arg"
  if (auto it = std::find(args.begin(), args.end(), "--my-custom-arg"); it != args.end())
  {
    handle_my_custom_arg();
    args.erase(it);
  }
}

//
// Step 2: Install the custom argument handler.
//         - This is done by defining a macro that invokes the custom argument handler.
//

// Install the custom argument handler:
// Either define this before any NVBench headers are included, or undefine and redefine:
#undef NVBENCH_MAIN_CUSTOM_ARGS_HANDLER
#define NVBENCH_MAIN_CUSTOM_ARGS_HANDLER(args) custom_arg_handler(args)

// Step 3: Define `main`
//
// After installing the custom argument handler, define the main function using:
//
// ```
// NVBENCH_MAIN
// ```
//
// Here, this is done at the end of this file.

/******************************************************************************
 * Unit test verification:
 ******************************************************************************/

// Track whether the args are found / handled.
bool h_custom_arg_found             = false;
bool h_handled_on_device            = false;
__device__ bool d_custom_arg_found  = false;
__device__ bool d_handled_on_device = false;

// Copy host values to device:
void copy_host_state_to_device()
{
  NVBENCH_CUDA_CALL(cudaMemcpyToSymbol(d_custom_arg_found, &h_custom_arg_found, sizeof(bool)));
  NVBENCH_CUDA_CALL(cudaMemcpyToSymbol(d_handled_on_device, &h_handled_on_device, sizeof(bool)));
}

// Copy device values to host:
void copy_device_state_to_host()
{
  NVBENCH_CUDA_CALL(cudaMemcpyFromSymbol(&h_custom_arg_found, d_custom_arg_found, sizeof(bool)));
  NVBENCH_CUDA_CALL(cudaMemcpyFromSymbol(&h_handled_on_device, d_handled_on_device, sizeof(bool)));
}

void handle_my_custom_arg()
{
  h_custom_arg_found = true;
  copy_host_state_to_device();
}

void verify()
{
  copy_device_state_to_host();
  if (!h_custom_arg_found)
  {
    throw std::runtime_error("Custom argument not detected.");
  }
  if (!h_handled_on_device)
  {
    throw std::runtime_error("Custom argument not handled on device.");
  }
}

// Install a verification check to ensure the custom argument was handled.
// Use the `PRE` finalize hook to ensure we check device state before resetting the context.
#undef NVBENCH_MAIN_FINALIZE_CUSTOM_PRE
#define NVBENCH_MAIN_FINALIZE_CUSTOM_PRE() verify()

// Simple kernel/benchmark to make sure that the handler can successfully modify CUDA state:
__global__ void kernel()
{
  if (d_custom_arg_found)
  {
    d_handled_on_device = true;
  }
}
void bench(nvbench::state &state)
{
  state.exec([](nvbench::launch &) { kernel<<<1, 1>>>(); });
}
NVBENCH_BENCH(bench);

// Define the customized main function:
NVBENCH_MAIN
