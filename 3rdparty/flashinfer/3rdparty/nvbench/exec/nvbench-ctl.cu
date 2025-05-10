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

#include <nvbench/main.cuh>

#include <vector>

int main(int argc, char const *const *argv)
try
{
  // If no args, substitute a new argv that prints the version
  std::vector<const char*> alt_argv;
  if (argc == 1)
  {
    alt_argv.push_back("--version");
    alt_argv.push_back(nullptr);
    argv = alt_argv.data();
  }

  NVBENCH_MAIN_PARSE(argc, argv);
  NVBENCH_CUDA_CALL(cudaDeviceReset());
  return 0;
}
catch (std::exception & e)
{
  std::cerr << "\nNVBench encountered an error:\n\n" << e.what() << "\n";
  return 1;
}
catch (...)
{
  std::cerr << "\nNVBench encountered an unknown error.\n";
  return 1;
}
