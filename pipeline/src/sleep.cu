#include <cuda.h>
#include <chrono>
#include <cuda/std/chrono>

__global__ void cudaSleep(int us) {
    auto start = cuda::std::chrono::high_resolution_clock::now();
    while (cuda::std::chrono::duration_cast<cuda::std::chrono::microseconds>(cuda::std::chrono::high_resolution_clock::now() - start).count() < us);
    {
       
    }
}