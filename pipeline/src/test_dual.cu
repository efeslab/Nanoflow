#include "dualWrapper.cuh"
#include <iostream>
#include "spdlog/sinks/basic_file_sink.h"
#include <string>

int main() {
    DualWrapper<128, 128, 32, 64, 64, 32, 1, 5, cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor> dw;
    int M = 128;
    int N = 128;
    int K = 128;

    dw.set_shape(M, N, K);
    cutlass::half_t *host_tensors[7];
    for (int i = 0; i < 7; i++) {
        host_tensors[i] = new cutlass::half_t[M*N];
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < N; k++) {
                host_tensors[i][j*N+k] = cutlass::half_t(float((j+k))/20/128);

            }
        }
    }

    cutlass::half_t *device_tensors[7];
    for (int i = 0; i < 7; i++) {
        cudaMalloc(&device_tensors[i], M*N*sizeof(cutlass::half_t));
        cudaMemcpy(device_tensors[i], host_tensors[i], M*N*sizeof(cutlass::half_t), cudaMemcpyHostToDevice);
    }
    vortexWeight b1, b2;
    b1.ptr = (half* )device_tensors[1];
    b1.N = N;
    b1.K = K;
    b2.ptr = (half* )device_tensors[2];
    b2.N = N;
    b2.K = K;
    
    dw.set_weight(b1,b2);


    pllmTensor<cutlass::half_t> a = pllmTensor<cutlass::half_t>(device_tensors[0], M, K, PllmLayout::ROW_MAJOR);
    pllmTensor<cutlass::half_t> c = pllmTensor<cutlass::half_t>(device_tensors[3], M, N, PllmLayout::ROW_MAJOR);
    pllmTensor<cutlass::half_t> d0 = pllmTensor<cutlass::half_t>(device_tensors[4], M, N, PllmLayout::ROW_MAJOR);
    pllmTensor<cutlass::half_t> d1 = pllmTensor<cutlass::half_t>(device_tensors[5], M, N, PllmLayout::ROW_MAJOR);
    pllmTensor<cutlass::half_t> d2 = pllmTensor<cutlass::half_t>(device_tensors[6], M, N, PllmLayout::ROW_MAJOR);

    dw.setA(a);
    dw.setC(c);
    dw.setD(d0, d1, d2);
    dw.init();
    dw.set_weight(b1,b2);
    dw.setStream(0);
    std::string private_file_name = "dual.txt";
	auto private_logger = spdlog::basic_logger_mt("private_logger", private_file_name, true);
    dw.run().log(private_logger);
    cudaDeviceSynchronize();

    // copy back d0, d1, d2
    for (int i = 4; i < 7; i++) {
        cudaMemcpy(host_tensors[i], device_tensors[i], M*N*sizeof(cutlass::half_t), cudaMemcpyDeviceToHost);
    }

    for (int i = 4; i < 7; i++) {
        for (int j = 0; j < M; j++) {
            for (int k = 0; k < N; k++) {
                std::cout << host_tensors[i][j*N+k] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    return 0;
}