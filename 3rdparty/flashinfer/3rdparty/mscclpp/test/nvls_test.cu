// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <stdio.h>

#if (USE_NVLS)
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <mscclpp/gpu.hpp>

#define CUCHECK(cmd)                                     \
  do {                                                   \
    auto err = cmd;                                      \
    if (err != 0) {                                      \
      printf("Cuda failure %d: Line %d", err, __LINE__); \
      exit(-1);                                          \
    }                                                    \
  } while (false)

// AR kernel snippet for sm_90 only

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define MULTIMEM_ST(val, ptr)                                                                                   \
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), \
               "r"(val.w)                                                                                       \
               : "memory");
// specific PTX for fp16 reduction. bf16 would be multimem.ld_reduce.global.add.v4.bf16x2 etc
#define MULTIMEM_LD(val, ptr)                                     \
  asm("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];" \
      : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)        \
      : "l"(ptr)                                                  \
      : "memory");
#else
#define MULTIMEM_ST(val, ptr)
#define MULTIMEM_LD(val, ptr)
#endif

__global__ void init_kernel(float* uc_ptr, int size, int myrank, int nranks) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < size; idx += blockDim.x * gridDim.x) {
    uc_ptr[idx] = myrank + idx;
  }
}

__global__ void check_correctness(float* uc_ptr, int size, int myrank, int nranks) {
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < size; idx += blockDim.x * gridDim.x) {
    float expected = (float)((nranks * (nranks - 1)) / 2 + nranks * idx);
    if (abs(uc_ptr[idx] - expected) > 0.01 * expected) {
      printf("error! idx %d: %f != %f\n", idx, uc_ptr[idx], expected);
    }
  }
}

__global__ void testing(float* mc_ptr, int size, int myrank, int nranks) {
  // for allreduce we dont even need an UC pointer. just using same mc_ptr for in-place reduction
  // line is assumed to be 16B 4 ints of 8 halves
  int my_st = ((int64_t)size * (int64_t)myrank) / (int64_t)nranks;
  int my_en = ((int64_t)size * (int64_t)(myrank + 1)) / (int64_t)nranks;

  int my_offset = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
  int my_step = blockDim.x * gridDim.x * 4;

  for (int idx = my_st + my_offset; idx < my_en; idx += my_step) {
    [[maybe_unused]] uint4 val;
    MULTIMEM_LD(val, mc_ptr + idx);
    MULTIMEM_ST(val, mc_ptr + idx);
  }
}

int main() {
  int myrank, nranks;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  cudaSetDevice(myrank);

  size_t size = 1024 * 1024 * 512;
  CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  CUmulticastObjectProp mcProp = {};
  mcProp.numDevices = nranks;
  mcProp.size = size;
  mcProp.handleTypes = handleType;

  size_t minGran, gran;
  gran = 0;
  minGran = 0;
  CUCHECK(cuMulticastGetGranularity(&minGran, &mcProp, CU_MULTICAST_GRANULARITY_MINIMUM));
  CUCHECK(cuMulticastGetGranularity(&gran, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));

  if (!myrank) printf("nvls multicast granularity: gran = %lu, minGrad = %lu\n", gran, minGran);
  size_t mcSize = ((size + gran - 1) / gran) * gran;
  mcProp.size = mcSize;

  CUmemGenericAllocationHandle handle;
  // only one rank creates the multicast object
  if (!myrank) CUCHECK(cuMulticastCreate(&handle, &mcProp));

  int fd = 0;
  if (!myrank) CUCHECK(cuMemExportToShareableHandle(&fd, handle, handleType, 0 /*flags*/));

  // some ugly UDS business
  //  Borrow ipcsocket.{c,h} from nccl code
  // in cuda 12.4 new fabric handle type is available so instead it would be possible to use MPI_Allgather for the
  // exported handles
  //  moreover it would the only way to do it on GraceHopper systems, since UDS is limited to single Unix node

  pid_t currentPid = getpid();
  MPI_Bcast(&fd, sizeof(fd), MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Bcast(&currentPid, sizeof(currentPid), MPI_CHAR, 0, MPI_COMM_WORLD);
  int pidFd = syscall(SYS_pidfd_open, currentPid, 0);

  // MPI_Bcast(&fd, sizeof(fd), MPI_CHAR, 0, MPI_COMM_WORLD);
  // everyone else would now have same multicast object
  int peerFd = 0;
  peerFd = syscall(SYS_pidfd_getfd, pidFd, fd, 0);
  if (myrank) CUCHECK(cuMemImportFromShareableHandle(&handle, reinterpret_cast<void*>(peerFd), handleType));
  MPI_Barrier(MPI_COMM_WORLD);

  close(fd);
  // end of ugly UDS business
  // everyone adds device(s), no syncs required, just need to ensure bindmem happens after all this is called
  int mydev = myrank;
  CUCHECK(cuMulticastAddDevice(handle, mydev));
  MPI_Barrier(MPI_COMM_WORLD);

  CUmemGenericAllocationHandle memhandle;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = mydev;
  prop.requestedHandleTypes = handleType;

  // allocate physical memory (data buffer)
  CUCHECK(cuMemCreate(&memhandle, size, &prop, 0 /*flags*/));

  void* uc_va;
  void* mc_va;
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = mydev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  // Map a VA to UC space
  CUCHECK(cuMemAddressReserve((CUdeviceptr*)&uc_va, size, minGran, 0U, 0));
  cudaMemset(uc_va, 0, size);
  CUCHECK(cuMemMap((CUdeviceptr)uc_va, size, 0, memhandle, 0));
  // set access on UC address
  CUCHECK(cuMemSetAccess((CUdeviceptr)uc_va, size, &accessDesc, 1));

  // everyone binds memory to the multicast
  CUCHECK(cuMulticastBindMem(handle, 0 /*mcOffset*/, memhandle, 0 /*memOffset*/, size, 0));
  MPI_Barrier(MPI_COMM_WORLD);
  // usual VA business: map both MC and PA to two different VA addresses

  // Map a VA to MC space
  CUCHECK(cuMemAddressReserve((CUdeviceptr*)&mc_va, mcSize, minGran, 0U, 0));
  CUCHECK(cuMemMap((CUdeviceptr)mc_va, mcSize, 0, handle, 0));
  // set access on MC address
  CUCHECK(cuMemSetAccess((CUdeviceptr)mc_va, mcSize, &accessDesc, 1));

  int rept = 10;
  int block_size = 1024;
  int nblocks = 16;

  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  init_kernel<<<nblocks, block_size>>>((float*)uc_va, size / sizeof(float), myrank, nranks);
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  testing<<<nblocks, block_size>>>((float*)mc_va, size / sizeof(float), myrank, nranks);
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);
  check_correctness<<<nblocks, block_size>>>((float*)uc_va, size / sizeof(float), myrank, nranks);
  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);

  for (size_t input_size = 1024; input_size <= size; input_size *= 2) {
    // warmup
    for (int i = 0; i < rept; i++) {
      testing<<<nblocks, block_size>>>((float*)mc_va, input_size / sizeof(float), myrank, nranks);
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    double st = MPI_Wtime();
    for (int i = 0; i < rept; i++) {
      testing<<<nblocks, block_size>>>((float*)mc_va, input_size / sizeof(float), myrank, nranks);
    }
    cudaDeviceSynchronize();
    double en = MPI_Wtime();
    double time = (en - st) / rept;
    if (!myrank)
      printf("input_size %ld | Time = %f us, alg_bw = %f (GBps)\n", input_size, time * 1e6, input_size / 1e9 / time);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}

#else  // !(USE_NVLS)

int main() {
  printf("This test requires NVLS to be enabled\n");
  return 0;
}

#endif  // !(USE_NVLS)
