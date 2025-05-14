import torch
import triton
import platform_config
from operations.impl_base import OperationImpl
from pybind_triton_kernels.triton_gemm.src.kernels import gemm_kernel_persistent

class GEMMTorchImpl(OperationImpl):
    category_tag = "torch"
    impl_tag_profile = "torch"
    def config(self, impl_tag, parameter_map):
        self.alpha = self.op_base.alpha
        self.bias = self.op_base.bias
        self.beta = 0.0
        if self.bias:
            self.beta = self.op_base.beta
    
    def run(self, B):
        with torch.cuda.stream(self.stream):
            D = self.outputs["D"].tensor
            A = self.inputs["A"].tensor
            
            if self.bias:
                C = self.inputs["C"].tensor
                D.copy_(A.matmul(B) * self.alpha + C * self.beta)
            else:
                D.copy_(A.matmul(B) * self.alpha)

class GEMMTritonImpl(OperationImpl):
    category_tag = "triton"
    def config(self, impl_tag, parameter_map):
        self.M = self.batch_size
        self.N = self.op_base.N
        self.K = self.op_base.K
        self.alpha = self.op_base.alpha
        self.bias = self.op_base.bias
        self.beta = 0.0
        if self.bias:
            self.beta = self.op_base.beta

    def run(self, B):
        with torch.cuda.stream(self.stream):
            D = self.outputs["D"].tensor
            A = self.inputs["A"].tensor
            C = self.inputs["C"].tensor if self.bias else torch.empty((self.M, self.N), dtype=torch.float16, device=self.device)

            stride_am, stride_ak = A.stride()
            stride_bk, stride_bn = B.stride()
            stride_cm, stride_cn = C.stride()

            NUM_SMS = torch.cuda.get_device_properties(self.device).multi_processor_count

            grid = lambda META: (
                min(
                    NUM_SMS,
                    triton.cdiv(self.M, META["BLOCK_SIZE_M"]) * triton.cdiv(self.N, META["BLOCK_SIZE_N"]),
                ),
            )

            gemm_kernel_persistent[grid](
                A,
                B,
                C,
                self.M,
                self.N,
                self.K,
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(1),
                C.stride(0),
                C.stride(1),
                alpha=self.alpha,
                beta=self.beta,
                NUM_SMS=NUM_SMS,
            )
            D.copy_(C)


if platform_config.PLATFORM_CUDA:
    import bind_gemm
    class GEMMCudaImpl(OperationImpl):
        category_tag = "cuda"
        impl_tag_profile = "SM90_128_256_64_2_1_1_1_RowMajor_RowMajor_RowMajor_auto"
        def config(self, impl_tag, parameter_map):
            if self.batch_size > 0:
                self.name = self.op_base.name
                self.M = self.batch_size
                self.N = self.op_base.N
                self.K = self.op_base.K
                self.alpha = self.op_base.alpha
                self.bias = self.op_base.bias
                self.beta = 0.0
                # print("M:", self.M, "N:", self.N, "K:", self.K)
                # print("alpha:", self.alpha, "beta:", self.beta)
                if self.bias:
                    self.beta = self.op_base.beta
                    bind_gemm.configGEMM(impl_tag, self.name, self.inputs["A"].tensor, self.inputs["C"].tensor, self.outputs["D"].tensor, self.M, self.N, self.K, self.alpha, self.beta)
                else:
                    # print("GEMMCudaImpl config", self.name, "M:", self.M, "N:", self.N, "K:", self.K)
                    bind_gemm.configGEMM(impl_tag, self.name, self.inputs["A"].tensor, torch.empty((self.M, self.N), dtype=torch.float16, device=self.device), self.outputs["D"].tensor, self.M, self.N, self.K, self.alpha, self.beta)

        # def profile(self, impl_tag):

        def run(self, B):
            if self.batch_size > 0:
                bind_gemm.gemmLauncher(self.name, B, self.stream_handle)