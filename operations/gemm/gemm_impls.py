import torch
import platform_config
from operations.impl_base import OperationImpl

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

if platform_config.PLATFORM_CUDA:
    import pybind.build.bind_gemm as bind_gemm
    class GEMMCudaImpl(OperationImpl):
        category_tag = "cuda"
        impl_tag_profile = "SM90_128_256_64_2_1_1_1_RowMajor_RowMajor_RowMajor_auto"
        def config(self, impl_tag, parameter_map):
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
                bind_gemm.configGEMM(impl_tag, self.name, self.inputs["A"].tensor, torch.empty((self.M, self.N), dtype=torch.float16, device=f"cuda:{self.device_id}"), self.outputs["D"].tensor, self.M, self.N, self.K, self.alpha, self.beta)

        # def profile(self, impl_tag):

        def run(self, B):
            bind_gemm.gemmLauncher(self.name, B, self.stream_handle)