import torch
import nanoflow.platform_config as platform_config
from nanoflow.operations import OperationImpl

if platform_config.PLATFORM_CUDA:
    from nanoflow.utils.green_ctx import set_sm_count_target
else:

    def set_sm_count_target(sm_count):
        pass


class GEMMTorchImpl(OperationImpl):
    category_tag = "torch"
    impl_tag_profile = "torch"

    def config(self, impl_tag, parameter_map):
        self.alpha = self.op_base.alpha
        self.bias = self.op_base.bias
        self.beta = 0.0
        if self.bias:
            self.beta = self.op_base.beta

        # if self.op_base.sm_count is not None:
        #     set_sm_count_target(self.op_base.sm_count)

    def run(self, A, B, C, D):
        with torch.cuda.stream(self.stream):
            if self.op_base.sm_count is not None:
                set_sm_count_target(self.op_base.sm_count)
            if self.bias or self.alpha != 1:
                torch.addmm(C, A, B, beta=self.beta, alpha=self.alpha, out=D)
            else:
                torch.matmul(A, B, out=D)


if platform_config.PLATFORM_AITER:

    class GEMMAiterImpl(OperationImpl):
        category_tag = "aiter"

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


# if platform_config.PLATFORM_CUDA:
#     import bind_gemm
#     class GEMMCudaImpl(OperationImpl):
#         category_tag = "cuda"
#         impl_tag_profile = "SM90_128_256_64_2_1_1_1_RowMajor_RowMajor_RowMajor_auto"
#         def config(self, impl_tag, parameter_map):
#             if self.batch_size > 0:
#                 self.name = self.op_base.name
#                 self.M = self.batch_size
#                 self.N = self.op_base.tp_N
#                 self.K = self.op_base.tp_K
#                 self.alpha = self.op_base.alpha
#                 self.beta = self.op_base.beta
#                 # print("M:", self.M, "N:", self.N, "K:", self.K)
#                 # print("alpha:", self.alpha, "beta:", self.beta)
#                 bind_gemm.configGEMM(impl_tag, self.name, self.M, self.N, self.K, self.alpha, self.beta)

#         # def profile(self, impl_tag):

#         def run(self, A, B, C, D):
#             with torch.cuda.stream(self.stream):
#                 if self.batch_size > 0:
#                     bind_gemm.gemmLauncher(self.name, A, B, C, D, self.stream_handle)
