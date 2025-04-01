from numpy import isin
import torch
import sys
import time
import nvtx
sys.path.append('../../pybind/build')
from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
import bind_gemm
from operations.impl_base import OperationImpl

class GEMMTorchImpl(OperationImpl):
    category_tag = "torch"
    impl_tag_profile = "torch"
    def config(self, impl_tag, parameter_map):
        self.alpha = parameter_map["alpha"]
        self.bias = parameter_map["bias"]
        self.beta = 0.0
        if self.bias:
            self.beta = parameter_map["beta"]
    
    def run(self, A, B, C, D):
        if self.bias:
            D.copy_(A.matmul(B) * self.alpha + C * self.beta)
        else:
            D.copy_(A.matmul(B) * self.alpha)

class GEMMCudaImpl(OperationImpl):
    category_tag = "cuda"
    impl_tag_profile = "SM90_128_256_64_2_1_1_1_RowMajor_RowMajor_RowMajor_auto"
    def config(self, impl_tag, parameter_map):
        self.name = parameter_map["name"]
        self.M = parameter_map["M"]
        self.N = parameter_map["N"]
        self.K = parameter_map["K"]
        self.alpha = parameter_map["alpha"]
        self.bias = parameter_map["bias"]
        self.beta = 0.0
        if self.bias:
            self.beta = parameter_map["beta"]
            bind_gemm.configGEMM(impl_tag, self.name, self.inputs["A"].tensor, self.inputs["C"].tensor, self.outputs["D"].tensor, self.M, self.N, self.K, self.alpha, self.beta)
        else:
            bind_gemm.configGEMM(impl_tag, self.name, self.inputs["A"].tensor, torch.empty((self.M, self.N), dtype=torch.float16, device=self.inputs["A"].tensor.device), self.outputs["D"].tensor, self.M, self.N, self.K, self.alpha, self.beta)

    # def profile(self, impl_tag):

    def run(self, B):
        bind_gemm.gemmLauncher(self.name, B)

class GEMM(Operations):
    def __init__(self, name, bias = False):
        super().__init__(name)
        if bias:
            self.inputs = {
                "A": IOWrapper(self, 'A', IOBufferType.FULL),
                "C": IOWrapper(self, 'C', IOBufferType.FULL)
            }
        else:
            self.inputs = {
                "A": IOWrapper(self, 'A', IOBufferType.FULL)
            }
        self.outputs = {
            "D": IOWrapper(self, 'D', IOBufferType.FULL)
        }
        self.weights = {
            "B": WeightWrapper()
        }
        self.bias = bias
        self.alpha = 1.0
        if self.bias:
            self.beta = 1.0
        else:
            self.beta = 0.0
        self.impl_map = {}
        self.init_impl_map()

    def setParameter(self, alpha = 1, beta = 0):
        self.alpha = alpha
        self.beta = beta
        if self.bias == False and self.beta != 0:
            raise ValueError("beta should be 0 when bias is not used")
        if self.bias == True and self.beta == 0:
            raise ValueError("beta should not be 0 when bias is used")
        return self

    def init_impl_map(self):
        self.add_impl(GEMMTorchImpl)
        self.add_impl(GEMMCudaImpl)
    
    def setShape(self, N, K):
        self.N = N
        self.K = K
        self.weights["B"].shape = (self.K, self.N)
    
    def setBatchSize(self, M):
        self.M = M
        if self.name == "O":
            self.inputs["A"].shape = (self.M, 32, 128)
        else:
            self.inputs["A"].shape = (self.M, self.K)
        if self.bias:
            self.inputs["C"].shape = (self.M, self.N)
        self.outputs["D"].shape = (self.M, self.N)
        
    def profile(self):
        # print("Get into profile", self.name)
        parameters_map = {
            "M": 2,
            "N": self.N,
            "K": self.K,
            "alpha": self.alpha,
            "bias": self.bias,
            "beta": self.beta
        }

        # check the similarity of the outputs
        A = torch.randn((2, self.K), dtype=torch.float16, device='cuda')
        B = torch.randn((self.K, self.N), dtype=torch.float16, device='cuda')
        C = torch.randn((2, self.N), dtype=torch.float16, device='cuda')
        output_list = []
        for _, impl in self.impl_map.items():
            # print(impl.impl_tag_profile)
            impl_instance = impl()
            out = torch.zeros((2, self.N), dtype=torch.float16, device='cuda')
            
            impl_instance.config(impl.impl_tag_profile, parameters_map)
            impl_instance.run(A, B, C, out)
            output_list.append(out)
            # print("finish the implentation", impl_instance.category_tag)
        
        self.checkConsistencyBetweenImpl(output_list)
        # print("Finish checking consistency")

        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        for batch_size in batch_sizes:
            parameters_map["M"] = batch_size
            D = torch.zeros((batch_size, self.N), dtype=torch.float16, device='cuda')
            for _, impl in self.impl_map.items():
                impl_instance = impl()
                impl_instance.config(impl.impl_tag_profile, parameters_map)
                category_tag = impl_instance.category_tag
                total_latency = 0
                for round in range(rounds):
                    A = torch.randn((batch_size, self.K), dtype=torch.float16, device='cuda')
                    B = torch.randn((self.K, self.N), dtype=torch.float16, device='cuda')
                    C = torch.randn((batch_size, self.N), dtype=torch.float16, device='cuda')
                    # record the time
                    start_time = time.time()
                    impl_instance.run(A, B, C, D)
                    if round > 0:
                        total_latency += time.time() - start_time
                average_time = total_latency / rounds
                print("name: {}, batch_size: {}, average_time: {}".format(self.name + f"_{category_tag}", batch_size, average_time))
                self.cursor.execute('''
                    INSERT INTO performance (keyword, batch_size, average_time)
                    VALUES (?, ?, ?)
                    ''', (self.name + f"_{category_tag}", batch_size, average_time))
        self.conn.commit()
    
    def run(self, layer):
        with nvtx.annotate("GEMM_prepare"):
            A = self.inputs["A"].tensor
            if self.bias:
                C = self.inputs["C"].tensor
            else:
                C = torch.empty((self.M, self.N), dtype=torch.float16, device=self.inputs["A"].tensor.device)
        
            B = self.weights["B"].weight_map[layer]
        with nvtx.annotate("GEMM_run"):
            self.impl.run(A, B, C, self.outputs["D"].tensor)
    
    def processWeight(self, global_weight_map, total_layers, cached = False):
        self.weights["B"].weight_map = {}
        if not isinstance(self.weight_name, list):
            self.weight_name = [self.weight_name]
        if any(['{layer}' in name for name in self.weight_name]):
            for l in range(total_layers):
                self.weights["B"].weight_map[l] = torch.cat([global_weight_map[name.format(layer=l)].t() for name in self.weight_name], dim=1).contiguous()
        else:
            weight_tensor = torch.cat([global_weight_map[name].t() for name in self.weight_name], dim=1).contiguous()
            for l in range(total_layers):
                self.weights["B"].weight_map[l] = weight_tensor
        # for l in range(total_layers):
        #     for name in self.weight_name:
        #         if name.format(layer=l) in global_weight_map:
        #             del global_weight_map[name.format(layer=l)]
        # torch.cuda.empty_cache()
        # device = torch.cuda.current_device()
        # reserved_memory = torch.cuda.memory_reserved(device)
        # print(f"Reserved memory: {reserved_memory / 1024 / 1024} MB")

class GEMM_Layer(Operations):
    def __init__(self, layer, operator_device):
        self.operator_device = operator_device
        self.name = f"{operator_device.name}_{layer}"
        self.layer = layer
        self.inputs = operator_device.inputs
        self.outputs = operator_device.outputs
        self.weights = operator_device.weights
        self.impl = operator_device.impl

    def run(self):
        with nvtx.annotate("GEMM_run"):
            self.operator_device.impl.run(self.weights["B"].weight_map[self.layer])