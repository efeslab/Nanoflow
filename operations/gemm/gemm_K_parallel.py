from numpy import isin
import torch
import time
import sqlite3
from utils.prof_marker import prof_marker
import platform_config
from operations.operation_base import Operations, Operation_Layer
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer

from operations.gemm.gemm_impls import GEMMTorchImpl, GEMMTritonImpl, GEMMCudaImpl

class GEMM_K_Parallel(Operations):
    def __init__(self, name, device, bias = False):
        super().__init__(name, device)
        if bias:
            self.inputs = {
                "A": IOWrapper(self, 'A', device).is_input(),
                "C": IOWrapper(self, 'C', device).is_input()
            }
        else:
            self.inputs = {
                "A": IOWrapper(self, 'A', device).is_input()
            }
        self.outputs = {
            "D": IOWrapper(self, 'D', device).is_output()
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
        self.op_layer = GEMM_K_Parallel_Layer

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
        self.add_impl(GEMMTritonImpl)
        if platform_config.PLATFORM_CUDA:
            self.add_impl(GEMMCudaImpl)
    
    def setShape(self, N, K, tp_idx=0, tp_size=1):
        self.tp_idx = tp_idx
        self.tp_size = tp_size
        # print("tp_idx", self.tp_idx, "tp_size", self.tp_size)
        self.N = N
        self.K = K
        self.tp_N = N
        self.tp_K = K // tp_size
        print("name", self.name, "N:", self.tp_N, "K:", self.tp_K)
        self.weights["B"].shape = (self.tp_K, self.tp_N)
        self.inputs["A"].init_shape((0, self.tp_K))
        if self.bias:
            self.inputs["C"].init_shape((0, self.tp_N)) # bias is a whole buffer, will be used by multiplying a beta.
        self.outputs["D"].init_shape((0, self.tp_N))
        return self
    
    def copy_nano(self, index):
        new_op = GEMM_K_Parallel(f"{self.name}{index}", self.device, self.bias)
        new_op.weights = self.weights
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.N, self.K, self.tp_idx, self.tp_size).setParameter(self.alpha, self.beta)
        new_op.set_stream(self.stream)
        
        self.nano_ops.append(new_op)

        return new_op

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

    def processWeight(self, global_weight_map, cached_weight_map, cached, device):
        if not isinstance(self.weight_name, list):
            self.weight_name = [self.weight_name]
        if not cached:
            offset = self.tp_idx % self.tp_size
            for l in self.layer_list:
                weights_list = []
                for name in self.weight_name:
                    stride = global_weight_map[name.format(layer=l)].shape[1] // self.tp_size
                    scope = (slice(None), slice(offset * stride, (offset + 1) * stride))
                    weights_list.append(global_weight_map[name.format(layer=l)][scope].t())
                cached_weight_map[f"{self.name}_layer_{l}"] = torch.cat(weights_list, dim=1).contiguous()
        if cached:
            self.weights["B"].weight_map = {}
            weight_wrapper = self.weights["B"]
            for l in self.layer_list:
                weight_wrapper.weight_map[l] = cached_weight_map[f"{self.name}_layer_{l}"].to(device, non_blocking=True)
                assert weight_wrapper.weight_map[l].shape == weight_wrapper.shape, f"name = {self.weight_name}, expected shape = {weight_wrapper.shape}, layer = {l}, real shape = {weight_wrapper.weight_map[l].shape}"
            
        # torch.cuda.empty_cache()
        # device = torch.cuda.current_device()
        # reserved_memory = torch.cuda.memory_reserved(device)
        # print(f"Reserved memory: {reserved_memory / 1024 / 1024} MB")

        
class GEMM_K_Parallel_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        with prof_marker("GEMM_run"):
            # with prof_marker("Allocate Sliced C"):
            C = self.inputs["C"].tensor if self.parent.bias else torch.empty((self.parent.batch_size, self.parent.N), dtype=torch.float16, device=self.device)
            self.impl.run(self.inputs["A"].tensor, self.weights["B"].weight_map[self.layer], C, self.outputs["D"].tensor)
