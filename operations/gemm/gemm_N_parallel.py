from numpy import isin
import torch
import sys
import time
from utils.prof_marker import prof_marker
import platform_config
from operations.operation_base import Operations, Operation_Device, Operation_Layer
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer

from operations.gemm.gemm_impls import GEMMTorchImpl, GEMMTritonImpl, GEMMCudaImpl

class GEMM_N_Parallel(Operations):
    def __init__(self, name, bias = False):
        super().__init__(name)
        if bias:
            self.inputs = {
                "A": IOWrapper(self, 'A'),
                "C": IOWrapper(self, 'C')
            }
        else:
            self.inputs = {
                "A": IOWrapper(self, 'A')
            }
        self.outputs = {
            "D": IOWrapper(self, 'D')
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
        self.op_device = GEMM_N_Parallel_Device

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
    
    def setShape(self, N, K, tp_size=1, strides=[]):
        self.tp_size = tp_size
        self.N = N // tp_size
        self.K = K
        print("name", self.name, "N:", self.N, "K:", self.K)
        self.weights["B"].shape = (self.K, self.N)
        self.updateChildrenIOShape()
        return self
    
    def copy_nano(self, index):
        new_op = GEMM_N_Parallel(f"{self.name}{index}", self.bias)
        new_op.weights = self.weights
        new_op.expand_all_gpu_and_layers(len(self.device_list), 32)
        new_op.setShape(self.N, self.K, self.tp_size).setParameter(self.alpha, self.beta)
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

    def processWeight(self, global_weight_map, cached_weight_map, cached = False):
        self.weights["B"].weight_map = {}
        weight_wrapper = self.weights["B"]
        if not isinstance(self.weight_name, list):
            self.weight_name = [self.weight_name]
        if cached:
            for device_id in self.device_list:
                weight_wrapper.weight_map[device_id] = {}
                for l in self.layer_list:
                    weight_wrapper.weight_map[device_id][l] = cached_weight_map[f"{self.name}_device_{device_id}_layer_{l}"].to(f'cuda:{device_id}')
                    assert weight_wrapper.weight_map[device_id][l].shape == weight_wrapper.shape, f"name = {self.weight_name}, expected shape = {weight_wrapper.shape}, layer = {l}, real shape = {weight_wrapper.weight_map[device_id][l].shape}"
    
        elif not cached:
            for device_id in self.device_list:
                weight_wrapper.weight_map[device_id] = {}
                offset = device_id % self.tp_size
                for l in self.layer_list:
                    weights_list = []
                    for name in self.weight_name:
                        stride = global_weight_map[name.format(layer=l)].shape[0] // self.tp_size
                        scope = (slice(offset * stride, (offset + 1) * stride), slice(None))
                        weights_list.append(global_weight_map[name.format(layer=l)][scope].to(f'cuda:{device_id}').t())
                    weight_wrapper.weight_map[device_id][l] = torch.cat(weights_list, dim=1).contiguous()

                    cached_weight_map[f"{self.name}_device_{device_id}_layer_{l}"] = weight_wrapper.weight_map[device_id][l].to("cpu")
                    assert weight_wrapper.weight_map[device_id][l].shape == weight_wrapper.shape, f"name = {self.weight_name}, expected shape = {weight_wrapper.shape}, layer = {l}, real shape = {weight_wrapper.weight_map[device_id][l].shape}"
        # torch.cuda.empty_cache()
        # device = torch.cuda.current_device()
        # reserved_memory = torch.cuda.memory_reserved(device)
        # print(f"Reserved memory: {reserved_memory / 1024 / 1024} MB")
    
class GEMM_N_Parallel_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.op_layer = GEMM_N_Parallel_Layer

    def setShapeForIOWrappers(self):
        self.inputs["A"].init_shape((0, self.parent.K))
        if self.parent.bias:
            self.inputs["C"].init_shape((0, self.parent.N))
        self.outputs["D"].init_shape((0, self.parent.N))
        

class GEMM_N_Parallel_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer=layer, op_device=op_device)

    def run(self):
        with prof_marker("GEMM_run"):
            self.impl.run(self.weights["B"].weight_map[self.device_id][self.layer])
