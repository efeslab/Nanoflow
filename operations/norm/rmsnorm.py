import torch
import time
import sqlite3

import platform_config
from operations.operation_base import Operations, Operation_Layer
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer

from operations.impl_base import OperationImpl

class LayerNormTorchImpl(OperationImpl):
    category_tag = "torch"
    def run(self, input, weight, output, epsilon):
        with torch.cuda.stream(self.stream):
            # print("using torch")
            x = input.to(torch.float32)
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)
            normalized_x = x / rms
            output.copy_(normalized_x.to(torch.float16) * weight)


if platform_config.PLATFORM_TRITON:
    from .triton.kernels.rmsnorm import rms_norm as triton_rms_norm
    class LayerNormTritonImpl(OperationImpl):
        category_tag = "triton"
        def run(self, x, weight, output, epsilon):
            with torch.cuda.stream(self.stream):
                triton_rms_norm(x, weight, output, epsilon)


if platform_config.PLATFORM_CUDA:
    import bind_rms_norm
    class LayerNormCudaImpl(OperationImpl):
        category_tag = "cuda"
        def run(self, x, weight, output, epsilon):
            # print("using cuda")
            if self.batch_size > 0:
                with torch.cuda.stream(self.stream):
                    bind_rms_norm.rms_norm(output, x, weight, epsilon, self.stream_handle)


if platform_config.PLATFORM_AITER:
    from aiter.ops.rmsnorm import rms_norm as aiter_rms_norm

    class LayerNormAiterImpl(OperationImpl):
        category_tag = "aiter"
        def run(self, x, weight, output, epsilon):
            with torch.cuda.stream(self.stream):
                output.copy_(aiter_rms_norm(x, weight, epsilon))


class LayerNorm(Operations):
    def __init__(self, name, device, nano_idx=None):
        super().__init__(name, device, nano_idx)
        self.inputs = {
            "input": IOWrapper(self, 'input', device).is_input(),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', device).is_output(),
        }
        self.weights = {
            "weight": WeightWrapper(self),
        }
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = LayerNorm_Layer

    def init_impl_map(self):
        self.add_impl(LayerNormTorchImpl)
        if platform_config.PLATFORM_TRITON:
            self.add_impl(LayerNormTritonImpl)
        if platform_config.PLATFORM_AITER:
            self.add_impl(LayerNormAiterImpl)
        if platform_config.PLATFORM_CUDA:
            self.add_impl(LayerNormCudaImpl)
    
    def setShape(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.weights["weight"].shape = (self.hidden_dim,)
        self.inputs["input"].init_shape((0, self.hidden_dim))
        self.outputs["output"].init_shape((0, self.hidden_dim))
    
    def copy_nano(self, index):
        new_op = LayerNorm(self.name, self.device, nano_idx=index)
        new_op.weights = self.weights
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.hidden_dim)

        self.nano_ops.append(new_op)

        return new_op

    def init_profile_database(self):
        for _, impl in self.impl_map.items():
            self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS "{impl.category_tag}" (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_size   INTEGER,
                sm_count INTEGER,
                hidden_dim INTEGER,
                average_time_ms REAL
            );
            ''')
            
    def store_profile_database(self, category_tag, impl_tag, average_elapsed_ms):
        print(f"Name: {self.name}, Category: {category_tag}, Batch Size: {self.batch_size}, Average Time: {average_elapsed_ms} ms")
        self.cursor.execute(f'''
            INSERT OR IGNORE INTO {category_tag} (batch_size, sm_count, hidden_dim, average_time_ms)
            VALUES (?, ?, ?, ?)
            ''', (self.batch_size, self.sm_count, self.hidden_dim, average_elapsed_ms))

    def run(self, layer):
        self.impl.run(self.inputs["input"].tensor, self.weights["weight"].weight_map[layer], self.outputs["output"].tensor, epsilon = 1e-5)
    
    def profile_run(self):
        self.run(self.layer_list[0])
    
    def processWeight(self, global_weight_map, weight_path, cached, device):
        return process_weight_layer(global_weight_map, self.weight_name, self.weights["weight"], self.layer_list, weight_path, cached, device)
        

class LayerNorm_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        self.parent.run(self.layer)