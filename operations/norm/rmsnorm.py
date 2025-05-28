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

if platform_config.PLATFORM_CUDA:
    import bind_rms_norm
    class LayerNormCudaImpl(OperationImpl):
        category_tag = "cuda"
        def run(self, x, weight, output, epsilon):
            # print("using cuda")
            if self.batch_size > 0:
                bind_rms_norm.rms_norm(output, x, weight, epsilon, self.stream_handle)

class LayerNorm(Operations):
    def __init__(self, name, device):
        super().__init__(name, device)
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
        if platform_config.PLATFORM_CUDA:
            self.add_impl(LayerNormCudaImpl)
    
    def setShape(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.weights["weight"].shape = (self.hidden_dim,)
        self.inputs["input"].init_shape((0, self.hidden_dim))
        self.outputs["output"].init_shape((0, self.hidden_dim))
    
    def copy_nano(self, index):
        new_op = LayerNorm(f"{self.name}{index}", self.device)
        new_op.weights = self.weights
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.hidden_dim)
        new_op.set_stream(self.stream)

        self.nano_ops.append(new_op)

        return new_op

    def profile(self):
        self.conn = sqlite3.connect('../profiling/LayerNorm.db')
        self.cursor = self.conn.cursor()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        hidden_dim = 4096

        # check the similarity of the outputs
        self.batch_size = 2
        x = torch.randn(self.batch_size, hidden_dim, dtype=torch.float16, device='cuda')
        weight = torch.randn(hidden_dim, dtype=torch.float16, device='cuda')
        output_list = []
        for _, impl in self.impl_map.items():
            out = torch.zeros((self.batch_size, hidden_dim), dtype=torch.float16, device='cuda')
            impl(self, None, self.device).run(x, weight, out, 1e-5)
            output_list.append(out)
            self.cursor.execute(f'''
                DROP TABLE IF EXISTS "{impl.category_tag}";
            ''')
            self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS "{impl.category_tag}" (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_size   INTEGER,
                hidden_dim INTEGER,
                average_time_ms REAL
            );
            ''')

        self.conn.commit()
        self.checkConsistencyBetweenImpl(output_list)

        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]

        for batch_size in batch_sizes:
            self.batch_size = batch_size
            out = torch.zeros((self.batch_size, hidden_dim), dtype=torch.float16, device='cuda')
            for _, impl in self.impl_map.items():
                impl_instance = impl(self, None, self.device)
                category_tag = impl_instance.category_tag
                latency_list = torch.empty(rounds-1, dtype=torch.float32, device='cuda')
                for round in range(rounds):
                    x = torch.randn((self.batch_size, hidden_dim), dtype=torch.float16, device='cuda')
                    weight = torch.randn(hidden_dim, dtype=torch.float16, device='cuda')
                    # record the time
                    start.record()
                    impl_instance.run(x, weight, out, 1e-5)
                    end.record()
                    torch.cuda.synchronize()
                    if round > 0:
                        # calculate the elapsed time
                        elapsed_ms = start.elapsed_time(end)
                        latency_list[round-1] = elapsed_ms
                print(f"Name: {self.name}, Category: {category_tag}, Batch Size: {batch_size}, Average Time: {latency_list.mean().item()} ms, Variance: {latency_list.var().item()}, latency[0]: {latency_list[0].item()} ms")
                # print("latency list:", latency_list.tolist())
                average_time_ms = latency_list.mean().item()
                self.cursor.execute(f'''
                    INSERT INTO {category_tag} (batch_size, hidden_dim, average_time_ms)
                    VALUES (?, ?, ?)
                    ''', (batch_size, hidden_dim, average_time_ms))
        self.conn.commit()
    
    def processWeight(self, global_weight_map, weight_path, cached, device):
        return process_weight_layer(global_weight_map, self.weight_name, self.weights["weight"], self.layer_list, weight_path, cached, device)
        

class LayerNorm_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        self.impl.run(self.inputs["input"].tensor, self.weights["weight"].weight_map[self.layer], self.outputs["output"].tensor, epsilon = 1e-5)