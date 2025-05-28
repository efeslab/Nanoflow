import sqlite3
import torch
import time

from operations.operation_base import Operations, Operation_Layer
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none
import platform_config as config
from operations.impl_base import OperationImpl

class SiluMultiplyTorchImpl(OperationImpl):
    category_tag = "torch"
    def run(self, x, output):
        with torch.cuda.stream(self.stream):
            A, B = torch.split(x, x.shape[-1] // 2, dim=-1)
            output.copy_(A * torch.nn.functional.silu(B))

if config.PLATFORM_CUDA:
    import bind_silu_multiply
    class SiluMultiplyCudaImpl(OperationImpl):
        category_tag = "cuda"
        def run(self, x, output):
            if self.batch_size > 0:
                bind_silu_multiply.silu_multiply(x, output, self.stream_handle)

class Activation(Operations):
    def __init__(self, name, device):
        super().__init__(name, device)
        self.inputs = {
            "input": IOWrapper(self, 'input', device).is_input(),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', device).is_output(),
        }
        self.act_fn = torch.nn.SiLU()
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = Activation_Layer
        
    
    def init_impl_map(self):
        self.add_impl(SiluMultiplyTorchImpl)
        if config.PLATFORM_CUDA:
            self.add_impl(SiluMultiplyCudaImpl)
        
    def setShape(self, N, tp_idx=0, tp_size=1):
        self.N = N // tp_size
        self.tp_idx = tp_idx
        self.tp_size = tp_size
        tp_N = N // tp_size
        self.inputs["input"].init_shape((0, tp_N * 2))
        self.outputs["output"].init_shape((0, tp_N))
    
    def copy_nano(self, index):
        new_op = Activation(f"{self.name}{index}", self.device)
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.N, self.tp_idx, self.tp_size)
        new_op.set_stream(self.stream)

        self.nano_ops.append(new_op)

        return new_op

    def profile(self):
        # check the similarity of the outputs
        self.conn = sqlite3.connect('../profiling/Activation.db')
        self.cursor = self.conn.cursor()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        N = 7 * 1024
        self.batch_size = 2
        x = torch.randn(self.batch_size, N * 2, dtype=torch.float16, device='cuda')
        output_list = []
        for _, impl in self.impl_map.items():
            out = torch.zeros((self.batch_size, N), dtype=torch.float16, device='cuda')
            impl(self, None, self.device).run(x, out)
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

        # profile the performance
        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]

        for batch_size in batch_sizes:
            out = torch.zeros((batch_size, N), dtype=torch.float16, device='cuda')
            for _, impl in self.impl_map.items():
                impl_instance = impl(self, None, self.device)
                category_tag = impl_instance.category_tag
                latency_list = torch.empty(rounds-1, dtype=torch.float32, device='cuda')
                for round in range(rounds):
                    x = torch.randn(batch_size, N * 2, dtype=torch.float16, device='cuda')
                    # record the time
                    start.record()
                    impl_instance.run(x, out)
                    end.record()
                    torch.cuda.synchronize()
                    if round > 0:
                        elapsed_ms = start.elapsed_time(end)
                        latency_list[round-1] = elapsed_ms

                average_time_ms = latency_list.mean().item()
                print(f"Name: {self.name}, Category: {category_tag}, Batch Size: {batch_size}, Average Time: {average_time_ms} ms, Variance: {latency_list.var().item()}, latency[0]: {latency_list[0].item()} ms")
                self.cursor.execute(f'''
                    INSERT INTO {category_tag} (batch_size, hidden_dim, average_time_ms)
                    VALUES (?, ?, ?);
                ''', (batch_size, N, average_time_ms))
        self.conn.commit()


class Activation_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)
    
    def run(self):
        self.impl.run(self.inputs["input"].tensor, self.outputs["output"].tensor)