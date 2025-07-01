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
    def __init__(self, name, device, nano_idx=None):
        super().__init__(name, device, nano_idx)
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
        new_op = Activation(self.name, self.device, nano_idx=index)
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.N, self.tp_idx, self.tp_size)

        self.nano_ops.append(new_op)

        return new_op

    def init_profile_db(self):
        for _, impl in self.impl_map.items():
            self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS "{impl.category_tag}" (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_size   INTEGER,
                sm_count     INTEGER,
                hidden_dim   INTEGER,
                average_time_ms REAL
            );
            ''')

    def store_profile_db(self, category_tag, impl_tag, average_elapsed_ms):
        print(f"Name: {self.name}, Category: {category_tag}, Batch Size: {self.batch_size}, Average Time: {average_elapsed_ms} ms")
        self.cursor.execute(f'''
            INSERT OR IGNORE INTO {category_tag} (batch_size, sm_count, hidden_dim, average_time_ms)
            VALUES (?, ?, ?, ?);
        ''', (self.batch_size, self.sm_count, self.N, average_elapsed_ms))

    def run(self):
        self.impl.run(self.inputs["input"].tensor, self.outputs["output"].tensor)

    def profile_run(self):
        self.run()


class Activation_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)
    
    def run(self):
        self.parent.run()