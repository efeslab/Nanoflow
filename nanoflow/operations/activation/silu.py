import sqlite3
import torch
import time

from nanoflow.operations import Operations, Operation_Layer, OperationImpl
from nanoflow.core import IOWrapper, WeightWrapper
import nanoflow.platform_config as config


class SiluMultiplyTorchImpl(OperationImpl):
    category_tag = "torch"
    def run(self, x, output, act_fn):
        with torch.cuda.stream(self.stream):
            if act_fn == "silu_mul":
                A, B = torch.split(x, x.shape[-1] // 2, dim=-1)
                output.copy_(A * torch.nn.functional.silu(B))
            elif act_fn == "silu":
                output.copy_(torch.nn.functional.silu(x))
            elif act_fn == "sigmoid":
                output.copy_(torch.sigmoid(x))
            else:
                raise ValueError(f"Unsupported activation function: {act_fn}")

if config.PLATFORM_AITER:
    from aiter.ops.activation import silu_and_mul as aiter_silu_multiply
    class SiluMultiplyAiterImpl(OperationImpl):
        category_tag = "aiter"
        def run(self, x, output, act_fn):
            assert act_fn == "silu_mul", "Aiter implementation only supports silu_mul"
            with torch.cuda.stream(self.stream):
                aiter_silu_multiply(output, x)


if config.PLATFORM_CUDA:
    import nanoflow.pybind.build.bind_silu_multiply as bind_silu_multiply
    class SiluMultiplyCudaImpl(OperationImpl):
        category_tag = "cuda"
        def run(self, x, output, act_fn):
            assert act_fn == "silu_mul", "CUDA implementation only supports silu_mul"
            if self.batch_size > 0:
                bind_silu_multiply.silu_multiply(x, output, self.stream_handle)

class Activation(Operations):
    def __init__(self, name, device, act_fn="silu_mul", nano_idx=None):
        super().__init__(name, device, nano_idx)
        self.inputs = {
            "input": IOWrapper(self, 'input', device).is_input(),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', device).is_output(),
        }
        self.act_fn = act_fn
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = Activation_Layer
        
    
    def init_impl_map(self):
        self.add_impl(SiluMultiplyTorchImpl)
        if config.PLATFORM_AITER:
            self.add_impl(SiluMultiplyAiterImpl)
        if config.PLATFORM_CUDA:
            self.add_impl(SiluMultiplyCudaImpl)
        
    def setShape(self, N, tp_rank=0, tp_size=1):
        self.N = N
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_N = N // tp_size
        if self.act_fn == "silu_mul":
            self.inputs["input"].init_shape((0, self.tp_N * 2))
        else:
            self.inputs["input"].init_shape((0, self.tp_N))
        self.outputs["output"].init_shape((0, self.tp_N))

        return self
    
    def copy_nano(self, index):
        new_op = Activation(self.name, self.device, nano_idx=index)
        new_op.set_category(self.category)
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.N, self.tp_rank, self.tp_size)

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
        ''', (self.batch_size, self.sm_count, self.tp_N, average_elapsed_ms))

    def run(self):
        self.impl.run(self.inputs["input"].tensor, self.outputs["output"].tensor, act_fn=self.act_fn)

    def profile_run(self):
        self.run()


class Activation_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)
    
    def run(self):
        self.parent.run()