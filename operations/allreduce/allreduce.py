import torch
import torch.distributed as dist

import platform_config
from utils.prof_marker import prof_marker
from operations.operation_base import Operations, Operation_Layer
from core.IOWrapper import IOWrapper
from operations.impl_base import OperationImpl

class AllReduceTorchImpl(OperationImpl):
    category_tag = "torch"
    def __init__(self, op_base, stream, device):
        super().__init__(op_base, stream, device)
        self.tp_size = op_base.tp_size
        self.subgroup = op_base.subgroup
        self.N = op_base.N
        # self.reduce_buffer = op_base.inputs["input"].tensor.clone()
    
    def run(self, input, output):
        with torch.cuda.stream(self.stream):
            # temp = input.clone()
            work = dist.all_reduce(input, op=dist.ReduceOp.SUM, group=self.subgroup, async_op=True)
            work.wait()
            output.copy_(input)

class AllReduce(Operations):
    def __init__(self, name, device, nano_idx=None):
        super().__init__(name, device, nano_idx)
        self.inputs = {
            "input": IOWrapper(self, 'input', device).is_input()
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', device).is_output()
        }
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = AllReduce_Layer
    
    def init_impl_map(self):
        self.add_impl(AllReduceTorchImpl)

    def setShape(self, N, tp_idx, tp_size):
        self.N = N
        self.tp_idx = tp_idx
        self.tp_size = tp_size
        self.inputs["input"].init_shape((0, self.N))
        self.outputs["output"].init_shape((0, self.N))

    def update(self, subgroup):
        self.subgroup = subgroup

    def copy_nano(self, index):
        new_op = AllReduce(self.name, self.device, nano_idx=index)
        new_op.set_category(self.category)
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.N, self.tp_idx, self.tp_size)
        new_op.update(self.subgroup)

        self.nano_ops.append(new_op)

        return new_op

    def init_profile_db(self):
        for _, impl in self.impl_map.items():
            self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS "{impl.category_tag}" (
                id           INTEGER PRIMARY KEY AUTOINCREMENT, 
                batch_size   INTEGER,
                sm_count INTEGER,
                N INTEGER,
                average_time_ms REAL
            );
            ''')

    def store_profile_db(self, category_tag, impl_tag, average_elapsed_ms):
        print(f"Name: {self.name}, Category: {category_tag}, Batch Size: {self.batch_size}, Average Time: {average_elapsed_ms} ms")
        self.cursor.execute(f'''
            INSERT OR IGNORE INTO {category_tag} (batch_size, sm_count, N, average_time_ms)
            VALUES (?, ?, ?, ?);
        ''', (self.batch_size, self.sm_count, self.N, average_elapsed_ms))

    def run(self):
        self.impl.run(self.inputs["input"].tensor, self.outputs["output"].tensor)

    def profile_run(self):
        self.run()

class AllReduce_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer, op_device)
        
    def run(self):
        self.parent.run()