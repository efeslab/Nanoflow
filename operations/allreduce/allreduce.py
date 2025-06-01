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
    def __init__(self, name, device):
        super().__init__(name, device)
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

class AllReduce_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer, op_device)
        
    def run(self):
        self.impl.run(self.inputs["input"].tensor, self.outputs["output"].tensor)