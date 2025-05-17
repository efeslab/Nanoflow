import torch
import torch.distributed as dist

import platform_config
from operations.operation_base import Operations, Operation_Layer
from core.IOWrapper import IOWrapper
from operations.impl_base import OperationImpl


class AllGatherTorchImpl(OperationImpl):
    category_tag = "torch"
    def __init__(self, op_base, stream, device):
        super().__init__(op_base, stream, device)
        self.tp_size = op_base.tp_size
        self.subgroup = op_base.subgroup
    
    def run(self, input, output):
        with torch.cuda.stream(self.stream):
            gather_list = [torch.empty_like(input) for _ in range(self.tp_size)]
            dist.all_gather(gather_list, input, group=self.subgroup)
            out = torch.cat(gather_list, dim=1)
            output.copy_(out)

class AllGather(Operations):
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
        self.op_layer = AllGather_Layer
    
    def init_impl_map(self):
        self.add_impl(AllGatherTorchImpl)

    def setShape(self, N, tp_idx, tp_size):
        self.N = N
        self.tp_idx = tp_idx
        self.tp_size = tp_size
        self.inputs["input"].init_shape((0, self.N // self.tp_size))
        self.outputs["output"].init_shape((0, self.N))
    
    def update(self, subgroup):
        self.subgroup = subgroup
    
class AllGather_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer, op_device)
        
    def run(self):
        self.impl.run(self.inputs["input"].tensor, self.outputs["output"].tensor)