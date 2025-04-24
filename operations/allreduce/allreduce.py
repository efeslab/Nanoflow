import torch
import torch.distributed as dist

import platform_config
from operations.operation_base import Operations, Operation_Device, Operation_Layer
from core.IOWrapper import IOWrapper
from operations.impl_base import OperationImpl

class AllReduceTorchImpl(OperationImpl):
    category_tag = "torch"
    def __init__(self, op_base, device_id):
        super().__init__(op_base, device_id)
        self.tp_size = op_base.tp_size
        self.subgroup = op_base.subgroup
        print(f"subgroup: {self.subgroup}")
    
    def run(self, input_tensor, output_tensor):
        # print("using torch")
        dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM, group=self.subgroup)
        output_tensor.copy_(input_tensor)

class AllReduce(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "input": IOWrapper(self, 'input')
        }
        self.outputs = {
            "output": IOWrapper(self, 'output')
        }
        self.impl_map = {}
        self.init_impl_map()
        self.op_device = AllReduce_Device
    
    def init_impl_map(self):
        self.add_impl(AllReduceTorchImpl)
    
    def setShape(self, N, tp_size):
        self.N = N
        self.tp_size = tp_size
        for op_device in self.children:
            op_device.setShapeForIOWrappers()

class AllReduce_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.op_layer = AllReduce_Layer
    
    def setShapeForIOWrappers(self):
        self.inputs["input"].init_shape((0, self.parent.N))
        self.outputs["output"].init_shape((0, self.parent.N))

class AllReduce_Layer(Operation_Layer):
    def __init__(self, parent, layer):
        super().__init__(parent, layer)

    def run(self):
        self.impl.run(self.inputs["input"].tensor, self.outputs["output"].tensor)