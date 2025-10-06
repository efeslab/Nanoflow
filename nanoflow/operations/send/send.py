import torch
import platform_config

from operations.operation_base import Operations, Operation_Device, Operation_Layer
from core.IOWrapper import IOWrapper
from operations.impl_base import OperationImpl



class Send(Operations):
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
        self.op_device = Send_Device

class Send_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.op_layer = Send_Layer

class Send_Layer(Operation_Layer):
    def __init__(self, parent, layer):
        super().__init__(parent, layer)
        self.input = None
        self.output = None
        self.tp_size = parent.tp_size
        self.subgroup = parent.subgroup
    
