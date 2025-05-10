import torch
from transformers import AutoTokenizer
from operations.operation_base import Operations, Operation_Device, Operation_Layer
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer

class GlobalInput(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            # "new_token": IOWrapper(self, 'new_token', dtype=torch.int32)
        }
        self.outputs = {
            "tokens": IOWrapper(self, 'tokens', dtype=torch.int32)
        }
        self.op_device = GlobalInput_Device
    
    def profile(self):
        pass


class GlobalInput_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device) 
        self.op_layer = GlobalInput_Layer

    def setShapeForIOWrappers(self):
        self.outputs["tokens"].init_shape((0,))
        # self.outputs["tokens"].tensor[:self.batch_size].copy_(torch.tensor([0] * self.batch_size, dtype=torch.int32))

class GlobalInput_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer=layer, op_device=op_device)
    
    def run(self):
        pass
    

class GlobalOutput(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "tokens": IOWrapper(self, 'tokens', dtype=torch.int32)
        }
        self.outputs = {
            "new_token": IOWrapper(self, 'new_token', dtype=torch.int32)
        }
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.op_device = GlobalOutput_Device

    def profile(self):
        pass

class GlobalOutput_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)      
        self.op_layer = GlobalOutput_Layer

    def setShapeForIOWrappers(self):
        self.inputs["tokens"].init_shape((0,))
        self.outputs["new_token"].init_shape((0,))

class GlobalOutput_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer=layer, op_device=op_device)

    def run(self):
        pass
    