import torch
from transformers import AutoTokenizer
from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer

class GlobalInput(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            # "new_token": IOWrapper(self, 'new_token', IOBufferType.FULL, dtype=torch.int32)
        }
        self.outputs = {
            "tokens": IOWrapper(self, 'tokens', IOBufferType.FULL, dtype=torch.int32)
        }
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.outputs["tokens"].shape = (self.batch_size,)
        # self.outputs["tokens"].tensor[:self.batch_size].copy_(torch.tensor([0] * self.batch_size, dtype=torch.int32))
    
    def profile(self):
        pass

    def run(self, layer):
        pass

class GlobalInput_Layer(Operations):
    def __init__(self, layer, operator_device):
        self.name = f"{operator_device.name}_{layer}"
        self.layer = layer
        self.inputs = operator_device.inputs
        self.outputs = operator_device.outputs
    
    def run(self):
        pass
    

class GlobalOutput(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "tokens": IOWrapper(self, 'tokens', IOBufferType.FULL, dtype=torch.int32)
        }
        self.outputs = {
            "new_token": IOWrapper(self, 'new_token', IOBufferType.FULL, dtype=torch.int32)
        }
        self.model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["tokens"].shape = (self.batch_size,)
        self.outputs["new_token"].shape = (1,)

    def profile(self):
        pass

    def run(self, layer):
        # self.outputs["new_token"].tensor.copy_(self.inputs["tokens"].tensor[-1])
        pass

class GlobalOutput_Layer(Operations):
    def __init__(self, layer, operator_device):
        self.name = f"{operator_device.name}_{layer}"
        self.layer = layer
        self.inputs = operator_device.inputs
        self.outputs = operator_device.outputs

    def run(self):
        pass
    