import torch
from transformers import AutoTokenizer
from operations.operation_base import Operations, Operation_Device, Operation_Layer
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

    def expand_gpu(self, gpu_list):
        for i in gpu_list:
            i_str = str(i)
            name = self.name + "_" + i_str
            op_device = GlobalInput_Device(self, self.name, i)
            self.children.append(op_device)
        
        return self.children

class GlobalInput_Device(Operation_Device):
    def __init__(self, op_general, name, device):
        super().__init__(op_general, name, device)      


class GlobalInput_Layer(Operations):
    def __init__(self, layer, operator_device):
        self.operator_device = operator_device
        self.name = f"{operator_device.name}_{layer}"
        self.layer = layer
        self.inputs = operator_device.inputs
        self.outputs = operator_device.outputs
        self.weights = operator_device.weights
        self.impl = operator_device.impl
    
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

    def expand_gpu(self, gpu_list):
        for i in gpu_list:
            i_str = str(i)
            name = self.name + "_" + i_str
            op_device = GlobalOutput_Device(self, self.name, i)
            self.children.append(op_device)
        
        return self.children

class GlobalOutput_Device(Operation_Device):
    def __init__(self, op_general, name, device):
        super().__init__(op_general, name, device)      

class GlobalOutput_Layer(Operation_Layer):
    def __init__(self, layer, operator_device):
        self.operator_device = operator_device
        self.name = f"{operator_device.name}_{layer}"
        self.layer = layer
        self.inputs = operator_device.inputs
        self.outputs = operator_device.outputs
        self.weights = operator_device.weights
        self.impl = operator_device.impl

    def run(self):
        pass
    