from typing import List

class OperationImpl:
    category_tag = None
    def __init__(self, op_base, stream, device):
        self.impl_tag = None
        self.op_base = op_base
        self.stream = stream
        self.stream_handle = stream.cuda_stream
        self.device = device
        self.batch_size = op_base.children[device].batch_size
        self.inputs = op_base.children[device].inputs
        self.outputs = op_base.children[device].outputs
        self.weights = op_base.children[device].weights
    
    @staticmethod
    def list_tags(self) -> List[str]:
        return [""]
    
    def config(self, impl_tag, parameter_map = {}):
        self.impl_tag = impl_tag
        
    def run(self, *args, **kwargs):
        pass