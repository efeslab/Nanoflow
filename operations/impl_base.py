from typing import List

class OperationImpl:
    category_tag = None
    def __init__(self, op_base, device_id):
        self.impl_tag = None
        self.op_base = op_base
        self.device_id = device_id
        self.batch_size = op_base.children[device_id].batch_size
        self.inputs = op_base.children[device_id].inputs
        self.outputs = op_base.children[device_id].outputs
        self.weights = op_base.children[device_id].weights
    
    @staticmethod
    def list_tags(self) -> List[str]:
        return [""]
    
    def config(self, impl_tag, parameter_map = {}):
        self.impl_tag = impl_tag
        
    def run(self, *args, **kwargs):
        pass