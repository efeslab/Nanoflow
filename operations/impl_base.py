from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from operations.operation_base import Operations

class OperationImpl:
    category_tag: str
    def __init__(self, op_base: "Operations", stream, device):
        self.impl_tag = None
        self.op_base = op_base
        self.stream = stream
        self.stream_handle = stream.cuda_stream if hasattr(stream, 'cuda_stream') else 0
        self.device = device
        self.batch_size = op_base.batch_size
        self.inputs = op_base.inputs
        self.outputs = op_base.outputs
        self.weights = op_base.weights
    
    @staticmethod
    def list_tags() -> List[str]:
        return [""]
    
    def config(self, impl_tag, parameter_map = {}):
        self.impl_tag = impl_tag
        
    def run(self, *args, **kwargs):
        pass