from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType

class Copy(Operations):
    """Virtual copy operation for memory sharing between multiple consumers"""
    def __init__(self, name):
        super().__init__(name)
        self.isVirtual = True
        self.inputs = {
            "input": IOWrapper(self, 'input', IOBufferType.FULL)
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.FULL)
        }
        
    def setBatchSize(self):
        if len(self.inputs["input"].prev) == 0:
            print(f"Copy operation input '{self.name}' has no input connections!\n")
            return False
        if len(self.outputs["output"].next) < 2:
            print(f"Copy operation output '{self.name}' has less than two output connections!\n")
            return False
        
        # Get input shape from connected predecessor
        self.inputs["input"].shape = self.inputs["input"].prev[0].shape
        self.outputs["output"].shape = self.inputs["input"].shape
        return True
