from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType

class Redist(Operations):
    """Virtual redistribution operation for partition/aggregate"""
    def __init__(self, name, mode="partition"):
        super().__init__(name)
        self.isVirtual = True
        self.mode = mode
        self.inputs = {
            "input": IOWrapper(self, 'input', IOBufferType.FULL)
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.FULL)
        }

    def setBatchSize(self):
        if self.mode == "partition":
            if len(self.inputs["input"].prev) != 1:
                print(f"Redist partition intput operation '{self.name}' has no wrong input connections!\n")
                return False
            if len(self.outputs["output"].next) < 2:
                print(f"Redist partition output operation '{self.name}' has no wrong output connections!\n")
                return False
            
            # Get shape from the only predecessor
            self.inputs["input"].shape = self.inputs["input"].prev[0].shape
            self.outputs["output"].shape = self.inputs["input"].shape
        else:
            if len(self.inputs["input"].prev) < 2:
                print(f"Redist aggregate input operation '{self.name}' has no wrong input connections!\n")
                return False
            if len(self.outputs["output"].next) != 1:
                print(f"Redist aggregate operation '{self.name}' has no wrong output connections!\n")
                return False
            
            # Aggregate shape from predecessors
            total_rows = sum(n.shape[0] for n in self.inputs["input"].prev)
            remaining_shape = self.inputs["input"].prev[0].shape[1:]
            shape = (total_rows, *remaining_shape)
            self.inputs["input"].shape = shape
            self.outputs["output"].shape = shape

        return True