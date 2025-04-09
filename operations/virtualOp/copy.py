from operations.operation_base import Operations
from core.IOWrapper import IOWrapper
from core.IOWrapper import IOWrapper_Device


class Copy(Operations):
    """Virtual copy operation for memory sharing between multiple consumers"""
    def __init__(self, name, num_outputs=1):
        super().__init__(name)
        self.name = name
        self.isVirtual = True
        self.inputs = {
            "input": IOWrapper(self, "input")
        }
        self.outputs = dict([(f"output_{i}", IOWrapper(self, f"output_{i}")) for i in range(num_outputs)])
        self.io = IOWrapper(self, "io")
        self.op_device = Copy_Device
        
    def check(self):
        if len(self.io.prev) == 0:
            raise Exception(f"Copy operation '{self.name}' has no prev connections!\n")
        if len(self.io.next) < 2:
            raise Exception(f"Copy operation '{self.name}' has less than two next connections!\n")
        if isinstance(self.io.prev[0].owner , Copy):
            raise Exception(f"Copy operation '{self.name}' connect with Copy '{self.io.prev[0].owner.name}' operation!\n")
        # Get input shape from connected predecessor
        self.io.shape = self.io.prev[0].shape

class Copy_Device(Copy):
    def __init__(self, op_general, name, device):
        super().__init__(name)
        self.base_io = op_general.io
        self.io = IOWrapper_Device(owner=self,name=self.base_io.name, dtype=self.base_io.dtype)
        self.base_io.append_child(self.io)

    def setBatchSize(self, wrapper):
        self.io.shape = wrapper.shape