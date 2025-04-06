from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.IOWrapper import IOWrapper_Device


class Copy(Operations):
    """Virtual copy operation for memory sharing between multiple consumers"""
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.isVirtual = True
        self.io = IOWrapper(self, name, IOBufferType.FULL)
        
    def setBatchSize(self):
        if len(self.io.prev) == 0:
            raise Exception(f"Copy operation '{self.name}' has no prev connections!\n")
        if len(self.io.next) < 2:
            raise Exception(f"Copy operation '{self.name}' has less than two next connections!\n")
        if isinstance(self.io.prev[0].owner , Copy):
            raise Exception(f"Copy operation '{self.name}' connect with Copy '{self.io.prev[0].owner.name}' operation!\n")
        # Get input shape from connected predecessor
        self.io.shape = self.io.prev[0].shape

    def expand_gpu(self, gpu_list):
        for i in gpu_list:
            i_str = str(i)
            name = self.name + "_" + i_str
            op_device = Copy_Device(self, self.name, i)
            self.children.append(op_device)
        
        return self.children

class Copy_Device(Copy):
    def __init__(self, op_general, name, device):
        super().__init__(name)
        base_io = op_general.io
        dev_io = IOWrapper_Device(owner=self,name=base_io.name, IOtype=base_io.IOtype, dtype=base_io.dtype)
        base_io.children.append(dev_io)