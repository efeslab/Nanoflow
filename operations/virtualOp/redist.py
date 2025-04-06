from enum import Enum
from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.IOWrapper import IOWrapper_Device


class RedistMode(Enum):
    PARTITION = "partition"
    AGGREGATE = "aggregate"

class Redist(Operations):
    """Virtual redistribution operation for partition/aggregate"""
    def __init__(self, name, mode=RedistMode.PARTITION):
        super().__init__(name)
        self.name = name
        self.isVirtual = True
        self.mode = mode
        self.io = IOWrapper(self, name, IOBufferType.FULL)
        self.children = []

    def setBatchSize(self):
        if self.mode == RedistMode.PARTITION:
            assert len(self.io.prev) == 1, \
                f"Redist partition input operation '{self.name}' must have exactly 1 input"
            assert len(self.io.next) >= 2, \
                f"Redist partition output operation '{self.name}' must have at least 2 outputs"
            
            if isinstance(self.io.prev[0].owner, Redist) and self.io.prev[0].owner.mode == RedistMode.PARTITION:
                raise Exception(f"Partition operation '{self.name}' connect with Partition '{self.io.prev[0].owner.name}' operation!\n")
            # Get shape from the only predecessor
            self.io.shape = self.io.prev[0].shape

        elif self.mode == RedistMode.AGGREGATE:
            assert len(self.io.prev) >= 2, \
                f"Redist aggregate input operation '{self.name}' must have at least 2 inputs"
            assert len(self.io.next) == 1, \
                f"Redist aggregate output operation '{self.name}' must have exactly 1 output"
            
            if isinstance(self.io.prev[0].owner, Redist):
                raise Exception(f"Aggregate operation '{self.name}' connect with Redist '{self.io.prev[0].owner.name}' operation!\n")
            # Aggregate shape from predecessors
            total_rows = sum(n.shape[0] for n in self.io.prev)
            remaining_shape = self.io.prev[0].shape[1:]
            self.io.shape = (total_rows, *remaining_shape)


    def expand_gpu(self, gpu_list):
        for i in gpu_list:
            i_str = str(i)
            name = self.name + "_" + i_str
            op_device = Redist_Device(self, self.name, i)
            self.children.append(op_device)
        
        return self.children

class Redist_Device(Redist):
    def __init__(self, op_general, name, device):
        super().__init__(name)
        base_io = op_general.io
        dev_io = IOWrapper_Device(owner=self,name=base_io.name, IOtype=base_io.IOtype, dtype=base_io.dtype)
        base_io.children.append(dev_io)
       
       
