from enum import Enum
import numpy as np
import torch
from collections import defaultdict

class IOBufferType(Enum):
    FULL = 1
    DiscontinousPartition = 2
    ContinousPartition = 3
    PartialSum = 4

# To Do: delete the attributes that related to tensor that should not belong to a base IOWrapper anymore
class IOWrapper:
    def __init__(self, owner, name, IOtype, dtype=torch.float16):
        self.owner = owner  # owner is now an Operations object or similar
        self.name = name
        self.prev = []
        self.next = []
        self.prev_depend_on_prev_layer = []
        self.shape = None # shape [0] is non-contiguous dimension, shape [1] is contiguous dimension
        self.ptr = 0
        self.IOtype = IOtype
        self.tensor: torch.Tensor = None
        self.transform = None
        self.tensor_offset = 0
        self.dtype = dtype
        self.real_deps = defaultdict(list)  # {curr_wrapper: [(real_prev_operation, prev_depend_on_prev_layer)]}
        self.children = []  # [IOWrapper_Device]
    
    @property
    def tensor_range(self):
        if self.shape is None:
            return None
        return (self.tensor_offset, self.tensor_offset + self.shape[0])
    
    @property
    def fullName(self):
        owner_name = self.owner.name if hasattr(self.owner, "name") else str(self.owner)
        return f"{owner_name}_{self.name}"
    
    def chain(self, next_wrapper, depend_on_prev=False):
        self.next.append(next_wrapper) if next_wrapper not in self.next else None # self.next prepared for memory allocation
        next_wrapper.prev.append(self) # self.prev prepared for executor graph
        next_wrapper.prev_depend_on_prev_layer.append(depend_on_prev)
        # check dtype must be the same
        if self.dtype != next_wrapper.dtype:
            raise Exception(f"Error: {self.fullName} and {next_wrapper.fullName} has different dtype")
        
        return next_wrapper
    
    def append_child(self, child_wrapper):
        self.children.append(child_wrapper)

    def __rshift__(self, next_wrapper):
        return self.chain(next_wrapper)
    
    def toStr(self):
        # name, prev = [], next = []
        return f"{self.fullName}, prev = {[p.fullName for p in self.prev]}, next = {[n.fullName for n in self.next]}"
    

class IOWrapper_Device:
    def __init__(self, owner, name, IOtype, dtype=torch.float16):
        self.owner = owner  # owner is now an Operations object or similar
        self.name = name
        self.IOtype = IOtype
        self.dtype = dtype
        self.shape = None # shape [0] is non-contiguous dimension, shape [1] is contiguous dimension
        self.tensor: torch.Tensor = None     
        self.tensor_offset = 0
    
    @property
    def tensor_range(self):
        if self.shape is None:
            return None
        return (self.tensor_offset, self.tensor_offset + self.shape[0])
    
    @property
    def fullName(self):
        owner_name = self.owner.name if hasattr(self.owner, "name") else str(self.owner)
        return f"{owner_name}_{self.name}"