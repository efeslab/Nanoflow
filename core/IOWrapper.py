from enum import Enum
import numpy as np
import torch
from collections import defaultdict

# To Do: delete the attributes that related to tensor that should not belong to a base IOWrapper anymore
class IOWrapper:
    def __init__(self, owner, name, dtype=torch.float16):
        self.owner = owner  # owner is now an Operations object or similar
        self.name = name
        self.prev = []
        self.next = []
        self.prev_depend_on_prev_layer = []
        self.shape = None # shape [0] is non-contiguous dimension, shape [1] is contiguous dimension
        self.ptr = 0
        self.transform = None
        self.dtype = dtype
        self.real_deps = defaultdict(list)  # {curr_wrapper: [(real_prev_operation, prev_depend_on_prev_layer)]}
        self.children = []  # [IOWrapper_Device]
    
    @property
    def fullName(self):
        owner_name = self.owner.name if hasattr(self.owner, "name") else str(self.owner)
        return f"{owner_name}_{self.name}"
    
    def chain(self, next_wrapper, depend_on_prev):
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
        depend_on_prev = False
        if isinstance(next_wrapper, tuple):
            next_wrapper, depend_on_prev = next_wrapper
        print("IOWrapper __rshift__", depend_on_prev)
        return self.chain(next_wrapper, depend_on_prev)
    
    def toStr(self):
        # name, prev = [], next = []
        return f"{self.fullName}, prev = {[p.fullName for p in self.prev]}, next = {[n.fullName for n in self.next]}"
    

class IOWrapper_Device:
    def __init__(self, owner, name, device_id, dtype=torch.float16, base_wrapper=None):
        self.owner = owner  # owner is now an Operations object or similar
        self.name = name
        self.device_id = device_id
        self.dtype = dtype
        self.base_wrapper = base_wrapper
        self.tensor_shape = None # shape [0] is non-contiguous dimension, shape [1] is contiguous dimension
        self.batch_size = None
        self.whole_buffer: torch.Tensor = None
        # self.tensor: torch.Tensor = None     
        self.tensor_offset = 0
        self.is_input_wrapper = False
        self.is_output_wrapper = False
    
    def set_whole_buffer(self, buffer):
        self.whole_buffer = buffer

    def set_tensor_offset(self, offset):
        self.tensor_offset = offset

    def init_shape(self, shape):
        self.tensor_shape = shape

    def is_input(self):
        self.is_input_wrapper = True
        return self

    def is_output(self):
        self.is_output_wrapper = True
        return self

    @property
    def shape(self):
        if self.batch_size is None or self.tensor_shape is None:
            return None
        return (self.batch_size, *self.tensor_shape[1:])

    @property
    def tensor(self):
        if self.whole_buffer is None:
            return None
        return self.whole_buffer[self.tensor_offset : self.tensor_offset + self.batch_size]

    @property
    def fullName(self):
        owner_name = self.owner.name if hasattr(self.owner, "name") else str(self.owner)
        return f"{owner_name}_{self.name}"

    @property
    def next(self):
        return [op_base.children[self.device_id] for op_base in self.base_wrapper.next]
    
    @property
    def prev(self):
        return [op_base.children[self.device_id] for op_base in self.base_wrapper.prev]
    
    @property
    def prev_depend_on_prev_layer(self):
        return self.base_wrapper.prev_depend_on_prev_layer