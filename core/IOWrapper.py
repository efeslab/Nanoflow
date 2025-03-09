from enum import Enum
import numpy as np
import torch

class IOBufferType(Enum):
    FULL = 1
    DiscontinousPartition = 2
    ContinousPartition = 3
    PartialSum = 4

class IOBufferTransform(Enum):
    Replicate = 1
    Partition = 2
    Aggregate = 3

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
        self.child = []
        self.transform = None
        self.tensor_offset = 0
        self.dtype = dtype
        
    
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
        self.next.append(next_wrapper)
        next_wrapper.prev.append(self)
        next_wrapper.prev_depend_on_prev_layer.append(depend_on_prev)
        # check dtype must be the same
        if self.dtype != next_wrapper.dtype:
            raise Exception(f"Error: {self.fullName} and {next_wrapper.fullName} has different dtype")
        
        return next_wrapper
    
    def __rshift__(self, next_wrapper):
        return self.chain(next_wrapper)
    
    def toStr(self):
        # name, prev = [], next = []
        return f"{self.fullName}, prev = {[p.fullName for p in self.prev]}, next = {[n.fullName for n in self.next]}"
    
    def checkCorrectPartition(self):
        # actually only three case: same io, aggregate, partition. Cannot discontious partition

        # case 1: No next wrapper
        if len(self.next) == 0:
            return True
        
        # pre-check: all next types are the same (differ from self is OK)
        if not (all(n.IOtype == self.next[0].IOtype for n in self.next)):
            print(f"Error: {self.fullName} has different IOtype with its next")
            for wrapper in self.next:
                print(f"  {wrapper.fullName}, IOtype = {wrapper.IOtype}")
            return False
        
        next_type = self.next[0].IOtype
        
        # case 2: same IO type, same size
        if next_type == self.IOtype:
            if all(n.shape == self.shape for n in self.next):
                self.transform = IOBufferTransform.Replicate
                return True
            else:
                print(f"[Replicate Failure] {self.fullName}: Expected shape {self.shape}")
                for n in self.next:
                    print(f"  {n.fullName}, shape = {n.shape}")
                return False

        # case 3: Partition
        if self.IOtype == IOBufferType.FULL and next_type == IOBufferType.ContinousPartition:
            total_rows = sum(n.shape[0] for n in self.next)
            if total_rows == self.shape[0]:
                self.child = self.next
                self.transform = IOBufferTransform.Partition
                return True
            else:
                print(f"[ContinousPartition Failure] {self.fullName}: Sum of partition rows is {total_rows} but expected {self.shape[0]}.")
                for n in self.next:
                    print(f"  {n.fullName}, shape = {n.shape}")
                return False
            
            
        # case 3: aggregate
        if self.IOtype == IOBufferType.ContinousPartition and next_type == IOBufferType.FULL:
            # consider all next wrappers' previous wrappers
            for n in self.next:
                for p in n.prev:
                    if p.IOtype != IOBufferType.ContinousPartition:
                        print(f"[ContinousAggregation Failure] {self.fullName}: Expected all next wrappers' previous wrappers to be ContinousPartition.")
                        return False
                if sum(p.shape[0] for p in n.prev) != n.shape[0]:
                    print(f"[ContinousAggregation Failure] {self.fullName}: Sum of partition rows is {sum(p.shape[0] for p in n.prev)} but expected {n.shape[0]}.")
                    for p in n.prev:
                        print(f"  {p.fullName}, shape = {p.shape}")
                    return False
                n.child = n.prev
            self.transform = IOBufferTransform.Aggregate
            return True
        
        # Fallback: configuration does not match any recognized partition scheme.
        next_types = [n.IOtype for n in self.next]
        next_shapes = [n.shape for n in self.next]
        print(f"[Partition Check Failure] {self.fullName}: Unrecognized partition configuration.")
        print(f"Self IOtype: {self.IOtype}, shape: {self.shape}")
        print(f"Next partitions IOtypes: {next_types}")
        print(f"Next partitions shapes: {next_shapes}")
        return False