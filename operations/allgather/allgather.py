import torch
from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer

class AllGather(Operations):
    def __init__(self, name, nranks):
        super().__init__(name)
        self.inputs = {
            "input": IOWrapper(self, 'input', IOBufferType.DiscontinousPartition)
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.FULL)
        }
        self.nranks = nranks
    
    def setShape(self, N):
        self.N = N
        self.inputs["input"].shape = (self.N // self.nranks, self.nranks)
        self.outputs["output"].shape = (self.N,)
    
    def setBatchSize(self, M):
        self.M = M
        self.inputs["input"].shape = (self.M, self.N // self.nranks)
        self.outputs["output"].shape = (self.M, self.N)
    
    def run(self, layer):
        self.outputs["output"].tensor.copy_(self.inputs["input"].tensor.repeat(1, self.nranks))
    
    def processWeight(self, global_weight_map, total_layers, cached=False):
        pass
