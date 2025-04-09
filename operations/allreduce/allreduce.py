import torch
from operations.operation_base import Operations
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer

class AllReduce(Operations):
    def __init__(self, name, nranks):
        super().__init__(name)
        self.inputs = {
            "input": IOWrapper(self, 'input')
        }
        self.outputs = {
            "output": IOWrapper(self, 'output')
        }
        self.nranks = nranks
    
    def setShape(self, N):
        self.N = N
        self.inputs["input"].shape = (self.N,)
        self.outputs["output"].shape = (self.N,)
    
    def setBatchSize(self, M):
        self.M = M
        self.inputs["input"].shape = (self.M, self.N)
        self.outputs["output"].shape = (self.M, self.N)
    
    def run(self, layer):
        self.outputs["output"].tensor.copy_(self.inputs["input"].tensor.sum(dim=0) / self.nranks)
    
    def processWeight(self, global_weight_map, total_layers, cached=False):
        pass