import torch
from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
from operations.impl_base import OperationImpl


class AllGatherTorch(OperationImpl):
    category_tag = "torch"
    def run(self, input, output):
        output.copy_(input.repeat(1, self.nranks))


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
        self.impl_map = {}
        self.init_impl_map()
    
    def init_impl_map(self):
        self.add_impl(AllGatherTorch)

    def setShape(self, N):
        self.N = N
        self.inputs["input"].shape = (self.N // self.nranks, self.nranks)
        self.outputs["output"].shape = (self.N,)
    
    def setBatchSize(self, M):
        self.M = M
        self.inputs["input"].shape = (self.M, self.N // self.nranks)
        self.outputs["output"].shape = (self.M, self.N)
    
    def run(self, layer):
        self.impl.run(self.inputs["input"].tensor, self.outputs["output"].tensor)
        # self.outputs["output"].tensor.copy_(self.inputs["input"].tensor.repeat(1, self.nranks))
    
    def processWeight(self, global_weight_map, total_layers, cached=False):
        pass
