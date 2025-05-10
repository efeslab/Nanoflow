class GEMM_DistN(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "A": IOWrapper(self, 'A', IOBufferType.FULL),
            "C": IOWrapper(self, 'C', IOBufferType.NPartition)
        }
        self.outputs = {
            "D": IOWrapper(self, 'D', IOBufferType.NPartition)
        }
        self.weights = {
            "B": IOWrapper(self, 'B', IOBufferType.NPartition)
        }
        
class AG(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "input": IOWrapper(self, 'input', IOBufferType.NPartition),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.NPartition)
        }

class WeightManager:
    def __init__(self):
        self.weights = {}
    
    def register(self, IOwrapper):
        self.weights[IOwrapper.fullName] = IOwrapper