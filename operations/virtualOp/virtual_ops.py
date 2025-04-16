from operations.operation_base import Operations, Operation_Device
from core.IOWrapper import IOWrapper
from core.IOWrapper import IOWrapper_Device

def link_allinputs_to_alloutputs(inputs, outputs):
    for _, input_wrapper in inputs.items():
        for _, output_wrapper in outputs.items():
            input_wrapper >> output_wrapper

class Copy(Operations):
    """Virtual copy operation for memory sharing between multiple consumers"""
    def __init__(self, name, num_outputs):
        super().__init__(name)
        self.name = name
        self.isVirtual = True
        self.isCopy = True
        self.isRedist = False
        self.inputs = {
            "input": IOWrapper(self, "input")
        }
        self.outputs = dict([(f"output_{i}", IOWrapper(self, f"output_{i}")) for i in range(num_outputs)])
        link_allinputs_to_alloutputs(self.inputs, self.outputs)
        # link_firstinputs_to_alloutputs(self.inputs["input"], self.outputs)
        self.op_device = Copy_Device

    def checkConnection(self):
        for name, IOwrapper in self.inputs.items():
            if len(IOwrapper.prev) == 0:
                raise Exception(f"Operation {self.name}, Input {name} is not connected")
        for name, IOwrapper in self.outputs.items():
            if len(IOwrapper.next) == 0:
                print(f"Operation {self.name}, Output {name} is not connected")
                raise Exception(f"Operation {self.name}, Output {name} is not connected")

    def check(self):
        pass
        # if len(self.io.prev) == 0:
        #     raise Exception(f"Copy operation '{self.name}' has no prev connections!\n")
        # if len(self.io.next) < 2:
        #     raise Exception(f"Copy operation '{self.name}' has less than two next connections!\n")
        # if isinstance(self.io.prev[0].owner , Copy):
        #     raise Exception(f"Copy operation '{self.name}' connect with Copy '{self.io.prev[0].owner.name}' operation!\n")
        # # Get input shape from connected predecessor
        # self.io.shape = self.io.prev[0].shape

class Copy_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.isCopy = parent.isCopy
        self.isRedist = parent.isRedist

    def setBatchSize(self, batch_size):
        pass
        

class Redist(Operations):
    """Virtual redistribution operation for partition/aggregate"""
    def __init__(self, name, num_inputs, num_outputs):
        super().__init__(name)
        self.name = name
        self.isVirtual = True
        self.isCopy = False
        self.isRedist = True
        self.inputs = dict(
            [(f"input_{i}", IOWrapper(self, f"input_{i}")) for i in range(num_inputs)]
        )
        self.outputs = dict(
            [(f"output_{i}", IOWrapper(self, f"output_{i}")) for i in range(num_outputs)]
        )
        link_allinputs_to_alloutputs(self.inputs, self.outputs)
        # link_firstinputs_to_alloutputs(self.inputs["input_0"], self.outputs)
        self.children = []
        self.op_device = Redist_Device

    def checkConnection(self):
        for name, IOwrapper in self.inputs.items():
            if len(IOwrapper.prev) == 0:
                raise Exception(f"Operation {self.name}, Input {name} is not connected")
        for name, IOwrapper in self.outputs.items():
            if len(IOwrapper.next) == 0:
                raise Exception(f"Operation {self.name}, Output {name} is not connected")

    def check(self):
        pass
        # if self.mode == RedistMode.PARTITION:
        #     assert len(self.io.prev) == 1, \
        #         f"Redist partition input operation '{self.name}' must have exactly 1 input"
        #     assert len(self.io.next) >= 2, \
        #         f"Redist partition output operation '{self.name}' must have at least 2 outputs"
            
        #     if isinstance(self.io.prev[0].owner, Redist) and self.io.prev[0].owner.mode == RedistMode.PARTITION:
        #         raise Exception(f"Partition operation '{self.name}' connect with Partition '{self.io.prev[0].owner.name}' operation!\n")

        # elif self.mode == RedistMode.AGGREGATE:
        #     assert len(self.io.prev) >= 2, \
        #         f"Redist aggregate input operation '{self.name}' must have at least 2 inputs"
        #     assert len(self.io.next) == 1, \
        #         f"Redist aggregate output operation '{self.name}' must have exactly 1 output"
            
        #     if isinstance(self.io.prev[0].owner, Redist):
        #         raise Exception(f"Aggregate operation '{self.name}' connect with Redist '{self.io.prev[0].owner.name}' operation!\n")

class Redist_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.isCopy = parent.isCopy
        self.isRedist = parent.isRedist
       
    def setBatchSize(self, batch_size):
        pass
