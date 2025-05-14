from operations.operation_base import Operations, Operation_Device
from core.IOWrapper import IOWrapper
from core.IOWrapper import IOWrapper_Device

def link_allinputs_to_alloutputs(inputs, outputs):
    for _, input_wrapper in inputs.items():
        for _, output_wrapper in outputs.items():
            input_wrapper >> output_wrapper

class Copy(Operations):
    """Virtual copy operation for memory sharing between multiple consumers"""
    def __init__(self, name, num_inputs, num_outputs):
        super().__init__(name)
        self.name = name
        self.isVirtual = True
        self.isCopy = True
        self.isRedist = False
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.inputs = dict([(f"input_{i}", IOWrapper(self, f"input_{i}")) for i in range(num_inputs)])
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

class Copy_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.isCopy = parent.isCopy
        self.isRedist = parent.isRedist
        self.num_inputs = parent.num_inputs
        self.num_outputs = parent.num_outputs

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
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.inputs = dict(
            [(f"input_{i}", IOWrapper(self, f"input_{i}")) for i in range(num_inputs)]
        )
        self.outputs = dict(
            [(f"output_{i}", IOWrapper(self, f"output_{i}")) for i in range(num_outputs)]
        )
        link_allinputs_to_alloutputs(self.inputs, self.outputs)
        # link_firstinputs_to_alloutputs(self.inputs["input_0"], self.outputs)
        self.children = {}
        self.op_device = Redist_Device

    def checkConnection(self):
        for name, IOwrapper in self.inputs.items():
            if len(IOwrapper.prev) == 0:
                raise Exception(f"Operation {self.name}, Input {name} is not connected")
        for name, IOwrapper in self.outputs.items():
            if len(IOwrapper.next) == 0:
                raise Exception(f"Operation {self.name}, Output {name} is not connected")

    def set_input(self, updated_input):
        assert len(self.inputs) == 1, "Nano_Dist operation should have only one input"
        updated_input.owner = self
        updated_input.children = {}
        self.inputs["input_0"] = updated_input
        self.relink_inputs_and_outputs()

    def set_output(self, updated_output):
        assert len(self.outputs) == 1, "Nano_Dist operation should have only one output"
        updated_output.owner = self
        updated_output.children = {}
        self.outputs["output_0"] = updated_output
        self.relink_inputs_and_outputs()

    def clear_inputs_and_outputs_links(self):
        for input_wrapper in self.inputs.values():
            input_wrapper.next =[]
        for output_wrapper in self.outputs.values():
            output_wrapper.prev = []
            output_wrapper.prev_depend_on_prev_layer = []
            output_wrapper.prev_depend_on_next_layer = []

    def relink_inputs_and_outputs(self):
        for input_wrapper in self.inputs.values():
            input_wrapper.nano_dist_next = []
        for output_wrapper in self.outputs.values():
            output_wrapper.nano_dist_prev = []
            output_wrapper.nano_dist_prev_depend_on_prev_layer = []
        for input_wrapper in self.inputs.values():
            for output_wrapper in self.outputs.values():
                input_wrapper.nano_dist_next.append(output_wrapper)
                output_wrapper.nano_dist_prev.append(input_wrapper)
                output_wrapper.nano_dist_prev_depend_on_prev_layer.append(False)
                output_wrapper.nano_dist_prev_depend_on_next_layer.append(False)

class Redist_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.isCopy = parent.isCopy
        self.isRedist = parent.isRedist
        self.num_inputs = parent.num_inputs
        self.num_outputs = parent.num_outputs
       
    def setBatchSize(self, batch_size):
        pass
