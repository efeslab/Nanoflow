import torch
from transformers import AutoTokenizer
from nanoflow.operations import Operations, Operation_Layer
from nanoflow.core import IOWrapper, WeightWrapper


class GlobalInput(Operations):
    def __init__(self, name, device):
        super().__init__(name, device)
        self.inputs = {
            # "new_token": IOWrapper(self, 'new_token', dtype=torch.int32)
        }
        self.outputs = {
            "tokens": IOWrapper(self, "tokens", device, dtype=torch.int32).is_output()
        }
        self.op_layer = GlobalInput_Layer

    def setShape(self):
        self.outputs["tokens"].init_shape((0,))

        return self

    def init_profile_db(self):
        pass

    def store_profile_db(self, category_tag, impl_tag, average_elapsed_ms):
        pass

    def run(self):
        pass

    def profile_run(self):
        pass


class GlobalInput_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        self.parent.run()


class GlobalOutput(Operations):
    def __init__(self, name, device):
        super().__init__(name, device)
        self.inputs = {
            "tokens": IOWrapper(self, "tokens", device, dtype=torch.int32).is_input(),
        }
        self.outputs = {
            "new_token": IOWrapper(
                self, "new_token", device, dtype=torch.int32
            ).is_output()
        }
        self.op_layer = GlobalOutput_Layer

    def setShape(self):
        self.inputs["tokens"].init_shape((0,))
        self.outputs["new_token"].init_shape((0,))

        return self

    def init_profile_db(self):
        pass

    def store_profile_db(self, category_tag, impl_tag, average_elapsed_ms):
        pass

    def run(self):
        pass

    def profile_run(self):
        pass


class GlobalOutput_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        self.parent.run()
