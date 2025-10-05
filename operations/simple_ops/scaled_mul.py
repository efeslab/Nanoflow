import torch
import time
import sqlite3

import platform_config
from operations.operation_base import Operations, Operation_Layer
from core.IOWrapper import IOWrapper

from operations.impl_base import OperationImpl


class ScaledMulTorchImpl(OperationImpl):
    category_tag = "torch"

    def run(self, input_0, input_1, output):
        with torch.cuda.stream(self.stream):
            output.copy_(input_0 * input_1)


class ScaledMul(Operations):
    def __init__(self, name, device, nano_idx=None):
        super().__init__(name, device, nano_idx)
        self.inputs = {
            "input_0": IOWrapper(self, "input_0", device).is_input(),
            "input_1": IOWrapper(self, "input_1", device).is_input(),
        }
        self.outputs = {
            "output": IOWrapper(self, "output", device).is_output(),
        }
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = ScaledMul_Layer

    def init_impl_map(self):
        self.add_impl(ScaledMulTorchImpl)

    def setShape(self, N):
        self.N = N
        self.inputs["input_0"].init_shape((0, 1))
        self.inputs["input_1"].init_shape((0, self.N))
        self.outputs["output"].init_shape((0, self.N))

        return self

    def copy_nano(self, index):
        new_op = ScaledMul(self.name, self.device, nano_idx=index)
        new_op.set_category(self.category)
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.N)

        self.nano_ops.append(new_op)

        return new_op

    def run(self):
        self.impl.run(
            self.inputs["input_0"].tensor,
            self.inputs["input_1"].tensor,
            self.outputs["output"].tensor,
        )


class ScaledMul_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        self.parent.run()
