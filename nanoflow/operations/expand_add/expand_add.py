import torch
import time
import sqlite3

import nanoflow.platform_config as platform_config
from nanoflow.operations import Operations, Operation_Layer, OperationImpl
from nanoflow.core import IOWrapper, WeightWrapper, process_bias_list


class ExpandAddTorchImpl(OperationImpl):
    category_tag = "torch"

    def run(self, input, weight, output):
        with torch.cuda.stream(self.stream):
            output.copy_(input + weight)

        torch.cuda.synchronize()  # for debug
        print("ExpandAdd output:", output)


class ExpandAdd(Operations):
    def __init__(self, name, device, nano_idx=None):
        super().__init__(name, device, nano_idx)
        self.inputs = {
            "input": IOWrapper(self, "input", device).is_input(),
        }
        self.outputs = {
            "output": IOWrapper(self, "output", device).is_output(),
        }
        self.weights = {
            "weight": WeightWrapper(self),
        }
        self.M = None
        self.bias_map: dict[int, torch.Tensor] = {}
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = ExpandAdd_Layer

    def init_impl_map(self):
        self.add_impl(ExpandAddTorchImpl)

    def setShape(self, N):
        self.N = N
        self.weights["weight"].shape = self.N
        self.inputs["input"].init_shape((0, self.N))
        self.outputs["output"].init_shape((0, self.N))

        return self

    def copy_nano(self, index):
        new_op = ExpandAdd(self.name, self.device, nano_idx=index)
        new_op.set_category(self.category)
        new_op.weights = self.weights
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.N)

        self.nano_ops.append(new_op)

        return new_op

    def run(self, layer):
        with torch.cuda.stream(self.stream):
            current_M = self.inputs["input"].batch_size
            if self.M != current_M:
                self.M = current_M
                for l in self.layer_list:
                    self.bias_map[l] = (
                        self.weights["weight"].weight_map[l].expand(self.M, -1)
                    )

        self.impl.run(
            self.inputs["input"].tensor,
            self.bias_map[layer],
            self.outputs["output"].tensor,
        )

    def processWeight(self, global_weight_map, cached_weight_map, cached, device):
        process_bias_list(
            global_weight_map,
            self.weight_name,
            self.weights["weight"],
            self.layer_list,
            cached_weight_map,
            cached,
            device,
        )


class ExpandAdd_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        self.parent.run(self.layer)
