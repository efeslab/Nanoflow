import torch
import torch.distributed as dist

from nanoflow.operations import Operations, Operation_Layer, OperationImpl
from nanoflow.core.IOWrapper import IOWrapper
from nanoflow.utils.prof_marker import prof_marker


class AllGatherTorchImpl(OperationImpl):
    category_tag = "torch"

    def __init__(self, op_base, stream, device):
        super().__init__(op_base, stream, device)
        self.world_size = op_base.world_size
        self.subgroup = op_base.subgroup
        self.N = op_base.N
        self.gather_list = [torch.empty((self.batch_size, self.N // self.world_size),
                                        dtype=torch.float16, device=device) for _ in range(self.world_size)]

    def run(self, input, output):
        with torch.cuda.stream(self.stream):
            with prof_marker(f"AllGatherTorchImpl.run all_gather"):
                dist.all_gather(self.gather_list, input, group=self.subgroup)
            with prof_marker(f"AllGatherTorchImpl.run cat"):
                out = torch.cat(self.gather_list, dim=1)
            with prof_marker(f"AllGatherTorchImpl.run copy"):
                output.copy_(out)


class AllGather(Operations):
    def __init__(self, name, device, nano_idx=None):
        super().__init__(name, device, nano_idx)
        self.inputs = {
            "input": IOWrapper(self, 'input', device).is_input()
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', device).is_output()
        }
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = AllGather_Layer

    def init_impl_map(self):
        self.add_impl(AllGatherTorchImpl)

    def setShape(self, N, rank, world_size):
        self.N = N
        self.rank = rank
        self.world_size = world_size
        self.inputs["input"].init_shape((0, self.N // self.world_size))
        self.outputs["output"].init_shape((0, self.N))

        return self

    def update(self, subgroup):
        self.subgroup = subgroup

    def run(self):
        self.impl.run(self.inputs["input"].tensor,
                      self.outputs["output"].tensor)

    def profile_run(self):
        self.run()
class AllGather_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer, op_device)

    def run(self):
        self.parent.run()
