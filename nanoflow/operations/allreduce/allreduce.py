import torch
import torch.distributed as dist

from nanoflow.operations import Operations, Operation_Layer, OperationImpl
from nanoflow.core.IOWrapper import IOWrapper
from nanoflow.pybind.build.bind_all_reduce import NCCLWrapper


class AllReduceTorchImpl(OperationImpl):
    category_tag = "torch"

    def __init__(self, op_base, stream, device):
        super().__init__(op_base, stream, device)
        self.subgroup = op_base.subgroup  # for torch allreduce

    def run(self, input, output):
        with torch.cuda.stream(self.stream):
            work = dist.all_reduce(
                input, op=dist.ReduceOp.SUM, group=self.subgroup, async_op=True)
            work.wait()

            output.copy_(input)


class AllReduceNCCLImpl(OperationImpl):
    category_tag = "nccl"

    def __init__(self, op_base, stream, device):
        super().__init__(op_base, stream, device)
        self.nccl_wrapper = op_base.nccl_wrapper  # for nccl allreduce

    def run(self, input, output):
        with torch.cuda.stream(self.stream):
            self.nccl_wrapper.all_reduce(input, output, "sum")


class AllReduce(Operations):
    def __init__(self, name, device, nano_idx=None):
        super().__init__(name, device, nano_idx)
        self.inputs = {"input": IOWrapper(self, "input", device).is_input()}
        self.outputs = {"output": IOWrapper(
            self, "output", device).is_output()}
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = AllReduce_Layer
        self.nccl_wrapper = None

    def init_impl_map(self):
        self.add_impl(AllReduceTorchImpl)
        self.add_impl(AllReduceNCCLImpl)

    def setShape(self, N, rank, world_size):
        self.N = N
        self.rank = rank
        self.world_size = world_size
        self.inputs["input"].init_shape((0, self.N))
        self.outputs["output"].init_shape((0, self.N))

        return self

    def update(self, subgroup, rank, world_size, unique_nccl_ids):
        self.subgroup = subgroup
        self.rank = rank
        self.world_size = world_size
        self.unique_nccl_ids = unique_nccl_ids
        # print("unique_nccl_id:", self.unique_nccl_id)
        if unique_nccl_ids is not None:
            self.nccl_wrapper = NCCLWrapper(
                self.rank, self.world_size, self.unique_nccl_ids[0]
            )

    def copy_nano(self, index):
        new_op = AllReduce(self.name, self.device, nano_idx=index)
        new_op.set_category(self.category)
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.N, self.rank, self.world_size)
        if self.unique_nccl_ids is not None:
            new_op.update(
                self.subgroup,
                self.rank,
                self.world_size,
                self.unique_nccl_ids[index + 1: index + 2],
            )
        else:
            new_op.update(self.subgroup, self.rank, self.world_size, None)
        # new_op.nccl_wrapper = self.nccl_wrapper

        self.nano_ops.append(new_op)

        return new_op

    def init_profile_db(self):
        for _, impl in self.impl_map.items():
            self.cursor.execute(
                f"""
            CREATE TABLE IF NOT EXISTS "{impl.category_tag}" (
                id           INTEGER PRIMARY KEY AUTOINCREMENT, 
                batch_size   INTEGER,
                sm_count INTEGER,
                N INTEGER,
                average_time_ms REAL
            );
            """
            )

    def store_profile_db(self, category_tag, impl_tag, average_elapsed_ms):
        print(
            f"Name: {self.name}, Category: {category_tag}, Batch Size: {self.batch_size}, Average Time: {average_elapsed_ms} ms"
        )
        self.cursor.execute(
            f"""
            INSERT OR IGNORE INTO {category_tag} (batch_size, sm_count, N, average_time_ms)
            VALUES (?, ?, ?, ?);
        """,
            (self.batch_size, self.sm_count, self.N, average_elapsed_ms),
        )

    def run(self):
        self.impl.run(self.inputs["input"].tensor,
                      self.outputs["output"].tensor)

    def profile_run(self):
        self.run()


class AllReduce_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer, op_device)

    def run(self):
        self.parent.run()
