import torch
import torch.distributed as dist

import nanoflow.platform_config as platform_config
from nanoflow.utils.prof_marker import prof_marker
from nanoflow.operations import Operations, Operation_Layer, OperationImpl
from nanoflow.core.IOWrapper import IOWrapper
from nanoflow.pybind.build.bind_all_reduce import NCCLWrapper


class AllReduceTorchImpl(OperationImpl):
    category_tag = "torch"

    def __init__(self, op_base, stream, device):
        super().__init__(op_base, stream, device)
        self.tp_size = op_base.tp_size
        self.subgroup = op_base.subgroup
        self.rank = op_base.rank
        self.world_size = op_base.tp_size
        self.N = op_base.N
        self.nccl_wrapper = op_base.nccl_wrapper

    def run(self, input, output):
        with torch.cuda.stream(self.stream):
            # self.nccl_wrapper.all_reduce_inplace(input, "sum")

            # work = dist.all_reduce(input, op=dist.ReduceOp.SUM, group=self.subgroup, async_op=True)
            # work.wait()

            # output.copy_(input)
            self.nccl_wrapper.all_reduce(input, output, "sum")


class AllReduce(Operations):
    def __init__(self, name, device, nano_idx=None):
        super().__init__(name, device, nano_idx)
        self.inputs = {"input": IOWrapper(self, "input", device).is_input()}
        self.outputs = {"output": IOWrapper(self, "output", device).is_output()}
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = AllReduce_Layer
        self.nccl_wrapper = None

    def init_impl_map(self):
        self.add_impl(AllReduceTorchImpl)

    def setShape(self, N, tp_idx, tp_size):
        self.N = N
        self.tp_idx = tp_idx
        self.tp_size = tp_size
        self.inputs["input"].init_shape((0, self.N))
        self.outputs["output"].init_shape((0, self.N))

    def update(self, subgroup, rank, tp_size, unique_nccl_ids):
        self.subgroup = subgroup
        self.rank = rank
        self.tp_size = tp_size
        self.unique_nccl_ids = unique_nccl_ids
        # print("unique_nccl_id:", self.unique_nccl_id)
        if unique_nccl_ids is not None:
            self.nccl_wrapper = NCCLWrapper(
                self.rank, self.tp_size, self.unique_nccl_ids[0]
            )

    def copy_nano(self, index):
        new_op = AllReduce(self.name, self.device, nano_idx=index)
        new_op.set_category(self.category)
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.N, self.tp_idx, self.tp_size)
        if self.unique_nccl_ids is not None:
            new_op.update(
                self.subgroup,
                self.rank,
                self.tp_size,
                self.unique_nccl_ids[index + 1 : index + 2],
            )
        else:
            new_op.update(self.subgroup, self.rank, self.tp_size, None)
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

    # def profile_all(self):
    #     with prof_marker(f"batchsize:{self.batch_size}"):
    #         start = torch.cuda.Event(enable_timing=True)
    #         end = torch.cuda.Event(enable_timing=True)

    #         for _, impl in self.impl_map.items():
    #             self.impl = impl(self, self.stream, self.device)
    #             category_tag = impl.category_tag
    #             is_profiled = self.is_profiled_in_db(category_tag)
    #             if is_profiled:
    #                 # If already profiled, we can skip profiling
    #                 continue
    #             # loop in the impl configs
    #             for impl_tag, para_map in self.impl_configs_map[category_tag]:
    #                 self.impl.config(impl_tag, para_map)
    #                 self.profile_update()

    #                 # warm up for 10 cycles.
    #                 for _ in range(10):
    #                     self.profile_run()

    #                 rounds = 100
    #                 start.record(self.stream)
    #                 with torch.cuda.stream(self.stream):
    #                     for round in range(rounds):
    #                         self.profile_run()
    #                 end.record(self.stream)
    #                 torch.cuda.synchronize()
    #                 elapsed_ms = start.elapsed_time(end)
    #                 average_elapsed_ms = elapsed_ms / rounds
    #                 # Store to results
    #                 if self.is_save_db:
    #                     self.store_profile_db(category_tag, impl_tag, average_elapsed_ms)
    #         self.conn.commit()

    def run(self):
        self.impl.run(self.inputs["input"].tensor, self.outputs["output"].tensor)

    def profile_run(self):
        self.run()


class AllReduce_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer, op_device)

    def run(self):
        self.parent.run()
