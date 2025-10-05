import platform
import torch
import math
import time

import platform_config
from operations.operation_base import Operations, Operation_Layer
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper
from operations.impl_base import OperationImpl
from kvcache.kv import KVCacheNone, KVCacheTorch, DistKVPool, BatchedDistKVCache
from utils.prof_marker import prof_marker
from utils.util_functions import tensor_offset_to_req_idx

        
if platform_config.PLATFORM_CUDA:
    import bind_ropeappend
    class RopeAppendCudaImpl(OperationImpl):
        category_tag = "cuda"
        def __init__(self, op_base, stream, device):
            super().__init__(op_base, stream, device)
            self.num_qo_heads = op_base.num_qo_heads // op_base.tp_size
            self.rope_scale = op_base.factor
            self.rope_theta = op_base.theta

        def run(self, kqv, k_data, v_data, output):
            with prof_marker("RopeAppendCuda: SplitRopeAppend"):
                bind_ropeappend.splitRopeAppend(
                    k_data,
                    v_data,
                    kqv,
                    output,
                    self.op_base.rev_input_indptr,
                    self.op_base.per_token_offset,
                    self.num_qo_heads,
                    self.rope_scale,
                    self.rope_theta,
                    0.0,
                    0.0,
                    self.stream.cuda_stream
                )

class RopeAppendFlashinfer(Operations):
    def __init__(
        self,
        name,
        device,
        theta,
        factor=1.0,
        nano_idx=None
    ):
        """
        Args:
            name (str): The name of this operator.
            theta (float): The base used to compute the inverse frequency (typically set from config.rope_theta).
            factor (float): Scaling factor used in llama3.
        """
        super().__init__(name, device, nano_idx)
        self.inputs = {"kqv": IOWrapper(self, "kqv", device).is_input()}
        self.outputs = {"q": IOWrapper(self, "q", device).is_output()}
        self.externals: dict[str, BatchedDistKVCache | KVCacheNone]
        
        # Save RoPE configuration.
        self.theta = theta  # typically config.rope_theta
        self.factor = factor

        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = RopeAppendFlashinfer_Layer

    def init_impl_map(self):
        if platform_config.PLATFORM_CUDA:
            self.add_impl(RopeAppendCudaImpl)

    def setShape(self, num_kv_heads, num_qo_heads, head_dim, tp_size=1):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.tp_size = tp_size
        self.inputs["kqv"].init_shape((
            0,
            (self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim // self.tp_size,
        ))
        # The output "q" has shape [batch_size, num_qo_heads * head_dim]
        self.outputs["q"].init_shape((0, self.num_qo_heads * self.head_dim // self.tp_size))

        return self

    def update(self, qo_indicies, decode_batchsize):
        if self.isNanoSplit:
            for nano_op in self.nano_ops:
                nano_op.update(qo_indicies, decode_batchsize)
        else:
            """Stores the starting indices for the query/key segments."""
            io = self.inputs["kqv"]
            self.qo_indicies = qo_indicies
            self.kv_indptr =  self.externals["KVCache"].kv_indptr
            self.kv_indices = self.externals["KVCache"].kv_indices
            self.kv_last_page_len = self.externals["KVCache"].kv_last_page_len

            self.rev_input_indptr = self.externals["KVCache"].rev_input_indptr[io.tensor_offset: io.tensor_offset + io.batch_size]
            self.per_token_offset = self.externals["KVCache"].per_token_offset[io.tensor_offset: io.tensor_offset + io.batch_size]
            self.page_size = self.externals["KVCache"].page_size
            self.decode_batchsize = decode_batchsize

            bind_ropeappend.updateKVCache(self.kv_indptr, self.kv_indices, self.kv_last_page_len, len(self.kv_last_page_len), self.page_size, self.num_kv_heads // self.tp_size, self.head_dim)

    def copy_nano(self, index):
        new_op = RopeAppendFlashinfer(self.name, self.device, self.theta, self.factor, nano_idx=index)
        new_op.set_category(self.category)
        new_op.externals = self.externals
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim, self.tp_size)
        self.nano_ops.append(new_op)

        return new_op

    def setup_profile_custom(self):
        super().setup_profile_custom()
        self.k_ptr, self.v_ptr = self.externals["KVCache"].get_whole_kv_data(self.layer_list[0])
        

    def init_profile_db(self):
        for _, impl in self.impl_map.items():
            self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS "{impl.category_tag}" (
                id           INTEGER PRIMARY KEY AUTOINCREMENT, 
                batch_size   INTEGER,
                sm_count INTEGER,
                head_dim INTEGER,
                num_qo_heads INTEGER,
                num_kv_heads INTEGER,
                average_time_ms REAL
            );
            ''')

    def store_profile_db(self, category_tag, impl_tag, average_elapsed_ms):
        print(f"Name: {self.name}, Category: {category_tag}, Batch Size: {self.batch_size}, Average Time: {average_elapsed_ms} ms")
        self.cursor.execute(f'''
            INSERT OR IGNORE INTO {category_tag} (batch_size, sm_count, head_dim, num_qo_heads, num_kv_heads, average_time_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (self.batch_size, self.sm_count, self.head_dim, self.num_qo_heads, self.num_kv_heads, average_elapsed_ms))

    def run(self, k_ptr, v_ptr):
        self.impl.run(self.inputs["kqv"].tensor, k_ptr, v_ptr, self.outputs["q"].tensor)
    
    def profile_run(self):
        self.run(self.k_ptr, self.v_ptr)

class RopeAppendFlashinfer_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)
        self.k_data_ptr, self.v_data_ptr = base_op.externals["KVCache"].get_whole_kv_data(self.layer)

    def run(self):
        self.parent.run(
            self.k_data_ptr,
            self.v_data_ptr
        )
        