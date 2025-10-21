import logging
import torch
import numpy as np
import time

import nanoflow.platform_config as platform_config
from nanoflow.operations import Operations, Operation_Layer, OperationImpl

from nanoflow.core import IOWrapper, WeightWrapper
from nanoflow.kvcache.kv import KVCacheNone, KVCacheTorch, DistKVPool, BatchedDistKVCache
from nanoflow.utils.util_functions import tensor_offset_to_req_idx
from nanoflow.utils.prof_marker import prof_marker

if platform_config.PLATFORM_CUDA:
    import flashinfer

    class DecAttnBatchedCudaImpl(OperationImpl):
        category_tag = "batched_cuda"

        def __init__(self, op_base, stream, device):
            super().__init__(op_base, stream, device)
            self.workspace_buffer = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8).to(self.device)
            self.wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                float_workspace_buffer=self.workspace_buffer, kv_layout="HND", use_cuda_graph=False, use_tensor_cores=True
            )
            self.num_qo_heads = op_base.num_qo_heads // op_base.tp_size
            self.num_kv_heads = op_base.num_kv_heads // op_base.tp_size
            self.head_dim = op_base.head_dim
            # print("DecAttnBatchedCudaImpl initialized with cuda stream:", self.stream.cuda_stream)

        def plan(self, kv_indptr, kv_indices, kv_last_page_len, page_size):
            with prof_marker("DecAttnBatchedCudaImpl.plan"):
                with torch.cuda.stream(self.stream):
                    # print("DecAttnBatchedCudaImpl.plan")
                    # print("kv_indptr: ", kv_indptr)
                    # print("kv_indices: ", kv_indices)
                    # print("kv_last_page_len: ", kv_last_page_len)
                    # print("page_size: ", page_size)
                    # print("num_qo_heads: ", self.num_qo_heads)
                    # print("num_kv_heads: ", self.num_kv_heads)
                    # print("head_dim: ", self.head_dim)

                    self.wrapper.plan(
                        kv_indptr,
                        kv_indices,
                        kv_last_page_len,
                        self.num_qo_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        page_size,
                        logits_soft_cap=0.0,
                        pos_encoding_mode="NONE",
                        q_data_type=torch.float16,
                        kv_data_type=torch.float16
                    )

        def run(self, Q, kv_tuple, output):
            with torch.cuda.stream(self.stream):
                if Q.shape[0] == 0:
                    return
                Q = Q.view(-1, self.num_qo_heads, self.head_dim)
                output = output.view(-1, self.num_qo_heads, self.head_dim)
                with prof_marker("DecAttnBatchedCudaImpl.run"):
                    self.wrapper.run(Q, kv_tuple, out=output)
                output = output.view(-1, self.num_qo_heads * self.head_dim)


class DecAttnFlashinfer(Operations):
    def __init__(self, name, device):
        super().__init__(name, device)
        self.inputs = {
            "Q": IOWrapper(self, 'Q', device).is_input(),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', device).is_output()
        }
        self.externals: dict[str, BatchedDistKVCache | KVCacheNone]
        self.impl_map = {}
        self.init_impl_map()
        self.batched_decode_wrapper = None
        self.op_layer = DecAttnFlashinfer_Layer

    def init_impl_map(self):
        if platform_config.PLATFORM_CUDA:
            self.add_impl(DecAttnBatchedCudaImpl)

    def setShape(self, num_kv_heads, num_qo_heads, head_dim, tp_size=1):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.tp_size = tp_size
        q_dim = num_qo_heads * head_dim // tp_size
        self.inputs["Q"].init_shape((0, q_dim))
        self.outputs["output"].init_shape((0, q_dim))

        return self

    def update(self, qo_indicies):
        self.qo_indicies = qo_indicies
        io = self.inputs["Q"]
        start_req_idx = tensor_offset_to_req_idx(qo_indicies, io.tensor_offset)
        end_req_idx = tensor_offset_to_req_idx(
            qo_indicies, io.tensor_offset + io.batch_size)

        self.kv_indptr = self.externals["KVCache"].kv_indptr[start_req_idx: end_req_idx + 1]
        self.kv_indices = self.externals["KVCache"].kv_indices
        self.kv_last_page_len = self.externals["KVCache"].kv_last_page_len[start_req_idx: end_req_idx]

        self.page_size = self.externals["KVCache"].page_size
        if start_req_idx != end_req_idx:
            self.impl.plan(self.kv_indptr, self.kv_indices,
                           self.kv_last_page_len, self.page_size)

    def profile_update(self):
        self.impl.plan(self.kv_indptr, self.kv_indices,
                       self.kv_last_page_len, self.page_size)

    def setup_profile_custom(self):
        super().setup_profile_custom()
        self.k_data_ptr, self.v_data_ptr = self.externals["KVCache"].get_whole_kv_data(
            self.layer_list[0])
        self.kv_tuple = tuple([self.k_data_ptr, self.v_data_ptr])

    def init_profile_db(self):
        for _, impl in self.impl_map.items():
            self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS "{impl.category_tag}" (
                id           INTEGER PRIMARY KEY AUTOINCREMENT, 
                batch_size   INTEGER,
                seq_len INTEGER,
                sm_count INTEGER,
                head_dim INTEGER,
                num_qo_heads INTEGER,
                num_kv_heads INTEGER,
                average_time_ms REAL,
                UNIQUE(batch_size, seq_len, sm_count)
            );
            ''')

    def is_profiled_in_db(self, category_tag):
        self.cursor.execute(f'''
            SELECT * FROM {category_tag}
            WHERE batch_size = ? AND seq_len = ? AND sm_count = ?
        ''', (self.batch_size, self.externals["KVCache"].get_seqlen(0) - 1, self.sm_count))
        row = self.cursor.fetchone()
        if row is not None:
            print(
                f"Name: {self.name}, Category: {category_tag}, Batch Size: {self.batch_size}, Seq Len: {self.externals['KVCache'].get_seqlen(0)} already profiled.")
            return True
        return False

    def store_profile_db(self, category_tag, impl_tag, average_elapsed_ms):
        print(
            f"Name: {self.name}, Category: {category_tag}, Batch Size: {self.batch_size}, Average Time: {average_elapsed_ms} ms")
        seq_len = self.externals["KVCache"].get_seqlen(0) - 1
        print(f"seq_len: {seq_len}")
        self.cursor.execute(f'''
            INSERT OR IGNORE INTO {category_tag} (batch_size, seq_len, sm_count, head_dim, num_qo_heads, num_kv_heads, average_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (self.batch_size, seq_len, self.sm_count, self.head_dim, self.num_qo_heads, self.num_kv_heads, average_elapsed_ms))

    def run(self, kv_tuple):
        self.impl.run(self.inputs["Q"].tensor,
                      kv_tuple, self.outputs["output"].tensor)

    def profile_run(self):
        self.run(self.kv_tuple)


class DecAttnFlashinfer_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)
        self.k_data_ptr, self.v_data_ptr = base_op.externals["KVCache"].get_whole_kv_data(
            self.layer)
        self.kv_tuple = tuple([self.k_data_ptr, self.v_data_ptr])

    def run(self):
        self.parent.run(self.kv_tuple)


if platform_config.PLATFORM_CUDA:
    import flashinfer.prefill

    class PFAttnBatchedCudaImpl(OperationImpl):
        category_tag = "batched_cuda"

        def __init__(self, op_base, stream, device):
            super().__init__(op_base, stream, device)
            self.workspace_buffer = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8).to(self.device)
            self.wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, "HND"
            )
            self.num_qo_heads = op_base.num_qo_heads // op_base.tp_size
            self.num_kv_heads = op_base.num_kv_heads // op_base.tp_size
            self.head_dim = op_base.head_dim

        def plan(self, qo_indicies, kv_indptr, kv_indices, kv_last_page_len, page_size,
                 causal=True, logits_soft_cap=0.0, pos_encoding_mode="NONE"):
            # print("PFAttnBatchedCudaImpl.plan")
            # print("qo_indicies: ", qo_indicies)
            # print("kv_indptr: ", kv_indptr)
            # print("kv_indices: ", kv_indices)
            # print("kv_last_page_len: ", kv_last_page_len)
            # print("page_size: ", page_size)
            # print("causal: ", causal)
            # print("logits_soft_cap: ", logits_soft_cap)
            # print("pos_encoding_mode: ", pos_encoding_mode)
            # print("num_qo_heads: ", self.num_qo_heads)
            # print("num_kv_heads: ", self.num_kv_heads)
            # print("head_dim: ", self.head_dim)
            with torch.cuda.stream(self.stream):
                self.wrapper.plan(
                    qo_indicies,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    self.num_qo_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    page_size,
                    causal=causal,
                    logits_soft_cap=logits_soft_cap,
                    pos_encoding_mode=pos_encoding_mode
                )

        def run(self, Q, kv_tuple, output):
            with torch.cuda.stream(self.stream):
                if Q.shape[0] == 0:
                    return
                Q = Q.view(-1, self.num_qo_heads, self.head_dim)
                output = output.view(-1, self.num_qo_heads, self.head_dim)
                # print("PFAttnBatchedCudaImpl")
                # print("Q shape: ", Q.shape)
                # print("Q: ", Q)
                # print("kv_tuple shape: ", kv_tuple[0].shape)
                # print("kv_tuple: ", kv_tuple[0])

                self.wrapper.run(Q, kv_tuple, out=output)

                output = output.view(-1, self.num_qo_heads * self.head_dim)


class PFAttnFlashinfer(Operations):
    def __init__(self, name, device, nano_idx=None):
        super().__init__(name, device, nano_idx=nano_idx)
        self.inputs = {
            "Q": IOWrapper(self, 'Q', device).is_input(),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', device).is_output()
        }
        # Note: for consistency with other operators (like RopeAppend), we expect the external KV cache to be
        # available as "KVCache". If needed, you can change the key name.
        self.externals: dict[str, BatchedDistKVCache | KVCacheNone]
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = PFAttnFlashinfer_Layer

    def init_impl_map(self):
        if platform_config.PLATFORM_CUDA:
            self.add_impl(PFAttnBatchedCudaImpl)

    def setShape(self, num_kv_heads, num_qo_heads, head_dim, tp_size=1):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.tp_size = tp_size
        q_dim = num_qo_heads * head_dim // tp_size
        self.inputs["Q"].init_shape((0, q_dim))
        self.outputs["output"].init_shape((0, q_dim))

        return self

    def update(self, qo_indicies,
               causal=True, logits_soft_cap=0.0, pos_encoding_mode="NONE"):
        """Stores the query offset indices for each batch element.  
        qo_indicies should be a list (or tensor) of length (batch_size + 1) such that for each batch index i,  
        the query slice is Q[qo_indicies[i]:qo_indicies[i+1], :].
        """
        if self.isNanoSplit:
            for nano_op in self.nano_ops:
                nano_op.update(qo_indicies, causal=causal, logits_soft_cap=logits_soft_cap, pos_encoding_mode=pos_encoding_mode)
        else:
            io = self.inputs["Q"]
            start_req_idx = tensor_offset_to_req_idx(qo_indicies, io.tensor_offset)
            end_req_idx = tensor_offset_to_req_idx(
                qo_indicies, io.tensor_offset + io.batch_size)

            self.qo_indicies = torch.tensor(
                qo_indicies[start_req_idx: end_req_idx + 1], dtype=torch.int32, device=self.device) - io.tensor_offset
            self.kv_indptr = self.externals["KVCache"].kv_indptr[start_req_idx: end_req_idx + 1]
            self.kv_indices = self.externals["KVCache"].kv_indices
            self.kv_last_page_len = self.externals["KVCache"].kv_last_page_len[start_req_idx: end_req_idx]

            self.page_size = self.externals["KVCache"].page_size
            self.causal = causal
            self.logits_soft_cap = logits_soft_cap
            self.pos_encoding_mode = pos_encoding_mode
            # print("qo_indicies: ", self.qo_indicies)
            # print("qo_indicies dtype: ", self.qo_indicies.dtype)
            # print("kv_indptr: ", self.kv_indptr)
            # print("kv_indptr dtype: ", self.kv_indptr.dtype)
            # print("kv_indices: ", self.kv_indices)
            # print("kv_indices dtype: ", self.kv_indices.dtype)
            # print("kv_last_page_len: ", self.kv_last_page_len)
            # print("kv_last_page_len dtype: ", self.kv_last_page_len.dtype)
            if start_req_idx != end_req_idx:
                self.impl.plan(self.qo_indicies, self.kv_indptr, self.kv_indices, self.kv_last_page_len, self.page_size,
                            causal=self.causal, logits_soft_cap=self.logits_soft_cap, pos_encoding_mode=self.pos_encoding_mode)
            # print("qo_indicies: ", qo_indicies)
            # print("qo_indicies dtype: ", qo_indicies.dtype)

    def copy_nano(self, index):
        new_op = PFAttnFlashinfer(self.name, self.device, nano_idx=index)
        new_op.set_category(self.category)
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.num_kv_heads, self.num_qo_heads,
                        self.head_dim, self.tp_size)
        self.nano_ops.append(new_op)
        return new_op

    def profile_update(self):
        self.impl.plan(self.qo_indicies, self.kv_indptr, self.kv_indices, self.kv_last_page_len, self.page_size,
                       causal=self.causal, logits_soft_cap=self.logits_soft_cap, pos_encoding_mode=self.pos_encoding_mode)

    def setup_profile_custom(self):
        super().setup_profile_custom()
        self.k_data_ptr, self.v_data_ptr = self.externals["KVCache"].get_whole_kv_data(
            self.layer_list[0])
        self.kv_tuple = tuple([self.k_data_ptr, self.v_data_ptr])

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
        print(
            f"Name: {self.name}, Category: {category_tag}, Batch Size: {self.batch_size}, Average Time: {average_elapsed_ms} ms")
        self.cursor.execute(f'''
            INSERT OR IGNORE INTO {category_tag} (batch_size, sm_count, head_dim, num_qo_heads, num_kv_heads, average_time_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (self.batch_size, self.sm_count, self.head_dim, self.num_qo_heads, self.num_kv_heads, average_elapsed_ms))

    def run(self, kv_tuple):
        self.impl.run(self.inputs["Q"].tensor,
                      kv_tuple, self.outputs["output"].tensor)

    def profile_run(self):
        self.run(self.kv_tuple)


class PFAttnFlashinfer_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)
        self.k_data_ptr, self.v_data_ptr = base_op.externals["KVCache"].get_whole_kv_data(
            self.layer)
        self.kv_tuple = tuple([self.k_data_ptr, self.v_data_ptr])

    def run(self):
        self.parent.run(self.kv_tuple)
