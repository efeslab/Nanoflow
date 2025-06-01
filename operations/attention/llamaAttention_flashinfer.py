import torch
import numpy as np
import time

from operations.operation_base import Operations, Operation_Layer
from utils.prof_marker import prof_marker
import platform_config
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
from operations.impl_base import OperationImpl
from kvcache.kv import KVCacheNone, KVCacheTorch, DistKVPool, BatchedDistKVCache
from utils.util_functions import tensor_offset_to_req_idx

if platform_config.PLATFORM_CUDA:
    import flashinfer
    class DecAttnCudaImpl(OperationImpl):
        category_tag = "cuda"
        def __init__(self, op_base, stream, device):
            super().__init__(op_base, stream, device)
            self.num_qo_heads = op_base.num_qo_heads
            self.num_kv_heads = op_base.num_kv_heads
            self.head_dim = op_base.head_dim

        def run(self, layer, qo_indicies,  Q, KVCache, output
        ):
            if Q.shape[0] == 0:
                return
            scale = 1.0 / (self.head_dim ** 0.5)
            # Compute group size: how many query heads correspond to one key/value head.
            group_size = self.num_qo_heads // self.num_kv_heads
            for i in range(len(qo_indicies) - 1):
                # Retrieve the query slice for this batch element.
                start = qo_indicies[i]
                end = qo_indicies[i + 1]

                sub_q = Q[start:end, :]  # shape: [n_q, num_qo_heads * head_dim]
                sub_q = sub_q.view(-1, self.num_qo_heads, self.head_dim)
                
                sub_k, sub_v = KVCache[layer].get(i) # [n_k, num_kv_heads * head_dim]

                n_k = sub_k.shape[0]

                sub_k = sub_k.view(n_k, self.num_kv_heads, self.head_dim)
                sub_v = sub_v.view(n_k, self.num_kv_heads, self.head_dim)
                sub_q = sub_q.squeeze(0)
                out = flashinfer.single_decode_with_kv_cache(sub_q, sub_k, sub_v, use_tensor_cores=True)

                out = out.reshape(-1, self.num_qo_heads * self.head_dim)
                output[start:end, :].copy_(out)

    class DecAttnBatchedCudaImpl(OperationImpl):
        category_tag = "batched_cuda"
        def __init__(self, op_base, stream, device):
            super().__init__(op_base, stream, device)
            self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8).to(self.device)
            self.wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer, "HND", False, True
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
                            data_type=torch.float16,
                            q_data_type=torch.float16
                        )

        def run(self, layer, qo_indicies,  Q, kv_tuple, KVCache, output):
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
        self.externals = {
            "KVCache": None
        }
        self.impl_map = {}
        self.init_impl_map()
        self.batched_decode_wrapper = None
        self.op_layer = DecAttnFlashinfer_Layer

    def init_impl_map(self):
        if platform_config.PLATFORM_CUDA:
            self.add_impl(DecAttnCudaImpl)
            self.add_impl(DecAttnBatchedCudaImpl)
    
    def setShape(self, num_kv_heads, num_qo_heads, head_dim, tp_size=1):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.tp_size = tp_size
        q_dim = num_qo_heads * head_dim // tp_size
        self.inputs["Q"].init_shape((0, q_dim))
        self.outputs["output"].init_shape((0, q_dim))
    
    def update(self, qo_indicies):
        self.qo_indicies = qo_indicies
        io = self.inputs["Q"]
        start_req_idx = tensor_offset_to_req_idx(qo_indicies, io.tensor_offset)
        end_req_idx = tensor_offset_to_req_idx(qo_indicies, io.tensor_offset + io.batch_size)

        self.kv_indptr =  self.externals["KVCache"].kv_indptr[start_req_idx: end_req_idx + 1]
        self.kv_indices = self.externals["KVCache"].kv_indices
        self.kv_last_page_len = self.externals["KVCache"].kv_last_page_len[start_req_idx: end_req_idx]

        self.page_size = self.externals["KVCache"].page_size
        if self.impl.category_tag == "batched_cuda":
            self.impl.plan(self.kv_indptr, self.kv_indices, self.kv_last_page_len, self.page_size)
    
    def profile(self):
        pass

class DecAttnFlashinfer_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)
        self.k_data_ptr, self.v_data_ptr = base_op.externals["KVCache"].get_whole_kv_data(self.layer)
        self.kv_tuple = tuple([self.k_data_ptr, self.v_data_ptr])

    def run(self):
        Q = self.inputs["Q"].tensor
        # self.operator_device.parent.impl.run(Q, self.kv_tuple, self.outputs["output"].tensor)
        self.impl.run(self.layer, self.parent.qo_indicies, Q, self.kv_tuple, self.parent.externals["KVCache"], self.outputs["output"].tensor)
    


if platform_config.PLATFORM_CUDA:
    class PFAttnCudaImpl(OperationImpl):
        category_tag = "cuda"
        def __init__(self, op_base, stream, device):
            super().__init__(op_base, stream, device)
            self.num_qo_heads = op_base.num_qo_heads
            self.num_kv_heads = op_base.num_kv_heads
            self.head_dim = op_base.head_dim
        def run(self, layer, qo_indicies, Q, KVCache, output
        ):
            if Q.shape[0] == 0:
                return
            # print("PFAttnCudaImpl")
            # print("Q shape: ", Q.shape)
            # print("Q: ", Q)
            scale = 1.0 / (self.head_dim ** 0.5)
            # Compute group size: how many query heads correspond to one key/value head.
            group_size = self.num_qo_heads // self.num_kv_heads

            for i in range(len(qo_indicies) - 1):
                # Retrieve the query slice for this batch element.
                start = qo_indicies[i]
                end = qo_indicies[i + 1]
                # Q is expected to be flattened as [n_total, num_qo_heads * head_dim];
                # extract the sub-tensor corresponding to this batch element.
                sub_q = Q[start:end, :]  # shape: [n_q, num_qo_heads * head_dim]
                sub_q = sub_q.view(-1, self.num_qo_heads, self.head_dim)

                sub_k, sub_v = KVCache[layer].get(i)
                n_k = sub_k.shape[0]

                # Reshape keys and values so that the head dimension is explicit.
                # New shapes: [n_k, num_kv_heads, head_dim]
                sub_k = sub_k.view(n_k, self.num_kv_heads, self.head_dim)
                sub_v = sub_v.view(n_k, self.num_kv_heads, self.head_dim)
                sub_q = sub_q.contiguous()
                sub_k = sub_k.contiguous()
                sub_v = sub_v.contiguous()
                
                out = flashinfer.single_prefill_with_kv_cache(sub_q, sub_k, sub_v, causal=True)
                out = out.reshape(-1, self.num_qo_heads * self.head_dim)

                output[start:end, :].copy_(out)

    class PFAttnBatchedCudaImpl(OperationImpl):
        category_tag = "batched_cuda"
        def __init__(self, op_base, stream, device):
            super().__init__(op_base, stream, device)
            self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8).to(self.device)
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

        def run(self, layer, qo_indicies, Q, kv_tuple, KVCache, output):
            with torch.cuda.stream(self.stream):
                if Q.shape[0] == 0:
                    return
                Q = Q.view(-1, self.num_qo_heads, self.head_dim)
                output = output.view(-1, self.num_qo_heads, self.head_dim)
                # print("PFAttnBatchedCudaImpl")
                # print("qo_indicies: ", qo_indicies)
                # print("Q shape: ", Q.shape)
                # print("Q: ", Q)
                # print("kv_tuple shape: ", kv_tuple[0].shape)
                # print("kv_tuple: ", kv_tuple[0])


                self.wrapper.run(Q, kv_tuple, out=output)

                output = output.view(-1, self.num_qo_heads * self.head_dim)


class PFAttnFlashinfer(Operations):
    def __init__(self, name, device):
        super().__init__(name, device)
        self.inputs = {
            "Q": IOWrapper(self, 'Q', device).is_input(),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', device).is_output()
        }
        # Note: for consistency with other operators (like RopeAppend), we expect the external KV cache to be
        # available as "KVCache". If needed, you can change the key name.
        self.externals = {
            "KVCache": None
        }
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = PFAttnFlashinfer_Layer

    def init_impl_map(self):
        if platform_config.PLATFORM_CUDA:
            self.add_impl(PFAttnCudaImpl)
            self.add_impl(PFAttnBatchedCudaImpl)
    
    def setShape(self, num_kv_heads, num_qo_heads, head_dim, tp_size=1):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.tp_size = tp_size
        q_dim = num_qo_heads * head_dim // tp_size
        self.inputs["Q"].init_shape((0, q_dim))
        self.outputs["output"].init_shape((0, q_dim))
    
    def update(self, qo_indicies,
             causal=True, logits_soft_cap=0.0, pos_encoding_mode="NONE"):
        """Stores the query offset indices for each batch element.  
        qo_indicies should be a list (or tensor) of length (batch_size + 1) such that for each batch index i,  
        the query slice is Q[qo_indicies[i]:qo_indicies[i+1], :].
        """
        io = self.inputs["Q"]
        start_req_idx = tensor_offset_to_req_idx(qo_indicies, io.tensor_offset)
        end_req_idx = tensor_offset_to_req_idx(qo_indicies, io.tensor_offset + io.batch_size)

        self.qo_indicies = torch.tensor(qo_indicies[start_req_idx: end_req_idx + 1], dtype=torch.int32, device=self.device) - io.tensor_offset
        self.kv_indptr =  self.externals["KVCache"].kv_indptr[start_req_idx: end_req_idx + 1]
        self.kv_indices = self.externals["KVCache"].kv_indices
        self.kv_last_page_len = self.externals["KVCache"].kv_last_page_len[start_req_idx: end_req_idx]

        self.page_size = self.externals["KVCache"].page_size
        # print("qo_indicies: ", self.qo_indicies)
        # print("qo_indicies dtype: ", self.qo_indicies.dtype)
        # print("kv_indptr: ", self.kv_indptr)
        # print("kv_indptr dtype: ", self.kv_indptr.dtype)
        # print("kv_indices: ", self.kv_indices)
        # print("kv_indices dtype: ", self.kv_indices.dtype)
        # print("kv_last_page_len: ", self.kv_last_page_len)
        # print("kv_last_page_len dtype: ", self.kv_last_page_len.dtype)
        if self.impl.category_tag == "batched_cuda":
            # Only plan for the batched CUDA implementation. 
            self.impl.plan(self.qo_indicies, self.kv_indptr, self.kv_indices, self.kv_last_page_len, self.page_size,
                causal=causal, logits_soft_cap=logits_soft_cap, pos_encoding_mode=pos_encoding_mode)
        # print("qo_indicies: ", qo_indicies)
        # print("qo_indicies dtype: ", qo_indicies.dtype) 

    def profile(self):
        input_q = torch.randn(2, self.q_dim, dtype=torch.float16, device='cuda')
        k_data = torch.randn(2, self.num_kv_heads* self.head_dim, dtype=torch.float16, device='cuda')
        v_data = torch.randn(2, self.num_kv_heads* self.head_dim, dtype=torch.float16, device='cuda')
        output_list = []
        for category_tag, impl in self.impl_map.items():
            out = torch.zeros((2, self.q_dim), dtype=torch.float16, device='cuda')
            print("name: ", self.name + f"_{category_tag}")
            if category_tag == "cuda":
                torch_kv_cache = KVCacheTorch()
                torch_kv_cache.put(0, k_data, v_data)

                impl().run(0, self.head_dim, self.num_qo_heads, self.num_kv_heads, torch.tensor([0, 2], dtype=torch.int32), input_q, [torch_kv_cache], out)
                print("output: ", out)
                output_list.append(out)
            elif category_tag == "batched_cuda":
                kv_pool = DistKVPool(1, self.num_kv_heads, self.head_dim, 2048, 7, 1)
                batched_kv_cache = BatchedDistKVCache(kv_pool, 0)

                k_data = k_data.view(-1, self.num_kv_heads, self.head_dim)
                v_data = v_data.view(-1, self.num_kv_heads, self.head_dim)
                batched_kv_cache.pre_allocate(0, 2)
                batched_kv_cache._pool.put_for_profile(0, 2, k_data, v_data)
                impl().run(0, self.head_dim, self.num_qo_heads, self.num_kv_heads, torch.tensor([0, 2], dtype=torch.int32), input_q, [batched_kv_cache], out)
                print("output: ", out)
                output_list.append(out)

        self.checkConsistencyBetweenImpl(output_list)

class PFAttnFlashinfer_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)
        self.k_data_ptr, self.v_data_ptr = base_op.externals["KVCache"].get_whole_kv_data(self.layer)
        self.kv_tuple = tuple([self.k_data_ptr, self.v_data_ptr])

    def run(self):
        Q = self.inputs["Q"].tensor
        self.impl.run(self.layer, self.parent.qo_indicies, Q, self.kv_tuple, self.parent.externals["KVCache"], self.outputs["output"].tensor)
        