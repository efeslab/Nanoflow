import torch
import numpy as np
import time

from operations.operation_base import Operations, Operation_Device, Operation_Layer
from utils.prof_marker import prof_marker
import platform_config
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
from operations.impl_base import OperationImpl
from kvcache.kv import KVCacheNone, KVCacheTorch, DistKVPool, BatchedDistKVCache
from utils.help_functions import tensor_offset_to_req_idx

if platform_config.PLATFORM_CUDA:
    import flashinfer
    class DecAttnCudaImpl(OperationImpl):
        category_tag = "cuda"
        def __init__(self, op_base, stream, device_id):
            super().__init__(op_base, stream, device_id)
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
        def __init__(self, op_base, stream, device_id):
            super().__init__(op_base, stream, device_id)
            self.workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(self.device_id)
            self.wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
                    self.workspace_buffer, "HND", False, True
                )
            self.num_qo_heads = op_base.num_qo_heads
            self.num_kv_heads = op_base.num_kv_heads
            self.head_dim = op_base.head_dim
        
        def plan(self, kv_indptr, kv_indices, kv_last_page_len, page_size):
            with prof_marker("DecAttnBatchedCudaImpl.plan"):
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
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "Q": IOWrapper(self, 'Q')
        }
        self.outputs = {
            "output": IOWrapper(self, 'output')
        }
        self.externals = {
            "KVCache": None
        }
        self.impl_map = {}
        self.init_impl_map()
        self.batched_decode_wrapper = None
        self.op_device = DecAttnFlashinfer_Device

    def init_impl_map(self):
        if platform_config.PLATFORM_CUDA:
            self.add_impl(DecAttnCudaImpl)
            self.add_impl(DecAttnBatchedCudaImpl)
    
    def setShape(self, num_kv_heads, num_qo_heads, head_dim, tp_size=1):
        self.num_kv_heads = num_kv_heads // tp_size
        self.num_qo_heads = num_qo_heads // tp_size
        self.head_dim = head_dim
        self.q_dim = num_qo_heads * head_dim
        self.updateChildrenIOShape()
    
    def update(self, qo_indicies, device_id):
        self.qo_indicies = qo_indicies
        io_device = self.children[device_id].inputs["Q"]
        start_req_idx = tensor_offset_to_req_idx(qo_indicies, io_device.tensor_offset)
        end_req_idx = tensor_offset_to_req_idx(qo_indicies, io_device.tensor_offset + io_device.batch_size)

        self.kv_indptr =  self.externals["KVCache"].kv_indptr_devices[device_id][start_req_idx: end_req_idx + 1]
        self.kv_indices = self.externals["KVCache"].kv_indices_devices[device_id]
        self.kv_last_page_len = self.externals["KVCache"].kv_last_page_len_devices[device_id][start_req_idx: end_req_idx]

        self.page_size = self.externals["KVCache"].page_size
        if self.impl.category_tag == "batched_cuda":
            self.impl.plan(self.kv_indptr, self.kv_indices, self.kv_last_page_len, self.page_size)
    
    def profile(self):
        pass
    
class DecAttnFlashinfer_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.op_layer = DecAttnFlashinfer_Layer 

    def setShapeForIOWrappers(self):
        self.inputs["Q"].init_shape((0, self.parent.num_qo_heads* self.parent.head_dim))
        self.outputs["output"].init_shape((0, self.parent.num_qo_heads * self.parent.head_dim))

class DecAttnFlashinfer_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer, op_device=op_device)
        self.k_data_ptr, self.v_data_ptr = op_device.externals["KVCache"].get_whole_kv_data(self.device_id, self.layer)
        self.kv_tuple = tuple([self.k_data_ptr, self.v_data_ptr])

    def run(self):
        Q = self.inputs["Q"].tensor
        # self.operator_device.parent.impl.run(Q, self.kv_tuple, self.outputs["output"].tensor)
        self.impl.run(self.layer, self.parent.parent.qo_indicies,  Q, self.kv_tuple, self.parent.externals["KVCache"], self.outputs["output"].tensor)
    


if platform_config.PLATFORM_CUDA:
    class PFAttnCudaImpl(OperationImpl):
        category_tag = "cuda"
        def __init__(self, op_base, stream, device_id):
            super().__init__(op_base, stream, device_id)
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
        def __init__(self, op_base, stream, device_id):
            super().__init__(op_base, stream, device_id)
            self.workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(self.device_id)
            self.wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, "HND"
            )
            self.num_qo_heads = op_base.num_qo_heads
            self.num_kv_heads = op_base.num_kv_heads
            self.head_dim = op_base.head_dim
        
        def plan(self, qo_indicies, kv_indptr, kv_indices, kv_last_page_len, page_size,
                causal=True, logits_soft_cap=0.0, pos_encoding_mode="NONE"):
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
                self.wrapper.run(Q, kv_tuple, out=output)

                output = output.view(-1, self.num_qo_heads * self.head_dim)


class PFAttnFlashinfer(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "Q": IOWrapper(self, 'Q'),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output')
        }
        # Note: for consistency with other operators (like RopeAppend), we expect the external KV cache to be
        # available as "KVCache". If needed, you can change the key name.
        self.externals = {
            "KVCache": None
        }
        self.impl_map = {}
        self.init_impl_map()
        self.op_device = PFAttnFlashinfer_Device

    def init_impl_map(self):
        if platform_config.PLATFORM_CUDA:
            self.add_impl(PFAttnCudaImpl)
            self.add_impl(PFAttnBatchedCudaImpl)
    
    def setShape(self, num_kv_heads, num_qo_heads, head_dim, tp_size=1):
        self.num_kv_heads = num_kv_heads // tp_size
        self.num_qo_heads = num_qo_heads // tp_size
        self.head_dim = head_dim
        self.q_dim = num_qo_heads * head_dim
        self.updateChildrenIOShape()
    
    def update(self, qo_indicies, device_id,
             causal=True, logits_soft_cap=0.0, pos_encoding_mode="NONE"):
        """Stores the query offset indices for each batch element.  
        qo_indicies should be a list (or tensor) of length (batch_size + 1) such that for each batch index i,  
        the query slice is Q[qo_indicies[i]:qo_indicies[i+1], :].
        """
        io_device = self.children[device_id].inputs["Q"]
        start_req_idx = tensor_offset_to_req_idx(qo_indicies, io_device.tensor_offset)
        end_req_idx = tensor_offset_to_req_idx(qo_indicies, io_device.tensor_offset + io_device.batch_size)

        self.qo_indicies = torch.tensor(qo_indicies[start_req_idx: end_req_idx + 1], dtype=torch.int32, device=f"cuda:{device_id}") - io_device.tensor_offset
        self.kv_indptr =  self.externals["KVCache"].kv_indptr_devices[device_id][start_req_idx: end_req_idx + 1]
        self.kv_indices = self.externals["KVCache"].kv_indices_devices[device_id]
        self.kv_last_page_len = self.externals["KVCache"].kv_last_page_len_devices[device_id][start_req_idx: end_req_idx]

        self.page_size = self.externals["KVCache"].page_size
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

    
class PFAttnFlashinfer_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.op_layer = PFAttnFlashinfer_Layer 

    def setShapeForIOWrappers(self):
        self.inputs["Q"].init_shape((0, self.parent.num_qo_heads * self.parent.head_dim))
        self.outputs["output"].init_shape((0, self.parent.num_qo_heads * self.parent.head_dim))


class PFAttnFlashinfer_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer=layer, op_device=op_device)
        self.k_data_ptr, self.v_data_ptr = op_device.externals["KVCache"].get_whole_kv_data(self.device_id, self.layer)
        self.kv_tuple = tuple([self.k_data_ptr, self.v_data_ptr])

    def run(self):
        Q = self.inputs["Q"].tensor
        # self.operator_device.parent.impl.run(Q, self.kv_tuple, self.outputs["output"].tensor)
        self.impl.run(self.layer, self.parent.parent.qo_indicies,  Q, self.kv_tuple, self.parent.externals["KVCache"], self.outputs["output"].tensor)
        