import platform
import torch
import math
import time

import platform_config
from operations.rope.help_functions import apply_rope
from operations.operation_base import Operations, Operation_Device, Operation_Layer
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
from operations.impl_base import OperationImpl
from kvcache.kv import KVCacheNone, KVCacheTorch, DistKVPool, BatchedDistKVCache
from utils.prof_marker import prof_marker 



class RopeAppendTorchImpl(OperationImpl):
    category_tag = "torch"
    def __init__(self, op_base, device_id):
        super().__init__(op_base, device_id)
        self.rope_type = op_base.rope_type
        self.theta = op_base.theta
        self.original_max_position_embeddings = op_base.original_max_position_embeddings
        self.low_freq_factor = op_base.low_freq_factor
        self.high_freq_factor = op_base.high_freq_factor
        self.factor = op_base.factor
        self.num_kv_heads = op_base.num_kv_heads
        self.num_qo_heads = op_base.num_qo_heads
        self.head_dim = op_base.head_dim
        
    def run(self, layer, kqv, KVCache, k_data, v_data, output, decode_flag, offset=0):
        # Determine the number of elements for each slice.
        layout_strides = [
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_qo_heads * self.head_dim,
        ]
        # Split kqv into key, query, and value (here assumed to be in the order: k, q, v).
        # print("kqv shape:", kqv.shape)
        k, v, q = torch.split(kqv, layout_strides, dim=1)
        k = k.contiguous()
        v = v.contiguous()
        q = q.contiguous()

        # Process each batch element.
        # print(self.op_base.qo_indicies)
        for i in range(len(self.op_base.qo_indicies) - 1):
            start = self.op_base.qo_indicies[i]
            end = self.op_base.qo_indicies[i + 1]
            sub_q = q[start:end, :]
            sub_k = k[start:end, :]
            if not decode_flag or KVCache.get_indices(layer, i) is None:
                last_offest = 0
            else:
                last_offest = KVCache.get_indices(layer, i)[0]
            apply_rope(
                self.rope_type,
                self.theta,
                self.original_max_position_embeddings,
                self.low_freq_factor,
                self.high_freq_factor,
                self.factor,
                sub_q,
                output=sub_q,
                offset=last_offest
            )
            apply_rope(
                self.rope_type,
                self.theta,
                self.original_max_position_embeddings,
                self.low_freq_factor,
                self.high_freq_factor,
                self.factor,
                sub_k,
                output=sub_k,
                offset=last_offest
            )

            # Write the updated values back.
            q[start:end, :] = sub_q
            k[start:end, :] = sub_k

            # Update the external KVCache with the new key and value.
            KVCache.put(layer, i, sub_k, v[start:end, :])
        output.copy_(q)
        
if platform_config.PLATFORM_CUDA:
    import bind_ropeappend
    class RopeAppendCudaImpl(OperationImpl):
        category_tag = "cuda"
        def __init__(self, op_base, device_id):
            super().__init__(op_base, device_id)
            # self.page_size = op_base.page_size
            self.num_kv_heads = op_base.num_kv_heads
            self.num_qo_heads = op_base.num_qo_heads
            self.head_dim = op_base.head_dim
            
        def run(self, layer,  kqv, KVCache, k_data, v_data, output, decode_flag, offset=0):
            with prof_marker("RopeAppendCuda: SplitRopeAppend"):
                bind_ropeappend.splitRopeAppend(
                    k_data,
                    v_data,
                    kqv,
                    output,
                    self.op_base.rev_input_indptr,
                    self.op_base.per_token_offset,
                    len(self.op_base.qo_indicies) - 1,
                    self.op_base.page_size,
                    self.num_kv_heads,
                    self.num_qo_heads,
                    self.head_dim,
                    1.0,
                    500000.0,
                    0.0,
                    0.0
                )

class RopeAppend(Operations):
    def __init__(
        self,
        name,
        rope_type="llama3",
        theta=10000.0,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ):
        """
        Args:
            name (str): The name of this operator.
            rope_type (str): The type of RoPE implementation to use. For llama3, pass "llama3".
            theta (float): The base used to compute the inverse frequency (typically set from config.rope_theta).
            factor (float): Scaling factor used in llama3.
            low_freq_factor (float): Lower bound frequency factor (llama3).
            high_freq_factor (float): Upper bound frequency factor (llama3).
            original_max_position_embeddings (int): The original maximum context length used in pretraining.
        """
        super().__init__(name)
        self.inputs = {"kqv": IOWrapper(self, "kqv")}
        self.outputs = {"q": IOWrapper(self, "q")}
        self.externals = {"KVCache": None, "k_data": None, "v_data": None}
        
        # Save RoPE configuration.
        self.rope_type = rope_type
        self.theta = theta  # typically config.rope_theta
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position_embeddings = original_max_position_embeddings

        self.impl_map = {}
        self.init_impl_map()
        self.op_device = RopeAppend_Device

    def init_impl_map(self):
        self.add_impl(RopeAppendTorchImpl)
        if platform_config.PLATFORM_CUDA:
            self.add_impl(RopeAppendCudaImpl)

    def setShape(self, num_kv_heads, num_qo_heads, head_dim):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        for op_device in self.children:
            op_device.setShapeForIOWrappers()

    def update(self, page_size, qo_indicies, kv_indptr, kv_indices, kv_last_page_len, rev_input_indptr, per_token_offset, decode_flag=False):
        """Stores the starting indices for the query/key segments."""
        self.page_size = page_size
        self.qo_indicies = qo_indicies
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.kv_last_page_len = kv_last_page_len
        self.rev_input_indptr = rev_input_indptr
        self.per_token_offset = per_token_offset
        self.decode_flag = decode_flag
        if self.impl.category_tag == "cuda":
            bind_ropeappend.updateKVCache(self.kv_indptr, self.kv_indices, self.kv_last_page_len, len(self.kv_last_page_len), self.page_size, self.num_kv_heads, self.num_qo_heads, self.head_dim)

    def profile(self):
        input_kqv = torch.randn(2, (self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim, dtype=torch.float16, device='cuda')
        output_list = []
        for category_tag, impl in self.impl_map.items():
            out = torch.zeros((2, self.num_qo_heads * self.head_dim), dtype=torch.float16, device='cuda')
            # print("name:", category_tag)
            if category_tag == "torch":
                impl().run(0, self.head_dim, self.num_qo_heads, self.num_kv_heads, torch.tensor([0, 2], dtype=torch.int32).cuda(), input_kqv, [KVCacheNone()], self.rope_type, self.theta, self.original_max_position_embeddings, self.low_freq_factor, self.high_freq_factor, self.factor, out, False)

                output_list.append(out)

                impl().run(0, self.head_dim, self.num_qo_heads, self.num_kv_heads, torch.tensor([0, 2], dtype=torch.int32).cuda(), input_kqv, [KVCacheTorch()], self.rope_type, self.theta, self.original_max_position_embeddings, self.low_freq_factor, self.high_freq_factor, self.factor, out, False)

            elif category_tag == "cuda":
                kv_pool = DistKVPool(1, self.num_kv_heads, self.head_dim, 2048, 7, 1)
                batchde_kv = BatchedDistKVCache(kv_pool, 0)
                impl().run(0, self.head_dim, self.num_qo_heads, self.num_kv_heads, torch.tensor([0, 2], dtype=torch.int32).cuda(), input_kqv, [batchde_kv], self.rope_type, self.theta, self.original_max_position_embeddings, self.low_freq_factor, self.high_freq_factor, self.factor, out, False)

            # print("out:", out)
            output_list.append(out)
        
        self.checkConsistencyBetweenImpl(output_list)
        # print("RopeAppend profile passed")
        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        for batch_size in batch_sizes:
            output = torch.zeros((batch_size, self.num_qo_heads * self.head_dim), dtype=torch.float16, device='cuda')
            for _, impl in self.impl_map.items():
                impl_instance = impl()
                category_tag = impl_instance.category_tag
                if category_tag == "torch":
                    kv_caches_choices = ['nokv', 'torch']
                elif category_tag == "cuda":
                    kv_caches_choices = ['flashinfer']

                total_latency = 0
                for kv_choice in kv_caches_choices:
                    for round in range(rounds):
                        input_kqv = torch.randn(batch_size, (self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim, dtype=torch.float16, device='cuda')
                        if kv_choice == 'nokv':
                            kv_caches = [KVCacheNone()]
                        elif kv_choice == 'torch':
                            kv_caches = [KVCacheTorch()]
                        elif kv_choice == 'flashinfer':
                            kv_pool = DistKVPool(1, self.num_kv_heads, self.head_dim, 2048, 7, 1)
                            batchde_kv = BatchedDistKVCache(kv_pool, 0)
                            kv_caches = [batchde_kv]

                        start_time = time.time()
                        impl_instance.run(0, self.head_dim, self.num_qo_heads, self.num_kv_heads, torch.tensor([0, batch_size], dtype=torch.int32).cuda(), input_kqv, kv_caches, self.rope_type, self.theta, self.original_max_position_embeddings, self.low_freq_factor, self.high_freq_factor, self.factor, output, False)
                        if round > 0:
                            total_latency += time.time() - start_time

                    average_time = total_latency / rounds
                    print("name: {}, batch_size: {}, average_time: {}".format(self.name + f"_{category_tag}" + f"with_{kv_caches[0].name}", batch_size, average_time))
                    self.cursor.execute('''
                    INSERT INTO performance (keyword, batch_size, average_time)
                    VALUES (?, ?, ?)
                    ''', (self.name + f"_{category_tag}" + f"with_{kv_caches[0].name}", batch_size, average_time))
        self.conn.commit()
    
class RopeAppend_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.op_layer = RopeAppend_Layer

    def setShapeForIOWrappers(self):
        # The input tensor "kqv" is assumed to have a flattened layout:
        # [batch_size, (num_qo_heads + 2 * num_kv_heads) * head_dim]
        self.inputs["kqv"].init_shape((
            0,
            (self.parent.num_qo_heads + 2 * self.parent.num_kv_heads) * self.parent.head_dim,
        ))
        # The output "q" has shape [batch_size, num_qo_heads * head_dim]
        self.outputs["q"].init_shape((0, self.parent.num_qo_heads * self.parent.head_dim))

class RopeAppend_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer, op_device)
        self.k_data_ptr, self.v_data_ptr = op_device.externals["KVCache"].get_whole_kv_data(self.device_id, self.layer)

    def run(self):
        self.impl.run(self.layer, self.inputs["kqv"].tensor, self.externals["KVCache"], self.k_data_ptr, self.v_data_ptr, self.outputs["q"].tensor, self.parent.parent.decode_flag, offset=0)
        