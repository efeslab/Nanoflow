import logging
import torch
import time

import platform_config
from operations.rope.help_functions import apply_rope # type: ignore[import]
from operations.operation_base import Operations, Operation_Device, Operation_Layer
from core.IOWrapper import IOWrapper
from operations.impl_base import OperationImpl
from kvcache.kv import KVCacheNone, KVCacheTorch, DistKVPool, BatchedDistKVCache
from utils.prof_marker import prof_marker
from utils.help_functions import tensor_offset_to_req_idx


def _apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)

class RopeAppendTorchImpl(OperationImpl):
    category_tag = "torch"
    def __init__(self, op_base, stream, device_id):
        super().__init__(op_base, stream, device_id)
        self.rope_type = op_base.rope_type
        if self.rope_type == "llama3":
            self.base = 500000.0
            self.rotary_dim = 128
        self.theta = op_base.theta
        self.original_max_position_embeddings = op_base.original_max_position_embeddings
        self.low_freq_factor = op_base.low_freq_factor
        self.high_freq_factor = op_base.high_freq_factor
        self.factor = op_base.factor
        self.num_kv_heads = op_base.num_kv_heads
        self.num_qo_heads = op_base.num_qo_heads
        self.head_dim = op_base.head_dim
    
    def config(self, impl_tag, parameter_map):
        if impl_tag == "withKVCache":
            self.use_kv_cache = True
        elif impl_tag == "withoutKVCache":
            self.use_kv_cache = False
        else:
            raise ValueError(f"Unknown impl_tag: {impl_tag}")
    
    def run(self, layer, kqv, KVCache, output, offset=0):
        with torch.cuda.stream(self.stream):
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

            qo_indicies = self.op_base.qo_indicies
            input_req_idx = self.op_base.input_req_idx
            
            # Process each batch element.
            for i, global_index in enumerate(input_req_idx):
                start = qo_indicies[i]
                end = qo_indicies[i + 1]
                sub_q = q[start:end, :]
                sub_k = k[start:end, :]
                
                last_offset = 0
                if self.use_kv_cache:
                    last_offset = KVCache.get_indices(layer, global_index)

                apply_rope(
                    self.rope_type,
                    self.theta,
                    self.original_max_position_embeddings,
                    self.low_freq_factor,
                    self.high_freq_factor,
                    self.factor,
                    sub_q,
                    output=sub_q,
                    offset=last_offset
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
                    offset=last_offset
                )

                # Write the updated values back.
                q[start:end, :] = sub_q
                k[start:end, :] = sub_k

                # Update the external KVCache with the new key and value.
                KVCache.put(layer, global_index, sub_k, v[start:end, :])
            output.copy_(q)
        
class RopeAppendTorch(Operations):
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
        self.externals = {"KVCache": None}
        
        # Save RoPE configuration.
        self.rope_type = rope_type
        self.theta = theta  # typically config.rope_theta
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.original_max_position_embeddings = original_max_position_embeddings

        self.impl_map = {}
        self.init_impl_map()
        self.op_device = RopeAppendTorch_Device

    def init_impl_map(self):
        self.add_impl(RopeAppendTorchImpl)

    def setShape(self, num_kv_heads, num_qo_heads, head_dim, tp_size=1):
        self.num_kv_heads = num_kv_heads // tp_size
        self.num_qo_heads = num_qo_heads // tp_size
        self.head_dim = head_dim
        self.updateChildrenIOShape()

    def update(self, qo_indicies, decode_batchsize, device_id):
        if self.isNanoSplit:
            for nano_op in self.nano_ops:
                nano_op.update(qo_indicies, decode_batchsize, device_id)
        else:
            """Stores the starting indices for the query/key segments."""
            io_device = self.children[device_id].inputs["kqv"]
            start_req_idx = tensor_offset_to_req_idx(qo_indicies, io_device.tensor_offset)
            end_req_idx = tensor_offset_to_req_idx(qo_indicies, io_device.tensor_offset + io_device.batch_size)

            self.qo_indicies = torch.tensor(qo_indicies[start_req_idx:end_req_idx + 1]) - io_device.tensor_offset
            self.input_req_idx = self.externals["KVCache"].input_req_idx[start_req_idx:end_req_idx]

    def copy_nano(self, index):
        new_op = RopeAppendTorch(f"{self.name}{index}", self.rope_type, self.theta, self.factor, self.low_freq_factor, self.high_freq_factor, self.original_max_position_embeddings)
        new_op.externals = self.externals
        new_op.expand_all_gpu_and_layers(len(self.device_list), 32)
        new_op.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
        new_op.set_stream(self.stream)
        self.nano_ops.append(new_op)

        return new_op

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
    
class RopeAppendTorch_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.op_layer = RopeAppendTorch_Layer

    def setShapeForIOWrappers(self):
        # The input tensor "kqv" is assumed to have a flattened layout:
        # [batch_size, (num_qo_heads + 2 * num_kv_heads) * head_dim]
        self.inputs["kqv"].init_shape((
            0,
            (self.parent.num_qo_heads + 2 * self.parent.num_kv_heads) * self.parent.head_dim,
        ))
        # The output "q" has shape [batch_size, num_qo_heads * head_dim]
        self.outputs["q"].init_shape((0, self.parent.num_qo_heads * self.parent.head_dim))

class RopeAppendTorch_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer, op_device)

    def run(self):
        self.impl.run(self.layer, self.inputs["kqv"].tensor, self.externals["KVCache"], self.outputs["q"].tensor, offset=0)
        