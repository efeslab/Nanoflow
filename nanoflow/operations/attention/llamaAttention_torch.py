import torch

from nanoflow.operations.operation_base import Operations, Operation_Layer
from nanoflow.core.IOWrapper import IOWrapper
from nanoflow.operations.impl_base import OperationImpl
from nanoflow.utils.util_functions import tensor_offset_to_req_idx


class DecAttnTorchImpl(OperationImpl):
    category_tag = "torch"
    def __init__(self, op_base: "DecAttnTorch", stream, device):
        super().__init__(op_base, stream, device)
        self.num_qo_heads = op_base.num_qo_heads // op_base.tp_size
        self.num_kv_heads = op_base.num_kv_heads // op_base.tp_size
        self.head_dim = op_base.head_dim

    def run(self, layer, Q, KVCache, output
    ):
        with torch.cuda.stream(self.stream):
            if Q.shape[0] == 0:
                return
            scale = 1.0 / (self.head_dim ** 0.5)
            # Compute group size: how many query heads correspond to one key/value head.
            group_size = self.num_qo_heads // self.num_kv_heads

            qo_indicies = self.op_base.qo_indicies
            input_req_idx = self.op_base.input_req_idx

            for i, global_index in enumerate(input_req_idx):
                # Retrieve the query slice for this batch element.
                start = qo_indicies[i]
                end = qo_indicies[i + 1]

                sub_q = Q[start:end, :]  # shape: [n_q, num_qo_heads * head_dim]
                sub_q = sub_q.view(-1, self.num_qo_heads, self.head_dim)
                
                sub_k, sub_v = KVCache.get(layer, global_index) # [n_k, num_kv_heads * head_dim]
                n_k = sub_k.shape[0]

                sub_k = sub_k.view(n_k, self.num_kv_heads, self.head_dim)
                sub_v = sub_v.view(n_k, self.num_kv_heads, self.head_dim)

                sub_k = sub_k.repeat_interleave(group_size, dim=1)
                sub_v = sub_v.repeat_interleave(group_size, dim=1)

                scores = torch.einsum("qhd,khd->qhk", sub_q, sub_k) * scale

                attn_weights = torch.softmax(scores, dim=-1)
                
                # Compute attention output as the weighted sum over the value vectors.
                # Resulting shape: [n_q, num_qo_heads, head_dim]
                out = torch.einsum("qhk,khd->qhd", attn_weights, sub_v)
                # Flatten heads back to shape: [n_q, num_qo_heads * head_dim]
                out = out.reshape(-1, self.num_qo_heads * self.head_dim)
                # Write the computed output into the operator's output tensor.
                output[start:end, :].copy_(out)

class DecAttnTorch(Operations):
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
        self.op_layer = DecAttnTorch_Layer

    def init_impl_map(self):
        self.add_impl(DecAttnTorchImpl)
    
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
        end_req_idx = tensor_offset_to_req_idx(qo_indicies, io.tensor_offset + io.batch_size)

        self.input_req_idx = self.externals["KVCache"].input_req_idx[start_req_idx:end_req_idx]

    def run(self, layer):
        self.impl.run(layer, self.inputs["Q"].tensor, self.externals["KVCache"], self.outputs["output"].tensor)

class DecAttnTorch_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        self.parent.run(self.layer)
    
class PFAttnTorchImpl(OperationImpl):
    category_tag = "torch"
    def __init__(self, op_base: "PFAttnTorch", stream, device):
        super().__init__(op_base, stream, device)
        self.num_qo_heads = op_base.num_qo_heads // op_base.tp_size
        self.num_kv_heads = op_base.num_kv_heads // op_base.tp_size
        self.head_dim = op_base.head_dim

    def run(self, layer, qo_indicies, Q, KVCache, output
    ):
        with torch.cuda.stream(self.stream):
            if Q.shape[0] == 0:
                return
            scale = 1.0 / (self.head_dim ** 0.5)
            # Compute group size: how many query heads correspond to one key/value head.
            group_size = self.num_qo_heads // self.num_kv_heads

            qo_indicies = self.op_base.qo_indicies
            input_req_idx = self.op_base.input_req_idx

            for i, global_index in enumerate(input_req_idx):
                # Retrieve the query slice for this batch element.
                start = qo_indicies[i]
                end = qo_indicies[i + 1]
                # Q is expected to be flattened as [n_total, num_qo_heads * head_dim];
                # extract the sub-tensor corresponding to this batch element.
                sub_q = Q[start:end, :]  # shape: [n_q, num_qo_heads * head_dim]
                sub_q = sub_q.view(-1, self.num_qo_heads, self.head_dim)

                sub_k, sub_v = KVCache.get(layer, global_index)
                n_k = sub_k.shape[0]

                sub_k = sub_k.view(n_k, self.num_kv_heads, self.head_dim)
                sub_v = sub_v.view(n_k, self.num_kv_heads, self.head_dim)
                # Expand (repeat) the keys and values so that they align with the query heads.
                sub_k = sub_k.repeat_interleave(group_size, dim=1)
                sub_v = sub_v.repeat_interleave(group_size, dim=1)

                scores = torch.einsum("qhd,khd->qhk", sub_q, sub_k) * scale

                n_q = sub_q.shape[0]
                n_k = sub_k.shape[0]
                past_length = max(n_k - n_q, 0)
                
                if past_length > 0:
                    new_mask = torch.tril(torch.ones(n_q, n_q, dtype=torch.bool, device=scores.device))
                    # For the past tokens (first past_length keys), we allow full attention.
                    past_mask = torch.ones(n_q, past_length, dtype=torch.bool, device=scores.device)
                    # Concatenate the masks along the key dimension.
                    causal_mask = torch.cat([past_mask, new_mask], dim=1)  # shape: [n_q, n_k]
                else:
                    # If there is no past context (i.e. n_k == n_q), use a standard lower-triangular mask.
                    causal_mask = torch.tril(torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device))

                scores = scores.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))
                
                # Apply softmax over the key dimension.
                attn_weights = torch.softmax(scores, dim=-1)
                
                out = torch.einsum("qhk,khd->qhd", attn_weights, sub_v)

                out = out.reshape(-1, self.num_qo_heads * self.head_dim)
                # Write the computed output into th e operator's output tensor.
                output[start:end, :].copy_(out)

class PFAttnCPUTorchImpl(OperationImpl):
    category_tag = "torch_cpu"
    def __init__(self, op_base: "PFAttnTorch", stream, device):
        super().__init__(op_base, stream, device)
        self.num_qo_heads = op_base.num_qo_heads // op_base.tp_size
        self.num_kv_heads = op_base.num_kv_heads // op_base.tp_size
        self.head_dim = op_base.head_dim

    def run(self, layer, qo_indicies, Q, KVCache, output
    ):
        # This implementation is similar to PFAttnTorchImpl but runs on CPU.
        # It is provided for completeness and may not be optimized for performance.
        pass  # Implementation would go here.

class PFAttnTorch(Operations):
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
        self.op_layer = PFAttnTorch_Layer

    def init_impl_map(self):
        self.add_impl(PFAttnTorchImpl)
    
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
        io = self.inputs["Q"]
        start_req_idx = tensor_offset_to_req_idx(qo_indicies, io.tensor_offset)
        end_req_idx = tensor_offset_to_req_idx(qo_indicies, io.tensor_offset + io.batch_size)

        self.qo_indicies = torch.tensor(qo_indicies[start_req_idx:end_req_idx + 1]) - io.tensor_offset
        self.input_req_idx = self.externals["KVCache"].input_req_idx[start_req_idx:end_req_idx]

    def run(self, layer):
        self.impl.run(layer, self.qo_indicies, self.inputs["Q"].tensor, self.externals["KVCache"], self.outputs["output"].tensor)

class PFAttnTorch_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)
    
    def run(self):
        self.parent.run(self.layer)