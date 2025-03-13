import torch
import time
import flashinfer

from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
from operations.impl_base import OperationImpl
# from flash_attn import flash_attn_func

class DecAttnTorchImpl(OperationImpl):
    category_tag = "torch"
    def run(self, scale, head_dim, num_qo_heads, group_size, q, k, v, output
    ):
        # print("Get into DecAttnTorchImpl")
        # print("q: ", q.shape)
        # print("q: ", q)
        # print("k: ", k.shape)
        # print("v: ", v.shape)
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)
        # print("using torch")
        # Compute attention scores.
        # Use Einstein summation notation: for each query (q) and head (h), compute dot product with each key (k)
        # resulting in scores of shape: [n_q, num_qo_heads, n_k]
        scores = torch.einsum("qhd,khd->qhk", q, k) * scale
        
        # Apply softmax over the key dimension.
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Compute attention output as the weighted sum over the value vectors.
        # Resulting shape: [n_q, num_qo_heads, head_dim]
        out = torch.einsum("qhk,khd->qhd", attn_weights, v)
        # Flatten heads back to shape: [n_q, num_qo_heads * head_dim]
        out = out.reshape(-1, num_qo_heads * head_dim)
        # Write the computed output into the operator's output tensor.
        output.copy_(out)

class DecAttnCudaImpl(OperationImpl):
    category_tag = "cuda"
    def run(self, scale, head_dim, num_qo_heads, group_size, q, k, v, output
    ):
        if q.shape[0] == 0:
            return
        q = q.squeeze(0)
        out = flashinfer.single_decode_with_kv_cache(q, k, v, use_tensor_cores=True)

        out = out.reshape(-1, num_qo_heads * head_dim)
        output.copy_(out)

class DecAttn(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "Q": IOWrapper(self, 'Q', IOBufferType.ContinousPartition),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.ContinousPartition)
        }
        self.externals = {
            "KVCache": None
        }
        self.impl_map = {}
        self.init_impl_map()

    def init_impl_map(self):
        self.add_impl(DecAttnTorchImpl)
        self.add_impl(DecAttnCudaImpl)
    
    def setShape(self, num_kv_heads, num_qo_heads, head_dim):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.q_dim = num_qo_heads * head_dim
        
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["Q"].shape = (self.batch_size, self.q_dim)
        self.outputs["output"].shape = (self.batch_size, self.q_dim)
    
    def update(self, qo_indicies):
        self.qo_indicies = qo_indicies
    
    def profile(self):
        scale = 1.0 / (self.head_dim ** 0.5)
        group_size = self.num_qo_heads // self.num_kv_heads

        # check the similarity of the outputs
        input_q = torch.randn(1, self.num_qo_heads, self.head_dim, dtype=torch.float16, device='cuda')
        input_k = torch.randn(2, self.num_kv_heads, self.head_dim, dtype=torch.float16, device='cuda')
        input_v = torch.randn(2, self.num_kv_heads, self.head_dim, dtype=torch.float16, device='cuda')
        output_list = []
        for _, impl in self.impl_map.items():
            out = torch.zeros((1, self.q_dim), dtype=torch.float16, device='cuda')
            impl().run(scale, self.head_dim, self.num_qo_heads, group_size, input_q, input_k, input_v, out)
            output_list.append(out)
        
        self.checkConsistencyBetweenImpl(output_list)
        
        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        for batch_size in batch_sizes:
            output = torch.zeros((batch_size, self.q_dim), dtype=torch.float16, device='cuda')
            for _, impl in self.impl_map.items():
                impl_instance = impl()
                category_tag = impl.category_tag
                total_latency = 0
                for round in range(rounds):
                    input_q = torch.randn((1, self.num_qo_heads, self.head_dim), dtype=torch.float16, device='cuda')
                    input_k = torch.randn((batch_size, self.num_kv_heads, self.head_dim), dtype=torch.float16, device='cuda')
                    input_v = torch.randn((batch_size, self.num_kv_heads, self.head_dim), dtype=torch.float16, device='cuda')

                    # record the time
                    start_time = time.time()
                    impl_instance.run(scale, self.head_dim, self.num_qo_heads, group_size, input_q, input_k, input_v, output)
                    if round > 0:
                        total_latency += time.time() - start_time
                
                average_time = total_latency / rounds
                print("name: {}, batch_size: {}, average_time: {}".format(self.name + f"_{category_tag}", batch_size, average_time))
                self.cursor.execute('''
                    INSERT INTO performance (keyword, batch_size, average_time)
                    VALUES (?, ?, ?)
                    ''', (self.name + f"_{category_tag}", batch_size, average_time))
        self.conn.commit()

    def run(self, layer):
        Q = self.inputs["Q"].tensor
        scale = 1.0 / (self.head_dim ** 0.5)
        # Compute group size: how many query heads correspond to one key/value head.
        group_size = self.num_qo_heads // self.num_kv_heads
        for i in range(len(self.qo_indicies) - 1):
            # Retrieve the query slice for this batch element.
            start = self.qo_indicies[i]
            end = self.qo_indicies[i + 1]

            sub_q = Q[start:end, :]  # shape: [n_q, num_qo_heads * head_dim]
            sub_q = sub_q.view(-1, self.num_qo_heads, self.head_dim)
            
            sub_k, sub_v = self.externals["KVCache"].get(layer, i) # [n_k, num_kv_heads * head_dim]
            n_k = sub_k.shape[0]

            sub_k = sub_k.view(n_k, self.num_kv_heads, self.head_dim)
            sub_v = sub_v.view(n_k, self.num_kv_heads, self.head_dim)

            # print('before DecAttn run')
            # print(self.outputs["output"].tensor[start:end, :])
            # print(self.batch_size)
            self.impl.run(scale, self.head_dim, self.num_qo_heads, group_size, sub_q, sub_k, sub_v, self.outputs["output"].tensor[start:end, :])
            # print('after DecAttn run')
            # print(self.outputs["output"].tensor[start:end, :])
        # print(torch.allclose(self.outputs["output"].tensor, Q, rtol=1e-03, atol=1e-05))
    
class PFAttnTorchImpl(OperationImpl):
    category_tag = "torch"

    def run(self, scale, head_dim, num_qo_heads, group_size, q, k, v, output
    ):
                    
        # Expand (repeat) the keys and values so that they align with the query heads.
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)
        # print("using torch")
        
        # Compute attention scores.
        # Use Einstein summation notation: for each query (q) and head (h), compute dot product with each key (k)
        # resulting in scores of shape: [n_q, num_qo_heads, n_k]
        scores = torch.einsum("qhd,khd->qhk", q, k) * scale
        
        # -------------------------------
        # Add Causal Mask to Attention
        # -------------------------------
        # Determine the number of query tokens.
        n_q = q.shape[0]
        n_k = k.shape[0]
        # Assume that the KV cache has 'past' tokens (from previous timesteps) and new tokens,
        # so that n_k = past_length + n_q.
        past_length = max(n_k - n_q, 0)
        
        if past_length > 0:
            # For the new tokens (the last n_q keys), build a lower-triangular mask.
            # For each query position i (0-indexed among the new tokens), allow attending only
            # to new keys with positions <= i.
            new_mask = torch.tril(torch.ones(n_q, n_q, dtype=torch.bool, device=scores.device))
            # For the past tokens (first past_length keys), we allow full attention.
            past_mask = torch.ones(n_q, past_length, dtype=torch.bool, device=scores.device)
            # Concatenate the masks along the key dimension.
            causal_mask = torch.cat([past_mask, new_mask], dim=1)  # shape: [n_q, n_k]
        else:
            # If there is no past context (i.e. n_k == n_q), use a standard lower-triangular mask.
            causal_mask = torch.tril(torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device))
        
        # Expand the causal mask to match the scores' shape: [n_q, num_qo_heads, n_k].
        # Then mask out disallowed (future) positions by setting them to -inf.
        scores = scores.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))
        
        # Apply softmax over the key dimension.
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Compute attention output as the weighted sum over the value vectors.
        # Resulting shape: [n_q, num_qo_heads, head_dim]
        out = torch.einsum("qhk,khd->qhd", attn_weights, v)
        # Flatten heads back to shape: [n_q, num_qo_heads * head_dim]
        out = out.reshape(-1, num_qo_heads * head_dim)
        # Write the computed output into the operator's output tensor.
        output.copy_(out)

class PFAttnCudaImpl(OperationImpl):
    category_tag = "cuda"
    def run(self, scale, head_dim, num_qo_heads, group_size, q, k, v, output
    ):
        if q.shape[0] == 0:
            return
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        
        # out = flash_attn_func(q, k, v, causal=True, softmax_scale=scale)
        out = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
        out = out.reshape(-1, num_qo_heads * head_dim)

        # print(f"output shape: {output.shape}")
        output.copy_(out)

class PFAttn(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "Q": IOWrapper(self, 'Q', IOBufferType.ContinousPartition),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.ContinousPartition)
        }
        # Note: for consistency with other operators (like RopeAppend), we expect the external KV cache to be
        # available as "KVCache". If needed, you can change the key name.
        self.externals = {
            "KVCache": None
        }
        self.impl_map = {}
        self.init_impl_map()

    def init_impl_map(self):
        self.add_impl(PFAttnTorchImpl)
        self.add_impl(PFAttnCudaImpl)
    
    def setShape(self, num_kv_heads, num_qo_heads, head_dim):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.q_dim = num_qo_heads * head_dim
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["Q"].shape = (self.batch_size, self.q_dim)
        self.outputs["output"].shape = (self.batch_size, self.q_dim)
    
    def update(self, qo_indicies):
        """Stores the query offset indices for each batch element.  
        qo_indicies should be a list (or tensor) of length (batch_size + 1) such that for each batch index i,  
        the query slice is Q[qo_indicies[i]:qo_indicies[i+1], :].
        """
        self.qo_indicies = qo_indicies

    def profile(self):
        scale = 1.0 / (self.head_dim ** 0.5)
        # Compute group size: how many query heads correspond to one key/value head.
        group_size = self.num_qo_heads // self.num_kv_heads

        # check the similarity of the outputs
        input_q = torch.randn(2, self.num_qo_heads, self.head_dim, dtype=torch.float16, device='cuda')
        input_k = torch.randn(2, self.num_kv_heads, self.head_dim, dtype=torch.float16, device='cuda')
        input_v = torch.randn(2, self.num_kv_heads, self.head_dim, dtype=torch.float16, device='cuda')
        output_list = []
        for _, impl in self.impl_map.items():
            out = torch.zeros((2, self.q_dim), dtype=torch.float16, device='cuda')
            impl().run(scale, self.head_dim, self.num_qo_heads, group_size, input_q, input_k, input_v, out)
            output_list.append(out)
        
        self.checkConsistencyBetweenImpl(output_list)

        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        for batch_size in batch_sizes:
            output = torch.zeros((batch_size, self.q_dim), dtype=torch.float16, device='cuda')
            for _, impl in self.impl_map.items():
                impl_instance = impl()
                category_tag = impl.category_tag
                total_latency = 0
                for round in range(rounds):
                    input_q = torch.randn((batch_size, self.num_qo_heads, self.head_dim), dtype=torch.float16, device='cuda')
                    input_k = torch.randn((batch_size, self.num_kv_heads, self.head_dim), dtype=torch.float16, device='cuda')
                    input_v = torch.randn((batch_size, self.num_kv_heads, self.head_dim), dtype=torch.float16, device='cuda')
                    # record the time
                    start_time = time.time()
                    impl_instance.run(scale, self.head_dim, self.num_qo_heads, group_size, input_q, input_k, input_v, output)
                    if round > 0:
                        total_latency += time.time() - start_time
                
                average_time = total_latency / rounds
                print("name: {}, batch_size: {}, average_time: {}".format(self.name + f"_{category_tag}", batch_size, average_time))
                self.cursor.execute('''
                    INSERT INTO performance (keyword, batch_size, average_time)
                    VALUES (?, ?, ?)
                    ''', (self.name + f"_{category_tag}", batch_size, average_time))
        self.conn.commit()
    
    def run(self, layer):
        Q = self.inputs["Q"].tensor
        scale = 1.0 / (self.head_dim ** 0.5)
        # Compute group size: how many query heads correspond to one key/value head.
        group_size = self.num_qo_heads // self.num_kv_heads

        for i in range(len(self.qo_indicies) - 1):
            # Retrieve the query slice for this batch element.
            start = self.qo_indicies[i]
            end = self.qo_indicies[i + 1]
            # Q is expected to be flattened as [n_total, num_qo_heads * head_dim];
            # extract the sub-tensor corresponding to this batch element.
            sub_q = Q[start:end, :]  # shape: [n_q, num_qo_heads * head_dim]
            sub_q = sub_q.view(-1, self.num_qo_heads, self.head_dim)

            sub_k, sub_v = self.externals["KVCache"].get(layer, i)
            n_k = sub_k.shape[0]

            # Reshape keys and values so that the head dimension is explicit.
            # New shapes: [n_k, num_kv_heads, head_dim]
            sub_k = sub_k.view(n_k, self.num_kv_heads, self.head_dim)
            sub_v = sub_v.view(n_k, self.num_kv_heads, self.head_dim)

            # print('before PFAttn run')
            # print(self.outputs["output"].tensor[start:end, :])
            # print(self.batch_size)
            self.impl.run(scale, self.head_dim, self.num_qo_heads, group_size, sub_q, sub_k, sub_v, self.outputs["output"].tensor[start:end, :])
            # print('after PFAttn run')
            # print(self.outputs["output"].tensor[start:end, :])
        # print(torch.allclose(self.outputs["output"].tensor, Q, rtol=1e-03, atol=1e-05))