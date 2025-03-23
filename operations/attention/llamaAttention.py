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
    def run(self, layer, head_dim, num_qo_heads, num_kv_heads, qo_indicies,  Q, KVCache, output
    ):
        scale = 1.0 / (head_dim ** 0.5)
        # Compute group size: how many query heads correspond to one key/value head.
        group_size = num_qo_heads // num_kv_heads
        for i in range(len(qo_indicies) - 1):
            # Retrieve the query slice for this batch element.
            start = qo_indicies[i]
            end = qo_indicies[i + 1]

            sub_q = Q[start:end, :]  # shape: [n_q, num_qo_heads * head_dim]
            sub_q = sub_q.view(-1, num_qo_heads, head_dim)
            
            sub_k, sub_v = KVCache.get(layer, i) # [n_k, num_kv_heads * head_dim]
            n_k = sub_k.shape[0]

            sub_k = sub_k.view(n_k, num_kv_heads, head_dim)
            sub_v = sub_v.view(n_k, num_kv_heads, head_dim)

            sub_k = sub_k.repeat_interleave(group_size, dim=1)
            sub_v = sub_v.repeat_interleave(group_size, dim=1)

            scores = torch.einsum("qhd,khd->qhk", sub_q, sub_k) * scale

            attn_weights = torch.softmax(scores, dim=-1)
            
            # Compute attention output as the weighted sum over the value vectors.
            # Resulting shape: [n_q, num_qo_heads, head_dim]
            out = torch.einsum("qhk,khd->qhd", attn_weights, sub_v)
            # Flatten heads back to shape: [n_q, num_qo_heads * head_dim]
            out = out.reshape(-1, num_qo_heads * head_dim)
            # Write the computed output into the operator's output tensor.
            output[start:end, :].copy_(out)

class DecAttnCudaImpl(OperationImpl):
    category_tag = "cuda"
    def run(self, layer, head_dim, num_qo_heads, num_kv_heads, qo_indicies,  Q, KVCache, output
    ):
        if Q.shape[0] == 0:
            return
        scale = 1.0 / (head_dim ** 0.5)
        # Compute group size: how many query heads correspond to one key/value head.
        group_size = num_qo_heads // num_kv_heads
        for i in range(len(qo_indicies) - 1):
            # Retrieve the query slice for this batch element.
            start = qo_indicies[i]
            end = qo_indicies[i + 1]

            sub_q = Q[start:end, :]  # shape: [n_q, num_qo_heads * head_dim]
            sub_q = sub_q.view(-1, num_qo_heads, head_dim)
            
            sub_k, sub_v = KVCache[layer].get(i) # [n_k, num_kv_heads * head_dim]

            n_k = sub_k.shape[0]

            sub_k = sub_k.view(n_k, num_kv_heads, head_dim)
            sub_v = sub_v.view(n_k, num_kv_heads, head_dim)
            sub_q = sub_q.squeeze(0)
            out = flashinfer.single_decode_with_kv_cache(sub_q, sub_k, sub_v, use_tensor_cores=True)

            out = out.reshape(-1, num_qo_heads * head_dim)
            output[start:end, :].copy_(out)

class DecAttnBatchedCudaImpl(OperationImpl):
    category_tag = "batched_cuda"
    def run(self, layer, head_dim, num_qo_heads, num_kv_heads, qo_indicies,  Q, KVCache, output
    ):
        if Q.shape[0] == 0:
            return
        Q = Q.view(-1, num_qo_heads, head_dim)
        kv_indptr, kv_indices, kv_last_page_len = KVCache[layer].update()
        k_data, v_data = KVCache[layer].get_whole_kv_data()
        kv_data = [k_data, v_data]
        kv_data = tuple(kv_data)
        workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
        wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
            workspace_buffer, "HND"
        )
        wrapper.plan(
            torch.tensor(kv_indptr, dtype=torch.int32).cuda(),
            torch.tensor(kv_indices, dtype=torch.int32).cuda(),
            torch.tensor(kv_last_page_len, dtype=torch.int32).cuda(),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            KVCache[layer].get_pool().page_size,
            logits_soft_cap=0.0,
            pos_encoding_mode="NONE",
            data_type=torch.float16,
            q_data_type=torch.float16
        )

        Q.contiguous()

        o = wrapper.run(Q, kv_data)
        output.copy_(o.reshape(-1, num_qo_heads * head_dim))


    

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
        self.add_impl(DecAttnBatchedCudaImpl)
    
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
        self.impl.run(layer, self.head_dim, self.num_qo_heads, self.num_kv_heads, self.qo_indicies, Q, self.externals["KVCache"], self.outputs["output"].tensor)
    
class PFAttnTorchImpl(OperationImpl):
    category_tag = "torch"

    def run(self, layer, head_dim, num_qo_heads, num_kv_heads, qo_indicies, Q, KVCache, output
    ):
        scale = 1.0 / (head_dim ** 0.5)
        # Compute group size: how many query heads correspond to one key/value head.
        group_size = num_qo_heads // num_kv_heads

        for i in range(len(qo_indicies) - 1):
            # Retrieve the query slice for this batch element.
            start = qo_indicies[i]
            end = qo_indicies[i + 1]
            # Q is expected to be flattened as [n_total, num_qo_heads * head_dim];
            # extract the sub-tensor corresponding to this batch element.
            sub_q = Q[start:end, :]  # shape: [n_q, num_qo_heads * head_dim]
            sub_q = sub_q.view(-1, num_qo_heads, head_dim)

            sub_k, sub_v = KVCache.get(layer, i)
            n_k = sub_k.shape[0]

            sub_k = sub_k.view(n_k, num_kv_heads, head_dim)
            sub_v = sub_v.view(n_k, num_kv_heads, head_dim)
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

            out = out.reshape(-1, num_qo_heads * head_dim)
            # Write the computed output into th e operator's output tensor.
            output[start:end, :].copy_(out)

class PFAttnCudaImpl(OperationImpl):
    category_tag = "cuda"
    def run(self, layer, head_dim, num_qo_heads, num_kv_heads, qo_indicies, Q, KVCache, output
    ):
        if Q.shape[0] == 0:
            return
        # print("Q shape: ", Q.shape)
        # print("Q: ", Q)
        scale = 1.0 / (head_dim ** 0.5)
        # Compute group size: how many query heads correspond to one key/value head.
        group_size = num_qo_heads // num_kv_heads

        for i in range(len(qo_indicies) - 1):
            # Retrieve the query slice for this batch element.
            start = qo_indicies[i]
            end = qo_indicies[i + 1]
            # Q is expected to be flattened as [n_total, num_qo_heads * head_dim];
            # extract the sub-tensor corresponding to this batch element.
            sub_q = Q[start:end, :]  # shape: [n_q, num_qo_heads * head_dim]
            sub_q = sub_q.view(-1, num_qo_heads, head_dim)

            sub_k, sub_v = KVCache[layer].get(i)
            n_k = sub_k.shape[0]

            # Reshape keys and values so that the head dimension is explicit.
            # New shapes: [n_k, num_kv_heads, head_dim]
            sub_k = sub_k.view(n_k, num_kv_heads, head_dim)
            sub_v = sub_v.view(n_k, num_kv_heads, head_dim)
            sub_q = sub_q.contiguous()
            sub_k = sub_k.contiguous()
            sub_v = sub_v.contiguous()
            
            out = flashinfer.single_prefill_with_kv_cache(sub_q, sub_k, sub_v, causal=True)
            out = out.reshape(-1, num_qo_heads * head_dim)

            output[start:end, :].copy_(out)

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
        self.impl.run(layer, self.head_dim, self.num_qo_heads, self.num_kv_heads, self.qo_indicies, Q, self.externals["KVCache"], self.outputs["output"].tensor)