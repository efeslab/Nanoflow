import torch
import time
from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
from flash_attn import flash_attn_func


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
        pass

    def run(self, layer):
        # Q = self.inputs["Q"].tensor
        # KVdata = self.externals["KVdata"].tensor
        # # Q: (batch_size, q_dim), KVdata: (batch_size, 2 * num_kv_heads * head_dim)
        # # output: (batch_size, q_dim)
        # self.outputs["output"].tensor = Q @ KVdata.T
        pass
    
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

        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        scale = 1.0 / (self.head_dim ** 0.5)
        group_size = self.num_qo_heads // self.num_kv_heads

        for batch_size in batch_sizes:
            total_latency_new = 0
            total_latency_old = 0
            for round in range(rounds):
                input_q = torch.randn((batch_size, self.q_dim), dtype=torch.float16, device='cuda')
                input_k = torch.randn((batch_size, self.num_kv_heads * self.head_dim), dtype=torch.float16, device='cuda')
                input_v = torch.randn((batch_size, self.num_kv_heads * self.head_dim), dtype=torch.float16, device='cuda')
                output = torch.zeros((batch_size, self.q_dim), dtype=torch.float16, device='cuda')
                start_time = time.time()
                input_q = input_q.view(-1, self.num_qo_heads, self.head_dim)
                input_q = input_q.unsqueeze(0)

                input_k = input_k.view(-1, self.num_kv_heads, self.head_dim)
                n_k = input_k.shape[0]
                expanded_k_old = input_k.repeat_interleave(group_size, dim=1)
                expanded_k = expanded_k_old.unsqueeze(0)

                input_v = input_v.view(-1, self.num_kv_heads, self.head_dim)
                expanded_v_old = input_v.repeat_interleave(group_size, dim=1)
                expanded_v = expanded_v_old.unsqueeze(0)

                output = flash_attn_func(input_q, expanded_k, expanded_v, causal=True, softmax_scale=scale)
                if round > 0: # warm up
                    latency = time.time() - start_time
                    total_latency_new += latency


                start_time = time.time()

                sub_q = input_q.view(-1, self.num_qo_heads, self.head_dim)
                # Compute attention scores.
                # Use Einstein summation notation: for each query (q) and head (h), compute dot product with each key (k)
                # resulting in scores of shape: [n_q, num_qo_heads, n_k]
                scores = torch.einsum("qhd,khd->qhk", sub_q, expanded_k_old) * scale
                
                # -------------------------------
                # Add Causal Mask to Attention
                # -------------------------------
                # Determine the number of query tokens.
                n_q = sub_q.shape[0]
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
                out = torch.einsum("qhk,khd->qhd", attn_weights, expanded_v_old)
                # Flatten heads back to shape: [n_q, num_qo_heads * head_dim]
                out = out.reshape(-1, self.num_qo_heads * self.head_dim)
                # Write the computed output into the operator's output tensor.
                if round > 0:
                    latency_old = time.time() - start_time
                    total_latency_old += latency_old
            average_time = total_latency_new / rounds
            average_time_old = total_latency_old / rounds
            print(f"name: {self.name}, batch_size: {batch_size}, average_time: {average_time}")
            print(f"name: {self.name}_old, batch_size: {batch_size}, average_time: {average_time_old}")

            self.cursor.execute('''
                INSERT OR REPLACE INTO performance (id, keyword, batch_size, average_time)
                VALUES ((SELECT id FROM performance WHERE keyword = ? AND batch_size = ?), ?, ?, ?)
            ''', (self.name, batch_size, self.name, batch_size, average_time))
            self.cursor.execute('''
                INSERT OR REPLACE INTO performance (id, keyword, batch_size, average_time)
                VALUES ((SELECT id FROM performance WHERE keyword = ? AND batch_size = ?), ?, ?, ?)
            ''', (self.name + "_old", batch_size, self.name + "_old", batch_size, average_time_old))
            self.conn.commit()

    
    def run(self, layer):
        """
        For each batch element, this method retrieves the query portion (Q) of shape
        [n_q, num_qo_heads * head_dim] and then uses the external KV cache to get the stored keys and values.
        
        The attention operation is performed per head as follows:
        1. Reshape the query tensor to [n_q, num_qo_heads, head_dim].
        2. Retrieve the cached keys and values (sub_k and sub_v) and reshape them to
            [n_k, num_kv_heads, head_dim] (where n_k is the number of cached keys).
        3. Compute the group size as: group_size = num_qo_heads // num_kv_heads.
            This tells you how many query heads correspond to each key/value head.
        4. Expand (repeat) the key and value tensors along the head dimension so that each query head
            has a corresponding key and value (i.e. each key/value head is repeated group_size times).
        5. Compute attention scores for each head via dot-product scaling (using the factor 1/sqrt(head_dim)).
        6. **Apply a causal mask over the scores** so that each query position can only attend to
            allowed keys (i.e. those coming from the past or up to the current position among the new tokens).
        7. Apply softmax over the key dimension to obtain attention weights.
        8. Use the attention weights to compute a weighted sum over the value vectors.
        9. Flatten the output back to shape [n_q, num_qo_heads * head_dim] and write it to the output.
        
        The final output is written to self.outputs["output"].tensor.
        """
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
            # Reshape queries into separate heads.
            # New shape: [n_q, num_qo_heads, head_dim]
            sub_q = sub_q.view(-1, self.num_qo_heads, self.head_dim)
            
            # Retrieve cached keys and values from the external KV cache.
            # Expected shapes:
            #   sub_k: [n_k, num_kv_heads * head_dim]
            #   sub_v: [n_k, num_kv_heads * head_dim]
            sub_k, sub_v = self.externals["KVCache"].get(layer, i)
            n_k = sub_k.shape[0]
            # Reshape keys and values so that the head dimension is explicit.
            # New shapes: [n_k, num_kv_heads, head_dim]
            sub_k = sub_k.view(n_k, self.num_kv_heads, self.head_dim)
            sub_v = sub_v.view(n_k, self.num_kv_heads, self.head_dim)
            
            # Expand (repeat) the keys and values so that they align with the query heads.
            # Each key/value head is repeated group_size times.
            # New shapes after repeat: [n_k, num_qo_heads, head_dim]
            expanded_k = sub_k.repeat_interleave(group_size, dim=1)
            expanded_v = sub_v.repeat_interleave(group_size, dim=1)
            
            # # Compute attention scores.
            # # Use Einstein summation notation: for each query (q) and head (h), compute dot product with each key (k)
            # # resulting in scores of shape: [n_q, num_qo_heads, n_k]
            # scores = torch.einsum("qhd,khd->qhk", sub_q, expanded_k) * scale
            
            # # -------------------------------
            # # Add Causal Mask to Attention
            # # -------------------------------
            # # Determine the number of query tokens.
            # n_q = sub_q.shape[0]
            # # Assume that the KV cache has 'past' tokens (from previous timesteps) and new tokens,
            # # so that n_k = past_length + n_q.
            # past_length = max(n_k - n_q, 0)
            
            # if past_length > 0:
            #     # For the new tokens (the last n_q keys), build a lower-triangular mask.
            #     # For each query position i (0-indexed among the new tokens), allow attending only
            #     # to new keys with positions <= i.
            #     new_mask = torch.tril(torch.ones(n_q, n_q, dtype=torch.bool, device=scores.device))
            #     # For the past tokens (first past_length keys), we allow full attention.
            #     past_mask = torch.ones(n_q, past_length, dtype=torch.bool, device=scores.device)
            #     # Concatenate the masks along the key dimension.
            #     causal_mask = torch.cat([past_mask, new_mask], dim=1)  # shape: [n_q, n_k]
            # else:
            #     # If there is no past context (i.e. n_k == n_q), use a standard lower-triangular mask.
            #     causal_mask = torch.tril(torch.ones(n_q, n_k, dtype=torch.bool, device=scores.device))
            
            # # Expand the causal mask to match the scores' shape: [n_q, num_qo_heads, n_k].
            # # Then mask out disallowed (future) positions by setting them to -inf.
            # scores = scores.masked_fill(~causal_mask.unsqueeze(1), float("-inf"))
            
            # # Apply softmax over the key dimension.
            # attn_weights = torch.softmax(scores, dim=-1)
            
            # # Compute attention output as the weighted sum over the value vectors.
            # # Resulting shape: [n_q, num_qo_heads, head_dim]
            # out = torch.einsum("qhk,khd->qhd", attn_weights, expanded_v)
            # # Flatten heads back to shape: [n_q, num_qo_heads * head_dim]
            # out = out.reshape(-1, self.num_qo_heads * self.head_dim)
            # # Write the computed output into the operator's output tensor.
            # self.outputs["output"].tensor[start:end, :].copy_(out)

            # Use flash attention to compute the attention scores and output.
            sub_q = sub_q.unsqueeze(0)
            expanded_k = expanded_k.unsqueeze(0)
            expanded_v = expanded_v.unsqueeze(0)
            embed_dim = self.head_dim
            num_heads = self.num_qo_heads
            sub_q = sub_q.contiguous()
            expanded_k = expanded_k.contiguous()
            expanded_v = expanded_v.contiguous()

            output = flash_attn_func(sub_q, expanded_k, expanded_v, causal=True, softmax_scale=scale)
            output = output.reshape(-1, self.num_qo_heads * self.head_dim)

            # print(f"output shape: {output.shape}")
            self.outputs["output"].tensor[start:end, :].copy_(output)