import logging
import torch

from operations.operation_base import Operations, Operation_Device, Operation_Layer
from core.IOWrapper import IOWrapper
from operations.impl_base import OperationImpl
from kvcache.kv import KVCacheFANoPage
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # type: ignore[import]


class DecAttnFANoPageImpl(OperationImpl):
    r"""FlashAttention implementation of the DecAttn operator.

    This implementation uses the flash_attn library to perform the decoding attention.
    """

    category_tag = "flash_attn_no_page"

    def __init__(self, op_base: "DecAttnFA", stream: torch.cuda.Stream, device_id: int) -> None:
        r"""Initialize the DecAttn operator.

        Parameters
        ----------
        op_base : DecAttn
            The base operator it implements.
        device_id : int
            The device ID.
        """
        super().__init__(op_base, stream, device_id)  # type: ignore
        self.device_id = device_id
        self.num_qo_heads = int(op_base.num_qo_heads)  # type: ignore
        self.num_kv_heads = int(op_base.num_kv_heads)  # type: ignore
        self.head_dim = int(op_base.head_dim)  # type: ignore
        self.scale = 1.0 / (self.head_dim**0.5)
    
    def run(
        self,
        layer: int,
        qo_indicies: torch.Tensor,
        Q: torch.Tensor,
        kv_tuple: None,
        KVCache: KVCacheFANoPage,
        output: torch.Tensor,
    ) -> None:
        r"""Run the DecAttn operator.
        
        Parameters
        ----------
        layer : int
            The layer index.g
        qo_indicies : torch.Tensor
            The indices that mark the start and end of the query slices for each batch request.
        Q : torch.Tensor
            The query tensor.
            Shape: [n_total, num_qo_heads * head_dim]
        kv_tuple : tuple[torch.Tensor | None, torch.Tensor | None]
            Unrelated.
        KVCache : KVCacheFANoPage
            The KV cache in flash_attn layout. Note that here we use the last_kv it stores.
            Shape: list of [batch_size, max_seq_len, num_kv_heads, head_dim]
        output : torch.Tensor
            The output tensor.
            Shape: [n_total, num_qo_heads * head_dim]
        
        Notes
        -----
        Deprecated. Use `run` instead.
        """
        if Q.shape[0] == 0:
            return        
        q = Q.view(-1, 1, self.num_qo_heads, self.head_dim)
        k_cache, v_cache = KVCache.get_whole_kv_data(self.device_id, layer)
        assert k_cache is not None and v_cache is not None
        o = flash_attn_with_kvcache( # type: ignore
            q,
            k_cache,
            v_cache,
            cache_seqlens=KVCache.get_whole_indices(),
        )
        assert isinstance(o, torch.Tensor)
        o = o.view(-1, self.num_qo_heads * self.head_dim)

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            q = q.view(-1, self.num_qo_heads * self.head_dim)
            k_list: list[torch.Tensor] = []
            v_list: list[torch.Tensor] = []
            cache_seqlens = KVCache.get_whole_indices().tolist()  # type: ignore
            for i, seq_len in enumerate(cache_seqlens):  # type: ignore
                k_list.append(k_cache[i].view(-1, self.num_kv_heads * self.head_dim)[:seq_len])  # type: ignore
                v_list.append(v_cache[i].view(-1, self.num_kv_heads * self.head_dim)[:seq_len])  # type: ignore
            k = torch.cat(k_list, dim=0)
            v = torch.cat(v_list, dim=0)
            logging.debug(f"q.shape {q.shape}\nq {q}\nk.shape {k.shape}\nk {k}\nv.shape {v.shape}\nv {v}\no.shape {o.shape}\no {o}")
        output.copy_(o)


class DecAttnFA(Operations):
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
        self.op_device = DecAttn_Device

    def init_impl_map(self):
        self.add_impl(DecAttnFANoPageImpl)
    
    def setShape(self, num_kv_heads, num_qo_heads, head_dim):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.q_dim = num_qo_heads * head_dim
        for op_device in self.children:
            op_device.setShapeForIOWrappers()
    
    def update(self, cumsum_input: list[int], device_id: int):
        self.qo_indicies = torch.tensor(cumsum_input, dtype=torch.int32, device=f"cuda:{device_id}")
    
    def profile(self):
        pass
    
class DecAttn_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.op_layer = DecAttn_Layer 

    def setShapeForIOWrappers(self):
        self.inputs["Q"].init_shape((0, self.parent.num_qo_heads* self.parent.head_dim))
        self.outputs["output"].init_shape((0, self.parent.num_qo_heads * self.parent.head_dim))

class DecAttn_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer, op_device=op_device)
        self.k_data_ptr, self.v_data_ptr = op_device.externals["KVCache"].get_whole_kv_data(self.device_id, self.layer)
        self.kv_tuple = tuple([self.k_data_ptr, self.v_data_ptr])

    def run(self):
        Q = self.inputs["Q"].tensor
        # self.operator_device.parent.impl.run(Q, self.kv_tuple, self.outputs["output"].tensor)
        self.impl.run(self.layer, self.parent.parent.qo_indicies,  Q, self.kv_tuple, self.parent.externals["KVCache"], self.outputs["output"].tensor)


class PFAttnFANoPageImpl(OperationImpl):
    r"""FlashAttention implementation of the PFAttn operator.

    This implementation uses the flash_attn library to perform the prefill attention.
    """

    category_tag = "flash_attn_no_page"

    def __init__(self, op_base: "PFAttnFA", stream: torch.cuda.Stream, device_id: int):
        r"""Initialize the PFAttn operator.

        Parameters
        ----------
        op_base : PFAttn
            The base operator it implements.
        device_id : int
            The device ID.
        """
        super().__init__(op_base, stream, device_id)  # type: ignore
        self.device_id = device_id
        self.num_qo_heads = int(op_base.num_qo_heads)  # type: ignore
        self.num_kv_heads = int(op_base.num_kv_heads)  # type: ignore
        self.head_dim = int(op_base.head_dim)  # type: ignore
        self.scale = 1.0 / (self.head_dim**0.5)


    def run(
        self,
        layer: int,
        qo_indicies: torch.Tensor,
        Q: torch.Tensor,
        kv_tuple: None,
        KVCache: KVCacheFANoPage,
        output: torch.Tensor,
    ):
        r"""Run the PFAttn operator.
        
        Parameters
        ----------
        layer : int
            The layer index.g
        qo_indicies : torch.Tensor
            The indices that mark the start and end of the query slices for each batch request.
        Q : torch.Tensor
            The query tensor.
            Shape: [n_total, num_qo_heads * head_dim]
        kv_tuple : tuple[torch.Tensor | None, torch.Tensor | None]
            Unrelated.
        KVCache : KVCacheFANoPage
            The KV cache in flash_attn layout. Note that here we use the last_kv it stores.
            Shape: list of [batch_size, max_seq_len, num_kv_heads, head_dim]
        output : torch.Tensor
            The output tensor.
            Shape: [n_total, num_qo_heads * head_dim]
        """
        if Q.shape[0] == 0:
            return
        q = Q.view(-1, self.num_qo_heads, self.head_dim)
        k, v = KVCache.get_last_kv(self.device_id, layer)
        o = flash_attn_varlen_func( # type: ignore
            q,
            k,
            v,
            cu_seqlens_q=qo_indicies,
            cu_seqlens_k=qo_indicies,
            max_seqlen_q=self.op_base.max_seqlen_q, # type: ignore
            max_seqlen_k=self.op_base.max_seqlen_k, # type: ignore
            softmax_scale=self.scale,
            causal=True
        )
        assert isinstance(o, torch.Tensor)
        o = o.view(-1, self.num_qo_heads * self.head_dim)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            q = q.view(-1, self.num_qo_heads * self.head_dim)
            k = k.view(-1, self.num_kv_heads * self.head_dim)
            v = v.view(-1, self.num_kv_heads * self.head_dim)
            logging.debug(f"q.shape {q.shape}\nq {q}\nk.shape {k.shape}\nk {k}\nv.shape {v.shape}\nv {v}\no.shape {o.shape}\no {o}")
        output.copy_(o)

class PFAttnFA(Operations):
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
        self.op_device = PFAttnFA_Device

    def init_impl_map(self):
        self.add_impl(PFAttnFANoPageImpl)
    
    def setShape(self, num_kv_heads, num_qo_heads, head_dim):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.q_dim = num_qo_heads * head_dim
        for op_device in self.children:
            op_device.setShapeForIOWrappers()
    
    def update(self, cumsum_input: list[int], device_id: int):
        self.qo_indicies = torch.tensor(cumsum_input, dtype=torch.int32, device=f"cuda:{device_id}")
        seq_lens = self.qo_indicies.diff()
        self.max_seqlen_q = torch.max(seq_lens).item()
        self.max_seqlen_k = self.max_seqlen_q

    def profile(self):
        raise NotImplementedError("Profile not implemented for PFAttnFA")

    
class PFAttnFA_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.op_layer = PFAttnFA_Layer 

    def setShapeForIOWrappers(self):
        self.inputs["Q"].init_shape((0, self.parent.num_qo_heads * self.parent.head_dim))
        self.outputs["output"].init_shape((0, self.parent.num_qo_heads * self.parent.head_dim))


class PFAttnFA_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer=layer, op_device=op_device)
        self.k_data_ptr, self.v_data_ptr = op_device.externals["KVCache"].get_whole_kv_data(self.device_id, self.layer)
        self.kv_tuple = tuple([self.k_data_ptr, self.v_data_ptr])

    
    def run(self):
        Q = self.inputs["Q"].tensor
        # self.operator_device.parent.impl.run(Q, self.kv_tuple, self.outputs["output"].tensor)
        self.impl.run(self.layer, self.parent.parent.qo_indicies,  Q, self.kv_tuple, self.parent.externals["KVCache"], self.outputs["output"].tensor)
