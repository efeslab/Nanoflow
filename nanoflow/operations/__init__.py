# only init the operation that depends on Base class and IO/WeightWrapper


from .operation_base import NanoOpInfo, Operations, Operation_Layer
from .impl_base import OperationImpl

from .activation.silu import Activation
from .allreduce.allreduce import AllReduce
from .embedding.embedding import GenEmbedding
from .globalOp.globalOp import GlobalInput, GlobalOutput
from .gemm.gemm_N_parallel import GEMM_N_Parallel
from .gemm.gemm_K_parallel import GEMM_K_Parallel
from .norm.rmsnorm import LayerNorm
from .sampling.max_sampling import Sampling
from .rope.rope_flashinfer import RopeAppendFlashinfer
from .attention.llamaAttention_flashinfer import (
    DecAttnFlashinfer,
    PFAttnFlashinfer,
)
from .virtualOp.virtual_ops import Copy, Redist