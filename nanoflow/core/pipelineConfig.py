from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PipelineConfig:
    pipeline_name_prefix: str
    pipeline_name: str 
    num_kv_heads: int
    num_qo_heads: int
    head_dim: int
    vocab_size: int
    hidden_dim: int
    intermediate_dim: int
    num_layers: int
    rms_norm_eps: float
    rope_theta: float
    page_size: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    dp_size: int
    dp_rank: int
    unique_nccl_ids: list[str]
