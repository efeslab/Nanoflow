from pathlib import Path
import os

class PipelineConfig:
    pipeline_name: str 
    cached_weight_dir: str
    profile_dir: str

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

    def has_cached_weight(self) -> bool:
        flag = Path(self.cached_weight_dir).exists()
        os.makedirs(self.cached_weight_dir, exist_ok=True)
        return flag
    
    def profile_data_path(self) -> str:
        return f"../profile_data/{self.pipeline_name}"