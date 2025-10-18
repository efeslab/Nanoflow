from nanoflow.core.pipelineConfig import PipelineConfig

class Qwen2MoEConfig(PipelineConfig):
    def __init__(
        self,
        multi_gpu_mode = False,
        vocab_size = 151936,
        hidden_dim = 3584,
        intermediate_dim = 18944,
        num_layers = 28,
        num_qo_heads = 28,
        num_kv_heads = 4,
        head_dim = 128,
        rms_norm_eps = 1e-06,
        rope_theta = 1000000.0,
        moe_intermediate_dim = 2560,
        shared_expert_intermediate_dim = 20480,
        num_experts_per_tok = 8,
        num_experts = 64,
        num_shared_experts = 1,
        norm_topk_prob = False,
        page_size = 16,
        tp_size = 1,
        tp_rank = 0,
        pp_size = 1,
        pp_rank = 0,
        dp_size = 1,
        dp_rank = 0,
        ep_size = 1,
        ep_rank = 0,
        kv_cache_type = "flashinfer",
        unique_nccl_ids = [],
    ):
        if multi_gpu_mode:
            self.pipeline_name = f"Qwen2-MoE-57B-{kv_cache_type}-allreduce-TP{tp_size}-PP{pp_size}-DP{dp_size}-EP{ep_size}"
        else:
            self.pipeline_name = f"Qwen2-MoE-57B-{kv_cache_type}"
        self.cached_weight_dir = f"../cached_weights/{self.pipeline_name}"
        self.profile_dir = f"../profile_data/{self.pipeline_name}"

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.num_layers = num_layers
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.moe_intermediate_dim = moe_intermediate_dim
        self.shared_expert_intermediate_dim = shared_expert_intermediate_dim
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.norm_topk_prob = norm_topk_prob
        self.page_size = page_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.kv_cache_type = kv_cache_type
        self.unique_nccl_ids = unique_nccl_ids