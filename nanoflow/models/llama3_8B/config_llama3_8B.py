from nanoflow.core.pipelineConfig import PipelineConfig

class Llama3_8B_Config(PipelineConfig):
    def __init__(
        self,
        multi_gpu_mode = False,
        vocab_size = 128256,
        hidden_dim = 4096,
        intermediate_dim = 14 * 1024,
        num_layers = 32,
        num_qo_heads = 32,
        num_kv_heads = 8,
        head_dim = 128,
        rms_norm_eps = 1e-05,
        rope_theta = 500000.0,
        page_size = 16,
        world_size = 1,
        world_rank = 0,
        tp_size = 1,
        tp_rank = 0,
        pp_size = 1,
        pp_rank = 0,
        dp_size = 1,
        dp_rank = 0,
        kv_cache_type = "flashinfer",
        unique_nccl_ids = [],
    ):
        self.multi_gpu_mode = multi_gpu_mode
        
        if multi_gpu_mode:
            self.pipeline_name = f"Llama3-8B-{kv_cache_type}-allreduce-TP{tp_size}-PP{pp_size}-DP{dp_size}"
        else:
            self.pipeline_name = f"Llama3-8B-{kv_cache_type}"
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
        self.page_size = page_size
        self.world_size = world_size
        self.world_rank = world_rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.kv_cache_type = kv_cache_type
        self.unique_nccl_ids = unique_nccl_ids