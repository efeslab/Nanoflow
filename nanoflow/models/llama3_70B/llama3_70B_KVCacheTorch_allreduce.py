import torch
import torch.distributed as dist

from nanoflow.operations import NanoOpInfo

from nanoflow.operations import (
    GlobalInput,
    GlobalOutput,
    GenEmbedding,
    LayerNorm,
    GEMM_N_Parallel,
    GEMM_K_Parallel,
    RopeAppendTorch,
    DecAttnTorch,
    PFAttnTorch,
    Activation,
    AllReduce,
    Sampling,
    Copy,
    Redist,
)

from nanoflow.kvcache.kv import KVCacheTorch

from nanoflow.core.basePipeline import BasePipeline
from nanoflow.core import CategoryType
from nanoflow.core.nanobatchSplit import split_nanobatch

from .config_llama3_70B import Llama3_70B_Config


class Pipeline(BasePipeline):
    def __init__(self, cfg: Llama3_70B_Config) -> None:
        # Set parameters as instance variables.
        super().__init__(
            pipeline_name=cfg.pipeline_name,
            cache_weight_name=cfg.cache_weight_name,
            cached_weight_dir=cfg.cached_weight_dir,
            profile_dir=cfg.profile_dir,
            num_layers=cfg.num_layers,
            world_size=cfg.world_size,
            world_rank=cfg.world_rank,
            categories=[CategoryType.COMP, CategoryType.NET],
        )

        self.num_kv_heads = cfg.num_kv_heads
        self.num_qo_heads = cfg.num_qo_heads
        self.head_dim = cfg.head_dim
        self.vocab_size = cfg.vocab_size
        self.hidden_dim = cfg.hidden_dim
        self.intermediate_dim = cfg.intermediate_dim
        self.rms_norm_eps = cfg.rms_norm_eps
        self.rope_theta = cfg.rope_theta
        self.page_size = cfg.page_size
        self.tp_size = cfg.tp_size
        self.tp_rank = cfg.tp_rank
        self.unique_nccl_ids = cfg.unique_nccl_ids

        self.kqv_heads = self.num_qo_heads + 2 * self.num_kv_heads

    def init_external_data(self) -> None:
        print("Initializing external data...")
        self.kv_cache = KVCacheTorch(self.num_kv_heads, self.head_dim, tp_size=self.tp_size)

    def init_operations(self) -> None:
        self.global_input = (
            GlobalInput("GlobalInput", self.device).setShape().first_only()
        )
        self.global_input_layers = self.global_input.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.global_input)

        self.gen_embedding = (
            GenEmbedding("GenEmbedding", self.device)
            .setWeightName("model.embed_tokens.weight")
            .setShape(self.hidden_dim, self.vocab_size)
            .first_only()
        )
        self.gen_embedding_layers = self.gen_embedding.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.gen_embedding)

        self.layerNormAttn = LayerNorm(
            "LayerNormAttn", self.device, eps=self.rms_norm_eps
        ).setWeightName("model.layers.{layer}.input_layernorm.weight").setShape(self.hidden_dim)
        self.layerNormAttn_layers = self.layerNormAttn.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.layerNormAttn)

        self.kqv = GEMM_N_Parallel("KQV", self.device).setWeightName(
            [
                "model.layers.{layer}.self_attn.k_proj.weight",
                "model.layers.{layer}.self_attn.v_proj.weight",
                "model.layers.{layer}.self_attn.q_proj.weight",
            ]
        ).setShape(
            self.kqv_heads * self.head_dim,
            self.hidden_dim,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        ).setParameter(1.0, 0.0)
        self.kqv_layers = self.kqv.expand_layer(self.layer_list)
        self.original_model_operations.append(self.kqv)

        self.ropeAppend = RopeAppendTorch("RopeAppend", self.device, theta=self.rope_theta).setShape(
            self.num_kv_heads, self.num_qo_heads, self.head_dim, tp_size=self.tp_size
        )
        self.ropeAppend.externals["KVCache"] = self.kv_cache
        self.ropeAppend_layers = self.ropeAppend.expand_layer(self.layer_list)
        self.original_model_operations.append(self.ropeAppend)

        self.decAttn = DecAttnTorch("DecAttn", self.device).setShape(
            self.num_kv_heads, self.num_qo_heads, self.head_dim, tp_size=self.tp_size
        )
        self.decAttn.externals["KVCache"] = self.kv_cache
        self.decAttn_layers = self.decAttn.expand_layer(self.layer_list)
        self.original_model_operations.append(self.decAttn)

        self.pfAttn = PFAttnTorch("PFAttn", self.device).setShape(
            self.num_kv_heads, self.num_qo_heads, self.head_dim, tp_size=self.tp_size
        )
        self.pfAttn.externals["KVCache"] = self.kv_cache
        self.pfAttn_layers = self.pfAttn.expand_layer(self.layer_list)
        self.original_model_operations.append(self.pfAttn)

        self.o = GEMM_K_Parallel("O", self.device, bias=True).setWeightName(
            "model.layers.{layer}.self_attn.o_proj.weight"
        ).setShape(
            self.hidden_dim, self.hidden_dim, tp_rank=self.tp_rank, tp_size=self.tp_size
        ).setParameter(1.0, 1.0 / self.tp_size)
        self.o_layers = self.o.expand_layer(self.layer_list)
        self.original_model_operations.append(self.o)

        self.allReduce_o = AllReduce("AllReduceO", self.device).setShape(
            self.hidden_dim, rank=self.tp_rank, world_size=self.tp_size
        )
        self.allReduce_o_layers = self.allReduce_o.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.allReduce_o)

        self.layerNormFFN = LayerNorm(
            "LayerNormFFN", device=self.device, eps=self.rms_norm_eps
        ).setWeightName("model.layers.{layer}.post_attention_layernorm.weight").setShape(self.hidden_dim)
        self.layerNormFFN_layers = self.layerNormFFN.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.layerNormFFN)

        self.ug = GEMM_N_Parallel("UG", self.device).setWeightName(
            [
                "model.layers.{layer}.mlp.up_proj.weight",
                "model.layers.{layer}.mlp.gate_proj.weight",
            ]
        ).setShape(
            self.intermediate_dim * 2,
            self.hidden_dim,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        ).setParameter(1.0, 0.0)
        self.ug_layers = self.ug.expand_layer(self.layer_list)
        self.original_model_operations.append(self.ug)

        self.activation = Activation("Activation", self.device).setShape(
            self.intermediate_dim, tp_rank=self.tp_rank, tp_size=self.tp_size
        )
        self.activation_layers = self.activation.expand_layer(self.layer_list)
        self.original_model_operations.append(self.activation)

        self.d = GEMM_K_Parallel("D", self.device, bias=True).setWeightName(
            "model.layers.{layer}.mlp.down_proj.weight"
        ).setShape(
            self.hidden_dim,
            self.intermediate_dim,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
        ).setParameter(1.0, 1.0 / self.tp_size)
        self.d_layers = self.d.expand_layer(self.layer_list)
        self.original_model_operations.append(self.d)

        self.allReduce_d = AllReduce("AllReduceD", self.device).setShape(
            self.hidden_dim, rank=self.tp_rank, world_size=self.tp_size
        )
        self.allReduce_d_layers = self.allReduce_d.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.allReduce_d)

        self.getLogits = (
            GEMM_N_Parallel("GetLogits", self.device)
            .setWeightName("lm_head.weight")
            .setShape(self.vocab_size, self.hidden_dim)
            .setParameter(alpha=1.0, beta=0.0)
            .last_only()
        )
        self.getLogits_layers = self.getLogits.expand_layer(self.layer_list)
        self.original_model_operations.append(self.getLogits)

        self.modelLayerNorm = (
            LayerNorm("ModelLayerNorm", self.device, eps=self.rms_norm_eps)
            .setWeightName("model.norm.weight")
            .setShape(self.hidden_dim)
            .last_only()
        )
        self.modelLayerNorm_layers = self.modelLayerNorm.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.modelLayerNorm)

        self.sample = Sampling("Sampling", self.device).setShape(
            self.vocab_size).last_only()
        self.sample_layers = self.sample.expand_layer(self.layer_list)
        self.original_model_operations.append(self.sample)

        self.global_output = GlobalOutput(
            "GlobalOutput", self.device).setShape().last_only()
        self.global_output_layers = self.global_output.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.global_output)

        self.copy_embedding = Copy(
            "CopyEmbedding", self.device, num_inputs=2, num_outputs=2
        )
        self.original_virtual_operations.append(self.copy_embedding)

        self.copy_o = Copy("CopyO", self.device, num_inputs=1, num_outputs=2)
        self.original_virtual_operations.append(self.copy_o)

        self.copy_d = Copy("CopyD", self.device, num_inputs=1, num_outputs=2)
        self.original_virtual_operations.append(self.copy_d)

        self.redist_p = Redist(
            "RedistPartition", self.device, num_inputs=1, num_outputs=2
        )
        self.original_virtual_operations.append(self.redist_p)

        self.redist_a = Redist(
            "RedistAggregation", self.device, num_inputs=2, num_outputs=1
        )
        self.original_virtual_operations.append(self.redist_a)

        self.model_operations = self.original_model_operations
        self.virtual_operations = self.original_virtual_operations
        self.all_operations = (
            self.model_operations + self.virtual_operations
        )
        for operation in self.model_operations:
            self.all_layer_operations.extend(operation.children)
        # NOTE(Ziren): for further nanosplit or auto search, which should keep the original operations since we need to change the strategy of optimization in the runtime.

    def init_dependency(self) -> None:
        self.global_input.outputs["tokens"] >> self.gen_embedding.inputs["token"]

        self.gen_embedding.outputs["output"] >> self.copy_embedding.inputs["input_0"]
        self.copy_embedding.outputs["output_0"] >> self.layerNormAttn.inputs["input"]
        self.copy_embedding.outputs["output_1"] >> self.o.inputs["C"]

        self.layerNormAttn.outputs["output"] >> self.kqv.inputs["A"]

        self.kqv.outputs["D"] >> self.ropeAppend.inputs["kqv"]

        self.ropeAppend.outputs["q"] >> self.redist_p.inputs["input_0"]
        self.redist_p.outputs["output_0"] >> self.decAttn.inputs["Q"]
        self.redist_p.outputs["output_1"] >> self.pfAttn.inputs["Q"]

        self.decAttn.outputs["output"] >> self.redist_a.inputs["input_0"]
        self.pfAttn.outputs["output"] >> self.redist_a.inputs["input_1"]
        self.redist_a.outputs["output_0"] >> self.o.inputs["A"]

        self.o.outputs["D"] >> self.allReduce_o.inputs["input"]
        self.allReduce_o.outputs["output"] >> self.copy_o.inputs["input_0"]

        self.copy_o.outputs["output_0"] >> self.layerNormFFN.inputs["input"]
        self.copy_o.outputs["output_1"] >> self.d.inputs["C"]

        self.layerNormFFN.outputs["output"] >> self.ug.inputs["A"]

        self.ug.outputs["D"] >> self.activation.inputs["input"]

        self.activation.outputs["output"] >> self.d.inputs["A"]

        self.d.outputs["D"] >> self.allReduce_d.inputs["input"]
        self.allReduce_d.outputs["output"] >> self.copy_d.inputs["input_0"]

        self.copy_d.outputs["output_0"] >> self.modelLayerNorm.inputs["input"]
        self.copy_d.outputs["output_1"] >> (self.copy_embedding.inputs["input_1"], True)

        self.modelLayerNorm.outputs["output"] >> self.getLogits.inputs["A"]

        self.getLogits.outputs["D"] >> self.sample.inputs["logits"]

        self.sample.outputs["tokens"] >> self.global_output.inputs["tokens"]

        for operation in self.all_operations:
            operation.checkConnection()

    def init_category(self):
        # set category for loop operations
        self.layerNormAttn.set_category(CategoryType.COMP)
        self.kqv.set_category(CategoryType.COMP)
        self.ropeAppend.set_category(CategoryType.COMP)
        self.decAttn.set_category(CategoryType.COMP)
        self.pfAttn.set_category(CategoryType.COMP)
        self.layerNormFFN.set_category(CategoryType.COMP)
        self.o.set_category(CategoryType.COMP)
        self.allReduce_o.set_category(CategoryType.NET)
        self.ug.set_category(CategoryType.COMP)
        self.activation.set_category(CategoryType.COMP)
        self.d.set_category(CategoryType.COMP)
        self.allReduce_d.set_category(CategoryType.NET)

    def apply_batch_size(self) -> None:
        print(
            "Configuring batch sizes: global_batch_size =",
            self.global_batch_size,
            ", decode_batch_size =",
            self.decode_batch_size,
        )
        self.global_input.setBatchSize(self.global_batch_size)
        self.decAttn.setBatchSize(self.decode_batch_size)

    def config_algorithm(self) -> None:
        print("Configuring algorithms...")
        params = {
            "use_cuda_graph": self.is_cuda_graph_enabled,
        }

        self.gen_embedding.config_tag("torch", params)

        if self.is_auto_search_enabled:
            for op in self.model_operations:
                print(
                    f"op.name: {op.name}, op.original_name: {op.original_name}")
                if op.original_name in self.profile_result["operations"]:
                    algo_tag = self.profile_result["operations"][op.original_name][
                        op.name
                    ]["algo_tag"]
                    op.config_tag(algo_tag, params)
        else:
            self.layerNormAttn.config_tag("torch", params)
            self.kqv.config_tag("torch", params)
            self.ropeAppend.config_tag("torch:withKVCache", params)
            self.decAttn.config_tag("torch", params)
            self.pfAttn.config_tag("torch", params)
            self.layerNormFFN.config_tag("torch", params)
            self.o.config_tag("torch", params)
            self.allReduce_o.config_tag("torch", params)
            self.ug.config_tag("torch", params)
            self.activation.config_tag("torch", params)
            self.d.config_tag("torch", params)
            self.allReduce_d.config_tag("torch", params)

        self.getLogits.config_tag("torch", params)
        self.modelLayerNorm.config_tag("torch", params)
        self.sample.config_tag("torch", params)

    def config_network(self) -> None:
        dist.init_process_group(
            backend="nccl", rank=self.world_rank, world_size=self.world_size
        )
        tp_group_idx = self.tp_rank // self.tp_size
        print("tp_group_idx: ", tp_group_idx, "tp_size: ", self.tp_size)
        self.tp_group = dist.new_group(
            ranks=[
                i
                for i in range(
                    tp_group_idx *
                    self.tp_size, (tp_group_idx + 1) * self.tp_size
                )
            ]
        )
        # print("tp_group in main: ", self.tp_group)
        print("Updating network operations with NCCL IDs...")
        # print("original unique_nccl_ids: ", self.unique_nccl_ids)
        self.allReduce_o.update(
            self.tp_group, self.tp_rank, self.tp_size, self.unique_nccl_ids[0:5]
        )
        self.allReduce_d.update(
            self.tp_group, self.tp_rank, self.tp_size, self.unique_nccl_ids[5:10]
        )

    def nanobatch_split(self, total_batchsize, decode_batch_size):
        pass

    def post_update_ops(self, input_req_idx: list[int], input_tensor: torch.Tensor, cumsum_input: list[int], decode_batch_size: int) -> None:
        assert self.kv_cache is not None, "KV cache not initialized"
        self.kv_cache.update(
            cumsum_input,
            input_req_idx,
            decode_batch_size,
            use_cuda_graph=(not self.plan_cuda_graph)
            and self.is_cuda_graph_enabled,
        )
        self.global_input.outputs["tokens"].tensor.copy_(input_tensor)
        self.ropeAppend.update(cumsum_input, decode_batch_size)
        self.decAttn.update(cumsum_input)
        self.pfAttn.update(cumsum_input)
