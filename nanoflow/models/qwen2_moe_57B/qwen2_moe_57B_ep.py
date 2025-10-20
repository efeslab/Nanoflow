import copy
import json
from pathlib import Path
from typing import Any, Optional
import torch

from nanoflow.operations import NanoOpInfo, Operations, Operation_Layer

from nanoflow.operations import Add, ScaledMul

from nanoflow.operations import (
    GlobalInput,
    GlobalOutput,
    GenEmbedding,
    LayerNorm,
    GEMM_N_Parallel,
    Activation,
    ExpandAdd,
    FusedMoE,
    AllReduce,
    Sampling,
    RopeAppendFlashinfer,
    DecAttnFlashinfer,
    PFAttnFlashinfer,
    Copy,
    Redist,
)


from nanoflow.kvcache.kv import KVCacheNone, DistKVPool, BatchedDistKVCache

from nanoflow.core.basePipeline import BasePipeline
from nanoflow.core import WeightManager, CategoryType
from nanoflow.core.bufferAllocate import BufferAllocator
from nanoflow.core.executor import Executor
from nanoflow.core.nanobatchSplit import split_nanobatch

from nanoflow.utils.green_ctx import split_device_green_ctx_by_sm_count
from nanoflow.utils.prof_marker import prof_marker

from .config_qwen2_moe_57B import Qwen2MoEConfig


class Pipeline(BasePipeline):
    def __init__(self, cfg: Qwen2MoEConfig) -> None:
        # Set parameters as instance variables.
        super().__init__(
            pipeline_name=cfg.pipeline_name,
            cached_weight_dir=cfg.cached_weight_dir,
            profile_dir=cfg.profile_dir,
            num_layers=cfg.num_layers,
            world_size=cfg.world_size,
            world_rank=cfg.world_rank,
            categories=[CategoryType.COMP, CategoryType.MEM],
        )

        self.num_kv_heads = cfg.num_kv_heads
        self.num_qo_heads = cfg.num_qo_heads
        self.head_dim = cfg.head_dim
        self.vocab_size = cfg.vocab_size
        self.hidden_dim = cfg.hidden_dim
        self.intermediate_dim = cfg.intermediate_dim
        self.moe_intermediate_dim = cfg.moe_intermediate_dim
        self.shared_expert_intermediate_dim = cfg.shared_expert_intermediate_dim
        self.num_experts_per_tok = cfg.num_experts_per_tok
        self.num_experts = cfg.num_experts
        self.norm_topk_prob = False
        self.num_shared_experts = cfg.num_shared_experts
        self.rms_norm_eps = cfg.rms_norm_eps
        self.rope_theta = cfg.rope_theta
        self.page_size = cfg.page_size
        self.ep_rank = cfg.ep_rank
        self.ep_size = cfg.ep_size
        self.unique_nccl_ids = cfg.unique_nccl_ids

        # for expert parallel
        self.experts_per_rank = self.num_experts // self.ep_size
        self.exper_start = self.ep_rank * self.experts_per_rank
        self.exper_end = self.exper_start + self.experts_per_rank
        self.experts_range = range(self.exper_start, self.exper_end)

        self.kqv_heads = self.num_qo_heads + 2 * self.num_kv_heads

    def init_external_data(self) -> None:
        print("Initializing external data...")
        # self.kv_pool = DistKVPool(
        #     self.num_layers,
        #     self.num_kv_heads,
        #     self.head_dim,
        #     2048,
        #     self.page_size,
        #     1,
        #     self.device,
        # )
        self.kv_pool = DistKVPool(
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            2048 * 28,
            self.page_size,
            1,
            self.device,
        )

        self.kv_cache = BatchedDistKVCache(self.kv_pool)

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

        self.layerNormAttn = (
            LayerNorm("LayerNormAttn", self.device, eps=self.rms_norm_eps)
            .setWeightName("model.layers.{layer}.input_layernorm.weight")
            .setShape(self.hidden_dim)
        )
        self.layerNormAttn_layers = self.layerNormAttn.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.layerNormAttn)

        self.kqv = (
            GEMM_N_Parallel("KQV", self.device)
            .setWeightName(
                [
                    "model.layers.{layer}.self_attn.k_proj.weight",
                    "model.layers.{layer}.self_attn.v_proj.weight",
                    "model.layers.{layer}.self_attn.q_proj.weight",
                ]
            )
            .setShape(self.kqv_heads * self.head_dim, self.hidden_dim)
            .setParameter(alpha=1.0, beta=0.0)
        )
        self.kqv_layers = self.kqv.expand_layer(self.layer_list)
        self.original_model_operations.append(self.kqv)

        self.kqv_bias = (
            ExpandAdd("KQVBias", self.device)
            .setWeightName(
                [
                    "model.layers.{layer}.self_attn.k_proj.bias",
                    "model.layers.{layer}.self_attn.v_proj.bias",
                    "model.layers.{layer}.self_attn.q_proj.bias",
                ]
            )
            .setShape(self.kqv_heads * self.head_dim)
        )
        self.kqv_bias_layers = self.kqv_bias.expand_layer(self.layer_list)
        self.original_model_operations.append(self.kqv_bias)

        self.ropeAppend = RopeAppendFlashinfer(
            "RopeAppend", self.device, theta=self.rope_theta
        ).setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
        self.ropeAppend.externals["KVCache"] = self.kv_cache
        self.ropeAppend_layers = self.ropeAppend.expand_layer(self.layer_list)
        self.original_model_operations.append(self.ropeAppend)

        self.decAttn = DecAttnFlashinfer("DecAttn", self.device).setShape(
            self.num_kv_heads, self.num_qo_heads, self.head_dim
        )
        self.decAttn.externals["KVCache"] = self.kv_cache
        self.decAttn_layers = self.decAttn.expand_layer(self.layer_list)
        self.original_model_operations.append(self.decAttn)

        self.pfAttn = PFAttnFlashinfer("PFAttn", self.device).setShape(
            self.num_kv_heads, self.num_qo_heads, self.head_dim
        )
        self.pfAttn.externals["KVCache"] = self.kv_cache
        self.pfAttn_layers = self.pfAttn.expand_layer(self.layer_list)
        self.original_model_operations.append(self.pfAttn)

        self.o = (
            GEMM_N_Parallel("O", self.device, bias=True)
            .setWeightName("model.layers.{layer}.self_attn.o_proj.weight")
            .setShape(self.hidden_dim, self.hidden_dim)
            .setParameter(alpha=1.0, beta=1.0)
        )
        self.o_layers = self.o.expand_layer(self.layer_list)
        self.original_model_operations.append(self.o)

        self.layerNormFFN = (
            LayerNorm("LayerNormFFN", self.device, eps=self.rms_norm_eps)
            .setWeightName("model.layers.{layer}.post_attention_layernorm.weight")
            .setShape(self.hidden_dim)
        )
        self.layerNormFFN_layers = self.layerNormFFN.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.layerNormFFN)

        self.gate = (
            GEMM_N_Parallel("Gate", self.device)
            .setWeightName("model.layers.{layer}.mlp.gate.weight")
            .setShape(N=self.num_experts, K=self.hidden_dim)
            .setParameter(alpha=1.0, beta=0.0)
        )
        self.gate_layers = self.gate.expand_layer(self.layer_list)
        self.original_model_operations.append(self.gate)

        self.fused_moe = (
            FusedMoE("FusedMoE", device=self.device)
            .setShape(
                num_experts=self.num_experts,
                moe_intermediate_dim=self.moe_intermediate_dim,
                hidden_dim=self.hidden_dim,
                top_k=self.num_experts_per_tok,
                norm_topk_prob=self.norm_topk_prob,
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
            )
            .setWeightName(
                [
                    [
                        [
                            "model.layers.{layer}"
                            + f".mlp.experts.{expert}.up_proj.weight",
                            "model.layers.{layer}"
                            + f".mlp.experts.{expert}.gate_proj.weight",
                        ]
                        for expert in self.experts_range
                    ],
                    [
                        "model.layers.{layer}"
                        + f".mlp.experts.{expert}.down_proj.weight"
                        for expert in self.experts_range
                    ],
                ]
            )
        )
        self.fused_moe_layers = self.fused_moe.expand_layer(self.layer_list)
        self.original_model_operations.append(self.fused_moe)

        self.allReduce_fused_moe = AllReduce("AllReduceFusedMoE", self.device).setShape(
            self.hidden_dim, rank=self.ep_rank, world_size=self.ep_size)
        self.allReduce_fused_moe_layers = self.allReduce_fused_moe.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.allReduce_fused_moe)

        self.shared_expert_gate = (
            GEMM_N_Parallel("SharedExpertGate", self.device)
            .setWeightName("model.layers.{layer}.mlp.shared_expert_gate.weight")
            .setShape(N=self.num_shared_experts, K=self.hidden_dim)
            .setParameter(alpha=1.0, beta=0.0)
        )
        self.shared_expert_gate_layers = self.shared_expert_gate.expand_layer(
            self.layer_list
        )
        self.original_model_operations.append(self.shared_expert_gate)

        self.shared_expert_activation = Activation(
            "SharedExpertActivation", self.device, act_fn="sigmoid"
        ).setShape(self.num_shared_experts)
        self.shared_expert_activation_layers = (
            self.shared_expert_activation.expand_layer(self.layer_list)
        )
        self.original_model_operations.append(self.shared_expert_activation)

        self.shared_ug = (
            GEMM_N_Parallel("SharedUG", self.device)
            .setWeightName(
                [
                    "model.layers.{layer}.mlp.shared_expert.up_proj.weight",
                    "model.layers.{layer}.mlp.shared_expert.gate_proj.weight",
                ]
            )
            .setShape(N=2 * self.shared_expert_intermediate_dim, K=self.hidden_dim)
            .setParameter(alpha=1.0, beta=0.0)
        )
        self.shared_ug_layers = self.shared_ug.expand_layer(self.layer_list)
        self.original_model_operations.append(self.shared_ug)

        self.shared_activation = Activation(
            "SharedActivation", self.device, act_fn="silu_mul"
        ).setShape(N=self.shared_expert_intermediate_dim)
        self.shared_activation_layers = self.shared_activation.expand_layer(
            self.layer_list
        )
        self.original_model_operations.append(self.shared_activation)

        self.shared_d = (
            GEMM_N_Parallel("SharedD", self.device)
            .setWeightName(
                "model.layers.{layer}.mlp.shared_expert.down_proj.weight",
            )
            .setShape(N=self.hidden_dim, K=self.shared_expert_intermediate_dim)
            .setParameter(alpha=1.0, beta=0.0)
        )
        self.shared_d_layers = self.shared_d.expand_layer(self.layer_list)
        self.original_model_operations.append(self.shared_d)

        self.shared_mul = ScaledMul(
            "SharedMul", self.device).setShape(self.hidden_dim)
        self.shared_mul_layers = self.shared_mul.expand_layer(self.layer_list)
        self.original_model_operations.append(self.shared_mul)

        self.add_experts = Add(
            "AddExperts", self.device).setShape(self.hidden_dim)
        self.add_experts_layers = self.add_experts.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.add_experts)

        self.add_down_bias = Add(
            "AddDownBias", self.device).setShape(self.hidden_dim)
        self.add_down_bias_layers = self.add_down_bias.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.add_down_bias)

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

        self.sample = (
            Sampling("Sampling", self.device).setShape(
                self.vocab_size).last_only()
        )
        self.sample_layers = self.sample.expand_layer(self.layer_list)
        self.original_model_operations.append(self.sample)

        self.global_output = (
            GlobalOutput("GlobalOutput", self.device).setShape().last_only()
        )
        self.global_output_layers = self.global_output.expand_layer(
            self.layer_list)
        self.original_model_operations.append(self.global_output)

        self.copy_embedding = Copy(
            "CopyEmbedding", self.device, num_inputs=2, num_outputs=2
        )
        self.original_virtual_operations.append(self.copy_embedding)

        self.copy_o = Copy("CopyO", self.device, num_inputs=1, num_outputs=2)
        self.original_virtual_operations.append(self.copy_o)

        self.copy_layernormffn = Copy(
            "CopyLayerNormFFN", self.device, num_inputs=1, num_outputs=4
        )
        self.original_virtual_operations.append(self.copy_layernormffn)

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

        self.kqv.outputs["D"] >> self.kqv_bias.inputs["input"]

        self.kqv_bias.outputs["output"] >> self.ropeAppend.inputs["kqv"]

        self.ropeAppend.outputs["q"] >> self.redist_p.inputs["input_0"]
        self.redist_p.outputs["output_0"] >> self.decAttn.inputs["Q"]
        self.redist_p.outputs["output_1"] >> self.pfAttn.inputs["Q"]

        self.decAttn.outputs["output"] >> self.redist_a.inputs["input_0"]
        self.pfAttn.outputs["output"] >> self.redist_a.inputs["input_1"]
        self.redist_a.outputs["output_0"] >> self.o.inputs["A"]

        self.o.outputs["D"] >> self.copy_o.inputs["input_0"]
        self.copy_o.outputs["output_0"] >> self.layerNormFFN.inputs["input"]
        self.copy_o.outputs["output_1"] >> self.add_down_bias.inputs["input_0"]

        self.layerNormFFN.outputs["output"] >> self.copy_layernormffn.inputs["input_0"]
        self.copy_layernormffn.outputs["output_0"] >> self.gate.inputs["A"]
        self.copy_layernormffn.outputs["output_1"] >> self.fused_moe.inputs["x"]
        (
            self.copy_layernormffn.outputs["output_2"]
            >> self.shared_expert_gate.inputs["A"]
        )
        self.copy_layernormffn.outputs["output_3"] >> self.shared_ug.inputs["A"]

        self.gate.outputs["D"] >> self.fused_moe.inputs["router_logits"]

        self.fused_moe.outputs["output"] >> self.allReduce_fused_moe.inputs["input"]
        self.allReduce_fused_moe.outputs["output"] >> self.add_experts.inputs["input_0"]

        (
            self.shared_expert_gate.outputs["D"]
            >> self.shared_expert_activation.inputs["input"]
        )

        self.shared_ug.outputs["D"] >> self.shared_activation.inputs["input"]
        self.shared_activation.outputs["output"] >> self.shared_d.inputs["A"]

        (
            self.shared_expert_activation.outputs["output"]
            >> self.shared_mul.inputs["input_0"]
        )
        self.shared_d.outputs["D"] >> self.shared_mul.inputs["input_1"]

        self.shared_mul.outputs["output"] >> self.add_experts.inputs["input_1"]

        self.add_experts.outputs["output"] >> self.add_down_bias.inputs["input_1"]

        self.add_down_bias.outputs["output"] >> self.copy_d.inputs["input_0"]
        self.copy_d.outputs["output_0"] >> (
            self.copy_embedding.inputs["input_1"], True)
        self.copy_d.outputs["output_1"] >> self.modelLayerNorm.inputs["input"]

        self.modelLayerNorm.outputs["output"] >> self.getLogits.inputs["A"]

        self.getLogits.outputs["D"] >> self.sample.inputs["logits"]

        self.sample.outputs["tokens"] >> self.global_output.inputs["tokens"]

        for operation in self.all_operations:
            operation.checkConnection()

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

        self.gen_embedding.config_tag("cuda", params)

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
            self.layerNormAttn.config_tag("cuda", params)
            self.kqv.config_tag("torch", params)
            self.kqv_bias.config_tag("torch", params)
            self.ropeAppend.config_tag("cuda", params)
            self.decAttn.config_tag("batched_cuda", params)
            self.pfAttn.config_tag("batched_cuda", params)
            self.layerNormFFN.config_tag("cuda", params)
            self.o.config_tag("torch", params)
            self.gate.config_tag("torch", params)
            self.fused_moe.config_tag("cutlass", params)
            self.allReduce_fused_moe.config_tag("nccl", params)
            self.shared_expert_gate.config_tag("torch", params)
            self.shared_expert_activation.config_tag("torch", params)
            self.shared_ug.config_tag("torch", params)
            self.shared_activation.config_tag("cuda", params)
            self.shared_d.config_tag("torch", params)
            self.shared_mul.config_tag("torch", params)
            self.add_experts.config_tag("torch", params)
            self.add_down_bias.config_tag("torch", params)

        self.getLogits.config_tag("torch", params)
        self.modelLayerNorm.config_tag("cuda", params)
        self.sample.config_tag("cuda", params)

    def config_network(self) -> None:
        print("Updating network operations with NCCL IDs...")
        # print("original unique_nccl_ids: ", self.unique_nccl_ids)
        self.allReduce_fused_moe.update(
            None, rank=self.ep_rank, world_size=self.ep_size, unique_nccl_ids=self.unique_nccl_ids[0:5]
        )

    def nanobatch_split(self) -> None:
        op_nanobatch_info_map: dict[str, tuple[NanoOpInfo, ...]] = {}
        extra_links: dict[str, list[tuple[str, bool]]] = {}
        if self.is_auto_search_enabled:
            operations = self.profile_result["operations"]
            for op_basename, op_info in operations.items():
                split_info_list = []
                for nano_op_name, nano_op_info in op_info.items():
                    split_info_list.append(
                        NanoOpInfo(
                            batch_idx=nano_op_info["batch_idx"],
                            batch_size=nano_op_info["batch_size"],
                        )
                    )
                    extra_links[nano_op_name] = nano_op_info["extra_dep"]

                op_nanobatch_info_map[op_basename] = tuple(split_info_list)
        else:
            info = (
                NanoOpInfo(batch_idx=0, batch_size=self.decode_batch_size),
                NanoOpInfo(
                    batch_idx=1,
                    batch_size=self.global_batch_size - self.decode_batch_size,
                ),
            )
            op_nanobatch_info_map = {
                "LayerNormAttn": copy.deepcopy(info),
                "KQV": copy.deepcopy(info),
                "RopeAppend": copy.deepcopy(info),
                "O": copy.deepcopy(info),
                "LayerNormFFN": copy.deepcopy(info),
                "UG": copy.deepcopy(info),
                "Activation": copy.deepcopy(info),
                "D": copy.deepcopy(info),
            }
            extra_links = {}

        print("op_nanobatch_info_map", op_nanobatch_info_map)
        print("extra_links", extra_links)

        model_ops, addtional_virtual_ops = split_nanobatch(
            self.original_model_operations, op_nanobatch_info_map, extra_links
        )
        self.model_operations = model_ops
        self.all_operations = []
        self.all_layer_operations = []
        for op in model_ops + self.virtual_operations + addtional_virtual_ops:
            print("op.name", op.name, op.batch_size)
            self.all_operations.append(op)
        for operation in model_ops:
            self.all_layer_operations.extend(operation.children)

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
