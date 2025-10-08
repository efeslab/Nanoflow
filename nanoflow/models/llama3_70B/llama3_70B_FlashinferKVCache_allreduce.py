import copy
import json
from pathlib import Path
from typing import Any, Optional
import torch
import torch.distributed as dist


from nanoflow.operations import NanoOpInfo, Operations, Operation_Layer

from nanoflow.operations import (
    GlobalInput,
    GlobalOutput,
    GenEmbedding,
    LayerNorm,
    GEMM_N_Parallel,
    GEMM_K_Parallel,
    RopeAppendFlashinfer,
    DecAttnFlashinfer,
    PFAttnFlashinfer,
    Activation,
    AllReduce,
    Sampling,
    Copy,
    Redist,
)

from nanoflow.kvcache.kv import KVCacheNone, DistKVPool, BatchedDistKVCache
from nanoflow.core import WeightManager, CategoryType
from nanoflow.core.bufferAllocate import BufferAllocator
from nanoflow.core.executor import Executor
from nanoflow.core.nanobatchSplit import split_nanobatch

from nanoflow.utils.prof_marker import prof_marker
from nanoflow.utils.green_ctx import split_device_green_ctx_by_sm_count

from nanoflow.pybind.build.bind_all_reduce import NCCLWrapper

from .config_llama3_70B import Llama3_70B_Config

class Pipeline:
    pipeline_name_prefix = "Llama3-70B-with-2-allreduce"

    def __init__(
        self,
        cfg: Llama3_70B_Config,
    ):
        # Set parameters as instance variables.
        self.pipeline_name = cfg.pipeline_name
        self.cached_weight_dir = cfg.cached_weight_dir
        self.profile_dir = cfg.profile_dir
        
        self.num_kv_heads = cfg.num_kv_heads
        self.num_qo_heads = cfg.num_qo_heads
        self.head_dim = cfg.head_dim
        self.vocab_size = cfg.vocab_size
        self.hidden_dim = cfg.hidden_dim
        self.intermediate_dim = cfg.intermediate_dim
        self.num_layers = cfg.num_layers
        self.rms_norm_eps = cfg.rms_norm_eps
        self.rope_theta = cfg.rope_theta
        self.page_size = cfg.page_size
        self.tp_size = cfg.tp_size
        self.tp_rank = cfg.tp_rank
        self.pp_size = cfg.pp_size
        self.pp_rank = cfg.pp_rank
        self.dp_size = cfg.dp_size
        self.dp_rank = cfg.dp_rank
        self.kv_cache_type = cfg.kv_cache_type
        self.unique_nccl_ids = cfg.unique_nccl_ids

        self.kqv_heads = self.num_qo_heads + 2 * self.num_kv_heads
        self.global_batch_size: Optional[int] = None
        self.decode_batch_size: Optional[int] = None
        self.layer_list = [i for i in range(self.num_layers)]
        self.num_cuda_devices = torch.cuda.device_count()

        self.profile_result: dict[str, Any] | None = None
        # self.categories = [CategoryType.COMP, CategoryType.MEM, CategoryType.NET]
        self.categories = [CategoryType.COMP, CategoryType.NET]

        self.buffer_fixed: bool = False
        self.is_auto_search_enabled: bool = False
        self.is_cuda_graph_enabled: bool = False
        self.plan_cuda_graph: bool = False

        assert (
            self.pp_size * self.dp_size * self.tp_size == self.num_cuda_devices
        ), f"num_cuda_devices {self.num_cuda_devices} should be equal to pp_size * dp_size * tp_size {self.pp_size * self.dp_size * self.tp_size}"

    def set_device(self, rank, device):
        self.rank = rank
        self.device = device

    def init(self, weight_path, cached=False):
        self.init_streams()
        self.init_external_data()
        self.init_operations()
        self.init_category()
        self.init_dependency()
        self.init_set_weight(weight_path, cached)
        self.config_network(self.rank)
        self.update_network_ops()

    def init_set_weight(self, weight_path, cached):
        weight_manager = WeightManager(
            self.pipeline_name,
            self.cached_weight_dir,
            weight_path,
            cached,
            self.device,
        )
        weight_manager.set_weight(self.model_operations, self.device)

    def init_cached_weight(self, weight_path):
        self.kv_cache = KVCacheNone()
        self.init_operations()
        self.init_set_weight(weight_path, False)

    def init_streams(self):
        self.main_stream = torch.cuda.Stream()
        self.total_sm = 132
        self.sm_counts = [
            # Assuming SM counts are in increments of 8
            i for i in range(8, 128, 8)
        ]
        # [8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120]
        num_sm_counts = len(self.sm_counts)
        self.streams_test = {
            "GEMM_Test": (torch.cuda.Stream(), self.total_sm),
        }

        self.streams: dict[CategoryType,
                           dict[int, tuple[torch._C.Stream, int]]] = {}
        for category in self.categories:
            self.streams[category] = {}

            for i in range((num_sm_counts + 1) // 2):
                sm_count_1 = self.sm_counts[i]
                sm_count_2 = self.sm_counts[num_sm_counts - 1 - i]
                # print(
                #     f"Creating green context streams for SM counts: {sm_count_1}, {sm_count_2}"
                # )

                (stream_1, stream_2, _), _ = split_device_green_ctx_by_sm_count(
                    torch.device(self.device), [sm_count_1, sm_count_2]
                )
                self.streams[category][sm_count_1] = (stream_1, sm_count_1)
                self.streams[category][sm_count_2] = (stream_2, sm_count_2)
            self.streams[category][self.total_sm] = (
                torch.cuda.Stream(), self.total_sm)

        # Create green context streams for testing
        self.profile_streams: dict[str, tuple[torch._C.Stream, int]] = {}
        for i in range((num_sm_counts + 1) // 2):
            sm_count_1 = self.sm_counts[i]
            sm_count_2 = self.sm_counts[num_sm_counts - 1 - i]
            # print(
            #     f"Creating green context streams for SM counts: {sm_count_1}, {sm_count_2}"
            # )

            (stream_1, stream_2, _), _ = split_device_green_ctx_by_sm_count(
                torch.device(self.device), [sm_count_1, sm_count_2]
            )
            self.profile_streams[f"TEST_{i}"] = (stream_1, sm_count_1)
            self.profile_streams[f"TEST_{num_sm_counts - 1 - i}"] = (
                stream_2,
                sm_count_2,
            )
        self.profile_streams[f"TEST_TOTAL"] = (
            torch.cuda.Stream(), self.total_sm)

    def init_external_data(self):
        print("Initializing external data...")
        H100_TP4_num_pages = 2048 * 14
        H200_TP2_num_pages = 2048 * 12
        H200_TP4_num_pages = 2048 * 36
        H200_TP8_num_pages = 2048 * 84
        self.kv_pool = DistKVPool(
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            H200_TP2_num_pages,
            self.page_size,
            self.tp_size,
            self.device,
        )
        self.kv_cache = BatchedDistKVCache(self.kv_pool)

    def reset(self):
        print("Resetting pipeline state...")
        # reset kv cache
        self.kv_cache.reset()

        # reset batch size and decode batch size
        self.global_batch_size = None
        self.decode_batch_size = None

    def init_operations(self):
        self.original_model_operations: list[Operations] = []
        self.original_virtual_operations: list[Operations] = []

        self.global_input = GlobalInput(
            "GlobalInput", self.device).setShape().first_only()
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

        self.ropeAppend = RopeAppendFlashinfer(
            "RopeAppend", self.device, theta=self.rope_theta
        ).setShape(
            self.num_kv_heads, self.num_qo_heads, self.head_dim, tp_size=self.tp_size
        )
        self.ropeAppend.externals["KVCache"] = self.kv_cache
        self.ropeAppend_layers = self.ropeAppend.expand_layer(self.layer_list)
        self.original_model_operations.append(self.ropeAppend)

        self.decAttn = DecAttnFlashinfer("DecAttn", self.device).setShape(
            self.num_kv_heads, self.num_qo_heads, self.head_dim, tp_size=self.tp_size
        )
        self.decAttn.externals["KVCache"] = self.kv_cache
        self.decAttn_layers = self.decAttn.expand_layer(self.layer_list)
        self.original_model_operations.append(self.decAttn)

        self.pfAttn = PFAttnFlashinfer("PFAttn", self.device).setShape(
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
            .setParameter(1.0, 0.0)
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
            # NOTE(Ziren): for further nanosplit or auto search, which should keep the original operations since we need to change the strategy of optimization in the runtime.
        )

        self.all_layer_operations: list[Operation_Layer] = []
        for operation in self.model_operations:
            self.all_layer_operations.extend(operation.children)

    def init_dependency(self):
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
        self.copy_d.outputs["output_1"] >> (
            self.copy_embedding.inputs["input_1"], 1)

        self.modelLayerNorm.outputs["output"] >> self.getLogits.inputs["A"]

        self.getLogits.outputs["D"] >> self.sample.inputs["logits"]

        self.sample.outputs["tokens"] >> self.global_output.inputs["tokens"]

        for operation in self.all_operations:
            operation.checkConnection()

    def init_executor(self):
        print("Initializing executor...")
        self.executor = Executor(self.all_layer_operations, self.layer_list)
        self.executor.plan_layer_ordering()

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

    def clear_batch_size(self):
        # init the batchsize to None
        for op in self.all_operations:
            op.setBatchSize(None)

    def config_batch_size(self):
        print(
            "Configuring batch sizes: global_batch_size =",
            self.global_batch_size,
            ", decode_batch_size =",
            self.decode_batch_size,
        )
        self.global_input.setBatchSize(self.global_batch_size)
        self.decAttn.setBatchSize(self.decode_batch_size)

    def config_algorithm(self):
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
            self.ropeAppend.config_tag("cuda", params)
            self.decAttn.config_tag("batched_cuda", params)
            self.pfAttn.config_tag("batched_cuda", params)
            self.layerNormFFN.config_tag("cuda", params)
            self.o.config_tag("torch", params)
            self.allReduce_o.config_tag("nccl", params)
            self.ug.config_tag("torch", params)
            self.activation.config_tag("cuda", params)
            self.d.config_tag("torch", params)
            self.allReduce_d.config_tag("nccl", params)

        self.getLogits.config_tag("torch", params)
        self.modelLayerNorm.config_tag("cuda", params)
        self.sample.config_tag("cuda", params)

    def config_streams(self):
        print("Configuring streams...")
        for operation in self.original_model_operations:
            operation.set_stream((self.main_stream, self.total_sm))

        if self.is_auto_search_enabled:
            for op in self.model_operations:
                print(
                    f"op.name: {op.name}, op.original_name: {op.original_name}")
                if op.original_name in self.profile_result["operations"]:
                    sm_count = self.profile_result["operations"][op.original_name][
                        op.name
                    ]["p_value"]
                    op.set_stream(self.streams[op.category][sm_count])

    def config_profile_streams(self, stream_tuple):
        for operation in self.model_operations:
            operation.set_stream(stream_tuple)

    def config_network(self, rank):
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=self.num_cuda_devices
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

    def update_network_ops(self):
        print("Updating network operations with NCCL IDs...")
        # print("original unique_nccl_ids: ", self.unique_nccl_ids)
        self.allReduce_o.update(
            self.tp_group, self.rank, self.tp_size, self.unique_nccl_ids[0:5]
        )
        self.allReduce_d.update(
            self.tp_group, self.rank, self.tp_size, self.unique_nccl_ids[5:10]
        )

    def nanobatch_split(self):
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
            raise ValueError("Auto search is not enabled")
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

    def update_allocate_buffers(self):
        print("Allocating buffers...")
        # Build list of buffers(op_device)
        buffers_list = []
        for operation in self.all_operations:
            for _, wrapper in operation.inputs.items():
                buffers_list.append(wrapper)
            for _, wrapper in operation.outputs.items():
                buffers_list.append(wrapper)

        # Allocate buffers for each devices seperatly
        bufferAllocator = BufferAllocator(buffers_list)
        bufferAllocator.create_dependency_graph()
        bufferAllocator.set_all_batchsize_by_linear_programming()

        bufferAllocator.allocate_buffer(self.device)
        print(
            f"Total allocated: {bufferAllocator.total_allocated / 1024 / 1024} MB in {self.device}"
        )

    def update(
        self,
        new_input_infos,
        decode_batch_size=0,
        is_profile=False,
        stream_name: str = "TEST_TOTAL",
        profile_result_path: Optional[str] = None,
        use_auto_search: bool = False,
        use_nano_split: bool = False,
        use_cuda_graph: bool = False,
    ):
        # preprocess new_input_infos
        with prof_marker("update_step_0"):
            self.input_req_idx = []
            self.input_ids = []
            for item in new_input_infos:
                # print("item", item)
                self.input_req_idx.append(item[0])
                self.input_ids.append(item[1])
        with prof_marker("update_step_1"):
            # concatenate input_ids into a single tensor
            flattened = [
                item for sublist in self.input_ids for item in sublist]
            global_batch_size = len(flattened)
        with prof_marker("update_step_3"):
            input_tensor = torch.tensor(
                flattened, dtype=torch.int32, device=self.device
            )

        # some assertions and configuration settings
        if (
            global_batch_size != self.global_batch_size
            or decode_batch_size != self.decode_batch_size
        ):
            self.buffer_fixed = False
        else:
            self.buffer_fixed = True

        self.plan_cuda_graph = False
        if use_cuda_graph and self.is_cuda_graph_enabled:
            assert (
                decode_batch_size == self.decode_batch_size
                and global_batch_size == self.global_batch_size
            ), "decode_batch_size and global_batch_size must be the same when use_cuda_graph is True"
        elif use_cuda_graph and not self.is_cuda_graph_enabled:
            self.plan_cuda_graph = True
        else:
            self.is_cuda_graph_enabled = False
        self.is_cuda_graph_enabled = use_cuda_graph

        self.is_auto_search_enabled = use_auto_search
        if profile_result_path is not None and use_auto_search:
            with open(profile_result_path, "r") as f:
                self.profile_result = json.load(f)
        elif profile_result_path is None and use_auto_search:
            raise ValueError(
                "profile_result_path must be provided when use_auto_search is True"
            )

        # update if batch size or decode batch size has changed
        if not self.buffer_fixed:
            with prof_marker("update_step_2"):
                self.global_batch_size = global_batch_size
                self.decode_batch_size = decode_batch_size
                # print(f"batch_size: {self.batch_size}")
                # print("decode_batch_size: ", decode_batch_size)
                self.clear_batch_size()
                self.config_batch_size()
                if use_nano_split:
                    self.nanobatch_split()
                self.update_allocate_buffers()
                # print("finish update_allocate_buffers")
                if is_profile:
                    self.config_profile_streams(
                        self.profile_streams[stream_name])
                else:
                    self.config_streams()
                self.config_algorithm()
                self.init_executor()
                print("Executor plan_layer_ordering finished")

        with prof_marker("update_step_4"):
            request_length = torch.tensor(
                [len(x) for x in self.input_ids], dtype=torch.int32, device="cpu"
            )
        with prof_marker("update_step_5"):
            self.cumsum_input = torch.cat(
                [
                    torch.tensor([0], dtype=torch.int32, device="cpu"),
                    torch.cumsum(request_length, dim=0, dtype=torch.int32),
                ]
            ).tolist()

        # update in any cases
        with prof_marker("update_step_6"):
            self.kv_cache.update(
                self.cumsum_input,
                self.input_req_idx,
                decode_batch_size,
                use_cuda_graph=(not self.plan_cuda_graph)
                and self.is_cuda_graph_enabled,
            )
        with prof_marker("update_step_7"):
            self.global_input.outputs["tokens"].tensor.copy_(input_tensor)
        with prof_marker("update_step_8"):
            self.ropeAppend.update(self.cumsum_input, decode_batch_size)
        with prof_marker("update_step_9"):
            self.decAttn.update(self.cumsum_input)
        with prof_marker("update_step_10"):
            self.pfAttn.update(self.cumsum_input)
        # print("Update finished")

    def run(self, file_name="out-tp-test", filefolder_name="llama3-kv-out-tp-test"):

        temp_out = torch.zeros(self.global_batch_size,
                               dtype=torch.int32, device="cuda")

        # os.makedirs(f"./{filefolder_name}", exist_ok=True)

        self.executor.execute(
            temp_out,
            self.main_stream,
            plan_cuda_graph=self.plan_cuda_graph,
            is_cuda_graph_enabled=self.is_cuda_graph_enabled,
        )
        # self.executor.print_debug(temp_out, file_name, filefolder_name=filefolder_name)

        with prof_marker("after_execute_before_return"):
            temp_out = temp_out.cpu()
        with prof_marker("after_execute_step_1"):
            new_tokens = [[temp_out[idx - 1].item()]
                          for idx in self.cumsum_input[1:]]
        with prof_marker("after_execute_step_2"):
            output = []
        with prof_marker("after_execute_step_3"):
            for req_idx, new_token in zip(self.input_req_idx, new_tokens):
                # print(f"req_idx: {req_idx}, new_token: {new_token}")
                output.append((req_idx, new_token))
        return output

    def terminate(self):
        print(f"Rank {self.rank}:  Terminating process group...")
        dist.destroy_process_group()
        print(f"Rank {self.rank}:  Process group terminated.")

    # profile related functions
    def init_profile_data(self, append_mode=False):
        is_save_db = True if self.rank == 0 else False
        print(
            "Initializing profile data for pipeline:",
            self.pipeline_name,
            "Append mode:",
            append_mode,
            "Save DB:",
            is_save_db,
        )
        for operation in self.model_operations:
            operation.setup_profile(
                self.profile_dir, append_mode, is_save_db=is_save_db
            )

    def profile_run(self):
        for operation in self.model_operations:
            if operation.batch_size > 0:
                with prof_marker(f"{operation.name}"):
                    print("Operation name:", operation.name)
                    operation.profile_all()
        torch.cuda.synchronize()

    def profile_print(self):
        for operation in self.model_operations:
            operation.print_profile()
