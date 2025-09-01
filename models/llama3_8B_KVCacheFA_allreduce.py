import copy
import torch
import os
import torch.distributed as dist

from operations.allreduce.allreduce import AllReduce
from operations.operation_base import NanoOpInfo, Operations
from operations.activation.silu import Activation
from operations.embedding.embedding import GenEmbedding
from operations.globalOp.globalOp import GlobalInput, GlobalOutput
from operations.gemm.gemm_N_parallel import GEMM_N_Parallel
from operations.gemm.gemm_K_parallel import GEMM_K_Parallel
from operations.norm.rmsnorm import LayerNorm
from operations.rope.rope_fa import RopeAppendBatched
from operations.sampling.max_sampling import Sampling
from operations.rope.rope_flashinfer import RopeAppendFlashinfer
from operations.attention.llamaAttention_flashattn import PFAttnFA
from operations.attention.llamaAttention_vllm import DecPagedAttn
from operations.virtualOp.virtual_ops import Copy, Redist
from kvcache.kv import KVCacheNone, KVCachevLLM
from core.weightManager import WeightManager
from core.bufferAllocate import BufferAllocator
from core.executor import Executor
from core.nanobatchSplit import split_nanobatch
from utils.cu_mask import create_streams_with_cumask
from utils.prof_marker import prof_marker

gemm_stream_with_dc_sm = 256
gemm_stream_with_pf_sm = 48
dc_stream_sm = 288
pf_stream_sm = 16


class Pipeline:
    def __init__(
        self,
        TP_idx: int,
        TP_size: int,
        max_seq_len: int = 1024,
        max_batch_size: int = 768,
    ):
        # Set parameters as instance variables.
        self.pipeline_name = f"Llama3-8B-TP{TP_size}-allreduce"
        self.num_kv_heads = 8
        self.num_qo_heads = 32
        self.kqv_heads = self.num_qo_heads + 2 * self.num_kv_heads
        self.head_dim = 128
        self.vocab_size = 128256
        self.hidden_dim = 4096
        self.intermediate_dim = 14 * 1024
        self.max_batch_size = max_batch_size
        self.batch_size = 0
        self.num_layers = 32
        self.device: str = "cuda:0"
        self.layer_list = [i for i in range(self.num_layers)]
        self.num_devices = torch.cuda.device_count()
        self.page_size = 64
        self.tp_idx = TP_idx
        self.tp_size = TP_size
        self.pp_size = 1
        self.dp_size = 1
        self.max_seq_len = max_seq_len
        self.kv_cache: KVCachevLLM

    def set_device(self, rank: int, device: str):
        self.rank = rank
        self.device = device

    def init(self, weight_path, cached=False):
        self.init_streams()
        self.init_external_data()
        self.init_operations()
        self.init_dependency()
        self.init_set_shape()
        self.init_set_weight(weight_path, cached)
        self.config_network(self.rank)
        self.update_network_ops()

    def init_streams(self):
        self.main_stream = torch.cuda.Stream()
        gemm_stream = torch.cuda.Stream()
        gemm_stream_with_dc, dc_stream = create_streams_with_cumask(
            [gemm_stream_with_dc_sm, gemm_stream_with_pf_sm], self.device
        )
        gemm_stream_with_pf, pf_stream = create_streams_with_cumask(
            [gemm_stream_with_pf_sm, gemm_stream_with_dc_sm], self.device
        )
        network_stream = torch.cuda.Stream()

        # self.streams = {
        #     "GEMM": (gemm_stream, None),
        #     "GEMM_WITH_DC": (gemm_stream_with_dc, None),
        #     "GEMM_WITH_PF": (gemm_stream_with_pf, None),
        #     "DC_ATTN": (dc_stream, None),
        #     "PF_ATTN": (pf_stream, None),
        #     "NETWORK": (network_stream, None),
        # }
        self.streams = {
            "GEMM": (gemm_stream, None),
            "GEMM_WITH_DC": (gemm_stream, None),
            "GEMM_WITH_PF": (gemm_stream, None),
            "DC_ATTN": (gemm_stream, None),
            "PF_ATTN": (gemm_stream, None),
            "NETWORK": (gemm_stream, None),
        }

    def init_external_data(self, for_test=False):
        self.kv_cache = KVCachevLLM(
            num_layers=self.num_layers,
            num_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seqlen=self.max_seq_len,
            max_batch_size=self.max_batch_size,
            device_id=self.rank,
            tp_size=self.tp_size,
        )

    def init_operations(self):
        self.global_input = GlobalInput("GlobalInput", self.device).first_only()
        self.global_input_layers = self.global_input.expand_layer(self.layer_list)

        self.gen_embedding = (
            GenEmbedding("GenEmbedding", self.device)
            .setWeightName("model.embed_tokens.weight")
            .first_only()
        )
        self.gen_embedding_layers = self.gen_embedding.expand_layer(self.layer_list)

        self.layerNormAttn = LayerNorm("LayerNormAttn", self.device).setWeightName(
            "model.layers.{layer}.input_layernorm.weight"
        )
        self.layerNormAttn_layers = self.layerNormAttn.expand_layer(self.layer_list)

        self.kqv = GEMM_N_Parallel("KQV", self.device).setWeightName(
            [
                "model.layers.{layer}.self_attn.k_proj.weight",
                "model.layers.{layer}.self_attn.v_proj.weight",
                "model.layers.{layer}.self_attn.q_proj.weight",
            ]
        )
        self.kqv_layers = self.kqv.expand_layer(self.layer_list)

        self.ropeAppend = RopeAppendBatched("RopeAppend", self.device)
        self.ropeAppend.externals["KVCache"] = self.kv_cache
        self.ropeAppend_layers = self.ropeAppend.expand_layer(self.layer_list)

        self.decAttn = DecPagedAttn("DecAttn", self.device)
        self.decAttn.externals["KVCache"] = self.kv_cache
        self.decAttn_layers = self.decAttn.expand_layer(self.layer_list)

        self.pfAttn = PFAttnFA("PFAttn", self.device)
        self.pfAttn.externals["KVCache"] = self.kv_cache
        self.pfAttn_layers = self.pfAttn.expand_layer(self.layer_list)

        self.o = GEMM_K_Parallel("O", self.device, bias=True).setWeightName(
            "model.layers.{layer}.self_attn.o_proj.weight"
        )
        self.o_layers = self.o.expand_layer(self.layer_list)

        self.allReduce_o = AllReduce("AllReduceO", self.device)
        self.allReduce_o_layers = self.allReduce_o.expand_layer(self.layer_list)

        self.layerNormFFN = LayerNorm("LayerNormFFN", self.device).setWeightName(
            "model.layers.{layer}.post_attention_layernorm.weight"
        )
        self.layerNormFFN_layers = self.layerNormFFN.expand_layer(self.layer_list)

        self.ug = GEMM_N_Parallel("UG", self.device).setWeightName(
            [
                "model.layers.{layer}.mlp.up_proj.weight",
                "model.layers.{layer}.mlp.gate_proj.weight",
            ]
        )
        self.ug_layers = self.ug.expand_layer(self.layer_list)

        self.activation = Activation("Activation", self.device)
        self.activation_layers = self.activation.expand_layer(self.layer_list)

        self.d = GEMM_K_Parallel("D", self.device, bias=True).setWeightName(
            "model.layers.{layer}.mlp.down_proj.weight"
        )
        self.d_layers = self.d.expand_layer(self.layer_list)

        self.allReduce_d = AllReduce("AllReduceD", self.device)
        self.allReduce_d_layers = self.allReduce_d.expand_layer(self.layer_list)

        self.getLogits = (
            GEMM_N_Parallel("GetLogits", self.device)
            .setWeightName("lm_head.weight")
            .last_only()
        )
        self.getLogits_layers = self.getLogits.expand_layer(self.layer_list)

        self.modelLayerNorm = (
            LayerNorm("ModelLayerNorm", self.device)
            .setWeightName("model.norm.weight")
            .last_only()
        )
        self.modelLayerNorm_layers = self.modelLayerNorm.expand_layer(self.layer_list)

        self.sample = Sampling("Sampling", self.device).last_only()
        self.sample_layers = self.sample.expand_layer(self.layer_list)

        self.global_output = GlobalOutput("GlobalOutput", self.device).last_only()
        self.global_output_layers = self.global_output.expand_layer(self.layer_list)

        self.copy_embedding = Copy(
            "CopyEmbedding", self.device, num_inputs=2, num_outputs=2
        )

        self.copy_o = Copy("CopyO", self.device, num_inputs=1, num_outputs=2)

        self.copy_d = Copy("CopyD", self.device, num_inputs=1, num_outputs=2)

        self.redist_p = Redist(
            "RedistPartition", self.device, num_inputs=1, num_outputs=2
        )

        self.redist_a = Redist(
            "RedistAggregation", self.device, num_inputs=2, num_outputs=1
        )

        # Save operations in an instance variable
        self.operation_list = [
            self.global_input,
            self.gen_embedding,
            self.layerNormAttn,
            self.kqv,
            self.ropeAppend,
            self.decAttn,
            self.pfAttn,
            self.o,
            self.allReduce_o,
            self.layerNormFFN,
            self.ug,
            self.activation,
            self.d,
            self.allReduce_d,
            self.modelLayerNorm,
            self.getLogits,
            self.sample,
            self.global_output,
        ]
        self.virtual_operation_list = [
            self.copy_embedding,
            self.copy_o,
            self.copy_d,
            self.redist_p,
            self.redist_a,
        ]

        self.op_for_buffer_allocation = []
        self.op_layers = []
        for op in self.operation_list + self.virtual_operation_list:
            self.op_for_buffer_allocation.append(op)
        for operation in self.operation_list:
            self.op_layers.extend(operation.children)

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
        self.copy_d.outputs["output_0"] >> (self.copy_embedding.inputs["input_1"], True)
        self.copy_d.outputs["output_1"] >> self.modelLayerNorm.inputs["input"]

        self.modelLayerNorm.outputs["output"] >> self.getLogits.inputs["A"]

        self.getLogits.outputs["D"] >> self.sample.inputs["logits"]

        self.sample.outputs["tokens"] >> self.global_output.inputs["tokens"]

        for operation in self.operation_list + self.virtual_operation_list:
            operation.checkConnection()

    def init_executor(self):
        # assert 0 <= device_id < self.num_devices, "device_id should be in range [0, num_devices)"
        self.executor = Executor(self.op_layers, self.layer_list)
        self.executor.plan_layer_ordering()

    def init_set_shape(self):
        self.global_input.setShape()
        self.gen_embedding.setShape(self.hidden_dim, self.vocab_size)
        self.layerNormAttn.setShape(self.hidden_dim)
        self.kqv.setShape(
            self.kqv_heads * self.head_dim,
            self.hidden_dim,
            tp_idx=self.tp_idx,
            tp_size=self.tp_size,
        ).setParameter(1.0, 0.0)
        self.decAttn.setShape(
            self.num_kv_heads, self.num_qo_heads, self.head_dim, tp_size=self.tp_size
        )
        self.pfAttn.setShape(
            self.num_kv_heads, self.num_qo_heads, self.head_dim, tp_size=self.tp_size
        )
        self.ropeAppend.setShape(
            self.num_kv_heads, self.num_qo_heads, self.head_dim, tp_size=self.tp_size
        )
        self.o.setShape(
            self.hidden_dim, self.hidden_dim, tp_idx=self.tp_idx, tp_size=self.tp_size
        ).setParameter(1.0, 1.0 / self.tp_size)
        self.allReduce_o.setShape(
            self.hidden_dim, tp_idx=self.tp_idx, tp_size=self.tp_size
        )
        self.layerNormFFN.setShape(self.hidden_dim)
        self.ug.setShape(
            self.intermediate_dim * 2,
            self.hidden_dim,
            tp_idx=self.tp_idx,
            tp_size=self.tp_size,
        ).setParameter(1.0, 0.0)
        self.activation.setShape(
            self.intermediate_dim, tp_idx=self.tp_idx, tp_size=self.tp_size
        )
        self.d.setShape(
            self.hidden_dim,
            self.intermediate_dim,
            tp_idx=self.tp_idx,
            tp_size=self.tp_size,
        ).setParameter(1.0, 1.0 / self.tp_size)
        self.allReduce_d.setShape(
            self.hidden_dim, tp_idx=self.tp_idx, tp_size=self.tp_size
        )
        self.modelLayerNorm.setShape(self.hidden_dim)
        self.getLogits.setShape(self.vocab_size, self.hidden_dim).setParameter(1.0, 0.0)
        self.sample.setShape(self.vocab_size)
        self.global_output.setShape()

    def init_cached_weight(self, weight_path):
        self.kv_cache = KVCacheNone()
        self.init_operations()
        self.init_set_shape()
        self.init_set_weight(weight_path, False)

    def init_set_weight(self, weight_path, cached):
        weight_manager = WeightManager(
            self.pipeline_name, weight_path, cached, self.device
        )
        weight_manager.set_weight(self.operation_list, self.device)

    def clear_batch_size(self):
        # init the batchsize to None
        for op in self.op_for_buffer_allocation:
            op.setBatchSize(None)

    def config_batch_size(self, decode_batchsize):
        self.global_input.setBatchSize(self.batch_size)
        self.decAttn.setBatchSize(decode_batchsize)

    def config_algorithm(self):
        gemm_tag = "torch"
        self.gen_embedding.config_tag(gemm_tag)
        self.decAttn.config_tag("vllm")
        self.pfAttn.config_tag("flash_attn_batched")
        self.layerNormAttn.config_tag(["torch", "torch"])
        self.activation.config_tag(["torch", "torch"])
        self.kqv.config_tag([gemm_tag, gemm_tag])
        self.ropeAppend.config_tag(["flash_attn_batched", "flash_attn_batched"])
        self.layerNormFFN.config_tag(["torch", "torch"])
        self.o.config_tag([gemm_tag, gemm_tag])
        self.allReduce_o.config_tag("torch")
        self.ug.config_tag([gemm_tag, gemm_tag])
        self.d.config_tag([gemm_tag, gemm_tag])
        self.allReduce_d.config_tag("torch")
        self.modelLayerNorm.config_tag("torch")
        self.sample.config_tag(gemm_tag)
        self.getLogits.config_tag(gemm_tag)

    def config_network(self, rank=0):
        dist.init_process_group(backend="nccl", rank=rank, world_size=self.num_devices)
        tp_group_idx = self.tp_idx // self.tp_size
        print("tp_group_idx: ", tp_group_idx, "tp_size: ", self.tp_size)
        self.tp_group = dist.new_group(
            ranks=[
                i
                for i in range(
                    tp_group_idx * self.tp_size, (tp_group_idx + 1) * self.tp_size
                )
            ]
        )
        # print("tp_group in main: ", self.tp_group)

    def config_streams(self):
        # manually set streams for running.
        self.global_input.set_stream(self.streams["GEMM"])
        self.gen_embedding.set_stream(self.streams["GEMM"])
        self.layerNormAttn.set_stream(
            [self.streams["GEMM_WITH_PF"], self.streams["GEMM_WITH_DC"]]
        )
        self.kqv.set_stream(
            [self.streams["GEMM_WITH_PF"], self.streams["GEMM_WITH_DC"]]
        )
        self.ropeAppend.set_stream(
            [self.streams["GEMM_WITH_PF"], self.streams["GEMM_WITH_DC"]]
        )
        self.decAttn.set_stream(self.streams["DC_ATTN"])
        self.pfAttn.set_stream(self.streams["PF_ATTN"])
        self.o.set_stream([self.streams["GEMM_WITH_PF"], self.streams["GEMM_WITH_DC"]])
        self.allReduce_o.set_stream(self.streams["NETWORK"])
        self.layerNormFFN.set_stream(
            [self.streams["GEMM_WITH_PF"], self.streams["GEMM_WITH_DC"]]
        )
        self.ug.set_stream([self.streams["GEMM_WITH_PF"], self.streams["GEMM_WITH_DC"]])
        self.activation.set_stream(
            [self.streams["GEMM_WITH_PF"], self.streams["GEMM_WITH_DC"]]
        )
        self.d.set_stream([self.streams["GEMM_WITH_PF"], self.streams["GEMM_WITH_DC"]])
        self.allReduce_d.set_stream(self.streams["NETWORK"])
        self.modelLayerNorm.set_stream(self.streams["GEMM"])
        self.sample.set_stream(self.streams["GEMM"])
        self.getLogits.set_stream(self.streams["GEMM"])
        self.global_output.set_stream(self.streams["GEMM"])

    def profile_config_streams(self, stream_tuple):
        for operation in self.operation_list:
            operation.set_stream(stream_tuple)

    def update_network_ops(self):
        self.allReduce_o.update(self.tp_group)
        self.allReduce_d.update(self.tp_group)

    def nanobatch_split(self, total_batchsize, decode_batchsize):
        info = (
            NanoOpInfo(batch_idx=0, batch_size=decode_batchsize),
            NanoOpInfo(batch_idx=1, batch_size=total_batchsize - decode_batchsize),
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

        new_operation_list, addtional_virtual_ops = split_nanobatch(
            self.operation_list, op_nanobatch_info_map, {}
        )
        self.op_for_buffer_allocation = []
        self.new_operation_list = new_operation_list
        self.op_layers = []
        for op in (
            new_operation_list + self.virtual_operation_list + addtional_virtual_ops
        ):
            print("op.name", op.name)
            self.op_for_buffer_allocation.append(op)
        for operation in new_operation_list:
            self.op_layers.extend(operation.children)

    def update(
        self,
        new_input_infos,
        decode_batch_size=0,
        is_profile=False,
        stream_name: str = "GEMM_Test",
        profile_result_path: str = None,
        use_auto_search: bool = False,
        use_nano_split: bool = False,
        use_cuda_graph: bool = False,
    ):
        self.input_req_idx = []
        self.input_ids = []
        with prof_marker("update_step_0"):
            for item in new_input_infos:
                # print("item", item)
                self.input_req_idx.append(item[0])
                self.input_ids.append(item[1])
        with prof_marker("update_step_1"):
            # concatenate input_ids into a single tensor
            flattened = [item for sublist in self.input_ids for item in sublist]
        with prof_marker("update_step_2"):
            if (
                len(flattened) != self.batch_size
                or decode_batch_size != self.decode_batch_size
            ):
                self.batch_size = len(flattened)
                self.decode_batch_size = decode_batch_size
                # print(f"batch_size: {self.batch_size}")
                # print("decode_batchsize: ", decode_batchsize)
                self.clear_batch_size()
                self.config_batch_size(decode_batch_size)
                if use_nano_split:
                    self.nanobatch_split(self.batch_size, decode_batch_size)
                self.update_allocate_buffers()
                # print("finish update_allocate_buffers")
                self.config_streams()
                self.config_algorithm()
                self.init_executor()
        with prof_marker("update_step_3"):
            input_tensor = torch.tensor(
                flattened, dtype=torch.int32, device=self.device
            )
            # get cumulative sum of the number of tokens in each input
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
        with prof_marker("update_step_6"):
            self.kv_cache.update(self.input_req_idx, self.cumsum_input)
        with prof_marker("update_step_7"):
            assert self.global_input.outputs["tokens"].tensor is not None
            self.global_input.outputs["tokens"].tensor.copy_(input_tensor)
        with prof_marker("update_step_8"):
            self.ropeAppend.update(self.cumsum_input, decode_batch_size, self.device)
        with prof_marker("update_step_9"):
            self.decAttn.update(self.cumsum_input, self.device)
        with prof_marker("update_step_10"):
            self.pfAttn.update(self.cumsum_input, self.device)

    def update_allocate_buffers(self):
        # Build list of buffers(op_device)
        buffers_list = []
        for operation in self.op_for_buffer_allocation:
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

    def run(self, file_name="out-operator_layer_test", filefolder_name="llama3-kv-out-rope_test"):
        temp_out = torch.zeros(self.batch_size, dtype=torch.int32, device="cuda")

        self.executor.execute(temp_out, self.main_stream)
        # self.executor.print_debug(temp_out, f"{file_name}_{self.rank}", filefolder_name=f"{filefolder_name}_{self.rank}")

        with prof_marker("after_execute_before_return"):
            temp_out = temp_out.cpu()
        with prof_marker("after_execute_step_1"):
            new_tokens = [[temp_out[idx - 1].item()] for idx in self.cumsum_input[1:]]
        with prof_marker("after_execute_step_2"):
            output = []
        with prof_marker("after_execute_step_3"):
            for req_idx, new_token in zip(self.input_req_idx, new_tokens):
                # print(f"req_idx: {req_idx}, new_token: {new_token}")
                output.append((req_idx, new_token))
        return output

    def terminate(self):
        dist.destroy_process_group()
