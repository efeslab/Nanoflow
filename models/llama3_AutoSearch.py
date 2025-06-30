import torch

from operations.activation.silu import Activation
from operations.embedding.embedding import GenEmbedding
from operations.globalOp.globalOp import GlobalInput, GlobalOutput
from operations.gemm.gemm_N_parallel import GEMM_N_Parallel
from operations.norm.rmsnorm import LayerNorm
from operations.sampling.max_sampling import Sampling
from operations.rope.rope_torch import RopeAppendTorch
from operations.attention.llamaAttention_torch import DecAttnTorch, PFAttnTorch
from operations.virtualOp.virtual_ops import Copy, Redist
from core.bufferAllocate import BufferAllocator
from core.executor import Executor
from core.nanobatchSplit import split_nanobatch
from utils.green_context import create_greenctx


class Pipeline():
    def __init__(self):
        # Set parameters as instance variables.
        self.pipeline_name = "Llama3-8B"
        self.num_kv_heads = 8
        self.num_qo_heads = 32
        self.kqv_heads = self.num_qo_heads + 2 * self.num_kv_heads
        self.head_dim = 128
        self.vocab_size = 128256
        self.hidden_dim = 4096
        self.intermediate_dim = 14 * 1024
        self.batch_size = None
        self.decode_batch_size = None
        self.num_layers = 32
        self.layer_list = [i for i in range(self.num_layers)]
        self.page_size = 64
        self.device = "cuda:0"

    def init(self):
        self.init_streams()
        self.init_operations()
        self.init_dependency()
        self.init_set_shape()

    def init_streams(self):
        gemm_stream_with_pf, pf_stream, gemm_stream_with_pf_sm, pf_stream_sm = create_greenctx(0.85, 0.15, 0)
        gemm_stream_with_dc, dc_stream, gemm_stream_with_dc_sm, dc_stream_sm = create_greenctx(0.7, 0.3, 0)

        self.streams = {
            "GEMM_Test": (torch.cuda.Stream(), gemm_stream_with_pf_sm + pf_stream_sm),
            "PF_ATTN": (pf_stream, pf_stream_sm),
            "DC_ATTN": (dc_stream, dc_stream_sm),
        }

        # Create green context streams for testing
        test_stream_01, test_stream_09, test_stream_01_sm, test_stream_09_sm = create_greenctx(0.1, 0.9, 0)
        test_stream_02, test_stream_08, test_stream_02_sm, test_stream_08_sm = create_greenctx(0.2, 0.8, 0)
        test_stream_03, test_stream_07, test_stream_03_sm, test_stream_07_sm = create_greenctx(0.3, 0.7, 0)
        test_stream_04, test_stream_06, test_stream_04_sm, test_stream_06_sm = create_greenctx(0.4, 0.6, 0)
        test_stream_05, _, test_stream_05_sm, _ = create_greenctx(0.5, 0.5, 0)

        self.sm_counts = [test_stream_01_sm, test_stream_02_sm, test_stream_03_sm, test_stream_04_sm, test_stream_05_sm,
                    test_stream_06_sm, test_stream_07_sm, test_stream_08_sm, test_stream_09_sm,
                    gemm_stream_with_pf_sm + pf_stream_sm]

    def init_operations(self):
        self.global_input    = GlobalInput("GlobalInput", self.device).first_only()
        self.global_input_layers = self.global_input.expand_layer(self.layer_list)

        self.gen_embedding  = GenEmbedding("GenEmbedding", self.device).setWeightName("model.embed_tokens.weight").first_only()
        self.gen_embedding_layers = self.gen_embedding.expand_layer(self.layer_list)

        self.layerNormAttn   = LayerNorm("LayerNormAttn", self.device).setWeightName("model.layers.{layer}.input_layernorm.weight")
        self.layerNormAttn_layers = self.layerNormAttn.expand_layer(self.layer_list)

        self.kqv             = GEMM_N_Parallel("KQV", self.device).setWeightName([
            "model.layers.{layer}.self_attn.k_proj.weight",
            "model.layers.{layer}.self_attn.v_proj.weight",
            "model.layers.{layer}.self_attn.q_proj.weight"
        ])
        self.kqv_layers = self.kqv.expand_layer(self.layer_list)

        self.ropeAppend      = RopeAppendTorch("RopeAppend", self.device)
        self.ropeAppend_layers = self.ropeAppend.expand_layer(self.layer_list)


        self.decAttn         = DecAttnTorch("DecAttn", self.device)
        self.decAttn_layers = self.decAttn.expand_layer(self.layer_list)

        self.pfAttn          = PFAttnTorch("PFAttn", self.device)
        self.pfAttn_layers = self.pfAttn.expand_layer(self.layer_list)

        self.o               = GEMM_N_Parallel("O", self.device, bias=True).setWeightName("model.layers.{layer}.self_attn.o_proj.weight")
        self.o_layers = self.o.expand_layer(self.layer_list)

        self.layerNormFFN    = LayerNorm("LayerNormFFN", self.device).setWeightName("model.layers.{layer}.post_attention_layernorm.weight")
        self.layerNormFFN_layers = self.layerNormFFN.expand_layer(self.layer_list)

        self.ug              = GEMM_N_Parallel("UG", self.device).setWeightName([
            "model.layers.{layer}.mlp.up_proj.weight",
            "model.layers.{layer}.mlp.gate_proj.weight"
        ])
        self.ug_layers = self.ug.expand_layer(self.layer_list)

        self.activation      = Activation("Activation", self.device)
        self.activation_layers = self.activation.expand_layer(self.layer_list)

        self.d               = GEMM_N_Parallel("D", self.device, bias=True).setWeightName("model.layers.{layer}.mlp.down_proj.weight")
        self.d_layers = self.d.expand_layer(self.layer_list)


        self.getLogits       = GEMM_N_Parallel("GetLogits", self.device).setWeightName("lm_head.weight").last_only()
        self.getLogits_layers = self.getLogits.expand_layer(self.layer_list)


        self.modelLayerNorm  = LayerNorm("ModelLayerNorm", self.device).setWeightName("model.norm.weight").last_only()
        self.modelLayerNorm_layers = self.modelLayerNorm.expand_layer(self.layer_list)

        self.sample          = Sampling("Sampling", self.device).last_only()
        self.sample_layers = self.sample.expand_layer(self.layer_list)

        self.global_output   = GlobalOutput("GlobalOutput", self.device).last_only()
        self.global_output_layers = self.global_output.expand_layer(self.layer_list)

        self.copy_embedding = Copy("CopyEmbedding", self.device, num_inputs=2, num_outputs=2)

        self.copy_o = Copy("CopyO", self.device, num_inputs=1, num_outputs=2)

        self.copy_d = Copy("CopyD", self.device, num_inputs=1, num_outputs=2)

        self.redist_p = Redist("RedistPartition", self.device, num_inputs=1, num_outputs=2)

        self.redist_a = Redist("RedistAggregation", self.device, num_inputs=2, num_outputs=1)

        # Save operations in an instance variable
        self.operation_list = [
            self.global_input, self.gen_embedding, self.layerNormAttn, self.kqv, self.ropeAppend,
            self.decAttn, self.pfAttn, self.o, self.layerNormFFN, self.ug, self.activation, self.d,
            self.modelLayerNorm, self.getLogits, self.sample, self.global_output
        ]
        self.virtual_operation_list = [self.copy_embedding, self.copy_o, self.copy_d, self.redist_p, self.redist_a]

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

        self.o.outputs["D"] >> self.copy_o.inputs["input_0"]
        self.copy_o.outputs["output_0"] >> self.layerNormFFN.inputs["input"]
        self.copy_o.outputs["output_1"] >> self.d.inputs["C"]

        self.layerNormFFN.outputs["output"] >> self.ug.inputs["A"]

        self.ug.outputs["D"] >> self.activation.inputs["input"]

        self.activation.outputs["output"] >> self.d.inputs["A"]


        self.d.outputs["D"] >> self.copy_d.inputs["input_0"]
        self.copy_d.outputs["output_0"] >> (self.copy_embedding.inputs["input_1"], True)
        self.copy_d.outputs["output_1"] >> self.modelLayerNorm.inputs["input"]

        self.modelLayerNorm.outputs["output"] >> self.getLogits.inputs["A"]

        self.getLogits.outputs["D"] >> self.sample.inputs["logits"]

        self.sample.outputs["tokens"] >> self.global_output.inputs["tokens"]
        
        for operation in self.operation_list + self.virtual_operation_list:
            operation.checkConnection()
    
    def init_executor(self):
        # assert 0 <= device_id < self.num_cuda_devices, "device_id should be in range [0, num_devices)"
        self.executor = Executor(self.op_layers, self.layer_list)
        self.executor.plan_layer_ordering()

    def init_set_shape(self):
        self.global_input.setShape()
        self.gen_embedding.setShape(self.hidden_dim, self.vocab_size)
        self.layerNormAttn.setShape(self.hidden_dim)
        self.kqv.setShape(self.kqv_heads * self.head_dim, self.hidden_dim).setParameter(1.0, 0.0)
        self.decAttn.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
        self.pfAttn.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
        self.ropeAppend.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
        self.o.setShape(self.hidden_dim, self.hidden_dim).setParameter(1.0, 1.0)
        self.layerNormFFN.setShape(self.hidden_dim)
        self.ug.setShape(self.intermediate_dim * 2, self.hidden_dim).setParameter(1.0, 0.0)
        self.d.setShape(self.hidden_dim, self.intermediate_dim).setParameter(1.0, 1.0)
        self.activation.setShape(self.intermediate_dim)
        self.modelLayerNorm.setShape(self.hidden_dim)
        self.getLogits.setShape(self.vocab_size, self.hidden_dim).setParameter(1.0, 0.0)
        self.sample.setShape(self.vocab_size)
        self.global_output.setShape()

    def config_batch_size(self, decode_batchsize):
        self.global_input.setBatchSize(self.batch_size)
        self.decAttn.setBatchSize(decode_batchsize)

    def config_category(self):
        self.global_input.set_category("GEMM")
        self.gen_embedding.set_category("GEMM")
        self.layerNormAttn.set_category("GEMM")
        self.kqv.set_category("GEMM")
        self.ropeAppend.set_category("GEMM")
        self.decAttn.set_category("GEMV")
        self.pfAttn.set_category("GEMV")
        self.layerNormFFN.set_category("GEMM")
        self.o.set_category("GEMM")
        self.ug.set_category("GEMM")
        self.activation.set_category("GEMM")
        self.d.set_category("GEMM")
        self.modelLayerNorm.set_category("GEMM")
        self.sample.set_category("GEMM")
        self.getLogits.set_category("GEMM")

    def config_streams(self):
        # Set stream for auto-search case
        self.global_input.set_stream(self.streams["GEMM_Test"])
        self.gen_embedding.set_stream(self.streams["GEMM_Test"])
        self.layerNormAttn.set_stream([self.streams["GEMM_Test"], self.streams["GEMM_Test"]])
        self.kqv.set_stream([self.streams["GEMM_Test"], self.streams["GEMM_Test"]])
        self.ropeAppend.set_stream([self.streams["GEMM_Test"], self.streams["GEMM_Test"]])
        self.decAttn.set_stream(self.streams["DC_ATTN"])
        self.pfAttn.set_stream(self.streams["DC_ATTN"])
        self.layerNormFFN.set_stream([self.streams["GEMM_Test"], self.streams["GEMM_Test"]])
        self.o.set_stream([self.streams["GEMM_Test"], self.streams["GEMM_Test"]])
        self.ug.set_stream([self.streams["GEMM_Test"], self.streams["GEMM_Test"]])
        self.activation.set_stream([self.streams["GEMM_Test"], self.streams["GEMM_Test"]])
        self.d.set_stream([self.streams["GEMM_Test"], self.streams["GEMM_Test"]])
        self.modelLayerNorm.set_stream(self.streams["GEMM_Test"])
        self.sample.set_stream(self.streams["GEMM_Test"])
        self.getLogits.set_stream(self.streams["GEMM_Test"])
        self.global_output.set_stream(self.streams["GEMM_Test"])

    def nanobatch_split(self, total_batchsize, decode_batchsize):
        op_nanobatch_info_map = {
            "LayerNormAttn": (2, (decode_batchsize, total_batchsize - decode_batchsize)),
            "KQV": (2, (decode_batchsize, total_batchsize - decode_batchsize)),
            "RopeAppend": (2, (decode_batchsize, total_batchsize - decode_batchsize)),
            "O": (2, (decode_batchsize, total_batchsize - decode_batchsize)),
            "LayerNormFFN": (2, (decode_batchsize, total_batchsize - decode_batchsize)),
            "UG": (2, (decode_batchsize, total_batchsize - decode_batchsize)),
            "Activation": (2, (decode_batchsize, total_batchsize - decode_batchsize)),
            "D": (2, (decode_batchsize, total_batchsize - decode_batchsize)),
        }
        extra_links = {}

        new_operation_list, addtional_virtual_ops = split_nanobatch(self.operation_list, op_nanobatch_info_map, extra_links)
        self.op_for_buffer_allocation = []
        self.new_operation_list = new_operation_list
        self.op_layers = []
        for op in new_operation_list + self.virtual_operation_list + addtional_virtual_ops:
            print("op.name", op.name)
            self.op_for_buffer_allocation.append(op)
        for operation in new_operation_list:
            self.op_layers.extend(operation.children)
        
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
        print(f"Total allocated: {bufferAllocator.total_allocated / 1024 / 1024} MB in {self.device}")