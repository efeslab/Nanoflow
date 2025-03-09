import transformers
import os
os.environ["HF_HOME"] = "/code/hf"
from operations.activation.silu import Activation
from operations.embedding.embedding import GenEmbedding
from operations.globalOp.globalOp import GlobalInput, GlobalOutput
from operations.gemm.gemm import GEMM, NPartitionGEMM, KPartitionGEMM
from operations.norm.rmsnorm import LayerNorm
from operations.sampling.max_sampling import Sampling
from operations.rope.rope import RopeAppend
from operations.attention.llamaAttention import DecAttn, PFAttn
from operations.operation_base import Operations
from kvcache.kvnone import KVCacheNone
from core.weightManager import WeightManager
from core.bufferAllocate import BufferAllocator
from core.executor import Executor
import torch


class Pipeline():
    def __init__(self):
        # Set parameters as instance variables.
        self.num_kv_heads = 8
        self.num_qo_heads = 64
        self.kqv_heads = self.num_qo_heads + 2 * self.num_kv_heads
        self.head_dim = 128
        self.vocab_size = 128256
        self.hidden_dim = 8192
        self.intermediate_dim = 28 * 1024
        self.batch_size = 7
        self.layer = 3
        self.nranks = 8

    def init(self, weight_path):
        self.init_external_data()
        self.init_operations()
        self.init_dependency()
        self.init_set_shape()
        self.init_set_weight(weight_path)

    def init_external_data(self):
        self.kv_cache = KVCacheNone()

    def init_operations(self):
        self.global_input    = GlobalInput("GlobalInput").first_only()

        self.gen_embedding   = GenEmbedding("GenEmbedding").setWeightName("model.embed_tokens.weight").first_only()

        self.layerNormAttn   = LayerNorm("LayerNormAttn").setWeightName("model.layers.{layer}.input_layernorm.weight")

        self.kqv             = NPartitionGEMM("KQV", False).setWeightName([
            "model.layers.{layer}.self_attn.q_proj.weight",
            "model.layers.{layer}.self_attn.k_proj.weight",
            "model.layers.{layer}.self_attn.v_proj.weight"
        ])

        self.ropeAppend      = RopeAppend("RopeAppend")
        self.ropeAppend.externals["KVCache"] = self.kv_cache

        self.decAttn         = DecAttn("DecAttn")
        self.decAttn.externals["KVCache"] = self.kv_cache

        self.pfAttn          = PFAttn("PFAttn")
        self.pfAttn.externals["KVCache"] = self.kv_cache

        self.o               = NPartitionGEMM("O").setWeightName("model.layers.{layer}.self_attn.o_proj.weight")

        self.layerNormFFN    = LayerNorm("LayerNormFFN").setWeightName("model.layers.{layer}.post_attention_layernorm.weight")

        self.ug              = NPartitionGEMM("UG", False).setWeightName([
            "model.layers.{layer}.mlp.up_proj.weight",
            "model.layers.{layer}.mlp.gate_proj.weight"
        ])

        self.activation      = Activation("Activation")

        self.d               = KPartitionGEMM("D").setWeightName("model.layers.{layer}.mlp.down_proj.weight")

        self.getLogits       = GEMM("GetLogits", False).setWeightName("lm_head.weight")
        self.getLogits.last_layer_only = True

        self.modelLayerNorm  = LayerNorm("ModelLayerNorm").setWeightName("model.norm.weight")
        self.modelLayerNorm.last_layer_only = True

        self.sample          = Sampling("Sampling")
        self.sample.last_layer_only = True

        self.global_output   = GlobalOutput("GlobalOutput")
        self.global_output.last_layer_only = True

        # Save operations in an instance variable.
        self.operation_list:list[Operations] = [
            self.global_input, self.gen_embedding, self.layerNormAttn, self.kqv, self.ropeAppend,
            self.decAttn, self.pfAttn, self.o, self.layerNormFFN, self.ug, self.activation, self.d,
            self.modelLayerNorm, self.getLogits, self.sample, self.global_output
        ]
        
        for op in self.operation_list:
            op.set_nranks(self.nranks)
    
    def init_dependency(self):
        self.global_input.outputs["tokens"] >> self.gen_embedding.inputs["token"]

        self.gen_embedding.outputs["output"] >> self.layerNormAttn.inputs["input"]

        self.layerNormAttn.outputs["output"] >> self.kqv.inputs["A"]

        self.kqv.outputs["D"] >> self.ropeAppend.inputs["kqv"]

        self.ropeAppend.outputs["q"] >> self.decAttn.inputs["Q"]
        self.ropeAppend.outputs["q"] >> self.pfAttn.inputs["Q"]

        self.decAttn.outputs["output"] >> self.o.inputs["A"]
        self.pfAttn.outputs["output"] >> self.o.inputs["A"]
        self.gen_embedding.outputs["output"] >> self.o.inputs["C"]

        self.o.outputs["D"] >> self.layerNormFFN.inputs["input"]

        self.layerNormFFN.outputs["output"] >> self.ug.inputs["A"]

        self.ug.outputs["D"] >> self.activation.inputs["input"]

        self.activation.outputs["output"] >> self.d.inputs["A"]

        self.o.outputs["D"] >> self.d.inputs["C"]
        # Additional dependency: d feeds back to layerNormAttn and o.inputs["C"]
        self.d.outputs["D"].chain(self.layerNormAttn.inputs["input"], True)
        self.d.outputs["D"].chain(self.o.inputs["C"], True)
        self.d.outputs["D"] >> self.modelLayerNorm.inputs["input"]

        self.modelLayerNorm.outputs["output"] >> self.getLogits.inputs["A"]

        self.getLogits.outputs["D"] >> self.sample.inputs["logits"]

        self.sample.outputs["tokens"] >> self.global_output.inputs["tokens"]
        
        for operation in self.operation_list:
            operation.checkConnection()
    
    def init_set_shape(self):
        self.gen_embedding.setShape(self.hidden_dim, self.vocab_size)
        self.layerNormAttn.setShape(self.hidden_dim)
        self.kqv.setShape(self.kqv_heads * self.head_dim, self.hidden_dim)
        self.decAttn.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
        self.pfAttn.setShape(self.num_kv_heads // self.nranks, self.num_qo_heads // self.nranks, self.head_dim)
        self.ropeAppend.setShape(self.num_kv_heads // self.nranks, self.num_qo_heads // self.nranks, self.head_dim)
        self.o.setShape(self.hidden_dim, self.hidden_dim)
        self.layerNormFFN.setShape(self.hidden_dim)
        self.ug.setShape(self.intermediate_dim * 2, self.hidden_dim)
        self.d.setShape(self.hidden_dim, self.intermediate_dim)
        self.activation.setShape(self.intermediate_dim)
        self.modelLayerNorm.setShape(self.hidden_dim)
        self.getLogits.setShape(self.vocab_size, self.hidden_dim)
        self.sample.setShape(self.vocab_size)
    
    def init_set_weight(self, weight_path):
        weight_manager = WeightManager()
        weight_manager.load_from_safe_tensor(weight_path)
        weight_manager.set_weight(self.operation_list, self.layer)
    
    def config_batch_size(self):
        self.gen_embedding.setBatchSize(self.batch_size)
        self.layerNormAttn.setBatchSize(self.batch_size)
        self.kqv.setBatchSize(self.batch_size)
        self.decAttn.setBatchSize(0)
        self.pfAttn.setBatchSize(self.batch_size)
        self.ropeAppend.setBatchSize(self.batch_size)
        self.layerNormFFN.setBatchSize(self.batch_size)
        self.ug.setBatchSize(self.batch_size)
        self.activation.setBatchSize(self.batch_size)
        self.o.setBatchSize(self.batch_size)
        self.d.setBatchSize(self.batch_size)
        self.modelLayerNorm.setBatchSize(self.batch_size)
        self.getLogits.setBatchSize(self.batch_size)
        self.sample.setBatchSize(self.batch_size)
        self.global_input.setBatchSize(self.batch_size)
        self.global_output.setBatchSize(self.batch_size)
    
    def config_algorithm(self):
        pass

    def config(self):
        self.config_batch_size()
        self.config_algorithm()
    
    def update(self, input_ids):
        
        self.update_allocate_buffers()
        # concatenate input_ids into a single tensor
        flattened = [item for sublist in input_ids for item in sublist]
        input_tensor = torch.tensor(flattened, dtype=torch.int32, device='cuda')
        # get cumulative sum of the number of tokens in each input
        request_length = torch.tensor([len(x) for x in input_ids], dtype=torch.int32)
        cumsum_input = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(request_length, dim=0)])
        print(f"cumsum_input: {cumsum_input}")
        print(f"input_tensor: {input_tensor}")
        
        self.global_input.outputs["tokens"].tensor[:input_tensor.shape[0]].copy_(input_tensor)
        self.ropeAppend.update(cumsum_input)
        self.decAttn.update(cumsum_input)
        self.pfAttn.update(cumsum_input)
        

    
    def update_allocate_buffers(self):
        # Build list of buffers.
        buffers_list = []
        for operation in self.operation_list:
            for _, wrapper in operation.inputs.items():
                buffers_list.append(wrapper)
            for _, wrapper in operation.outputs.items():
                buffers_list.append(wrapper)
        # Allocate buffers.
        bufferAllocator = BufferAllocator(buffers_list)
        bufferAllocator.allocate_buffer()
        print(f"Total allocated: {bufferAllocator.total_allocated / 1024 / 1024} MB")

    def run(self):
        executor = Executor(self.operation_list, self.layer)
        executor.plan_layer_ordering()
        # executor.draw_ordered_graph()
        print(executor.ordered_operations)
        # executor.execute({})
        executor.print_debug("out.txt")