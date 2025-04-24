import transformers
import torch
import nvtx
import os, sys
sys.path.append("../")
sys.path.append('../pybind/build')
os.environ["HF_HOME"] = "/code/hf"

from operations.operation_base import Operations
from operations.activation.silu import Activation
from operations.embedding.embedding import GenEmbedding
from operations.globalOp.globalOp import GlobalInput, GlobalOutput
from operations.gemm.gemm_N_parallel import GEMM_N_Parallel
from operations.norm.rmsnorm import LayerNorm
from operations.sampling.max_sampling import Sampling
from operations.rope.rope_torch import RopeAppendTorch
from operations.attention.llamaAttention_torch import DecAttnTorch, PFAttnTorch
from operations.virtualOp.virtual_ops import Copy, Redist
from kvcache.kv import KVCacheNone
from core.weightManager import WeightManager
from core.bufferAllocate import BufferAllocator
from core.executor import Executor



class Pipeline():
    def __init__(self):
        # Set parameters as instance variables.
        self.pipeline_name = "Llama3-7B"
        self.num_kv_heads = 8
        self.num_qo_heads = 32
        self.kqv_heads = self.num_qo_heads + 2 * self.num_kv_heads
        self.head_dim = 128
        self.vocab_size = 128256
        self.hidden_dim = 4096
        self.intermediate_dim = 14 * 1024
        self.batch_size = None
        self.num_layers = 32
        self.num_devices = torch.cuda.device_count()
        self.page_size = 64

    def init(self, weight_path, cached=False):
        self.init_external_data()
        self.init_operations()
        self.init_dependency()
        self.init_set_shape()
        self.init_set_weight(weight_path, cached)
        self.init_streams()
        self.config_streams()

    def init_external_data(self):
        self.kv_cache = KVCacheNone()

    def init_operations(self):
        self.global_input    = GlobalInput("GlobalInput").first_only()
        self.global_input_devices, self.global_input_layers_per_device = self.global_input.expand_all_gpu_and_layers(self.num_devices, self.num_layers)

        self.gen_embedding  = GenEmbedding("GenEmbedding").setWeightName("model.embed_tokens.weight").first_only()
        self.gen_embedding_devices, self.gen_embedding_layers_per_device = self.gen_embedding.expand_all_gpu_and_layers(self.num_devices, self.num_layers)

        self.layerNormAttn   = LayerNorm("LayerNormAttn").setWeightName("model.layers.{layer}.input_layernorm.weight")
        self.layerNormAttn_devices, self.layerNormAttn_layers_per_device = self.layerNormAttn.expand_all_gpu_and_layers(self.num_devices, self.num_layers)

        self.kqv             = GEMM_N_Parallel("KQV").setWeightName([
            "model.layers.{layer}.self_attn.k_proj.weight",
            "model.layers.{layer}.self_attn.v_proj.weight",
            "model.layers.{layer}.self_attn.q_proj.weight"
        ])
        self.kqv_devices, self.kqv_layers_per_device = self.kqv.expand_all_gpu_and_layers(self.num_devices, self.num_layers)

        self.ropeAppend      = RopeAppendTorch("RopeAppend")
        self.ropeAppend.externals["KVCache"] = self.kv_cache
        self.ropeAppend_devices, self.ropeAppend_layers_per_device = self.ropeAppend.expand_all_gpu_and_layers(self.num_devices, self.num_layers)


        self.decAttn         = DecAttnTorch("DecAttn")
        self.decAttn.externals["KVCache"] = self.kv_cache
        self.decAttn_devices, self.decAttn_layers_per_device = self.decAttn.expand_all_gpu_and_layers(self.num_devices, self.num_layers)

        self.pfAttn          = PFAttnTorch("PFAttn")
        self.pfAttn.externals["KVCache"] = self.kv_cache
        self.pfAttn_devices, self.pfAttn_layers_per_device = self.pfAttn.expand_all_gpu_and_layers(self.num_devices, self.num_layers)

        self.o               = GEMM_N_Parallel("O", True).setWeightName("model.layers.{layer}.self_attn.o_proj.weight")
        self.o_devices, self.o_layers_per_device = self.o.expand_all_gpu_and_layers(self.num_devices, self.num_layers)

        self.layerNormFFN    = LayerNorm("LayerNormFFN").setWeightName("model.layers.{layer}.post_attention_layernorm.weight")
        self.layerNormFFN_devices, self.layerNormFFN_layers_per_device = self.layerNormFFN.expand_all_gpu_and_layers(self.num_devices, self.num_layers)

        self.ug              = GEMM_N_Parallel("UG").setWeightName([
            "model.layers.{layer}.mlp.up_proj.weight",
            "model.layers.{layer}.mlp.gate_proj.weight"
        ])
        self.ug_devices, self.ug_layers_per_device = self.ug.expand_all_gpu_and_layers(self.num_devices, self.num_layers)

        self.activation      = Activation("Activation")
        self.activation_devices, self.activation_layers_per_device = self.activation.expand_all_gpu_and_layers(self.num_devices, self.num_layers)

        self.d               = GEMM_N_Parallel("D", True).setWeightName("model.layers.{layer}.mlp.down_proj.weight")
        self.d_devices, self.d_layers_per_device = self.d.expand_all_gpu_and_layers(self.num_devices, self.num_layers)


        self.getLogits       = GEMM_N_Parallel("GetLogits").setWeightName("lm_head.weight").last_only()
        self.getLogits_devices, self.getLogits_layers_per_device = self.getLogits.expand_all_gpu_and_layers(self.num_devices, self.num_layers)


        self.modelLayerNorm  = LayerNorm("ModelLayerNorm").setWeightName("model.norm.weight").last_only()
        self.modelLayerNorm_devices, self.modelLayerNorm_layers_per_device = self.modelLayerNorm.expand_all_gpu_and_layers(self.num_devices, self.num_layers)

        self.sample          = Sampling("Sampling").last_only()
        self.sample_devices, self.sample_layers_per_device = self.sample.expand_all_gpu_and_layers(self.num_devices, self.num_layers)

        self.global_output   = GlobalOutput("GlobalOutput").last_only()
        self.global_output_devices, self.global_output_layers_per_device = self.global_output.expand_all_gpu_and_layers(self.num_devices, self.num_layers)

        self.copy_embedding = Copy("CopyEmbedding", num_outputs=3)
        self.copy_embedding_devices = self.copy_embedding.expand_gpu(self.num_devices)

        self.copy_o = Copy("CopyO", num_outputs=2)
        self.copy_o_devices = self.copy_o.expand_gpu(self.num_devices)

        self.copy_d = Copy("CopyD", num_outputs=3)
        self.copy_d_devices = self.copy_d.expand_gpu(self.num_devices)

        self.redist_p = Redist("RedistPartition", num_inputs=1, num_outputs=2)
        self.redist_p_devices = self.redist_p.expand_gpu(self.num_devices)

        self.redist_a = Redist("RedistAggregation", num_inputs=2, num_outputs=1)
        self.redist_a_devices = self.redist_a.expand_gpu(self.num_devices)

        # Save operations in an instance variable
        self.operation_list = [
            self.global_input, self.gen_embedding, self.layerNormAttn, self.kqv, self.ropeAppend,
            self.decAttn, self.pfAttn, self.o, self.layerNormFFN, self.ug, self.activation, self.d,
            self.modelLayerNorm, self.getLogits, self.sample, self.global_output
        ]
        self.virtual_operation_list = [self.copy_embedding, self.copy_o, self.copy_d, self.redist_p, self.redist_a]

        self.operation_device_list = []
        self.operation_layers_per_device = []
        for i in range(self.num_devices):
            op_devices = []
            op_layers = []
            for op in self.operation_list + self.virtual_operation_list:
                op_devices.append(op.children[i])
            for operation in self.operation_list:
                op_layers.extend(operation.op_layers_per_device[i])
            self.operation_device_list.append(op_devices)
            self.operation_layers_per_device.append(op_layers)

    def init_dependency(self):
        self.global_input.outputs["tokens"] >> self.gen_embedding.inputs["token"]

        self.gen_embedding.outputs["output"] >> self.copy_embedding.inputs["input"]
        self.copy_embedding.outputs["output_0"] >> self.layerNormAttn.inputs["input"]
        self.copy_embedding.outputs["output_1"] >> self.o.inputs["C"]
        self.copy_embedding.outputs["output_2"] >> self.d.outputs["D"]

        self.layerNormAttn.outputs["output"] >> self.kqv.inputs["A"]

        self.kqv.outputs["D"] >> self.ropeAppend.inputs["kqv"]

        self.ropeAppend.outputs["q"] >> self.redist_p.inputs["input_0"]
        self.redist_p.outputs["output_0"] >> self.decAttn.inputs["Q"]
        self.redist_p.outputs["output_1"] >> self.pfAttn.inputs["Q"]

        self.decAttn.outputs["output"] >> self.redist_a.inputs["input_0"]
        self.pfAttn.outputs["output"] >> self.redist_a.inputs["input_1"]
        self.redist_a.outputs["output_0"] >> self.o.inputs["A"]

        self.o.outputs["D"] >> self.copy_o.inputs["input"]
        self.copy_o.outputs["output_0"] >> self.layerNormFFN.inputs["input"]
        self.copy_o.outputs["output_1"] >> self.d.inputs["C"]

        self.layerNormFFN.outputs["output"] >> self.ug.inputs["A"]

        self.ug.outputs["D"] >> self.activation.inputs["input"]

        self.activation.outputs["output"] >> self.d.inputs["A"]


        self.d.outputs["D"] >> self.copy_d.inputs["input"]
        self.copy_d.outputs["output_0"] >> (self.layerNormAttn.inputs["input"], True)
        self.copy_d.outputs["output_1"] >> (self.o.inputs["C"], True)
        self.copy_d.outputs["output_2"] >> self.modelLayerNorm.inputs["input"]

        self.modelLayerNorm.outputs["output"] >> self.getLogits.inputs["A"]

        self.getLogits.outputs["D"] >> self.sample.inputs["logits"]

        self.sample.outputs["tokens"] >> self.global_output.inputs["tokens"]
        
        for operation in self.operation_list + self.virtual_operation_list:
            operation.checkConnection()
    
    
    def init_executor(self, device_id=0):
        assert 0 <= device_id < self.num_devices, "device_id should be in range [0, num_devices)"
        self.executor = Executor(self.operation_layers_per_device[device_id], self.num_layers)
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
    
    def init_set_weight(self, weight_path, cached):
        weight_manager = WeightManager(self.pipeline_name, self.num_devices, cached)
        if not cached:
            print("load from safe tensor")
            weight_manager.load_from_safe_tensor(weight_path)
        weight_manager.set_weight(self.operation_list)
        torch.cuda.empty_cache()

    def config_batch_size(self, decode_flag, device_id=0):
        # init the batchsize to None
        for op_device in self.operation_device_list[device_id]:
            op_device.setBatchSize(None)

        self.global_input_devices[device_id].setBatchSize(self.batch_size)
        self.decAttn_devices[device_id].setBatchSize(0)
        if decode_flag:
            self.decAttn_devices[device_id].setBatchSize(self.batch_size)

    def config_algorithm(self, device_id=0):
        self.gen_embedding.config_tag("torch", device_id)
        self.layerNormAttn.config_tag("torch", device_id)
        self.activation.config_tag("torch", device_id)
        self.kqv.config_tag("torch", device_id)
        self.ropeAppend.config_tag("torch", device_id)
        self.decAttn.config_tag("torch", device_id)
        self.pfAttn.config_tag("torch", device_id)
        self.layerNormFFN.config_tag("torch", device_id)
        self.o.config_tag("torch", device_id)
        self.ug.config_tag("torch", device_id)
        self.d.config_tag("torch", device_id)
        self.modelLayerNorm.config_tag("torch", device_id)
        self.sample.config_tag("torch", device_id)
        self.getLogits.config_tag("torch", device_id)

    def init_streams(self):
        GEMM_STREAM = torch.cuda.Stream()
        GEMV_STREAM = torch.cuda.Stream()
        NETWORK_STREAM = torch.cuda.Stream()
        OTHER_STREAM = torch.cuda.Stream()
        self.streams = {
            "GEMM": GEMM_STREAM,
            "GEMV": GEMV_STREAM,
            "NETWORK": NETWORK_STREAM,
            "OTHER": OTHER_STREAM
        }

    def config_streams(self):
        self.global_input.set_stream(self.streams["OTHER"])
        self.gen_embedding.set_stream(self.streams["GEMM"])
        self.layerNormAttn.set_stream(self.streams["GEMM"])
        self.activation.set_stream(self.streams["GEMM"])
        self.kqv.set_stream(self.streams["GEMM"])
        self.ropeAppend.set_stream(self.streams["GEMM"])
        self.decAttn.set_stream(self.streams["GEMV"])
        self.pfAttn.set_stream(self.streams["GEMM"])
        self.layerNormFFN.set_stream(self.streams["GEMM"])
        self.o.set_stream(self.streams["GEMM"])
        self.ug.set_stream(self.streams["GEMM"])
        self.d.set_stream(self.streams["GEMM"])
        self.modelLayerNorm.set_stream(self.streams["GEMM"])
        self.sample.set_stream(self.streams["GEMM"])
        self.getLogits.set_stream(self.streams["GEMM"])
        self.global_output.set_stream(self.streams["OTHER"])
    
    def update(self, input_ids, decode_flag=False, device_id=0):
        
        self.input_ids = input_ids
        # concatenate input_ids into a single tensor
        flattened = [item for sublist in input_ids for item in sublist]
        if len(flattened) != self.batch_size:
            self.batch_size = len(flattened)
            self.config_batch_size(decode_flag, device_id)
            self.update_allocate_buffers(device_id)
            print("finish update_allocate_buffers")
            self.config_algorithm(device_id)
            self.init_executor(device_id)
        input_tensor = torch.tensor(flattened, dtype=torch.int32, device=f'cuda:{device_id}')
        # get cumulative sum of the number of tokens in each input
        request_length = torch.tensor([len(x) for x in input_ids], dtype=torch.int32, device=f'cuda:{device_id}')
        self.cumsum_input = torch.cat([torch.tensor([0], dtype=torch.int32, device=f'cuda:{device_id}'), torch.cumsum(request_length, dim=0, dtype=torch.int32)])

        self.kv_cache.update(self.cumsum_input)

        self.global_input.children[device_id].outputs["tokens"].tensor[:input_tensor.shape[0]].copy_(input_tensor)
        self.ropeAppend.update(self.cumsum_input, decode_flag)
        self.decAttn.update(self.cumsum_input)
        self.pfAttn.update(self.cumsum_input)
        
    def update_allocate_buffers(self, device_id):
        # Build list of buffers(op_device)
        buffers_list = []
        for operation in self.operation_device_list[device_id]:
            for _, wrapper in operation.inputs.items():
                buffers_list.append(wrapper)
            for _, wrapper in operation.outputs.items():
                buffers_list.append(wrapper)

        # Allocate buffers for each devices seperatly
        bufferAllocator = BufferAllocator(buffers_list)
        bufferAllocator.create_dependency_graph()
        bufferAllocator.set_all_batchsize_by_linear_programming()
        
        bufferAllocator.allocate_buffer(device_id)
        print(f"Total allocated: {bufferAllocator.total_allocated / 1024 / 1024} MB in device {device_id}")

    def profile(self):
        for operation in self.operation_list:
            operation.profile()

    def search_profile_data(self):
        operation_base = Operations()
        operation_base.search_profile_data()

    def run(self, rank=0, file_name="out-operator_layer_test", filefolder_name="llama3-kv-out-rope_test"):

        temp_out = torch.zeros(self.batch_size, dtype=torch.int32, device='cuda')

        os.makedirs(f"./{filefolder_name}", exist_ok=True)

        self.executor.execute({}, temp_out)
        # self.executor.print_debug(file_name, rank, filefolder_name=filefolder_name, output=temp_out)

        with nvtx.annotate("after_execute_before_return"):
            temp_out = temp_out.cpu()
            new_tokens = [ [temp_out[idx-1].item()] for idx in self.cumsum_input[1:] ]
        return new_tokens

if __name__ == "__main__":
    # remove the file performance.db
    try:
        os.remove("performance.db")
    except:
        pass
    pipeline = Pipeline()
    pipeline.init_external_data()
    pipeline.init_operations()
    pipeline.init_set_shape()
    pipeline.config_algorithm()
    pipeline.profile()
    pipeline.activation.search_profile_data()