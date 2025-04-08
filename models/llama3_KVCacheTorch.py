import transformers
import os, sys

from utils.prof_marker import prof_marker
sys.path.append("../")
sys.path.append('../pybind/build')
os.environ["HF_HOME"] = "/code/hf"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from operations.operation_base import Operations
from operations.activation.silu import Activation, Activation_Layer
from operations.embedding.embedding import GenEmbedding, GenEmbedding_Layer
from operations.globalOp.globalOp import GlobalInput, GlobalInput_Layer, GlobalOutput, GlobalOutput_Layer
from operations.gemm.gemm import GEMM, GEMM_Layer
from operations.norm.rmsnorm import LayerNorm, LayerNorm_Layer
from operations.sampling.max_sampling import Sampling, Sampling_Layer
from operations.rope.rope import RopeAppend, RopeAppend_Layer
from operations.attention.llamaAttention import DecAttn, DecAttn_Layer, PFAttn, PFAttn_Layer
from operations.virtualOp.copy import Copy
from operations.virtualOp.redist import Redist, RedistMode
from core.weightManager import WeightManager
from core.bufferAllocate import BufferAllocator
from core.executor import Executor
import torch
from kvcache.kv import KVCacheTorch



class Pipeline():
    def __init__(self):
        # Set parameters as instance variables.
        self.num_kv_heads = 8
        self.num_qo_heads = 32
        self.kqv_heads = self.num_qo_heads + 2 * self.num_kv_heads
        self.head_dim = 128
        self.vocab_size = 128256
        self.hidden_dim = 4096
        self.intermediate_dim = 14 * 1024
        self.batch_size = 7
        self.layer = 32
        self.actual_layer_range = [i for i in range(self.layer)]

    def init(self, weight_path):
        self.init_external_data()
        self.init_operations()
        self.init_dependency()
        self.init_executor()
        self.init_set_shape()
        self.init_set_weight(weight_path)

    def init_external_data(self):
        self.kv_cache = KVCacheTorch()

    def init_operations(self):
        self.global_input    = GlobalInput("GlobalInput").first_only()
        self.global_input_devices = self.global_input.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.global_input_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.global_input_layers_per_device.append([GlobalInput_Layer(0, self.global_input_devices[i])])

        self.gen_embedding   = GenEmbedding("GenEmbedding").setWeightName("model.embed_tokens.weight").first_only()
        self.gen_embedding_devices = self.gen_embedding.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.gen_embedding_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.gen_embedding_layers_per_device.append(self.gen_embedding_devices[i].expand_layer([0]))

        self.layerNormAttn   = LayerNorm("LayerNormAttn").setWeightName("model.layers.{layer}.input_layernorm.weight")
        self.layerNormAttn_devices = self.layerNormAttn.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.layerNormAttn_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.layerNormAttn_layers_per_device.append(self.layerNormAttn_devices[i].expand_layer(self.actual_layer_range))

        self.kqv             = GEMM("KQV").setWeightName([
            "model.layers.{layer}.self_attn.k_proj.weight",
            "model.layers.{layer}.self_attn.v_proj.weight",
            "model.layers.{layer}.self_attn.q_proj.weight"
        ])
        self.kqv_devices = self.kqv.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.kqv_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.kqv_layers_per_device.append(self.kqv_devices[i].expand_layer(self.actual_layer_range))


        self.ropeAppend      = RopeAppend("RopeAppend")
        self.ropeAppend.externals["KVCache"] = self.kv_cache
        self.ropeAppend_devices = self.ropeAppend.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.ropeAppend_layers_per_device_per_device = []
        for i in range(torch.cuda.device_count()):
            self.ropeAppend_layers_per_device_per_device.append(self.ropeAppend_devices[i].expand_layer(self.actual_layer_range))

        self.decAttn         = DecAttn("DecAttn")
        self.decAttn.externals["KVCache"] = self.kv_cache
        
        self.decAttn_devices = self.decAttn.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        # print(self.decAttn_devices[0].inputs["Q"])
        self.decAttn_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.decAttn_layers_per_device.append(self.decAttn_devices[i].expand_layer(self.actual_layer_range))

        self.pfAttn          = PFAttn("PFAttn")
        self.pfAttn.externals["KVCache"] = self.kv_cache
        self.pfAttn_devices = self.pfAttn.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        # print(self.pfAttn_devices[0].inputs["Q"])
        self.pfAttn_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.pfAttn_layers_per_device.append(self.pfAttn_devices[i].expand_layer(self.actual_layer_range))

        self.o               = GEMM("O", True).setWeightName("model.layers.{layer}.self_attn.o_proj.weight")
        self.o_devices = self.o.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.o_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.o_layers_per_device.append(self.o_devices[i].expand_layer(self.actual_layer_range))

        self.layerNormFFN    = LayerNorm("LayerNormFFN").setWeightName("model.layers.{layer}.post_attention_layernorm.weight")
        self.layerNormFFN_devices = self.layerNormFFN.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.layerNormFFN_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.layerNormFFN_layers_per_device.append(self.layerNormFFN_devices[i].expand_layer(self.actual_layer_range))

        self.ug              = GEMM("UG").setWeightName([
            "model.layers.{layer}.mlp.up_proj.weight",
            "model.layers.{layer}.mlp.gate_proj.weight"
        ])
        self.ug_devices = self.ug.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.ug_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.ug_layers_per_device.append(self.ug_devices[i].expand_layer(self.actual_layer_range))


        self.activation      = Activation("Activation")
        self.activation_devices = self.activation.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.activation_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.activation_layers_per_device.append(self.activation_devices[i].expand_layer(self.actual_layer_range))

        self.d               = GEMM("D", True).setWeightName("model.layers.{layer}.mlp.down_proj.weight")
        self.d_devices = self.d.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.d_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.d_layers_per_device.append(self.d_devices[i].expand_layer(self.actual_layer_range))

        self.getLogits       = GEMM("GetLogits").setWeightName("lm_head.weight")
        self.getLogits.last_layer_only = True
        self.getLogits_devices = self.getLogits.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.getLogits_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.getLogits_layers_per_device.append([GEMM_Layer(self.actual_layer_range[-1], self.getLogits_devices[i])])

        self.modelLayerNorm  = LayerNorm("ModelLayerNorm").setWeightName("model.norm.weight")
        self.modelLayerNorm.last_layer_only = True
        self.modelLayerNorm_devices = self.modelLayerNorm.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.modelLayerNorm_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.modelLayerNorm_layers_per_device.append([LayerNorm_Layer(self.actual_layer_range[-1], self.modelLayerNorm_devices[i])])

        self.sample          = Sampling("Sampling")
        self.sample.last_layer_only = True
        self.sample_devices = self.sample.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.sample_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.sample_layers_per_device.append([Sampling_Layer(self.actual_layer_range[-1], self.sample_devices[i])])

        self.global_output   = GlobalOutput("GlobalOutput")
        self.global_output.last_layer_only = True
        self.global_output_devices = self.global_output.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.global_output_layers_per_device = []
        for i in range(torch.cuda.device_count()):
            self.global_output_layers_per_device.append([GlobalOutput_Layer(self.actual_layer_range[-1], self.global_output_devices[i])])

        self.copy_o = Copy("CopyO")
        self.copy_o_devices = self.copy_o.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.copy_d = Copy("CopyD")
        self.copy_d_devices = self.copy_d.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.redist_p = Redist("RedistPartition", RedistMode.PARTITION)
        self.redist_p_devices = self.redist_p.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])
        self.redist_a = Redist("RedistAggregation", RedistMode.AGGREGATE)
        self.redist_a_devices = self.redist_a.expand_gpu([torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())])



        self.virtual_operation_list = [self.copy_o, self.copy_d, self.redist_p, self.redist_a]

        # Save operations in an instance variable
        self.operation_list = [
            self.global_input, self.gen_embedding, self.layerNormAttn, self.kqv, self.ropeAppend,
            self.decAttn, self.pfAttn, self.o, self.layerNormFFN, self.ug, self.activation, self.d,
            self.modelLayerNorm, self.getLogits, self.sample, self.global_output
        ]

        self.operation_layers_per_device = []

        for i in range(torch.cuda.device_count()):
            layers = [
                self.global_input_layers_per_device[i],
                self.gen_embedding_layers_per_device[i],
                self.layerNormAttn_layers_per_device[i],
                self.kqv_layers_per_device[i],
                self.ropeAppend_layers_per_device_per_device[i],  # Seems like a typo: consider fixing the name
                self.decAttn_layers_per_device[i],
                self.pfAttn_layers_per_device[i],
                self.o_layers_per_device[i],
                self.layerNormFFN_layers_per_device[i],
                self.ug_layers_per_device[i],
                self.activation_layers_per_device[i],
                self.d_layers_per_device[i],
                self.modelLayerNorm_layers_per_device[i],
                self.getLogits_layers_per_device[i],
                self.sample_layers_per_device[i],
                self.global_output_layers_per_device[i]
            ]
            self.operation_layers_per_device.append(layers)
    
    def init_dependency(self):
        self.global_input.outputs["tokens"] >> self.gen_embedding.inputs["token"]

        self.gen_embedding.outputs["output"] >> self.copy_d.io
        self.copy_d.io  >> self.layerNormAttn.inputs["input"]
        self.copy_d.io.real_deps[self.layerNormAttn.inputs["input"]].append((self.gen_embedding, False))

        self.layerNormAttn.outputs["output"] >> self.kqv.inputs["A"]

        self.kqv.outputs["D"] >> self.ropeAppend.inputs["kqv"]

        self.ropeAppend.outputs["q"] >>  self.redist_p.io
        self.redist_p.io >>  self.decAttn.inputs["Q"]
        self.redist_p.io >> self.pfAttn.inputs["Q"]
        self.redist_p.io.real_deps[self.decAttn.inputs["Q"]].append((self.ropeAppend, False))
        self.redist_p.io.real_deps[self.pfAttn.inputs["Q"]].append((self.ropeAppend, False))
 
        self.decAttn.outputs["output"] >> self.redist_a.io
        self.pfAttn.outputs["output"] >> self.redist_a.io
        self.redist_a.io >> self.o.inputs["A"]
        self.redist_a.io.real_deps[self.o.inputs["A"]].append((self.pfAttn, False))
        self.redist_a.io.real_deps[self.o.inputs["A"]].append((self.decAttn, False))
        
        self.copy_d.io >> self.o.inputs["C"]
        self.copy_d.io.real_deps[self.o.inputs["C"]].append((self.gen_embedding, False))

        self.o.outputs["D"] >> self.copy_o.io
        self.copy_o.io >> self.layerNormFFN.inputs["input"]
        self.copy_o.io.real_deps[self.layerNormFFN.inputs["input"]].append((self.o, False))

        self.layerNormFFN.outputs["output"] >> self.ug.inputs["A"]

        self.ug.outputs["D"] >> self.activation.inputs["input"]

        self.activation.outputs["output"] >> self.d.inputs["A"]

        self.copy_o.io >> self.d.inputs["C"]
        self.copy_o.io.real_deps[self.d.inputs["C"]].append((self.o, False))

        self.d.outputs["D"] >> self.copy_d.io
        self.copy_d.io.chain(self.layerNormAttn.inputs["input"], True)
        self.copy_d.io.real_deps[self.layerNormAttn.inputs["input"]].append((self.d, True))
        self.copy_d.io.chain(self.o.inputs["C"], True)
        self.copy_d.io.real_deps[self.o.inputs["C"]].append((self.d, True))
        self.copy_d.io >> self.modelLayerNorm.inputs["input"]
        self.copy_d.io.real_deps[self.modelLayerNorm.inputs["input"]].append((self.d, False))

        self.modelLayerNorm.outputs["output"] >> self.getLogits.inputs["A"]

        self.getLogits.outputs["D"] >> self.sample.inputs["logits"]

        self.sample.outputs["tokens"] >> self.global_output.inputs["tokens"]
        
        for operation in self.operation_list + self.virtual_operation_list:
            operation.checkConnection()
            
    def init_executor(self):
        self.executor = Executor(self.operation_list, self.operation_layers_per_device[0], self.layer)
        # self.executor.plan_layer_ordering()
        self.executor.plan_layer_ordering_using_operator_layers()
    
    def init_set_shape(self):
        self.gen_embedding.setShape(self.hidden_dim, self.vocab_size)
        self.layerNormAttn.setShape(self.hidden_dim)
        self.kqv.setShape(self.kqv_heads * self.head_dim, self.hidden_dim)
        self.decAttn.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
        self.pfAttn.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
        self.ropeAppend.setShape(self.num_kv_heads, self.num_qo_heads, self.head_dim)
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
        torch.cuda.empty_cache()
    
    def config_batch_size_devices(self, decode_flag, i):
        # print(f"batch_size: {self.batch_size}")
        self.gen_embedding_devices[i].setBatchSize(self.batch_size)
        self.layerNormAttn_devices[i].setBatchSize(self.batch_size)
        self.kqv_devices[i].setBatchSize(self.batch_size)
        self.decAttn_devices[i].setBatchSize(0)
        self.pfAttn_devices[i].setBatchSize(self.batch_size)
        if decode_flag:
            self.decAttn_devices[i].setBatchSize(self.batch_size)
            self.pfAttn_devices[i].setBatchSize(0)
        self.ropeAppend_devices[i].setBatchSize(self.batch_size)
        # print(self.decAttn_devices[i].inputs["Q"].shape)
        self.layerNormFFN_devices[i].setBatchSize(self.batch_size)
        self.ug_devices[i].setBatchSize(self.batch_size)
        self.activation_devices[i].setBatchSize(self.batch_size)
        self.o_devices[i].setBatchSize(self.batch_size)
        self.d_devices[i].setBatchSize(self.batch_size)
        self.modelLayerNorm_devices[i].setBatchSize(self.batch_size)
        self.getLogits_devices[i].setBatchSize(self.batch_size)
        self.sample_devices[i].setBatchSize(self.batch_size)
        self.global_input_devices[i].setBatchSize(self.batch_size)
        self.global_output_devices[i].setBatchSize(self.batch_size)
        self.copy_o_devices[i].setBatchSize(self.o_devices[i].outputs["D"])
        self.copy_d_devices[i].setBatchSize(self.d_devices[i].outputs["D"])
        self.redist_p_devices[i].setBatchSize(self.ropeAppend_devices[i].outputs["q"])
        self.redist_a_devices[i].setBatchSize(self.o_devices[i].inputs["A"])
    
    def config_algorithm(self):
        self.gen_embedding.config_tag("torch")
        self.layerNormAttn.config_tag("torch")
        self.activation.config_tag("torch")
        self.kqv.config_tag("torch", {"M" : self.batch_size, "N": self.kqv_heads * self.head_dim, "K": self.hidden_dim, "alpha": 1.0, "bias": False})
        self.ropeAppend.config_tag("torch")
        self.decAttn.config_tag("torch")
        self.pfAttn.config_tag("torch")
        self.layerNormFFN.config_tag("torch")

        # self.ug.config_tag("cuda:SM90_128_256_64_2_1_1_1_RowMajor_RowMajor_RowMajor_auto", {"M" : self.batch_size, "N": self.intermediate_dim * 2, "K": self.hidden_dim, "alpha": 1.0, "bias": False})
        # self.d.config_tag("cuda:SM90_128_256_64_2_1_1_1_RowMajor_RowMajor_RowMajor_auto", {"M" : self.batch_size, "N": self.hidden_dim, "K": self.intermediate_dim, "alpha": 1.0, "bias": True, "beta": 1.0})
        # self.getLogits.config_tag("cuda:SM90_128_256_64_2_1_1_1_RowMajor_RowMajor_RowMajor_auto", {"M" : self.batch_size, "N": self.vocab_size, "K": self.hidden_dim, "alpha": 1.0, "bias": False})
        # self.kqv.config_tag("torch", {"alpha": 1.0, "bias" : False})
        self.o.config_tag("torch", {"alpha": 1.0, "bias" : True, "beta": 1.0})
        self.ug.config_tag("torch", {"alpha": 1.0, "bias" : False})
        self.d.config_tag("torch", {"alpha": 1.0, "bias" : True, "beta": 1.0})
        self.modelLayerNorm.config_tag("torch")
        self.sample.config_tag("torch")
        self.getLogits.config_tag("torch", {"alpha": 1.0, "bias" : False})

    def config(self, decode_flag=False):
        self.config_batch_size(decode_flag)
        self.config_algorithm()
    
    def update(self, input_ids, decode_flag=False):
        # print("update, " ,self.decAttn_devices[0].inputs["Q"])
        self.input_ids = input_ids
        # concatenate input_ids into a single tensor
        flattened = [item for sublist in input_ids for item in sublist]
        self.batch_size = len(flattened)
        # print(f"batch_size: {self.batch_size}")
        self.config_batch_size_devices(decode_flag, 0)
        # print("after config, " ,self.decAttn_devices[0].inputs["Q"])
        self.update_allocate_buffers()
        self.config_algorithm()
        input_tensor = torch.tensor(flattened, dtype=torch.int32, device='cuda')
        # get cumulative sum of the number of tokens in each input
        request_length = torch.tensor([len(x) for x in input_ids], dtype=torch.int32)
        self.cumsum_input = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(request_length, dim=0)])
        # print(f"cumsum_input: {self.cumsum_input}")
        # print(f"input_tensor: {input_tensor}")
        
        self.global_input.children[0].outputs["tokens"].tensor[:input_tensor.shape[0]].copy_(input_tensor)
        self.ropeAppend.update(0, self.cumsum_input, 0, 0, 0, 0, 0, decode_flag)
        self.decAttn.update(self.cumsum_input, 0, 0, 0, 0, 0, 0, 0)
        self.pfAttn.update(self.cumsum_input, 0, 0, 0, 0, 0, 0, 0)
        
    def update_allocate_buffers(self):
        # Build list of buffers.
        buffers_list = []
        for operation in self.operation_list:
            for _, wrapper in operation.inputs.items():
                buffers_list.append(wrapper)
                # print(f"input of {operation.name}: {wrapper.name}, {wrapper}")
            for _, wrapper in operation.outputs.items():
                buffers_list.append(wrapper)
                # print(f"output of {operation.name}: {wrapper.name}, {wrapper}")
        for operation in self.virtual_operation_list:
            buffers_list.append(operation.io)
        # Allocate buffers.
        bufferAllocator = BufferAllocator(buffers_list)
        bufferAllocator.create_dependency_graph()
        bufferAllocator.allocate_buffer(0)
        print(f"Total allocated: {bufferAllocator.total_allocated / 1024 / 1024} MB")

    def profile(self):
        for operation in self.operation_list:
            operation.profile()

    def search_profile_data(self):
        operation_base = Operations()
        operation_base.search_profile_data()

    def run(self):
        with prof_marker("initialize_executor"):
            # executor = Executor(self.operation_list, self.layer)
            # executor.plan_layer_ordering()
            # executor.draw_ordered_graph()
            # print(executor.ordered_operations)

            temp_out = torch.zeros(self.batch_size, dtype=torch.int32, device='cuda')
            # os.makedirs("./llama3-kv-out-gt", exist_ok=True)


        self.executor.execute_using_operator_layers({}, temp_out)

        # self.executor.print_debug_using_operator_layers("out-gt", filefolder_name="llama3-kv-out-gt", output=temp_out)

        with prof_marker("after_execute_before_return"):
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