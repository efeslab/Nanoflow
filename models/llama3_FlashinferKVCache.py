import torch
import os

from operations.operation_base import Operations
from operations.activation.silu import Activation
from operations.embedding.embedding import GenEmbedding
from operations.globalOp.globalOp import GlobalInput, GlobalOutput
from operations.gemm.gemm_N_parallel import GEMM_N_Parallel
from operations.norm.rmsnorm import LayerNorm
from operations.sampling.max_sampling import Sampling
from operations.rope.rope_flashinfer import RopeAppendFlashinfer
from operations.attention.llamaAttention_flashinfer import DecAttnFlashinfer, PFAttnFlashinfer
from operations.virtualOp.virtual_ops import Copy, Redist
from kvcache.kv import KVCacheNone, DistKVPool, BatchedDistKVCache
from core.weightManager import WeightManager
from core.bufferAllocate import BufferAllocator
from core.executor import Executor
from core.nanobatchSplit import split_nanobatch
from utils.prof_marker import prof_marker



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
        self.num_layers = 32
        self.layer_list = [i for i in range(self.num_layers)]
        self.page_size = 80
        self.device = "cuda:0"

    def set_device(self, device):
        self.device = device

    def init(self, weight_path, cached=False):
        self.init_streams()
        self.init_external_data()
        self.init_operations()
        self.init_dependency()
        self.init_set_shape()
        self.init_set_weight(weight_path, cached)
        self.config_streams()

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

    def init_external_data(self):
        self.kv_pool = DistKVPool(self.num_layers, self.num_kv_heads, self.head_dim, 2048* 2, self.page_size, 1, self.device)
        self.kv_cache = BatchedDistKVCache(self.kv_pool)

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

        self.ropeAppend      = RopeAppendFlashinfer("RopeAppend", self.device)
        self.ropeAppend.externals["KVCache"] = self.kv_cache
        self.ropeAppend_layers = self.ropeAppend.expand_layer(self.layer_list)


        self.decAttn         = DecAttnFlashinfer("DecAttn", self.device)
        self.decAttn.externals["KVCache"] = self.kv_cache
        self.decAttn_layers = self.decAttn.expand_layer(self.layer_list)

        self.pfAttn          = PFAttnFlashinfer("PFAttn", self.device)
        self.pfAttn.externals["KVCache"] = self.kv_cache
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
    
    def init_cached_weight(self, weight_path):
        self.kv_cache = KVCacheNone()
        self.init_operations()
        self.init_set_shape()
        self.init_set_weight(weight_path, False)

    def init_set_weight(self, weight_path, cached):
        weight_manager = WeightManager(self.pipeline_name, weight_path, cached, self.device)
        weight_manager.set_weight(self.operation_list, self.device)

    def clear_batch_size(self):
        # init the batchsize to None
        for op in self.op_for_buffer_allocation:
            op.setBatchSize(None)

    def config_batch_size(self, decode_batchsize):
        self.global_input.setBatchSize(self.batch_size)
        self.decAttn.setBatchSize(decode_batchsize)

    def config_algorithm(self):
        gemm_tag = "cuda:SM90_128_256_64_2_1_1_1_RowMajor_RowMajor_RowMajor_auto"
        self.gen_embedding.config_tag("cuda")
        self.layerNormAttn.config_tag(["cuda", "cuda"])
        # self.layerNormAttn.config_tag("cuda")
        self.activation.config_tag(["cuda", "cuda"])
        # self.activation.config_tag("cuda")
        # self.kqv.config_tag(gemm_tag)
        self.kqv.config_tag([gemm_tag, gemm_tag])
        # self.kqv.config_tag(["cuda:128_128_32_64_64_32_3_5_RowMajor_RowMajor_RowMajor", "cuda:128_128_32_64_64_32_3_5_RowMajor_RowMajor_RowMajor"])
        # self.kqv.config_tag("triton")
        self.ropeAppend.config_tag(["cuda", "cuda"])
        # self.ropeAppend.config_tag("cuda")
        self.decAttn.config_tag("batched_cuda")
        self.pfAttn.config_tag("batched_cuda")
        # self.layerNormFFN.config_tag("cuda")
        self.layerNormFFN.config_tag(["cuda", "cuda"])
        # self.o.config_tag(gemm_tag)
        self.o.config_tag([gemm_tag, gemm_tag])
        # self.o.config_tag(["cuda:128_128_32_64_64_32_1_5_RowMajor_RowMajor_RowMajor", "cuda:128_128_32_64_64_32_2_5_RowMajor_RowMajor_RowMajor"])
        # self.ug.config_tag(gemm_tag)
        self.ug.config_tag([gemm_tag, gemm_tag])
        # self.ug.config_tag(["cuda:128_128_32_64_64_32_1_5_RowMajor_RowMajor_RowMajor", "cuda:128_128_32_64_64_32_2_5_RowMajor_RowMajor_RowMajor"])
        # self.d.config_tag(gemm_tag)
        self.d.config_tag([gemm_tag, gemm_tag])
        # self.d.config_tag(["cuda:128_128_32_64_64_32_1_5_RowMajor_RowMajor_RowMajor", "cuda:128_128_32_64_64_32_2_5_RowMajor_RowMajor_RowMajor"])
        self.modelLayerNorm.config_tag("cuda")
        self.sample.config_tag("cuda")
        self.getLogits.config_tag(gemm_tag)
        # self.getLogits.config_tag("cuda:128_256_32_64_64_32_1_3_RowMajor_RowMajor_RowMajor")

    def config_streams(self):
        self.global_input.set_stream(self.streams["GEMM"])
        self.gen_embedding.set_stream(self.streams["GEMM"])
        self.layerNormAttn.set_stream(self.streams["GEMM"])
        self.activation.set_stream(self.streams["GEMM"])
        self.kqv.set_stream(self.streams["GEMM"])
        self.ropeAppend.set_stream(self.streams["GEMM"])
        self.decAttn.set_stream(self.streams["GEMV"])
        self.pfAttn.set_stream(self.streams["GEMV"])
        self.layerNormFFN.set_stream(self.streams["GEMM"])
        self.o.set_stream(self.streams["GEMM"])
        self.ug.set_stream(self.streams["GEMM"])
        self.d.set_stream(self.streams["GEMM"])
        self.modelLayerNorm.set_stream(self.streams["GEMM"])
        self.sample.set_stream(self.streams["GEMM"])
        self.getLogits.set_stream(self.streams["GEMM"])
        self.global_output.set_stream(self.streams["GEMM"])

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
        extra_links = {
            # TODO: add extra links for virtual ops
            # "KQV0": ("KQV1", False, False),
            # "RopeAppend0": ("RopeAppend1", False, False),
            "RopeAppend0": ("O1", False, False),
            "RopeAppend1": ("O0", False, True),
        }

        new_operation_list, addtional_virtual_ops = split_nanobatch(self.operation_list, op_nanobatch_info_map, extra_links)
        self.op_for_buffer_allocation = []
        self.op_layers = []
        for op in new_operation_list + self.virtual_operation_list + addtional_virtual_ops:
            print("op.name", op.name)
            self.op_for_buffer_allocation.append(op)
        for operation in new_operation_list:
            self.op_layers.extend(operation.children)
    
    def update(self, new_input_infos, decode_batchsize=0):
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
            if len(flattened) != self.batch_size:
                self.batch_size = len(flattened)
                # print(f"batch_size: {self.batch_size}")
                # print("decode_batchsize: ", decode_batchsize)
                self.clear_batch_size()
                self.config_batch_size(decode_batchsize)
                self.nanobatch_split(self.batch_size, decode_batchsize)
                self.update_allocate_buffers()
                # print("finish update_allocate_buffers")
                self.config_algorithm()
                self.init_executor()
        with prof_marker("update_step_3"):
            input_tensor = torch.tensor(flattened, dtype=torch.int32, device=self.device)
            # get cumulative sum of the number of tokens in each input
        with prof_marker("update_step_4"):
            request_length = torch.tensor([len(x) for x in self.input_ids], dtype=torch.int32, device='cpu')
        with prof_marker("update_step_5"):
            self.cumsum_input = torch.cat([torch.tensor([0], dtype=torch.int32, device='cpu'), torch.cumsum(request_length, dim=0, dtype=torch.int32)]).tolist()
        with prof_marker("update_step_6"):
            self.kv_cache.update(self.cumsum_input, self.input_req_idx, decode_batchsize)
        with prof_marker("update_step_7"):
            self.global_input.outputs["tokens"].tensor.copy_(input_tensor)
        with prof_marker("update_step_8"):
            self.ropeAppend.update(self.cumsum_input, decode_batchsize)
        with prof_marker("update_step_9"):
            self.decAttn.update(self.cumsum_input)
        with prof_marker("update_step_10"):
            self.pfAttn.update(self.cumsum_input)
        
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

    def profile(self):
        for operation in self.operation_list:
            operation.profile()

    def search_profile_data(self):
        operation_base = Operations()
        operation_base.search_profile_data()

    def run(self, file_name="./test_data/llama3-8B-flashinfer", filefolder_name="./test_data/llama3-8B-flashinfer_folder"):

        temp_out = torch.zeros(self.batch_size, dtype=torch.int32, device='cuda')

        os.makedirs(f"./{filefolder_name}", exist_ok=True)

        self.executor.execute({}, temp_out)
        # self.executor.print_debug(file_name, filefolder_name=filefolder_name, output=temp_out)

        with prof_marker("after_execute_before_return"):
            temp_out = temp_out.cpu()
        with prof_marker("after_execute_step_1"):
            new_tokens = [ [temp_out[idx-1].item()] for idx in self.cumsum_input[1:]]
        with prof_marker("after_execute_step_2"):
            output = []
        with prof_marker("after_execute_step_3"):
            for req_idx, new_token in zip(self.input_req_idx, new_tokens):
                # print(f"req_idx: {req_idx}, new_token: {new_token}")
                output.append((req_idx, new_token))
        return output

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