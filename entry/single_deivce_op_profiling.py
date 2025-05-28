import os, sys
sys.path.append("../")
sys.path.append("../pybind/build")

from operations.operation_base import Operations
from operations.activation.silu import Activation
from operations.embedding.embedding import GenEmbedding
from operations.globalOp.globalOp import GlobalInput, GlobalOutput
from operations.gemm.gemm_N_parallel import GEMM_N_Parallel
from operations.norm.rmsnorm import LayerNorm
from operations.sampling.max_sampling import Sampling
from operations.rope.rope_flashinfer import RopeAppendFlashinfer
from operations.attention.llamaAttention_flashinfer import DecAttnFlashinfer, PFAttnFlashinfer



class Profiling_Operations():
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

    def init_operations(self):
        self.gen_embedding  = GenEmbedding("GenEmbedding", self.device)

        self.layerNorm   = LayerNorm("LayerNorm", self.device)

        self.gemm_n_parallel   = GEMM_N_Parallel("GEMM_N_Parallel", self.device)

        self.ropeAppend      = RopeAppendFlashinfer("RopeAppend", self.device)

        self.decAttn         = DecAttnFlashinfer("DecAttn", self.device)

        self.pfAttn          = PFAttnFlashinfer("PFAttn", self.device)

        self.activation      = Activation("Activation", self.device)

        self.sample          = Sampling("Sampling", self.device)

        # Save operations in an instance variable
        self.operation_list = [
            self.gen_embedding, self.layerNorm, self.gemm_n_parallel, self.ropeAppend,
            self.decAttn, self.pfAttn, self.activation, self.sample
        ]

    def profile(self):
        # self.gen_embedding.profile()
        # self.layerNorm.profile()
        self.gemm_n_parallel.profile()
        # self.ropeAppend.profile()
        # self.decAttn.profile()
        # self.pfAttn.profile()
        # self.activation.profile()
        # self.sample.profile()
    
    def print_profile(self):
        # self.gen_embedding.print_profile()
        # self.layerNorm.print_profile()
        self.gemm_n_parallel.print_profile()
        # self.ropeAppend.print_profile()
        # self.decAttn.print_profile()
        # self.pfAttn.print_profile()
        # self.activation.print_profile()
        # self.sample.print_profile()

os.makedirs("../profiling", exist_ok=True)

profiler = Profiling_Operations()
profiler.init_operations()
profiler.profile()
profiler.print_profile()