from numpy import isin
import torch
import time
import sqlite3
from utils.prof_marker import prof_marker
import platform_config
from operations.operation_base import Operations, Operation_Layer
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer

from operations.gemm.gemm_impls import GEMMTorchImpl, GEMMCudaImpl

class GEMM_K_Parallel(Operations):
    def __init__(self, name, device, bias = False, nano_idx=None):
        super().__init__(name, device, nano_idx)
        if bias:
            self.inputs = {
                "A": IOWrapper(self, 'A', device).is_input(),
                "C": IOWrapper(self, 'C', device).is_input()
            }
        else:
            self.inputs = {
                "A": IOWrapper(self, 'A', device).is_input()
            }
        self.outputs = {
            "D": IOWrapper(self, 'D', device).is_output()
        }
        self.weights = {
            "B": WeightWrapper()
        }
        self.bias = bias
        self.alpha = 1.0
        if self.bias:
            self.beta = 1.0
        else:
            self.beta = 0.0
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = GEMM_K_Parallel_Layer

    def setParameter(self, alpha = 1.0, beta = 0.0):
        self.alpha = alpha
        self.beta = beta
        if self.bias == False and self.beta != 0:
            raise ValueError("beta should be 0 when bias is not used")
        if self.bias == True and self.beta == 0:
            raise ValueError("beta should not be 0 when bias is used")
        return self

    def init_impl_map(self):
        self.add_impl(GEMMTorchImpl)
        if platform_config.PLATFORM_CUDA:
            from operations.gemm.gemm_impls import GEMMCudaImpl
            self.add_impl(GEMMCudaImpl)
    
    def setShape(self, N, K, tp_idx=0, tp_size=1):
        self.tp_idx = tp_idx
        self.tp_size = tp_size
        # print("tp_idx", self.tp_idx, "tp_size", self.tp_size)
        self.N = N
        self.K = K
        self.tp_N = N
        self.tp_K = K // tp_size
        print("name", self.name, "N:", self.tp_N, "K:", self.tp_K)
        self.weights["B"].shape = (self.tp_K, self.tp_N)
        self.inputs["A"].init_shape((0, self.tp_K))
        if self.bias:
            self.inputs["C"].init_shape((0, self.tp_N)) # bias is a whole buffer, will be used by multiplying a beta.
        self.outputs["D"].init_shape((0, self.tp_N))
        return self
    
    def copy_nano(self, index):
        new_op = GEMM_K_Parallel(self.name, self.device, self.bias, nano_idx=index)
        new_op.weights = self.weights
        new_op.expand_layer(self.layer_list)
        new_op.setShape(self.N, self.K, self.tp_idx, self.tp_size).setParameter(self.alpha, self.beta)
        
        self.nano_ops.append(new_op)

        return new_op

    def init_profile_db(self):
        for _, impl in self.impl_map.items():
            self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS "{impl.category_tag}" (
                id           INTEGER PRIMARY KEY AUTOINCREMENT, 
                batch_size   INTEGER,
                sm_count INTEGER,
                N INTEGER,
                K INTEGER,
                alpha REAL,
                bias BOOLEAN,
                beta REAL,
                average_time_ms REAL,
                GFLOPS REAL,
                impl_tag TEXT,
                UNIQUE (batch_size, sm_count, impl_tag)
            );
            ''')
    
    def store_profile_db(self, category_tag, impl_tag, average_elapsed_ms):
        # Calculate the average time
        print(f"Name: {self.name}, Category: {category_tag}, impl_tag: {impl_tag}, Batch Size: {self.batch_size}, Average Time: {average_elapsed_ms} ms")
        GFLOPS = (2 * self.batch_size * self.tp_N * self.tp_K) / average_elapsed_ms / 1e6 # in GigaFLOPS
        self.cursor.execute(f'''
            INSERT OR IGNORE INTO {category_tag} (batch_size, sm_count, N, K, alpha, bias, beta, average_time_ms, GFLOPS, impl_tag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (self.batch_size, self.sm_count, self.tp_N, self.tp_K, self.alpha, self.bias, self.beta, average_elapsed_ms, GFLOPS, impl_tag))

    def init_impl_configs(self):
        self.impl_configs_map = {}
        for _, impl in self.impl_map.items():
            category_tag = impl.category_tag
            if category_tag == "cuda":
                from pybind.src.generate_gemm.genGEMM import GetAllH100GemmCanonicalNames
                names = GetAllH100GemmCanonicalNames()
                # print(f"GetAllH100GemmCanonicalNames: {names}")
                self.impl_configs_map[category_tag] = [
                    ("SM90_256_128_64_2_1_1_1_RowMajor_RowMajor_RowMajor_auto", None),
                    # ("SM90_256_128_64_2_1_1_1_RowMajor_RowMajor_RowMajor_warpspecialized_cooperative_epi_nosmem", None),
                    # (name, None) for name in names if "RowMajor_RowMajor_RowMajor" in name
                ]
            else:
                self.impl_configs_map[category_tag] = [
                    (None, None)
                ]

    def run(self, layer):
        with prof_marker("GEMM_run"):
            # with prof_marker("Allocate Sliced C"):
            C = self.inputs["C"].tensor if self.bias else torch.empty((self.batch_size, self.N), dtype=torch.float16, device=self.device)
            self.impl.run(self.inputs["A"].tensor, self.weights["B"].weight_map[layer], C, self.outputs["D"].tensor)

    def profile_run(self):
        self.run(self.layer_list[0])

    def processWeight(self, global_weight_map, cached_weight_map, cached, device):
        if not isinstance(self.weight_name, list):
            self.weight_name = [self.weight_name]
        if not cached:
            offset = self.tp_idx % self.tp_size
            for l in self.layer_list:
                weights_list = []
                for name in self.weight_name:
                    stride = global_weight_map[name.format(layer=l)].shape[1] // self.tp_size
                    scope = (slice(None), slice(offset * stride, (offset + 1) * stride))
                    weights_list.append(global_weight_map[name.format(layer=l)][scope].t())
                cached_weight_map[f"{self.name}_layer_{l}"] = torch.cat(weights_list, dim=1).contiguous()
        if cached:
            self.weights["B"].weight_map = {}
            weight_wrapper = self.weights["B"]
            for l in self.layer_list:
                weight_wrapper.weight_map[l] = cached_weight_map[f"{self.name}_layer_{l}"].to(device, non_blocking=True)
                assert weight_wrapper.weight_map[l].shape == weight_wrapper.shape, f"name = {self.weight_name}, expected shape = {weight_wrapper.shape}, layer = {l}, real shape = {weight_wrapper.weight_map[l].shape}"
            
        # torch.cuda.empty_cache()
        # device = torch.cuda.current_device()
        # reserved_memory = torch.cuda.memory_reserved(device)
        # print(f"Reserved memory: {reserved_memory / 1024 / 1024} MB")

        
class GEMM_K_Parallel_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        self.parent.run(self.layer)
