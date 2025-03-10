from numpy import isin
import torch
import sys
import time
sys.path.append('../../pybind/build')
from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
import bind_gemm
from operations.impl_base import OperationImpl


# class GEMMNoBias(Operations):
#     def __init__(self, name):
#         super().__init__(name)
#         self.inputs = {
#             "A": IOWrapper(self, 'A', IOBufferType.FULL),
#         }
#         self.outputs = {
#             "D": IOWrapper(self, 'D', IOBufferType.FULL)
#         }
#         self.weights = {
#             "B": WeightWrapper()
#         }
    
#     def setShape(self, N, K):
#         self.N = N
#         self.K = K        
#         self.weights["B"].shape = (self.K, self.N)
    
#     def setBatchSize(self, M):
#         self.M = M
#         self.inputs["A"].shape = (self.M, self.K)
#         self.outputs["D"].shape = (self.M, self.N)

#     def profile(self):
#         rounds = 100
#         batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
#         for batch_size in batch_sizes:
#             total_latency = 0
#             for round in range(rounds):
#                 A = torch.randn((batch_size, self.K), dtype=torch.float16, device='cuda')
#                 B = torch.randn((self.K, self.N), dtype=torch.float16, device='cuda')
#                 C = torch.zeros((batch_size, self.N), dtype=torch.float16, device='cuda')
#                 # record the time
#                 start_time = time.time()
#                 bind_gemm.gemmLauncher(A, B, C, C, batch_size, self.N, self.K, 1, 1)
#                 if round > 0:
#                     latency = time.time() - start_time
#                     total_latency += latency
#             average_time = total_latency / rounds
#             print("name: {}, batch_size: {}, average_time: {}".format(self.name, batch_size, average_time))
#             self.cursor.execute('''
#                 INSERT OR REPLACE INTO performance (id, keyword, batch_size, average_time)
#                 VALUES ((SELECT id FROM performance WHERE keyword = ? AND batch_size = ?), ?, ?, ?)
#             ''', (self.name, batch_size, self.name, batch_size, average_time))
#             self.conn.commit()

        
#     def run(self, layer):
#         A = self.inputs["A"].tensor
#         B = self.weights["B"].weight_map[layer]
#         C = torch.zeros((self.M, self.N), dtype=torch.float16, device=self.inputs["A"].tensor.device)
#         bind_gemm.gemmLauncher(A, B, C, self.outputs["D"].tensor, self.M, self.N, self.K, 1, 1)

#         # self.outputs["D"].tensor.copy_(A.matmul(B))
    
#     def processWeight(self, global_weight_map, total_layers, cached = False):
#         return process_weight_layer(global_weight_map, self.weight_name, self.weights["B"], total_layers, cached)

# class GEMMCombineWeight(GEMMNoBias):
#     def setWeightName(self, name_list):
#         self.weight_name = name_list
#         return self
    
#     def processWeight(self, global_weight_map, total_layers, cached=False):
#         self.weights["B"].weight_map = {}
#         for l in range(total_layers):
#             self.weights["B"].weight_map[l] = torch.cat([global_weight_map[name.format(layer=l)].t() for name in self.weight_name], dim=1).contiguous()
class GEMMTorchImpl(OperationImpl):
    category_tag = "torch"
    def config(self, tag, parameter_map):
        self.alpha = parameter_map["alpha"]
        self.beta = parameter_map["beta"]
    
    def run(self, A, B, C, D):
        D.copy_(A.matmul(B) * self.alpha + C * self.beta)


class GEMM(Operations):
    def __init__(self, name, bias = False):
        super().__init__(name)
        if bias:
            self.inputs = {
                "A": IOWrapper(self, 'A', IOBufferType.FULL),
                "C": IOWrapper(self, 'C', IOBufferType.FULL)
            }
        else:
            self.inputs = {
                "A": IOWrapper(self, 'A', IOBufferType.FULL)
            }
        self.outputs = {
            "D": IOWrapper(self, 'D', IOBufferType.FULL)
        }
        self.weights = {
            "B": WeightWrapper()
        }
        self.bias = bias
        if bias:
            self.beta = 1
        else:
            self.beta = 0
        self.alpha = 1
        
    def setParameter(self, alpha = 1, beta = 0):
        self.alpha = alpha
        self.beta = beta
        if self.bias == False and self.beta != 0:
            raise ValueError("beta should be 0 when bias is not used")
        if self.bias == True and self.beta == 0:
            raise ValueError("beta should not be 0 when bias is used")
        return self
    
    def init_impl_map(self):
        self.add_impl(GEMMTorchImpl)
    
    def setShape(self, N, K):
        self.N = N
        self.K = K
        self.weights["B"].shape = (self.K, self.N)
    
    def setBatchSize(self, M):
        self.M = M
        self.inputs["A"].shape = (self.M, self.K)
        if self.bias:
            self.inputs["C"].shape = (self.M, self.N)
        self.outputs["D"].shape = (self.M, self.N)
        
    def profile(self):
        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        for batch_size in batch_sizes:
            total_latency = 0
            for round in range(rounds):
                A = torch.randn((batch_size, self.K), dtype=torch.float16, device='cuda')
                B = torch.randn((self.K, self.N), dtype=torch.float16, device='cuda')
                C = torch.randn((batch_size, self.N), dtype=torch.float16, device='cuda')
                D = torch.zeros((batch_size, self.N), dtype=torch.float16, device='cuda')
                # record the time
                start_time = time.time()
                bind_gemm.gemmLauncher(A, B, C, D, batch_size, self.N, self.K, 1, 1)
                if round > 0:
                    latency = time.time() - start_time
                    total_latency += latency
            average_time = total_latency / rounds
            print("name: {}, batch_size: {}, average_time: {}".format(self.name, batch_size, average_time))
            self.cursor.execute('''
                INSERT OR REPLACE INTO performance (id, keyword, batch_size, average_time)
                VALUES ((SELECT id FROM performance WHERE keyword = ? AND batch_size = ?), ?, ?, ?)
            ''', (self.name, batch_size, self.name, batch_size, average_time))
            self.conn.commit()
    
    def run(self, layer):
        A = self.inputs["A"].tensor
        if self.bias:
            C = self.inputs["C"].tensor
        else:
            C = torch.empty((self.M, self.N), dtype=torch.float16, device=self.inputs["A"].tensor.device)
        
        B = self.weights["B"].weight_map[layer]
        
        bind_gemm.gemmLauncher(A, B, C, self.outputs["D"].tensor, self.M, self.N, self.K, self.alpha, self.beta)

        # self.outputs["D"].tensor.copy_(A.matmul(B) + C)
    
    def processWeight(self, global_weight_map, total_layers, cached = False):
        self.weights["B"].weight_map = {}
        if not isinstance(self.weight_name, list):
            self.weight_name = [self.weight_name]
        for l in range(total_layers):
            self.weights["B"].weight_map[l] = torch.cat([global_weight_map[name.format(layer=l)].t() for name in self.weight_name], dim=1).contiguous()