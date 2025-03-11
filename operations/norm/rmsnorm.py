import torch
import time
from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
import bind_rms_norm
from operations.impl_base import OperationImpl

class LayerNormTorchImpl(OperationImpl):
    category_tag = "torch"
    def run(self, x, weight, output, epsilon):
        # print("using torch")
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)
        normalized_x = x / rms
        output.copy_(normalized_x.to(torch.float16) * weight)

class LayerNormCudaImpl(OperationImpl):
    category_tag = "cuda"
    def run(self, x, weight, output, epsilon):
        # print("using cuda")
        bind_rms_norm.rms_norm(output, x, weight, epsilon)

class LayerNorm(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "input": IOWrapper(self, 'input', IOBufferType.FULL),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.FULL)
        }
        self.weights = {
            "weight": WeightWrapper(),
        }
        self.impl_map = {}
        self.init_impl_map()

    def init_impl_map(self):
        self.add_impl(LayerNormTorchImpl)
        self.add_impl(LayerNormCudaImpl)
    
    def setShape(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.weights["weight"].shape = (self.hidden_dim,)
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["input"].shape = (self.batch_size, self.hidden_dim)
        self.outputs["output"].shape = (self.batch_size, self.hidden_dim)

    def profile(self):
        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        for batch_size in batch_sizes:
            total_latency = 0
            for round in range(rounds):
                x = torch.randn((batch_size, self.hidden_dim), dtype=torch.float16, device='cuda')
                weight = torch.randn((self.hidden_dim,), dtype=torch.float16, device='cuda')
                out = torch.zeros((batch_size, self.hidden_dim), dtype=torch.float16, device='cuda')
                # record the time
                start_time = time.time()
                bind_rms_norm.rms_norm(out, x, weight, 1e-5)
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
        x = self.inputs["input"].tensor
        epsilon = 1e-5
        weight_val = self.weights["weight"].weight_map[layer]
        
        self.impl.run(x, weight_val, self.outputs["output"].tensor, epsilon)
    
    def processWeight(self, global_weight_map, total_layers, cached = False):
        return process_weight_layer(global_weight_map, self.weight_name, self.weights["weight"], total_layers, cached)
        