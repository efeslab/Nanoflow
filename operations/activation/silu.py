import torch
import sys
import time
sys.path.append('../../pybind/build')

from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
import bind_silu_multiply
from operations.impl_base import OperationImpl

class SiluMultiplyTorchImpl(OperationImpl):
    category_tag = "torch"
    def run(self, x, output):
        A, B = torch.split(x, x.shape[-1] // 2, dim=-1)
        output.copy_(A * torch.nn.functional.silu(B))
        
class SiluMultiplyCudaImpl(OperationImpl):
    category_tag = "cuda"
    def run(self, x, output):
        bind_silu_multiply.silu_multiply(x, output)

class Activation(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "input": IOWrapper(self, 'input', IOBufferType.FULL),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.FULL)
        }
        self.act_fn = torch.nn.SiLU()
        self.impl_map = {}
        self.init_impl_map()
    
    def init_impl_map(self):
        self.add_impl(SiluMultiplyTorchImpl)
        self.add_impl(SiluMultiplyCudaImpl)
        
    def setShape(self, N):
        self.N = N
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["input"].shape = (self.batch_size, self.N * 2)
        self.outputs["output"].shape = (self.batch_size, self.N)

    def profile(self):
        
        # warm up
        x = torch.randn(2, self.N * 2, dtype=torch.float16, device='cuda')
        out = torch.zeros((2, self.N), dtype=torch.float16, device='cuda')
        bind_silu_multiply.silu_multiply(x, out)

        # profile the performance
        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        for batch_size in batch_sizes:
            total_latency_new = 0
            total_latency_old = 0
            for round in range(rounds):
                x = torch.randn(batch_size, self.N * 2, dtype=torch.float16, device='cuda')
                out = torch.zeros((batch_size, self.N), dtype=torch.float16, device='cuda')
                # record the time
                start_time = time.time()
                bind_silu_multiply.silu_multiply(x, out)
                latency_new = time.time() - start_time
                # print(latency_new)
                total_latency_new += latency_new
                start_time_old = time.time()
                A, B = torch.split(x, self.N, dim=-1)
                out.copy_(A * self.act_fn(B))
                latency_old = time.time() - start_time_old
                total_latency_old += latency_old

            average_time_old = total_latency_old / rounds
            average_time_new = total_latency_new / rounds
            print("name: {}, batch_size: {}, average_time: {}".format(self.name, batch_size, average_time_new))
            print("name: {}, batch_size: {}, average_time: {}".format(self.name + "_old", batch_size, average_time_old))
            self.cursor.execute('''
                INSERT OR REPLACE INTO performance (id, keyword, batch_size, average_time)
                VALUES ((SELECT id FROM performance WHERE keyword = ? AND batch_size = ?), ?, ?, ?)
            ''', (self.name, batch_size, self.name, batch_size, average_time_new))
            self.cursor.execute('''
                INSERT OR REPLACE INTO performance (id, keyword, batch_size, average_time)
                VALUES ((SELECT id FROM performance WHERE keyword = ? AND batch_size = ?), ?, ?, ?)
            ''', (self.name + "_old", batch_size, self.name + "_old", batch_size, average_time_old))
            self.conn.commit()
    
    def search_profile_data(self):
        self.cursor.execute('''
            SELECT * FROM performance
        ''')
        rows = self.cursor.fetchall()
        for row in rows:
            print(row)
        
    def run(self, layer):
        x = self.inputs["input"].tensor
        self.impl.run(x, self.outputs["output"].tensor)