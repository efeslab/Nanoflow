import torch
import sys
import time
sys.path.append('../../pybind/build')

from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper
from core.processWeight import process_weight_no_transpose
import bind_genEmbedding
from operations.impl_base import OperationImpl

class GenEmbeddingTorch(OperationImpl):
    category_tag = "torch"
    def run(self, tokens, embedding, output):
        print("using torch")
        output.copy_(embedding[tokens])
        
class GenEmbeddingCuda(OperationImpl):
    category_tag = "cuda"
    def run(self, tokens, embedding, output):
        print("using cuda")
        bind_genEmbedding.genEmbedding(tokens, embedding, output)
        
class GenEmbedding(Operations):
    
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "token": IOWrapper(self, 'token', IOBufferType.FULL, dtype=torch.int32),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.FULL)
        }
        self.weights = {
            "embedding": WeightWrapper()
        }
        self.impl_map = {}
        self.init_impl_map()
    
    def init_impl_map(self):
        self.add_impl(GenEmbeddingTorch)
        self.add_impl(GenEmbeddingCuda)
        
    def setShape(self, hidden_dim, vocab_size):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.weights["embedding"].shape = (vocab_size, hidden_dim)
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["token"].shape = (self.batch_size,)
        self.outputs["output"].shape = (self.batch_size, self.hidden_dim)
    
    def profile(self):
        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        embedding = torch.randn(self.vocab_size, self.hidden_dim, dtype=torch.float16, device='cuda')
        for batch_size in batch_sizes:
            total_latency = 0
            for round in range(rounds):
                input = torch.randint(self.vocab_size, (batch_size,), dtype=torch.int32, device='cuda')
                output = torch.zeros((batch_size, self.hidden_dim), dtype=torch.float16, device='cuda')
                # record the time
                start_time = time.perf_counter()
            
                self.impl.run(input, embedding, output)

                if round > 0:
                    latency = time.perf_counter() - start_time
                    total_latency += latency
            average_time = total_latency / rounds
            print("name: {}, batch_size: {}, average_time: {}".format(self.name, batch_size, average_time))
            self.cursor.execute('''
                INSERT OR REPLACE INTO performance (id, keyword, batch_size, average_time)
                VALUES ((SELECT id FROM performance WHERE keyword = ? AND batch_size = ?), ?, ?, ?)
            ''', (self.name, batch_size, self.name, batch_size, average_time))
            self.conn.commit()
            

    def run(self, layer):
        # Retrieve token indices from input and the embedding matrix from weight_map.
        tokens = self.inputs["token"].tensor  # expected shape: (1, batch_size)
        embedding = self.weights["embedding"].weight_map[layer]  # expected shape: (vocab_size, hidden_dim)
        self.impl.run(tokens, embedding, self.outputs["output"].tensor)
    
    def processWeight(self, global_weight_map, total_layers, cached = False):
        return process_weight_no_transpose(global_weight_map, self.weight_name, self.weights["embedding"], total_layers, cached)
    