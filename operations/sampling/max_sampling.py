import torch
import time
from operations.operation_base import Operations
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
import bind_sample
from operations.impl_base import OperationImpl

class SamplingTorchImpl(OperationImpl):
    category_tag = "torch"
    def run(self, logits, tokens):
        print("using torch")
        tokens.copy_(torch.argmax(logits, dim=1))

class SamplingCudaImpl(OperationImpl):
    category_tag = "cuda"
    def run(self, logits, tokens):
        print("using cuda")
        maxvals = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
        bind_sample.SampleMax(logits, maxvals, tokens)


class Sampling(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "logits": IOWrapper(self, 'logits', IOBufferType.FULL)
        }
        self.outputs = {
            "tokens": IOWrapper(self, 'tokens', IOBufferType.FULL, dtype=torch.int32)
        }
        self.impl_map = {}
        self.init_impl_map()
    
    def init_impl_map(self):
        self.add_impl(SamplingTorchImpl)
        self.add_impl(SamplingCudaImpl)
    
    def setShape(self, vocab_size):
        self.vocab_size = vocab_size
        
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["logits"].shape = (self.batch_size, self.vocab_size)
        self.outputs["tokens"].shape = (self.batch_size,)
    
    def profile(self):
        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        for batch_size in batch_sizes:
            total_latency = 0
            for round in range(rounds):
                logits = torch.randn((batch_size, self.vocab_size), dtype=torch.float16, device='cuda')
                maxvals = torch.zeros(logits.shape[0], dtype=logits.dtype, device=logits.device)
                tokens = torch.zeros((batch_size,), dtype=torch.int32, device='cuda')
                # record the time
                start_time = time.time()
                bind_sample.SampleMax(logits, maxvals, tokens)
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
        logits = self.inputs["logits"].tensor
        # print("logits: ", logits)
    
        self.impl.run(logits, self.outputs["tokens"].tensor)