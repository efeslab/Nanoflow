import torch
import time

import platform_config
from operations.operation_base import Operations, Operation_Layer
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper
from core.processWeight import process_weight_no_transpose

from operations.impl_base import OperationImpl

class GenEmbeddingTorchImpl(OperationImpl):
    category_tag = "torch"
    def run(self, tokens, embedding, output):
        with torch.cuda.stream(self.stream):
            # print("using torch")
            output.copy_(embedding[tokens])

if platform_config.PLATFORM_CUDA:
    import bind_genEmbedding
    class GenEmbeddingCudaImpl(OperationImpl):
        category_tag = "cuda"
        def run(self, tokens, embedding, output):
            # print("using cuda")
            if self.batch_size > 0:
                bind_genEmbedding.genEmbedding(tokens, embedding, output, self.stream_handle)
            
class GenEmbedding(Operations):
    def __init__(self, name, device):
        super().__init__(name, device)
        self.inputs = {
            "token": IOWrapper(self, 'token', device, dtype=torch.int32).is_input(),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', device).is_output(),
        }
        self.weights = {
            "embedding": WeightWrapper(self)
        }
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = GenEmbedding_Layer
    
    def init_impl_map(self):
        self.add_impl(GenEmbeddingTorchImpl)
        if platform_config.PLATFORM_CUDA:
            self.add_impl(GenEmbeddingCudaImpl)
        
    def setShape(self, hidden_dim, vocab_size):
        self.N = hidden_dim
        self.vocab_size = vocab_size
        self.weights["embedding"].shape = (self.vocab_size, self.N)
        self.inputs["token"].init_shape((0,))
        self.outputs["output"].init_shape((0, self.N))
    
    
    def profile(self):
        # check the similarity of the outputs
        tokens = torch.randint(self.vocab_size, (2,), dtype=torch.int32, device='cuda')
        embedding = torch.randn(self.vocab_size, self.hidden_dim, dtype=torch.float16, device='cuda')
        output_list = []
        for _, impl in self.impl_map.items():
            out = torch.zeros((2, self.hidden_dim), dtype=torch.float16, device='cuda')
            impl().run(tokens, embedding, out)
            output_list.append(out)

        self.checkConsistencyBetweenImpl(output_list)

        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]

        for batch_size in batch_sizes:
            output = torch.zeros((batch_size, self.hidden_dim), dtype=torch.float16, device='cuda')
            for _, impl in self.impl_map.items():
                impl_instance = impl()
                category_tag = impl.category_tag
                total_latency = 0
                for round in range(rounds):
                    tokens = torch.randint(self.vocab_size, (batch_size,), dtype=torch.int32, device='cuda')
                    start = time.time()
                    impl_instance.run(tokens, embedding, output)
                    torch.cuda.synchronize()
                    if round > 0:
                        total_latency += time.time() - start
                average_time = total_latency / rounds
                print("name: {}, batch_size: {}, average_time: {}".format(self.name + f"_{category_tag}", batch_size, average_time))
                self.cursor.execute('''
                    INSERT INTO performance (keyword, batch_size, average_time)
                    VALUES (?, ?, ?)
                    ''', (self.name + f"_{category_tag}", batch_size, average_time))
        self.conn.commit()
    
    def processWeight(self, global_weight_map, cached_weight_map, cached, device):
        return process_weight_no_transpose(global_weight_map, self.weight_name, self.weights["embedding"], self.layer_list, cached_weight_map, cached, device)

class GenEmbedding_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)
    
    def run(self):
        self.impl.run(self.inputs["token"].tensor, self.weights["embedding"].weight_map[self.layer], self.outputs["output"].tensor)