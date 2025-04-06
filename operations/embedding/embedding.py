import torch
import sys
import time
sys.path.append('../../pybind/build')

from operations.operation_base import Operations, Operation_Device, Operation_Layer
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper
from core.processWeight import process_weight_no_transpose
import bind_genEmbedding
from operations.impl_base import OperationImpl

class GenEmbeddingTorchImpl(OperationImpl):
    category_tag = "torch"
    def run(self, tokens, embedding, output):
        # print("using torch")
        output.copy_(embedding[tokens])
        
class GenEmbeddingCudaImpl(OperationImpl):
    category_tag = "cuda"
    def run(self, tokens, embedding, output):
        # print("using cuda")
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
        self.add_impl(GenEmbeddingTorchImpl)
        self.add_impl(GenEmbeddingCudaImpl)
        
    def setShape(self, hidden_dim, vocab_size):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.weights["embedding"].shape = (vocab_size, hidden_dim)
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["token"].shape = (self.batch_size,)
        self.outputs["output"].shape = (self.batch_size, self.hidden_dim)
    
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
        
    def run(self, layer):
        # Retrieve token indices from input and the embedding matrix from weight_map.
        tokens = self.inputs["token"].tensor  # expected shape: (1, batch_size)
        embedding = self.weights["embedding"].weight_map[layer]  # expected shape: (vocab_size, hidden_dim)
        self.impl.run(tokens, embedding, self.outputs["output"].tensor)
    
    def processWeight(self, global_weight_map, total_layers, cached = False):
        return process_weight_no_transpose(global_weight_map, self.weight_name, self.weights["embedding"], total_layers, cached)

    def expand_gpu(self, gpu_list):
        for i in gpu_list:
            i_str = str(i)
            name = self.name + "_" + i_str
            op_device = GenEmbedding_Device(self, self.name, i)
            self.children.append(op_device)
        
        return self.children
    
class GenEmbedding_Device(Operation_Device):
    def __init__(self, op_general, name, device):
        super().__init__(op_general, name, device)
        
    def expand_layer(self, layer_list):
        for i in layer_list:
            op_layer = GenEmbedding_Layer(i, self)
            self.children.append(op_layer)
        
        return self.children

class GenEmbedding_Layer(Operation_Layer):
    def __init__(self, layer, operator_device):
        self.operator_device = operator_device
        self.name = f"{operator_device.name}_{layer}"
        self.layer = layer
        self.inputs = operator_device.inputs
        self.outputs = operator_device.outputs
        self.weights = operator_device.weights
        self.impl = operator_device.impl
    
    def run(self):
        self.operator_device.parent.impl.run(self.inputs["token"].tensor, self.weights["embedding"].weight_map[self.layer], self.outputs["output"].tensor)