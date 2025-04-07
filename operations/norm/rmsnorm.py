import torch
import time
import platform_config
from operations.operation_base import Operations, Operation_Device, Operation_Layer
from core.IOWrapper import IOWrapper, IOBufferType
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer

from operations.impl_base import OperationImpl

class LayerNormTorchImpl(OperationImpl):
    category_tag = "torch"
    def run(self, x, weight, output, epsilon):
        # print("using torch")
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)
        normalized_x = x / rms
        output.copy_(normalized_x.to(torch.float16) * weight)

if platform_config.PLATFORM_CUDA:
    import bind_rms_norm
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
        if platform_config.PLATFORM_CUDA:
            self.add_impl(LayerNormCudaImpl)
    
    def setShape(self, hidden_dim):
        self.hidden_dim = hidden_dim
        self.weights["weight"].shape = (self.hidden_dim,)

    def profile(self):
        # check the similarity of the outputs
        x = torch.randn(2, self.hidden_dim, dtype=torch.float16, device='cuda')
        weight = torch.randn(self.hidden_dim, dtype=torch.float16, device='cuda')
        output_list = []
        for _, impl in self.impl_map.items():
            out = torch.zeros((2, self.hidden_dim), dtype=torch.float16, device='cuda')
            impl().run(x, weight, out, 1e-5)
            output_list.append(out)
        self.checkConsistencyBetweenImpl(output_list)

        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        for batch_size in batch_sizes:
            out = torch.zeros((batch_size, self.hidden_dim), dtype=torch.float16, device='cuda')
            for _, impl in self.impl_map.items():
                impl_instance = impl()
                category_tag = impl_instance.category_tag
                total_latency = 0
                for round in range(rounds):
                    x = torch.randn((batch_size, self.hidden_dim), dtype=torch.float16, device='cuda')
                    weight = torch.randn(self.hidden_dim, dtype=torch.float16, device='cuda')
                    # record the time
                    start_time = time.time()
                    impl_instance.run(x, weight, out, 1e-5)
                    if round > 0:
                        total_latency += time.time() - start_time
                average_time = total_latency / rounds
                print("name: {}, batch_size: {}, average_time: {}".format(self.name + f"_{category_tag}", batch_size, average_time))
                self.cursor.execute('''
                    INSERT INTO performance (keyword, batch_size, average_time)
                    VALUES (?, ?, ?)
                    ''', (self.name + f"_{category_tag}", batch_size, average_time))
        self.conn.commit()

    def run(self, layer):
        x = self.inputs["input"].tensor
        weight_val = self.weights["weight"].weight_map[layer]
        
        self.impl.run(x, weight_val, self.outputs["output"].tensor, epsilon = 1e-5)
    
    def processWeight(self, global_weight_map, total_layers, cached = False):
        return process_weight_layer(global_weight_map, self.weight_name, self.weights["weight"], total_layers, cached)
        
    def expand_gpu(self, gpu_list):
        for i in gpu_list:
            i_str = str(i)
            name = self.name + "_" + i_str
            op_device = LayerNorm_Device(self, self.name, i)
            self.children.append(op_device)
        
        return self.children
    
class LayerNorm_Device(Operation_Device):
    def __init__(self, op_general, name, device):
        super().__init__(op_general, name, device)

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["input"].shape = (self.batch_size, self.parent.hidden_dim)
        self.outputs["output"].shape = (self.batch_size, self.parent.hidden_dim)

    def expand_layer(self, layer_list):
        for i in layer_list:
            op_layer = LayerNorm_Layer(i, self)
            self.children.append(op_layer)
        
        return self.children
        

class LayerNorm_Layer(Operation_Layer):
    def __init__(self, layer, operator_device):
        self.operator_device = operator_device
        self.name = f"{operator_device.name}_{layer}"
        self.layer = layer
        self.inputs = operator_device.inputs
        self.outputs = operator_device.outputs
        self.weights = operator_device.weights
        self.impl = operator_device.impl

    def run(self):
        self.operator_device.parent.impl.run(self.inputs["input"].tensor, self.weights["weight"].weight_map[self.layer], self.outputs["output"].tensor, epsilon = 1e-5)