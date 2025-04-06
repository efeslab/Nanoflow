import torch
import sys
import time
sys.path.append('../../pybind/build')

from operations.operation_base import Operations, Operation_Device, Operation_Layer
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
        # check the similarity of the outputs
        x = torch.randn(2, self.N * 2, dtype=torch.float16, device='cuda')
        output_list = []
        for _, impl in self.impl_map.items():
            out = torch.zeros((2, self.N), dtype=torch.float16, device='cuda')
            impl().run(x, out)
            output_list.append(out)
        self.checkConsistencyBetweenImpl(output_list)

        # profile the performance
        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        for batch_size in batch_sizes:
            out = torch.zeros((batch_size, self.N), dtype=torch.float16, device='cuda')
            for _, impl in self.impl_map.items():
                impl_instance = impl()
                category_tag = impl_instance.category_tag
                total_latency = 0
                for round in range(rounds):
                    x = torch.randn(batch_size, self.N * 2, dtype=torch.float16, device='cuda')
                    # record the time
                    start_time = time.time()
                    impl_instance.run(x, out)
                    if round > 0:
                        total_latency += time.time() - start_time

                average_time = total_latency / rounds
                print("name: {}, batch_size: {}, average_time: {}".format(self.name + f"_{category_tag}", batch_size, average_time))
                self.cursor.execute('''
                    INSERT INTO performance (keyword, batch_size, average_time)
                    VALUES (?, ?, ?)
                    ''', (self.name + f"_{category_tag}", batch_size, average_time))
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

    def expand_gpu(self, gpu_list):
        for i in gpu_list:
            i_str = str(i)
            name = self.name + "_" + i_str
            op_device = Activation_Device(self, self.name, i)
            self.children.append(op_device)
        
        return self.children

class Activation_Device(Operation_Device):
    def __init__(self, op_general, name, device):
        super().__init__(op_general, name, device)    

    def expand_layer(self, layer_list):
        for i in layer_list:
            op_layer = Activation_Layer(i, self)
            self.children.append(op_layer)
        
        return self.children

class Activation_Layer(Operations):
    def __init__(self, layer, operation_device):
        self.operator_device = operation_device
        self.name = f"{operation_device.name}_{layer}"
        self.layer = layer
        self.inputs = operation_device.inputs
        self.outputs = operation_device.outputs
        self.impl = operation_device.impl
    
    def run(self):
        self.operator_device.parent.impl.run(self.inputs["input"].tensor, self.outputs["output"].tensor)