import torch
import sys
import time
sys.path.append('../../pybind/build')

from operations.operation_base import Operations, Operation_Device, Operation_Layer
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer
import platform_config as config
from operations.impl_base import OperationImpl

class SiluMultiplyTorchImpl(OperationImpl):
    category_tag = "torch"
    def run(self, x, output):
        with torch.cuda.stream(self.stream):
            A, B = torch.split(x, x.shape[-1] // 2, dim=-1)
            output.copy_(A * torch.nn.functional.silu(B))

if config.PLATFORM_CUDA:
    import bind_silu_multiply
    class SiluMultiplyCudaImpl(OperationImpl):
        category_tag = "cuda"
        def run(self, x, output):
            if self.batch_size > 0:
                bind_silu_multiply.silu_multiply(x, output, self.stream_handle)

class Activation(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "input": IOWrapper(self, 'input'),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output')
        }
        self.act_fn = torch.nn.SiLU()
        self.impl_map = {}
        self.init_impl_map()
        self.op_device = Activation_Device
    
    def init_impl_map(self):
        self.add_impl(SiluMultiplyTorchImpl)
        if config.PLATFORM_CUDA:
            self.add_impl(SiluMultiplyCudaImpl)
        
    def setShape(self, N, tp_size=1):
        self.N = N // tp_size
        self.updateChildrenIOShape()
    
    def copy_nano(self, index):
        new_op = Activation(f"{self.name}{index}")
        new_op.expand_all_gpu_and_layers(len(self.device_list), 32)
        new_op.setShape(self.N)
        new_op.set_stream(self.stream)

        self.nano_ops.append(new_op)

        return new_op

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

class Activation_Device(Operation_Device):
    def __init__(self, parent, device):
        super().__init__(parent, device)
        self.op_layer = Activation_Layer

    def setShapeForIOWrappers(self):
        self.inputs["input"].init_shape((0, self.parent.N * 2))
        self.outputs["output"].init_shape((0, self.parent.N))

class Activation_Layer(Operation_Layer):
    def __init__(self, layer, op_device):
        super().__init__(layer=layer, op_device=op_device)
    
    def run(self):
        self.impl.run(self.inputs["input"].tensor, self.outputs["output"].tensor)