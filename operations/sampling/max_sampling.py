import torch
import time
import platform_config
from operations.operation_base import Operations, Operation_Device, Operation_Layer
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer

from operations.impl_base import OperationImpl

class SamplingTorchImpl(OperationImpl):
    category_tag = "torch"
    def run(self, logits, tokens):
        # print("using torch")
        tokens.copy_(torch.argmax(logits, dim=1))

if platform_config.PLATFORM_CUDA:
    import bind_sample
    class SamplingCudaImpl(OperationImpl):
        category_tag = "cuda"
        def run(self, logits, tokens):
            # print("using cuda")
            bind_sample.SampleMax(logits, tokens)


class Sampling(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "logits": IOWrapper(self, 'logits')
        }
        self.outputs = {
            "tokens": IOWrapper(self, 'tokens', dtype=torch.int32)
        }
        self.impl_map = {}
        self.init_impl_map()
        self.op_device = Sampling_Device
    
    def init_impl_map(self):
        self.add_impl(SamplingTorchImpl)
        if platform_config.PLATFORM_CUDA:
            self.add_impl(SamplingCudaImpl)
    
    def setShape(self, vocab_size):
        self.vocab_size = vocab_size
        
    
    def profile(self):
        maxvals = torch.zeros(2, dtype=torch.float16, device='cuda')
        # check the similarity of the outputs
        logits = torch.randn(2, self.vocab_size, dtype=torch.float16, device='cuda')
        output_list = []
        for _, impl in self.impl_map.items():
            out = torch.zeros((2,), dtype=torch.int32, device='cuda')
            impl().run(logits, maxvals, out)
            output_list.append(out)
        self.checkConsistencyBetweenImpl(output_list)

        rounds = 100
        batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024]
        for batch_size in batch_sizes:
            out = torch.zeros((batch_size,), dtype=torch.int32, device='cuda')
            for _, impl in self.impl_map.items():
                impl_instance = impl()
                category_tag = impl_instance.category_tag
                total_latency = 0
                for round in range(rounds):
                    logits = torch.randn((batch_size, self.vocab_size), dtype=torch.float16, device='cuda')
                    # record the time
                    start_time = time.time()
                    impl_instance.run(logits, maxvals, out)
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
        logits = self.inputs["logits"].tensor
        # print("logits: ", logits)

        self.impl.run(logits, self.outputs["tokens"].tensor)

class Sampling_Device(Operation_Device):
    def __init__(self, op_general, name, device):
        super().__init__(op_general, name, device)
        self.op_layer = Sampling_Layer   

    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["logits"].shape = (self.batch_size, self.parent.vocab_size)
        self.outputs["tokens"].shape = (self.batch_size,)


class Sampling_Layer(Operation_Layer):
    def __init__(self, layer, operator_device):
        self.operator_device = operator_device
        self.name = f"{operator_device.name}_{layer}"
        self.layer = layer
        self.inputs = operator_device.inputs
        self.outputs = operator_device.outputs
        self.impl = operator_device.impl
    
    def run(self):
        self.operator_device.parent.impl.run(self.inputs["logits"].tensor, self.outputs["tokens"].tensor)