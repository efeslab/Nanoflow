import torch
import time
import platform_config
from operations.operation_base import Operations, Operation_Layer
from core.IOWrapper import IOWrapper
from core.weightWrapper import WeightWrapper    
from core.processWeight import process_weight_none, process_weight_layer

from operations.impl_base import OperationImpl

class SamplingTorchImpl(OperationImpl):
    category_tag = "torch"
    def run(self, logits, tokens):
        with torch.cuda.stream(self.stream):
            # print("using torch")
            tokens.copy_(torch.argmax(logits, dim=1))

if platform_config.PLATFORM_CUDA:
    import bind_sample
    class SamplingCudaImpl(OperationImpl):
        category_tag = "cuda"
        def run(self, logits, tokens):
            # print("using cuda")
            bind_sample.SampleMax(logits, tokens, self.stream_handle)


class Sampling(Operations):
    def __init__(self, name: str, device: str):
        super().__init__(name, device)
        self.inputs = {
            "logits": IOWrapper(self, 'logits', device).is_input(),
        }
        self.outputs = {
            "tokens": IOWrapper(self, 'tokens', device, dtype=torch.int32).is_output(),
        }
        self.impl_map = {}
        self.init_impl_map()
        self.op_layer = Sampling_Layer
    
    def init_impl_map(self):
        self.add_impl(SamplingTorchImpl)
        if platform_config.PLATFORM_CUDA:
            self.add_impl(SamplingCudaImpl)
    
    def setShape(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.inputs["logits"].init_shape((0, self.vocab_size))
        self.outputs["tokens"].init_shape((0,))

    def init_profile_db(self):
        for _, impl in self.impl_map.items():
            self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS "{impl.category_tag}" (
                id           INTEGER PRIMARY KEY AUTOINCREMENT, 
                batch_size   INTEGER,
                sm_count INTEGER,
                vocab_size INTEGER,
                average_time_ms REAL
            );
            ''')

    def store_profile_db(self, category_tag, impl_tag, average_elapsed_ms):
        print(f"Name: {self.name}, Category: {category_tag}, Batch Size: {self.batch_size}, Average Time: {average_elapsed_ms} ms")
        self.cursor.execute(f'''
            INSERT OR IGNORE INTO {category_tag} (batch_size, sm_count, vocab_size, average_time_ms)
            VALUES (?, ?, ?, ?)
            ''', (self.batch_size, self.sm_count, self.vocab_size, average_elapsed_ms))

    def run(self):
        self.impl.run(self.inputs["logits"].tensor, self.outputs["tokens"].tensor)

    def profile_run(self):
        return self.run()

class Sampling_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)
    
    def run(self):
        self.impl.run(self.inputs["logits"].tensor, self.outputs["tokens"].tensor)