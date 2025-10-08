import torch

import nanoflow.platform_config as platform_config
from nanoflow.operations import Operations, Operation_Layer, OperationImpl
from nanoflow.core import IOWrapper, WeightWrapper, process_weight_no_transpose
from nanoflow.utils.prof_marker import prof_marker


class GenEmbeddingTorchImpl(OperationImpl):
    category_tag = "torch"

    def run(self, tokens, embedding, output):
        with torch.cuda.stream(self.stream):
            # print("using torch")
            output.copy_(embedding[tokens])


if platform_config.PLATFORM_CUDA:
    import nanoflow.pybind.build.bind_genEmbedding as bind_genEmbedding

    class GenEmbeddingCudaImpl(OperationImpl):
        category_tag = "cuda"

        def run(self, tokens, embedding, output):
            # print("using cuda")
            if self.batch_size > 0:
                bind_genEmbedding.genEmbedding(
                    tokens, embedding, output, self.stream_handle
                )


class GenEmbedding(Operations):
    def __init__(self, name, device):
        super().__init__(name, device)
        self.inputs = {
            "token": IOWrapper(self, "token", device, dtype=torch.int32).is_input(),
        }
        self.outputs = {
            "output": IOWrapper(self, "output", device).is_output(),
        }
        self.weights = {"embedding": WeightWrapper(self)}
        self.init_impl_map()
        self.op_layer = GenEmbedding_Layer

    def init_impl_map(self):
        self.impl_map = {}
        self.add_impl(GenEmbeddingTorchImpl)
        if platform_config.PLATFORM_CUDA:
            self.add_impl(GenEmbeddingCudaImpl)

    def setShape(self, hidden_dim, vocab_size):
        self.N = hidden_dim
        self.vocab_size = vocab_size
        self.weights["embedding"].shape = (self.vocab_size, self.N)
        self.inputs["token"].init_shape((0,))
        self.outputs["output"].init_shape((0, self.N))

        return self

    def init_profile_db(self):
        for _, impl in self.impl_map.items():
            self.cursor.execute(
                f"""
            CREATE TABLE IF NOT EXISTS "{impl.category_tag}" (
                id           INTEGER PRIMARY KEY AUTOINCREMENT, 
                batch_size   INTEGER,
                sm_count INTEGER,
                vocab_size INTEGER,
                hidden_dim INTEGER,
                average_time_ms REAL
            );
            """
            )

    def store_profile_db(self, category_tag, impl_tag, average_elapsed_ms):
        print(
            f"Name: {self.name}, Category: {category_tag}, Batch Size: {self.batch_size}, Average Time: {average_elapsed_ms} ms"
        )
        self.cursor.execute(
            f"""
            INSERT OR IGNORE INTO {category_tag} (batch_size, sm_count, vocab_size, hidden_dim, average_time_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                self.batch_size,
                self.sm_count,
                self.vocab_size,
                self.N,
                average_elapsed_ms,
            ),
        )

    def run(self, layer):
        self.impl.run(
            self.inputs["token"].tensor,
            self.weights["embedding"].weight_map[layer],
            self.outputs["output"].tensor,
        )

    def profile_run(self):
        self.run(self.layer_list[0])

    def processWeight(self, global_weight_map, cached_weight_map, cached, device):
        return process_weight_no_transpose(
            global_weight_map,
            self.weight_name,
            self.weights["embedding"],
            self.layer_list,
            cached_weight_map,
            cached,
            device,
        )


class GenEmbedding_Layer(Operation_Layer):
    def __init__(self, layer, base_op):
        super().__init__(layer, base_op)

    def run(self):
        self.parent.run(self.layer)
