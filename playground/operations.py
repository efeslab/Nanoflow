from enum import Enum

class IOBufferType(Enum):
    FULL = 1
    NPartition = 2
    MPartition = 3
    KPartition = 4

class IOWrapper:
    def __init__(self, owner, name, IOtype):
        self.owner = owner  # owner is now an Operations object or similar
        self.name = name
        self.prev = None
        self.next = None
        self.shape = None
        self.ptr = 0
        self.IOtype = IOtype
    
    @property
    def fullName(self):
        owner_name = self.owner.name if hasattr(self.owner, "name") else str(self.owner)
        return f"{owner_name}_{self.name}"
    
    def chain(self, next_wrapper, first=False, last=False):
        assert self.IOtype == next_wrapper.IOtype, f"IOBufferType mismatch: {self.IOtype} != {next_wrapper.IOtype}"
        if self.next is not None:
            self.next.append(next_wrapper)
        else:
            self.next = [next_wrapper]
        if next_wrapper.prev is None:
            next_wrapper.prev = []
        next_wrapper.prev.append(self)
        return next_wrapper
    
    def __rshift__(self, next_wrapper):
        return self.chain(next_wrapper)

class Operations:
    def __init__(self, name):
        self.inputs = {}
        self.outputs = {}
        self.weights = {}
        self.externals = {}
        self.name = name
        self.first_layer_only = False
        self.last_layer_only = False
    
    @property
    def prerequisites(self):
        dep = []
        for _, input_wrapper in self.inputs.items():
            for dep_wrapper in input_wrapper.prev:
                dep.append(dep_wrapper.owner)
        return dep
        
    def checkDep(self):
        for name, IOwrapper in self.inputs.items():
            if IOwrapper.prev is None:
                raise Exception(f"Operation {self.name}, Input {name} is not connected")
    
    def initOutputBuffer(self):
        for _, wrapper in self.outputs.items():
            wrapper.owner = self
    
    def __str__(self):
        return self.name    

# -------------------------------------------------------------------
# Operations definitions (ordered loosely by type)

class GlobalInput(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {}
        self.outputs = {
            "tokens": IOWrapper(self, 'tokens', IOBufferType.FULL)
        }
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size

class GenEmbedding(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "token": IOWrapper(self, 'token', IOBufferType.FULL),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.FULL)
        }
        self.weights = {
            "embedding": IOWrapper(self, 'embedding', IOBufferType.FULL)
        }
    
    def setShape(self, hidden_dim, vocab_size):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["token"].shape = (1, self.batch_size)

class LayerNorm(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "input": IOWrapper(self, 'input', IOBufferType.FULL),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.FULL)
        }
        self.weights = {}
    
    def setShape(self, hidden_dim):
        self.hidden_dim = hidden_dim
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["input"].shape = (self.batch_size, self.hidden_dim)
        self.outputs["output"].shape = (self.batch_size, self.hidden_dim)

class GEMMNoBias(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "A": IOWrapper(self, 'A', IOBufferType.FULL),
        }
        self.outputs = {
            "D": IOWrapper(self, 'D', IOBufferType.FULL)
        }
        self.weights = {
            "B": IOWrapper(self, 'B', IOBufferType.FULL)
        }
    
    def setShape(self, N, K):
        self.N = N
        self.K = K
    
    def setBatchSize(self, M):
        self.M = M
        self.inputs["A"].shape = (self.M, self.K)
        self.outputs["D"].shape = (self.M, self.N)
        self.weights["B"].shape = (self.K, self.N)

class RopeAppend(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "kqv": IOWrapper(self, 'kqv', IOBufferType.FULL),
        }
        self.outputs = {
            "q": IOWrapper(self, 'q', IOBufferType.FULL),
        }
        self.externals = {
            "KVdata": IOWrapper(self, 'KVdata', IOBufferType.FULL)
        }
    
    def setShape(self, num_kv_heads, num_qo_heads, head_dim):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["kqv"].shape = (self.batch_size, self.num_qo_heads + 2 * self.num_kv_heads * self.head_dim)
        self.outputs["q"].shape = (self.batch_size, self.num_qo_heads, self.head_dim)

class DecAttn(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "Q": IOWrapper(self, 'Q', IOBufferType.MPartition),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.MPartition)
        }
        self.externals = {
            "KVdata": IOWrapper(self, 'KVdata', IOBufferType.FULL)
        }
    
    def setShape(self, num_kv_heads, num_qo_heads, head_dim):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.q_dim = num_qo_heads * head_dim
        
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["Q"].shape = (self.batch_size, self.q_dim)
        self.outputs["output"].shape = (self.batch_size, self.q_dim)

class PFAttn(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "Q": IOWrapper(self, 'Q', IOBufferType.MPartition),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.MPartition)
        }
        self.externals = {
            "KVdata": IOWrapper(self, 'KVdata', IOBufferType.FULL)
        }
    
    def setShape(self, num_kv_heads, num_qo_heads, head_dim):
        self.num_kv_heads = num_kv_heads
        self.num_qo_heads = num_qo_heads
        self.head_dim = head_dim
        self.q_dim = num_qo_heads * head_dim
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["Q"].shape = (self.batch_size, self.q_dim)
        self.outputs["output"].shape = (self.batch_size, self.q_dim)

class GEMM(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "A": IOWrapper(self, 'A', IOBufferType.FULL),
            "C": IOWrapper(self, 'C', IOBufferType.FULL)
        }
        self.outputs = {
            "D": IOWrapper(self, 'D', IOBufferType.FULL)
        }
        self.weights = {
            "B": IOWrapper(self, 'B', IOBufferType.FULL)
        }
        
    def setShape(self, N, K):
        self.N = N
        self.K = K
    
    def setBatchSize(self, M):
        self.M = M
        self.inputs["A"].shape = (self.M, self.K)
        self.inputs["C"].shape = (self.M, self.N)
        self.outputs["D"].shape = (self.M, self.N)
        self.weights["B"].shape = (self.K, self.N)

class Activation(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "input": IOWrapper(self, 'input', IOBufferType.FULL),
        }
        self.outputs = {
            "output": IOWrapper(self, 'output', IOBufferType.FULL)
        }
        
    def setShape(self, N):
        self.N = N
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        

class Sampling(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "logits": IOWrapper(self, 'logits', IOBufferType.FULL)
        }
        self.outputs = {
            "tokens": IOWrapper(self, 'tokens', IOBufferType.FULL)
        }
    
    def setShape(self, vocab_size):
        self.vocab_size = vocab_size
        
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["logits"].shape = (self.batch_size, self.vocab_size)
        self.outputs["tokens"].shape = (1, self.batch_size)

class GlobalOutput(Operations):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {
            "tokens": IOWrapper(self, 'tokens', IOBufferType.FULL)
        }
        self.outputs = {}
    
    def setBatchSize(self, batch_size):
        self.batch_size = batch_size
        self.inputs["tokens"].shape = (1, self.batch_size)

# -------------------------------------------------------------------
# Instantiate operations (in operation_list order)

global_input    = GlobalInput("GlobalInput")
global_input.first_layer_only = True

gen_embedding   = GenEmbedding("GenEmbedding")
gen_embedding.first_layer_only = True

layerNormAttn   = LayerNorm("LayerNormAttn")

kqv             = GEMMNoBias("KQV")

ropeAppend      = RopeAppend("RopeAppend")

decAttn         = DecAttn("DecAttn")

pfAttn          = PFAttn("PFAttn")

o               = GEMM("O")

layerNormFFN    = LayerNorm("LayerNormFFN")

ug              = GEMMNoBias("UG")

activation      = Activation("Activation")

d               = GEMM("D")

modelLayerNorm  = LayerNorm("ModelLayerNorm")
modelLayerNorm.last_layer_only = True

sample          = Sampling("Sampling")
sample.last_layer_only = True

global_output   = GlobalOutput("GlobalOutput")
global_output.last_layer_only = True

# -------------------------------------------------------------------
# Define chaining among operations

global_input.outputs["tokens"] >> gen_embedding.inputs["token"]

gen_embedding.outputs["output"] >> layerNormAttn.inputs["input"]

layerNormAttn.outputs["output"] >> kqv.inputs["A"]

kqv.outputs["D"] >> ropeAppend.inputs["kqv"]

# Chain external dependency for key/value data
ropeAppend.externals["KVdata"] >> decAttn.externals["KVdata"]
ropeAppend.externals["KVdata"] >> pfAttn.externals["KVdata"]

ropeAppend.outputs["q"] >> decAttn.inputs["Q"]
ropeAppend.outputs["q"] >> pfAttn.inputs["Q"]

decAttn.outputs["output"] >> o.inputs["A"]
gen_embedding.outputs["output"] >> o.inputs["C"]

o.outputs["D"] >> layerNormFFN.inputs["input"]

layerNormFFN.outputs["output"] >> ug.inputs["A"]

ug.outputs["D"] >> activation.inputs["input"]

activation.outputs["output"] >> d.inputs["A"]

o.outputs["D"] >> d.inputs["C"]

d.outputs["D"] >> modelLayerNorm.inputs["input"]

# Additional dependency: d feeds back to layerNormAttn
d.outputs["D"] >> layerNormAttn.inputs["input"]

modelLayerNorm.outputs["output"] >> sample.inputs["logits"]

sample.outputs["tokens"] >> global_output.inputs["tokens"]

# -------------------------------------------------------------------
# Collect all operations in a list (for later processing)
operation_list = [
    global_input, gen_embedding, layerNormAttn, kqv, ropeAppend,
    decAttn, pfAttn, o, layerNormFFN, ug, activation, d,
    modelLayerNorm, sample, global_output
]

# -------------------------------------------------------------------
# Example: Check dependencies for each operation and print prerequisite info
for operation in operation_list:
    operation.checkDep()

for operation in operation_list:
    print(f"Operation: {operation.name}, dep = {[str(dep) for dep in operation.prerequisites]}")

# -------------------------------------------------------------------
# Set model parameters for all operations

num_kv_heads = 8
num_qo_heads = 32
kqv_heads = num_qo_heads + 2 * num_kv_heads
head_dim = 128
vocab_size = 128 * 1024
hidden_dim = 4096
intermediate_dim = 14 * 1024

gen_embedding.setShape(hidden_dim, vocab_size)
layerNormAttn.setShape(hidden_dim)
kqv.setShape(kqv_heads * head_dim, hidden_dim)
decAttn.setShape(num_kv_heads, num_qo_heads, head_dim)
ropeAppend.setShape(num_kv_heads, num_qo_heads, head_dim)
o.setShape(hidden_dim, hidden_dim)
layerNormFFN.setShape(hidden_dim)
ug.setShape(intermediate_dim * 2, hidden_dim)
d.setShape(hidden_dim, intermediate_dim)
activation.setShape(intermediate_dim)
pfAttn.setShape(num_kv_heads, num_qo_heads, head_dim)
modelLayerNorm.setShape(hidden_dim)
sample.setShape(vocab_size)


batch_size = 1024
gen_embedding.setBatchSize(batch_size)
layerNormAttn.setBatchSize(batch_size)
kqv.setBatchSize(batch_size)
decAttn.setBatchSize(batch_size)
ropeAppend.setBatchSize(batch_size)
layerNormFFN.setBatchSize(batch_size)
ug.setBatchSize(batch_size)
activation.setBatchSize(batch_size)
o.setBatchSize(batch_size)
d.setBatchSize(batch_size)
modelLayerNorm.setBatchSize(batch_size)
sample.setBatchSize(batch_size)
global_input.setBatchSize(batch_size)
global_output.setBatchSize(batch_size)



# -------------------------------------------------------------------
# Initialize output buffers before runtime
for operation in operation_list:
    operation.initOutputBuffer()

# -------------------------------------------------------------------
# (Further steps such as buffer initialization, scheduling, and run would follow here)
