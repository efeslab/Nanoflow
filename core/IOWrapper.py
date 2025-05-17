import torch

# To Do: delete the attributes that related to tensor that should not belong to a base IOWrapper anymore
class IOWrapper:
    def __init__(self, owner, name, device, dtype=torch.float16):
        self.owner = owner  # owner is now an Operations object or similar
        self.name = name
        self.device = device
        self.prev = []
        self.next = []
        self.prev_depend_on_prev_layer = []
        self.prev_depend_on_next_layer = []
        self.nano_dist_prev = []
        self.nano_dist_next = []
        self.nano_dist_prev_depend_on_prev_layer = []
        self.nano_dist_prev_depend_on_next_layer = []

        self.ptr = 0
        self.transform = None
        self.dtype = dtype
        self.tensor_shape = None # shape [0] is non-contiguous dimension, shape [1] is contiguous dimension
        self.batch_size = None
        self.tensor_offset = 0
        self.whole_buffer: torch.Tensor = None  
        self.is_input_wrapper = None
        self.is_output_wrapper = None
    
    @property
    def fullName(self):
        owner_name = self.owner.name if hasattr(self.owner, "name") else str(self.owner)
        return f"{owner_name}_{self.name}"
    
    def chain(self, next_wrapper, depend_on_prev, depend_on_next):
        self.next.append(next_wrapper) if next_wrapper not in self.next else None # self.next prepared for memory allocation
        next_wrapper.prev.append(self) # self.prev prepared for executor graph
        next_wrapper.prev_depend_on_prev_layer.append(depend_on_prev)
        next_wrapper.prev_depend_on_next_layer.append(depend_on_next)
        # check dtype must be the same
        if self.dtype != next_wrapper.dtype:
            raise Exception(f"Error: {self.fullName} and {next_wrapper.fullName} has different dtype")
        
        return next_wrapper

    def __rshift__(self, next_wrapper):
        depend_on_prev = False
        depend_on_next = False
        if isinstance(next_wrapper, tuple):
            if len(next_wrapper) == 2:
                next_wrapper, depend_on_prev = next_wrapper
            elif len(next_wrapper) == 3:
                next_wrapper, depend_on_prev, depend_on_next = next_wrapper
        # print("IOWrapper __rshift__", depend_on_prev)
        return self.chain(next_wrapper, depend_on_prev, depend_on_next)
    
    def toStr(self):
        # name, prev = [], next = []
        return f"{self.fullName}, prev = {[p.fullName for p in self.prev]}, next = {[n.fullName for n in self.next]}"

    def set_whole_buffer(self, buffer):
        self.whole_buffer = buffer

    def set_tensor_offset(self, offset):
        self.tensor_offset = offset

    def init_shape(self, shape):
        self.tensor_shape = shape

    def is_input(self):
        self.is_input_wrapper = True
        return self

    def is_output(self):
        self.is_output_wrapper = True
        return self
    
    def is_intersect(self, other: "IOWrapper"):
        if not torch.equal(self.whole_buffer, other.whole_buffer):
            return False
        if self.batch_size == 0 or other.batch_size == 0:
            return False
        return not (self.tensor_offset + self.batch_size <= other.tensor_offset or self.tensor_offset >= other.tensor_offset + other.batch_size)

    @property
    def shape(self):
        if self.batch_size is None or self.tensor_shape is None:
            return None
        return (self.batch_size, *self.tensor_shape[1:])

    @property
    def tensor(self):
        if self.whole_buffer is None:
            return None
        return self.whole_buffer[self.tensor_offset : self.tensor_offset + self.batch_size]

    @property
    def actual_next(self):
        # print("base_wrapper.name", self.base_wrapper.name)
        # print("base_wrapper.owner.name", self.base_wrapper.owner.name)
        # for io_base in self.base_wrapper.next + self.base_wrapper.nano_dist_next:
        #     print("io_base.owner.name", io_base.owner.name)
        #     print("io_base.fullName", io_base.fullName)
        #     print("io_base.children", io_base.children)
        return [io for io in self.next + self.nano_dist_next]
    
    @property
    def actual_prev(self):
        
        return [io for io in self.prev + self.nano_dist_prev]
    
    @property
    def actual_prev_depend_on_prev_layer(self):
        return self.prev_depend_on_prev_layer + self.nano_dist_prev_depend_on_prev_layer

    @property
    def actual_prev_depend_on_next_layer(self):
        return self.prev_depend_on_next_layer + self.nano_dist_prev_depend_on_next_layer