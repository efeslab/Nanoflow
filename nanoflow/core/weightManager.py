import torch, os
import tqdm
import os
import safetensors
import json
import time

USE_FAST_URING = False

try:
    import nanoflow.pybind.build.fast_uring as fast_uring
    USE_FAST_URING = True
except ImportError:
    print("fast_uring not found, using numpy instead")
    import numpy as np

class WeightManager():
    def __init__(self, cache_weight_name, cached_weight_dir, weight_path, cached, device):
        self.cache_weight_name = cache_weight_name
        self.cached = cached
        self.cached_weight_dir = cached_weight_dir
        self.weight_map = {}
        self.processed_weight_map = {}
        self.processed_weight_metadata = {}

        if cached:
            self.load_from_disk(device)
        if not cached:
            self.load_from_safe_tensor(weight_path)
    
    def load_from_safe_tensor(self, tensor_path):
        print("load from safe tensor")
        for file in tqdm.tqdm(os.listdir(tensor_path)):
            if file.endswith(".safetensors"):
                tensors = safetensors.safe_open(os.path.join(tensor_path, file), 'pt')
                for name in tensors.keys():
                    tensor = tensors.get_tensor(name)
                    self.weight_map[name] = tensor.half() # make all the tensor fp16
                    # print(f"load tensor {name} from {file} with shape {tensor.shape}")
    
    def load_from_disk(self, device):
        print("load weight from disk")
        meta_data = json.load(open(os.path.join(self.cached_weight_dir, f"{self.cache_weight_name}_{device}_metadata.json"), "r"))
        file = os.path.join(self.cached_weight_dir, f"{self.cache_weight_name}_{device}.bin")
        start_load_time = time.time()
        # use torch.load to load the tensor
        if USE_FAST_URING:
            ten = fast_uring.load_fp16(file, threads=32)
        else:
            ten = torch.from_numpy(np.fromfile(file, dtype=np.float16))
        t1 = time.time()
        mb_s = ten.numel()*2 / 1e6 / (t1 - start_load_time)
        print(f"Weight takes {ten.numel() * 2 / 1024 / 1024 / 1024:.2f} GB")
        print(f"Loaded {mb_s:,.1f} MB/s with {ten.numel():,} elements")
        
        start_load_to_device_time = time.time()
        ten = ten.to(device, non_blocking=True)
        print(f"load tensor to device time: {time.time() - start_load_to_device_time:.2f}s")

        for name, metadata in meta_data.items():
            offset = metadata["offset"]
            shape = metadata["shape"]
            dtype = metadata["dtype"]
            size = metadata["size"]
            # print(f"load tensor {name} with shape {shape} and offset {offset}")
            self.processed_weight_map[name] = ten[offset:offset + size].view(shape)
        
        print(f"load weight time: {time.time() - start_load_time:.2f}s")
    
    def set_weight(self, operation_list, device):
        print("set weight start")
        start_time = time.time()
        for op in operation_list:
            op.processWeight(self.weight_map, self.processed_weight_map, cached=self.cached, device=device)
        if not self.cached:
            print("save weight to disk")
            total_el = 0
            offsets = []
            for weight_name, weight_tensor in self.processed_weight_map.items():
                # print(f"weight name: {weight_name}, shape: {weight_tensor.shape}, dtype: {weight_tensor.dtype}, size: {weight_tensor.numel()}")
                t = weight_tensor.contiguous()
                
                offsets.append(total_el)
                self.processed_weight_metadata[weight_name] = {
                    "offset": total_el,
                    "shape": t.shape,
                    "dtype": str(t.dtype),
                    "size": t.numel(),
                }
                total_el += t.numel()
            # print("creating flat tensor")
            flat = torch.empty(total_el, dtype=torch.float16)
            # print(f"flat tensor size: {flat.size()}, dtype: {flat.dtype}")
            for t, offset in zip(self.processed_weight_map.values(), offsets):
                flat[offset:offset + t.numel()].copy_(t.contiguous().view(-1))

            with open(os.path.join(self.cached_weight_dir, f"{self.cache_weight_name}_{device}.bin"), "wb") as f:
                f.write(flat.numpy().tobytes())

            json.dump(self.processed_weight_metadata, open(os.path.join(self.cached_weight_dir, f"{self.cache_weight_name}_{device}_metadata.json"), "w"))
            # breakpoint()
        
        print(f"set weight time: {time.time() - start_time:.2f}s")