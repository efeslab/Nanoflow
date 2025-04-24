import torch
import tqdm
import os
import safetensors
class WeightManager():
    def __init__(self, pipeline_name, num_devices, cached = False):
        self.pipeline_name = pipeline_name
        self.cached = cached
        self.cached_weight_path = f"../cached_weights"
        self.weight_map = {}
        self.cached_weight_map = {}
        # create the filefolder "../cached_weights/pipeline_name" if not exist
        if cached:
            self.cached_weight_map = torch.load(os.path.join(self.cached_weight_path, f"{self.pipeline_name}.pt"))
        else:
            os.makedirs(self.cached_weight_path, exist_ok=True)
    
    def load_from_safe_tensor(self, tensor_path):
        for file in tqdm.tqdm(os.listdir(tensor_path)):
            if file.endswith(".safetensors"):
                tensors = safetensors.safe_open(os.path.join(tensor_path, file), 'pt')
                for name in tensors.keys():
                    tensor = tensors.get_tensor(name)
                    self.weight_map[name] = tensor.half() # make all the tensor fp16
    
    def set_weight(self, operation_list):
        for op in operation_list:
            op.processWeight(self.weight_map, self.cached_weight_map, cached=self.cached)
        if not self.cached:
            torch.save(self.cached_weight_map, os.path.join(self.cached_weight_path, f"{self.pipeline_name}.pt"))