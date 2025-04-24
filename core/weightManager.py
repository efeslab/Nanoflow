import tqdm
import os
import safetensors
class WeightManager():
    def __init__(self, pipeline_name, num_devices, cached = False):
        self.pipeline_name = pipeline_name
        self.cached = cached
        self.cached_weight_path = f"../cached_weights/{self.pipeline_name}"
        self.weight_map = {}
        # create the filefolder "../cached_weights/pipeline_name" if not exist
        if not cached:
            #remove the folder if it exists
            if os.path.exists(self.cached_weight_path):
                import shutil
                shutil.rmtree(self.cached_weight_path)
            os.makedirs(self.cached_weight_path)
            for i in range(num_devices):
                os.makedirs(f"{self.cached_weight_path}/device_{i}", exist_ok=True)
    
    def load_from_safe_tensor(self, tensor_path):
        for file in tqdm.tqdm(os.listdir(tensor_path)):
            if file.endswith(".safetensors"):
                tensors = safetensors.safe_open(os.path.join(tensor_path, file), 'pt')
                for name in tensors.keys():
                    tensor = tensors.get_tensor(name)
                    self.weight_map[name] = tensor.half() # make all the tensor fp16
    
    def set_weight(self, operation_list):
        for op in operation_list:
            op.processWeight(self.weight_map, self.cached_weight_path, cached=self.cached)
