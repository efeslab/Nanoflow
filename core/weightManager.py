import tqdm
import os
import safetensors
class WeightManager():
    def load_tensors(tensor_path):
        original_tensors ={}
        for file in tqdm.tqdm(os.listdir(tensor_path)):
            if file.endswith(".safetensors"):
                tensors = safetensors.safe_open(os.path.join(tensor_path, file), 'pt')
                for name in tensors.keys():
                    tensor = tensors.get_tensor(name)
                    original_tensors[name] = tensor
        return original_tensors


    def __init__(self):
        pass
    
    def load_from_safe_tensor(self, tensor_path, num_devices):
        self.weight_map = WeightManager.load_tensors(tensor_path)
        self.weights_per_device = [{} for _ in range(num_devices)]
        for device_id in range(num_devices):
            # make all the tensor fp16
            for key in self.weight_map.keys():
                # self.weight_map[key] = self.weight_map[key].half().to('cuda')
                self.weights_per_device[device_id][key] = self.weight_map[key].half().to(f'cuda:{device_id}')
    
    def set_weight(self, operation_list, total_devices, total_layers):
        for op in operation_list:
            op.processWeight(self.weights_per_device, total_devices, total_layers)
