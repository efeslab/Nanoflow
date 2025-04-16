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
    
    def load_from_safe_tensor(self, tensor_path, num_devices, tp_size=1):
        self.weight_map = WeightManager.load_tensors(tensor_path)
        self.weights_per_device = [{} for _ in range(num_devices)]
        for device_id in range(num_devices):
            offset = device_id % tp_size
            # make all the tensor fp16
            for key in self.weight_map.keys():
                # if the key contains "self_attn", the key is a string.
                if "o_proj" in key or "down_proj" in key:
                    assert self.weight_map[key].shape[0] % tp_size == 0, f"key: {key}, shape: {self.weight_map[key].shape}, tp_size: {tp_size}"
                    stride = self.weight_map[key].shape[0] // tp_size
                    start = offset * stride
                    end = start + stride
                    self.weights_per_device[device_id][key] = self.weight_map[key][start:end].half().to(f'cuda:{device_id}')
                elif "embed_tokens" in key:
                    assert self.weight_map[key].shape[1] % tp_size == 0, f"key: {key}, shape: {self.weight_map[key].shape}, tp_size: {tp_size}"
                    stride = self.weight_map[key].shape[1] // tp_size
                    start = offset * stride
                    end = start + stride
                    self.weights_per_device[device_id][key] = self.weight_map[key][:, start:end].half().to(f'cuda:{device_id}')
                else:
                    # self.weight_map[key] = self.weight_map[key].half().to('cuda')
                    self.weights_per_device[device_id][key] = self.weight_map[key].half().to(f'cuda:{device_id}')
    
    def set_weight(self, operation_list, total_devices, total_layers):
        for op in operation_list:
            op.processWeight(self.weights_per_device, total_devices, total_layers)
