import safetensors
import sys
sys.path.append('../build')
import pllm_python
import torch
import os
import tqdm
import time
from transformers import LlamaTokenizer
import concurrent.futures
import re

def to_vortex_weight(tensor):
    w = pllm_python.VortexWeight()
    if len(tensor.size()) == 2:
        w.N = tensor.size()[0]
        w.K = tensor.size()[1]
    else:
        w.N = 1
        w.K = tensor.size()[0]
    w.setPtr(tensor.data_ptr())
    # print(f"load weight of size {w.N}x{w.K}")
    return w



def find_max_i_from_dict(d):
    # Regex pattern to match keys in the format 'O1_i'
    pattern = r'^D_(\d+)$'
    
    # Extract the integers from the matching keys
    integers = [int(re.match(pattern, key).group(1)) for key in d.keys() if re.match(pattern, key)]
    
    # Return the maximum integer found, or None if no match
    return max(integers, default=None)

def load_single_weight(i, weight_path):
    print("Loading tensors from weight_rank_{}.pt".format(i))
    original_tensors = torch.load(weight_path + 'weight_rank_{}.pt'.format(i))
    print("original_tensor data type: ", original_tensors["O1_0"].dtype)
    num_layers = find_max_i_from_dict(original_tensors) + 1
    print(f"layers = {num_layers}")
    for key in original_tensors:
        if isinstance(original_tensors[key], torch.Tensor):
            original_tensors[key] = original_tensors[key].to(torch.float16)
    model_weight = pllm_python.VortexModelWeight()
    model_weight.model_layernorm = to_vortex_weight(original_tensors['ModelNorm'])
    model_weight.embedding = to_vortex_weight(original_tensors['Embed'])
    model_weight.lm_head = to_vortex_weight(original_tensors['LmHead'])
    layer_weights = []
    for l in range(num_layers):
        layer_weight = pllm_python.VortexLayerWeight()
        layer_weight.W_O1 = to_vortex_weight(original_tensors[f"O1_{l}"])
        layer_weight.W_O2 = to_vortex_weight(original_tensors[f"O2_{l}"])
        layer_weight.W_U = to_vortex_weight(original_tensors[f"U_{l}"])
        layer_weight.W_G = to_vortex_weight(original_tensors[f"G_{l}"])
        layer_weight.W_D = to_vortex_weight(original_tensors[f"D_{l}"])
        layer_weight.W_KQV = to_vortex_weight(original_tensors[f"KQV_{l}"])        
        layer_weight.W_LN_Attention = to_vortex_weight(original_tensors[f"LNATT_{l}"])
        layer_weight.W_LN_FFN = to_vortex_weight(original_tensors[f"LNFFN_{l}"])
        if f"BKQV_{l}" in original_tensors: # compatibility for 8B models (local pipeline)
            # print(f"Loading BKQV_{l}")
            layer_weight.B_KQV = to_vortex_weight(original_tensors[f"BKQV_{l}"])
        layer_weights.append(layer_weight)
    model_weight.layer_weight = layer_weights
    return original_tensors, model_weight, i

def load_weights(nranks, vnranks, weight_path='./nanoflow_weight_8B/'):
    tensor_saved = []
    model_weights = []
    temp_list = {}
    print("in load_weights, before executor")
    start_time = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_single_weight, i, weight_path) for i in range(nranks)]
        
        for future in concurrent.futures.as_completed(futures):
            original_tensors, model_weight, i = future.result()
            temp_list[i] = (original_tensors, model_weight)
    end_time = time.perf_counter()
    print(f"Time taken to load weights: {end_time - start_time}")
    print("in load_weights, after executor")

    for i in range(nranks):
        tensor_saved.append(temp_list[i][0])
        model_weights.append(temp_list[i][1])
    # for i in range(nranks):
        # original_tensors, model_weight = load_single_weight(i, weight_path)
        # tensor_saved.append(original_tensors)
        # model_weights.append(model_weight)

    return tensor_saved, model_weights

if __name__ == "__main__":
    tensor_saved, model_weights = load_weights(1,1)
