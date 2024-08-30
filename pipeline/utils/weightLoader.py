import safetensors
import sys
sys.path.append('../build')
import pllm_python
import torch
import os
import tqdm
from transformers import LlamaTokenizer
import concurrent.futures
from typing import List

def to_vortex_weight(tensor):
    w = pllm_python.VortexWeight()
    if len(tensor.size()) == 2:
        w.N = tensor.size()[0]
        w.K = tensor.size()[1]
    else:
        w.N = 1
        w.K = tensor.size()[0]
    w.setPtr(tensor.data_ptr())
    return w

num_layers = 80

def load_single_weight(i, weight_path):
    original_tensors = torch.load(weight_path + 'weight_rank_{}.pt'.format(i))
    print("Loaded tensors from weight_rank_{}.pt".format(i))
    model_weight = pllm_python.VortexModelWeight()
    model_weight.model_layernorm = to_vortex_weight(original_tensors['ModelNorm'])
    model_weight.embedding = to_vortex_weight(original_tensors['Embed'])
    model_weight.lm_head = to_vortex_weight(original_tensors['LmHead'])
    layer_weights = []
    for l in range(num_layers):
        layer_weight = pllm_python.VortexLayerWeight()
        layer_weight.W_O1 = to_vortex_weight(original_tensors[f"O1_{l}"])
        layer_weight.W_O2 = to_vortex_weight(original_tensors[f"O2_{l}"])
        layer_weight.W_UG = to_vortex_weight(original_tensors[f"UG_{l}"])
        layer_weight.W_D = to_vortex_weight(original_tensors[f"D_{l}"])
        layer_weight.W_KQV = to_vortex_weight(original_tensors[f"KQV_{l}"])
        if l == 0:
            print(f"original_tensors[f'KQV_{l}']: ", original_tensors[f"KQV_{l}"])
        layer_weight.W_LN_Attention = to_vortex_weight(original_tensors[f"LNATT_{l}"])
        layer_weight.W_LN_FFN = to_vortex_weight(original_tensors[f"LNFFN_{l}"])
        layer_weights.append(layer_weight)
    model_weight.layer_weight = layer_weights
    return original_tensors, model_weight, i

def load_weights(nranks, vnranks, weight_path='./nanoflow_weight/'):
    tensor_saved = []
    model_weights = []
    temp_list = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_single_weight, i, weight_path) for i in range(nranks)]
        
        for future in concurrent.futures.as_completed(futures):
            original_tensors, model_weight, i = future.result()
            temp_list[i] = (original_tensors, model_weight)
    
    for i in range(nranks):
        tensor_saved.append(temp_list[i][0])
        model_weights.append(temp_list[i][1])
    # for i in range(nranks):
        # original_tensors, model_weight = load_single_weight(i, weight_path)
        # tensor_saved.append(original_tensors)
        # model_weights.append(model_weight)

    return tensor_saved, model_weights

if __name__ == "__main__":
    tensor_saved, model_weights = load_weights(8, 8)
