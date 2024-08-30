import torch
import time 
import numpy as np
import pllm_python
from torch.profiler import profile, record_function, ProfilerActivity
from typing import List

def toGPU(data, nranks, dtype, non_blocking=True):
    
    with record_function("construct tensor"):
        # convert data to torch tensors
        data_tensor = torch.tensor(data, dtype=dtype).pin_memory()
    # print internal type of data_tensor
    # print(data_tensor.dtype)
    # move data to GPU
    with record_function("move data to GPU"):
        device_tensor_list = []
        for i in range(nranks):
            device_tensor_list.append(data_tensor.to(f'cuda:{i}', non_blocking=non_blocking))
        device_tensor_addresses = [device_tensor.data_ptr() for device_tensor in device_tensor_list]
    return device_tensor_addresses, device_tensor_list

def toGPUTensor(data, nranks, dtype, non_blocking=True):
    # move data to GPU
    with record_function("move data to GPU"):
        device_tensor_list = []
        for i in range(nranks):
            device_tensor_list.append(data.to(f'cuda:{i}', non_blocking=non_blocking))
        device_tensor_addresses = [device_tensor.data_ptr() for device_tensor in device_tensor_list]
    return device_tensor_addresses, device_tensor_list

def toGPUShard(data: torch.tensor, nranks, dtype, non_blocking=True):
    # shard data into nranks parts
    shard_size = len(data) // nranks
    device_tensor_list = []
    for i in range(nranks):
        device_tensor_list.append(data[i*shard_size:(i+1)*shard_size].to(f'cuda:{i}', non_blocking=non_blocking))
    
    device_tensor_addresses = [device_tensor.data_ptr() for device_tensor in device_tensor_list]
    return device_tensor_addresses, device_tensor_list

def genInitData(rank, weight):
    data = pllm_python.VortexInitData()
    hiddenDim = 8192
    temp = torch.randn([hiddenDim, hiddenDim * 4], dtype=torch.half, device=f'cuda:{rank}')
    data.setTmpBuffer(temp.data_ptr(), temp.numel() * temp.element_size())
    data.tmp_buffer_size = 5*1024*1024*1024
    data.setWeight(weight, 0)

    return data


import json

def save_config(config, filename):
    # Convert the VortexConfigData object to a dictionary
    config_dict = {
        "gemm_op_tag": config.gemm_op_tag,
        "global_batch_size": config.globalBatchSize,
        "nanobatch_1_size": config.nanobatch1Size,
        "kqv1_size": config.kqv1Size,
        "kqv3_size": config.kqv3Size
    }
    # Write the dictionary to a JSON file
    with open(filename, 'w') as json_file:
        json.dump(config_dict, json_file, indent=4)

def load_config(filename):
    # Read the JSON file into a dictionary
    with open(filename, 'r') as json_file:
        config_dict = json.load(json_file)
    # Create a VortexConfigData object and set its attributes
    config = pllm_python.VortexConfigData()
    config.setGemmOpTag(config_dict["gemm_op_tag"])
    config.globalBatchSize = config_dict["global_batch_size"]
    config.nanobatch1Size = config_dict["nanobatch_1_size"]
    config.kqv1Size = config_dict["kqv1_size"]
    config.kqv3Size = config_dict["kqv3_size"]
    return config

def initUpdateData(
        nranks : int,
        decodePrefillBorder: int,
        prefillNum: int,
        input_embedding: int,
        input_indptr: int,
        kv_indicies: int,
        kv_indptr: int,
        kv_last_page_len: int,
        rev_input_indptr: int,
        per_token_offset: int,
        gemv_batch_size: List[int],
        gemv_block_num: List[int] ) -> List[pllm_python.VortexUpdateData]:
    updateDataList = []
    gemv_batch_size = np.array(gemv_batch_size, dtype=np.int32)
    gemv_block_num = np.array(gemv_block_num, dtype=np.int32)
    for i in range(nranks):
        updateData = pllm_python.VortexUpdateData()
        updateData.setGemvBatchSize(gemv_batch_size)
        updateData.setGemvNumBlocks(gemv_block_num)
        updateData.setInputEmbedding(input_embedding)
        updateData.setInputIndptr(input_indptr)
        updateData.setKVIndices(kv_indicies)
        updateData.setKVIndptr(kv_indptr)
        updateData.setKVLastPageLen(kv_last_page_len)
        updateData.decodePrefillBorder = decodePrefillBorder
        updateData.prefillNum = prefillNum
        updateData.setRevInputIndptr(rev_input_indptr)
        updateData.setPerTokenOffset(per_token_offset)
        updateDataList.append(updateData) # TODO: change rev_input and offset
    return updateDataList

if __name__ == "__main__":
    data = [i for i in range(1000)]
    nranks = 8
    
    t = time.time()
    toGPU(data, nranks, torch.int32)
    print("Time taken: ", time.time() - t)
    
    
    
