import sys
sys.path.append('../build')
sys.path.append('../utils')
import pllm_python
import torch 
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import subprocess
from pathlib import Path
import time
import json
from weightLoader import load_weights


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

def genVortexConfig(rank):
    gemm_op_tag = ["128_128_32_64_64_32_1_5_ColumnMajor_RowMajor_ColumnMajor",
    "128_128_32_64_64_32_1_4_ColumnMajor_RowMajor_ColumnMajor",
    "128_128_32_64_64_32_3_4_ColumnMajor_RowMajor_RowMajor",
    "128_128_32_64_64_32_1_5_ColumnMajor_RowMajor_RowMajor",
    "128_256_32_64_64_32_1_3_ColumnMajor_RowMajor_RowMajor",
    "128_128_32_64_64_32_1_4_ColumnMajor_RowMajor_RowMajor",
    "128_64_64_64_32_64_2_3_RowMajor_RowMajor_RowMajor",
    "128_128_32_64_64_32_2_5_RowMajor_RowMajor_RowMajor",
    "128_128_32_64_64_32_1_5_RowMajor_RowMajor_RowMajor",
    "128_128_32_64_64_32_1_5_RowMajor_RowMajor_RowMajor",
    "128_128_32_64_64_32_2_4_ColumnMajor_RowMajor_RowMajor"]
    global_batch_size = 2048
    nanobatch_1_size = 640
    kqv1_size = 256
    kqv3_size = 640
    
    config = pllm_python.VortexConfigData()
    config.setGemmOpTag(gemm_op_tag)
    config.globalBatchSize = global_batch_size
    config.nanobatch1Size = nanobatch_1_size
    config.kqv1Size = kqv1_size
    config.kqv3Size = kqv3_size
    return config

def genInitData(rank):
    data = pllm_python.VortexInitData()

    # input
    hiddenDim = 8192

    # init gemv
    head_dim = 128
    seqlen = 1024
    batch_size = 1350
    page_size = 16
    num_kv_heads = 1


    pages_per_seq = (seqlen + page_size - 1) // page_size
    num_pages = pages_per_seq * batch_size

    layer_num=5



    temp = torch.randn([hiddenDim, hiddenDim * 4], dtype=torch.half, device=f'cuda:{rank}')
    temp2 = torch.randn([hiddenDim, hiddenDim * 4], dtype=torch.half, device=f'cuda:{rank}')


    data.setTmpBuffer(temp.data_ptr(), temp.numel() * temp.element_size())
    data.tmp_buffer_size = 5*1024*1024*1024
    data.weight_size = 5*1024*1024*1024
    return data

def genUpdateData(rank):
    data = pllm_python.VortexUpdateData()

    # input

    decodePrefillBorder = 0
    denseBatch = 2048
    hiddenDim = 8192

    input_embedding = torch.ones(denseBatch, dtype=torch.int32, device='cpu')
    input_embedding[1] = 6324
    input_embedding[2] = 29892
    input_embedding[3] = 29871
    
    for i in range(4, denseBatch):
        input_embedding[i] = input_embedding[i%4]

    input_indptr_cpu = [0]
    for i in range(decodePrefillBorder):
        input_indptr_cpu.append(input_indptr_cpu[-1] + 1)

    # init gemv

    seqlen = 1024
    page_size = 16
    
    
    prefill_num = denseBatch // 4
    prefill_tokens = [4] * int(prefill_num)
    prefill_pages = [(i + page_size - 1) // page_size for i in prefill_tokens] # important!!!!
    total_prefill_pages = sum(prefill_pages)

    
    for i in range(prefill_num):
        input_indptr_cpu.append(input_indptr_cpu[-1] + prefill_tokens[i])
        
    input_indptr = torch.tensor(input_indptr_cpu, dtype=torch.int32, device='cpu')   
    print ("input_indptr", input_indptr)
    
    rev_input_indptr_cpu = []
    for i in range(decodePrefillBorder+prefill_num):
        for j in range(input_indptr_cpu[i], input_indptr_cpu[i+1]):
            rev_input_indptr_cpu.append(i)
    rev_input_indptr = torch.tensor(rev_input_indptr_cpu, dtype=torch.int32, device='cpu')
    print ("rev_input_indptr", rev_input_indptr)
    
    per_token_offset_cpu = []
    for i in range(decodePrefillBorder+prefill_num):
        for j in range(input_indptr_cpu[i], input_indptr_cpu[i+1]):
            per_token_offset_cpu.append(j-input_indptr_cpu[i])
    per_token_offset = torch.tensor(per_token_offset_cpu, dtype=torch.int32, device='cpu')
    print ("per_token_offset", per_token_offset)

    pages_per_seq = (seqlen + page_size - 1) // page_size
    num_pages = pages_per_seq * decodePrefillBorder + total_prefill_pages

    kv_indicies_host = torch.zeros(num_pages, dtype=torch.int32, device='cpu')
    kv_indptr_host = torch.zeros(decodePrefillBorder+prefill_num +1, dtype=torch.int32, device='cpu')
    kv_last_page_len_host = torch.zeros(decodePrefillBorder+prefill_num, dtype=torch.int32, device='cpu')

    for i in range(decodePrefillBorder):
        for p in range(pages_per_seq):
            kv_indicies_host[i * pages_per_seq + p] = i * pages_per_seq + p
        kv_indptr_host[i+1] = (i+1) * pages_per_seq
        kv_last_page_len_host[i] = (seqlen-1) % page_size+1
        
        
    # set prefill requests
    last_len_indecies = decodePrefillBorder * pages_per_seq
    for i in range(prefill_num):
        for p in range(prefill_pages[i]):
            kv_indicies_host[last_len_indecies + p] = last_len_indecies + p
        last_len_indecies += prefill_pages[i]
        kv_indptr_host[decodePrefillBorder + i + 1] = kv_indptr_host[decodePrefillBorder + i] + prefill_pages[i]
        kv_last_page_len_host[decodePrefillBorder + i] = (prefill_tokens[i]-1) % page_size+1
    
    kv_indicies = kv_indicies_host
    kv_indptr = kv_indptr_host
    kv_last_page_len = kv_last_page_len_host
    
    print("kv_indicies", kv_indicies)
    print("kv_indptr", kv_indptr)
    print("kv_last_page_len", kv_last_page_len)

    gemv_batch_size = np.array([0,0,0,0], dtype=np.int32)
    gemv_block_num = np.array([108,12,10,48], dtype=np.int32)
    data.setInputEmbedding(input_embedding.data_ptr())
    data.setInputIndptr(input_indptr.data_ptr())
    data.setKVIndices(kv_indicies.data_ptr())
    data.setKVIndptr(kv_indptr.data_ptr())
    data.setKVLastPageLen(kv_last_page_len.data_ptr())
    data.setGemvBatchSize(gemv_batch_size)
    data.setGemvNumBlocks(gemv_block_num)
    data.setRevInputIndptr(rev_input_indptr.data_ptr())
    data.setPerTokenOffset(per_token_offset.data_ptr())
    data.decodePrefillBorder = decodePrefillBorder
    data.prefillNum = prefill_num
    
    result_save = (gemv_batch_size, gemv_block_num,input_embedding, input_indptr, kv_indicies, kv_indptr, kv_last_page_len, rev_input_indptr, per_token_offset)

    return data, result_save


nranks = torch.cuda.device_count()
pllm_python.setRank(nranks,8)

data_array = []
config_array = []
update_array = []
save_data = []
tensor_saved, model_weights = load_weights(8,8, '../utils/nanoflow_weight/')
for i in range(nranks):
    init_data = genInitData(i)
    init_data.setWeight(model_weights[i],123)
    
    data_array.append(init_data)
    config_array.append(load_config("../config/2048.json"))
    (data, result_save) = genUpdateData(i)
    update_array.append(data)
    save_data.append(result_save)
# exit()

if False:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, use_cuda=True) as prof:
        with record_function("model_inference"):
            pllm_python.init(data_array)
            out = pllm_python.run()
            out = pllm_python.run()
            out = pllm_python.run()
            out = pllm_python.run()
            out = pllm_python.run()
            out = pllm_python.run()
            out = pllm_python.run()
            pllm_python.finalize()

    prof.export_chrome_trace("trace.json")
else:
    pllm_python.init(data_array, pllm_python.PipelineType.PLLM)
    pllm_python.config(config_array)
    pllm_python.update(update_array)
    for i in range(3):
        t = time.time()
        out = pllm_python.run()
        sampled_token_array = pllm_python.getPipelineOutput()
        for i in sampled_token_array:
            print(i)
        print(f"-----------------time {time.time()-t}-----------------")
    pllm_python.finalize()
    
