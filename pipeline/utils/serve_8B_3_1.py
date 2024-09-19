import sys
sys.path.append('../build')

import pllm_python
import os
import argparse
import time
import torch
from pybindUtil import toGPU, toGPUShard, initUpdateData, genInitData, load_config

from torch.profiler import profile, record_function, ProfilerActivity
from frontend import requestManager

import pickle
from collections import deque

from kv_cache import DistKVPool, DistKVCache, BatchedDistKVCache
from request_info import NewRequestInfo, NewRequestQueue, FlyRequestInfo
import numpy as np
from enum import Enum

pipetype = pllm_python.PipelineType.LOCAL
torch.set_printoptions(threshold=4000)
from weightLoader import load_weights
import logging
os.environ['HF_HOME'] = '../../../hf'
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaModel, LlamaForCausalLM

logging.basicConfig(level=logging.INFO, filename='serve.log', filemode='w')
# make logging only output the message without any prefix
logging.basicConfig(format='%(message)s')

class isRunning(Enum):
    STATE_SKIP_CYCLE = 0
    STATE_FIRST_CYCLE = 2
    STATE_RUNNING = 3

def log_tensor(tensor, name):
    logging.info(f"{name}: {tensor.size()}")
    logging.info(f"{tensor}")


class WorkingSet:
    """
    Wrapper class for denoting a working set of requests.
    Mainly used for calculating effective batch size.
    """
    
    def __init__(self) -> None:
        self._set : list[FlyRequestInfo] = []
        
    def put(self, req: FlyRequestInfo):
        self._set.append(req)
        
    def __getitem__(self, key):
        return self._set[key]
    
    @property
    def size(self) -> int:
        return len(self._set)
    
    @property
    def effective_bsz(self) -> int:
        if self.size == 0:
            return 0
        else:
            # sum of input lengths
            return sum([len(req.input) for req in self._set])
    
    def adjust_kv_cache(self) -> None:
        """
        Before each iteration, call this function to adjust 
        the KV-Cache metadata according to the len(req.input).
        """
        for req in self._set:
            req.kv_cache.allocate_tokens(len(req.input))

class Scheduler:
    def __init__(self, memory_pool: DistKVPool, request_queue: NewRequestQueue, weight: list[pllm_python.VortexModelWeight], decode_length, prefill_length):
        self._memory_pool = memory_pool
        self._request_queue = request_queue
        
        self.average_decode_length = decode_length
        self.average_prefill_length = prefill_length
        
        self._decode_workset = WorkingSet()
        self._prefill_workset = WorkingSet()
        
        self._new_decode_workset = WorkingSet()
        self._new_prefill_workset = WorkingSet()
        
        self._gemv_batch_size = [0, 0, 0, 0]
        self._gemv_num_blocks = [108, 108, 108, 108]
        
        self._nranks = memory_pool.num_devices
        self._gpu_tensors = []
        self._new_gpu_tensors = []
        self._is_running = isRunning.STATE_SKIP_CYCLE
        self._record_schedule_stat = []
        self._weight = weight
        self.pinned_keep_token_list = torch.zeros([1], dtype=torch.int32, device='cpu')
        self._old_token_length = 0
        self._old_prefill = WorkingSet()
        self._old_decode = WorkingSet()
        
    
    def init_pipe(self, filename: str):
        pllm_python.setModelConfig("/code/pllm/compute-bound/modelConfig/llama3.1-8B.json")
        pllm_python.setRank(self._nranks,1)
        data_array = []
        config_array = []
        for i in range(self._nranks):
            data_array.append(genInitData(i, self._weight[i]))
        pllm_python.init(data_array, pipetype)

        for i in range(self._nranks):
            config_array.append(load_config(filename))
        pllm_python.config(config_array)
        self.config_data = config_array[0]

        self._global_bsz = config_array[0].globalBatchSize
        self.pinned_input_embedding = torch.zeros([2048], dtype=torch.int32, device='cpu')

    def check_future_min_free_pages(self):
        total_pages = self._memory_pool.capacity
        max_seq_len = (self.average_decode_length + self.average_prefill_length + 2)
        # print("max_seq_len", max_seq_len)
        future_page_usage = np.zeros(max_seq_len, dtype=int)
        
        # Create an index array that represents the addition pattern
        index_range = np.arange(max_seq_len, dtype=int)
        
        # Accumulate the values using vectorized operations
        for req in self._decode_workset._set:
            current_seq_len = req.kv_cache.seqlen + 1
            future_page_usage[0:max_seq_len-current_seq_len] += \
                (index_range[:max_seq_len-current_seq_len] + current_seq_len + self._memory_pool.page_size - 1) \
                // self._memory_pool.page_size
        for req in self._prefill_workset._set:
            # logging.info(f"prefill req in future estimation")
            current_seq_len = self.average_prefill_length + 1
            future_page_usage[0:max_seq_len-current_seq_len] += \
                (index_range[:max_seq_len-current_seq_len] + current_seq_len + self._memory_pool.page_size - 1) \
                // self._memory_pool.page_size
        # find the maximum page usage
        # logging.info(f"future_page_usage: {future_page_usage}")
        max_page_usage = np.max(future_page_usage)
        # find available pages
        available_pages = total_pages - max_page_usage
        return available_pages
    
    def check_future_page_usage(self):
        total_pages = self._memory_pool.capacity
        max_seq_len = (self.average_decode_length + self.average_prefill_length + 2)
        # print("max_seq_len", max_seq_len)
        future_page_usage = np.zeros(max_seq_len, dtype=int)
        
        # Create an index array that represents the addition pattern
        index_range = np.arange(max_seq_len, dtype=int)
        
        # Accumulate the values using vectorized operations
        for req in self._decode_workset._set:
            current_seq_len = req.kv_cache.seqlen + 1
            future_page_usage[0:max_seq_len-current_seq_len] += \
                (index_range[:max_seq_len-current_seq_len] + current_seq_len + self._memory_pool.page_size - 1) \
                // self._memory_pool.page_size
        for req in self._prefill_workset._set:
            # logging.info(f"prefill req in future estimation")
            current_seq_len = self.average_prefill_length + 1
            future_page_usage[0:max_seq_len-current_seq_len] += \
                (index_range[:max_seq_len-current_seq_len] + current_seq_len + self._memory_pool.page_size - 1) \
                // self._memory_pool.page_size
        # find the maximum page usage
        # logging.info(f"future_page_usage: {future_page_usage}")
        max_page_usage = np.max(future_page_usage)
        # find available pages
        available_pages = total_pages - max_page_usage
        return future_page_usage

    # def update_future_page_usage(self, req: FlyRequestInfo):
    #     total_pages = self._memory_pool.capacity
    #     max_seq_len = (self.average_decode_length + self.average_prefill_length + 2)


    
        
    def schedule_req(self):
        with record_function("check_new_request"):
            # logging.info(f"Decode workset size: {self._decode_workset.size}")
            # logging.info(f"Prefill workset size: {self._prefill_workset.size}")
            # logging.info(f"Decode start tokens: {self._decode_workset.effective_bsz}")
            # logging.info(f"Prefill start tokens: {self._prefill_workset.effective_bsz}")
            # decode_idle_tokens = self._decode_bsz - self._decode_workset.effective_bsz
            decode_idle_tokens = self._global_bsz - self._decode_workset.effective_bsz
            # prefill_idle_tokens = self._prefill_bsz - self._prefill_workset.effective_bsz
            prefill_idle_tokens = self._global_bsz - self._prefill_workset.effective_bsz - self._decode_workset.effective_bsz
            # print(prefill_idle_tokens, self._prefill_bsz, self._prefill_workset.effective_bsz)
            # min_avail_pages = self.check_future_min_free_pages()
            future_page_usage = self.check_future_page_usage()
            new_min_avail_pages = self._memory_pool.capacity - np.max(future_page_usage)

            # current_page_usage = 0
            # for req in self._decode_workset._set + self._prefill_workset._set:
            #     current_page_usage += (req.kv_cache.seqlen + self._memory_pool.page_size - 1) // self._memory_pool.page_size
            # logging.info(f"current_page_usage before adjust: {current_page_usage}")
            # logging.info(f"Min available pages before adjust: {min_avail_pages}")
            while prefill_idle_tokens > 0 and decode_idle_tokens > 0 and self._request_queue.size > 0 \
                and new_min_avail_pages > (self.average_prefill_length + self.average_decode_length + self._memory_pool.page_size) // self._memory_pool.page_size:
                # Check whether there is new request
                if self._request_queue.size > 0:
                    # logging.info("add new request")
                    new_req = self._request_queue.get()
                    new_kv_cache = DistKVCache(self._memory_pool)
                    new_request_prompt_len = len(new_req.prompt)
                    # print("newreq len", new_request_prompt_len)
                    if new_request_prompt_len > prefill_idle_tokens:
                        # Chunked prefill
                        self._prefill_workset.put(FlyRequestInfo(
                            req_idx=new_req.req_idx,
                            input=new_req.prompt[:prefill_idle_tokens],
                            output=[],
                            prompt=new_req.prompt,
                            request_comein_time=new_req.start_time,
                            chunked_prefill=True,
                            kv_cache=new_kv_cache,
                            encode_latency=0.0,
                            decode_start_at=0.0,
                            decode_latency=0.0,
                            output_len=new_req.output_len,
                            input_len=new_request_prompt_len
                        ))
                        logging.info(f"Chunked prefill request: {new_req.req_idx}")
                        
                        # Computed in next iteration
                        self._new_prefill_workset.put(FlyRequestInfo(
                            req_idx=new_req.req_idx,
                            input=new_req.prompt[prefill_idle_tokens:],
                            output=[],
                            prompt=new_req.prompt,
                            request_comein_time=new_req.start_time,
                            chunked_prefill=False,
                            kv_cache=new_kv_cache,
                            encode_latency=0.0,
                            decode_start_at=0.0,
                            decode_latency=0.0,
                            output_len=new_req.output_len,
                            input_len=new_request_prompt_len
                        ))
                        prefill_idle_tokens = 0
                        decode_idle_tokens -= 1
                    else:
                        # Do not need be chunked
                        self._prefill_workset.put(FlyRequestInfo(
                            req_idx=new_req.req_idx,
                            input=new_req.prompt,
                            output=[],
                            prompt=new_req.prompt,
                            request_comein_time=new_req.start_time,
                            chunked_prefill=False,
                            kv_cache=new_kv_cache,
                            encode_latency=0.0,
                            decode_start_at=0.0,
                            decode_latency=0.0,
                            output_len=new_req.output_len,
                            input_len=new_request_prompt_len
                        ))
                        # print("newreq", new_req.start_time)
                        prefill_idle_tokens -= new_request_prompt_len
                        decode_idle_tokens -= 1
                # min_avail_pages = self.check_future_min_free_pages()
                # print("min avail pages", min_avail_pages)
                # update future page usage for the new request
                current_seq_len = self.average_prefill_length + 1
                max_seq_len = (self.average_decode_length + self.average_prefill_length + 2)
                index_range = np.arange(max_seq_len, dtype=int)
                future_page_usage[0:max_seq_len-current_seq_len] += \
                    (index_range[:max_seq_len-current_seq_len] + current_seq_len + self._memory_pool.page_size - 1) \
                    // self._memory_pool.page_size
                new_min_avail_pages = self._memory_pool.capacity - np.max(future_page_usage)
                # print("new min avail pages", new_min_avail_pages)
                # assert new_min_avail_pages == min_avail_pages, "Future page usage not updated correctly."

            if prefill_idle_tokens < 0 :
                assert self._prefill_workset.size == 1, "At most add one large prefill each iteration."
                prefill_idle_tokens = - prefill_idle_tokens

                # Chunk the prefill request
                self._new_prefill_workset.put(FlyRequestInfo(
                    req_idx=self._prefill_workset[0].req_idx,
                    input=self._prefill_workset[0].input[-prefill_idle_tokens:],
                    output=[],
                    prompt=self._prefill_workset[0].prompt,
                    request_comein_time=self._prefill_workset[0].request_comein_time,
                    chunked_prefill=False,
                    kv_cache=self._prefill_workset[0].kv_cache,
                    encode_latency=0.0,
                    decode_start_at=0.0,
                    decode_latency=0.0,
                    output_len=self._prefill_workset[0].output_len,
                    input_len=self._prefill_workset[0].input_len
                ))
                self._prefill_workset[0].input = self._prefill_workset[0].input[:-prefill_idle_tokens]
                self._prefill_workset[0].chunked_prefill = True
            self._record_schedule_stat.append((time.perf_counter(), self._prefill_workset.effective_bsz, self._decode_workset.effective_bsz))
            # logging.info(f"Decode scheduled total tokens: {self._decode_workset.effective_bsz}")
            # logging.info(f"Prefill scheduled total tokens: {self._prefill_workset.effective_bsz}")
    
    def bench_text_gen(self, retired_rq, actualRun = True):
        self._new_decode_workset = WorkingSet()
        self._new_prefill_workset = WorkingSet()
        t1 = time.perf_counter()
        # Check whether it is possible to start a new request
        
        self.schedule_req()
        with record_function("input_embedding"):
            input_ids = []
            if actualRun:
                # with record_function("input_ids_decode"):
                #     for req in self._decode_workset._set:
                #         input_ids.extend(req.input)
                #     assert len(input_ids) == self._decode_workset.effective_bsz, "Decode Input length should be correct."
                # t_embedding_decode = time.perf_counter()
                # logging.info(f"embedding decode:  {t_embedding_decode - t_wait_async_end}")

                with record_function("input_ids_prefill"):
                    for req in self._prefill_workset._set:
                        input_ids.extend(req.input)
                assert len(input_ids) == self._prefill_workset.effective_bsz 

                with record_function("input_ids_tensor"):
                    self.pinned_input_embedding = torch.tensor(input_ids, dtype=torch.int32, device='cpu')
                
        with record_function("prepare data"):
            # prefill/decode workset should be ready to run at this time
            self._prefill_workset.adjust_kv_cache()
            self._decode_workset.adjust_kv_cache()
            
            input_ids : list[int] = []
            input_indptr : list[int] = [0]
            prev_len : list[int] = []
            decodePrefillBorder = self._decode_workset.effective_bsz
            decode_kvs : list[DistKVCache] = []
            prefill_kvs : list[DistKVCache] = []
            
        with record_function("calc batch size"):
            t3 = time.perf_counter()
            self._gemv_batch_size[0] = min(decodePrefillBorder, self.config_data.kqv1Size)
            self._gemv_batch_size[1] = min(decodePrefillBorder - self._gemv_batch_size[0], self.config_data.nanobatch1Size - self.config_data.kqv1Size)
            self._gemv_batch_size[2] = min(decodePrefillBorder - self._gemv_batch_size[0] - self._gemv_batch_size[1], self.config_data.kqv3Size)
            self._gemv_batch_size[3] = decodePrefillBorder - self._gemv_batch_size[0] - self._gemv_batch_size[1] - self._gemv_batch_size[2]
        with record_function("prepare KV"):
            for req in self._decode_workset._set:
                input_indptr.append(input_indptr[-1] + len(req.input))
                prev_len.append(req.kv_cache.seqlen - 1) # check, -1 since we already append 1 additional token
                decode_kvs.append(req.kv_cache)
            assert input_indptr[-1] == decodePrefillBorder, "Input indptr should be correct."
            
            for req in self._prefill_workset._set:
                input_indptr.append(input_indptr[-1] + len(req.input))
                prev_len.append(req.kv_cache.seqlen - len(req.input)) 
                prefill_kvs.append(req.kv_cache)
            # Prepare batched KV-Cache metadata
            batched_kv_cache = BatchedDistKVCache(decode_kvs, prefill_kvs)
            
        with record_function("batch_kv"):  
            if actualRun:
                [kv_indices, kv_indptr, kv_last_page_len] = batched_kv_cache.toCPUPinned()
                
        with record_function("input_indptr"):
            if actualRun:
                self.pinned_input_indptr = torch.tensor(input_indptr, dtype=torch.int32, device='cpu').pin_memory()

        with record_function("rev_input_indptr"):
            if actualRun:
                rev_input_indptr_cpu = []
                for i in range(len(input_indptr)-1):
                    for j in range(input_indptr[i], input_indptr[i+1]):
                        rev_input_indptr_cpu.append(i)                
                self.pinned_rev_input_indptr_cpu = torch.tensor(rev_input_indptr_cpu, dtype=torch.int32, device='cpu').pin_memory()

        with record_function("per_token_offset"):
            if actualRun:
                per_token_offset_cpu = []
                for i in range(len(input_indptr)-1):
                    for j in range(input_indptr[i], input_indptr[i+1]):
                        per_token_offset_cpu.append(j-input_indptr[i]+prev_len[i])              
                self.pinned_per_token_offset_cpu = torch.tensor(per_token_offset_cpu, dtype=torch.int32, device='cpu').pin_memory()

        with record_function("update_data"):
            if actualRun:
                updateDatas = initUpdateData(
                    nranks = self._nranks,
                    decodePrefillBorder=decodePrefillBorder,
                    prefillNum= len(prefill_kvs),
                    input_embedding=self.pinned_input_embedding.data_ptr(),
                    input_indptr=self.pinned_input_indptr.data_ptr(),
                    kv_indicies=kv_indices,
                    kv_indptr=kv_indptr,
                    kv_last_page_len=kv_last_page_len,
                    rev_input_indptr= self.pinned_rev_input_indptr_cpu.data_ptr(),
                    per_token_offset= self.pinned_per_token_offset_cpu.data_ptr(),
                    gemv_batch_size=self._gemv_batch_size,
                    gemv_block_num=self._gemv_num_blocks,
                    keep_token_list=self.pinned_keep_token_list.data_ptr(),
                    keep_token_list_length=self._old_token_length,
                    prefill_tokens_num= self._prefill_workset.effective_bsz,
                )
                # logging.info(f"decodePrefillBorder: {decodePrefillBorder}")
                # logging.info(f"prefillNum: {len(prefill_kvs)}")
                # logging.info(f"gemv_batch_size: {self._gemv_batch_size}")
                # logging.info(f"gemv_num_blocks: {self._gemv_num_blocks}")
                # logging.info(f"effective decode bsz: {self._decode_workset.effective_bsz}")
                # logging.info(f"effective prefill bsz: {self._prefill_workset.effective_bsz}")
                # logging.info(f"total bsz: {self._prefill_workset.effective_bsz + self._decode_workset.effective_bsz}")
                # log_tensor(self.pinned_input_embedding, "input_embedding")
                # log_tensor(self.pinned_input_indptr, "input_indptr")
                # log_tensor(self.pinned_rev_input_indptr_cpu, "rev_input_indptr")
                # log_tensor(self.pinned_per_token_offset_cpu, "per_token_offset")
                # log_tensor(batched_kv_cache.pinned_kv_indices, "kv_indices")
                # log_tensor(batched_kv_cache.pinned_kv_indptr, "kv_indptr")
                # log_tensor(batched_kv_cache.pinned_kv_last_page_len, "kv_last_page_len")
                # decode_req_idx = [req.req_idx for req in self._decode_workset._set]
                # prefill_req_idx = [req.req_idx for req in self._prefill_workset._set]
                # logging.info(f"decode_req_idx: {decode_req_idx}")
                # logging.info(f"prefill_req_idx: {prefill_req_idx}")

        with record_function("wait"):
            if self._is_running == isRunning.STATE_FIRST_CYCLE or self._is_running == isRunning.STATE_RUNNING:
                pllm_python.run_async_wait()
            t_wait_async_end = time.perf_counter()
                
        with record_function("update"):
            # print("Start updating...")
            if actualRun:
                pllm_python.update(updateDatas)
            t_update_finish = time.perf_counter()
            logging.info(f"update time: {t_update_finish - t_wait_async_end}")
            # print("Updating FINISHED!")
        with record_function("run"):
            t_run_1 = time.perf_counter()
            if actualRun:
                pllm_python.run_async()
                # toGPUShard(self._embedding_table, self._nranks, torch.half)
                t_run_2 = time.perf_counter()
                if self._is_running == isRunning.STATE_SKIP_CYCLE:
                    self._is_running = isRunning.STATE_FIRST_CYCLE
                elif self._is_running == isRunning.STATE_FIRST_CYCLE:
                    self._is_running = isRunning.STATE_RUNNING

            else:
                t_run_2 = time.perf_counter()
                self._is_running = isRunning.STATE_SKIP_CYCLE
            
            # print("Run FINISHED!")
        self._old_token_length = self._decode_workset.effective_bsz + self._prefill_workset.effective_bsz
        with record_function("get output"):
            if self._is_running == isRunning.STATE_RUNNING:
                output_ids = pllm_python.getPipelineOutput()
                
                for i in range(self._old_prefill.size):
                    req = self._old_prefill[i]
                    if req.chunked_prefill == False:
                        # Not chunked, therefore valid output and append to decode_workset
                        token_idx = self._old_input_indptr[self._old_decode.size + i + 1] - 1
                        # logging.info(f"tokenidx {token_idx}")
                        req.output[-1] = output_ids[token_idx]
                        req.input = [req.output[-1]]
                t_output_for_old_prefill = time.perf_counter()
                logging.info(f"time output prefill: {t_output_for_old_prefill - t_wait_async_end}")
                for i in range(self._old_decode.size):
                    req = self._old_decode[i]
                    req.output[-1] = output_ids[self._old_input_indptr[i]]
                    # logging.info(f"tokenidx {self._old_input_indptr[i]}")
                    req.input = [req.output[-1]]
                t_output_for_old_decode = time.perf_counter()
                logging.info(f"time output decode: {t_output_for_old_decode - t_wait_async_end}")

        
        with record_function("predict_output"):
            t2 = time.perf_counter()
            print (f"Time taken: {t2 - t1}")
            keep_token_count = 0
            with record_function("get_output for decode"):
                self.pinned_keep_token_list = torch.zeros([self._global_bsz], dtype=torch.int32, device='cpu').pin_memory()         
                for i in range(self._decode_workset.size):
                    req = self._decode_workset[i]
                    
                    if len(req.output) == req.output_len - 1 : # or req.output[-1] == EOS_ID:
                        req.decode_latency = t2 - req.decode_start_at
                        req.finish()
                        retired_rq.append(req)
                        # print(req.output)
                    else:
                        self.pinned_keep_token_list[input_indptr[i]] = 1
                        keep_token_count += 1
                        req.output.append(0)
                        # Still decoding
                        req.input = [0]
                        self._new_decode_workset.put(req)
                

            with record_function("get_output for prefill"):
                for i in range(self._prefill_workset.size):
                    req = self._prefill_workset[i]
                    if req.chunked_prefill == False:
                        # Not chunked, therefore valid output and append to decode_workset
                        self.pinned_keep_token_list[input_indptr[self._decode_workset.size + i + 1] - 1] = 1
                        keep_token_count += 1
                        req.chunked_prefill = False
                        req.decode_start_at = t2
                        req.encode_latency = t2 - req.request_comein_time
                        req.input = [0]
                        req.output.append(0)
                        self._new_decode_workset.put(req)
            # Update workset
            print("keep_token_count: ", keep_token_count)
            
        self._old_decode = self._decode_workset
        self._old_prefill = self._prefill_workset
        self._decode_workset = self._new_decode_workset
        self._prefill_workset = self._new_prefill_workset
        self._old_input_indptr = input_indptr
        
            # wait for user input
            # input("Press Enter to continue...")
        

    
if __name__ == "__main__":
    
    model_name_or_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tensor_saved, model_weights = load_weights(1,1, './nanoflow_weight_8B_3_1/')
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--trace_path", type=str, required=True, help="Request trace to read from")
    arg_parser.add_argument("--run_cycles", type=int, default=2000, help="Run cycle")
    arg_parser.add_argument("--config_path", type=str, default="../config/llama3-8B/1024.json", help="Config path")
    arg_parser.add_argument("--skip_cycles", type=int, default=0, help="Skip cycle")
    arg_parser.add_argument("--est_cycle_time", type=float, default=0.175, help="Estimated cycle time")

    args = arg_parser.parse_args()
    
    nranks = torch.cuda.device_count()
    nranks = 1
    pool = DistKVPool(1,8,128,1024*100//16,16,nranks)
    
    request_manager = requestManager(args.trace_path)
    request_manager.read_request(model_name_or_path=model_name_or_path)
    
    scheduler = Scheduler(pool, request_manager.avaliable_request_queue, model_weights, request_manager.average_decode_length, request_manager.average_prefill_length)
    scheduler.init_pipe(args.config_path)

    retired_rq : list[FlyRequestInfo] = []
    totalCycle = 0

    skip_cycle = args.skip_cycles
    while request_manager.full_request_queue.size + request_manager.avaliable_request_queue.size > 0 and skip_cycle > 0:
        skip_cycle -= 1
        print("skip cycle: ", skip_cycle)
        request_manager.simulate_issue(args.est_cycle_time)
        scheduler.bench_text_gen(retired_rq, False)
    
    t1 = time.perf_counter()
    run_cycle = 200
    
    if request_manager.full_request_queue.size >0:
        request_manager.avaliable_request_queue.clear()
    request_manager.start_processing()
    original_size = len(retired_rq)
    
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, use_cuda=True) as prof:
    #     while request_manager.full_request_queue.size + request_manager.avaliable_request_queue.size > 0 and run_cycle > 0:
    #         logging.info(f"------------------ Cycle {totalCycle} ------------------")
    #         t0 = time.perf_counter()
    #         run_cycle -= 1
    #         totalCycle = totalCycle + 1
    #         request_manager.issue()
    #         scheduler.bench_text_gen(retired_rq, True)
    #         t2 = time.perf_counter()
    #         print("---------------------------------------------------Total time taken: ", time.perf_counter() - t0)
    # t_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # prof.export_chrome_trace(f"{t_str}.json")
    

    while request_manager.full_request_queue.size + request_manager.avaliable_request_queue.size > 0 and run_cycle > 0:
        logging.info(f"------------------ Cycle {totalCycle} ------------------")
        t0 = time.perf_counter()
        run_cycle -= 1
        totalCycle = totalCycle + 1
        request_manager.issue()
        scheduler.bench_text_gen(retired_rq, True)
        t2 = time.perf_counter()
        print("---------------------------------------------------Total time taken: ", time.perf_counter() - t0)
    
    pllm_python.run_async_wait()
    pllm_python.finalize()
    
    all_processed_request_list = retired_rq + scheduler._decode_workset._set
    # sort by request index
    all_processed_request_list.sort(key=lambda x: x.req_idx)
    
    # open output file
    out_name = args.trace_path + ".out"
    with open(out_name, 'w') as f:

        for req in all_processed_request_list:
            f.write(f"{req.prompt.tolist()+req.output}\n")
            f.write(f"{tokenizer.decode(req.prompt.tolist()+req.output, skip_special_tokens=True)}\n")

    

