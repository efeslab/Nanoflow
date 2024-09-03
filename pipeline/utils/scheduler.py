import sys
sys.path.append('../build')

import pllm_python

import argparse
import time
import torch
from pybindUtil import toGPU, toGPUShard, initUpdateData, genInitData, load_config

from torch.profiler import profile, record_function, ProfilerActivity
from frontend import requestManager

import pickle
from collections import deque
from typing import List

from kv_cache import DistKVPool, DistKVCache, BatchedDistKVCache
from request_info import NewRequestInfo, NewRequestQueue, FlyRequestInfo

pipetype = pllm_python.PipelineType.PLLM

def getPipeType(s: str):
    if s == "pllm":
        return pllm_python.PipelineType.PLLM
    if s == "pllm-offload":
        return pllm_python.PipelineType.PLLMOFFLOAD
    if s == "non-overlap":
        return pllm_python.PipelineType.NONOVERLAP
    if s == "nanobatch":
        return pllm_python.PipelineType.NANOBATCH
    print("Invalid pipeline type")
    exit(1)

class WorkingSet:
    """
    Wrapper class for denoting a working set of requests.
    Mainly used for calculating effective batch size.
    """
    
    def __init__(self) -> None:
        self._set : List[FlyRequestInfo] = []
        
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
    def __init__(self, memory_pool: DistKVPool, request_queue: NewRequestQueue):
        self._memory_pool = memory_pool
        self._request_queue = request_queue
        
        
        self._decode_workset = WorkingSet()
        self._prefill_workset = WorkingSet()
        
        self._new_decode_workset = WorkingSet()
        self._new_prefill_workset = WorkingSet()
        
        self._gemv_batch_size = [0, 0, 0, 0]
        self._gemv_num_blocks = [108, 108, 108, 108]
        
        self._hidden_dim = 8192
        self._nranks = memory_pool.num_devices
        # self._embedding_table = torch.randn([26000, self._hidden_dim], dtype=torch.float16, device='cpu').pin_memory()
        self._gpu_tensors = []
        self._new_gpu_tensors = []
        self._is_running = False
        self._record_schedule_stat = []
    
    def init_pipe(self, filename: str):
        pllm_python.setRank(self._nranks,8)
        data_array = []
        config_array = []
        weight_array = [pllm_python.VortexModelWeight() for i in range(self._nranks)]
        for i in range(self._nranks):
            pllm_python.createModelWeight(weight_array[i], i)
        for i in range(self._nranks):
            data_array.append(genInitData(i, weight_array[i]))
        pllm_python.init(data_array, pipetype)

        for i in range(self._nranks):
            config_array.append(load_config(filename))
        pllm_python.config(config_array)

        self.config_data = config_array[0]
        self._global_bsz = config_array[0].globalBatchSize
        self._input_embedding = torch.empty([self._global_bsz, self._hidden_dim], dtype=torch.float16, device='cpu').pin_memory()

    def schedule_req(self):
        with record_function("check_new_request"):
            # decode_idle_tokens = self._decode_bsz - self._decode_workset.effective_bsz
            decode_idle_tokens = self._global_bsz - self._decode_workset.effective_bsz
            # prefill_idle_tokens = self._prefill_bsz - self._prefill_workset.effective_bsz
            prefill_idle_tokens = self._global_bsz - self._prefill_workset.effective_bsz - self._decode_workset.effective_bsz
            # print(prefill_idle_tokens, self._prefill_bsz, self._prefill_workset.effective_bsz)

            while prefill_idle_tokens > 0 and decode_idle_tokens > 0 and self._request_queue.size > 0:
                # Check whether there is new request
                if self._request_queue.size > 0:
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
    
    def bench_text_gen(self, retired_rq, actualRun = True):
        self._new_decode_workset = WorkingSet()
        self._new_prefill_workset = WorkingSet()
        t1 = time.perf_counter()
        # Check whether it is possible to start a new request
        
        self.schedule_req()
        with record_function("prepare data"):
            # prefill/decode workset should be ready to run at this time
            self._prefill_workset.adjust_kv_cache()
            self._decode_workset.adjust_kv_cache()
            
            input_ids : List[int] = []
            input_indptr : List[int] = [0]
            prev_len : List[int] = []
            decodePrefillBorder = self._decode_workset.effective_bsz
            decode_kvs : List[DistKVCache] = []
            prefill_kvs : List[DistKVCache] = []
            
        with record_function("calc batch size"):
            t3 = time.perf_counter()
            self._gemv_batch_size[0] = min(decodePrefillBorder, self.config_data.kqv1Size)
            self._gemv_batch_size[1] = min(decodePrefillBorder - self._gemv_batch_size[0], self.config_data.nanobatch1Size - self.config_data.kqv1Size)
            self._gemv_batch_size[2] = min(decodePrefillBorder - self._gemv_batch_size[0] - self._gemv_batch_size[1], self.config_data.kqv3Size)
            self._gemv_batch_size[3] = decodePrefillBorder - self._gemv_batch_size[0] - self._gemv_batch_size[1] - self._gemv_batch_size[2]

            # print("schedule time: ", time.perf_counter() - t3)
        with record_function("prepare KV"):
            for req in self._decode_workset._set:
                input_ids.extend(req.input)
                input_indptr.append(input_indptr[-1] + len(req.input))
                prev_len.append(req.kv_cache.seqlen - 1) # check, -1 since we already append 1 additional token
                decode_kvs.append(req.kv_cache)
            assert input_indptr[-1] == decodePrefillBorder, "Input indptr should be correct."
            
            for req in self._prefill_workset._set:
                input_ids.extend(req.input)
                input_indptr.append(input_indptr[-1] + len(req.input))
                prev_len.append(req.kv_cache.seqlen - len(req.input)) 
                prefill_kvs.append(req.kv_cache)
            # Prepare batched KV-Cache metadata
            batched_kv_cache = BatchedDistKVCache(decode_kvs, prefill_kvs)


            
        # Embedding input_ids
        # with record_function("create input_embedding"):
            # # gen 2048 rand int from 0 to 26000
            # randNums = torch.randint(0, 26000, (2048,), dtype=torch.int32, device='cpu')
            # # input_embeding is init use randNums and embedding_table
            # input_embedding = (self._embedding_table[randNums]).pin_memory()
            
        # input_embedding = torch.empty([2048, self._hidden_dim], dtype=torch.float16, device='cpu').pin_memory()
        
        
        ########
        
        # with record_function("batch_kv prepare"):
        #     batched_kv_cache.prepare()
        
        
        

                
                
        with record_function("batch_kv"):  
            if actualRun:
                [kv_indices, kv_indptr, kv_last_page_len] = batched_kv_cache.toCPUPinned()
                
        with record_function("input_embedding"):
            if actualRun:
            # [device_embedding, self._device_embedding_tensor] = toGPUShard(self._input_embedding, self._nranks, torch.half)
                self.pinned_input_embedding = torch.tensor(self._input_embedding, dtype=torch.int32, device='cpu').pin_memory()
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
                # print ("rev_input_indptr", rev_input_indptr_cpu)

        with record_function("per_token_offset"):
            if actualRun:
                per_token_offset_cpu = []
                for i in range(len(input_indptr)-1):
                    for j in range(input_indptr[i], input_indptr[i+1]):
                        per_token_offset_cpu.append(j-input_indptr[i]+prev_len[i])              
                self.pinned_per_token_offset_cpu = torch.tensor(per_token_offset_cpu, dtype=torch.int32, device='cpu').pin_memory()
                # print ("per_token_offset", per_token_offset_cpu)

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
                    gemv_block_num=self._gemv_num_blocks
                )
            # print(batched_kv_cache.pinned_kv_indices)
            # print(batched_kv_cache.pinned_kv_indptr)
            # print(batched_kv_cache.pinned_kv_last_page_len)
        # self._new_gpu_tensors = [device_embedding, device_input_indptr, batched_kv_cache._kv_indices_tensor,\
        #     batched_kv_cache._kv_indptr_tensor, batched_kv_cache._kv_last_page_len_tensor]
        
        # self._gpu_tensors = self._new_gpu_tensors
        with record_function("wait"):
            if self._is_running:
                pllm_python.run_async_wait()
        with record_function("update"):
            # print("Start updating...")
            if actualRun:
                pllm_python.update(updateDatas)
            # print("Updating FINISHED!")
        with record_function("run"):
            t_run_1 = time.perf_counter()
            if actualRun:
                pllm_python.run_async()
                # toGPUShard(self._embedding_table, self._nranks, torch.half)
                t_run_2 = time.perf_counter()
                self._is_running = True
            else:
                t_run_2 = time.perf_counter()
                self._is_running = False
            
            # print("Run FINISHED!")
        
        with record_function("get_output"):
            output_ids = range(2048) # FIX later
            ########
            t2 = time.perf_counter()
            print (f"Time taken: {t2 - t1}")

            with record_function("get_output for decode"):
                for i in range(self._decode_workset.size):
                    req = self._decode_workset[i]
                    req.output.append(output_ids[input_indptr[i]])
                
                    if len(req.output) >= req.output_len: # note that prefill also produce one token
                        req.decode_latency = t2 - req.decode_start_at
                        req.finish()
                        retired_rq.append(req)
                    else:
                        # Still decoding
                        req.input = [req.output[-1]]
                        self._new_decode_workset.put(req)
            with record_function("get_output for prefill"):
                for i in range(self._prefill_workset.size):
                    req = self._prefill_workset[i]
                    if req.chunked_prefill == False:
                        # Not chunked, therefore valid output and append to decode_workset
                        req.output.append(output_ids[input_indptr[-(self._prefill_workset.size - i)] - 1])
                        req.input = [req.output[-1]]
                        req.chunked_prefill = False
                        req.decode_start_at = t2
                        req.encode_latency = t2 - req.request_comein_time
                        # print("t2 {}, req.request_comein_time {}".format(t2, req.request_comein_time))
                        
                        self._new_decode_workset.put(req)
            # Update workset
            self._decode_workset = self._new_decode_workset
            self._prefill_workset = self._new_prefill_workset
            # wait for user input
            # input("Press Enter to continue...")
        

    
def print_and_to_file(string, f):
    print(string)
    f.write(string + "\n")
    
if __name__ == "__main__":
    
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--profile_mode", type=bool, default=False, help="Run in profile mode")
    arg_parser.add_argument("--trace_path", type=str, required=True, help="Request trace to read from")
    arg_parser.add_argument("--output_path", type=str, default="", help="Output path to save result in csv format")
    arg_parser.add_argument("--skip_cycles", type=int, default=4000, help="Skip cycle")
    arg_parser.add_argument("--run_cycles", type=int, default=2000, help="Run cycle")
    arg_parser.add_argument("--pipeline_type", type=str, default="pllm", help="Pipeline type")
    arg_parser.add_argument("--config_path", type=str, default="../config/2048.json", help="Config path")
    arg_parser.add_argument("--est_cycle_time", type=float, default=0.175, help="Estimated cycle time")
    args = arg_parser.parse_args()
    
    pipetype = getPipeType(args.pipeline_type)
    print(f"Pipeline type: {pipetype}------------------------------")
    print(f"original_str" + args.pipeline_type)
    print(f"output_str" + args.output_path)
    f = open(args.output_path+".stat", "w")
    print_and_to_file (f"Start running {args.trace_path}", f)
    
    nranks = torch.cuda.device_count()
    pool = DistKVPool(1,8,128,100*1500,16,nranks)
    
    request_manager = requestManager(args.trace_path)
    request_manager.read_request()
    
    scheduler = Scheduler(pool, request_manager.avaliable_request_queue)
    scheduler.init_pipe(args.config_path)

    retired_rq : List[FlyRequestInfo] = []
    totalCycle = 0
    
    skip_cycle = args.skip_cycles
    while request_manager.full_request_queue.size + request_manager.avaliable_request_queue.size > 0 and skip_cycle > 0:
        skip_cycle -= 1
        print("skip cycle: ", skip_cycle)
        request_manager.simulate_issue(args.est_cycle_time)
        # print(request_manager.avaliable_request_queue.size)
        # if request_manager.avaliable_request_queue.size > 0:
        #     print(request_manager.avaliable_request_queue._queue[0].start_time)
        # input("Press Enter to continue...")
        scheduler.bench_text_gen(retired_rq, False)
        # get scheduler property
    
    outfile = open("stat_out6.schedule.out", "w")
    for item in scheduler._record_schedule_stat:
        outfile.write(f"{item[1]},{item[2]}\n")
    outfile.close()
        
    t1 = time.perf_counter()
    run_cycle = 2000
    if args.profile_mode:
        run_cycle = 10
    if request_manager.full_request_queue.size >0:
        request_manager.avaliable_request_queue.clear()
    request_manager.start_processing()
    original_size = len(retired_rq)
    

    if args.profile_mode:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, use_cuda=True) as prof:
            with record_function("model_inference"):
                    t0 = time.perf_counter()
                    while request_manager.full_request_queue.size + request_manager.avaliable_request_queue.size > 0 and run_cycle > 0:
                        run_cycle -= 1
                        print("totalCycle: ", totalCycle)
                        totalCycle = totalCycle + 1
                        request_manager.issue()
                        scheduler.bench_text_gen(retired_rq, True)
                        print("---------------------------------------------------Total time taken: ", time.perf_counter() - t0)
                        t0 = time.perf_counter()
    else:
        while request_manager.full_request_queue.size + request_manager.avaliable_request_queue.size > 0 and run_cycle > 0:
            t0 = time.perf_counter()
            run_cycle -= 1
            totalCycle = totalCycle + 1
            request_manager.issue()
            scheduler.bench_text_gen(retired_rq, True)
            t2 = time.perf_counter()
            print("---------------------------------------------------Total time taken: ", time.perf_counter() - t0)
    
    pllm_python.run_async_wait()
    pllm_python.finalize()
    # get current time in format of "YYYY-MM-DD-HH-MM-SS"
    
    if args.profile_mode:
        t_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        
        prof.export_chrome_trace(f"{t_str}.json")
        # compress the trace file to gz
        import gzip
        with open(f"{t_str}.json", 'rb') as f:
            with gzip.open(f"{t_str}.json.gz", 'wb') as f_out:
                f_out.writelines(f)
        exit()

    final_size = len(retired_rq)
    print("Done")
    t2 = time.perf_counter()

    print_and_to_file (f"TOTAL Time taken: {t2 - t1}", f)
    tps = (final_size - original_size) * \
                        (request_manager.average_decode_length + request_manager.average_prefill_length) /(t2-t1)
    print_and_to_file (f"Token per second: {tps}", f)
    # TODO: yile: assumes 8 GPUs for now, may need to change later
    tps_per_gpu = tps / 8
    print_and_to_file (f"per GPU: {tps_per_gpu}", f)
    
    print_and_to_file (f"Total Cycle: {totalCycle}", f)
    print_and_to_file (f"Cycle time taken: {(t2-t1)/totalCycle}", f)
    
    print_and_to_file (f"start_idx: {request_manager.start_idx}", f)
    # filter retired request 
    retired_rq = [req for req in retired_rq if req.req_idx >= request_manager.start_idx]
    print_and_to_file (f"Actual Retired request size: {len(retired_rq)}", f)
    
    total_ttft = 0
    total_tpot = 0
    normalize_latency = 0
    
    for req in retired_rq:
        total_ttft += req.encode_latency 
        total_tpot += req.decode_latency / req.output_len
        normalize_latency += (req.decode_start_at + req.decode_latency - req.request_comein_time)/req.output_len
    
    average_ttft = total_ttft / len(retired_rq)
    average_tpot = total_tpot / len(retired_rq)
    average_normalize_latency = normalize_latency / len(retired_rq)
    
    print_and_to_file (f"Average TTFT: {average_ttft}", f)
    print_and_to_file (f"Average TPOT: {average_tpot}", f)
    print_and_to_file (f"Average Normalize Latency: {average_normalize_latency}", f)
    
    if args.output_path != "":
        with open(args.output_path, 'w') as f:
            f.write("total_time,token_per_second,token_per_second_per_gpu,total_cycle,cycle_time,average_ttft,average_tpot,average_normalize_latency\n")
            f.write(f"{t2-t1},{tps},{tps_per_gpu},{totalCycle},{(t2-t1)/totalCycle},{average_ttft},{average_tpot},{average_normalize_latency}\n")
    
    f.close()
    # save retired request to csv
    with open(args.output_path + ".ret", 'w') as f:
        for req in retired_rq:
            f.write(f"{req.req_idx},{req.input_len},{req.output_len},{req.request_comein_time},{req.encode_latency},{req.decode_start_at},{req.decode_latency}\n")