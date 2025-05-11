import sys

from request_info import NewRequestInfo, NewRequestQueue, FlyRequestInfo
from transformers import AutoTokenizer
from loguru import logger

import queue
import time
import torch
import random

class requestManager:
    def __init__(self, filename: str, model_name_or_path = "meta-llama/Llama-2-70b-chat-hf"):
        logger.info(f"[frontend] init requestManager, model_name_or_path: {model_name_or_path}")
        self.file_queue = queue.Queue()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.available_request_queue = NewRequestQueue()
        self.full_request_queue = NewRequestQueue()
        self.start_time = 0
        self.passed_time = 0
        self.average_prefill_length = 0
        self.average_decode_length = 0
        self.pdr = 0
        self.start_idx = 0
        self.request_queue_budget = 500

        if filename is not None:
            self.file_queue.put(filename)
    
    def tokenize_and_put(self, req: NewRequestInfo, test_input = False):
        if not test_input:
            # print all infos of req
            # print("req.req_idx", req.req_idx)
            # print("req.prompt", req.prompt)
            # print("req.output_len", req.output_len)
            # print("req.start_time", req.start_time)
            req.prompt = self.tokenizer(req.prompt, return_tensors="pt")["input_ids"][0]
        else:
            req.prompt = torch.tensor(req.prompt, dtype=torch.int32)
        self.full_request_queue.put(req)
        temp_sum_prefill_length = len(req.prompt)
        temp_sum_decode_length = req.output_len

        return temp_sum_prefill_length, temp_sum_decode_length


    def read_request(self, random_input = False):
        logger.info(f"[frontend] read request")
        sum_prefill_length = 0
        sum_decode_length = 0
        if self.file_queue.empty():
            logger.error("[frontend] file_queue is empty")
            return
        file_name = self.file_queue.get()
        print(file_name)
        with open(file_name, 'r') as f:
            first = True
            for line in f:
                # if first:
                #     first = False
                #     continue
                req = NewRequestInfo()
                
                split_result = line.split(',')   
                req.req_idx = int(split_result[0])
                req.output_len = int(split_result[2])
                # if split_result[4] exists and == -1, it means this is real traces with real input ids
                # if (len(split_result) > 5) and (int(split_result[5]) == -1):
                #     input_tokens = split_result[4].strip().split(' ')
                #     # print("input_tokens", input_tokens)
                #     req.prompt = [int(token) for token in input_tokens]
                #     # print("req.prompt", req.prompt)
                #     req.prompt = torch.tensor(req.prompt, dtype=torch.int32)
                if int(split_result[1]) == -1:
                    # find the fourth comma and the rest is the prompt string
                    prompt_str = line.split(',', 3)[3]
                    if prompt_str[-1] == '\n':
                        prompt_str = prompt_str[:-1]
                    req.prompt = prompt_str
                    temp_sum_prefill_length, temp_sum_decode_length = self.tokenize_and_put(req)
                elif int(split_result[1]) == -2:
                    # find the fourth comma and the rest is the prompt string
                    prompt_str = line.split(',', 3)[3]
                    # prompt_str is now list of integers in form of [1, 2, 3, 4, 5]
                    prompt_str = prompt_str[1:]
                    prompt_str = prompt_str[:-1]
                    prompt_str = prompt_str.split(', ')
                    l = []
                    for i in prompt_str:
                        l.append(int(i))
                    req.prompt = torch.tensor(l, dtype=torch.int32)
                    temp_sum_prefill_length = len(req.prompt)
                    temp_sum_decode_length = int(split_result[2])
                    self.full_request_queue.put(req)
                else:
                    if random_input:
                        # print("[INFO] Random input")
                        req.prompt = [random.randint(0, self.tokenizer.vocab_size-1) for _ in range(int(split_result[1]))]
                    else:
                        # print("[INFO] Fixed input")
                        req.prompt = [i for i in range(int(split_result[1]))]
                    temp_sum_prefill_length, temp_sum_decode_length = self.tokenize_and_put(req, test_input=True)
                sum_prefill_length += temp_sum_prefill_length
                sum_decode_length += temp_sum_decode_length
        if self.full_request_queue.size != 0:
            self.average_prefill_length = sum_prefill_length // self.full_request_queue.size
            self.average_decode_length = sum_decode_length // self.full_request_queue.size
            self.pdr = self.average_prefill_length / self.average_decode_length
        
        
    
    @property
    def decode_batch_size(self):
        return int(2048//(1+self.pdr))
    
    def start_processing(self):
        self.start_time = time.perf_counter()
        if self.available_request_queue.size > 0:
            self.start_idx = self.available_request_queue._queue[0].req_idx
        elif self.full_request_queue.size > 0:
            self.start_idx = self.full_request_queue._queue[0].req_idx
        else :
            self.start_idx = -1
    
    def release_request(self):
        count = 0 
        while self.full_request_queue.size > 0 and count < self.request_queue_budget:
            count += 1
            temp = self.full_request_queue.get()
            self.available_request_queue.put(temp)
            
    
    def simulate_issue(self, estimate_cycle_time = 0.175):
        self.passed_time += estimate_cycle_time
        self.release_request()
    
    def issue(self):
        t = time.perf_counter()
        self.passed_time += t - self.start_time
        self.start_time = t
        self.release_request()
        


if __name__ == "__main__":
    request_manager = requestManager(sys.argv[1])
    request_manager.read_request()
    for i in range(100):
        request_manager.simulate_issue()
        print(request_manager.available_request_queue.size)
    
    request_manager.start_processing()
    for i in range(100):
        time.sleep(0.5)
        request_manager.issue()
        print(request_manager.available_request_queue.size)