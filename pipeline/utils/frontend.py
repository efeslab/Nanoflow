import sys
import os
os.environ['HF_HOME'] = '../../../hf'
from request_info import NewRequestInfo, NewRequestQueue, FlyRequestInfo
from transformers import AutoTokenizer

import time
import torch

class requestManager:
    def __init__(self, filename: str):
        self.filename = filename
        self.avaliable_request_queue = NewRequestQueue()
        self.full_request_queue = NewRequestQueue()
        self.start_time = 0
        self.passed_time = 0
        self.average_prefill_length = 0
        self.average_decode_length = 0
        self.pdr = 0
        self.start_idx = 0
        

    def read_request(self, model_name_or_path = "meta-llama/Llama-2-70b-chat-hf"):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        req_index = 0
        sum_prefill_length = 0
        sum_decode_length = 0
        print(self.filename)
        with open(self.filename, 'r') as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                req = NewRequestInfo()
                req.req_idx = req_index
                req_index += 1
                split_result = line.split(',')   
                
                req.output_len = int(split_result[2])
                req.start_time = float(split_result[3])
                if int(split_result[1]) == -1:
                    # find the fourth comma and the rest is the prompt string
                    prompt_str = line.split(',', 4)[4]
                    if prompt_str[-1] == '\n':
                        prompt_str = prompt_str[:-1]
                    input_tokens = tokenizer(prompt_str, return_tensors="pt")["input_ids"][0]
                    req.prompt = input_tokens
                else:
                    req.prompt = [i for i in range(int(split_result[1]))]
                    # convert to tensor
                    req.prompt = torch.tensor(req.prompt, dtype=torch.int32)
                self.full_request_queue.put(req)
                sum_prefill_length += len(req.prompt)
                sum_decode_length += req.output_len
        self.average_prefill_length = sum_prefill_length // self.full_request_queue.size
        self.average_decode_length = sum_decode_length // self.full_request_queue.size
        self.pdr = self.average_prefill_length / self.average_decode_length
        
        
    
    @property
    def decode_batch_size(self):
        return int(2048//(1+self.pdr))
    
    def start_processing(self):
        self.start_time = time.perf_counter()
        if self.avaliable_request_queue.size > 0:
            self.start_idx = self.avaliable_request_queue._queue[0].req_idx
        elif self.full_request_queue.size > 0:
            self.start_idx = self.full_request_queue._queue[0].req_idx
        else :
            self.start_idx = -1
    
    def release_request(self):
        while self.full_request_queue.size > 0 and self.full_request_queue._queue[0].start_time < self.passed_time:
            temp = self.full_request_queue.get()
            temp.start_time = time.perf_counter()
            self.avaliable_request_queue.put(temp)
            
    
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
        print(request_manager.avaliable_request_queue.size)
    
    request_manager.start_processing()
    for i in range(100):
        time.sleep(0.5)
        request_manager.issue()
        print(request_manager.avaliable_request_queue.size)