from collections import deque
from .offload import offloadData, offloadMetaData
import torch

class NewRequestInfo:
    """
    Request info for incoming request
    NOTE (Yilong): add support for offloading / onloading KV-Cache
    """
    req_idx: int
    prompt: list[int]
    output_len : int

class NewRequestQueue:
    """
    Thread-safe request deque as request buffer.
    """
    def __init__(self) -> None:
        self._queue = deque()
    
    @property
    def size(self) -> int:
        return len(self._queue)

    def put(self, req: NewRequestInfo):
        self._queue.append(req)
    
    def get(self) -> NewRequestInfo:
        assert len(self._queue) > 0, "Queue is empty"
        return self._queue.popleft()
    
    def peek_left(self) -> NewRequestInfo:
        assert len(self._queue) > 0, "Queue is empty"
        return self._queue[0]

    def pop(self) -> NewRequestInfo:
        assert len(self._queue) > 0, "Queue is empty"
        return self._queue.popleft()
    
    def clear(self) -> None:
        self._queue.clear()
        
    def __iter__(self):
        return iter(self._queue)

class FlyRequestInfo:
    """
    Request info for on-the-fly request
    NOTE (Yilong): add support for offloading / onloading KV-Cache
    """
    def __init__(self, req_idx: int, input: list[int], output: list[int], prompt: list[int], request_comein_time: float, 
                 chunked_prefill: bool, kv_cache, encode_latency: float, 
                 decode_start_at: float, decode_latency: float, output_len: int, input_len: int):
        self.req_idx = req_idx
        self.input = input
        self.output = output
        self.prompt = prompt
        self.chunked_prefill = chunked_prefill
        self.kv_cache = kv_cache
        self.encode_latency = encode_latency
        self.decode_start_at = decode_start_at
        self.decode_latency = decode_latency
        self.output_len = output_len
        self.input_len = input_len
        self.request_comein_time = request_comein_time
        self.finished = False
        self.offload_metadata = offloadMetaData()
        self.promise_pages = 0
        self.promise_page_list = []
        self.remaining_prompt = prompt


    def finish(self) -> None:
        self.kv_cache.release()
        self.finished = True

    # promise some page that would be received in next iteration, return the pages to offload
    def schedule_offload(self, promise_offload_page_num) -> list[int]:
        pages = self.kv_cache.indicies
        promised_last_idx = self.offload_metadata.promise(promise_offload_page_num)
        self.promise_pages = promise_offload_page_num
        self.promise_page_list = pages[promised_last_idx : promised_last_idx + promise_offload_page_num]
        return pages[promised_last_idx : promised_last_idx + promise_offload_page_num]

    def receive_offload(self, tensor, page_list) -> None:
        self.offload_metadata.append(tensor)
        assert self.promise_page_list == page_list, "Page list does not match"
        self.promise_pages = 0
    
    # number of unpromised, un-offloaded pages
    def offload_remaining_pages(self) -> int:
        assert self.promise_pages == 0, "Cannot check remaining pages when there are promised pages"
        return len(self.kv_cache.indicies) - self.offload_metadata.last_page()

    def offload_finished (self) -> bool:
        return self.promise_pages == 0 and self.offload_metadata.page_num == len(self.kv_cache.indicies)