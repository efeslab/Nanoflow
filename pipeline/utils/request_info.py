from collections import deque
from typing import List

class NewRequestInfo:
    """
    Request info for incoming request
    NOTE (Yilong): add support for offloading / onloading KV-Cache
    """
    req_idx: int
    prompt: List[int]
    output_len : int
    start_time: float

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
    
    def clear(self) -> None:
        self._queue.clear()

class FlyRequestInfo:
    """
    Request info for on-the-fly request
    NOTE (Yilong): add support for offloading / onloading KV-Cache
    """
    
    def __init__(self, req_idx: int, input: List[int], output: List[int], prompt: List[int], request_comein_time: float, 
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

    def finish(self) -> None:
        self.kv_cache.release()

    