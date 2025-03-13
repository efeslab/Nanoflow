import torch

class KVCache:
    pass

class KVCacheNone(KVCache):
    def __init__(self):
        self.cache = {}
    
    def put(self, layer, idx, key, value):
        if (layer, idx) in self.cache:
            old_key, old_value = self.cache[(layer, idx)]
            key = torch.cat([old_key, key], dim=0)
            value = torch.cat([old_value, value], dim=0)
        self.cache[(layer, idx)] = (key, value)

    def get(self, layer, idx):
        return self.cache.get((layer, idx), None)