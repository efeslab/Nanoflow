class KVCache:
    pass

class KVCacheNone(KVCache):
    def __init__(self):
        self.cache = {}
    
    def put(self, layer, idx, key, value):
        self.cache[(layer, idx)] = (key, value)

    def get(self, layer, idx):
        return self.cache.get((layer, idx), None)