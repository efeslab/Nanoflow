def prepare_weight(Pipeline, weight_map):
    pipeline = Pipeline()
    pipeline.init_cached_weight(weight_map)

def tensor_offset_to_req_idx(qo_indicies, tensor_offset):
    for idx, cum_batch_size in enumerate(qo_indicies):
        if cum_batch_size == tensor_offset:
            return idx
        elif cum_batch_size > tensor_offset:
            print("cum_batch_size", cum_batch_size)
            print("tensor_offset", tensor_offset)
            raise ValueError(f"tensor_offset {tensor_offset} is not valid")

def req_idx_to_tensor_offset(qo_indicies, req_idx):
    return qo_indicies[req_idx]