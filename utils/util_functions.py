import re


def prepare_weight(pipeline_weight_list, weight_map):
    for rank, device, pipeline in pipeline_weight_list:
        pipeline.set_device(rank, device)
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

def op_name_to_name_idx_layer(op_name: str) -> tuple[str, int, int]:
    basename, layer_idx = op_name.split("_")
    name_idx = re.match(r"([A-Za-z]+)([0-9]+)", basename)
    if name_idx is None:
        return basename, -1, int(layer_idx)
    return name_idx.groups()[0], int(name_idx.groups()[1]), int(layer_idx)
