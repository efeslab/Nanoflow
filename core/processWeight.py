def process_weight_none(global_weight_map, weight_name, weight_wrapper, total_layer, cached = False):
    return None

def process_weight_no_transpose(global_weight_map, weight_name, weight_wrapper, total_layer, cached = False):
    weight_wrapper.weight_map = {}
    for l in range(total_layer):
        weight_wrapper.weight_map[l] = global_weight_map[weight_name.format(layer = l)]
        assert weight_wrapper.weight_map[l].shape == weight_wrapper.shape, f"name = {weight_name}, shape = {weight_wrapper.shape}, layer = {l}, shape = {weight_wrapper.weight_map[l].shape}"
    return weight_wrapper

def process_weight_layer(global_weight_map, weight_name, weight_wrapper, total_layer, cached = False):
    weight_wrapper.weight_map = {}
    for l in range(total_layer):
        weight_wrapper.weight_map[l] = global_weight_map[weight_name.format(layer = l)].t()
        assert weight_wrapper.weight_map[l].shape == weight_wrapper.shape, f"name = {weight_name}, shape = {weight_wrapper.shape}, layer = {l}, shape = {weight_wrapper.weight_map[l].shape}"
    return weight_wrapper
