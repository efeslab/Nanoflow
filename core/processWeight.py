def process_weight_none(global_weight_map, weight_name, weight_wrapper, device_list, layer_list, cached_weight_map, cached = False):
    return None

def process_weight_no_transpose(global_weight_map, weight_name, weight_wrapper, device_list, layer_list, cached_weight_map, tp_size=1, tp_weights_row = False, cached = False):
    weight_wrapper.weight_map = {}
    if cached:
        for device_id in device_list:
            weight_wrapper.weight_map[device_id] = {}
            for l in layer_list:
                weight_wrapper.weight_map[device_id][l] = cached_weight_map[f"{weight_wrapper.owner.name}_device_{device_id}_layer_{l}"].to(f'cuda:{device_id}')
            assert weight_wrapper.weight_map[device_id][0].shape == weight_wrapper.shape, f"name = {weight_name}, expected shape = {weight_wrapper.shape}, layer = 0, real shape = {weight_wrapper.weight_map[device_id][0].shape}"
        return weight_wrapper
    if not cached:
        for device_id in device_list:
            weight_wrapper.weight_map[device_id] = {}
            offset = device_id % tp_size
            if tp_weights_row:
                stride = global_weight_map[weight_name.format(layer = 0)].shape[0] // tp_size
                scope = (slice(offset * stride, (offset+1) * stride), slice(None))
            else:
                stride = global_weight_map[weight_name.format(layer = 0)].shape[1] // tp_size
                scope = (slice(None), slice(offset * stride, (offset+1) * stride))
            for l in layer_list:
                weight_wrapper.weight_map[device_id][l] = global_weight_map[weight_name.format(layer = l)][scope].to(f'cuda:{device_id}')
                cached_weight_map[f"{weight_wrapper.owner.name}_device_{device_id}_layer_{l}"] = weight_wrapper.weight_map[device_id][l].to("cpu")
                assert weight_wrapper.weight_map[device_id][l].shape == weight_wrapper.shape, f"name = {weight_name}, expected shape = {weight_wrapper.shape}, layer = {l}, real shape = {weight_wrapper.weight_map[device_id][l].shape}"
        return weight_wrapper

def process_weight_layer(global_weight_map, weight_name, weight_wrapper, device_list, layer_list, cached_weight_map, cached = False):
    weight_wrapper.weight_map = {}
    if cached:
        for device_id in device_list:
            weight_wrapper.weight_map[device_id] = {}
            for l in layer_list:
                weight_wrapper.weight_map[device_id][l] = cached_weight_map[f"{weight_wrapper.owner.name}_device_{device_id}_layer_{l}"].to(f'cuda:{device_id}')
                assert weight_wrapper.weight_map[device_id][l].shape == weight_wrapper.shape, f"name = {weight_name}, expected shape = {weight_wrapper.shape}, layer = 0, real shape = {weight_wrapper.weight_map[device_id][0].shape}"
        return weight_wrapper
    for device_id in device_list:
        weight_wrapper.weight_map[device_id] = {}
        for l in layer_list:
            weight_wrapper.weight_map[device_id][l] = global_weight_map[weight_name.format(layer = l)].to(f'cuda:{device_id}').t()
            cached_weight_map[f"{weight_wrapper.owner.name}_device_{device_id}_layer_{l}"] = weight_wrapper.weight_map[device_id][l].to("cpu")
            assert weight_wrapper.weight_map[device_id][l].shape == weight_wrapper.shape, f"name = {weight_name}, expected shape = {weight_wrapper.shape}, layer = {l}, real shape = {weight_wrapper.weight_map[device_id][l].shape}"
    return weight_wrapper
