def process_weight_none(global_weight_map, weight_name, weight_wrapper, layer_list, cached_weight_map, cached, device):
    return None

def process_weight_no_transpose(global_weight_map, weight_name, weight_wrapper, layer_list, cached_weight_map, cached, device, tp_idx=0, tp_size=1, tp_weights_row = False):
    if not cached:
        offset = tp_idx % tp_size
        if tp_weights_row:
            stride = global_weight_map[weight_name.format(layer = 0)].shape[0] // tp_size
            scope = (slice(offset * stride, (offset+1) * stride), slice(None))
        else:
            stride = global_weight_map[weight_name.format(layer = 0)].shape[1] // tp_size
            scope = (slice(None), slice(offset * stride, (offset+1) * stride))
        for l in layer_list:
            cached_weight_map[f"{weight_wrapper.owner.name}_layer_{l}"] = global_weight_map[weight_name.format(layer = l)][scope]
    if cached:
        weight_wrapper.weight_map = {}
        for l in layer_list:
            weight_wrapper.weight_map[l] = cached_weight_map[f"{weight_wrapper.owner.name}_layer_{l}"].to(device, non_blocking=True)
        assert weight_wrapper.weight_map[0].shape == weight_wrapper.shape, f"name = {weight_name}, expected shape = {weight_wrapper.shape}, layer = 0, real shape = {weight_wrapper.weight_map[0].shape}"


def process_weight_layer(global_weight_map, weight_name, weight_wrapper, layer_list, cached_weight_map, cached, device):
    if not cached:
        for l in layer_list:
            cached_weight_map[f"{weight_wrapper.owner.name}_layer_{l}"] = global_weight_map[weight_name.format(layer = l)].t()

    if cached:
        weight_wrapper.weight_map = {}
        for l in layer_list:
            weight_wrapper.weight_map[l] = cached_weight_map[f"{weight_wrapper.owner.name}_layer_{l}"].to(device, non_blocking=True)
            assert weight_wrapper.weight_map[l].shape == weight_wrapper.shape, f"name = {weight_name}, expected shape = {weight_wrapper.shape}, layer = 0, real shape = {weight_wrapper.weight_map[0].shape}"
