import torch

## Default weight processing is transpose.


def process_weight_none(
    global_weight_map,
    weight_name,
    weight_wrapper,
    layer_list,
    cached_weight_map,
    cached,
    device,
):
    return None


def process_weight_no_transpose(
    global_weight_map,
    weight_name,
    weight_wrapper,
    layer_list,
    cached_weight_map,
    cached,
    device,
    tp_idx=0,
    tp_size=1,
    tp_split_row=True,
):
    owner_name = weight_wrapper.owner.name
    if not cached:
        offset = tp_idx % tp_size
        if tp_split_row:
            stride = global_weight_map[weight_name.format(layer=0)].shape[0] // tp_size
            scope = (slice(offset * stride, (offset + 1) * stride), slice(None))
        else:
            stride = global_weight_map[weight_name.format(layer=0)].shape[1] // tp_size
            scope = (slice(None), slice(offset * stride, (offset + 1) * stride))
        for l in layer_list:
            cached_weight_map[f"{owner_name}_layer_{l}"] = global_weight_map[
                weight_name.format(layer=l)
            ][scope]
    if cached:
        weight_wrapper.weight_map = {}
        for l in layer_list:
            weight_wrapper.weight_map[l] = cached_weight_map[
                f"{owner_name}_layer_{l}"
            ].to(device, non_blocking=True)
        assert (
            weight_wrapper.weight_map[0].shape == weight_wrapper.shape
        ), f"name = {weight_name}, expected shape = {weight_wrapper.shape}, layer = 0, real shape = {weight_wrapper.weight_map[0].shape}"


def process_weight(
    global_weight_map,
    weight_name,
    weight_wrapper,
    layer_list,
    cached_weight_map,
    cached,
    device,
):
    owner_name = weight_wrapper.owner.name
    if not cached:
        for l in layer_list:
            cached_weight_map[f"{owner_name}_layer_{l}"] = global_weight_map[
                weight_name.format(layer=l)
            ].t()

    if cached:
        weight_wrapper.weight_map = {}
        for l in layer_list:
            weight_wrapper.weight_map[l] = cached_weight_map[
                f"{owner_name}_layer_{l}"
            ].to(device, non_blocking=True)
            assert (
                weight_wrapper.weight_map[l].shape == weight_wrapper.shape
            ), f"name = {weight_name}, expected shape = {weight_wrapper.shape}, layer = 0, real shape = {weight_wrapper.weight_map[0].shape}"


def process_weight_list(
    global_weight_map,
    weight_name,
    weight_wrapper,
    layer_list,
    cached_weight_map,
    cached,
    device,
    tp_idx=0,
    tp_size=1,
    tp_split_row=True,
):
    owner_name = weight_wrapper.owner.name
    if not isinstance(weight_name, list):
        weight_name = [weight_name]
    if not cached:
        offset = tp_idx % tp_size
        for l in layer_list:
            weights_list = []
            for name in weight_name:
                if tp_split_row:
                    stride = global_weight_map[name.format(layer=l)].shape[0] // tp_size
                    scope = (slice(offset * stride, (offset + 1) * stride), slice(None))
                else:
                    stride = global_weight_map[name.format(layer=l)].shape[1] // tp_size
                    scope = (slice(None), slice(offset * stride, (offset + 1) * stride))
                weights_list.append(global_weight_map[name.format(layer=l)][scope].t())
            cached_weight_map[f"{owner_name}_layer_{l}"] = torch.cat(
                weights_list, dim=1
            ).contiguous()
    if cached:
        for l in layer_list:
            weight_wrapper.weight_map[l] = cached_weight_map[
                f"{owner_name}_layer_{l}"
            ].to(device, non_blocking=True)
            assert (
                weight_wrapper.weight_map[l].shape == weight_wrapper.shape
            ), f"name = {weight_name}, expected shape = {weight_wrapper.shape}, layer = {l}, real shape = {weight_wrapper.weight_map[l].shape}"
