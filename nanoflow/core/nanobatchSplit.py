from nanoflow.operations import NanoOpInfo, Operations
from nanoflow.operations.virtualOp.virtual_ops import Redist

def split_nanobatch(op_list: list[Operations], op_nano_info_map: dict[str, tuple[NanoOpInfo, ...]], extra_links):
    nano_op_list: list[Operations] = []
    additional_virtual_ops = []
    device = op_list[0].device
    for op in op_list:
        if (op.name not in op_nano_info_map):
            nano_op_list.append(op)
            continue
        elif len(op_nano_info_map[op.name]) == 1:
            op.setBatchSize(op_nano_info_map[op.name][0].batch_size)
            nano_op_list.append(op)
            continue

        nano_op_info_list = list(op_nano_info_map[op.name])
        op.isNanoSplit = True
        op.nano_ops = []

        redists_in = []
        redists_out = []
        for key, value in op.inputs.items():
            op_redist = Redist(f"Nano_Dist_{op.name}_{key}", device, 1, len(nano_op_info_list))
            op_redist.clear_inputs_and_outputs_links()
            op_redist.set_input(value)
            redists_in.append(op_redist)
            additional_virtual_ops.append(op_redist)

        for key, value in op.outputs.items():
            # print(f"Nano_Dist_{op.name}_{key}")
            op_redist = Redist(f"Nano_Dist_{op.name}_{key}", device, len(nano_op_info_list), 1)
            op_redist.clear_inputs_and_outputs_links()
            op_redist.set_output(value)
            redists_out.append(op_redist)
            additional_virtual_ops.append(op_redist)

        for info in nano_op_info_list:
            batch_idx = info.batch_idx
            copied_op = op.copy_nano(batch_idx)
            copied_op.setBatchSize(info.batch_size)

            for j, (key, value) in enumerate(copied_op.inputs.items()):
                redists_in[j].outputs[f"output_{batch_idx}"] >> value
            for j, (key, value) in enumerate(copied_op.outputs.items()):
                value >> redists_out[j].inputs[f"input_{batch_idx}"]
            
            nano_op_list.append(copied_op)
    
    nano_op_map = {op.name : op for op in nano_op_list}
    for key, list_of_values in extra_links.items():
        for value, depend_on_prev_layer in list_of_values:
            op_key = nano_op_map.get(key)
            op_value = nano_op_map.get(value)
            # print("key", key, "value", value)
            # find the name key and value in nano_op_list
            # print("op_key.name", op_key.name, "key", key)
            # print("op_value.name", op_value.name, "value", value)
            if op_key and op_value:
                op_key.append_dependency((op_value, depend_on_prev_layer))
            else:
                raise ValueError(f"Operation {key} or {value} not found in nano_op_list")

    return nano_op_list, additional_virtual_ops