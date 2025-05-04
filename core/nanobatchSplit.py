from typing import List
from operations.operation_base import Operations
from operations.virtualOp.virtual_ops import Redist

def split_nanobatch(op_list: List[Operations], op_nano_info_map, extra_links):
    nano_op_list = []
    additional_virtual_ops = []
    for op in op_list:
        if op.name not in op_nano_info_map:
            nano_op_list.append(op)
            continue
        # print("op.name", op.name, "nanobatch_split")
        num_nano_op, batch_range = op_nano_info_map[op.name]

        op.isNanoSplit = True
        op.nano_op_batchsizes = batch_range
        op.nano_ops = []

        redists_in = []
        redists_out = []
        for key, value in op.inputs.items():
            op_redist = Redist(f"Nano_Dist_{op.name}_{key}", 1, num_nano_op)
            op_redist.clear_inputs_and_outputs_links()
            op_redist.set_input(value)
            op_redist.expand_gpu(len(op.device_list))
            redists_in.append(op_redist)
            additional_virtual_ops.append(op_redist)

        for key, value in op.outputs.items():
            # print(f"Nano_Dist_{op.name}_{key}")
            op_redist = Redist(f"Nano_Dist_{op.name}_{key}", num_nano_op, 1)
            op_redist.clear_inputs_and_outputs_links()
            op_redist.set_output(value)
            op_redist.expand_gpu(len(op.device_list))
            redists_out.append(op_redist)
            additional_virtual_ops.append(op_redist)

        for i in range(num_nano_op):
            copied_op = op.copy_nano(i)
            for child in copied_op.children:
                child.setBatchSize(batch_range[i])

            for j, (key, value) in enumerate(copied_op.inputs.items()):
                redists_in[j].outputs[f"output_{i}"] >> value
            for j, (key, value) in enumerate(copied_op.outputs.items()):
                value >> redists_out[j].inputs[f"input_{i}"]

            nano_op_list.append(copied_op)
    
    nano_op_map = {op.name : op for op in nano_op_list}
    for key, value in extra_links.items():
        value, depend_on_prev_layer, depend_on_next_layer = value
        op_key = nano_op_map.get(key)
        op_value = nano_op_map.get(value)
        # print("key", key, "value", value)
        # find the name key and value in nano_op_list
        # print("op_key.name", op_key.name, "key", key)
        # print("op_value.name", op_value.name, "value", value)
        if op_key and op_value:
            op_value.append_dependency((op_key, depend_on_prev_layer, depend_on_next_layer))
        else:
            raise ValueError(f"Operation {key} or {value} not found in nano_op_list")

    return nano_op_list, additional_virtual_ops