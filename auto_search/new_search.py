
# from plotPipe import drawPipeline, createGraph, createLogicalDependencyGraph, createLogicalDependencyGraphOfNanoOps, createLogicalDependencyGraphOfLayeredNanoOps\
#     ,calcStartEndLayered,getCycleTime,drawPipelineLayered,createGraphOfLayeredNanoOps
import sys
sys.path.append('../')
sys.path.append('../pybind/build/')
from models.llama3_FlashinferKVCache import Pipeline
from core.executor import Executor

import itertools
from typing import Dict
from collections import defaultdict

profile_data_path = '../profile_data/Llama3-8B/'

hdim=4096
idim=14336
kqv_n = int(hdim * 1.5)

global_batch_size = 1024
decode_batch_size = 384
prefill_batch_size = 640
seq_len = 1024

# read profile
from profileAnalysis import getGemvTime, getByBatchsize

# getGemmProfile(profile_data_path, "O", 1024, ("RowMajor", "RowMajor", "RowMajor"))
getByBatchsize(profile_data_path, "O", 1024)
getGemvTime(profile_data_path, 384, 1024)


batch_size_range = list(range(128, global_batch_size+1, 128))
print("batch_size_range:", batch_size_range)

# create operations
pipeline = Pipeline()
pipeline.init_streams()
pipeline.init_external_data(for_test=True)
pipeline.init_operations()
pipeline.init_dependency()
pipeline.init_set_shape()

# set offsets and batch_sizes
pipeline.batch_size = global_batch_size
pipeline.clear_batch_size()
pipeline.config_batch_size(decode_batch_size)
pipeline.nanobatch_split(global_batch_size, decode_batch_size)
pipeline.update_allocate_buffers()

# set streams
pipeline.config_streams()


# get layered operation
from operations.operation_base import Operation_Layer
layer_num = 1
all_layered_ops: list[Operation_Layer] = []
for op in pipeline.new_operation_list:
    if op.first_layer_only or op.last_layer_only:
        continue
    layered_ops = op.children[:layer_num]
    all_layered_ops.extend(layered_ops)
print("all_layered_ops:", [op.name for op in all_layered_ops])
print("all_layered_ops original name:", [op.original_name for op in all_layered_ops])


# operations = [op for op in pipeline.op_layers if op.parent.first_layer_only == False and op.parent.last_layer_only == False]

# print("operations:", [op.name for op in operations])


for layer_op in all_layered_ops:
    print(layer_op.name)
    print("prerequisite operations:", [op for op in layer_op.prerequisites])
    if layer_op.original_name == "DecAttn":
        batch_size = layer_op.batch_size
        print("batch_size:", batch_size)
        duration = getGemvTime(profile_data_path, batch_size, seq_len)
        layer_op.duration_map[(batch_size, 1)] = duration
        print("duration_map:", layer_op.duration_map)
    else:
        batch_size = layer_op.batch_size
        print("batch_size:", batch_size)
        duration = getByBatchsize(profile_data_path, layer_op.original_name, batch_size)
        layer_op.duration_map[(batch_size, 1)] = duration
        print("duration_map:", layer_op.duration_map)

# create executor
executor = Executor(all_layered_ops, [i for i in range(layer_num)])
executor.plan_layer_ordering()

for layer_op in all_layered_ops:
    print(layer_op.name)
    print("prev_op_layer:", [op.name for op in layer_op.prev_op_layer])
    print("is_depended_on:", layer_op.is_depended_on)

import gurobipy as gp
from gurobipy import GRB

model = gp.Model("pipeline")
for layer_op in all_layered_ops:
    layer_op.initVariables(model) # add start_time, end_time

# create sequantial constraints
sequence_nano = {}
category_nano_op_map: dict[str, list[Operation_Layer]] = defaultdict(list)
for layer_op in all_layered_ops:
    category_nano_op_map[str(layer_op.stream.cuda_stream)].append(layer_op)

for nano_ops in category_nano_op_map.values():
    for op_1, op_2 in itertools.combinations(nano_ops, 2):
        sequence_nano[op_1, op_2] = model.addVar(vtype=GRB.BINARY, name=f"{op_1.name}_before_{op_2.name}")

# the completion time of all operations
C_max = model.addVar(vtype=GRB.CONTINUOUS, name="C_max")

# non-overlapping constraints
for type_name, nano_op_list in category_nano_op_map.items():
    for op1, op2 in itertools.combinations(nano_op_list, 2):
        model.addConstr(op1.end_time <= op2.start_time + 10 * (1 - sequence_nano[op1, op2]),
                        name=f"non_overlap_{op1.name}_{op2.name}_when_sequence_nano_1")
        model.addConstr(op2.end_time <= op1.start_time + 10 * sequence_nano[op1, op2],
                        name=f"non_overlap_{op1.name}_{op2.name}_when_sequence_nano_2")
    
# dependency constraints
for layer_op in all_layered_ops:
    for dep_op in layer_op.prev_op_layer:
        print("layer_op:", layer_op.name, "dep_op:", dep_op.name)
        model.addConstr(dep_op.end_time <= layer_op.start_time, 
                        name=f"dependency_{dep_op.name}_before_{layer_op.name}")

# fusion constraints
for layer_op in all_layered_ops:
    name = layer_op.original_name
    if name == "KQV":
        assert len(layer_op.prev_op_layer) == 1, f"Expected only one previous operation for {layer_op.name}, got {len(layer_op.prev_op_layer)}"
        prev_op = layer_op.prev_op_layer[0]
        model.addConstr(prev_op.end_time == layer_op.start_time, name=f"fusion_{prev_op.name}_to_{layer_op.name}")
    elif name == "RopeAppend":
        assert len(layer_op.prev_op_layer) == 1, f"Expected only one previous operation for {layer_op.name}, got {len(layer_op.prev_op_layer)}"
        prev_op = layer_op.prev_op_layer[0]
        model.addConstr(prev_op.end_time == layer_op.start_time, name=f"fusion_{prev_op.name}_to_{layer_op.name}")
    elif name == "LayerNormFFN":
        assert len(layer_op.prev_op_layer) == 1, f"Expected only one previous operation for {layer_op.name}, got {len(layer_op.prev_op_layer)}"
        prev_op = layer_op.prev_op_layer[0]
        model.addConstr(prev_op.end_time == layer_op.start_time, name=f"fusion_{prev_op.name}_to_{layer_op.name}")
    elif name == "Activation":
        assert len(layer_op.prev_op_layer) == 1, f"Expected only one previous operation for {layer_op.name}, got {len(layer_op.prev_op_layer)}"
        prev_op = layer_op.prev_op_layer[0]
        model.addConstr(prev_op.end_time == layer_op.start_time, name=f"fusion_{prev_op.name}_to_{layer_op.name}")


# C_max constraints
for layer_op in all_layered_ops:
    model.addConstr(layer_op.end_time <= C_max, name=f"C_max_{layer_op.name}")


model.setObjective(C_max, GRB.MINIMIZE)
model.setParam("Threads", 200)
# model.setParam("TimeLimit", 120)
model.optimize()

for layer_op in all_layered_ops:
    print(layer_op)

# verify the results
for layer_op in all_layered_ops:
    for dep_op in layer_op.prev_op_layer:
        if dep_op.end_time.X > layer_op.start_time.X + 0.001:
            print(f"Error: {dep_op.name} end time {dep_op.end_time} is greater than {layer_op.name} start time {layer_op.start_time}")
        # else:
        #     print(f"Dependency check passed: {dep_op.name} -> {layer_op.name}")
    

from matplotlib import pyplot as plt

# Assuming you have a list of NanoOperation instances called nano_operations_list
# and each NanoOperation has the required attributes.

# Extract unique operation types
operation_types = sorted(set(n.stream.cuda_stream for n in all_layered_ops))
y_positions = {op_type: i for i, op_type in enumerate(operation_types)}

fig, ax = plt.subplots(figsize=(30, 6))

for n in all_layered_ops:
    n_name = n.name
    # Access the optimized values of the variables
    start_time = n.start_time.X
    duration = n.duration_map[(n.batch_size, 1)]  # Assuming duration is stored in a map with batch size as key
    batch_size = n.batch_size
    op_type = n.stream.cuda_stream 

    y_position = y_positions[op_type]
    
    # Plot the operation as a horizontal bar
    ax.barh(y_position, duration, left=start_time, height=0.8, alpha=0.7, edgecolor="black")
    
    # Annotate with operation name and batch size
    label = f"{n.name}\nB={int(batch_size)}"
    ax.text(start_time + duration / 2, y_position, label, ha="center", va="center", color="black", fontsize=8)

# Set y-ticks and labels
ax.set_yticks(list(y_positions.values()))
ax.set_yticklabels(list(y_positions.keys()))
ax.set_xlabel("Time (s)")
ax.set_ylabel("Operation Type")
ax.set_title("Nano Operations Timeline with Batch Sizes")
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("nano_operations_timeline.png")
# plt.show()

# sort category_nano_op_map by start time
for op_type, nano_ops in category_nano_op_map.items():
    category_nano_op_map[op_type] = sorted(nano_ops, key=lambda x: x.start_time.X)

# create a extra_links
extra_links = {

}

# print sorted category_nano_op_map
for op_type, nano_ops in category_nano_op_map.items():
    print(f"{op_type}: {[f'{n.name}({n.start_time.X:.2f})' for n in nano_ops]}")
    extra_links[(nano_ops[-1].parent.name, nano_ops[0].parent.name)] = (True, False)
    for op, next_op in zip(nano_ops[:-1], nano_ops[1:]):
        prev_layer_dep = False
        next_layer_dep = False
        if op.layer == next_op.layer:
            pass
        elif op.layer == next_op.layer - 1:
            prev_layer_dep = True
        elif op.layer == next_op.layer + 1:
            next_layer_dep = True
        else:
            raise ValueError(f"Unexpected layer order: {op.name}({op.layer}) -> {next_op.name}({next_op.layer})")
        # print(f"Linking {op.name} to {next_op.name}, prev_layer_dep: {prev_layer_dep}, next_layer_dep: {next_layer_dep}")
        extra_links[(op.parent.name, next_op.parent.name)] = (prev_layer_dep, next_layer_dep)

print("extra_links:", extra_links)