import sys

sys.path.append("../")
sys.path.append("../pybind/build/")
import itertools
from collections import defaultdict

from models.llama3_AutoSearch import Pipeline
from models.llama3_8B_allreduce_AutoSearch import Pipeline as Pipeline_8B_AllReduce
from models.llama3_70B_allreduce_AutoSearch import Pipeline as Pipeline_70B_AllReduce
from core.executor import Executor
from core.categoryType import CategoryType
from utils.util_functions import op_name_to_name_idx_layer
from profileAnalysis import getGemvTimeAndSMCount, getByBatchsizeAndSMCount


<<<<<<< HEAD
global_batch_size = 2048
decode_batch_size = 640
=======
global_batch_size = 3072
decode_batch_size = 1280
# global_batch_size = 2048
# decode_batch_size = 640
>>>>>>> origin/master
# global_batch_size = 1024
# decode_batch_size = 384

seq_len = 1024

# create operations

# pipeline = Pipeline()
# stage1_figure_path = "8B_stage1_figure.png"
# stage2_figure_path = "8B_stage2_figure.png"
# dump_file = "8B_search_result.json"
# stage1_figure_path = "8B_stage1_figure_large_btz_reverse_v.pdf"
# stage2_figure_path = "8B_stage2_figure_large_btz_reverse_v.pdf"
# dump_file = "8B_search_result_large_btz_reverse_v.json"


<<<<<<< HEAD
pipeline = Pipeline_70B_AllReduce(TP_idx=0, TP_size=4)
# stage1_figure_path = "70B_stage1_figure.pdf"
# stage2_figure_path = "70B_stage2_figure.pdf"
# dump_file = "70B_search_result.json"
stage1_figure_path = "70B_stage1_figure_reverse_v.pdf"
stage2_figure_path = "70B_stage2_figure_reverse_v.pdf"
dump_file = "70B_search_result_reverse_v.json"
=======
# pipeline = Pipeline_70B_AllReduce(TP_idx=0, TP_size=4)
# stage1_figure_path = "70B_stage1_figure.pdf"
# stage2_figure_path = "70B_stage2_figure.pdf"
# dump_file = "70B_search_result.json"
# stage1_figure_path = "70B_stage1_figure_reverse_v3.pdf"
# stage2_figure_path = "70B_stage2_figure_reverse_v3.pdf"
# dump_file = "70B_search_result_reverse_v3.json"

pipeline = Pipeline_8B_AllReduce(TP_idx=0, TP_size=4)
stage1_figure_path = "8B_allreduce_naive_stage1_figure.pdf"
stage2_figure_path = "8B_allreduce_naive_stage2_figure.pdf"
dump_file = "8B_allreduce_naive_search_result.json"
>>>>>>> origin/master

profile_dir = pipeline.profile_dir


pipeline.init()
pipeline.global_batch_size = global_batch_size
pipeline.decode_batch_size = decode_batch_size
pipeline.config_batch_size()
pipeline.nanobatch_split()
pipeline.update_allocate_buffers()

# set streams
pipeline.config_streams()

# get layered operation
from operations.operation_base import Operation_Layer

layer_num = 1
all_layered_ops: list[Operation_Layer] = []
for op in pipeline.model_operations:
    if op.first_layer_only or op.last_layer_only:
        continue
    layered_ops = op.children[:layer_num]
    all_layered_ops.extend(layered_ops)
print("all_layered_ops:", [op.name for op in all_layered_ops])
print("all_layered_ops original name:", [op.original_name for op in all_layered_ops])


# operations = [op for op in pipeline.op_layers if op.parent.first_layer_only == False and op.parent.last_layer_only == False]

# print("operations:", [op.name for op in operations])


profile_sm_counts = pipeline.sm_counts
full_sm_counts = profile_sm_counts[-1]
print("sm_counts: ", profile_sm_counts)
print("full_sm_counts: ", full_sm_counts)

for layer_op in all_layered_ops:
    print(layer_op.name)
    print("prerequisite operations:", [op for op in layer_op.prerequisites])
    if layer_op.original_name == "DecAttn":
        batch_size = layer_op.batch_size
        print("batch_size:", batch_size)
        for sm_count in profile_sm_counts:
            print("sm_count:", sm_count)
            algo_tag, duration = getGemvTimeAndSMCount(
                profile_dir, batch_size, seq_len, sm_count
            )
            layer_op.duration_map[(batch_size, sm_count)] = duration
            layer_op.algo_tag_map[(batch_size, sm_count)] = algo_tag
    else:
        batch_size = layer_op.batch_size
        print("batch_size:", batch_size)
        for sm_count in profile_sm_counts:
            print("sm_count:", sm_count)
            algo_tag, duration = getByBatchsizeAndSMCount(
                profile_dir, layer_op.original_name, batch_size, sm_count
            )
            layer_op.duration_map[(batch_size, sm_count)] = duration
            layer_op.algo_tag_map[(batch_size, sm_count)] = algo_tag

# create executor
executor = Executor(all_layered_ops, [i for i in range(layer_num)])
executor.plan_layer_ordering()

for layer_op in all_layered_ops:
    print(layer_op.name)
    print("prev_op_layer:", [op.name for op in layer_op.prev_op_layer])
    print("is_depended_on:", layer_op.is_depended_on)

import gurobipy as gp
from gurobipy import GRB

model_stage_one = gp.Model("pipeline")
for layer_op in all_layered_ops:
    layer_op.initVariables(model_stage_one, full_sm_counts)  # add start_time, end_time

# create sequantial constraints
sequence_nano = {}
category_nano_op_map: dict[CategoryType, list[Operation_Layer]] = defaultdict(list)
for layer_op in all_layered_ops:
    category_nano_op_map[layer_op.category].append(layer_op)

for nano_ops in category_nano_op_map.values():
    for op_1, op_2 in itertools.combinations(nano_ops, 2):
        sequence_nano[op_1, op_2] = model_stage_one.addVar(
            vtype=GRB.BINARY, name=f"{op_1.name}_before_{op_2.name}"
        )

# the completion time of all operations
C_max = model_stage_one.addVar(vtype=GRB.CONTINUOUS, name="C_max")

# non-overlapping constraints
M = 100

for type_name, nano_op_list in category_nano_op_map.items():
    for op1, op2 in itertools.combinations(nano_op_list, 2):
        model_stage_one.addConstr(
            op1.end_time <= op2.start_time + M * (1 - sequence_nano[op1, op2]),
            name=f"non_overlap_{op1.name}_{op2.name}_when_sequence_nano_1",
        )
        model_stage_one.addConstr(
            op2.end_time <= op1.start_time + M * sequence_nano[op1, op2],
            name=f"non_overlap_{op1.name}_{op2.name}_when_sequence_nano_2",
        )

# dependency constraints
for layer_op in all_layered_ops:
    for dep_op in layer_op.prev_op_layer:
        print("layer_op:", layer_op.name, "dep_op:", dep_op.name)
        model_stage_one.addConstr(
            dep_op.end_time <= layer_op.start_time,
            name=f"dependency_{dep_op.name}_before_{layer_op.name}",
        )

# fusion constraints
for layer_op in all_layered_ops:
    name = layer_op.original_name
    if name == "KQV":
        assert (
            len(layer_op.prev_op_layer) == 1
        ), f"Expected only one previous operation for {layer_op.name}, got {len(layer_op.prev_op_layer)}"
        prev_op = layer_op.prev_op_layer[0]
        model_stage_one.addConstr(
            prev_op.end_time == layer_op.start_time,
            name=f"fusion_{prev_op.name}_to_{layer_op.name}",
        )
    elif name == "RopeAppend":
        assert (
            len(layer_op.prev_op_layer) == 1
        ), f"Expected only one previous operation for {layer_op.name}, got {len(layer_op.prev_op_layer)}"
        prev_op = layer_op.prev_op_layer[0]
        model_stage_one.addConstr(
            prev_op.end_time == layer_op.start_time,
            name=f"fusion_{prev_op.name}_to_{layer_op.name}",
        )
    elif name == "LayerNormFFN":
        assert (
            len(layer_op.prev_op_layer) == 1
        ), f"Expected only one previous operation for {layer_op.name}, got {len(layer_op.prev_op_layer)}"
        prev_op = layer_op.prev_op_layer[0]
        model_stage_one.addConstr(
            prev_op.end_time == layer_op.start_time,
            name=f"fusion_{prev_op.name}_to_{layer_op.name}",
        )
    elif name == "Activation":
        assert (
            len(layer_op.prev_op_layer) == 1
        ), f"Expected only one previous operation for {layer_op.name}, got {len(layer_op.prev_op_layer)}"
        prev_op = layer_op.prev_op_layer[0]
        model_stage_one.addConstr(
            prev_op.end_time == layer_op.start_time,
            name=f"fusion_{prev_op.name}_to_{layer_op.name}",
        )


# C_max constraints
for layer_op in all_layered_ops:
    model_stage_one.addConstr(layer_op.end_time <= C_max, name=f"C_max_{layer_op.name}")


model_stage_one.setObjective(C_max, GRB.MINIMIZE)
model_stage_one.setParam("Threads", 200)
# model_stage_one.setParam("TimeLimit", 120)
model_stage_one.optimize()

for layer_op in all_layered_ops:
    print(layer_op)

# verify the results
for layer_op in all_layered_ops:
    for dep_op in layer_op.prev_op_layer:
        if dep_op.end_time.X > layer_op.start_time.X + 0.001:
            print(
                f"Error: {dep_op.name} end time {dep_op.end_time} is greater than {layer_op.name} start time {layer_op.start_time}"
            )
        # else:
        #     print(f"Dependency check passed: {dep_op.name} -> {layer_op.name}")


from matplotlib import pyplot as plt

# Assuming you have a list of NanoOperation instances called nano_operations_list
# and each NanoOperation has the required attributes.

# Extract unique operation types
operation_types = sorted(set(str(n.category) for n in all_layered_ops))
y_positions = {op_type: i for i, op_type in enumerate(operation_types)}

fig, ax = plt.subplots(figsize=(30, 6))

for n in all_layered_ops:
    n_name = n.name
    # Access the optimized values of the variables
    start_time = n.start_time.X
    duration = n.duration_map[
        (n.batch_size, full_sm_counts)
    ]  # Assuming duration is stored in a map with batch size as key
    batch_size = n.batch_size
    op_type = str(n.category)

    y_position = y_positions[op_type]

    # Plot the operation as a horizontal bar
    ax.barh(
        y_position, duration, left=start_time, height=0.8, alpha=0.7, edgecolor="black"
    )

    # Annotate with operation name and batch size
    label = f"{n.name}\nB={int(batch_size)}"
    ax.text(
        start_time + duration / 2,
        y_position,
        label,
        ha="center",
        va="center",
        color="black",
        fontsize=8,
    )

# Set y-ticks and labels
ax.set_yticks(list(y_positions.values()))
ax.set_yticklabels(list(y_positions.keys()))
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Operation Type")
ax.set_title("Nano Operations Timeline with Batch Sizes")
ax.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(stage1_figure_path)
# plt.show()

# sort category_nano_op_map by start time
for op_type, nano_ops in category_nano_op_map.items():
    category_nano_op_map[op_type] = sorted(nano_ops, key=lambda x: x.start_time.X)

# Apply stream dependencies
for op_type, nano_ops in category_nano_op_map.items():
    print(f"{op_type}: {[f'{n.name}({n.start_time.X:.2f})' for n in nano_ops]}")
    nano_ops[0].parent.append_dependency((nano_ops[-1].parent, 1))
    for op, next_op in zip(nano_ops[:-1], nano_ops[1:]):
        print(f"Linking {op.name} to {next_op.name}")
        next_op.parent.append_dependency((op.parent, 0))
        # print(f"Linking {op.name} to {next_op.name}, prev_layer_dep: {prev_layer_dep}, next_layer_dep: {next_layer_dep}")

# second search stage
layer_num = 3
second_stage_nano_ops: list[Operation_Layer] = []
for op in pipeline.model_operations:
    if op.first_layer_only or op.last_layer_only:
        continue
    layered_ops = op.children[:layer_num]
    second_stage_nano_ops.extend(layered_ops)
print("second_stage_nano_ops:", [op.name for op in second_stage_nano_ops])
print(
    "second_stage_nano_ops original name:",
    [op.original_name for op in second_stage_nano_ops],
)

# Create sequantial constraints for the second stage
categories = set(op.category for op in second_stage_nano_ops)

<<<<<<< HEAD
category_nano_op_map_stage_two: dict[CategoryType, list[Operation_Layer]] = defaultdict(list)
=======
category_nano_op_map_stage_two: dict[CategoryType, list[Operation_Layer]] = defaultdict(
    list
)
>>>>>>> origin/master
for layer_op in second_stage_nano_ops:
    category_nano_op_map_stage_two[layer_op.category].append(layer_op)

# create executor
executor2 = Executor(second_stage_nano_ops, [i for i in range(layer_num)])
executor2.plan_layer_ordering()

for layer_op in second_stage_nano_ops:
    print(layer_op.name)
    print("prev_op_layer:", [op.name for op in layer_op.prev_op_layer])
    print("is_depended_on:", layer_op.is_depended_on)

for layer_op in second_stage_nano_ops:
    print(layer_op.name)
    if layer_op.original_name == "DecAttn":
        batch_size = layer_op.batch_size
        # print("batch_size:", batch_size)
        for sm_count in profile_sm_counts:
            # print("sm_count:", sm_count)
            algo_tag, duration = getGemvTimeAndSMCount(
                profile_dir, batch_size, seq_len, sm_count
            )
            layer_op.duration_map[(batch_size, sm_count)] = duration
            layer_op.algo_tag_map[(batch_size, sm_count)] = algo_tag
            # print("duration_map:", layer_op.duration_map)
    else:
        batch_size = layer_op.batch_size
        # print("batch_size:", batch_size)
        for sm_count in profile_sm_counts:
            # print("sm_count:", sm_count)
            algo_tag, duration = getByBatchsizeAndSMCount(
                profile_dir, layer_op.original_name, batch_size, sm_count
            )
            layer_op.duration_map[(batch_size, sm_count)] = duration
            layer_op.algo_tag_map[(batch_size, sm_count)] = algo_tag
            # print("duration_map:", layer_op.duration_map)

# Create the second stage model
second_stage_model = gp.Model("SecondStageOptimization")

# Initialize variables and constraints for each NanoOperationSecondStage
for op in second_stage_nano_ops:
    print(f"Initializing variables for operation: {op.name}")
    op.initVariablesStageTwo(second_stage_model, profile_sm_counts, categories)

# Add constraints for each operation
for op in second_stage_nano_ops:
    op.addInternalConstraintsStageTwo(second_stage_model)

# dependency constraints
for layer_op in second_stage_nano_ops:
    for dep_op in layer_op.prev_op_layer:
        print("layer_op:", layer_op.name, "dep_op:", dep_op.name)
        second_stage_model.addConstr(
            dep_op.end_time <= layer_op.start_time,
            name=f"dependency_{dep_op.name}_before_{layer_op.name}",
        )


# prepare overlapping constraints
M = 10
epsilon = 1e-3  # A small value to avoid numerical issues
is_overlapping = {}
delta_maps = {}
for (type_1, list_1), (type_2, list_2) in itertools.combinations(
    category_nano_op_map_stage_two.items(), 2
):
    print(f"Processing overlapping constraints between {type_1} and {type_2}")
    for op_1, op_2 in itertools.product(list_1, list_2):
        print(f"Adding overlapping constraints for {op_1.name} and {op_2.name}")
        is_overlap = second_stage_model.addVar(
            vtype=GRB.BINARY, name=f"{op_1.name}_overlap_{op_2.name}"
        )
        is_overlapping[(op_1.name, op_2.name)] = is_overlap
        is_overlapping[(op_2.name, op_1.name)] = is_overlap  # Ensure symmetry
        delta1 = second_stage_model.addVar(
            vtype=GRB.BINARY, name=f"{op_1.name}_{op_2.name}_delta_1"
        )
        delta1.Start = 1
        delta2 = second_stage_model.addVar(
            vtype=GRB.BINARY, name=f"{op_1.name}_{op_2.name}_delta_2"
        )
        delta2.Start = 1
        delta_maps[(op_1.name, op_2.name)] = (delta1, delta2)
        delta_maps[(op_2.name, op_1.name)] = (delta1, delta2)  # Ensure symmetry
        # Add constraints
        second_stage_model.addConstr(
            op_2.end_time <= epsilon + op_1.start_time + M * delta1,
            name=f"overlap_{op_1.name}_{op_2.name}_when_delta1",
        )
        second_stage_model.addConstr(
            op_2.end_time >= epsilon + op_1.start_time - M * (1 - delta1),
            name=f"overlap_{op_1.name}_{op_2.name}_when_not_delta1",
        )
        second_stage_model.addConstr(
            op_1.end_time <= epsilon + op_2.start_time + M * delta2,
            name=f"overlap_{op_1.name}_{op_2.name}_when_delta2",
        )
        second_stage_model.addConstr(
            op_1.end_time >= epsilon + op_2.start_time - M * (1 - delta2),
            name=f"overlap_{op_1.name}_{op_2.name}_when_not_delta2",
        )

        # add is_overlapping constraints
        second_stage_model.addConstr(
            is_overlap <= delta1,
            name=f"is_overlapping_{op_1.name}_{op_2.name}_when_delta1",
        )
        second_stage_model.addConstr(
            is_overlap <= delta2,
            name=f"is_overlapping_{op_1.name}_{op_2.name}_when_delta2",
        )
        second_stage_model.addConstr(
            is_overlap >= delta1 + delta2 - 1,
            name=f"is_overlapping_{op_1.name}_{op_2.name}_when_not_delta1_and_not_delta2",
        )

# resource constraints
category_lists = list(category_nano_op_map_stage_two.values())
num_categories = len(category_lists)
print("Number of categories:", num_categories)

for combo in itertools.product(*category_lists):
    print(
        f"Processing resource constraints for combination: {[op.name for op in combo]}"
    )
    for idx in range(num_categories):
        op = combo[idx]
<<<<<<< HEAD
        other_ops = combo[:idx] + combo[idx+1:]
=======
        other_ops = combo[:idx] + combo[idx + 1 :]
>>>>>>> origin/master
        # print(f"Adding resource constraints for {op.name} in category {idx}")
        # print(f"Other operations in the combination: {[other_op.name for other_op in other_ops]}")
        # Add constraints for each operation in the combination
        second_stage_model.addConstr(
            op.p_choice
            + gp.quicksum(
                other_op.p_choice * is_overlapping[(op.name, other_op.name)]
                for other_op in other_ops
            )
            <= full_sm_counts,
            name=f"resource_constraint_{op.name}_category_{idx}",
        )


# Makespan constraints
C_max_stage_two = second_stage_model.addVar(vtype=GRB.CONTINUOUS, name="C_max")
for op in second_stage_nano_ops:
    second_stage_model.addConstr(
        C_max_stage_two >= op.end_time, name=f"makespan_constraint_{op.name}"
    )

# Objective: Minimize makespan
second_stage_model.setObjective(C_max_stage_two, GRB.MINIMIZE)
# Optimize the model
second_stage_model.setParam("Threads", 200)
# second_stage_model.setParam("Heuristics", 0.1)

second_stage_model.optimize()

# Assuming you have a list of NanoOperation instances called nano_operations_list
# and each NanoOperation has the required attributes.

# set some extra dependencies
for category_list in category_lists:
    # sort this list by the element.start_time.X
    category_list.sort(key=lambda x: x.start_time.X)

for list1, list2 in itertools.combinations(category_lists, 2):
    filtered_list1 = [op for op in list1 if op_name_to_name_idx_layer(op.name)[2] == 1]
    filtered_list2 = [op for op in list2 if op_name_to_name_idx_layer(op.name)[2] == 1]
    print(f"filtered_list1: {[op.name for op in filtered_list1]}")
    print(f"filtered_list2: {[op.name for op in filtered_list2]}")
    for op1 in filtered_list1:
        # filter the elements in list2 that have smaller end_time.X than op1.start_time.X
<<<<<<< HEAD
        filtered_list2_for_op1 = [op2 for op2 in list2 if op2.end_time.X < op1.start_time.X + 2*epsilon]
=======
        filtered_list2_for_op1 = [
            op2 for op2 in list2 if op2.end_time.X < op1.start_time.X + 2 * epsilon
        ]
>>>>>>> origin/master
        if filtered_list2_for_op1:
            op2 = filtered_list2_for_op1[-1]
            print(f"Attempting to add dependency from {op2.name} to {op1.name}")
            if not (
                op1.is_extra_linked_before_op[op2.category]
                or op2.is_extra_linked_after_op[op1.category]
            ):
                print(f"Adding dependency from {op2.name} to {op1.name}")
                op2_layer = op_name_to_name_idx_layer(op2.name)[2]
                prev_layer_flag = 0
                if op2_layer == 0:
                    prev_layer_flag = 1
                elif op2_layer == 2:
                    prev_layer_flag = -1
                op1.parent.append_dependency((op2.parent, prev_layer_flag))
                op1.is_extra_linked_before_op[op2.category] = True
                op2.is_extra_linked_after_op[op1.category] = True

    for op2 in filtered_list2:
        # filter the elements in list1 that have smaller end_time.X than op2.start_time.X
<<<<<<< HEAD
        filtered_list1_for_op2 = [op1 for op1 in list1 if op1.end_time.X < op2.start_time.X + 2*epsilon]
=======
        filtered_list1_for_op2 = [
            op1 for op1 in list1 if op1.end_time.X < op2.start_time.X + 2 * epsilon
        ]
>>>>>>> origin/master
        if filtered_list1_for_op2:
            op1 = filtered_list1_for_op2[-1]
            print(f"Attempting to add dependency from {op1.name} to {op2.name}")
            if not (
                op2.is_extra_linked_before_op[op1.category]
                or op1.is_extra_linked_after_op[op2.category]
            ):
                print(f"Adding dependency from {op1.name} to {op2.name}")
                op1_layer = op_name_to_name_idx_layer(op1.name)[2]
                prev_layer_flag = 0
                if op1_layer == 0:
                    prev_layer_flag = 1
                elif op1_layer == 2:
                    prev_layer_flag = -1
                op2.parent.append_dependency((op1.parent, prev_layer_flag))
                op2.is_extra_linked_before_op[op1.category] = True
                op1.is_extra_linked_after_op[op2.category] = True

# Extract unique operation types (categories) -> x axis
operation_types = sorted({str(n.category) for n in all_layered_ops})
x_positions = {op_type: i for i, op_type in enumerate(operation_types)}

# Taller figure since time is now vertical
fig, ax = plt.subplots(figsize=(15, 30))

for n in second_stage_nano_ops:
    # Values from optimizer
    start_time = n.start_time.X
    p_value = round(n.p_choice.X)
    duration = n.duration_map[(n.batch_size, p_value)]
    op_type = str(n.category)
    x = x_positions[op_type]

    # Vertical bar: height=duration, bottom=start_time
    ax.bar(x, duration, bottom=start_time, width=0.8, alpha=0.7, edgecolor="black")

    # Put the label rotated along the bar to save horizontal space
    if duration > 0.03:  # Only label if the bar is tall enough
        label = f"{n.name} L{n.layer} SM {p_value}"
    else:
        label = None
<<<<<<< HEAD
    ax.text(x, start_time + duration/2, label,
            ha="center", va="center", rotation=0, fontsize=10, color="black")
=======
    ax.text(
        x,
        start_time + duration / 2,
        label,
        ha="center",
        va="center",
        rotation=0,
        fontsize=10,
        color="black",
    )
>>>>>>> origin/master

# Axes & labels (now swapped)
ax.set_xticks(list(x_positions.values()))
ax.set_xticklabels(list(x_positions.keys()), rotation=45, ha="right")
ax.set_ylabel("Time (ms)")
ax.set_xlabel("Operation Type")
ax.set_title("Nano Operations Timeline with Batch Sizes (vertical)")
ax.grid(True, axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(stage2_figure_path)

# # Extract unique operation types
# operation_types = sorted(set(str(n.category) for n in all_layered_ops))
# y_positions = {op_type: i for i, op_type in enumerate(operation_types)}

# fig, ax = plt.subplots(figsize=(30, 6))

# for n in second_stage_nano_ops:
#     n_name = n.name
#     # Access the optimized values of the variables
#     start_time = n.start_time.X
#     duration = n.duration_map[(n.batch_size, n.p_choice.X)]  # Assuming duration is stored in a map with batch size as key
#     batch_size = n.batch_size
#     op_type = str(n.category)

#     y_position = y_positions[op_type]
<<<<<<< HEAD
    
#     # Plot the operation as a horizontal bar
#     ax.barh(y_position, duration, left=start_time, height=0.8, alpha=0.7, edgecolor="black")
    
#     # Annotate with operation name and batch size
#     label = f"{n.name}\nL{n.layer}\nP {n.p_choice.X}\n"
#     ax.text(start_time + duration / 2, y_position, label, ha="center", va="center", color="black", fontsize=12)
    
=======

#     # Plot the operation as a horizontal bar
#     ax.barh(y_position, duration, left=start_time, height=0.8, alpha=0.7, edgecolor="black")

#     # Annotate with operation name and batch size
#     label = f"{n.name}\nL{n.layer}\nP {n.p_choice.X}\n"
#     ax.text(start_time + duration / 2, y_position, label, ha="center", va="center", color="black", fontsize=12)

>>>>>>> origin/master
# # Set y-ticks and labels
# ax.set_yticks(list(y_positions.values()))
# ax.set_yticklabels(list(y_positions.keys()))
# ax.set_xlabel("Time (ms)")
# ax.set_ylabel("Operation Type")
# ax.set_title("Nano Operations Timeline with Batch Sizes")
# ax.grid(True, linestyle="--", alpha=0.6)

# plt.tight_layout()
# plt.savefig(stage2_figure_path)

output_overlap_map: defaultdict[str, dict[str, int]] = defaultdict(dict)
output_op_infos: dict[str, dict[str, dict]] = {}

for (op1_name, op2_name), is_overlap in is_overlapping.items():
    if is_overlap.X > 0.5:  # If the operations overlap
        output_overlap_map[op1_name][op2_name] = 1
    else:
        output_overlap_map[op1_name][op2_name] = 0

for op in second_stage_nano_ops:
    op_basename, op_batch_idx, op_layer_idx = op_name_to_name_idx_layer(op.name)
    if op_layer_idx != 1:
        continue
    p_value = op.p_choice.X
    duration = 0.0
    for sm_count in op.p_vars:
        duration += op.duration_map[(op.batch_size, sm_count)] * op.p_vars[sm_count].X
    algo_tag = op.algo_tag_map[(op.batch_size, p_value)]
    start_time = op.start_time.X
    finish_time = op.end_time.X
    str_extra_dep = [(elem0.name, elem1) for elem0, elem1 in op.parent.extra_dep]

    op_is_overlapping_with_others = False
    if op.name in output_overlap_map:
        for other_op_name, overlap_value in output_overlap_map[op.name].items():
            if overlap_value == 1:
                op_is_overlapping_with_others = True
                break
    if not op_is_overlapping_with_others:
        p_value = full_sm_counts

    if op_basename not in output_op_infos:
        output_op_infos[op_basename] = {}
    output_op_infos[op_basename][op.parent.name] = {
        "batch_idx": op_batch_idx,
        "batch_size": op.batch_size,
        "start_time": op.start_time.X,
        "end_time": op.end_time.X,
        "duration_time": op.duration_map[(op.batch_size, op.p_choice.X)],
        "p_value": p_value,
        "algo_tag": algo_tag,
        "extra_dep": str_extra_dep,
    }

    print(
        f"{op.name} starts {start_time:.3f} end {start_time + duration:.3f} p {p_value}, duration {duration}"
    )


import json

# Save the output to a JSON file
<<<<<<< HEAD
output_data = {
    "operations": output_op_infos
}
with open(dump_file, "w") as f:
    json.dump(output_data, f, indent=4)

print("delta_maps of LayerNormAttn0_0 and PFAttn_0:", delta_maps.get(("LayerNormAttn0_0", "PFAttn_0"), None))
print(is_overlapping.get(("LayerNormAttn0_0", "PFAttn_0"), None))
D1_0_layer_op = pipeline.d.nano_ops[1].children[0]
print("D1_0_layer_op:", D1_0_layer_op.name)
print("D1_0_layer_op start time:", D1_0_layer_op.start_time.X
      , "end time:", D1_0_layer_op.end_time.X, "p_choice:", D1_0_layer_op.p_choice.X
      , "duration:", D1_0_layer_op.duration_map[(D1_0_layer_op.batch_size, D1_0_layer_op.p_choice.X)])
=======
output_data = {"operations": output_op_infos}
with open(dump_file, "w") as f:
    json.dump(output_data, f, indent=4)
>>>>>>> origin/master
