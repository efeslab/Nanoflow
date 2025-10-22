import json
from matplotlib import pyplot as plt
from gurobipy import GRB
import gurobipy as gp
import itertools
from collections import defaultdict

from nanoflow.models.llama3_8B.llama3_AutoSearch import Pipeline
from nanoflow.models.llama3_8B.llama3_8B_allreduce_AutoSearch import Pipeline as Pipeline_8B_AllReduce
from nanoflow.models.llama3_8B.config_llama3_8B import Llama3_8B_Config

from nanoflow.models.llama3_70B.llama3_70B_allreduce_AutoSearch import Pipeline as Pipeline_70B_AllReduce
from nanoflow.models.llama3_70B.config_llama3_70B import Llama3_70B_Config

# get layered operation
from nanoflow.operations import Operation_Layer

from nanoflow.core.executor import Executor
from nanoflow.core.categoryType import CategoryType
from nanoflow.utils.util_functions import op_name_to_name_idx_layer
from profileAnalysis import getGemvTimeAndSMCount, getByBatchsizeAndSMCount

def stage_one() -> tuple[dict, float]:
    layer_num = 1
    all_layered_ops: list[Operation_Layer] = []
    for op in pipeline.model_operations:
        if op.first_layer_only or op.last_layer_only or op.batch_size == 0:
            continue
        layered_ops = op.children[:layer_num]
        all_layered_ops.extend(layered_ops)
    print("all_layered_ops:", [op.name for op in all_layered_ops])
    print("all_layered_ops original name:", [
        op.original_name for op in all_layered_ops])


    # operations = [op for op in pipeline.op_layers if op.parent.first_layer_only == False and op.parent.last_layer_only == False]

    # print("operations:", [op.name for op in operations])


    for layer_op in all_layered_ops:
        print(layer_op.name)
        # print("prerequisite operations:", [op for op in layer_op.prerequisites])
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

    # create executor
    executor = Executor(all_layered_ops, [i for i in range(layer_num)])
    executor.plan_layer_ordering()

    for layer_op in all_layered_ops:
        print(layer_op.name)
        print("prev_op_layer:", [op.name for op in layer_op.prev_op_layer])
        print("is_depended_on:", layer_op.is_depended_on)


    model_stage_one = gp.Model("pipeline")
    for layer_op in all_layered_ops:
        # add start_time, end_time
        layer_op.initVariablesStageOne(model_stage_one, full_sm_counts)

    # create sequantial constraints
    sequence_nano = {}
    category_nano_op_map: dict[CategoryType,
                            list[Operation_Layer]] = defaultdict(list)
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
                op1.end_time_stage_one <= op2.start_time_stage_one + M * (1 - sequence_nano[op1, op2]),
                name=f"non_overlap_{op1.name}_{op2.name}_when_sequence_nano_1",
            )
            model_stage_one.addConstr(
                op2.end_time_stage_one <= op1.start_time_stage_one + M * sequence_nano[op1, op2],
                name=f"non_overlap_{op1.name}_{op2.name}_when_sequence_nano_2",
            )

    # dependency constraints
    for layer_op in all_layered_ops:
        for dep_op in layer_op.prev_op_layer:
            print("layer_op:", layer_op.name, "dep_op:", dep_op.name)
            model_stage_one.addConstr(
                dep_op.end_time_stage_one <= layer_op.start_time_stage_one,
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
                prev_op.end_time_stage_one == layer_op.start_time_stage_one,
                name=f"fusion_{prev_op.name}_to_{layer_op.name}",
            )
        elif name == "RopeAppend":
            assert (
                len(layer_op.prev_op_layer) == 1
            ), f"Expected only one previous operation for {layer_op.name}, got {len(layer_op.prev_op_layer)}"
            prev_op = layer_op.prev_op_layer[0]
            model_stage_one.addConstr(
                prev_op.end_time_stage_one == layer_op.start_time_stage_one,
                name=f"fusion_{prev_op.name}_to_{layer_op.name}",
            )
        elif name == "LayerNormFFN":
            assert (
                len(layer_op.prev_op_layer) == 1
            ), f"Expected only one previous operation for {layer_op.name}, got {len(layer_op.prev_op_layer)}"
            prev_op = layer_op.prev_op_layer[0]
            model_stage_one.addConstr(
                prev_op.end_time_stage_one == layer_op.start_time_stage_one,
                name=f"fusion_{prev_op.name}_to_{layer_op.name}",
            )
        elif name == "Activation":
            assert (
                len(layer_op.prev_op_layer) == 1
            ), f"Expected only one previous operation for {layer_op.name}, got {len(layer_op.prev_op_layer)}"
            prev_op = layer_op.prev_op_layer[0]
            model_stage_one.addConstr(
                prev_op.end_time_stage_one == layer_op.start_time_stage_one,
                name=f"fusion_{prev_op.name}_to_{layer_op.name}",
            )


    # C_max constraints
    for layer_op in all_layered_ops:
        model_stage_one.addConstr(
            layer_op.end_time_stage_one <= C_max, name=f"C_max_{layer_op.name}")


    model_stage_one.setObjective(C_max, GRB.MINIMIZE)
    model_stage_one.setParam("Threads", 200)
    # model_stage_one.setParam("TimeLimit", 120)
    model_stage_one.optimize()

    # verify the results
    for layer_op in all_layered_ops:
        for dep_op in layer_op.prev_op_layer:
            if dep_op.end_time_stage_one.X > layer_op.start_time_stage_one.X + 0.001:
                print(
                    f"Error: {dep_op.name} end time {dep_op.end_time_stage_one} is greater than {layer_op.name} start time {layer_op.start_time_stage_one}"
                )
            # else:
            #     print(f"Dependency check passed: {dep_op.name} -> {layer_op.name}")

    period_time_stage_one = C_max.X
    print("period_time_stage_one: ", period_time_stage_one)

    # sort category_nano_op_map by start time
    for op_type, nano_ops in category_nano_op_map.items():
        category_nano_op_map[op_type] = sorted(
            nano_ops, key=lambda x: x.start_time_stage_one.X)

    # Apply stream dependencies
    for op_type, nano_ops in category_nano_op_map.items():
        print(
            f"{op_type}: {[f'{n.name}({n.start_time_stage_one.X:.2f})' for n in nano_ops]}")
        nano_ops[0].parent.append_dependency((nano_ops[-1].parent, 1))
        for op, next_op in zip(nano_ops[:-1], nano_ops[1:]):
            print(f"Linking {op.name} to {next_op.name}")
            next_op.parent.append_dependency((op.parent, 0))

    # Assuming you have a list of NanoOperation instances called nano_operations_list
    # and each NanoOperation has the required attributes.

    # Extract unique operation types
    operation_types = sorted(set(str(n.category) for n in all_layered_ops))
    y_positions = {op_type: i for i, op_type in enumerate(operation_types)}

    fig, ax = plt.subplots(figsize=(30, 6))
    output_op_infos_stage_one = {}

    for n in all_layered_ops:
        n_name = n.name
        # Access the optimized values of the variables
        start_time = n.start_time_stage_one.X
        duration = n.duration_map[
            (n.batch_size, full_sm_counts)
        ]  # Assuming duration is stored in a map with batch size as key
        algo_tag = n.algo_tag_map[(n.batch_size, full_sm_counts)]
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
        op_basename, op_batch_idx, op_layer_idx = op_name_to_name_idx_layer(n.name)
        if op_basename not in output_op_infos_stage_one:
            output_op_infos_stage_one[op_basename] = {}
        output_op_infos_stage_one[op_basename][n.parent.name] = {
            "batch_idx": op_batch_idx,
            "batch_size": n.batch_size,
            "start_time": start_time,
            "end_time": n.end_time_stage_one.X,
            "duration_time": duration,
            "p_value": full_sm_counts,
            "algo_tag": algo_tag,
            "extra_dep": [(elem0.name, elem1)
                        for elem0, elem1 in n.parent.extra_dep],
        }

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

    with open(stage1_dump_file, "w") as f:
        json.dump(output_op_infos_stage_one, f, indent=4)

    return output_op_infos_stage_one, period_time_stage_one

def stage_two(output_op_infos_stage_one, period_time_stage_one):
    # second search stage
    layer_num = 3
    second_stage_nano_ops: list[Operation_Layer] = []
    for op in pipeline.model_operations:
        if op.first_layer_only or op.last_layer_only or op.batch_size == 0:
            continue
        # if op.original_name == "LayerNormAttn":
        #     layered_ops = op.children[:2]
        # elif op.original_name == "KQV":
        #     layered_ops = op.children[:2]
        # elif op.original_name == "RopeAppend":
        #     layered_ops = op.children[:2]
        # elif op.original_name == "PFAttn":
        #     layered_ops = op.children[:2]
        # elif op.original_name == "O":
        #     layered_ops = op.children[:2]
        # elif op.original_name == "AllReduceO":
        #     layered_ops = op.children[:2]
        # elif op.original_name == "LayerNormFFN":
        #     layered_ops = op.children[:2]
        # elif op.original_name == "UG":
        #     layered_ops = op.children[:2]
        # elif op.original_name == "Activation":
        #     layered_ops = op.children[:2]
        # elif op.name == "D1":
        #     layered_ops = op.children[:2]
        # elif op.name == "D0":
        #     layered_ops = op.children[:2]
        # elif op.original_name == "AllReduceD":
        #     layered_ops = op.children[:2]
        # else:
        layered_ops = op.children[:layer_num]
        second_stage_nano_ops.extend(layered_ops)
    print("second_stage_nano_ops:", [op.name for op in second_stage_nano_ops])
    # Create sequantial constraints for the second stage
    categories = set(op.category for op in second_stage_nano_ops)

    category_nano_op_map_stage_two: dict[CategoryType, list[Operation_Layer]] = defaultdict(
        list
    )
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
        op.initVariablesStageTwo(second_stage_model, profile_sm_counts, categories, output_op_infos_stage_one, period_time_stage_one)

    # Add constraints for each operation
    for op in second_stage_nano_ops:
        op.addInternalConstraintsStageTwo(second_stage_model)

    # dependency constraints
    for layer_op in second_stage_nano_ops:
        for dep_op in layer_op.prev_op_layer:
            second_stage_model.addConstr(
                dep_op.end_time_stage_two <= layer_op.start_time_stage_two,
                name=f"dependency_{dep_op.name}_before_{layer_op.name}",
            )

    # prepare overlapping constraints
    M = 100
    epsilon = 1e-3  # A small value to avoid numerical issues
    is_overlapping = {}
    delta_maps = {}
    for (type_1, list_1), (type_2, list_2) in itertools.combinations(
        category_nano_op_map_stage_two.items(), 2
    ):
        # print(f"Processing overlapping constraints between {type_1} and {type_2}")
        for op_1, op_2 in itertools.product(list_1, list_2):
            # print(
            #     f"Adding overlapping constraints for {op_1.name} and {op_2.name}")
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
                op_2.end_time_stage_two <= epsilon + op_1.start_time_stage_two + M * delta1,
                name=f"overlap_{op_1.name}_{op_2.name}_when_delta1",
            )
            second_stage_model.addConstr(
                op_2.end_time_stage_two >= epsilon + op_1.start_time_stage_two - M * (1 - delta1),
                name=f"overlap_{op_1.name}_{op_2.name}_when_not_delta1",
            )
            second_stage_model.addConstr(
                op_1.end_time_stage_two <= epsilon + op_2.start_time_stage_two + M * delta2,
                name=f"overlap_{op_1.name}_{op_2.name}_when_delta2",
            )
            second_stage_model.addConstr(
                op_1.end_time_stage_two >= epsilon + op_2.start_time_stage_two - M * (1 - delta2),
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
        for idx in range(num_categories):
            op = combo[idx]
            other_ops = combo[:idx] + combo[idx + 1:]
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
            C_max_stage_two >= op.end_time_stage_two, name=f"makespan_constraint_{op.name}"
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
        category_list.sort(key=lambda x: x.start_time_stage_two.X)

    for list1, list2 in itertools.combinations(category_lists, 2):
        filtered_list1 = [op for op in list1 if op_name_to_name_idx_layer(op.name)[
            2] == 1]
        filtered_list2 = [op for op in list2 if op_name_to_name_idx_layer(op.name)[
            2] == 1]
        print(f"filtered_list1: {[op.name for op in filtered_list1]}")
        print(f"filtered_list2: {[op.name for op in filtered_list2]}")
        for op1 in filtered_list1:
            # filter the elements in list2 that have smaller end_time.X than op1.start_time.X
            filtered_list2_for_op1 = [
                op2 for op2 in list2 if op2.end_time_stage_two.X < op1.start_time_stage_two.X + 2 * epsilon
            ]
            if filtered_list2_for_op1:
                op2 = filtered_list2_for_op1[-1]
                print(
                    f"Attempting to add dependency from {op2.name} to {op1.name}")
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
            filtered_list1_for_op2 = [
                op1 for op1 in list1 if op1.end_time_stage_two.X < op2.start_time_stage_two.X + 2 * epsilon
            ]
            if filtered_list1_for_op2:
                op1 = filtered_list1_for_op2[-1]
                print(
                    f"Attempting to add dependency from {op1.name} to {op2.name}")
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
    operation_types = sorted({str(n.category) for n in second_stage_nano_ops})
    x_positions = {op_type: i for i, op_type in enumerate(operation_types)}

    # Taller figure since time is now vertical
    fig, ax = plt.subplots(figsize=(15, 30))

    for n in second_stage_nano_ops:
        # Values from optimizer
        start_time = n.start_time_stage_two.X
        p_value = round(n.p_choice.X)
        duration = n.duration_map[(n.batch_size, p_value)]
        op_type = str(n.category)
        x = x_positions[op_type]

        # Vertical bar: height=duration, bottom=start_time
        ax.bar(x, duration, bottom=start_time,
            width=0.8, alpha=0.7, edgecolor="black")

        # Put the label rotated along the bar to save horizontal space
        if duration > 0.03:  # Only label if the bar is tall enough
            label = f"{n.name} L{n.layer} SM {p_value}"
        else:
            label = None
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

    # Axes & labels (now swapped)
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(list(x_positions.keys()), rotation=45, ha="right")
    ax.set_ylabel("Time (ms)")
    ax.set_xlabel("Operation Type")
    ax.set_title("Nano Operations Timeline with Batch Sizes (vertical)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(stage2_figure_path)


    output_overlap_map: defaultdict[str, dict[str, int]] = defaultdict(dict)
    output_op_infos_stage_two: dict[str, dict[str, dict]] = {}

    for (op1_name, op2_name), is_overlap in is_overlapping.items():
        if is_overlap.X > 0.5:  # If the operations overlap
            output_overlap_map[op1_name][op2_name] = 1
        else:
            output_overlap_map[op1_name][op2_name] = 0

    for op in second_stage_nano_ops:
        op_basename, op_batch_idx, op_layer_idx = op_name_to_name_idx_layer(
            op.name)
        if op_layer_idx != 1:
            continue
        p_value = op.p_choice.X
        duration = 0.0
        for sm_count in op.p_vars:
            duration += op.duration_map[(op.batch_size,
                                        sm_count)] * op.p_vars[sm_count].X
        algo_tag = op.algo_tag_map[(op.batch_size, p_value)]
        start_time = op.start_time_stage_two.X
        finish_time = op.end_time_stage_two.X
        str_extra_dep = [(elem0.name, elem1)
                        for elem0, elem1 in op.parent.extra_dep]

        op_is_overlapping_with_others = False
        if op.name in output_overlap_map:
            for other_op_name, overlap_value in output_overlap_map[op.name].items():
                if overlap_value == 1:
                    op_is_overlapping_with_others = True
                    break
        if not op_is_overlapping_with_others:
            p_value = full_sm_counts

        if op_basename not in output_op_infos_stage_two:
            output_op_infos_stage_two[op_basename] = {}
        output_op_infos_stage_two[op_basename][op.parent.name] = {
            "batch_idx": op_batch_idx,
            "batch_size": op.batch_size,
            "start_time": op.start_time_stage_two.X,
            "end_time": op.end_time_stage_two.X,
            "duration_time": op.duration_map[(op.batch_size, op.p_choice.X)],
            "p_value": p_value,
            "algo_tag": algo_tag,
            "extra_dep": str_extra_dep,
        }

        print(
            f"{op.name} starts {start_time:.3f} end {start_time + duration:.3f} p {p_value}, duration {duration}"
        )


    # Save the output to a JSON file
    with open(stage2_dump_file, "w") as f:
        json.dump(output_op_infos_stage_two, f, indent=4)


if __name__ == "__main__":
    # global_batch_size = 3072
    # decode_batch_size = 1280
    # global_batch_size = 2048
    # decode_batch_size = 640
    # global_batch_size = 1024
    # decode_batch_size = 384
    global_batch_size = 8192
    decode_batch_size = 0


    seq_len = 1024
    # create operations

    # pipeline = Pipeline(cfg=Llama3_8B_Config())

    # pipeline = Pipeline_8B_AllReduce(cfg=Llama3_8B_Config(
    #     multi_gpu_mode=True, world_size=4, world_rank=0, tp_size=4, tp_rank=0))

    pipeline = Pipeline_70B_AllReduce(cfg=Llama3_70B_Config(
        multi_gpu_mode=True, world_size=4, world_rank=0, tp_size=4, tp_rank=0))

    profile_dir = pipeline.profile_dir

    # stage1_figure_path = f"result_pdf/simple_test_stage1_figure.pdf"
    # stage2_figure_path = f"result_pdf/simple_test_stage2_figure.pdf"
    # stage1_dump_file = f"result_json/simple_test_search_result_stage1.json"
    # stage2_dump_file = f"result_json/simple_test_search_result_stage2.json"

    stage1_figure_path = "result_pdf/prefill_only_stage1_figure.pdf"
    stage2_figure_path = "result_pdf/prefill_only_stage2_figure.pdf"
    stage1_dump_file = "result_json/prefill_only_search_result_stage1.json"
    stage2_dump_file = "result_json/prefill_only_search_result_stage2.json"


    pipeline.init_wo_weight()
    pipeline.global_batch_size = global_batch_size
    pipeline.decode_batch_size = decode_batch_size

    profile_sm_counts = pipeline.sm_counts
    full_sm_counts = pipeline.total_sm
    print("sm_counts: ", profile_sm_counts)
    print("full_sm_counts: ", full_sm_counts)

    pipeline.apply_batch_size()
    pipeline.nanobatch_split()
    pipeline.update_allocate_buffers()
    pipeline.config_streams()

    output_op_infos_stage_one, period_time_stage_one = stage_one()
    stage_two(output_op_infos_stage_one, period_time_stage_one)