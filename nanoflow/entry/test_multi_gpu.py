import time
import torch
import torch.multiprocessing as mp

from nanoflow.entry.common import (
    parse_args,
    setup_model_and_configs,
    ensure_weights,
    create_pipelines,
    create_shared_variables,
    prefill_context,
    start_workers,
    world_info,
    step_barrier,
)

from nanoflow.entry.worker_entry import worker_entry


def test_correctness():
    # Settings
    input_string = "Hi, who are you?"
    input_ids = arts.tokenizer.encode(input_string)
    input0 = [(i, input_ids.copy()) for i in range(2)]
    input1 = [(i, input_ids.copy()) for i in range(2, 4)]

    output_strings = {}
    for idx in range(4):
        output_strings[idx] = input_ids.copy()

    # Execute
    command.value = b"Execute"
    decode_bts.value = 0

    for queue in request_queues:
        queue.put((input0, None))
    step_barrier(barrier)

    new_tokens = result_queue.get()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)

    new_tokens.extend(input1)
    for queue in request_queues:
        queue.put((new_tokens, None))
    decode_bts.value = 2
    iterations = 20
    for i in range(iterations):
        print(f"Iteration {i + 1}/{iterations}")
        # Set the shared task value.
        step_barrier(barrier)
        new_tokens = result_queue.get()
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)
        for queue in request_queues:
            queue.put((new_tokens, None))
        decode_bts.value = 4

    # Terminate
    terminate_workers(processes, barrier)

    output_text = arts.tokenizer.batch_decode(
        list(output_strings.values()), skip_special_tokens=True
    )
    print(output_text)


def test_prefill_only():
    # Settings
    seq_len = 512
    num_prefill_reqs = 128

    prefill_context_ids = arts.tokenizer.encode(prefill_context)
    print("len(prefill_context_ids): ", len(
        prefill_context_ids), "seq_len: ", seq_len)
    assert seq_len <= len(
        prefill_context_ids), f"seq_len {seq_len} should be less than {len(prefill_context_ids)}"
    prefill_input_ids = prefill_context_ids[:seq_len]
    output_strings = {}

    # Execute
    command.value = b"Execute"
    decode_bts.value = 0
    next_decode_bts.value = 0
    auto_search_enabled.value = args.auto_search_enabled
    nano_split_enabled.value = args.nano_split_enabled
    plan_cuda_graph.value = args.plan_cuda_graph
    cuda_graph_enabled.value = args.cuda_graph_enabled
    plan_double_buffer.value = True
    double_buffer_enabled.value = False

    group_prefill_size = 16  # might encounter the illegal memory access issue when group_prefill_size is too large, like group_prefill_size* seq_len == 16384
    cycles = (num_prefill_reqs + group_prefill_size - 1) // group_prefill_size

    for i in range(cycles):
        if i == 1:
            plan_double_buffer.value = False
            double_buffer_enabled.value = True
            torch.cuda.cudart().cudaProfilerStart()
        print(f"Cycle {i + 1}/{cycles}")
        prefill_inputs = []
        next_prefill_inputs_infos = []
        if i == cycles - 1:
            for j in range(i * group_prefill_size, num_prefill_reqs):
                prefill_inputs.append((j, prefill_input_ids.copy()))
                next_prefill_inputs_infos.append(
                    (j + group_prefill_size, seq_len))
                output_strings[j] = prefill_input_ids.copy()
        else:
            for j in range(i * group_prefill_size, (i + 1) * group_prefill_size):
                prefill_inputs.append((j, prefill_input_ids.copy()))
                next_prefill_inputs_infos.append(
                    (j + group_prefill_size, seq_len))
                output_strings[j] = prefill_input_ids.copy()
        for queue in request_queues:
            queue.put_nowait((prefill_inputs, next_prefill_inputs_infos))

        step_barrier(barrier)

        new_tokens = result_queue.get(timeout=1)
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)
        # print("new_tokens: ", new_tokens)

    torch.cuda.cudart().cudaProfilerStop()

    # Terminate
    terminate_workers(processes, barrier)

    output_text = arts.tokenizer.batch_decode(
        list(output_strings.values())[:2], skip_special_tokens=True
    )
    print(output_text)


def test_decode_only():
    # Settings
    seq_len = 1024
    decode_batch_size = 512

    prefill_context_ids = arts.tokenizer.encode(prefill_context)
    print("len(prefill_context_ids): ", len(
        prefill_context_ids), "seq_len: ", seq_len)
    assert seq_len <= len(
        prefill_context_ids), f"seq_len {seq_len} should be less than {len(prefill_context_ids)}"
    prefill_input_ids = prefill_context_ids[:seq_len]

    decode_inputs = []
    next_decode_inputs_infos = []
    output_strings = {}

    # Execute
    command.value = b"Execute"

    group_prefill_size = 4  # might encounter the illegal memory access issue when group_prefill_size is too large, like group_prefill_size* seq_len == 16384
    cycles = (decode_batch_size + group_prefill_size - 1) // group_prefill_size

    for i in range(cycles):
        print(f"Cycle {i + 1}/{cycles}")
        prefill_inputs = []
        if i == cycles - 1:
            for j in range(i * group_prefill_size, decode_batch_size):
                prefill_inputs.append((j, prefill_input_ids.copy()))
                next_decode_inputs_infos.append((j, 1))
                output_strings[j] = prefill_input_ids.copy()
        else:
            for j in range(i * group_prefill_size, (i + 1) * group_prefill_size):
                prefill_inputs.append((j, prefill_input_ids.copy()))
                next_decode_inputs_infos.append((j, 1))
                output_strings[j] = prefill_input_ids.copy()
        for queue in request_queues:
            queue.put_nowait((prefill_inputs, None))

        step_barrier(barrier)

        new_tokens = result_queue.get(timeout=1)
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)
        decode_inputs.extend(new_tokens)
        # print("new_tokens: ", new_tokens)

    # prepare for the testing configuration

    for queue in request_queues:
        queue.put_nowait((decode_inputs, next_decode_inputs_infos))
    decode_bts.value = decode_batch_size
    next_decode_bts.value = decode_batch_size
    auto_search_enabled.value = args.auto_search_enabled
    nano_split_enabled.value = args.nano_split_enabled
    plan_cuda_graph.value = True
    cuda_graph_enabled.value = False
    plan_double_buffer.value = True
    double_buffer_enabled.value = False

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(20):
        print("Cycle: ", i)
        if i == 1:
            plan_cuda_graph.value = False
            cuda_graph_enabled.value = True
            plan_double_buffer.value = False
            double_buffer_enabled.value = True
        # Set the shared task value.
        step_barrier(barrier)
        new_tokens = result_queue.get(timeout=1)
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)

        # print("new_tokens: ", new_tokens)
        assert len(new_tokens) == decode_batch_size

        for queue in request_queues:
            queue.put_nowait((new_tokens, next_decode_inputs_infos))

    torch.cuda.cudart().cudaProfilerStop()

    # Terminate
    terminate_workers(processes, barrier)

    output_text = arts.tokenizer.batch_decode(
        list(output_strings.values())[:2], skip_special_tokens=True
    )
    print(output_text)


def test_performance():
    # Settings
    seq_len = 1024
    # seq_len = 2048
    # global_batch_size = 1024
    global_batch_size = 2048
    # global_batch_size = 3072
    # decode_batch_size = 128
    decode_batch_size = 640
    # decode_batch_size = 1280
    prefill_batch_size = global_batch_size - decode_batch_size

    prefill_context_ids = arts.tokenizer.encode(prefill_context)
    print("len(prefill_context_ids): ", len(
        prefill_context_ids), "seq_len: ", seq_len)
    assert seq_len <= len(
        prefill_context_ids), f"seq_len {seq_len} should be less than {len(prefill_context_ids)}"
    prefill_input_ids = prefill_context_ids[:seq_len]

    decode_inputs = []
    next_decode_inputs_infos = []
    output_strings = {}

    command.value = b"Execute"

    group_prefill_size = 4  # might encounter the illegal memory access issue when group_prefill_size is too large, like group_prefill_size* seq_len == 16384
    cycles = (decode_batch_size + group_prefill_size - 1) // group_prefill_size

    for i in range(cycles):
        print(f"Cycle {i + 1}/{cycles}")
        prefill_inputs = []
        if i == cycles - 1:
            for j in range(i * group_prefill_size, decode_batch_size):
                prefill_inputs.append((j, prefill_input_ids.copy()))
                next_decode_inputs_infos.append((j, 1))
                output_strings[j] = prefill_input_ids.copy()
        else:
            for j in range(i * group_prefill_size, (i + 1) * group_prefill_size):
                prefill_inputs.append((j, prefill_input_ids.copy()))
                next_decode_inputs_infos.append((j, 1))
                output_strings[j] = prefill_input_ids.copy()
        for queue in request_queues:
            queue.put_nowait((prefill_inputs, None))

        step_barrier(barrier)

        new_tokens = result_queue.get(timeout=1)
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)
        decode_inputs.extend(new_tokens)
        # print("new_tokens: ", new_tokens)

    # prepare for the testing configuration
    assert prefill_batch_size <= len(
        prefill_context_ids), f"prefill_batch_size {prefill_batch_size} should be less than {len(prefill_context_ids)}"
    output_strings[decode_batch_size] = prefill_context_ids[:prefill_batch_size].copy()
    decode_inputs.extend(
        [(decode_batch_size, prefill_context_ids[:prefill_batch_size].copy())]
    )
    next_decode_inputs_infos.extend(
        [(decode_batch_size+1, prefill_batch_size)]
    )
    for queue in request_queues:
        queue.put_nowait((decode_inputs, next_decode_inputs_infos))
    decode_bts.value = decode_batch_size
    next_decode_bts.value = decode_batch_size
    auto_search_enabled.value = args.auto_search_enabled
    nano_split_enabled.value = args.nano_split_enabled
    plan_cuda_graph.value = args.plan_cuda_graph
    cuda_graph_enabled.value = args.cuda_graph_enabled
    plan_double_buffer.value = True
    double_buffer_enabled.value = False

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(decode_batch_size, decode_batch_size + 20):
        if i == decode_batch_size + 1:
            plan_double_buffer.value = False
            double_buffer_enabled.value = True
        print("Cycle: ", i - decode_batch_size)
        next_prefill_idx = i + 1
        # Set the shared task value.
        step_barrier(barrier)
        new_tokens = result_queue.get(timeout=1)
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)

        new_tokens = new_tokens[:-1]
        next_decode_inputs_infos = next_decode_inputs_infos[:-1]
        # print("new_tokens: ", new_tokens)
        assert len(new_tokens) == decode_batch_size

        output_strings[next_prefill_idx] = prefill_context_ids[:prefill_batch_size].copy()

        new_tokens.extend(
            [(next_prefill_idx, prefill_context_ids[:prefill_batch_size].copy())]
        )
        next_decode_inputs_infos.extend(
            [(next_prefill_idx+1, prefill_batch_size)]
        )

        for queue in request_queues:
            queue.put_nowait((new_tokens, next_decode_inputs_infos))

    torch.cuda.cudart().cudaProfilerStop()

    # Terminate
    terminate_workers(processes, barrier)

    output_text = arts.tokenizer.batch_decode(
        list(output_strings.values())[:2], skip_special_tokens=True
    )
    print(output_text)


def profile():
    # Execute
    command.value = b"Profile"
    step_barrier(barrier)

    # Terminate
    terminate_workers(processes, barrier)


def terminate_workers(processes, barrier):
    # Terminate
    command.value = b"Terminate"
    step_barrier(barrier)

    for p in processes:
        p.join()

    print("All processes have finished.")


if __name__ == "__main__":
    mp.set_start_method("spawn")

    args = parse_args()
    world_size = world_info()
    arts = setup_model_and_configs(args)

    T0 = time.perf_counter()

    print("import modules, ", time.perf_counter() - T0)

    ensure_weights(arts.cfgs, arts.Pipeline, arts.weight_map)
    pipeline_list = create_pipelines(arts.cfgs, arts.Pipeline)
    command, decode_bts, next_decode_bts, auto_search_enabled, nano_split_enabled, plan_cuda_graph, cuda_graph_enabled, plan_double_buffer, double_buffer_enabled, barrier = create_shared_variables(world_size)
    request_queues = [mp.Queue(maxsize=1000) for _ in range(world_size)]
    result_queue = mp.Queue(maxsize=1000)
    processes = start_workers(
        0.0,
        world_size,
        args.affinity_module_path,
        request_queues,
        decode_bts,
        next_decode_bts,
        result_queue,
        barrier,
        pipeline_list,
        auto_search_enabled,
        arts.auto_search_path,
        nano_split_enabled,
        plan_cuda_graph,
        cuda_graph_enabled,
        plan_double_buffer,
        double_buffer_enabled,
        command,
        worker_entry,
    )

    if args.test == "correctness":
        # optionally: set a global used by test_correctness
        test_correctness()
    elif args.test == "performance":
        test_performance()
    elif args.test == "prefill_only":
        test_prefill_only()
    elif args.test == "decode_only":
        test_decode_only()
    elif args.test == "profile":
        profile()
    else:
        raise NotImplementedError(f"Unsupported test: {args.test}")
