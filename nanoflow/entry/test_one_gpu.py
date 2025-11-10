from nanoflow.utils.input_test import prefill_context
from nanoflow.utils.util_functions import prepare_weight
from nanoflow.utils.frontend import requestManager
from nanoflow.utils.prof_marker import prof_marker
from transformers import AutoTokenizer
import argparse
import torch


def test_performance():
    seq_len = 1024
    global_batch_size = 2048
    decode_batch_size = 480
    prefill_batch_size = global_batch_size - decode_batch_size

    prefill_context_ids = tokenizer.encode(prefill_context)

    prefill_input_ids = prefill_context_ids[:seq_len]
    output_strings = {}
    # initialize the reqs for first 384 requests
    decode_inputs = []
    for i in range(decode_batch_size):
        input = [(i, prefill_input_ids.copy())]
        pipeline.update(input)
        output_strings[i] = prefill_input_ids.copy()
        new_tokens = pipeline.run()
        for _, new_token in new_tokens:
            output_strings[i].extend(new_token)
        decode_inputs.extend(new_tokens)
        print("new_tokens: ", new_tokens)

    # prepare for the testing configuration
    output_strings[decode_batch_size] = prefill_context_ids[:prefill_batch_size].copy()
    decode_inputs.extend(
        [(decode_batch_size, prefill_context_ids[:prefill_batch_size].copy())]
    )
    pipeline.update(
        input_infos=decode_inputs,
        decode_batch_size=decode_batch_size,
        next_input_infos=decode_inputs,
        next_decode_batch_size=decode_batch_size,
        profile_result_path=auto_search_path,
        use_auto_search=False,
        use_cuda_graph=False,
        use_nano_split=False,
        plan_double_buffer=True,
    )

    torch.cuda.cudart().cudaProfilerStart()
    for i in range(decode_batch_size, decode_batch_size + 20):
        print("Cycle: ", i - decode_batch_size)
        next_prefill_idx = i + 1
        new_tokens = pipeline.run()
        with prof_marker(f"after_execute_step_4"):
            for req_idx, new_token in new_tokens:
                output_strings[req_idx].extend(new_token)
        # print("new_tokens: ", new_tokens)
        with prof_marker(f"after_execute_step_5"):
            new_tokens = new_tokens[:-1]
            decode_batchsize = len(new_tokens)
            assert decode_batchsize == decode_batch_size
        with prof_marker(f"after_execute_step_6"):
            output_strings[next_prefill_idx] = prefill_context_ids[
                :prefill_batch_size
            ].copy()
        with prof_marker(f"after_execute_step_7"):
            new_tokens.extend(
                [(next_prefill_idx, prefill_context_ids[:prefill_batch_size].copy())]
            )
        with prof_marker(f"after_execute_step_8"):
            pipeline.update(
                input_infos=new_tokens,
                decode_batch_size=decode_batchsize,
                next_input_infos=new_tokens,
                next_decode_batch_size=decode_batchsize,
                profile_result_path=auto_search_path,
                use_auto_search=False,
                use_cuda_graph=False,
                use_nano_split=False,
                double_buffer_enabled=True,
            )

    torch.cuda.cudart().cudaProfilerStop()

    output_text = tokenizer.batch_decode(
        list(output_strings.values())[:1], skip_special_tokens=True
    )
    print(output_text)


def test_correctness(use_kv_cache=True):
    # input_strings = ["Hi, who are you?"]
    # input_strings = ["Hi, who are you?", "What's the weather today?"]
    input_string = "Hi, who are you?"
    # input_strings = [ "The university of washington is located in" for _ in range(16)]
    input_ids = tokenizer.encode(input_string)
    # print(input_ids)
    special_inputs_0 = [(0, input_ids.copy()), (1, input_ids.copy())]
    special_inputs_1 = [(2, input_ids.copy()), (3, input_ids.copy())]
    output_strings = {}
    for idx, tensor in special_inputs_0:
        output_strings[idx] = tensor

    for idx, tensor in special_inputs_1:
        output_strings[idx] = tensor

    pipeline.update(special_inputs_0)
    new_tokens = pipeline.run()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)
    decode_batchsize = len(new_tokens)
    assert decode_batchsize == 2

    # print("new_tokens: ", new_tokens)
    if use_kv_cache:
        new_tokens.extend(special_inputs_1)
        pipeline.update(new_tokens, decode_batchsize)
    else:
        new_tokens = [(0, output_strings[0]), (1, output_strings[1])]
        new_tokens.extend(special_inputs_1)
        pipeline.update(new_tokens, 0)

    new_tokens = pipeline.run()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)
    decode_batchsize = len(new_tokens)
    assert decode_batchsize == 4
    # print("new_tokens: ", new_tokens)

    if use_kv_cache:
        pipeline.update(new_tokens, decode_batchsize)
    else:
        new_tokens = [(i, output_strings[i]) for i in range(4)]
        pipeline.update(new_tokens, 0)

    for i in range(20):
        print("Cycle: ", i)
        new_tokens = pipeline.run()
        for req_idx, new_token in new_tokens:
            output_strings[req_idx].extend(new_token)
        decode_batchsize = len(new_tokens)
        assert decode_batchsize == 4
        # print("new_tokens: ", new_tokens)
        if use_kv_cache:
            pipeline.update(new_tokens, decode_batchsize)
        else:
            new_tokens = [(i, output_strings[i]) for i in range(4)]
            pipeline.update(new_tokens, 0)

    output_text = tokenizer.batch_decode(
        list(output_strings.values()), skip_special_tokens=True
    )
    print(output_text)


def test_one_cycle():
    input_string = "Hi, who are you?"
    # input_strings = [ "The university of washington is located in" for _ in range(16)]
    input_ids = tokenizer.encode(input_string)
    special_inputs_0 = [(0, input_ids.copy()), (1, input_ids.copy())]
    output_strings = {}
    for idx, tensor in special_inputs_0:
        output_strings[idx] = tensor

    pipeline.update(special_inputs_0)
    new_tokens = pipeline.run()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)

    output_text = tokenizer.batch_decode(
        list(output_strings.values()), skip_special_tokens=True
    )
    print(output_text)


def profile_one_cycle():
    # which length is 1912.
    prefill_context_ids = tokenizer.encode(prefill_context)

    pipeline.init_profile_data()

    stream_names = [f"TEST_{i}" for i in range(len(pipeline.sm_counts))] + [
        "TEST_TOTAL"
    ]
    for stream_name in stream_names:
        print(f"Stream: {stream_name}")
        pipeline.reset()

        # test for prefill
        total_batch_sizes = [
            128,
            256,
            384,
            512,
            640,
            768,
            896,
            1024,
            1152,
            1280,
            1408,
            1536,
            1664,
            1792,
            1920,
            2048,
        ]
        # total_batch_sizes = [1024]
        for idx, total_batch_size in enumerate(total_batch_sizes):
            input = [(idx, prefill_context_ids[:total_batch_size].copy())]

            pipeline.update(input, is_profile=True, stream_name=stream_name)
            pipeline.profile_run()

        # test for decode
        total_batch_sizes = [128, 256, 384, 512, 640]
        # total_batch_sizes = [384]
        # prepare the decode inputs for a special input_length
        input_length = 1024
        output_length = 0
        prefill_input_ids = prefill_context_ids[:input_length]

        for total_batch_size in total_batch_sizes:
            decode_inputs = []
            pipeline.reset()
            for i in range(total_batch_size):
                input = [(i, prefill_input_ids.copy())]
                pipeline.update(input, is_profile=True)
                new_tokens = pipeline.run()
                decode_inputs.extend(new_tokens)
                print("new_tokens: ", new_tokens)
                print("total_batch_size: ", total_batch_size)

            # decode profiling from input_length to input_length + output_length
            for i in range(output_length + 1):
                print("Cycle: ", i)
                pipeline.update(
                    decode_inputs,
                    total_batch_size,
                    is_profile=True,
                    stream_name=stream_name,
                )
                if i % 128 == 0:
                    pipeline.profile_run()

    print("All profiling data has been collected.")


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--test",
    choices=["correctness", "performance", "profile", "one_cycle"],
    default="correctness",
    help="Which test to run",
)
arg_parser.add_argument(
    "--model",
    default="Llama3-8B",
    help="Pick which Pipeline to instantiate",
)
arg_parser.add_argument(
    "--kvcache_type",
    default="flashinfer",
    help="Pick which KVCache to use",
)
args = arg_parser.parse_args()
print("Parse all args: ", args)

# request_queue = []
# request_manager = requestManager(args.trace_path, "meta-llama/Meta-Llama-3-8B-Instruct")
# request_manager.read_request()
# request_manager.release_request()
# # print(request_manager.available_request_queue)

# new_input_ids = []
# for req in request_manager.available_request_queue:
#     print("req.idx: ", req.req_idx)
#     print("req.prompt: ", req.prompt)
#     print("req.output_len: ", req.output_len)
#     new_input_ids.append((req.req_idx, req.prompt))
# print("new_input_ids: ", new_input_ids)

if args.model == "Llama3-8B":
    MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    from nanoflow.models.llama3_8B.config_llama3_8B import Llama3_8B_Config as Config
    if args.kvcache_type == "flashinfer":
        from nanoflow.models.llama3_8B.llama3_FlashinferKVCache import Pipeline
    elif args.kvcache_type == "torch":
        from nanoflow.models.llama3_8B.llama3_KVCacheTorch import Pipeline
    else:
        raise NotImplementedError(
            f"KVCache type {args.kvcache_type} not implemented yet.")
    
    cfg = Config(kv_cache_type=args.kvcache_type)
    weight_map = "/code/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
    auto_search_path = "../auto_search/8B_search_result_large_btz.json"
elif args.model == "Llama3-70B":
    MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
    from nanoflow.models.llama3_70B.config_llama3_70B import Llama3_70B_Config as Config
    if args.kvcache_type == "flashinfer":
        from nanoflow.models.llama3_70B.llama3_70B_FlashinferKVCache import Pipeline
    elif args.kvcache_type == "torch":
        raise NotImplementedError(
            f"KVCache type {args.kvcache_type} not implemented yet.")
    else:
        raise NotImplementedError(
            f"KVCache type {args.kvcache_type} not implemented yet.")
    cfg = Config(kv_cache_type=args.kvcache_type)
    weight_map = "/code/hf/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/28bd9fa9d94b23cb6ded08f92d5672b2aabe695f"
    auto_search_path = None
elif args.model == "Qwen1.5-MoE-A2.7B":
    MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
    from nanoflow.models.qwen2_moe.qwen2_moe import Pipeline
    from nanoflow.models.qwen2_moe.config_qwen2_moe import Qwen2MoEConfig as Config
    cfg = Config()

    weight_map = "/code/hf/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
    auto_search_path = None
elif args.model == "Qwen2-57B-A14B-Instruct":
    MODEL_ID = "Qwen/Qwen2-57B-A14B-Instruct"
    from nanoflow.models.qwen2_moe_57B.qwen2_moe_57B import Pipeline
    from nanoflow.models.qwen2_moe_57B.config_qwen2_moe_57B import Qwen2MoEConfig as Config
    cfg = Config()

    weight_map = "/code/hf/hub/models--Qwen--Qwen2-57B-A14B-Instruct/snapshots/50896d66b39f1425d63720541a66c7df13e053c0"
    auto_search_path = None
else:
    raise NotImplementedError(f"Model {args.model} not implemented yet.")
    # weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/28bd9fa9d94b23cb6ded08f92d5672b2aabe695f"
    # weight_map_amd_kan = "/work1/kasikci/kanzhu/models/llama3-8b"
    # weight_map_yi = "/app/llama3-8b"
print("--------------------------------")
print("Model ID: ", MODEL_ID)
print("--------------------------------")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

HAS_CACHED_WEIGHT = cfg.has_cached_weight()
print("HAS_CACHED_WEIGHT: ", HAS_CACHED_WEIGHT)

if not HAS_CACHED_WEIGHT:
    pipeline_weight_list = [Pipeline(cfg=cfg)]
    prepare_weight(pipeline_weight_list, weight_map)

pipeline = Pipeline(cfg=cfg)
pipeline.init(weight_map, cached=True)

print("Finish initializing the pipeline.")

if args.test == "correctness":
    test_correctness()
    # test_correctness(use_kv_cache=False)
elif args.test == "performance":
    test_performance()
elif args.test == "profile":
    profile_one_cycle()
elif args.test == "one_cycle":
    test_one_cycle()
