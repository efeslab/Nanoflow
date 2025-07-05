import sys
import argparse
sys.path.append("../")
sys.path.append("../utils")
sys.path.append('../pybind/build')

from utils.prof_marker import prof_marker
from utils.frontend import requestManager
from utils.util_functions import prepare_weight
from transformers import AutoTokenizer
from input_test import prefill_context


# from models.llama3_KVCacheTorch import Pipeline
from models.llama3_FlashinferKVCache import Pipeline

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-l", "--load_hf_weight", action="store_true", help="Load weights from huggingface")

args = arg_parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# input_strings = ["Hi, who are you?"]
# input_strings = ["Hi, who are you?", "What's the weather today?"]
input_strings = [ "Hi, who are you?" for _ in range(384)]
# input_strings = [ "The university of washington is located in" for _ in range(16)]
input_ids = [tokenizer.encode(s) for s in input_strings]
# print(input_ids)

# request_queue = []
# request_manager = requestManager(args.trace_path, "meta-llama/Meta-Llama-3-8B-Instruct")
# request_manager.read_request()
# request_manager.release_request()
# # print(request_manager.available_request_queue)

global_batch_size = 1024
decode_batch_size = 384

# new_input_ids = []
# for req in request_manager.available_request_queue:
#     print("req.idx: ", req.req_idx)
#     print("req.prompt: ", req.prompt)
#     print("req.output_len: ", req.output_len)
#     new_input_ids.append((req.req_idx, req.prompt))
# print("new_input_ids: ", new_input_ids)


decode_inputs_ids = [(i, input_ids[i]) for i in range(decode_batch_size)]

prefill_context_ids = tokenizer.encode(prefill_context) # which length is 1066.

weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
# weight_map_wzr = "/code/hf/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/28bd9fa9d94b23cb6ded08f92d5672b2aabe695f"
# weight_map_amd_kan = "/work1/kasikci/kanzhu/models/llama3-8b"
if args.load_hf_weight:
    pipeline_weight_list = [
        (i, f"cuda:{i}", Pipeline()) for i in range(1)
    ]
    prepare_weight(pipeline_weight_list, weight_map_wzr)

pipeline = Pipeline()
pipeline.init(weight_map_wzr, cached=True)

# torch.cuda.empty_cache()
# device = torch.cuda.current_device()
# reserved_memory = torch.cuda.memory_reserved(device)
# print(f"Reserved memory: {reserved_memory / 1024 / 1024} MB")
# pipeline.config()
def test_performance():
    input_length = 1024
    prefill_input_ids = [prefill_context_ids[:input_length] for _ in range(1000)]
    output_strings = {}
    # initialize the reqs for first 384 requests
    decode_inputs = []
    for i in range(decode_batch_size):
        input = [(i, prefill_input_ids[i])]
        pipeline.update(input)
        output_strings[i] = prefill_input_ids[i]
        new_tokens = pipeline.run()
        for _, new_token in new_tokens:
            output_strings[i].extend(new_token)
        decode_inputs.extend(new_tokens)
        print("new_tokens: ", new_tokens)

    output_strings[decode_batch_size] = prefill_input_ids[decode_batch_size]
    decode_inputs.extend([(decode_batch_size, prefill_input_ids[decode_batch_size])])
    pipeline.update(decode_inputs, decode_batch_size)

    for i in range(decode_batch_size, decode_batch_size + 50):
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
            output_strings[next_prefill_idx] = prefill_input_ids[next_prefill_idx]
        with prof_marker(f"after_execute_step_7"):
            new_tokens.extend([(next_prefill_idx, prefill_input_ids[next_prefill_idx])])
        with prof_marker(f"after_execute_step_8"):
            pipeline.update(new_tokens, decode_batchsize)

    output_text = tokenizer.batch_decode(list(output_strings.values())[:1], skip_special_tokens=True)
    print(output_text)

def test_correctness(use_kv_cache=True):
    special_inputs_0 = [(0, input_ids[0]), (1, input_ids[1])]
    special_inputs_1 = [(2, input_ids[2]), (3, input_ids[3])]
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

    output_text = tokenizer.batch_decode(list(output_strings.values()), skip_special_tokens=True)
    print(output_text)

def test_one_cycle():
    special_inputs_0 = [(0, input_ids[0]), (1, input_ids[1])]
    output_strings = {}
    for idx, tensor in special_inputs_0:
        output_strings[idx] = tensor

    pipeline.update(special_inputs_0)
    new_tokens = pipeline.run()
    for req_idx, new_token in new_tokens:
        output_strings[req_idx].extend(new_token)
    
    output_text = tokenizer.batch_decode(list(output_strings.values()), skip_special_tokens=True)
    print(output_text)

def profile_one_cycle():
    pipeline.init_profile_data()

    stream_names = [ f"TEST_{i}" for i in range(len(pipeline.sm_counts)) ] + ["TEST_TOTAL"]
    for stream_name in stream_names:
        print(f"Stream: {stream_name}")
        pipeline.reset()

        # test for prefill
        total_batch_sizes = [128, 256, 384, 512, 640, 768, 896, 1024]
        # total_batch_sizes = [1024]
        for idx, total_batch_size in enumerate(total_batch_sizes):
            input = [(idx, prefill_context_ids[:total_batch_size])]

            pipeline.update(input, is_profile=True, stream_name=stream_name)
            pipeline.profile_run()

        # test for decode
        total_batch_sizes = [128, 256, 384]
        # total_batch_sizes = [384]
        # prepare the decode inputs for a special input_length
        input_length = 1024
        output_length = 512
        prefill_input_ids = [prefill_context_ids[:input_length] for _ in range(1000)]

        for total_batch_size in total_batch_sizes:
            decode_inputs = []
            pipeline.reset()
            for i in range(total_batch_size):
                input = [(i, prefill_input_ids[i])]
                pipeline.update(input, is_profile=True)
                new_tokens = pipeline.run()
                decode_inputs.extend(new_tokens)
                print("new_tokens: ", new_tokens)
                print("total_batch_size: ", total_batch_size)

            # decode profiling from input_length to input_length + output_length
            for i in range(output_length + 1):
                print("Cycle: ", i)
                pipeline.update(decode_inputs, total_batch_size, is_profile=True, stream_name=stream_name)
                if i % 128 == 0:
                    pipeline.profile_run()
                    
    print("All profiling data has been collected.")

test_correctness()
# test_correctness(use_kv_cache=False)
# test_performance()
# test_one_cycle()
# profile_one_cycle()