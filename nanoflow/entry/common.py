import os
import time
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

from nanoflow.utils.util_functions import prepare_weight
from nanoflow.utils.input_test import prefill_context
from nanoflow.pybind.build.bind_all_reduce import NCCLWrapper

@dataclass
class CliArgs:
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    test: str = "correctness"
    model: str = "Llama3-8B"
    kvcache_type: str = "flashinfer" # ["none", "torch", "flashinfer"]
    network_type: str = "allreduce" # ["allreduce", "allgather"]
    use_cuda_graph: bool = False
    use_auto_search: bool = False
    use_nanosplit: bool = False
    affinity_module_path: Optional[str] = None

@dataclass
class ModelArtifacts:
    MODEL_ID: str
    weight_map: str
    Pipeline: type
    cfgs: list
    tokenizer: AutoTokenizer
    auto_search_path: Optional[str]

def setup_model_and_configs(args: CliArgs) -> ModelArtifacts:
    world_size = torch.cuda.device_count()
    print("world size: ", world_size)
    MULTI_GPU_MODE = True
    FULL_DATA_PARALLEL_MODE = args.data_parallel_size == world_size

    unique_nccl_ids = [NCCLWrapper.get_nccl_unique_id() for _ in range(10)]

    if args.model == "Llama3-70B":
        MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
        weight_map = "/code/hf/hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/28bd9fa9d94b23cb6ded08f92d5672b2aabe695f"
        from nanoflow.models.llama3_70B.config_llama3_70B import Llama3_70B_Config as Config
        assert world_size == args.data_parallel_size * args.tensor_parallel_size, "world_size should be equal to data_parallel_size * tensor_parallel_size"
        if args.kvcache_type == "flashinfer":
            if FULL_DATA_PARALLEL_MODE:
                from nanoflow.models.llama3_70B.llama3_70B_FlashinferKVCache import Pipeline
            else:
                if args.network_type == "allreduce":
                    from nanoflow.models.llama3_70B.llama3_70B_FlashinferKVCache_allreduce import Pipeline
                elif args.network_type == "allgather":
                    from nanoflow.models.llama3_70B.llama3_70B_FlashinferKVCache_allgather import Pipeline
                else:
                    raise NotImplementedError(
                        f"Network type {args.network_type} not implemented yet.")
        elif args.kvcache_type == "torch":
            if FULL_DATA_PARALLEL_MODE:
                raise NotImplementedError("Data parallel mode is not supported for torch kvcache type")
            else:
                if args.network_type == "allreduce":
                    from nanoflow.models.llama3_70B.llama3_70B_KVCacheTorch_allreduce import Pipeline
                elif args.network_type == "allgather":
                    from nanoflow.models.llama3_70B.llama3_70B_KVCacheTorch_allgather import Pipeline
                else:
                    raise NotImplementedError(
                        f"Network type {args.network_type} not implemented yet.")
        else:
            raise NotImplementedError(
                f"KVCache type {args.kvcache_type} not implemented yet.")


        cfgs = [Config(
            multi_gpu_mode=MULTI_GPU_MODE,
            world_size=world_size,
            world_rank=i,
            tp_size=args.tensor_parallel_size,
            tp_rank=i % args.tensor_parallel_size,
            dp_size=args.data_parallel_size,
            dp_rank=i // args.tensor_parallel_size,
            kv_cache_type=args.kvcache_type,
            network_type=args.network_type,
            unique_nccl_ids=unique_nccl_ids,
        ) for i in range(world_size)]

        auto_search_path = "/code/Nanoflow-python/nanoflow/auto_search/result_json/prefill_only_search_result_stage1.json"
        # auto_search_path = None

    elif args.model == "Llama3-8B":
        weight_map = "/code/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
        from nanoflow.models.llama3_8B.llama3_8B_FlashinferKVCache_allreduce import Pipeline
        from nanoflow.models.llama3_8B.config_llama3_8B import Llama3_8B_Config as Config
        cfgs = [Config(
            multi_gpu_mode=MULTI_GPU_MODE,
            world_size=world_size,
            world_rank=i,
            tp_size=args.tensor_parallel_size,
            tp_rank=i,
            unique_nccl_ids=unique_nccl_ids,
        ) for i in range(world_size)]
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct")
        auto_search_path = "../auto_search/search_result_json/8B_allreduce_search_result.json"

    elif args.model == "Qwen1.5-MoE-A2.7B-EP":
        MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
        weight_map = "/code/hf/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
        from nanoflow.models.qwen2_moe.qwen2_moe_ep import Pipeline
        from nanoflow.models.qwen2_moe.config_qwen2_moe import Qwen2MoEConfig as Config
        cfgs = [Config(
            multi_gpu_mode=MULTI_GPU_MODE,
            world_size=world_size,
            world_rank=i,
            ep_size=args.expert_parallel_size,
            ep_rank=i,
            unique_nccl_ids=unique_nccl_ids,
        ) for i in range(world_size)]

        auto_search_path = None
    
    elif args.model == "Qwen2-57B-A14B-Instruct-EP":
        MODEL_ID = "Qwen/Qwen2-57B-A14B-Instruct"
        weight_map = "/code/hf/hub/models--Qwen--Qwen2-57B-A14B-Instruct/snapshots/50896d66b39f1425d63720541a66c7df13e053c0"
        from nanoflow.models.qwen2_moe_57B.qwen2_moe_57B_ep import Pipeline
        from nanoflow.models.qwen2_moe_57B.config_qwen2_moe_57B import Qwen2MoEConfig as Config
        cfgs = [Config(
            multi_gpu_mode=MULTI_GPU_MODE,
            world_size=world_size,
            world_rank=i,
            ep_size=args.expert_parallel_size,
            ep_rank=i,
            unique_nccl_ids=unique_nccl_ids,
        ) for i in range(world_size)]

        auto_search_path = None

    elif args.model == "Qwen2-57B-A14B-Instruct-TP-EP":
        MODEL_ID = "Qwen/Qwen2-57B-A14B-Instruct"
        weight_map = "/code/hf/hub/models--Qwen--Qwen2-57B-A14B-Instruct/snapshots/50896d66b39f1425d63720541a66c7df13e053c0"
        from nanoflow.models.qwen2_moe_57B.qwen2_moe_57B_tp_ep import Pipeline
        from nanoflow.models.qwen2_moe_57B.config_qwen2_moe_57B import Qwen2MoEConfig as Config
        assert world_size == args.tensor_parallel_size == args.expert_parallel_size, "world_size should be equal to TP_size and EP_size"
        cfgs = [Config(
            multi_gpu_mode=MULTI_GPU_MODE,
            world_size=world_size,
            world_rank=i,
            tp_size=args.tensor_parallel_size,
            tp_rank=i,
            ep_size=args.expert_parallel_size,
            ep_rank=i,
            unique_nccl_ids=unique_nccl_ids,
        ) for i in range(world_size)]

        auto_search_path = None
    else:
        # from models.llama3_8B_KVCacheFA_TP2 import Pipeline
        raise ValueError("Unsupported model")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("--------------------------------")
    print("MODEL_ID: ", MODEL_ID)
    print("args: ", args.__dict__)
    print("--------------------------------")

    return ModelArtifacts(
        MODEL_ID=MODEL_ID,
        weight_map=weight_map,
        Pipeline=Pipeline,
        cfgs=cfgs,
        tokenizer=tokenizer,
        auto_search_path=auto_search_path,
    )

def ensure_weights(cfgs: list, Pipeline: type, weight_map: str) -> None:
    """Prepare weights if missing (same logic as original)."""
    HAS_CACHED_WEIGHT = cfgs[0].has_cached_weight()
    print("HAS_CACHED_WEIGHT: ", HAS_CACHED_WEIGHT)
    if not HAS_CACHED_WEIGHT:
        pipeline_weight_list = [Pipeline(cfg=cfgs[i]) for i in range(len(cfgs))]
        prepare_weight(pipeline_weight_list, weight_map)

def create_pipelines(cfgs: list, Pipeline: type):
    return [Pipeline(cfg=cfgs[i]) for i in range(len(cfgs))]

def create_shared_variables(world_size: int):
    command = mp.Array("c", 32)
    shared_decode_bts = mp.Value("i", 0)
    use_auto_search = mp.Value("i", 0)
    use_nanosplit = mp.Value("i", 0)
    use_cuda_graph = mp.Value("i", 0)

    # Create a Barrier for world_size workers plus the main process.
    barrier = mp.Barrier(world_size + 1)
    return command, shared_decode_bts, use_auto_search, use_nanosplit, use_cuda_graph, barrier

def world_info():
    return torch.cuda.device_count()

def start_workers(
    T0: float,
    world_size: int,
    affinity_module_path: Optional[str],
    request_queues,
    shared_decode_bts,
    result_queue,
    barrier,
    pipeline_list,
    use_auto_search,
    auto_search_path: Optional[str],
    use_nanosplit,
    use_cuda_graph,
    command,
    worker_fn: Callable,
    ):
    """Spawn one worker per rank with unified argument wiring."""
    processes = []
    for rank in range(world_size):
        args_tuple = (
        T0,
        rank,
        affinity_module_path,
        None if request_queues is None else request_queues[rank],
        shared_decode_bts,
        result_queue,
        barrier,
        pipeline_list[rank],
        use_auto_search,
        auto_search_path,
        use_nanosplit,
        use_cuda_graph,
        command,
        )
        p = mp.Process(target=worker_fn, args=args_tuple)
        p.start()
        processes.append(p)
    
    return processes

def step_barrier(barrier: mp.Barrier) -> None:
    barrier.wait()
    barrier.wait()