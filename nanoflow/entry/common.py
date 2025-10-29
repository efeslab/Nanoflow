import argparse

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
    auto_search_enabled: bool = False
    nano_split_enabled: bool = False
    plan_cuda_graph: bool = False
    cuda_graph_enabled: bool = False
    plan_double_buffer: bool = False
    double_buffer_enabled: bool = False
    affinity_module_path: Optional[str] = None

@dataclass
class ModelArtifacts:
    MODEL_ID: str
    weight_map: str
    Pipeline: type
    cfgs: list
    tokenizer: AutoTokenizer
    auto_search_path: Optional[str]

def parse_args() -> CliArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--data_parallel_size",
                            type=int, default=1, help="Data parallel size")
    p.add_argument("--tensor_parallel_size",
                            type=int, default=1, help="Tensor parallel size")
    p.add_argument("--expert_parallel_size",
                            type=int, default=1, help="Expert parallel size")
    p.add_argument("--test",
                            type=str, default="correctness", help="Which test to run")
    p.add_argument("--model",
                            type=str, default="8B", help="Pick which Pipeline to instantiate")
    p.add_argument("--kvcache_type",
                            type=str, default="flashinfer", help="Pick which KVCache to use")
    p.add_argument("--network_type",
                            type=str, default="allreduce", help="Pick which network type to use")
    p.add_argument("--auto_search_enabled", action="store_true",
                            help="Auto search enabled")
    p.add_argument("--nano_split_enabled", action="store_true",
                            help="Nanosplit enabled")
    p.add_argument("--plan_cuda_graph", action="store_true",
                            help="Plan CUDA graph")
    p.add_argument("--cuda_graph_enabled", action="store_true",
                            help="CUDA graph enabled")
    p.add_argument("--plan_double_buffer", action="store_true",
                            help="Plan double buffer")
    p.add_argument("--double_buffer_enabled", action="store_true",
                            help="Double buffer enabled")
    p.add_argument("--affinity_module_path",
                            type=str, default=None, help="Affinity module path")
    args = p.parse_args()

    return CliArgs(
        data_parallel_size=args.data_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        test=args.test,
        model=args.model,
        kvcache_type=args.kvcache_type,
        network_type=args.network_type,
        auto_search_enabled=args.auto_search_enabled,
        nano_split_enabled=args.nano_split_enabled,
        plan_cuda_graph=args.plan_cuda_graph,
        cuda_graph_enabled=args.cuda_graph_enabled,
        plan_double_buffer=args.plan_double_buffer,
        double_buffer_enabled=args.double_buffer_enabled,
        affinity_module_path=args.affinity_module_path,
    )

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
        MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
        weight_map = "/code/hf/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a"
        assert world_size == args.data_parallel_size * args.tensor_parallel_size, "world_size should be equal to data_parallel_size * tensor_parallel_size"
        from nanoflow.models.llama3_8B.config_llama3_8B import Llama3_8B_Config as Config
        if FULL_DATA_PARALLEL_MODE:
            from nanoflow.models.llama3_8B.llama3_FlashinferKVCache import Pipeline
        else:
            from nanoflow.models.llama3_8B.llama3_8B_FlashinferKVCache_allreduce import Pipeline
        cfgs = [Config(
            multi_gpu_mode=MULTI_GPU_MODE,
            world_size=world_size,
            world_rank=i,
            tp_size=args.tensor_parallel_size,
            tp_rank=i % args.tensor_parallel_size,
            dp_size=args.data_parallel_size,
            dp_rank=i // args.tensor_parallel_size,
            unique_nccl_ids=unique_nccl_ids,
        ) for i in range(world_size)]
        
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

    # mkdir for profiler
    if args.test == "profile":
        print("Do Profile")
        profile_data_path = cfgs[0].profile_data_path()
        import os
        os.makedirs(profile_data_path, exist_ok=True)

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
    decode_bts = mp.Value("i", 0)
    next_decode_bts = mp.Value("i", 0)
    auto_search_enabled = mp.Value("i", 0)
    nano_split_enabled = mp.Value("i", 0)
    plan_cuda_graph = mp.Value("i", 0)
    cuda_graph_enabled = mp.Value("i", 0)
    plan_double_buffer = mp.Value("i", 0)
    double_buffer_enabled = mp.Value("i", 0)

    # Create a Barrier for world_size workers plus the main process.
    barrier = mp.Barrier(world_size + 1)
    return command, decode_bts, next_decode_bts, auto_search_enabled, nano_split_enabled, plan_cuda_graph, cuda_graph_enabled, plan_double_buffer, double_buffer_enabled, barrier

def world_info():
    return torch.cuda.device_count()

def start_workers(
    T0: float,
    world_size: int,
    affinity_module_path: Optional[str],
    request_queues,
    decode_bts,
    next_decode_bts,
    result_queue,
    barrier,
    pipeline_list,
    auto_search_enabled,
    auto_search_path: Optional[str],
    nano_split_enabled,
    plan_cuda_graph,
    cuda_graph_enabled,
    plan_double_buffer,
    double_buffer_enabled,
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
        decode_bts,
        next_decode_bts,
        result_queue,
        barrier,
        pipeline_list[rank],
        auto_search_enabled,
        auto_search_path,
        nano_split_enabled,
        plan_cuda_graph,
        cuda_graph_enabled,
        plan_double_buffer,
        double_buffer_enabled,
        command,
        )
        p = mp.Process(target=worker_fn, args=args_tuple)
        p.start()
        processes.append(p)
    
    return processes

def step_barrier(barrier: mp.Barrier) -> None:
    barrier.wait()
    barrier.wait()