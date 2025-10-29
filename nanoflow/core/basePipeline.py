from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Optional
import copy

import torch

from nanoflow.kvcache.kv import KVCacheNone, KVCacheTorch, BatchedDistKVCache

from nanoflow.core import WeightManager, CategoryType
from nanoflow.core.bufferAllocate import BufferAllocator
from nanoflow.core.executor import Executor
from nanoflow.core.nanobatchSplit import split_nanobatch
from nanoflow.operations import Operations, Operation_Layer, NanoOpInfo
from nanoflow.utils.green_ctx import split_device_green_ctx_by_sm_count
from nanoflow.utils.prof_marker import prof_marker


class BasePipeline(ABC):
    """
    A reusable base class for NanoFlow-style pipelines.

    Subclasses implement model-specific graph construction, weights, and op updates
    via small, well-scoped hooks (see abstract methods below).
    """

    # --------- Required subclass API (hooks) ---------

    @abstractmethod
    def init_external_data(self) -> None:
        """Create external data structures (e.g., KV caches, pools)."""

    @abstractmethod
    def init_operations(self) -> None:
        """Instantiate ops and expand per-layer children into self.all_layer_operations."""

    @abstractmethod
    def init_dependency(self) -> None:
        """Wire op inputs/outputs together and call operation.checkConnection()."""

    @abstractmethod
    def apply_batch_size(self) -> None:
        """Apply batch sizes to key ops (e.g., setBatchSize on inputs/attention ops)."""

    @abstractmethod
    def config_algorithm(self) -> None:
        """Choose algorithms per-op (or from profile)."""

    @abstractmethod
    def post_update_ops(
        self,
        input_req_idx: list[int],
        input_tensor: torch.Tensor,
        cumsum_input: list[int],
        decode_batch_size: int,
        double_buffer_enabled: bool,
    ) -> None:
        """
        Given finalized inputs and computed cumsums, copy tensors to device and
        call per-op .update(...) where needed (KV cache, rope, attention, etc.).
        """

    # --------- Optional subclass API (override if needed) ---------
    def config_network(self) -> None:
        """Update ops with comm IDs, ranks, shapes, etc. after init."""
        pass

    def init_category(self) -> None:
        """Optionally tag ops with CategoryType for stream allocation."""
        pass

    def config_algorithm_auto_search(self, params: dict[str, Any]) -> None:
        """Choose algorithms per-op from profile."""
        for op in self.model_operations:
            print(f"op.name: {op.name}, op.original_name: {op.original_name}")
            if op.original_name in self.profile_result:
                algo_tag = self.profile_result[op.original_name][op.name]["algo_tag"]
                op.config_tag(algo_tag, params)

    def nanobatch_split(self) -> None:
        """Split the model operations into nano operations."""
        op_nanobatch_info_map: dict[str, tuple[NanoOpInfo, ...]] = {}
        extra_links: dict[str, list[tuple[str, bool]]] = {}
        if self.is_auto_search_enabled:
            for op_basename, op_info in self.profile_result.items():
                split_info_list = []
                for nano_op_name, nano_op_info in op_info.items():
                    split_info_list.append(
                        NanoOpInfo(
                            batch_idx=nano_op_info["batch_idx"],
                            batch_size=nano_op_info["batch_size"],
                        )
                    )
                    extra_links[nano_op_name] = nano_op_info["extra_dep"]

                op_nanobatch_info_map[op_basename] = tuple(split_info_list)
        else:
            raise NotImplementedError("Nanobatch split is not implemented without auto search")
        model_ops, addtional_virtual_ops = split_nanobatch(
            self.original_model_operations, op_nanobatch_info_map, extra_links
        )
        self.model_operations = model_ops
        self.all_operations = []
        self.all_layer_operations = []
        for op in model_ops + self.virtual_operations + addtional_virtual_ops:
            print("op.name", op.name, op.batch_size)
            self.all_operations.append(op)
        for operation in model_ops:
            self.all_layer_operations.extend(operation.children)

    def post_update_for_next_cycle_ops(self, next_input_req_idx: list[int], next_cumsum_input: list[int], next_decode_batch_size: int) -> None:
        pass

    # --------- Base: construction / config ---------
    def __init__(
        self,
        pipeline_name: str,
        cache_weight_name: str,
        cached_weight_dir: str,
        profile_dir: str,
        num_layers: int,
        world_size: int = 1,
        world_rank: int = 0,
        pp_size: int = 1,
        pp_rank: int = 0,
        categories: list[CategoryType] = [CategoryType.COMP, CategoryType.MEM],
    ) -> None:
        self.pipeline_name = pipeline_name
        self.cache_weight_name = cache_weight_name
        self.cached_weight_dir = cached_weight_dir
        self.profile_dir = profile_dir
        self.num_layers = num_layers
        self.start_layer_idx = pp_rank * num_layers // pp_size
        self.end_layer_idx = (pp_rank + 1) * num_layers // pp_size
        self.layer_list = [i for i in range(self.start_layer_idx, self.end_layer_idx)]
        self.world_size = world_size
        self.world_rank = world_rank
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.device = f"cuda:{world_rank}"
        self.categories = categories
        self.kv_cache: Optional[KVCacheNone | KVCacheTorch | BatchedDistKVCache] = None

        # execution / profiling flags
        self.buffer_fixed: bool = False
        self.auto_search_enabled: bool = False
        self.plan_cuda_graph: bool = False
        self.cuda_graph_enabled: bool = False
        self.plan_double_buffer: bool = False
        self.double_buffer_enabled: bool = False
        self.profile_result: dict[str, Any] | None = None

        # batch & shape
        self.global_batch_size: Optional[int] = None
        self.decode_batch_size: Optional[int] = None

        # op containers (subclass should fill these in init_operations)
        self.original_model_operations: list[Operations] = []
        self.original_virtual_operations: list[Operations] = []
        self.model_operations: list[Operations] = []
        self.virtual_operations: list[Operations] = []
        self.all_operations: list[Operations] = []
        self.all_layer_operations: list[Operation_Layer] = []

        # built later
        self.executor: Optional[Executor] = None

    # --------- Base: init lifecycle ---------
    def init(self, weight_path: str, cached: bool = False) -> None:
        self.init_streams()
        self.init_external_data()
        self.init_operations()
        self.init_category()
        self.init_dependency()
        self.init_set_weight(weight_path, cached)
        self.config_network()
    
    def init_wo_weight(self) -> None:
        self.init_streams()
        self.init_external_data()
        self.init_operations()
        self.init_category()
        self.init_dependency()
        self.config_network()

    def init_set_weight(self, weight_path: str, cached: bool) -> None:
        weight_manager = WeightManager(
            self.cache_weight_name,
            self.cached_weight_dir,
            weight_path,
            cached,
            self.device,
        )
        weight_manager.set_weight(self.model_operations, self.device)

    def init_cached_weight(self, weight_path: str) -> None:
        """Convenience: subclasses can override if they need special behavior."""
        self.kv_cache = KVCacheNone()
        self.init_operations()
        self.init_set_weight(weight_path, cached=False)

    # --------- Base: streams ---------
    def init_streams(self) -> None:
        self.main_stream = torch.cuda.Stream()

        # Assuming SM counts are in increments of 8 up to 120 (reserve 132 as total)
        sm_counts_for_greenctx = [i for i in range(8, 128, 8)]
        self.total_sm = 132  # default for H200; override in subclass if needed
        
        self.sm_counts = sm_counts_for_greenctx + [self.total_sm]
        
        # Category -> { sm_count : (stream, sm_count) }
        self.streams: dict[CategoryType,
                           dict[int, tuple[torch._C.Stream, int]]] = {}

        for category in self.categories:
            self.streams[category] = {}

            # build paired green contexts
            n = len(sm_counts_for_greenctx)
            for i in range((n + 1) // 2):
                sm1 = sm_counts_for_greenctx[i]
                sm2 = sm_counts_for_greenctx[n - 1 - i]
                (s1, s2, _), _ = split_device_green_ctx_by_sm_count(
                    torch.device(self.device), [sm1, sm2]
                )
                self.streams[category][sm1] = (s1, sm1)
                self.streams[category][sm2] = (s2, sm2)

            # full-SM default
            self.streams[category][self.total_sm] = (
                torch.cuda.Stream(), self.total_sm)

        # Simple test pool for profiling sweeps
        self.profile_streams: dict[str, tuple[torch._C.Stream, int]] = {}
        n = len(sm_counts_for_greenctx)
        for i in range((n + 1) // 2):
            sm1 = sm_counts_for_greenctx[i]
            sm2 = sm_counts_for_greenctx[n - 1 - i]
            (s1, s2, _), _ = split_device_green_ctx_by_sm_count(
                torch.device(self.device), [sm1, sm2]
            )
            self.profile_streams[f"TEST_{sm1}"] = (s1, sm1)
            self.profile_streams[f"TEST_{sm2}"] = (s2, sm2)
        self.profile_streams[f"TEST_{self.total_sm}"] = (
            torch.cuda.Stream(), self.total_sm)

    def config_streams(self) -> None:
        # Default: pin all model ops to main stream unless profile results say otherwise
        print(f"Configuring streams on {self.device}...")
        for op in self.original_model_operations:
            op.set_stream((self.main_stream, self.total_sm))

        if self.auto_search_enabled:
            assert self.profile_result is not None, "Profile result not initialized"
            for op in self.model_operations:
                if op.original_name in self.profile_result:
                    sm_count = self.profile_result[op.original_name][op.name]["p_value"]
                    op.set_stream(self.streams[op.category][sm_count])

    def config_profile_streams(self, stream_tuple: tuple[torch._C.Stream, int]) -> None:
        for op in self.model_operations:
            op.set_stream(stream_tuple)

    # --------- Base: executor / buffers ---------
    def init_executor(self) -> None:
        print("Initializing executor...")
        self.executor = Executor(self.all_layer_operations, self.layer_list)
        self.executor.plan_layer_ordering()

    def update_allocate_buffers(self) -> None:
        # Collect all buffer wrappers referenced by ops
        print(f"Allocating buffers on {self.device}...")
        buffers_list = []
        for operation in self.all_operations:
            buffers_list.extend(operation.inputs.values())
            buffers_list.extend(operation.outputs.values())

        bufferAllocator = BufferAllocator(buffers_list)
        bufferAllocator.create_dependency_graph()
        bufferAllocator.set_all_batchsize_by_linear_programming()
        bufferAllocator.allocate_buffer(self.device)

    # --------- Base: public controls ---------
    def reset(self) -> None:
        """Subclasses may override to also reset KV cache etc."""
        assert self.kv_cache is not None, "KV cache not initialized"
        self.kv_cache.reset()
        self.global_batch_size = None
        self.decode_batch_size = None

    def clear_batch_size(self) -> None:
        for op in self.all_operations:
            op.setBatchSize(None)

    # --------- Base: update lifecycle ---------
    def _prepare_inputs(self, input_infos: list[tuple[int, list[int]]]) -> tuple[
        int, torch.Tensor, list[int], list[list[int]]
    ]:
        with prof_marker("prepare_inputs"):
            input_req_idx = []
            input_ids = []
            for req_idx, ids in input_infos:
                input_req_idx.append(req_idx)
                input_ids.append(ids)

            flattened = [tok for seq in input_ids for tok in seq]
            global_batch_size = len(flattened)
        with prof_marker("prepare_inputs_create_input_tensor"):
            input_tensor = torch.tensor(
                flattened, dtype=torch.int32, device="cpu")
        return global_batch_size, input_tensor, input_req_idx, input_ids

    def _compute_cumsums(self, input_ids: list[list[int]]) -> list[int]:
        request_length = torch.tensor(
            [len(x) for x in input_ids], dtype=torch.int32, device="cpu"
        )
        cumsum_input = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device="cpu"),
                torch.cumsum(request_length, dim=0, dtype=torch.int32),
            ]
        ).tolist()
        return cumsum_input

    def update_for_next_cycle(
        self,
    ) -> None:
        if self.next_input_infos is None:
            return
        # -------- inputs --------
        with prof_marker("update_for_next_cycle_prepare_inputs"):
            _, _, next_input_req_idx, next_input_ids = self._prepare_inputs(
                self.next_input_infos)

        with prof_marker("update_for_next_cycle_compute_cumsum"):
            next_cumsum_input = self._compute_cumsums(next_input_ids)
        # -------- always update per-op state --------
        with prof_marker("update_for_next_cycle_post_ops"):
            self.post_update_for_next_cycle_ops(
                next_input_req_idx, next_cumsum_input, self.next_decode_batch_size)

    def update(
        self,
        input_infos: list[tuple[int, list[int]]],
        decode_batch_size: int = 0,
        next_input_infos: list[tuple[int, list[int]]] = None,
        next_decode_batch_size: int = 0,
        is_profile: bool = False,
        stream_name: str = "TEST_TOTAL",
        profile_result_path: Optional[str] = None,
        auto_search_enabled: bool = False,
        nano_split_enabled: bool = False,  # subclasses can ignore/override
        plan_cuda_graph: bool = False,
        cuda_graph_enabled: bool = False,
        plan_double_buffer: bool = False,
        double_buffer_enabled: bool = False,
    ) -> None:
        # -------- inputs --------
        with prof_marker("update_prepare_inputs"):
            global_batch_size, input_tensor, input_req_idx, input_ids = self._prepare_inputs(
                input_infos)
            self.input_req_idx = input_req_idx
            self.next_input_infos = next_input_infos
            self.next_decode_batch_size = next_decode_batch_size
            if self.next_input_infos is None:
                assert not (double_buffer_enabled or plan_double_buffer), "Double buffer related flags are not allowed when next_input_infos is None"
        # -------- flags --------
        self.buffer_fixed = (
            global_batch_size == self.global_batch_size
            and decode_batch_size == self.decode_batch_size
        )

        self.plan_cuda_graph = plan_cuda_graph
        self.cuda_graph_enabled = cuda_graph_enabled
        assert not (self.plan_cuda_graph and self.cuda_graph_enabled), "CUDA graph is not enabled when plan_cuda_graph is True"

        if self.cuda_graph_enabled:
            assert self.buffer_fixed, "When using CUDA graph, batch sizes must remain unchanged."

        self.auto_search_enabled = auto_search_enabled
        if self.auto_search_enabled:
            assert profile_result_path is not None, "profile_result_path must be provided when auto_search_enabled=True"
            with open(profile_result_path, "r") as f:
                self.profile_result = json.load(f)

        self.plan_double_buffer = plan_double_buffer
        self.double_buffer_enabled = double_buffer_enabled
        assert not (self.plan_double_buffer and self.double_buffer_enabled), "Double buffer is not enabled when plan_double_buffer is True"

        # -------- (re)configure if batch sizes changed --------
        if not self.buffer_fixed:
            with prof_marker("update_reconfig"):
                self.global_batch_size = global_batch_size
                self.decode_batch_size = decode_batch_size

                self.clear_batch_size()
                self.apply_batch_size()

                # optional: subclasses may split/nanosplit here
                if nano_split_enabled:
                    self.nanobatch_split()

                self.update_allocate_buffers()

                if is_profile:
                    self.config_profile_streams(
                        self.profile_streams[stream_name])
                else:
                    self.config_streams()

                self.config_algorithm()
                self.init_executor()
                print("Executor initialized")

            with prof_marker("update_compute_cumsum"):
                self.cumsum_input = self._compute_cumsums(input_ids)

        # -------- always update per-op state --------
        with prof_marker("update_post_ops"):
            self.post_update_ops(
                self.input_req_idx, input_tensor.to(self.device), self.cumsum_input, decode_batch_size)

    # --------- Base: run + simple profile helpers ---------
    def run(self) -> list[tuple[int, list[int]]]:
        assert self.executor is not None, "Executor not initialized. Call update() first."

        temp_out = torch.zeros(self.global_batch_size,
                               dtype=torch.int32, device=self.device)
        exec_handle = self.executor.execute(
            temp_out,
            self.main_stream,
            plan_cuda_graph=self.plan_cuda_graph,
            cuda_graph_enabled=self.cuda_graph_enabled,
        )
        if self.plan_double_buffer or self.double_buffer_enabled:
            self.update_for_next_cycle()

        torch.cuda.current_stream().wait_stream(exec_handle.stream)
        temp_out.copy_(exec_handle.src)

        if self.pp_rank == self.pp_size - 1:
            with prof_marker("after_execute_move_cpu"):
                temp_out = temp_out.cpu()

            # unpack new tokens by original request indices
            new_tokens = [[temp_out[idx - 1].item()]
                        for idx in self.cumsum_input[1:]]
            output: list[tuple[int, list[int]]] = []
            for req_idx, token in zip(self.input_req_idx, new_tokens):
                output.append((req_idx, token))
            return output
        else:
            return []

    def init_profile_data(self, append_mode: bool = False) -> None:
        for operation in self.model_operations:
            operation.setup_profile(
                self.profile_dir, append_mode=append_mode, is_save_db=self.world_rank == 0)

    def profile_run(self) -> None:
        for operation in self.model_operations:
            with prof_marker(f"{operation.name}"):
                print(f"Operation name: {operation.name}")
                operation.profile_all()
        torch.cuda.synchronize()

    def profile_print(self) -> None:
        for operation in self.model_operations:
            operation.print_profile()
