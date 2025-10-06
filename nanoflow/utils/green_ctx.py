import nvmath
import torch

from flashinfer.green_ctx import split_device_green_ctx_by_sm_count

__all__ = [
    "split_device_green_ctx_by_sm_count",
    "set_sm_count_target",
]


def set_sm_count_target(sm_count: int):
    handle = torch.cuda.current_blas_handle()
    nvmath.bindings.cublas.set_sm_count_target(handle, sm_count)  # type: ignore
