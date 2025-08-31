# prof_mark.py
from typing import Optional
import platform_config as config 
import contextlib

if config.PLATFORM_PROFILE_IN_TORCH:
    import torch

    @contextlib.contextmanager
    def prof_marker(name: str, color: Optional[str] = None):
        with torch.autograd.profiler.record_function(name):
            yield

elif config.PLATFORM_CUDA:
    import nvtx

    @contextlib.contextmanager
    def prof_marker(name: str, color: Optional[str] = None):
        with nvtx.annotate(name, color=color):
            yield

elif config.PLATFORM_ROCM:
    import pybind_amd.bind_marker.build.bind_marker as bind_marker

    @contextlib.contextmanager
    def prof_marker(name: str, color: Optional[str] = None):
        # Push the ROCm profiling range with the given name
        bind_marker.roctxRangePush(name)
        try:
            yield
        finally:
            # Ensure that the marker is popped when the context is exited
            bind_marker.roctxRangePop()

else:
    # Fallback: do nothing
    @contextlib.contextmanager
    def prof_marker(name: str, color: Optional[str] = None):
        yield
