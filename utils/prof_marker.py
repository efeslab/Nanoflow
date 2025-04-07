# prof_mark.py
import platform_config as config 
import contextlib

if config.PLATFORM_CUDA:
    import nvtx

    @contextlib.contextmanager
    def prof_marker(name: str):
        with nvtx.annotate(name):
            yield

elif config.PLATFORM_ROCM:
    import torch.profiler

    @contextlib.contextmanager
    def prof_marker(name: str):
        with torch.profiler.record_function(name):
            yield

else:
    # Fallback: do nothing
    @contextlib.contextmanager
    def prof_marker(name: str):
        yield
