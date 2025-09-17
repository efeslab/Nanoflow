import os, psutil

AFFINITY = {
    0: list(range(0, 16)) + list(range(128, 144)),
    1: list(range(16, 32)) + list(range(144, 160)),
    2: list(range(32, 48)) + list(range(160, 176)),
    3: list(range(48, 64)) + list(range(176, 192)),
    4: list(range(64, 80)) + list(range(192, 208)),
    5: list(range(80, 96)) + list(range(208, 224)),
    6: list(range(96, 112)) + list(range(224, 240)),
    7: list(range(112, 128)) + list(range(240, 256)),
}


def set_affinity_for_rank(rank, core_list):
    try:
        p = psutil.Process(os.getpid())
        p.cpu_affinity(core_list)
    except Exception:
        raise RuntimeError("Setting CPU Affinity failed.")

def tune_threads_like(cores):
     """ Optional: match OMP/BLAS/PyTorch thread counts to your core set. """ 
     n = max(1, len(cores)) 
     os.environ.setdefault("OMP_NUM_THREADS", str(n)) 
     os.environ.setdefault("MKL_NUM_THREADS", str(n)) 
     os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0") 
     try: 
        import torch 
        torch.set_num_threads(n) 
        torch.set_num_interop_threads(max(1, min(2, n // 2))) 
     except Exception:
         pass