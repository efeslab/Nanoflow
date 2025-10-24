def worker_entry(T0, rank, affinity_module_path, *rest):
    # --- Set CPU affinity (import here so parent stays light) ---
    try:
        _aff_mod = __import__(affinity_module_path, fromlist=[
                              "set_affinity_for_rank", "tune_threads_like", "AFFINITY"])
        _set_affinity_for_rank = getattr(_aff_mod, "set_affinity_for_rank")
        _tune_threads_like = getattr(_aff_mod, "tune_threads_like")
        _AFFINITY = getattr(_aff_mod, "AFFINITY")
        _cores = _AFFINITY.get(rank, [])
        if _cores:
            _set_affinity_for_rank(rank, _cores)
            _tune_threads_like(_cores)
        print(f"[rank {rank}] CPU affinity set to cores: {_cores}", flush=True)
    except Exception as _e:
        # Affinity is best-effort; don't crash the worker if unavailable
        print(
            f"[rank {rank}] CPU affinity setup skipped or failed: {_e}", flush=True)

    from nanoflow.core.worker import worker as real_worker

    return real_worker(T0, rank, *rest)