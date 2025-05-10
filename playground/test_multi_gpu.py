import os
import torch
import torch.multiprocessing as mp
from multiprocessing import SimpleQueue  # lighter weight than Queue in many cases
import time

def example_task(x):
    """
    A sample task function that creates a tensor, performs an operation,
    and returns the result. Each worker computes this on its own GPU.
    """
    # Pre-allocating or reusing tensors could be considered if x is always the same shape.
    # tensor = torch.tensor([x], device=torch.device("cuda"))
    result = x + 10
    return result

def worker(rank, world_size, task_queue, result_queue):
    # Set the device for the current process
    torch.cuda.set_device(rank)
    
    # (Optional) Set a start method if needed:
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29501'
    
    # Avoid printing inside the loop to reduce overhead.
    # print(f"Worker {rank} started.")
    
    while True:
        # Retrieve the task; this blocks until an item is available.
        # Instead of sending a function pointer, we just send the argument.
        task_data = task_queue.get()
        
        # Use None as a sentinel value to signal shutdown.
        if task_data is None:
            break
        
        # Execute the task (calling example_task directly)
        result = example_task(task_data)
        
        # Place the result into the result queue.
        # Here we don’t include the rank if not needed, but you could include it.
        result_queue.put(result)
    
    # print(f"Worker {rank} exiting.")

if __name__ == '__main__':
    # On Linux or macOS you might benefit from using 'fork'
    mp.set_start_method('fork', force=True)
    
    # Determine the number of available GPUs.
    world_size = torch.cuda.device_count()
    if world_size < 2:
        raise RuntimeError("This example requires at least 2 GPUs.")
    
    # Use SimpleQueue for lower overhead.
    task_queue = SimpleQueue()
    result_queue = SimpleQueue()
    
    # Spawn one worker per GPU; these processes will be long-running.
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, task_queue, result_queue))
        p.start()
        processes.append(p)
    
    iterations = 1000
    start_time = time.time()
    
    # Benchmark loop: For each iteration, send one task per worker and wait for all results.
    for i in range(iterations):
        # Submit a task for each worker.
        for _ in range(world_size):
            # Here the task is simply the iteration index.
            task_queue.put(i)
        
        # Wait for one result per worker.
        for _ in range(world_size):
            _ = result_queue.get()
    
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time for {} iterations: {:.4f} seconds".format(iterations, total_time))
    
    # Signal each worker to shutdown.
    for _ in range(world_size):
        task_queue.put(None)
    
    # Wait for all worker processes to finish.
    for p in processes:
        p.join()
