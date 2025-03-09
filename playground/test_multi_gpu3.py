import time
import torch.multiprocessing as mp
from multiprocessing import Value, Array, Barrier

def worker(rank, world_size, shared_int, shared_array, barrier):
    """
    Worker process: wait for a new task value from the main process,
    then compute (rank + shared_int) and store the result in shared_array[rank].
    """
    while True:
        # First barrier: wait until the main process writes a new task.
        barrier.wait()
        
        # Read the shared integer.
        with shared_int.get_lock():
            task_val = shared_int.value

        # Check for termination signal.
        if task_val == -1:
            # Call barrier a second time to keep the barrier count consistent.
            barrier.wait()
            break

        # Compute the result and write to the worker’s assigned index.
        shared_array[rank] = rank + task_val
        
        # Second barrier: wait until all workers finish computation.
        barrier.wait()
    
    # Worker exits gracefully.
    
if __name__ == '__main__':
    # For Unix-like systems, using the 'fork' start method can be faster.
    mp.set_start_method('fork', force=True)
    
    # Use the number of available GPUs (or set a fixed number). Here we use torch.cuda.device_count()
    # if you have GPUs; otherwise, you could simply set world_size = 4.
    world_size = 4
    
    iterations = 1000

    # Create a shared integer (for the task value) and a shared array to hold each worker's result.
    shared_int = Value('i', 0)    # 'i' stands for a signed integer.
    shared_array = Array('i', world_size)  # An array of integers with length equal to world_size.
    
    # Create a Barrier for world_size workers plus the main process.
    barrier = Barrier(world_size + 1)
    
    # Spawn one worker per GPU (or per unit of parallelism).
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker, args=(rank, world_size, shared_int, shared_array, barrier))
        p.start()
        processes.append(p)
    
    start_time = time.time()
    
    # For each iteration, update the shared integer, synchronize with the workers,
    # and let them compute and write their results.
    for i in range(iterations):
        # Set the shared task value.
        with shared_int.get_lock():
            shared_int.value = i
        
        # First barrier: signal to all workers that a new task is available.
        barrier.wait()
        
        # Second barrier: wait for all workers to finish processing.
        barrier.wait()
        
        # (Optional) Main process could inspect shared_array here if desired.
        # For example, expected value at index j is: j + i.
        # results = list(shared_array)
    
    # Signal termination: write -1 into the shared integer.
    with shared_int.get_lock():
        shared_int.value = -1
    
    # Execute the final two barrier waits so that all workers exit cleanly.
    barrier.wait()  # First barrier of termination iteration.
    barrier.wait()  # Second barrier of termination iteration.
    
    total_time = time.time() - start_time
    print("Total time for {} iterations: {:.4f} seconds".format(iterations, total_time))
    
    # Wait for all worker processes to finish.
    for p in processes:
        p.join()
