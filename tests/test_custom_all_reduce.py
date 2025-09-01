import os
import torch
import torch.distributed as dist

# Import the custom all-reduce module (after building)
try:
    import bind_all_reduce
    CUSTOM_ALL_REDUCE_AVAILABLE = True
except ImportError:
    print("Custom all-reduce module not available. Please build it first.")
    CUSTOM_ALL_REDUCE_AVAILABLE = False

def dist_setup() -> tuple[int, int]:
    """Setup distributed environment"""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    return rank, world_size

def test_custom_all_reduce():
    """Test the custom all-reduce implementation"""
    if not CUSTOM_ALL_REDUCE_AVAILABLE:
        return
    
    rank, world_size = dist_setup()
    
    # Create test tensor
    dim = 1024
    x = torch.randn(dim, dim, device=f"cuda:{rank}")
    original_x = x.clone()
    
    print(f"Rank {rank}: Original tensor sum: {x.sum().item():.4f}")
    
    # Test custom all-reduce
    custom_ar = bind_all_reduce.CustomAllReduce()
    custom_ar.init(rank, world_size)
    
    # Test sum all-reduce
    result = custom_ar.all_reduce(x, "sum")
    print(f"Rank {rank}: Custom all-reduce sum result: {result.sum().item():.4f}")
    
    # Test in-place all-reduce
    x_inplace = original_x.clone()
    custom_ar.all_reduce_inplace(x_inplace, "sum")
    print(f"Rank {rank}: Custom in-place all-reduce result: {x_inplace.sum().item():.4f}")
    
    # Compare with PyTorch native all-reduce
    x_torch = original_x.clone()
    dist.all_reduce(x_torch, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}: PyTorch native all-reduce result: {x_torch.sum().item():.4f}")
    
    # Verify results are the same
    torch.testing.assert_close(result, x_torch, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x_inplace, x_torch, rtol=1e-5, atol=1e-5)
    print(f"Rank {rank}: All results match! ✓")
    
    # Test convenience functions
    x_conv = original_x.clone()
    result_conv = bind_all_reduce.all_reduce_sum(x_conv, rank, world_size)
    print(f"Rank {rank}: Convenience function result: {result_conv.sum().item():.4f}")
    torch.testing.assert_close(result_conv, x_torch, rtol=1e-5, atol=1e-5)
    
    dist.destroy_process_group()

def test_different_operations():
    """Test different reduction operations"""
    if not CUSTOM_ALL_REDUCE_AVAILABLE:
        return
    
    rank, world_size = dist_setup()
    
    # Create test tensor with different values per rank
    x = torch.ones(100, device=f"cuda:{rank}") * (rank + 1)
    
    custom_ar = bind_all_reduce.CustomAllReduce()
    custom_ar.init(rank, world_size)
    
    # Test different operations
    operations = ["sum", "max", "min", "prod"]
    
    for op in operations:
        result = custom_ar.all_reduce(x, op)
        print(f"Rank {rank}: {op.upper()} result (first 5 elements): {result[:5]}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    if CUSTOM_ALL_REDUCE_AVAILABLE:
        print("Testing custom all-reduce implementation...")
        test_custom_all_reduce()
        print("\nTesting different operations...")
        test_different_operations()
    else:
        print("Skipping tests - custom all-reduce module not available") 