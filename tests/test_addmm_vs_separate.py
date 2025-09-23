import torch
import time

def test_addmm_vs_separate(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Compare performance between torch.addmm and separate matmul+add operations
    """
    assert(device == 'cuda'), "This test is designed to run on CUDA device for performance comparison."
    print(f"Running test on device: {device}")
    
    # Matrix dimensions (thousands level as requested)
    M, N, K = 2048, 2048, 2048
    
    # Create test matrices
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)
    C = torch.randn(M, N, device=device, dtype=torch.float32)
    
    # Warmup to avoid cold start effects
    print("Warming up...")
    for _ in range(10):
        _ = torch.addmm(C, A, B)
        _ = torch.matmul(A, B) + C
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    print(f"Matrix shapes: A={A.shape}, B={B.shape}, C={C.shape}")
    print("Starting performance test...")
    
    # Test torch.addmm
    print("\n=== Testing torch.addmm ===")
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for i in range(100):
        result1 = torch.addmm(C, A, B)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    addmm_time = time.perf_counter() - start_time
    
    print(f"torch.addmm: {addmm_time:.4f} seconds for 100 iterations")
    print(f"Average per iteration: {addmm_time/100*1000:.2f} ms")
    
    # Test separate matmul + add
    print("\n=== Testing torch.matmul + torch.add ===")
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    for i in range(100):
        matmul_result = torch.matmul(A, B)
        result2 = torch.add(C, matmul_result)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    separate_time = time.perf_counter() - start_time
    
    print(f"matmul + add: {separate_time:.4f} seconds for 100 iterations")
    print(f"Average per iteration: {separate_time/100*1000:.2f} ms")
    
    # Results comparison
    print(f"\n=== Performance Summary ===")
    speedup = separate_time / addmm_time
    print(f"torch.addmm is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    
    # Verify results are equivalent (within numerical precision)
    max_diff = torch.max(torch.abs(result1 - result2)).item()
    print(f"Max difference between results: {max_diff:.2e}")
    print("Results are equivalent!" if max_diff < 1e-5 else "Warning: Results differ!")

if __name__ == "__main__":
    # Run the test
    test_addmm_vs_separate()
    
    print("\n" + "="*50)
    print("To profile with Nsight Systems, run:")
    print("nsys profile --trace=cuda,nvtx python torch_perf_test.py")
    print("="*50)