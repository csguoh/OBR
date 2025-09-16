"""
Complete CUTLASS Sparse/Dense GEMM Python Interface with Memory Evaluation
Supports three comparisons: INT4 sparse, INT4 dense, FP16 dense
"""
import sys
import os
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load

# Global variables
cutlass_ops = None

def calculate_memory_usage(batch_size, seq_len, cin, cout, precision, is_sparse=False):
    """Calculate theoretical GPU memory usage"""
    # Input matrix A: batch_size * seq_len * cin
    input_elements = batch_size * seq_len * cin
    
    # Weight matrix B: cout * cin
    weight_elements = cout * cin
    
    # Output matrix C: batch_size * seq_len * cout
    output_elements = batch_size * seq_len * cout
    
    # Bytes per element based on precision
    if precision == 'FP16':
        bytes_per_element = 2
    elif precision == 'INT4':
        bytes_per_element = 0.5
    else:
        bytes_per_element = 4  # default FP32
    
    # Calculate memory for each component
    input_mem_bytes = input_elements * bytes_per_element
    weight_mem_bytes = weight_elements * bytes_per_element
    output_mem_bytes = output_elements * bytes_per_element
    
    # For sparse matrices, weight memory is reduced but we need metadata
    if is_sparse:
        weight_mem_bytes = weight_mem_bytes * 0.5  # 2:4 sparsity
        # Add metadata overhead (approximately 25% of compressed weight)
        metadata_bytes = weight_mem_bytes * 0.25
        weight_mem_bytes += metadata_bytes
    
    total_memory = input_mem_bytes + weight_mem_bytes + output_mem_bytes
    
    return {
        'input_memory_mb': input_mem_bytes / (1024 * 1024),
        'weight_memory_mb': weight_mem_bytes / (1024 * 1024),
        'output_memory_mb': output_mem_bytes / (1024 * 1024),
        'total_memory_mb': total_memory / (1024 * 1024)
    }

def load_cutlass_ops():
    """Load CUTLASS operations"""
    global cutlass_ops
    
    if cutlass_ops is not None:
        return True
    
    try:
        cutlass_dir = os.environ.get('CUTLASS_DIR', '/path/to/cutlass')
        
        cutlass_ops = load(
            name="cutlass_sparse_ops",
            sources=["/path/to/compare_with_three_baseline.cu"],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=[
                "-O3", "-std=c++17", "-arch=sm_80",
                f"-I{cutlass_dir}/include",
                f"-I{cutlass_dir}/tools/util/include"
            ],
            extra_ldflags=["-lcublas"],
            verbose=True
        )
        
        print("CUTLASS GEMM extension loaded successfully")
        return True
        
    except Exception as e:
        print(f"Loading failed: {e}")
        return False

def check_gpu():
    """Check GPU compatibility"""
    if not load_cutlass_ops():
        return False
    
    return cutlass_ops.check_gpu_compatibility()

# INT4 sparse GEMM functions
def create_sparse_gemm_batch(batch_size, seq_len, cin, cout):
    """Create batch-supported sparse GEMM instance"""
    if not load_cutlass_ops():
        raise RuntimeError("CUTLASS operations not loaded")
    
    return cutlass_ops.create_sparse_gemm_batch(batch_size, seq_len, cin, cout)

def create_sparse_gemm(m, n, k):
    """Create sparse GEMM instance (compatible with old interface)"""
    if not load_cutlass_ops():
        raise RuntimeError("CUTLASS operations not loaded")
    
    return cutlass_ops.create_sparse_gemm(m, n, k)

# INT4 dense GEMM functions
def create_dense_gemm_int4_batch(batch_size, seq_len, cin, cout):
    """Create batch-supported INT4 dense GEMM instance"""
    if not load_cutlass_ops():
        raise RuntimeError("CUTLASS operations not loaded")
    
    return cutlass_ops.create_dense_gemm_int4_batch(batch_size, seq_len, cin, cout)

def create_dense_gemm_int4(m, n, k):
    """Create INT4 dense GEMM instance (compatible with old interface)"""
    if not load_cutlass_ops():
        raise RuntimeError("CUTLASS operations not loaded")
    
    return cutlass_ops.create_dense_gemm_int4(m, n, k)

def benchmark_sparse_gemm_batch(batch_size, seq_len, cin, cout, warmup=5, iterations=10, verbose=True):
    """Benchmark batch sparse GEMM"""
    if verbose:
        print(f"CUTLASS INT4 2:4 Sparse GEMM Benchmark")
        print(f"Batch size: {batch_size} x {seq_len} x {cin} -> {batch_size} x {seq_len} x {cout}")
    
    if not check_gpu():
        raise RuntimeError("GPU incompatible")
    
    sparse_gemm = create_sparse_gemm_batch(batch_size, seq_len, cin, cout)
    
    if verbose:
        problem_size = sparse_gemm.get_problem_size()
        gemm_size = sparse_gemm.get_gemm_size()
        sparse_info = sparse_gemm.get_sparse_info()
        print(f"Problem size: batch={problem_size[0]}, seq_len={problem_size[1]}, cin={problem_size[2]}, cout={problem_size[3]}")
        print(f"GEMM size: M={gemm_size[0]}, N={gemm_size[1]}, K={gemm_size[2]}")
        print(f"Sparse config: kSparse={sparse_info[0]}, kElementsPerElementE={sparse_info[1]}")
    
    sparse_gemm.fill_random_data()
    sparse_gemm.sync_to_device()
    
    times = sparse_gemm.benchmark(warmup, iterations)
    
    if not times:
        raise RuntimeError("Benchmark failed")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Calculate TOPS (considering 2:4 sparsity)
    ops = 2.0 * batch_size * seq_len * cin * cout * 0.5  # 50% sparsity
    tops = ops / (avg_time * 1e12)
    gflops = ops / 1e9
    
    # Calculate memory usage
    memory_info = calculate_memory_usage(batch_size, seq_len, cin, cout, 'INT4', is_sparse=True)
    
    result = {
        'type': 'INT4_Sparse',
        'batch_size': batch_size,
        'seq_len': seq_len,
        'cin': cin,
        'cout': cout,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'tops': tops,
        'gflops': gflops,
        'operations': ops,
        'times': times,
        'memory_info': memory_info
    }
    
    if verbose:
        print(f"\n Performance Results:")
        print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Min time: {min_time:.2f} ms")
        print(f"  Max time: {max_time:.2f} ms")
        print(f"  Computing performance: {tops:.2f} TOPS")
        print(f"  Memory usage: {memory_info['total_memory_mb']:.2f} MB")
    
    return result

def benchmark_dense_gemm_int4_batch(batch_size, seq_len, cin, cout, warmup=5, iterations=10, verbose=True):
    """Benchmark batch INT4 dense GEMM"""
    if verbose:
        print(f" CUTLASS INT4 Dense GEMM Benchmark")
        print(f"Batch size: {batch_size} x {seq_len} x {cin} -> {batch_size} x {seq_len} x {cout}")
    
    if not check_gpu():
        raise RuntimeError("GPU incompatible")
    
    dense_gemm = create_dense_gemm_int4_batch(batch_size, seq_len, cin, cout)
    
    if verbose:
        problem_size = dense_gemm.get_problem_size()
        gemm_size = dense_gemm.get_gemm_size()
        print(f"Problem size: batch={problem_size[0]}, seq_len={problem_size[1]}, cin={problem_size[2]}, cout={problem_size[3]}")
        print(f"GEMM size: M={gemm_size[0]}, N={gemm_size[1]}, K={gemm_size[2]}")
    
    dense_gemm.fill_random_data()
    dense_gemm.sync_to_device()
    
    times = dense_gemm.benchmark(warmup, iterations)
    
    if not times:
        raise RuntimeError("Benchmark failed")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Calculate TOPS (full operations)
    ops = 2.0 * batch_size * seq_len * cin * cout
    tops = ops / (avg_time * 1e12)
    gflops = ops / 1e9
    
    # Calculate memory usage
    memory_info = calculate_memory_usage(batch_size, seq_len, cin, cout, 'INT4', is_sparse=False)
    
    result = {
        'type': 'INT4_Dense',
        'batch_size': batch_size,
        'seq_len': seq_len,
        'cin': cin,
        'cout': cout,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'tops': tops,
        'gflops': gflops,
        'operations': ops,
        'times': times,
        'memory_info': memory_info
    }
    
    if verbose:
        print(f"\nPerformance Results:")
        print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Min time: {min_time:.2f} ms")
        print(f"  Max time: {max_time:.2f} ms")
        print(f"  Computing performance: {tops:.2f} TOPS")
        print(f"  Memory usage: {memory_info['total_memory_mb']:.2f} MB")
    
    return result

def benchmark_dense_gemm_fp16_batch(batch_size, seq_len, cin, cout, iterations=10, verbose=True):
    """Benchmark batch FP16 dense GEMM (using PyTorch)"""
    if verbose:
        print(f"PyTorch FP16 Dense GEMM Benchmark")
        print(f"Batch size: {batch_size} x {seq_len} x {cin} -> {batch_size} x {seq_len} x {cout}")
    
    # Create random matrices
    X = torch.randn(batch_size, seq_len, cin, dtype=torch.float16, device='cuda')
    W = torch.randn(cout, cin, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(5):
        _ = torch.matmul(X, W.t())
    torch.cuda.synchronize()
    
    # Timing
    times = []
    for _ in range(iterations):
        start_time = time.time()
        Y = torch.matmul(X, W.t())
        torch.cuda.synchronize()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    # Calculate TOPS
    ops = 2.0 * batch_size * seq_len * cin * cout
    tops = ops / (avg_time * 1e12)
    gflops = ops / 1e9
    
    # Calculate memory usage
    memory_info = calculate_memory_usage(batch_size, seq_len, cin, cout, 'FP16', is_sparse=False)
    
    result = {
        'type': 'FP16_Dense',
        'batch_size': batch_size,
        'seq_len': seq_len,
        'cin': cin,
        'cout': cout,
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'tops': tops,
        'gflops': gflops,
        'operations': ops,
        'times': times,
        'memory_info': memory_info
    }
    
    if verbose:
        print(f"\n Performance Results:")
        print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Min time: {min_time:.2f} ms")
        print(f"  Max time: {max_time:.2f} ms")
        print(f"  Computing performance: {tops:.2f} TOPS")
        print(f"  Memory usage: {memory_info['total_memory_mb']:.2f} MB")
    
    return result

def compare_all_gemm_batch(batch_size, seq_len, cin, cout, warmup=5, iterations=10):
    """Compare three GEMM implementations: INT4 sparse, INT4 dense, FP16 dense"""
    print(f"\n Three GEMM Performance Comparison")
    print(f"Batch size: {batch_size} x {seq_len} x {cin} -> {batch_size} x {seq_len} x {cout}")
    print("=" * 80)
    
    # Test INT4 sparse
    sparse_result = benchmark_sparse_gemm_batch(batch_size, seq_len, cin, cout, warmup, iterations, verbose=False)
    
    # Test INT4 dense
    int4_dense_result = benchmark_dense_gemm_int4_batch(batch_size, seq_len, cin, cout, warmup, iterations, verbose=False)
    
    # Test FP16 dense
    fp16_dense_result = benchmark_dense_gemm_fp16_batch(batch_size, seq_len, cin, cout, iterations, verbose=False)
    
    # Calculate comparison metrics
    sparse_vs_int4dense = int4_dense_result['avg_time_ms'] / sparse_result['avg_time_ms']
    sparse_vs_fp16dense = fp16_dense_result['avg_time_ms'] / sparse_result['avg_time_ms']
    int4dense_vs_fp16dense = fp16_dense_result['avg_time_ms'] / int4_dense_result['avg_time_ms']
    
    # Memory comparison
    sparse_mem = sparse_result['memory_info']['total_memory_mb']
    int4_dense_mem = int4_dense_result['memory_info']['total_memory_mb']
    fp16_dense_mem = fp16_dense_result['memory_info']['total_memory_mb']
    
    # Output comparison results
    print(f"\n Performance Comparison Results:")
    print("-" * 80)
    print(f"{'Metric':<20} {'INT4 Sparse':<15} {'INT4 Dense':<15} {'FP16 Dense':<15}")
    print("-" * 80)
    print(f"{'Time (ms)':<20} {sparse_result['avg_time_ms']:<15.2f} {int4_dense_result['avg_time_ms']:<15.2f} {fp16_dense_result['avg_time_ms']:<15.2f}")
    print(f"{'TOPS':<20} {sparse_result['tops']:<15.2f} {int4_dense_result['tops']:<15.2f} {fp16_dense_result['tops']:<15.2f}")
    print(f"{'GFLOPS':<20} {sparse_result['gflops']:<15.2f} {int4_dense_result['gflops']:<15.2f} {fp16_dense_result['gflops']:<15.2f}")
    print(f"{'Memory (MB)':<20} {sparse_mem:<15.2f} {int4_dense_mem:<15.2f} {fp16_dense_mem:<15.2f}")
    print("-" * 80)
    print(f"\nSpeedup Ratios:")
    print(f"  INT4 Sparse vs INT4 Dense: {sparse_vs_int4dense:.2f}x")
    print(f"  INT4 Sparse vs FP16 Dense: {sparse_vs_fp16dense:.2f}x") 
    print(f"  INT4 Dense vs FP16 Dense: {int4dense_vs_fp16dense:.2f}x")
    print(f"\nMemory Usage (relative to FP16 Dense):")
    print(f"  INT4 Sparse: {sparse_mem/fp16_dense_mem*100:.1f}%")
    print(f"  INT4 Dense: {int4_dense_mem/fp16_dense_mem*100:.1f}%")
    print(f"  FP16 Dense: 100.0%")
    print(f"\nMemory Breakdown (MB):")
    print(f"  INT4 Sparse - Input: {sparse_result['memory_info']['input_memory_mb']:.2f}, Weight: {sparse_result['memory_info']['weight_memory_mb']:.2f}, Output: {sparse_result['memory_info']['output_memory_mb']:.2f}")
    print(f"  INT4 Dense  - Input: {int4_dense_result['memory_info']['input_memory_mb']:.2f}, Weight: {int4_dense_result['memory_info']['weight_memory_mb']:.2f}, Output: {int4_dense_result['memory_info']['output_memory_mb']:.2f}")
    print(f"  FP16 Dense  - Input: {fp16_dense_result['memory_info']['input_memory_mb']:.2f}, Weight: {fp16_dense_result['memory_info']['weight_memory_mb']:.2f}, Output: {fp16_dense_result['memory_info']['output_memory_mb']:.2f}")
    
    return {
        'sparse_result': sparse_result,
        'int4_dense_result': int4_dense_result,
        'fp16_dense_result': fp16_dense_result,
        'sparse_vs_int4dense': sparse_vs_int4dense,
        'sparse_vs_fp16dense': sparse_vs_fp16dense,
        'int4dense_vs_fp16dense': int4dense_vs_fp16dense
    }

def test_functionality():
    """Simple functionality test"""
    print(" Functionality Test")
    print("=" * 30)
    
    if not check_gpu():
        print(" GPU incompatible")
        return False
    
    try:
        is_verbose = False
        # Test INT4 sparse
        print("Testing INT4 sparse GEMM...")
        result1 = benchmark_sparse_gemm_batch(2, 128, 256, 128, warmup=2, iterations=3, verbose=is_verbose)
        print("INT4 sparse GEMM test passed")
        
        # Test INT4 dense
        print("Testing INT4 dense GEMM...")
        result2 = benchmark_dense_gemm_int4_batch(2, 128, 256, 128, warmup=2, iterations=3, verbose=is_verbose)
        print("INT4 dense GEMM test passed")
        
        # Test FP16 dense
        print("Testing FP16 dense GEMM...")
        result3 = benchmark_dense_gemm_fp16_batch(2, 128, 256, 128, iterations=3, verbose=is_verbose)
        print("FP16 dense GEMM test passed")
        
        print("All functionality tests passed")
        return True
    except Exception as e:
        print(f"Functionality test failed: {e}")
        return False

if __name__ == "__main__":
    # Functionality test
    if test_functionality():
        # Run performance comparison tests
        test_configs = [
            (32, 2048, 4096, 4096),
            (32, 128, 4096, 4096),
            (32, 256, 4096, 4096),
            (32, 512, 4096, 4096),
            (32, 1024, 4096, 4096),
            (32, 2048, 4096, 4096),
            (32, 4096, 4096, 4096),


        ] # [batch_size, seq_len, cin, cout]
        
        for batch_size, seq_len, cin, cout in test_configs:
            try:
                compare_all_gemm_batch(batch_size, seq_len, cin, cout, warmup=3, iterations=5)
            except Exception as e:
                print(f"Test failed: {e}")
        
        print("\nAll tests completed!")
    else:
        print("Basic functionality test failed")