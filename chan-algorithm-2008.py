import numpy as np
import time
import math
from typing import List, Tuple
import matplotlib.pyplot as plt

def generate_test_graph(n: int) -> np.ndarray:
    """Generate a test graph with random weights"""
    np.random.seed(42)
    adj_matrix = np.random.randint(1, 100, size=(n, n)).astype(float)
    # Set diagonal to 0
    for i in range(n):
        adj_matrix[i,i] = 0
    return adj_matrix

def floyd_warshall(adj_matrix: np.ndarray) -> np.ndarray:
    """Standard Floyd-Warshall implementation for verification"""
    n = len(adj_matrix)
    dist = adj_matrix.copy()
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i,k] + dist[k,j] < dist[i,j]:
                    dist[i,j] = dist[i,k] + dist[k,j]
    return dist

def compute_min_plus_product(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute min-plus matrix multiplication"""
    n, m1 = A.shape
    m2, p = B.shape
    assert m1 == m2, "Matrix dimensions don't match"
    
    C = np.full((n, p), float('inf'))
    for i in range(n):
        for j in range(p):
            for k in range(m1):
                C[i,j] = min(C[i,j], A[i,k] + B[k,j])
    return C

def han_basic_apsp(adj_matrix: np.ndarray) -> np.ndarray:
    """Basic implementation of APSP algorithm"""
    n = len(adj_matrix)
    D = adj_matrix.copy()
    
    # Block size calculation
    log_n = math.log2(n)
    log_log_n = math.log2(log_n) if n > 1 else 1
    block_size = max(1, int((log_n / log_log_n) ** 0.5))
    
    # Process blocks
    for phase in range(math.ceil(math.log2(n))):
        D_new = D.copy()
        
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                for k in range(0, n, block_size):
                    # Get block ranges
                    i_end = min(i + block_size, n)
                    j_end = min(j + block_size, n)
                    k_end = min(k + block_size, n)
                    
                    # Extract blocks
                    block_ik = D[i:i_end, k:k_end]
                    block_kj = D[k:k_end, j:j_end]
                    
                    # Compute block product
                    product = compute_min_plus_product(block_ik, block_kj)
                    
                    # Update result
                    D_new[i:i_end, j:j_end] = np.minimum(
                        D_new[i:i_end, j:j_end],
                        product
                    )
        
        # Check if we've reached fixed point
        if np.array_equal(D, D_new):
            break
            
        D = D_new
    
    return D

def verify_result(adj_matrix: np.ndarray, result: np.ndarray) -> bool:
    """Verify result against Floyd-Warshall"""
    correct = floyd_warshall(adj_matrix)
    
    if not np.allclose(correct, result, rtol=1e-5, atol=1e-5):
        not_equal = ~np.isclose(correct, result, rtol=1e-5, atol=1e-5)
        diff_indices = np.where(not_equal)
        print(f"\nFirst few differing elements:")
        for idx in zip(*diff_indices)[:5]:  # Show first 5 differences
            print(f"Position {idx}:")
            print(f"Expected: {correct[idx]}")
            print(f"Got: {result[idx]}")
        return False
    return True

def measure_execution_time() -> List[Tuple[int, float]]:
    """Measure execution time for different graph sizes"""
    results = []
    sizes = [32, 64, 128,256]  # Smaller sizes for testing
    
    for n in sizes:
        print(f"\nTesting size n={n}")
        graph = generate_test_graph(n)
        
        # First verify small sample
        if n == 32:
            print("\nVerifying small sample...")
            sample = graph[:4, :4].copy()
            result_sample = han_basic_apsp(sample)
            correct_sample = floyd_warshall(sample)
            print("Sample input:")
            print(sample)
            print("\nSample output:")
            print(result_sample)
            print("\nExpected output:")
            print(correct_sample)
        
        start_time = time.time()
        result = han_basic_apsp(graph)
        execution_time = time.time() - start_time
        
        # Verify result
        is_correct = verify_result(graph, result)
        if is_correct:
            print(f"✓ Verification passed for n={n}")
        else:
            print(f"✗ Verification failed for n={n}")
        
        results.append((n, execution_time))
        print(f"Time taken: {execution_time:.2f} seconds")
    
    return results

def plot_results(results: List[Tuple[int, float]]):
    """Plot results comparing actual vs theoretical time"""
    sizes, times = zip(*results)
    
    # Calculate theoretical times
    theoretical_times = [
        n**3 * math.log(math.log(n))**3 / math.log(n)**2 
        for n in sizes
    ]
    
    # Scale theoretical times
    scale_factor = max(times) / max(theoretical_times)
    theoretical_times = [t * scale_factor for t in theoretical_times]
    
    plt.figure(figsize=(12, 7))
    plt.plot(sizes, times, 'bo-', label='Actual Running Time', linewidth=2, markersize=8)
    plt.plot(sizes, theoretical_times, 'r--', label='Theoretical Complexity', linewidth=2)
    
    plt.xlabel('Graph Size (n)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title("Han's APSP Algorithm: Actual vs Theoretical Running Time", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.tight_layout()
    plt.savefig('han_apsp_analysis1.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Add debug code to check small example first
    print("Testing with small example first:")
    test_size = 4
    test_graph = generate_test_graph(test_size)
    print("\nTest graph:")
    print(test_graph)
    
    print("\nFloyd-Warshall result:")
    fw_result = floyd_warshall(test_graph)
    print(fw_result)
    
    print("\nOur algorithm result:")
    our_result = han_basic_apsp(test_graph)
    print(our_result)
    
    if np.allclose(fw_result, our_result, rtol=1e-5, atol=1e-5):
        print("\n✓ Small test passed!")
        # Continue with full testing
        results = measure_execution_time()
        plot_results(results)
        
        print("\nDetailed Results:")
        print("Size | Time (seconds)")
        print("-" * 20)
        for size, time_taken in results:
            print(f"{size:4d} | {time_taken:.4f}")
    else:
        print("\n✗ Small test failed!")
        print("Debug output:")
        print("Differences:")
        print(np.abs(fw_result - our_result))