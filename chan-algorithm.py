import numpy as np
import time
import math
from typing import List, Tuple, Set
import matplotlib.pyplot as plt

def generate_test_graph(n: int) -> np.ndarray:
    """Generate a test graph with random weights and some infinity edges"""
    adj_matrix = np.random.randint(1, 100, size=(n, n)).astype(float)
    for i in range(n):
        for j in range(n):
            if i == j:
                adj_matrix[i][j] = 0
            elif np.random.random() < 0.3:
                adj_matrix[i][j] = float('inf')
    return adj_matrix

class Cell:
    """Represents a geometric cell from the partition theorem"""
    def __init__(self):
        self.points = []  # List of point indices
        self.hyperplanes = []  # List of hyperplane indices

def compute_rect_product(A: np.ndarray, B: np.ndarray, word_size: int) -> np.ndarray:
    """
    Compute the distance product of rectangular matrices A and B using geometric approach
    A is n×d, B is d×n
    """
    n, d = A.shape
    assert B.shape == (d, n), "Invalid matrix dimensions"
    
    # Form point set from matrix A
    points = [A[i, :] for i in range(n)]
    
    # Form hyperplanes from matrix B
    hyperplanes = []
    for j in range(n):
        for k in range(d):
            for l in range(d):
                if k != l:
                    # Hyperplane equation: x_k + b_kj = x_l + b_lj
                    hyperplanes.append((j, k, l, B[k,j], B[l,j]))
    
    # Compute partition parameter r
    kappa = 3  # Geometric parameter, can be adjusted based on dimension
    r = int(n**((3-2.376)/(1+1/kappa)))  # Using ω ≈ 2.376 for matrix multiplication
    
    # Create cells using simplified partition (for demonstration)
    num_cells = (n//r + 1)**d
    cells = [Cell() for _ in range(num_cells)]
    
    # Assign points to cells
    for i, point in enumerate(points):
        cell_idx = 0
        valid_point = True
        for dim in range(d):
            if np.isinf(point[dim]):
                valid_point = False
                break
            cell_coord = min(int(point[dim] / (100/r)), n//r)
            cell_idx = cell_idx * (n//r + 1) + cell_coord
        
        if valid_point:
            cells[cell_idx].points.append(i)
    
    # Initialize result matrix
    C = np.full((n, n), -1, dtype=int)
    
    # Process each cell
    for cell in cells:
        if not cell.points:
            continue
            
        for j in range(n):
            # Pick arbitrary point from cell
            sample_point = points[cell.points[0]]
            
            # Find initial minimum index
            min_k = 0
            min_val = float('inf')
            for k in range(d):
                if not np.isinf(sample_point[k]) and not np.isinf(B[k,j]):
                    val = sample_point[k] + B[k,j]
                    if val < min_val:
                        min_val = val
                        min_k = k
                    
            # Set tentative values
            for i in cell.points:
                C[i,j] = min_k if min_val != float('inf') else -1
                
            # Correct values using hyperplanes
            for i in cell.points:
                point = points[i]
                for j2, k, l, bkj, blj in hyperplanes:
                    if j2 == j and not np.isinf(point[k]) and not np.isinf(bkj) and not np.isinf(point[l]) and not np.isinf(blj):
                        if point[k] + bkj < point[l] + blj:
                            C[i,j] = k
                            
    return C

def min_elements(C1: np.ndarray, C2: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute element-wise minimum of two matrices based on actual path lengths"""
    n = C1.shape[0]
    R = np.zeros_like(C1)
    
    for i in range(n):
        for j in range(n):
            # Handle infinite values
            if np.isinf(C1[i,j]):
                idx1 = -1
            else:
                idx1 = int(C1[i,j])
                
            if np.isinf(C2[i,j]):
                idx2 = -1
            else:
                idx2 = int(C2[i,j])
            
            val1 = A[i,idx1] + B[idx1,j] if idx1 >= 0 else float('inf')
            val2 = A[i,idx2] + B[idx2,j] if idx2 >= 0 else float('inf')
            R[i,j] = C1[i,j] if val1 <= val2 else C2[i,j]
    
    return R

def improved_apsp(adj_matrix: np.ndarray) -> np.ndarray:
    """Implementation of Chan's improved APSP algorithm"""
    n = len(adj_matrix)
    word_size = 64  # Typical word size
    d = max(1, int(0.1 * math.log(n) / math.log(word_size)))  # δ = 0.1
    
    # Divide matrices
    A_blocks = []
    B_blocks = []
    for i in range(0, n, d):
        end = min(i + d, n)
        A_blocks.append(adj_matrix[:, i:end])
        B_blocks.append(adj_matrix[i:end, :])
    
    # Initialize result
    C = np.full((n, n), float('inf'))
    
    # Process each block pair
    for A_i, B_i in zip(A_blocks, B_blocks):
        C_i = compute_rect_product(A_i, B_i, word_size)
        C = min_elements(C, C_i, adj_matrix, adj_matrix)
    
    return C

def measure_execution_time() -> List[Tuple[int, float]]:
    """Measure execution time for different graph sizes"""
    results = []
    sizes = [32, 64, 128, 256, 512]  # Adjusted for demonstration
    
    for n in sizes:
        print(f"\nTesting size n={n}")
        graph = generate_test_graph(n)
        
        # First verify small sample if this is the first test
        if n == 32:
            print("\nVerifying small sample...")
            sample = graph[:4, :4].copy()
            result_sample = improved_apsp(sample)
            correct_sample = verify_floyd_warshall(sample)
            print("Sample input:")
            print(sample)
            print("\nSample output:")
            print(result_sample)
            print("\nExpected output:")
            print(correct_sample)
        
        start_time = time.time()
        result = improved_apsp(graph)
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
def verify_floyd_warshall(adj_matrix: np.ndarray) -> np.ndarray:
    """Standard Floyd-Warshall implementation for verification"""
    n = len(adj_matrix)
    dist = adj_matrix.copy()
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if not np.isinf(dist[i,k]) and not np.isinf(dist[k,j]):
                    dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
    return dist

def plot_results(results: List[Tuple[int, float]]):
    """Plot the results comparing actual running time with theoretical complexity"""
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
    
    plt.xticks(sizes, [str(size) for size in sizes], rotation=0)
    plt.xlabel('Graph Size (n)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title("Chan's APSP Algorithm: Actual vs Theoretical Running Time", fontsize=14, pad=15)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.tight_layout()
    plt.savefig('chan_apsp_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
def verify_result(adj_matrix: np.ndarray, result: np.ndarray) -> bool:
    """Verify result against Floyd-Warshall"""
    expected = verify_floyd_warshall(adj_matrix)
    
    if not np.allclose(expected, result, rtol=1e-5, atol=1e-5, equal_nan=True):
        not_equal = ~np.isclose(expected, result, rtol=1e-5, atol=1e-5, equal_nan=True)
        diff_indices = np.where(not_equal)
        print("\nFirst few differing elements:")
        for idx in zip(*diff_indices)[:5]:  # Show first 5 differences
            print(f"Position {idx}:")
            print(f"Expected: {expected[idx]}")
            print(f"Got: {result[idx]}")
            print(f"Original value: {adj_matrix[idx]}")
        return False
    return True
if __name__ == "__main__":
    # Test with a small example first
    print("Testing with small example first:")
    test_size = 4
    np.random.seed(42)  # For reproducibility
    test_graph = generate_test_graph(test_size)
    print("\nTest graph:")
    print(test_graph)
    
    print("\nFloyd-Warshall result:")
    fw_result = verify_floyd_warshall(test_graph)
    print(fw_result)
    
    print("\nOur algorithm result:")
    our_result = improved_apsp(test_graph)
    print(our_result)
    
    if np.allclose(fw_result, our_result, rtol=1e-5, atol=1e-5, equal_nan=True):
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