import numpy as np
import time
import matplotlib.pyplot as plt
import math

def generate_test_graph(n: int) -> np.ndarray:
    """
    Generate a test graph with specific structure to match the paper's example.
    Args:
        n: Number of vertices
    Returns:
        adjacency matrix with weights
    """
    adj_matrix = np.random.randint(1, 100, size=(n, n)).astype(float)
    for i in range(n):
        for j in range(n):
            if i == j:
                adj_matrix[i][j] = 0
            elif np.random.random() < 0.3:
                adj_matrix[i][j] = float('inf')
    return adj_matrix

def han_takaoka_apsp(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Implementation of Han-Takaoka algorithm for All Pairs Shortest Paths
    """
    n = len(adj_matrix)
    dist = adj_matrix.copy()
    
    # Parameters as described in the paper
    r1 = 0.5  # Example value, can be optimized
    t1 = int(n**(1-r1))
    
    # Main algorithm implementation
    def process_submatrix(E: np.ndarray, F: np.ndarray) -> np.ndarray:
        rows_E, cols_E = E.shape
        cols_F = F.shape[1]
        result = np.full((rows_E, cols_F), float('inf'))
        
        # Modify r2 and block_size calculation to ensure they are not too small
        r2 = max(1, int(math.log(n)/math.log(math.log(n))))
        block_size = max(1, min(rows_E, cols_E, cols_F, int(r2 * math.log(n)/math.log(math.log(n)))))
        
        # Process blocks
        for i in range(0, rows_E, block_size):
            for j in range(0, cols_F, block_size):
                for k in range(0, cols_E, block_size):
                    actual_i_size = min(block_size, rows_E - i)
                    actual_j_size = min(block_size, cols_F - j)
                    actual_k_size = min(block_size, cols_E - k)
                    
                    E_block = E[i:i+actual_i_size, k:k+actual_k_size]
                    F_block = F[k:k+actual_k_size, j:j+actual_j_size]
                    
                    for bi in range(actual_i_size):
                        for bj in range(actual_j_size):
                            min_val = float('inf')
                            for bk in range(actual_k_size):
                                min_val = min(min_val, E_block[bi, bk] + F_block[bk, bj])
                            result[i+bi, j+bj] = min(result[i+bi, j+bj], min_val)
        
        return result
    
    # Divide matrices and process
    for k in range(t1):
        start_col = (k * n) // t1
        end_col = ((k + 1) * n) // t1
        
        A_k = dist[:, start_col:end_col]
        B_k = dist[start_col:end_col, :]
        
        C_k = process_submatrix(A_k, B_k)
        dist = np.minimum(dist, C_k)
    
    return dist

def measure_execution_time() -> list[tuple[int, float]]:
    """
    Measure execution time for different graph sizes
    """
    results = []
    sizes = [32, 64, 128, 256, 512, 1024]
    
    for n in sizes:
        print(f"Testing size n={n}")
        graph = generate_test_graph(n)
        
        start_time = time.time()
        _ = han_takaoka_apsp(graph)
        end_time = time.time()
        
        execution_time = end_time - start_time
        results.append((n, execution_time))
        print(f"Time taken: {execution_time:.2f} seconds")
    
    return results

def plot_results(results: list[tuple[int, float]]):
    """
    Plot the results and compare with theoretical complexity
    """
    sizes, times = zip(*results)
    
    theoretical_times = [n**3 * math.log(math.log(n)) / (math.log(n)**2) for n in sizes]
    max_actual = max(times)
    max_theoretical = max(theoretical_times)
    theoretical_times = [t * max_actual / max_theoretical for t in theoretical_times]
    
    plt.figure(figsize=(12, 7))
    plt.plot(sizes, times, 'bo-', label='Actual Running Time', linewidth=2, markersize=8)
    plt.plot(sizes, theoretical_times, 'r--', label='Theoretical Complexity', linewidth=2)
    
    plt.xticks(sizes, [str(size) for size in sizes], rotation=0)
    plt.xlabel('Graph Size (n)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('APSP Algorithm: Actual vs Theoretical Running Time', fontsize=14, pad=15)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.tight_layout()
    plt.savefig('complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run the experiment
results = measure_execution_time()
plot_results(results)