import numpy as np
import time
import matplotlib.pyplot as plt
import math

def generate_test_graph(n: int) -> np.ndarray:
    """
    Generate a test graph with specific structure similar to the original code.
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

def floyd_warshall(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Implementation of Floyd-Warshall algorithm for All Pairs Shortest Paths
    Args:
        adj_matrix: Input adjacency matrix with weights
    Returns:
        Matrix of shortest paths between all pairs of vertices
    """
    n = len(adj_matrix)
    dist = adj_matrix.copy()
    
    # Main algorithm implementation
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    return dist

def measure_execution_time() -> list[tuple[int, float]]:
    """
    Measure execution time for different graph sizes
    """
    results = []
    sizes = [32, 64, 128, 256, 512, 1024]  # Same sizes as original code
    
    for n in sizes:
        print(f"Testing size n={n}")
        graph = generate_test_graph(n)
        
        start_time = time.time()
        _ = floyd_warshall(graph)
        end_time = time.time()
        
        execution_time = end_time - start_time
        results.append((n, execution_time))
        print(f"Time taken: {execution_time:.2f} seconds")
    
    return results

def plot_results(results: list[tuple[int, float]]):
    """
    Plot the results and compare with theoretical complexity O(n³)
    """
    sizes, times = zip(*results)
    
    # For Floyd-Warshall, theoretical complexity is exactly O(n³)
    theoretical_times = [n**3 for n in sizes]
    max_actual = max(times)
    max_theoretical = max(theoretical_times)
    theoretical_times = [t * max_actual / max_theoretical for t in theoretical_times]
    
    plt.figure(figsize=(12, 7))
    plt.plot(sizes, times, 'bo-', label='Actual Running Time', linewidth=2, markersize=8)
    plt.plot(sizes, theoretical_times, 'r--', label='Theoretical Complexity O(n³)', linewidth=2)
    
    plt.xticks(sizes, [str(size) for size in sizes], rotation=0)
    plt.xlabel('Graph Size (n)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Floyd-Warshall Algorithm: Actual vs Theoretical Running Time', fontsize=14, pad=15)
    plt.legend(fontsize=11)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.yscale('log')
    plt.xscale('log', base=2)
    plt.tight_layout()
    plt.savefig('floyd_warshall_complexity.png', dpi=300, bbox_inches='tight')
    plt.close()

# Run the experiment
if __name__ == "__main__":
    results = measure_execution_time()
    plot_results(results)