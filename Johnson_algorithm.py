import numpy as np
import time
import heapq

def generate_test_graph(n: int) -> np.ndarray:
    """
    Generate a random graph with n vertices, where edges have random weights.
    """
    adj_matrix = np.random.randint(1, 100, size=(n, n)).astype(float)
    for i in range(n):
        for j in range(n):
            if i == j:
                adj_matrix[i][j] = 0
            elif np.random.random() < 0.3:  # ~30% chance for no edge
                adj_matrix[i][j] = float('inf')
    return adj_matrix

def bellman_ford(adj_matrix: np.ndarray, source: int) -> np.ndarray:
    """
    Bellman-Ford algorithm to compute shortest paths from the source node.
    """
    n = len(adj_matrix)
    dist = np.full(n, float('inf'))
    dist[source] = 0
    
    for _ in range(n - 1):
        for u in range(n):
            for v in range(n):
                if adj_matrix[u][v] != float('inf') and dist[u] + adj_matrix[u][v] < dist[v]:
                    dist[v] = dist[u] + adj_matrix[u][v]
    return dist

def dijkstra(adj_matrix: np.ndarray, source: int) -> np.ndarray:
    """
    Dijkstra's algorithm to compute shortest paths from the source node.
    """
    n = len(adj_matrix)
    dist = np.full(n, float('inf'))
    dist[source] = 0
    pq = [(0, source)]  # Priority queue (min-heap)
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v in range(n):
            if adj_matrix[u][v] != float('inf') and dist[u] + adj_matrix[u][v] < dist[v]:
                dist[v] = dist[u] + adj_matrix[u][v]
                heapq.heappush(pq, (dist[v], v))
    return dist

def johnsons_algorithm(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Implementation of Johnson's Algorithm for All Pairs Shortest Path (APSP).
    """
    n = len(adj_matrix)
    # Step 1: Add a new vertex (n) and connect it to all others with zero-weight edges.
    new_graph = np.vstack([adj_matrix, np.zeros(n)])
    new_graph = np.hstack([new_graph, np.zeros((n+1, 1))])
    
    # Step 2: Run Bellman-Ford from the new vertex (n) to compute potential function.
    potential = bellman_ford(new_graph, n)
    
    # Step 3: Re-weight the original graph using the potential values.
    reweighted_graph = adj_matrix.copy()
    for u in range(n):
        for v in range(n):
            if reweighted_graph[u][v] != float('inf'):
                reweighted_graph[u][v] += potential[u] - potential[v]
    
    # Step 4: Run Dijkstra for each vertex in the re-weighted graph.
    all_pairs_shortest_paths = np.full((n, n), float('inf'))
    for u in range(n):
        all_pairs_shortest_paths[u] = dijkstra(reweighted_graph, u)
    
    # Step 5: Revert the re-weighting.
    for u in range(n):
        for v in range(n):
            if all_pairs_shortest_paths[u][v] != float('inf'):
                all_pairs_shortest_paths[u][v] += potential[v] - potential[u]
    
    return all_pairs_shortest_paths

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
        _ = johnsons_algorithm(graph)
        end_time = time.time()
        
        execution_time = end_time - start_time
        results.append((n, execution_time))
        print(f"Time taken: {execution_time:.2f} seconds")
    
    return results

def plot_results(results: list[tuple[int, float]]):
    """
    Plot the results and compare with theoretical complexity
    """
    import matplotlib.pyplot as plt
    import math
    
    sizes, times = zip(*results)
    
    # Theoretical complexity for Johnson's Algorithm: O(V^2 log V + VE)
    theoretical_times = [n**2 * np.log(n) for n in sizes]
    
    max_actual = max(times)
    max_theoretical = max(theoretical_times)
    theoretical_times = [t * max_actual / max_theoretical for t in theoretical_times]
    
    plt.figure(figsize=(12, 7))
    plt.plot(sizes, times, 'bo-', label='Actual Running Time', linewidth=2, markersize=8)
    plt.plot(sizes, theoretical_times, 'r--', label='Theoretical Complexity', linewidth=2)
    
    plt.xticks(sizes, [str(size) for size in sizes], rotation=0)
    plt.xlabel('Graph Size (n)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Johnson Algorithm: Actual vs Theoretical Running Time', fontsize=14, pad=15)
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
