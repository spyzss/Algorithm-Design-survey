import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Set, Dict, Tuple
import math
import random
import sys

class Component:
    def __init__(self, vertices: Set[int], level: int, id: int):
        self.vertices = vertices    # 组件中的顶点集
        self.level = level         # 组件所在的层次
        self.id = id              # 组件的唯一标识
        self.children = []         # 子组件列表
        self.parent = None        # 父组件
        self.min_distance = float('inf')  # 组件中顶点的最小距离
        self.ix = 0              # 组件的索引，用于桶排序
        self.ix0 = 0             # 初始索引
        self.ix_infinity = 0     # 最大索引限制
        self.buckets = {}        # 组件的桶结构

class BucketStructure:
    """更完整的桶结构实现"""
    def __init__(self, n: int):
        self.n = n
        self.buckets = {}  # 主桶结构
        self.waste_bucket = set()  # 废弃桶
        self.relevant_range = {}  # 每个组件的相关范围
        
    def get_bucket_index(self, component: Component, value: float) -> int:
        """计算桶索引"""
        if value == float('inf'):
            return sys.maxsize
        if component.level > 0:
            return int(value / (2 ** (component.level - 1)))
        return int(value)
        
    def initialize_component_buckets(self, component: Component):
        """为组件初始化桶结构"""
        if component.min_distance == float('inf'):
            component.ix0 = 0
        else:
            component.ix0 = self.get_bucket_index(component, component.min_distance)
            
        if not hasattr(component, 'edges'):
            component.edges = []
            
        delta = 1
        if component.edges:
            delta = sum(w for _, w in component.edges) / (2 ** max(0, component.level - 1))
        component.ix_infinity = component.ix0 + int(delta)
        self.relevant_range[component.id] = (component.ix0, component.ix_infinity)
        
        # 初始化相关桶
        for idx in range(component.ix0, min(component.ix_infinity + 1, 1000000)):  # 限制桶的数量
            if (component.id, idx) not in self.buckets:
                self.buckets[(component.id, idx)] = set()

    def insert(self, component: Component, vertex: int, value: float):
        """将顶点插入到对应的桶中"""
        bucket_idx = self.get_bucket_index(component, value)
        if component.ix0 <= bucket_idx <= component.ix_infinity:
            self.buckets[(component.id, bucket_idx)].add(vertex)
        else:
            self.waste_bucket.add(vertex)
            
    def remove(self, component: Component, vertex: int, old_value: float):
        """从桶中移除顶点"""
        old_idx = self.get_bucket_index(component, old_value)
        if (component.id, old_idx) in self.buckets:
            self.buckets[(component.id, old_idx)].discard(vertex)
            
    def update(self, component: Component, vertex: int, old_value: float, new_value: float):
        """更新顶点在桶中的位置"""
        self.remove(component, vertex, old_value)
        self.insert(component, vertex, new_value)
        
    def get_min_bucket(self, component: Component) -> Set[int]:
        """获取组件的最小桶"""
        for idx in range(component.ix0, component.ix_infinity + 1):
            if (component.id, idx) in self.buckets and self.buckets[(component.id, idx)]:
                return self.buckets[(component.id, idx)]
        return set()

class ComponentTree:
    """组件树的完整实现"""
    def __init__(self, n: int, edges: List[Tuple[int, int, int]]):
        self.n = n
        self.edges = edges
        self.components = []  # 所有组件的列表
        self.vertex_to_component = {}  # 顶点到组件的映射
        self.next_id = 0
        
    def build_tree(self) -> Component:
        """构建组件树"""
        # 按照权重对边进行排序
        sorted_edges = sorted(self.edges, key=lambda x: x[2])
        
        # 初始化单顶点组件
        components = []
        for i in range(self.n):
            comp = Component({i}, 0, self.next_id)
            self.next_id += 1
            components.append(comp)
            self.vertex_to_component[i] = comp
            self.components.append(comp)
            
        current_level = 0
        current_weight = 0
        
        for u, v, w in sorted_edges:
            if math.log2(w) > current_level:
                # 创建新层次
                current_level = int(math.log2(w))
                self._create_new_level(components, current_level)
            
            # 合并组件
            comp_u = self._find_component(u)
            comp_v = self._find_component(v)
            if comp_u != comp_v:
                new_comp = self._merge_components(comp_u, comp_v, current_level)
                components.append(new_comp)
                
        return components[-1]  # 返回根组件
        
    def _create_new_level(self, components: List[Component], level: int):
        """创建新的层次"""
        new_components = []
        visited = set()
        
        for comp in components:
            if comp.id not in visited and comp.parent is None:
                new_comp = Component(comp.vertices.copy(), level, self.next_id)
                self.next_id += 1
                new_comp.children.append(comp)
                comp.parent = new_comp
                new_components.append(new_comp)
                self.components.append(new_comp)
                visited.add(comp.id)
                
        return new_components
        
    def _merge_components(self, comp1: Component, comp2: Component, level: int) -> Component:
        """合并两个组件"""
        new_vertices = comp1.vertices.union(comp2.vertices)
        new_comp = Component(new_vertices, level, self.next_id)
        self.next_id += 1
        
        new_comp.children = [comp1, comp2]
        comp1.parent = new_comp
        comp2.parent = new_comp
        
        for v in new_vertices:
            self.vertex_to_component[v] = new_comp
            
        self.components.append(new_comp)
        return new_comp
        
    def _find_component(self, vertex: int) -> Component:
        """找到顶点所属的组件"""
        return self.vertex_to_component[vertex]

def thorup_sssp(graph: List[List[Tuple[int, int]]], source: int, n: int) -> List[float]:
    """Thorup的线性时间SSSP算法的完整实现"""
    # 初始化距离数组
    distances = [float('inf')] * n
    distances[source] = 0
    visited = set()
    
    # 构建边的列表
    edges = []
    for u in range(n):
        for v, w in graph[u]:
            if u < v:  # 避免重复边
                edges.append((u, v, w))
                
    # 构建组件树
    component_tree = ComponentTree(n, edges)
    root = component_tree.build_tree()
    
    # 创建桶结构
    bucket_structure = BucketStructure(n)
    
    def visit_component(component: Component):
        """访问组件"""
        if component.level == 0:
            # 访问单个顶点
            vertex = list(component.vertices)[0]
            if vertex not in visited:
                visit_vertex(vertex)
            return
            
        # 初始化组件的桶
        if component.id not in bucket_structure.relevant_range:
            bucket_structure.initialize_component_buckets(component)
            
        while True:
            min_bucket = bucket_structure.get_min_bucket(component)
            if not min_bucket:
                break
                
            for child in component.children:
                if child.vertices.intersection(min_bucket):
                    visit_component(child)
                    
            component.ix += 1
            if component.ix > component.ix_infinity:
                break
                
    def visit_vertex(vertex: int):
        """访问顶点"""
        visited.add(vertex)
        for u, w in graph[vertex]:
            if distances[vertex] + w < distances[u]:
                old_dist = distances[u]
                distances[u] = distances[vertex] + w
                
                # 更新受影响组件的桶
                comp = component_tree._find_component(u)
                while comp is not None:
                    bucket_structure.update(comp, u, old_dist, distances[u])
                    comp = comp.parent
                    
    # 从根组件开始访问
    visit_component(root)
    
    return distances

def measure_execution_time() -> List[Tuple[int, float]]:
    """测试不同规模的执行时间"""
    results = []
    sizes = [1000, 2000, 4000, 8000, 16000]
    
    for n in sizes:
        print(f"Testing size n={n}")
        graph = [[] for _ in range(n)]
        
        # 生成稀疏图
        edges = set()
        for _ in range(3 * n):  # 平均度数为6的稀疏图
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v and (u, v) not in edges:
                w = random.randint(1, 1000)
                graph[u].append((v, w))
                graph[v].append((u, w))
                edges.add((u, v))
                edges.add((v, u))
                
        start_time = time.time()
        _ = thorup_sssp(graph, 0, n)
        end_time = time.time()
        
        execution_time = end_time - start_time
        results.append((n, execution_time))
        print(f"Time taken: {execution_time:.2f} seconds")
        
    return results

def plot_results(results: List[Tuple[int, float]]):
    """绘制结果对比图"""
    sizes, times = zip(*results)
    
    # 理论复杂度 O(m)，因为是稀疏图，m ≈ 3n
    theoretical_times = [3 * n for n in sizes]
    
    # 缩放理论时间以匹配实际时间
    max_actual = max(times)
    max_theoretical = max(theoretical_times)
    theoretical_times = [t * max_actual / max_theoretical for t in theoretical_times]
    
    plt.figure(figsize=(12, 7))
    plt.plot(sizes, times, 'bo-', label='Actual Running Time', linewidth=2, markersize=8)
    plt.plot(sizes, theoretical_times, 'r--', label='Linear Complexity O(m)', linewidth=2)
    
    plt.xlabel('Graph Size (n)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title("Thorup's SSSP Algorithm: Actual vs Theoretical Running Time", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig('thorup_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    results = measure_execution_time()
    plot_results(results)