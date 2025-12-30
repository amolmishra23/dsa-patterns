# Graphs Advanced - Practice List

## Problems by Pattern

### Topological Sort
- LC 207: Course Schedule (Medium)
- LC 210: Course Schedule II (Medium)
- LC 269: Alien Dictionary (Hard)
- LC 310: Minimum Height Trees (Medium)
- LC 1136: Parallel Courses (Medium)

### Union-Find
- LC 200: Number of Islands (Medium) - alternative
- LC 547: Number of Provinces (Medium)
- LC 684: Redundant Connection (Medium)
- LC 721: Accounts Merge (Medium)
- LC 990: Satisfiability of Equality Equations (Medium)
- LC 1319: Number of Operations to Make Network Connected (Medium)

### Shortest Path (Dijkstra)
- LC 743: Network Delay Time (Medium)
- LC 787: Cheapest Flights Within K Stops (Medium)
- LC 1514: Path with Maximum Probability (Medium)
- LC 1631: Path With Minimum Effort (Medium)

### MST (Minimum Spanning Tree)
- LC 1135: Connecting Cities With Minimum Cost (Medium)
- LC 1584: Min Cost to Connect All Points (Medium)

## Templates

```python
import heapq
from collections import defaultdict, deque

# Topological Sort (Kahn's BFS)
def topological_sort(n, edges):
    graph = defaultdict(list)
    indegree = [0] * n
    for u, v in edges:
        graph[u].append(v)
        indegree[v] += 1
    
    queue = deque([i for i in range(n) if indegree[i] == 0])
    order = []
    
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    
    return order if len(order) == n else []

# Union-Find with Path Compression and Union by Rank
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

# Dijkstra's Algorithm
def dijkstra(graph, start, n):
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]
    
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        for neighbor, weight in graph[node]:
            new_dist = d + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))
    
    return dist

# Bellman-Ford (handles negative weights)
def bellman_ford(n, edges, start):
    dist = [float('inf')] * n
    dist[start] = 0
    
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    
    # Check negative cycle
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # Negative cycle
    
    return dist

# Kruskal's MST
def kruskal_mst(n, edges):
    edges.sort(key=lambda x: x[2])  # Sort by weight
    uf = UnionFind(n)
    mst_cost = 0
    edges_used = 0
    
    for u, v, w in edges:
        if uf.union(u, v):
            mst_cost += w
            edges_used += 1
            if edges_used == n - 1:
                break
    
    return mst_cost if edges_used == n - 1 else -1
```

## Key Insights
- Topological sort: only for DAGs (no cycles)
- Union-Find: dynamic connectivity, cycle detection
- Dijkstra: non-negative weights, greedy
- Bellman-Ford: handles negative weights, detects negative cycles
- MST: Kruskal (sort edges) or Prim (grow from node)

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED GRAPH ALGORITHMS                                │
│                                                                             │
│  TOPOLOGICAL SORT (Kahn's Algorithm):                                       │
│                                                                             │
│  Graph: 0 → 1 → 3                                                           │
│         ↓   ↓                                                               │
│         2 → 4                                                               │
│                                                                             │
│  Initial indegree: [0, 1, 1, 1, 2]                                          │
│  Queue: [0]                                                                 │
│  Process 0 → Queue: [1, 2], Order: [0]                                      │
│  Process 1 → Queue: [2, 3], Order: [0, 1]                                   │
│  Process 2 → Queue: [3, 4], Order: [0, 1, 2]                                │
│  Process 3 → Queue: [4], Order: [0, 1, 2, 3]                                │
│  Process 4 → Order: [0, 1, 2, 3, 4] ✓                                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  UNION-FIND:                                                                │
│                                                                             │
│  Initial: parent = [0, 1, 2, 3, 4]  (each is its own parent)                │
│                                                                             │
│  union(0, 1):  0 ← 1                parent = [0, 0, 2, 3, 4]                 │
│  union(2, 3):  2 ← 3                parent = [0, 0, 2, 2, 4]                 │
│  union(0, 2):  0 ← 2                parent = [0, 0, 0, 2, 4]                 │
│                                                                             │
│  Tree structure:      0                                                     │
│                      /|\                                                    │
│                     1 2 (3 points to 2)                                     │
│                       |                                                     │
│                       3                                                     │
│                                                                             │
│  find(3) with path compression: 3 → 2 → 0, update 3 → 0                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DIJKSTRA'S ALGORITHM:                                                      │
│                                                                             │
│       1 ──(4)── 2                                                           │
│      /│         │\                                                          │
│    (1)│       (1)│(3)                                                       │
│    /  │         │  \                                                        │
│   0  (2)       (2)  4                                                       │
│    \  │         │  /                                                        │
│    (5)│       (1)│(1)                                                       │
│      \│         │/                                                          │
│       3 ──(2)── 5                                                           │
│                                                                             │
│  From node 0:                                                               │
│  dist = [0, ∞, ∞, ∞, ∞, ∞]                                                  │
│  Process 0: update 1(1), 3(5) → dist = [0, 1, ∞, 5, ∞, ∞]                   │
│  Process 1: update 2(5) → dist = [0, 1, 5, 5, ∞, ∞]                         │
│  Process 3: update 5(7) → dist = [0, 1, 5, 5, ∞, 7]                         │
│  ...                                                                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  KRUSKAL'S MST:                                                             │
│                                                                             │
│  Edges sorted by weight:                                                    │
│  (0,1,1), (2,3,1), (1,2,2), (0,3,3), (1,3,4)                               │
│                                                                             │
│  Process (0,1,1): union(0,1) ✓  MST cost = 1                                │
│  Process (2,3,1): union(2,3) ✓  MST cost = 2                                │
│  Process (1,2,2): union(1,2) ✓  MST cost = 4                                │
│  Process (0,3,3): same component, skip                                      │
│                                                                             │
│  MST:  0 ─(1)─ 1 ─(2)─ 2                                                    │
│                        │                                                    │
│                       (1)                                                   │
│                        │                                                    │
│                        3                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: Topological Sort
- [ ] LC 207: Course Schedule (Medium)
- [ ] LC 210: Course Schedule II (Medium)
- [ ] LC 269: Alien Dictionary (Hard)
- [ ] LC 310: Minimum Height Trees (Medium)
- [ ] LC 1136: Parallel Courses (Medium)

### Week 2: Union-Find
- [ ] LC 547: Number of Provinces (Medium)
- [ ] LC 684: Redundant Connection (Medium)
- [ ] LC 721: Accounts Merge (Medium)
- [ ] LC 990: Satisfiability of Equality Equations (Medium)
- [ ] LC 1319: Number of Operations to Make Network Connected (Medium)

### Week 3: Shortest Path & MST
- [ ] LC 743: Network Delay Time (Medium)
- [ ] LC 787: Cheapest Flights Within K Stops (Medium)
- [ ] LC 1514: Path with Maximum Probability (Medium)
- [ ] LC 1584: Min Cost to Connect All Points (Medium)
- [ ] LC 1631: Path With Minimum Effort (Medium)

---

## Common Mistakes

### 1. Topological Sort on Cyclic Graph
```python
# WRONG - not checking for cycles
def topo_sort(n, edges):
    # ... process all nodes
    return order  # May be incomplete if cycle exists!

# CORRECT - verify all nodes processed
def topo_sort(n, edges):
    # ... process nodes
    return order if len(order) == n else []  # Empty = cycle detected
```

### 2. Union-Find Without Path Compression
```python
# WRONG - O(n) find without compression
def find(self, x):
    while self.parent[x] != x:
        x = self.parent[x]
    return x

# CORRECT - O(α(n)) with path compression
def find(self, x):
    if self.parent[x] != x:
        self.parent[x] = self.find(self.parent[x])  # Compress path
    return self.parent[x]
```

### 3. Dijkstra with Negative Weights
```python
# WRONG - Dijkstra doesn't work with negative weights
# Use Bellman-Ford instead

# Dijkstra assumes: once a node is processed, its distance is final
# Negative weights can violate this assumption
```

### 4. Not Skipping Outdated Heap Entries
```python
# WRONG - processes outdated distances
while heap:
    d, node = heapq.heappop(heap)
    for neighbor, weight in graph[node]:
        # May process same node multiple times with wrong distance

# CORRECT - skip if already found better path
while heap:
    d, node = heapq.heappop(heap)
    if d > dist[node]:  # Outdated entry
        continue
    for neighbor, weight in graph[node]:
        ...
```

---

## Complexity Reference

| Algorithm | Time | Space | Use Case |
|-----------|------|-------|----------|
| Topo Sort (Kahn) | O(V + E) | O(V) | DAG ordering |
| Union-Find | O(α(n)) per op | O(n) | Dynamic connectivity |
| Dijkstra | O((V+E) log V) | O(V) | Non-negative weights |
| Bellman-Ford | O(V * E) | O(V) | Negative weights |
| Kruskal MST | O(E log E) | O(V) | Sparse graphs |
| Prim MST | O((V+E) log V) | O(V) | Dense graphs |

---

## Pattern Recognition

| See This | Think This |
|----------|------------|
| "Order of dependencies" | Topological sort |
| "Connected components" | Union-Find or DFS |
| "Detect cycle in undirected" | Union-Find |
| "Shortest path, positive weights" | Dijkstra |
| "Shortest path, negative weights" | Bellman-Ford |
| "K stops/edges limit" | Modified Bellman-Ford/BFS |
| "Minimum cost to connect all" | MST (Kruskal/Prim) |
| "Dynamic connectivity" | Union-Find |
