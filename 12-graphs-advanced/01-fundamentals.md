# Advanced Graphs - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE ADVANCED GRAPH ALGORITHMS                    │
│                                                                             │
│  TOPOLOGICAL SORT:                                                          │
│  ✓ "Course prerequisites"                                                   │
│  ✓ "Build order" / "Task dependencies"                                      │
│  ✓ "Detect cycle in directed graph"                                         │
│                                                                             │
│  UNION-FIND (Disjoint Set):                                                 │
│  ✓ "Connected components" (dynamic)                                         │
│  ✓ "Redundant connection"                                                   │
│  ✓ "Accounts merge"                                                         │
│  ✓ "Minimum spanning tree"                                                  │
│                                                                             │
│  DIJKSTRA'S ALGORITHM:                                                      │
│  ✓ "Shortest path" (weighted, non-negative)                                 │
│  ✓ "Network delay time"                                                     │
│  ✓ "Cheapest flights"                                                       │
│                                                                             │
│  BELLMAN-FORD:                                                              │
│  ✓ "Shortest path" (with negative weights)                                  │
│  ✓ "Detect negative cycle"                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning advanced graphs, ensure you understand:
- [ ] Basic graph concepts (vertices, edges, directed/undirected)
- [ ] Graph representations (adjacency list, matrix)
- [ ] BFS and DFS traversal
- [ ] Basic shortest path (unweighted BFS)
- [ ] Heap/Priority Queue operations

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED GRAPHS MEMORY MAP                               │
│                                                                             │
│                    ┌────────────────────┐                                   │
│         ┌─────────│  ADVANCED GRAPHS   │─────────┐                          │
│         │         └────────────────────┘         │                          │
│         ▼                                        ▼                          │
│  ┌─────────────┐                          ┌─────────────┐                   │
│  │ SHORTEST    │                          │ STRUCTURE   │                   │
│  │   PATH      │                          │ ALGORITHMS  │                   │
│  └──────┬──────┘                          └──────┬──────┘                   │
│         │                                        │                          │
│    ┌────┴────┬────────┐                ┌────────┴────────┐                  │
│    ▼         ▼        ▼                ▼                 ▼                  │
│ ┌──────┐ ┌──────┐ ┌──────┐      ┌──────────┐     ┌───────────┐             │
│ │Dijks-│ │Bell- │ │Floyd-│      │Topological│    │Union-Find │             │
│ │ tra  │ │ man  │ │Warsh.│      │   Sort    │    │   (DSU)   │             │
│ └──────┘ └──────┘ └──────┘      └──────────┘     └───────────┘             │
│                                                                             │
│  Algorithm Selection Guide:                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Scenario                      │ Algorithm                          │    │
│  ├───────────────────────────────┼────────────────────────────────────┤    │
│  │ Unweighted shortest path      │ BFS                                │    │
│  │ Weighted, non-negative        │ Dijkstra                           │    │
│  │ Negative weights allowed      │ Bellman-Ford                       │    │
│  │ All pairs shortest path       │ Floyd-Warshall                     │    │
│  │ Task ordering/dependencies    │ Topological Sort                   │    │
│  │ Dynamic connectivity          │ Union-Find                         │    │
│  │ Minimum Spanning Tree         │ Kruskal (Union-Find) or Prim       │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SHORTEST PATH DECISION TREE                              │
│                                                                             │
│  Is the graph weighted?                                                     │
│       │                                                                     │
│       ├── NO → Use BFS (O(V+E))                                             │
│       │                                                                     │
│       └── YES → Are there negative weights?                                 │
│                    │                                                        │
│                    ├── NO → Single source?                                  │
│                    │            │                                           │
│                    │            ├── YES → Dijkstra O((V+E) log V)           │
│                    │            │                                           │
│                    │            └── NO → Floyd-Warshall O(V³)               │
│                    │                                                        │
│                    └── YES → Need to detect negative cycle?                 │
│                                 │                                           │
│                                 ├── YES → Bellman-Ford O(V×E)               │
│                                 │                                           │
│                                 └── NO → Bellman-Ford or SPFA               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                    GRAPH STRUCTURE DECISION TREE                            │
│                                                                             │
│  Need to process nodes in dependency order?                                 │
│       │                                                                     │
│       ├── YES → Is it a DAG (Directed Acyclic Graph)?                       │
│       │            │                                                        │
│       │            ├── YES → Topological Sort                               │
│       │            │                                                        │
│       │            └── NO (has cycle) → No valid ordering exists            │
│       │                                                                     │
│       └── NO → Need to track connected components dynamically?              │
│                    │                                                        │
│                    ├── YES → Union-Find (DSU)                               │
│                    │                                                        │
│                    └── NO → Need minimum spanning tree?                     │
│                                 │                                           │
│                                 ├── YES → Kruskal or Prim                   │
│                                 │                                           │
│                                 └── NO → Use basic BFS/DFS                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Topological Sort

Ordering of vertices in a DAG (Directed Acyclic Graph) such that for every edge u→v, u comes before v.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TOPOLOGICAL SORT VISUALIZATION                           │
│                                                                             │
│  Course prerequisites:                                                      │
│  0 → 1 → 3                                                                  │
│  0 → 2 → 3                                                                  │
│                                                                             │
│        0                                                                    │
│       ↙ ↘                                                                   │
│      1   2                                                                  │
│       ↘ ↙                                                                   │
│        3                                                                    │
│                                                                             │
│  Valid orderings: [0, 1, 2, 3] or [0, 2, 1, 3]                             │
│  (0 must come before 1, 2; both must come before 3)                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Kahn's Algorithm (BFS)

```python
from collections import deque, defaultdict

def topological_sort_bfs(num_nodes: int, edges: list[list[int]]) -> list[int]:
    """
    Topological sort using Kahn's algorithm (BFS).

    Strategy:
    1. Calculate in-degree for each node
    2. Start with nodes having in-degree 0
    3. Process node, reduce in-degree of neighbors
    4. Add neighbors with in-degree 0 to queue

    Time: O(V + E)
    Space: O(V + E)

    Returns: Topological order, or empty list if cycle exists
    """
    # Build adjacency list and calculate in-degrees
    graph = defaultdict(list)
    in_degree = [0] * num_nodes

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    # Start with nodes having in-degree 0
    queue = deque()
    for node in range(num_nodes):
        if in_degree[node] == 0:
            queue.append(node)

    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        # Reduce in-degree of neighbors
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1

            # If in-degree becomes 0, add to queue
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If all nodes processed, valid ordering exists
    if len(result) == num_nodes:
        return result
    else:
        return []  # Cycle detected


def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    """
    Course Schedule (LC #207) - Can all courses be finished?

    This is essentially checking if topological sort is possible.
    """
    # Convert to edges: [course, prereq] means prereq → course
    edges = [[prereq, course] for course, prereq in prerequisites]

    result = topological_sort_bfs(numCourses, edges)
    return len(result) == numCourses
```

### DFS Approach

```python
def topological_sort_dfs(num_nodes: int, edges: list[list[int]]) -> list[int]:
    """
    Topological sort using DFS.

    Strategy:
    - DFS from each unvisited node
    - Add node to result AFTER processing all descendants
    - Reverse the result at the end

    Time: O(V + E)
    Space: O(V + E)
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    # States: 0 = unvisited, 1 = in current path, 2 = processed
    state = [0] * num_nodes
    result = []
    has_cycle = False

    def dfs(node: int) -> bool:
        """Returns True if cycle detected."""
        nonlocal has_cycle

        if state[node] == 1:
            return True  # Back edge = cycle!
        if state[node] == 2:
            return False  # Already processed

        state[node] = 1  # Mark as in current path

        for neighbor in graph[node]:
            if dfs(neighbor):
                has_cycle = True
                return True

        state[node] = 2  # Mark as processed
        result.append(node)  # Add after all descendants
        return False

    # Process all nodes
    for node in range(num_nodes):
        if state[node] == 0:
            if dfs(node):
                return []  # Cycle detected

    # Reverse to get correct order
    return result[::-1]
```

---

## Union-Find (Disjoint Set Union)

Data structure to track connected components with efficient union and find operations.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNION-FIND VISUALIZATION                                 │
│                                                                             │
│  Initial: Each element is its own set                                       │
│  {0} {1} {2} {3} {4}                                                        │
│                                                                             │
│  Union(0, 1):                                                               │
│  {0, 1} {2} {3} {4}                                                         │
│                                                                             │
│  Union(2, 3):                                                               │
│  {0, 1} {2, 3} {4}                                                          │
│                                                                             │
│  Union(0, 2):                                                               │
│  {0, 1, 2, 3} {4}                                                           │
│                                                                             │
│  Tree representation with path compression:                                 │
│       0                                                                     │
│      /|\                                                                    │
│     1 2 3                                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
class UnionFind:
    """
    Union-Find with path compression and union by rank.

    Time: O(α(n)) per operation, where α is inverse Ackermann (nearly O(1))
    Space: O(n)
    """

    def __init__(self, n: int):
        """
        Initialize n elements, each in its own set.

        parent[i] = parent of element i (initially itself)
        rank[i] = approximate depth of tree rooted at i
        """
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # Number of connected components

    def find(self, x: int) -> int:
        """
        Find root of element x with path compression.

        Path compression: Make all nodes on path point directly to root.
        This flattens the tree for faster future queries.
        """
        if self.parent[x] != x:
            # Recursively find root and update parent
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Union sets containing x and y.

        Union by rank: Attach smaller tree under larger tree.
        This keeps trees balanced.

        Returns: True if union happened (were in different sets)
        """
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # Already in same set

        # Union by rank: attach smaller tree under larger
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            # Same rank: choose one as root, increment its rank
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        self.count -= 1  # One less component
        return True

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)

    def get_count(self) -> int:
        """Return number of connected components."""
        return self.count
```

### Union-Find Applications

```python
def findRedundantConnection(edges: list[list[int]]) -> list[int]:
    """
    Find edge that creates a cycle (LC #684).

    Strategy:
    - Process edges one by one
    - Use Union-Find to check if edge connects already-connected nodes
    - First such edge is redundant

    Time: O(n * α(n)) ≈ O(n)
    Space: O(n)
    """
    n = len(edges)
    uf = UnionFind(n + 1)  # 1-indexed nodes

    for u, v in edges:
        if not uf.union(u, v):
            # u and v already connected - this edge is redundant
            return [u, v]

    return []


def countComponents(n: int, edges: list[list[int]]) -> int:
    """
    Count connected components (LC #323).

    Time: O(n + e * α(n))
    Space: O(n)
    """
    uf = UnionFind(n)

    for u, v in edges:
        uf.union(u, v)

    return uf.get_count()
```

---

## Dijkstra's Algorithm

Shortest path from source to all vertices in weighted graph (non-negative weights).

```python
import heapq
from collections import defaultdict

def dijkstra(n: int, edges: list[list[int]], source: int) -> list[int]:
    """
    Dijkstra's algorithm for shortest paths.

    Strategy:
    - Use min-heap to always process closest unvisited node
    - Relax edges: if shorter path found, update distance

    Time: O((V + E) log V)
    Space: O(V + E)

    Returns: List of shortest distances from source to each node
    """
    # Build adjacency list: node -> [(neighbor, weight), ...]
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        # For undirected: graph[v].append((u, w))

    # Distance array: infinity initially
    dist = [float('inf')] * n
    dist[source] = 0

    # Min-heap: (distance, node)
    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)

        # Skip if we've found a shorter path already
        if d > dist[u]:
            continue

        # Relax all edges from u
        for v, weight in graph[u]:
            new_dist = dist[u] + weight

            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return dist


def networkDelayTime(times: list[list[int]], n: int, k: int) -> int:
    """
    Network Delay Time (LC #743).

    Find time for signal to reach all nodes from node k.
    """
    dist = dijkstra(n, times, k - 1)  # Convert to 0-indexed

    max_time = max(dist)
    return max_time if max_time != float('inf') else -1
```

---

## Bellman-Ford Algorithm

Shortest path with negative weights (can detect negative cycles).

```python
def bellman_ford(n: int, edges: list[list[int]], source: int) -> list[int]:
    """
    Bellman-Ford algorithm for shortest paths.

    Handles negative weights, detects negative cycles.

    Strategy:
    - Relax all edges V-1 times
    - If any edge can still be relaxed, negative cycle exists

    Time: O(V * E)
    Space: O(V)

    Returns: List of shortest distances, or None if negative cycle
    """
    dist = [float('inf')] * n
    dist[source] = 0

    # Relax all edges V-1 times
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w

    # Check for negative cycle
    for u, v, w in edges:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None  # Negative cycle detected

    return dist


def findCheapestPrice(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
    """
    Cheapest Flights Within K Stops (LC #787).

    Modified Bellman-Ford with limited relaxations.
    """
    dist = [float('inf')] * n
    dist[src] = 0

    # At most k+1 relaxations (k stops = k+1 edges)
    for _ in range(k + 1):
        # Use copy to avoid using updated values in same iteration
        temp = dist[:]

        for u, v, price in flights:
            if dist[u] != float('inf'):
                temp[v] = min(temp[v], dist[u] + price)

        dist = temp

    return dist[dst] if dist[dst] != float('inf') else -1
```

---

## Algorithm Comparison

| Algorithm | Time | Space | Use Case |
|-----------|------|-------|----------|
| BFS | O(V + E) | O(V) | Unweighted shortest path |
| Dijkstra | O((V+E) log V) | O(V) | Weighted, non-negative |
| Bellman-Ford | O(V × E) | O(V) | Negative weights |
| Floyd-Warshall | O(V³) | O(V²) | All pairs shortest path |
| Topological Sort | O(V + E) | O(V) | DAG ordering |
| Union-Find | O(α(n)) per op | O(V) | Dynamic connectivity |

---

## Common Mistakes

```python
# ❌ WRONG: Using Dijkstra with negative weights
# Dijkstra assumes shortest path found is final - negative edges break this

# ✅ CORRECT: Use Bellman-Ford for negative weights


# ❌ WRONG: Not handling disconnected nodes in Dijkstra
dist = dijkstra(...)
return max(dist)  # May return infinity!

# ✅ CORRECT: Check for unreachable nodes
max_dist = max(dist)
return max_dist if max_dist != float('inf') else -1


# ❌ WRONG: Union-Find without path compression
def find(x):
    while parent[x] != x:
        x = parent[x]
    return x  # O(n) per operation!

# ✅ CORRECT: With path compression - O(α(n)) ≈ O(1)
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])  # Path compression
    return parent[x]
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"For this shortest path problem with weighted edges, I'll use Dijkstra's
algorithm. It works by greedily selecting the node with minimum distance
and relaxing its edges. Since all weights are non-negative, once we
process a node, we've found its shortest path."
```

### 2. What Interviewers Look For
- **Algorithm selection**: Know when to use each algorithm
- **Complexity analysis**: Time and space for each approach
- **Edge cases**: Disconnected graphs, negative cycles, empty input
- **Implementation details**: Heap usage, path compression

### 3. Common Follow-up Questions
- "What if there are negative weights?" → Use Bellman-Ford
- "Can you detect a cycle?" → Topological sort fails, or use DFS coloring
- "How would you reconstruct the path?" → Track parent pointers
- "What if we need all-pairs shortest path?" → Floyd-Warshall

### 4. Key Insights to Mention
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  KEY INSIGHTS FOR INTERVIEWS                                                │
│                                                                             │
│  Dijkstra:                                                                  │
│  • "Greedy works because non-negative weights mean we can't find a          │
│     shorter path later through a longer intermediate path"                  │
│                                                                             │
│  Bellman-Ford:                                                              │
│  • "We relax V-1 times because shortest path has at most V-1 edges"         │
│  • "If we can still relax after V-1 iterations, negative cycle exists"     │
│                                                                             │
│  Topological Sort:                                                          │
│  • "We process nodes with no dependencies first (in-degree 0)"              │
│  • "If we can't process all nodes, there's a cycle"                         │
│                                                                             │
│  Union-Find:                                                                │
│  • "Path compression + union by rank gives near-constant time"              │
│  • "α(n) is the inverse Ackermann function, practically ≤ 4"                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Next: Practice Problems

Continue to:
- [02-shortest-path.md](./02-shortest-path.md) - Dijkstra & Bellman-Ford problems
- [03-union-find.md](./03-union-find.md) - Union-Find applications
- [04-topological-sort.md](./04-topological-sort.md) - Topological sort problems
