# MST & Shortest Path - Deep Dive

## Minimum Spanning Tree (MST)

A spanning tree connects all vertices with minimum total edge weight.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MST ALGORITHMS COMPARISON                                │
│                                                                             │
│  PRIM'S ALGORITHM:                                                          │
│  • Start from any vertex                                                    │
│  • Greedily add closest vertex to current tree                              │
│  • Best for: Dense graphs                                                   │
│  • Time: O(E log V) with heap, O(V²) with array                            │
│                                                                             │
│  KRUSKAL'S ALGORITHM:                                                       │
│  • Sort all edges by weight                                                 │
│  • Greedily add edges that don't create cycle                               │
│  • Best for: Sparse graphs                                                  │
│  • Time: O(E log E)                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prim's Algorithm

### Intuition Development
```
PRIM'S STEP-BY-STEP:

Graph:
    A ---1--- B
    |         |
    4         2
    |         |
    C ---3--- D

Start from A:
  Visited: {A}
  Heap: [(1,B), (4,C)]

Pop (1,B):
  Visited: {A, B}
  Total: 1
  Heap: [(2,D), (4,C)]  ← add B's edges

Pop (2,D):
  Visited: {A, B, D}
  Total: 1 + 2 = 3
  Heap: [(3,C), (4,C)]

Pop (3,C):
  Visited: {A, B, D, C}
  Total: 3 + 3 = 6

MST edges: A-B, B-D, D-C (weight = 6)
```

```python
import heapq
from collections import defaultdict

def prim_mst(n: int, edges: list[list[int]]) -> int:
    """
    Find MST weight using Prim's algorithm.

    Strategy:
    - Start from vertex 0
    - Use min-heap to always add closest unvisited vertex
    - Track visited vertices to avoid cycles

    Args:
        n: Number of vertices
        edges: List of [u, v, weight]

    Returns:
        Total weight of MST, or -1 if not connected
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    # Min-heap: (weight, vertex)
    heap = [(0, 0)]  # Start from vertex 0 with cost 0
    visited = set()
    total_weight = 0

    while heap and len(visited) < n:
        weight, u = heapq.heappop(heap)

        if u in visited:
            continue

        visited.add(u)
        total_weight += weight

        # Add edges to unvisited neighbors
        for v, w in graph[u]:
            if v not in visited:
                heapq.heappush(heap, (w, v))

    # Check if all vertices connected
    return total_weight if len(visited) == n else -1
```

### Complexity
- **Time**: O(E log V)
- **Space**: O(V + E)

---

## Kruskal's Algorithm

### Intuition Development
```
KRUSKAL'S STEP-BY-STEP:

Edges sorted by weight:
  (A,B,1), (B,D,2), (C,D,3), (A,C,4)

Process (A,B,1):
  A and B in different components → INCLUDE
  Union(A, B)
  Total: 1, Edges: 1

Process (B,D,2):
  B and D in different components → INCLUDE
  Union(B, D)
  Total: 3, Edges: 2

Process (C,D,3):
  C and D in different components → INCLUDE
  Union(C, D)
  Total: 6, Edges: 3 = n-1 ← DONE!

Skip (A,C,4): A and C now in same component
```

```python
class UnionFind:
    """Union-Find for Kruskal's algorithm."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def kruskal_mst(n: int, edges: list[list[int]]) -> int:
    """
    Find MST weight using Kruskal's algorithm.

    Strategy:
    - Sort edges by weight
    - Add edges that don't create cycle (using Union-Find)
    - Stop when n-1 edges added

    Returns:
        Total weight of MST, or -1 if not connected
    """
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])

    uf = UnionFind(n)
    total_weight = 0
    edges_used = 0

    for u, v, w in edges:
        if uf.union(u, v):
            total_weight += w
            edges_used += 1

            if edges_used == n - 1:
                break

    return total_weight if edges_used == n - 1 else -1
```

### Complexity
- **Time**: O(E log E)
- **Space**: O(V)

---

## Problem 1: Min Cost to Connect All Points (LC #1584) - Medium

- [LeetCode](https://leetcode.com/problems/min-cost-to-connect-all-points/)

### Problem Statement
Connect all points with minimum total Manhattan distance.

### Examples
```
Input: points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
Output: 20

Visualization:
     (3,10)
       |
       |
  (0,0)---(2,2)---(5,2)---(7,0)
```

### Intuition Development
```
MANHATTAN DISTANCE:
dist(p1, p2) = |x1-x2| + |y1-y2|

This is a complete graph (every point connects to every other).
Use Prim's or Kruskal's to find MST.

For 5 points: C(5,2) = 10 edges possible
MST will have exactly 4 edges (n-1)
```

### Video Explanation
- [NeetCode - Min Cost to Connect All Points](https://www.youtube.com/watch?v=f7JOBJIC-NA)

### Solution
```python
def minCostConnectPoints(points: list[list[int]]) -> int:
    """
    Find MST of points using Prim's algorithm.
    """
    n = len(points)
    if n <= 1:
        return 0

    # Manhattan distance
    def dist(i, j):
        return abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])

    # Prim's algorithm
    visited = set()
    heap = [(0, 0)]  # (distance, point_index)
    total = 0

    while len(visited) < n:
        d, u = heapq.heappop(heap)

        if u in visited:
            continue

        visited.add(u)
        total += d

        # Add edges to unvisited points
        for v in range(n):
            if v not in visited:
                heapq.heappush(heap, (dist(u, v), v))

    return total
```

### Complexity
- **Time**: O(n² log n)
- **Space**: O(n²)

### Edge Cases
- Single point → return 0
- Two points → return Manhattan distance between them
- Collinear points
- Points with same coordinates

---

## Shortest Path Algorithms

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SHORTEST PATH COMPARISON                                 │
│                                                                             │
│  BFS (Unweighted):                                                          │
│  • Time: O(V + E)                                                           │
│  • Use: Unweighted graphs                                                   │
│                                                                             │
│  DIJKSTRA:                                                                  │
│  • Time: O((V + E) log V)                                                   │
│  • Use: Non-negative weights                                                │
│  • Cannot handle negative weights                                           │
│                                                                             │
│  BELLMAN-FORD:                                                              │
│  • Time: O(V × E)                                                           │
│  • Use: Negative weights, detect negative cycles                            │
│                                                                             │
│  FLOYD-WARSHALL:                                                            │
│  • Time: O(V³)                                                              │
│  • Use: All pairs shortest path                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dijkstra's Algorithm

### Intuition Development
```
DIJKSTRA STEP-BY-STEP:

    A --1-- B
    |       |
    4       2
    |       |
    C --1-- D

Source: A
Initial dist: {A:0, B:∞, C:∞, D:∞}
Heap: [(0, A)]

Pop (0, A):
  Relax A→B: dist[B] = min(∞, 0+1) = 1
  Relax A→C: dist[C] = min(∞, 0+4) = 4
  Heap: [(1, B), (4, C)]

Pop (1, B):
  Relax B→D: dist[D] = min(∞, 1+2) = 3
  Heap: [(3, D), (4, C)]

Pop (3, D):
  Relax D→C: dist[C] = min(4, 3+1) = 4 (no change)

Pop (4, C):
  Already optimal

Final: {A:0, B:1, C:4, D:3}
```

```python
def dijkstra(n: int, edges: list[list[int]], source: int) -> list[int]:
    """
    Find shortest paths from source using Dijkstra's algorithm.

    Strategy:
    - Use min-heap to process closest unvisited vertex
    - Relax edges: update distance if shorter path found

    Returns:
        List of shortest distances from source to each vertex
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, w in edges:
        graph[u].append((v, w))

    # Initialize distances
    dist = [float('inf')] * n
    dist[source] = 0

    # Min-heap: (distance, vertex)
    heap = [(0, source)]

    while heap:
        d, u = heapq.heappop(heap)

        # Skip if we've found shorter path
        if d > dist[u]:
            continue

        # Relax edges
        for v, w in graph[u]:
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    return dist
```

### Complexity
- **Time**: O((V + E) log V)
- **Space**: O(V + E)

---

## Bellman-Ford Algorithm

### Intuition Development
```
BELLMAN-FORD:
Relaxes ALL edges V-1 times.
Can detect negative cycles with one more pass.

Why V-1 iterations?
Shortest path has at most V-1 edges.
Each iteration guarantees at least one more correct distance.
```

```python
def bellman_ford(n: int, edges: list[list[int]], source: int) -> list[int]:
    """
    Find shortest paths using Bellman-Ford (handles negative weights).

    Strategy:
    - Relax all edges V-1 times
    - If any edge can still be relaxed, negative cycle exists

    Returns:
        List of shortest distances, or None if negative cycle
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
            return None  # Negative cycle exists

    return dist
```

### Complexity
- **Time**: O(V × E)
- **Space**: O(V)

---

## Problem 2: Network Delay Time (LC #743) - Medium

- [LeetCode](https://leetcode.com/problems/network-delay-time/)

### Problem Statement
Time for signal to reach all nodes from source.

### Examples
```
Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2

Graph:
    1 ← 2 → 3 → 4

From node 2:
  To 1: time = 1
  To 3: time = 1
  To 4: time = 1 + 1 = 2

Max time = 2
```

### Intuition Development
```
SIGNAL PROPAGATION:
Find shortest path from k to ALL nodes.
Answer = maximum of all shortest paths.

If any node unreachable → return -1
```

### Video Explanation
- [NeetCode - Network Delay Time](https://www.youtube.com/watch?v=EaphyqKU4PQ)

### Solution
```python
def networkDelayTime(times: list[list[int]], n: int, k: int) -> int:
    """
    Find time for signal to reach all nodes using Dijkstra.
    """
    # Build graph
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    # Dijkstra from source k
    dist = {k: 0}
    heap = [(0, k)]

    while heap:
        d, u = heapq.heappop(heap)

        if d > dist.get(u, float('inf')):
            continue

        for v, w in graph[u]:
            new_dist = d + w
            if new_dist < dist.get(v, float('inf')):
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    # Check if all nodes reachable
    if len(dist) != n:
        return -1

    return max(dist.values())
```

### Complexity
- **Time**: O((V + E) log V)
- **Space**: O(V + E)

### Edge Cases
- Single node → return 0
- Disconnected graph → return -1
- Self-loop (ignored for delay calculation)
- Multiple edges between same nodes

---

## Problem 3: Cheapest Flights Within K Stops (LC #787) - Medium

- [LeetCode](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

### Problem Statement
Cheapest price from src to dst with at most k stops.

### Examples
```
Input: n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]],
       src = 0, dst = 3, k = 1

Graph:
    0 → 1 → 3  (cost: 100 + 600 = 700, stops: 1) ✓
    0 → 1 → 2 → 3  (cost: 400, stops: 2) ✗ too many stops

Output: 700
```

### Intuition Development
```
MODIFIED BELLMAN-FORD:
Standard Dijkstra doesn't work because we need to track stops.

Key insight: Use k+1 relaxation rounds (k stops = k+1 edges)

Round 0: dist = [0, ∞, ∞, ∞]
Round 1 (1 edge): dist = [0, 100, ∞, ∞]
Round 2 (2 edges): dist = [0, 100, 200, 700]

Answer: dist[3] = 700
```

### Video Explanation
- [NeetCode - Cheapest Flights Within K Stops](https://www.youtube.com/watch?v=5eIK3zUdYmE)

### Solution
```python
def findCheapestPrice(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
    """
    Find cheapest flight with limited stops using modified Bellman-Ford.

    Strategy:
    - Bellman-Ford with k+1 iterations (k stops = k+1 edges)
    - Use copy of distances to avoid using updated values
    """
    dist = [float('inf')] * n
    dist[src] = 0

    # k+1 relaxations for k stops
    for _ in range(k + 1):
        temp = dist[:]

        for u, v, price in flights:
            if dist[u] != float('inf'):
                temp[v] = min(temp[v], dist[u] + price)

        dist = temp

    return dist[dst] if dist[dst] != float('inf') else -1
```

### Complexity
- **Time**: O(k × E)
- **Space**: O(V)

### Edge Cases
- k = 0 → direct flight only
- No path within k stops → -1
- Multiple paths with same cost
- src = dst → return 0

---

## Problem 4: Path with Maximum Probability (LC #1514) - Medium

- [LeetCode](https://leetcode.com/problems/path-with-maximum-probability/)

### Problem Statement
Find path with maximum probability from start to end.

### Examples
```
Input: n = 3, edges = [[0,1],[1,2],[0,2]],
       succProb = [0.5,0.5,0.2], start = 0, end = 2

Graph:
    0 --0.5-- 1 --0.5-- 2
     \                 /
      ----0.2---------

Path 0→2: prob = 0.2
Path 0→1→2: prob = 0.5 × 0.5 = 0.25

Output: 0.25
```

### Intuition Development
```
MODIFIED DIJKSTRA:
Instead of minimizing sum, MAXIMIZE product.

Use max-heap (negate probabilities).
Multiply instead of add.

Heap: [(-1.0, 0)]
Pop (-1.0, 0): prob[0] = 1.0
  → 1: new_prob = 1.0 × 0.5 = 0.5
  → 2: new_prob = 1.0 × 0.2 = 0.2

Pop (-0.5, 1): prob[1] = 0.5
  → 2: new_prob = 0.5 × 0.5 = 0.25 > 0.2 ✓

Answer: 0.25
```

### Video Explanation
- [NeetCode - Path with Maximum Probability](https://www.youtube.com/watch?v=kPsDTGcrzGM)

### Solution
```python
def maxProbability(n: int, edges: list[list[int]], succProb: list[float],
                   start: int, end: int) -> float:
    """
    Find maximum probability path using modified Dijkstra.

    Strategy:
    - Use max-heap (negate probabilities)
    - Multiply probabilities instead of adding
    """
    # Build graph
    graph = defaultdict(list)
    for i, (u, v) in enumerate(edges):
        graph[u].append((v, succProb[i]))
        graph[v].append((u, succProb[i]))

    # Max-heap: (-probability, node)
    prob = [0.0] * n
    prob[start] = 1.0
    heap = [(-1.0, start)]

    while heap:
        neg_p, u = heapq.heappop(heap)
        p = -neg_p

        if u == end:
            return p

        if p < prob[u]:
            continue

        for v, edge_prob in graph[u]:
            new_prob = p * edge_prob
            if new_prob > prob[v]:
                prob[v] = new_prob
                heapq.heappush(heap, (-new_prob, v))

    return 0.0
```

### Complexity
- **Time**: O((V + E) log V)
- **Space**: O(V + E)

### Edge Cases
- start = end → return 1.0
- No path exists → return 0.0
- All probabilities = 1.0
- Very small probabilities (floating point precision)

---

## Problem 5: Swim in Rising Water (LC #778) - Hard

- [LeetCode](https://leetcode.com/problems/swim-in-rising-water/)

### Problem Statement
Find minimum time to swim from (0,0) to (n-1,n-1).

### Examples
```
Input: grid = [[0,2],[1,3]]

At time 0: can only be at (0,0)
At time 1: water at (1,0)=1, can move to (1,0)
At time 2: water at (0,1)=2, can move to (0,1)
At time 3: water at (1,1)=3, can reach destination

Output: 3
```

### Intuition Development
```
KEY INSIGHT:
Time to reach cell = max(time to reach previous, cell elevation)

This is like finding a path where we minimize the MAXIMUM edge weight.

MODIFIED DIJKSTRA:
Instead of summing distances, take maximum.

Grid:
  [0, 2]
  [1, 3]

Heap: [(0, 0, 0)]
Pop (0, 0, 0):
  → (0,1): time = max(0, 2) = 2
  → (1,0): time = max(0, 1) = 1
  Heap: [(1, 1, 0), (2, 0, 1)]

Pop (1, 1, 0):
  → (1,1): time = max(1, 3) = 3
  Heap: [(2, 0, 1), (3, 1, 1)]

Pop (2, 0, 1):
  → (1,1): time = max(2, 3) = 3 (already have 3)

Pop (3, 1, 1): DESTINATION! Return 3
```

### Video Explanation
- [NeetCode - Swim in Rising Water](https://www.youtube.com/watch?v=amvrKlMLuGY)

### Solution
```python
def swimInWater(grid: list[list[int]]) -> int:
    """
    Find minimum time to swim across grid.

    Strategy (Modified Dijkstra):
    - Time to reach cell = max(time to reach previous, cell elevation)
    - Use min-heap to process cells by minimum time
    """
    n = len(grid)

    # Min-heap: (time, row, col)
    heap = [(grid[0][0], 0, 0)]
    visited = [[False] * n for _ in range(n)]

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while heap:
        time, r, c = heapq.heappop(heap)

        if r == n - 1 and c == n - 1:
            return time

        if visited[r][c]:
            continue
        visited[r][c] = True

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < n and 0 <= nc < n and not visited[nr][nc]:
                # Time = max of current time and cell elevation
                new_time = max(time, grid[nr][nc])
                heapq.heappush(heap, (new_time, nr, nc))

    return -1
```

### Complexity
- **Time**: O(n² log n)
- **Space**: O(n²)

### Edge Cases
- 1×1 grid → return grid[0][0]
- All same elevation → return that elevation
- Destination has highest elevation
- Starting point has highest elevation

---

## Summary: MST & Shortest Path

| Algorithm | Time | Space | Use Case |
|-----------|------|-------|----------|
| Prim's | O(E log V) | O(V) | Dense graphs MST |
| Kruskal's | O(E log E) | O(V) | Sparse graphs MST |
| BFS | O(V + E) | O(V) | Unweighted |
| Dijkstra | O((V+E) log V) | O(V) | Non-negative weights |
| Bellman-Ford | O(V × E) | O(V) | Negative weights |
| Floyd-Warshall | O(V³) | O(V²) | All pairs |

---

## Practice More Problems

- [ ] LC #1135 - Connecting Cities With Minimum Cost
- [ ] LC #1168 - Optimize Water Distribution in a Village
- [ ] LC #882 - Reachable Nodes In Subdivided Graph
- [ ] LC #1334 - Find the City With the Smallest Number of Neighbors
