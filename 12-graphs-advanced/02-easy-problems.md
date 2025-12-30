# Advanced Graphs - Practice Problems

## Problem 1: Course Schedule (LC #207) - Medium

- [LeetCode](https://leetcode.com/problems/course-schedule/)

### Problem Statement
Determine if you can finish all courses given prerequisites.

### Examples
```
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true (take course 0 then 1)

Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false (circular dependency)
```

### Video Explanation
- [NeetCode - Course Schedule](https://www.youtube.com/watch?v=EgI5nU9etnU)

### Intuition
```
The key insight: This is a CYCLE DETECTION problem!

If there's a cycle in the dependency graph, we can't complete all courses.

Visual: Prerequisites form a directed graph

       [1,0] means: 0 → 1 (take 0 before 1)

       Case 1: No cycle - CAN finish
       ┌───┐    ┌───┐
       │ 0 │───→│ 1 │
       └───┘    └───┘

       Case 2: Cycle - CANNOT finish
       ┌───┐    ┌───┐
       │ 0 │───→│ 1 │
       └───┘←───└───┘
         ↑_______↓

Algorithm: Topological Sort (Kahn's BFS)
1. Count in-degrees (how many prereqs each course has)
2. Start with courses having 0 prereqs
3. "Take" a course, reduce in-degree of dependent courses
4. If we take all courses, no cycle exists!
```

### Solution
```python
from collections import deque, defaultdict

def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    """
    Check if all courses can be finished (no cycle in DAG).

    Strategy: Topological sort using Kahn's algorithm (BFS).
    If we can process all nodes, no cycle exists.

    Time: O(V + E)
    Space: O(V + E)
    """
    # Build adjacency list and in-degree count
    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # Start with courses having no prerequisites
    queue = deque()
    for i in range(numCourses):
        if in_degree[i] == 0:
            queue.append(i)

    # Process courses
    courses_taken = 0

    while queue:
        course = queue.popleft()
        courses_taken += 1

        # Reduce in-degree for dependent courses
        for next_course in graph[course]:
            in_degree[next_course] -= 1

            if in_degree[next_course] == 0:
                queue.append(next_course)

    # If all courses processed, no cycle
    return courses_taken == numCourses
```

### Edge Cases
- No prerequisites → return True
- Self-dependency → return False
- Single course → return True
- All courses independent → return True
- Circular dependency chain → return False

---

## Problem 2: Course Schedule II (LC #210) - Medium

- [LeetCode](https://leetcode.com/problems/course-schedule-ii/)

### Problem Statement
Return the ordering of courses to finish all courses.

### Video Explanation
- [NeetCode - Course Schedule II](https://www.youtube.com/watch?v=Akt3glAwyfY)

### Intuition
```
Same as Course Schedule I, but now we need the ORDER!

Just record the order in which we "take" courses.

Topological Sort gives us a valid ordering:
- Courses with no prereqs come first
- Dependent courses come after their prerequisites

Example: 4 courses, prereqs = [[1,0],[2,0],[3,1],[3,2]]

       0 → 1 → 3
       ↓   ↗
       2 ──┘

Valid orders: [0,1,2,3] or [0,2,1,3]
```

### Solution
```python
def findOrder(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    """
    Find valid course ordering (topological sort).

    Same as Course Schedule I, but return the order.

    Time: O(V + E)
    Space: O(V + E)
    """
    # Build graph and in-degrees
    graph = defaultdict(list)
    in_degree = [0] * numCourses

    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1

    # Start with courses having no prerequisites
    queue = deque()
    for i in range(numCourses):
        if in_degree[i] == 0:
            queue.append(i)

    # Process and record order
    order = []

    while queue:
        course = queue.popleft()
        order.append(course)

        for next_course in graph[course]:
            in_degree[next_course] -= 1

            if in_degree[next_course] == 0:
                queue.append(next_course)

    # Return order if all courses can be taken
    return order if len(order) == numCourses else []
```

### Edge Cases
- No prerequisites → return [0, 1, ..., n-1]
- Cycle → return []
- Multiple valid orders → return any valid one
- Single course → return [0]
- Linear dependency chain → return in order

---

## Problem 3: Network Delay Time (LC #743) - Medium

- [LeetCode](https://leetcode.com/problems/network-delay-time/)

### Problem Statement
Find time for signal to reach all nodes from source k.

### Examples
```
Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2
```

### Video Explanation
- [NeetCode - Network Delay Time](https://www.youtube.com/watch?v=EaphyqKU4PQ)

### Intuition
```
This is DIJKSTRA'S ALGORITHM - shortest path from single source!

Visual: Weighted directed graph

        times = [[2,1,1],[2,3,1],[3,4,1]], k = 2

             1
        2 ──────→ 1
        │
        │ 1
        ↓
        3 ──────→ 4
             1

        From node 2:
        - Reach 1 in time 1
        - Reach 3 in time 1
        - Reach 4 in time 1+1 = 2

        Max time = 2 (time for signal to reach ALL nodes)

Dijkstra's Algorithm:
1. Use min-heap to always process closest unvisited node
2. Update distances to neighbors
3. Track max distance to any node
```

### Solution
```python
import heapq
from collections import defaultdict

def networkDelayTime(times: list[list[int]], n: int, k: int) -> int:
    """
    Find time for signal to reach all nodes using Dijkstra's algorithm.

    Strategy:
    - Build weighted graph
    - Run Dijkstra from source k
    - Return max distance (time for last node to receive signal)

    Time: O((V + E) log V)
    Space: O(V + E)
    """
    # Build adjacency list: node -> [(neighbor, weight), ...]
    graph = defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    # Dijkstra's algorithm
    dist = {k: 0}  # node -> shortest distance from k
    heap = [(0, k)]  # (distance, node)

    while heap:
        d, u = heapq.heappop(heap)

        # Skip if we've found shorter path
        if d > dist.get(u, float('inf')):
            continue

        # Relax edges
        for v, weight in graph[u]:
            new_dist = d + weight

            if new_dist < dist.get(v, float('inf')):
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    # Check if all nodes reachable
    if len(dist) != n:
        return -1

    return max(dist.values())
```

### Edge Cases
- Source can't reach all nodes → return -1
- Single node → return 0
- No edges → return -1 if n > 1
- Multiple paths to same node → take shortest
- Negative weights → not applicable (all positive)

---

## Problem 4: Cheapest Flights Within K Stops (LC #787) - Medium

- [LeetCode](https://leetcode.com/problems/cheapest-flights-within-k-stops/)

### Problem Statement
Find cheapest price from src to dst with at most k stops.

### Examples
```
Input: n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]]
       src = 0, dst = 3, k = 1
Output: 700 (0 -> 1 -> 3)
```

### Video Explanation
- [NeetCode - Cheapest Flights](https://www.youtube.com/watch?v=5eIK3zUdYmE)

### Intuition
```
This is BELLMAN-FORD with a twist: limited number of edges!

Why not Dijkstra? Because we have a CONSTRAINT (k stops).
Dijkstra finds shortest path but can't limit edges used.

Bellman-Ford naturally works in "rounds":
- Round 1: Paths with 1 edge
- Round 2: Paths with 2 edges
- ...
- Round k+1: Paths with k+1 edges (k stops)

Visual:
        0 ──100──→ 1 ──600──→ 3
        ↑         │
       100        100
        │         ↓
        └─────── 2 ──200──→ 3

k=1 means at most 1 stop (2 edges):
- 0→1→3 = 700 ✓ (1 stop)
- 0→1→2→3 = 400 ✗ (2 stops, exceeds k)

Answer: 700
```

### Solution
```python
def findCheapestPrice(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
    """
    Find cheapest flight with at most k stops using modified Bellman-Ford.

    Strategy:
    - Bellman-Ford with limited iterations (k+1 edges = k stops)
    - Use copy of distances to avoid using updated values in same iteration

    Time: O(k * E)
    Space: O(V)
    """
    # Initialize distances
    dist = [float('inf')] * n
    dist[src] = 0

    # Relax edges k+1 times (k stops = k+1 edges)
    for _ in range(k + 1):
        # Use copy to avoid using updated values in same iteration
        temp = dist[:]

        for u, v, price in flights:
            if dist[u] != float('inf'):
                temp[v] = min(temp[v], dist[u] + price)

        dist = temp

    return dist[dst] if dist[dst] != float('inf') else -1


def findCheapestPrice_bfs(n: int, flights: list[list[int]], src: int, dst: int, k: int) -> int:
    """
    Alternative: BFS with level tracking.

    Time: O(k * E)
    Space: O(V)
    """
    from collections import deque, defaultdict

    # Build graph
    graph = defaultdict(list)
    for u, v, price in flights:
        graph[u].append((v, price))

    # BFS: (node, cost, stops)
    dist = [float('inf')] * n
    dist[src] = 0
    queue = deque([(src, 0, 0)])  # (node, cost, stops)

    while queue:
        node, cost, stops = queue.popleft()

        if stops > k:
            continue

        for neighbor, price in graph[node]:
            new_cost = cost + price

            if new_cost < dist[neighbor]:
                dist[neighbor] = new_cost
                queue.append((neighbor, new_cost, stops + 1))

    return dist[dst] if dist[dst] != float('inf') else -1
```

### Edge Cases
- k = 0 → direct flight only
- No path within k stops → return -1
- src == dst → return 0
- Cheaper path with more stops → might not be valid
- Multiple paths with same cost → return that cost

---

## Problem 5: Redundant Connection (LC #684) - Medium

- [LeetCode](https://leetcode.com/problems/redundant-connection/)

### Problem Statement
Find the edge that creates a cycle in an undirected graph.

### Examples
```
Input: edges = [[1,2],[1,3],[2,3]]
Output: [2,3]
```

### Video Explanation
- [NeetCode - Redundant Connection](https://www.youtube.com/watch?v=FXWRE67PLL0)

### Intuition
```
This is UNION-FIND for cycle detection!

A tree with n nodes has exactly n-1 edges.
We're given n edges → one edge is redundant (creates cycle).

Union-Find approach:
- Process edges one by one
- If two nodes are ALREADY connected, this edge creates a cycle!

Visual:
        edges = [[1,2],[1,3],[2,3]]

        Step 1: [1,2] → Union 1 and 2
        1 ─── 2

        Step 2: [1,3] → Union 1 and 3
        1 ─── 2
        │
        3

        Step 3: [2,3] → 2 and 3 already connected!
        1 ─── 2
        │     │  ← This edge creates cycle!
        3 ────┘

        Answer: [2,3]
```

### Solution
```python
class UnionFind:
    """Union-Find with path compression and union by rank."""

    def __init__(self, n: int):
        self.parent = list(range(n + 1))
        self.rank = [0] * (n + 1)

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """
        Union two sets.

        Returns False if already in same set (edge creates cycle).
        """
        root_x, root_y = self.find(x), self.find(y)

        if root_x == root_y:
            return False  # Already connected - cycle!

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True


def findRedundantConnection(edges: list[list[int]]) -> list[int]:
    """
    Find edge that creates cycle using Union-Find.

    Strategy:
    - Process edges one by one
    - If edge connects already-connected nodes, it's redundant

    Time: O(n * α(n)) ≈ O(n)
    Space: O(n)
    """
    n = len(edges)
    uf = UnionFind(n)

    for u, v in edges:
        if not uf.union(u, v):
            # u and v already connected - this edge is redundant
            return [u, v]

    return []
```

### Edge Cases
- Only one redundant edge (guaranteed by problem)
- Multiple edges could be redundant → return last one
- Self-loop → would be detected as redundant
- Edge appears multiple times → first creates connection, later redundant
- Tree with extra edge → exactly one cycle

---

## Problem 6: Number of Connected Components (LC #323) - Medium

- [LeetCode](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

### Problem Statement
Count connected components in undirected graph.

### Video Explanation
- [NeetCode - Connected Components](https://www.youtube.com/watch?v=8f1XPm4WOUc)

### Intuition
```
Union-Find: Start with n components, merge as we process edges.

Visual: n=5, edges = [[0,1],[1,2],[3,4]]

        Initially: 5 components (each node is its own)
        [0] [1] [2] [3] [4]

        Edge [0,1]: Merge 0 and 1 → 4 components
        [0─1] [2] [3] [4]

        Edge [1,2]: Merge 1 and 2 → 3 components
        [0─1─2] [3] [4]

        Edge [3,4]: Merge 3 and 4 → 2 components
        [0─1─2] [3─4]

        Answer: 2 components
```

### Solution
```python
def countComponents(n: int, edges: list[list[int]]) -> int:
    """
    Count connected components using Union-Find.

    Time: O(n + e * α(n))
    Space: O(n)
    """
    uf = UnionFind(n)

    for u, v in edges:
        uf.union(u, v)

    # Count unique roots
    roots = set()
    for i in range(n):
        roots.add(uf.find(i))

    return len(roots)


def countComponents_dfs(n: int, edges: list[list[int]]) -> int:
    """
    Alternative: DFS approach.

    Time: O(V + E)
    Space: O(V + E)
    """
    # Build adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()
    components = 0

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    for i in range(n):
        if i not in visited:
            dfs(i)
            components += 1

    return components
```

### Edge Cases
- No edges → n components
- All nodes connected → 1 component
- Single node → 1 component
- Nodes numbered 0 to n-1 → handle all
- Duplicate edges → union handles idempotently

---

## Problem 7: Alien Dictionary (LC #269) - Hard

- [LeetCode](https://leetcode.com/problems/alien-dictionary/)

### Problem Statement
Derive order of letters in alien language from sorted dictionary.

### Examples
```
Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"
```

### Video Explanation
- [NeetCode - Alien Dictionary](https://www.youtube.com/watch?v=6kTZYvNNyps)

### Intuition
```
Build a graph from word comparisons, then topological sort!

Key insight: Compare ADJACENT words to find letter ordering.

Visual: words = ["wrt","wrf","er","ett","rftt"]

        Compare "wrt" vs "wrf": t < f
        Compare "wrf" vs "er":  w < e
        Compare "er" vs "ett":  r < t
        Compare "ett" vs "rftt": e < r

        Graph: w → e → r → t → f

        Topological sort: "wertf"

Edge case: ["abc", "ab"] is INVALID!
- Longer word can't come before its prefix
```

### Solution
```python
def alienOrder(words: list[str]) -> str:
    """
    Find letter ordering using topological sort.

    Strategy:
    1. Build graph from adjacent word comparisons
    2. Topological sort to find valid ordering

    Time: O(total chars)
    Space: O(unique chars)
    """
    # Build graph: char -> set of chars that come after
    graph = defaultdict(set)
    in_degree = {char: 0 for word in words for char in word}

    # Compare adjacent words to build graph
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]

        # Check for invalid case: prefix comes after longer word
        if len(w1) > len(w2) and w1.startswith(w2):
            return ""

        # Find first different character
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break

    # Topological sort (Kahn's algorithm)
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []

    while queue:
        char = queue.popleft()
        result.append(char)

        for next_char in graph[char]:
            in_degree[next_char] -= 1
            if in_degree[next_char] == 0:
                queue.append(next_char)

    # Check if all characters included (no cycle)
    if len(result) != len(in_degree):
        return ""

    return ''.join(result)
```

### Edge Cases
- Single word → return all unique chars
- ["abc", "ab"] → invalid (prefix comes after)
- Cycle in ordering → return ""
- All same characters → any order valid
- Letters not appearing in comparisons → can be anywhere

---

## Problem 8: Minimum Spanning Tree (LC #1584) - Medium

- [LeetCode](https://leetcode.com/problems/min-cost-to-connect-all-points/)

### Problem Statement
Find minimum cost to connect all points.

### Video Explanation
- [NeetCode - Min Cost to Connect All Points](https://www.youtube.com/watch?v=f7JOBJIC-NA)

### Intuition
```
This is MINIMUM SPANNING TREE (MST)!

Two algorithms: Prim's and Kruskal's

PRIM'S (grow tree from one node):
1. Start from any node
2. Always add the CHEAPEST edge to an unvisited node
3. Repeat until all nodes connected

KRUSKAL'S (grow forest by merging):
1. Sort ALL edges by weight
2. Add edges in order if they don't create cycle
3. Use Union-Find for cycle detection

Visual: Points = [[0,0], [2,2], [3,10], [5,2], [7,0]]

        Connect all points with minimum total distance.

        (3,10)
           │
           │ (large distance)
           │
        (0,0)───(2,2)───(5,2)───(7,0)

MST connects all nodes with minimum total edge weight.
```

### Solution
```python
def minCostConnectPoints(points: list[list[int]]) -> int:
    """
    Find MST using Prim's algorithm.

    Strategy:
    - Start from any point
    - Greedily add closest unvisited point
    - Use heap to efficiently find minimum edge

    Time: O(n² log n)
    Space: O(n²)
    """
    n = len(points)
    if n <= 1:
        return 0

    # Calculate Manhattan distance
    def dist(i, j):
        return abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])

    # Prim's algorithm
    visited = set()
    heap = [(0, 0)]  # (distance, point_index)
    total_cost = 0

    while len(visited) < n:
        cost, u = heapq.heappop(heap)

        if u in visited:
            continue

        visited.add(u)
        total_cost += cost

        # Add edges to unvisited points
        for v in range(n):
            if v not in visited:
                heapq.heappush(heap, (dist(u, v), v))

    return total_cost


def minCostConnectPoints_kruskal(points: list[list[int]]) -> int:
    """
    Alternative: Kruskal's algorithm with Union-Find.

    Time: O(n² log n)
    Space: O(n²)
    """
    n = len(points)

    # Generate all edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
            edges.append((dist, i, j))

    # Sort edges by weight
    edges.sort()

    # Kruskal's algorithm
    uf = UnionFind(n)
    total_cost = 0
    edges_used = 0

    for cost, u, v in edges:
        if uf.union(u, v):
            total_cost += cost
            edges_used += 1

            if edges_used == n - 1:
                break

    return total_cost
```

### Edge Cases
- Single point → return 0
- Two points → return distance between them
- All points collinear → MST is a line
- Points with same coordinates → distance 0
- Large coordinates → use long/int64 for distance calc

---

## Summary: Advanced Graph Problems

| # | Problem | Algorithm | Time |
|---|---------|-----------|------|
| 1 | Course Schedule | Topological Sort (BFS) | O(V + E) |
| 2 | Course Schedule II | Topological Sort | O(V + E) |
| 3 | Network Delay | Dijkstra | O((V+E) log V) |
| 4 | Cheapest Flights | Modified Bellman-Ford | O(k * E) |
| 5 | Redundant Connection | Union-Find | O(n * α(n)) |
| 6 | Connected Components | Union-Find / DFS | O(V + E) |
| 7 | Alien Dictionary | Topological Sort | O(chars) |
| 8 | Min Spanning Tree | Prim's / Kruskal's | O(n² log n) |

---

## Practice More Problems

- [ ] LC #332 - Reconstruct Itinerary
- [ ] LC #399 - Evaluate Division
- [ ] LC #685 - Redundant Connection II
- [ ] LC #1135 - Connecting Cities With Minimum Cost
- [ ] LC #1192 - Critical Connections in a Network

