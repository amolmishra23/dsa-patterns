# Graphs Basics - Easy/Medium Problems

## Problem 1: Number of Islands (LC #200) - Medium

- [LeetCode](https://leetcode.com/problems/number-of-islands/)

### Problem Statement
Given an m x n 2D grid of '1's (land) and '0's (water), return the number of islands. An island is surrounded by water and formed by connecting adjacent lands horizontally or vertically.

### Video Explanation
- [NeetCode - Number of Islands](https://www.youtube.com/watch?v=pV2kpPD66nE)
- [Take U Forward - Number of Islands](https://www.youtube.com/watch?v=muncqlKJrH0)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FLOOD FILL / DFS TO COUNT CONNECTED COMPONENTS                             │
│                                                                             │
│  grid:                                                                      │
│  ["1","1","0","0","0"]                                                     │
│  ["1","1","0","0","0"]                                                     │
│  ["0","0","1","0","0"]                                                     │
│  ["0","0","0","1","1"]                                                     │
│                                                                             │
│  Island 1:  ██        Island 2:    █     Island 3:     ██                  │
│             ██                                                              │
│                                                                             │
│  Algorithm:                                                                 │
│  1. Scan grid for '1' (unvisited land)                                     │
│  2. When found, start DFS/BFS to mark entire island as visited             │
│  3. Increment island count                                                  │
│  4. Continue scanning                                                       │
│                                                                             │
│  DFS explores all 4 directions: UP, DOWN, LEFT, RIGHT                      │
│  Mark visited by changing '1' to '0' (or use visited set)                  │
│                                                                             │
│  Result: 3 islands                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Examples
```
Input: grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
Output: 3
```

### Solution
```python
def numIslands(grid: list[list[str]]) -> int:
    """
    Count islands using DFS.

    Strategy:
    - For each unvisited land cell, start DFS
    - Mark all connected land as visited
    - Each DFS start = one island

    Time: O(m * n)
    Space: O(m * n) for recursion in worst case
    """
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r: int, c: int):
        # Boundary and water check
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return

        # Mark as visited
        grid[r][c] = '0'

        # Explore all 4 directions
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1

    return count
```

### Edge Cases
- Empty grid → return 0
- Grid with all water → return 0
- Grid with all land (single island) → return 1
- Single cell grid → return 1 or 0 based on cell value
- Islands touching corners only (not adjacent) → separate islands

---

## Problem 2: Clone Graph (LC #133) - Medium

- [LeetCode](https://leetcode.com/problems/clone-graph/)

### Problem Statement
Given a reference of a node in a connected undirected graph, return a deep copy (clone) of the graph.

### Video Explanation
- [NeetCode - Clone Graph](https://www.youtube.com/watch?v=mQeF6bN8hMk)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DEEP COPY WITH HASH MAP TO TRACK CLONES                                    │
│                                                                             │
│  Original:          Clone:                                                  │
│    1 --- 2            1' --- 2'                                            │
│    |     |            |      |                                              │
│    4 --- 3            4' --- 3'                                            │
│                                                                             │
│  Challenge: Nodes have circular references (neighbors)                     │
│                                                                             │
│  Solution: Use hash map {original_node: cloned_node}                       │
│                                                                             │
│  DFS Process:                                                               │
│  1. Visit node 1, create clone 1', add to map                              │
│  2. For each neighbor of 1, recursively clone                              │
│  3. If already cloned (in map), return existing clone                      │
│  4. Connect cloned neighbors to cloned node                                │
│                                                                             │
│  Key: Check map BEFORE creating clone to handle cycles                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors else []


def cloneGraph(node: Node) -> Node:
    """
    Clone graph using DFS with hash map.

    Strategy:
    - Use hash map to track cloned nodes
    - DFS to traverse and clone each node

    Time: O(V + E)
    Space: O(V)
    """
    if not node:
        return None

    cloned = {}  # original -> clone

    def dfs(original: Node) -> Node:
        # Return if already cloned
        if original in cloned:
            return cloned[original]

        # Create clone
        clone = Node(original.val)
        cloned[original] = clone

        # Clone neighbors
        for neighbor in original.neighbors:
            clone.neighbors.append(dfs(neighbor))

        return clone

    return dfs(node)
```

### Edge Cases
- Empty/null node → return None
- Single node graph → return clone of that single node
- Graph with self-loops → hash map handles this
- Disconnected nodes → not possible per problem (connected graph)
- Duplicate values in nodes → use node reference, not value, as key

---

## Problem 3: Pacific Atlantic Water Flow (LC #417) - Medium

- [LeetCode](https://leetcode.com/problems/pacific-atlantic-water-flow/)

### Problem Statement
Find cells that can flow to both Pacific and Atlantic oceans.

### Video Explanation
- [NeetCode - Pacific Atlantic](https://www.youtube.com/watch?v=s-VkcjHqkGI)

### Solution
```python
def pacificAtlantic(heights: list[list[int]]) -> list[list[int]]:
    """
    Find cells that flow to both oceans.

    Strategy:
    - Reverse thinking: start from ocean borders
    - DFS from Pacific border, mark reachable cells
    - DFS from Atlantic border, mark reachable cells
    - Return intersection

    Time: O(m * n)
    Space: O(m * n)
    """
    if not heights:
        return []

    rows, cols = len(heights), len(heights[0])
    pacific = set()
    atlantic = set()

    def dfs(r: int, c: int, visited: set, prev_height: int):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            (r, c) in visited or heights[r][c] < prev_height):
            return

        visited.add((r, c))

        dfs(r + 1, c, visited, heights[r][c])
        dfs(r - 1, c, visited, heights[r][c])
        dfs(r, c + 1, visited, heights[r][c])
        dfs(r, c - 1, visited, heights[r][c])

    # DFS from Pacific (top and left)
    for c in range(cols):
        dfs(0, c, pacific, heights[0][c])
    for r in range(rows):
        dfs(r, 0, pacific, heights[r][0])

    # DFS from Atlantic (bottom and right)
    for c in range(cols):
        dfs(rows - 1, c, atlantic, heights[rows - 1][c])
    for r in range(rows):
        dfs(r, cols - 1, atlantic, heights[r][cols - 1])

    # Return intersection
    return list(pacific & atlantic)
```

### Edge Cases
- Empty grid → return []
- Single cell → return [[0,0]] (can flow to both oceans)
- All same height → all cells can flow to both
- Strictly increasing from corner → only corner cells
- Grid with valleys → water can't flow uphill

---

## Problem 4: Rotting Oranges (LC #994) - Medium

- [LeetCode](https://leetcode.com/problems/rotting-oranges/)

### Problem Statement
Find minimum time for all oranges to rot (BFS from all rotten).

### Video Explanation
- [NeetCode - Rotting Oranges](https://www.youtube.com/watch?v=y704fEOx0s0)

### Solution
```python
from collections import deque

def orangesRotting(grid: list[list[int]]) -> int:
    """
    Find time for all oranges to rot using multi-source BFS.

    Strategy:
    - Start BFS from all initially rotten oranges
    - Each level = 1 minute
    - Count fresh oranges, check if all rotted

    Time: O(m * n)
    Space: O(m * n)
    """
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    # Find all rotten oranges and count fresh
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c, 0))
            elif grid[r][c] == 1:
                fresh += 1

    if fresh == 0:
        return 0

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    minutes = 0

    while queue:
        r, c, time = queue.popleft()
        minutes = max(minutes, time)

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[nr][nc] = 2  # Mark as rotten
                fresh -= 1
                queue.append((nr, nc, time + 1))

    return minutes if fresh == 0 else -1
```

### Edge Cases
- No fresh oranges → return 0
- No rotten oranges but fresh exist → return -1
- Fresh orange isolated (surrounded by empty) → return -1
- All rotten → return 0
- Single cell → return 0 if rotten/empty, depends on state

---

## Problem 5: Surrounded Regions (LC #130) - Medium

- [LeetCode](https://leetcode.com/problems/surrounded-regions/)

### Problem Statement
Capture all 'O's not connected to border.

### Video Explanation
- [NeetCode - Surrounded Regions](https://www.youtube.com/watch?v=9z2BunfoZ5Y)

### Solution
```python
def solve(board: list[list[str]]) -> None:
    """
    Capture surrounded regions.

    Strategy:
    - Mark border-connected 'O's as safe (DFS from borders)
    - Convert remaining 'O's to 'X'
    - Restore safe cells to 'O'

    Time: O(m * n)
    Space: O(m * n)
    """
    if not board:
        return

    rows, cols = len(board), len(board[0])

    def dfs(r: int, c: int):
        if r < 0 or r >= rows or c < 0 or c >= cols or board[r][c] != 'O':
            return

        board[r][c] = 'S'  # Mark as safe
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    # Mark border-connected 'O's as safe
    for r in range(rows):
        dfs(r, 0)
        dfs(r, cols - 1)
    for c in range(cols):
        dfs(0, c)
        dfs(rows - 1, c)

    # Convert: 'O' -> 'X', 'S' -> 'O'
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 'O':
                board[r][c] = 'X'
            elif board[r][c] == 'S':
                board[r][c] = 'O'
```

### Edge Cases
- Empty board → no changes
- All 'X' → no changes needed
- All 'O' on border → none captured
- Single row/column → all O's touch border
- O's forming a ring around X's → O's not captured

---

## Problem 6: Course Schedule (LC #207) - Medium

- [LeetCode](https://leetcode.com/problems/course-schedule/)

### Problem Statement
Check if all courses can be completed (cycle detection).

### Video Explanation
- [NeetCode - Course Schedule](https://www.youtube.com/watch?v=EgI5nU9etnU)

### Solution
```python
def canFinish(numCourses: int, prerequisites: list[list[int]]) -> bool:
    """
    Check if courses can be completed (no cycle).

    Strategy:
    - Build adjacency list
    - DFS with cycle detection (visiting state)

    Time: O(V + E)
    Space: O(V + E)
    """
    from collections import defaultdict

    # Build graph
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    # States: 0 = unvisited, 1 = visiting, 2 = visited
    state = [0] * numCourses

    def has_cycle(course: int) -> bool:
        if state[course] == 1:
            return True  # Cycle detected
        if state[course] == 2:
            return False  # Already processed

        state[course] = 1  # Mark as visiting

        for next_course in graph[course]:
            if has_cycle(next_course):
                return True

        state[course] = 2  # Mark as visited
        return False

    # Check each course
    for course in range(numCourses):
        if has_cycle(course):
            return False

    return True
```

### Edge Cases
- No prerequisites → return True (all courses independent)
- Self-dependency [[1,1]] → return False (cycle)
- Single course → return True
- Linear chain of dependencies → return True
- Two courses depend on each other → return False

---

## Problem 7: Course Schedule II (LC #210) - Medium

- [LeetCode](https://leetcode.com/problems/course-schedule-ii/)

### Problem Statement
Return order to complete all courses (topological sort).

### Video Explanation
- [NeetCode - Course Schedule II](https://www.youtube.com/watch?v=Akt3glAwyfY)

### Solution
```python
def findOrder(numCourses: int, prerequisites: list[list[int]]) -> list[int]:
    """
    Find course order using topological sort.

    Strategy:
    - DFS-based topological sort
    - Add to result after processing all dependencies

    Time: O(V + E)
    Space: O(V + E)
    """
    from collections import defaultdict

    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    result = []
    state = [0] * numCourses  # 0=unvisited, 1=visiting, 2=visited

    def dfs(course: int) -> bool:
        if state[course] == 1:
            return False  # Cycle
        if state[course] == 2:
            return True

        state[course] = 1

        for next_course in graph[course]:
            if not dfs(next_course):
                return False

        state[course] = 2
        result.append(course)
        return True

    for course in range(numCourses):
        if not dfs(course):
            return []

    return result[::-1]  # Reverse for correct order
```

### Edge Cases
- No prerequisites → return [0, 1, 2, ..., n-1]
- Cycle exists → return []
- Single course → return [0]
- Multiple valid orderings → return any valid one
- All courses depend on course 0 → 0 comes first

---

## Problem 8: Number of Connected Components (LC #323) - Medium

- [LeetCode](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

### Problem Statement
Count connected components in undirected graph.

### Video Explanation
- [NeetCode - Connected Components](https://www.youtube.com/watch?v=8f1XPm4WOUc)

### Solution
```python
def countComponents(n: int, edges: list[list[int]]) -> int:
    """
    Count connected components using Union-Find.

    Time: O(E * α(V)) ≈ O(E)
    Space: O(V)
    """
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])  # Path compression
        return parent[x]

    def union(x: int, y: int) -> bool:
        px, py = find(x), find(y)
        if px == py:
            return False

        # Union by rank
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

        return True

    components = n
    for a, b in edges:
        if union(a, b):
            components -= 1

    return components
```

### Edge Cases
- No edges → n components (each node isolated)
- n-1 edges and connected → 1 component
- Single node → 1 component
- All nodes connected to one hub → 1 component
- Two separate cliques → 2 components

---

## Problem 9: Graph Valid Tree (LC #261) - Medium

- [LeetCode](https://leetcode.com/problems/graph-valid-tree/)

### Problem Statement
Check if edges form a valid tree (connected, no cycles).

### Video Explanation
- [NeetCode - Graph Valid Tree](https://www.youtube.com/watch?v=bXsUuownnoQ)

### Solution
```python
def validTree(n: int, edges: list[list[int]]) -> bool:
    """
    Check if graph is a valid tree.

    Tree properties:
    - n-1 edges
    - Connected (one component)
    - No cycles

    Time: O(V + E)
    Space: O(V + E)
    """
    # Tree must have exactly n-1 edges
    if len(edges) != n - 1:
        return False

    # Build adjacency list
    from collections import defaultdict
    graph = defaultdict(list)
    for a, b in edges:
        graph[a].append(b)
        graph[b].append(a)

    # BFS to check connectivity
    visited = set([0])
    queue = [0]

    while queue:
        node = queue.pop(0)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    # All nodes should be reachable
    return len(visited) == n
```

### Edge Cases
- n=1, no edges → True (single node is a tree)
- n nodes, n edges → False (has cycle)
- n nodes, n-2 edges → False (disconnected)
- Edges forming a line → True
- Self-loop → False

---

## Problem 10: Walls and Gates (LC #286) - Medium

- [LeetCode](https://leetcode.com/problems/walls-and-gates/)

### Problem Statement
Fill each empty room with distance to nearest gate.

### Video Explanation
- [NeetCode - Walls and Gates](https://www.youtube.com/watch?v=e69C6xhiSQE)

### Solution
```python
def wallsAndGates(rooms: list[list[int]]) -> None:
    """
    Fill distances using multi-source BFS from gates.

    Strategy:
    - Start BFS from all gates simultaneously
    - Each level = distance + 1

    Time: O(m * n)
    Space: O(m * n)
    """
    if not rooms:
        return

    rows, cols = len(rooms), len(rooms[0])
    INF = 2147483647
    queue = deque()

    # Find all gates
    for r in range(rows):
        for c in range(cols):
            if rooms[r][c] == 0:
                queue.append((r, c))

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        r, c = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and rooms[nr][nc] == INF:
                rooms[nr][nc] = rooms[r][c] + 1
                queue.append((nr, nc))
```

### Edge Cases
- No gates → all rooms stay INF
- No empty rooms → nothing to fill
- All walls → nothing to fill
- Single gate → distances radiate from it
- Room surrounded by walls → stays INF

---

## Summary: Graph Basics Problems

| # | Problem | Algorithm | Time |
|---|---------|-----------|------|
| 1 | Number of Islands | DFS flood fill | O(mn) |
| 2 | Clone Graph | DFS + hash map | O(V+E) |
| 3 | Pacific Atlantic | Reverse DFS | O(mn) |
| 4 | Rotting Oranges | Multi-source BFS | O(mn) |
| 5 | Surrounded Regions | Border DFS | O(mn) |
| 6 | Course Schedule | Cycle detection | O(V+E) |
| 7 | Course Schedule II | Topological sort | O(V+E) |
| 8 | Connected Components | Union-Find | O(E) |
| 9 | Valid Tree | Edge count + BFS | O(V+E) |
| 10 | Walls and Gates | Multi-source BFS | O(mn) |

---

## Practice More Problems

- [ ] LC #547 - Number of Provinces
- [ ] LC #684 - Redundant Connection
- [ ] LC #695 - Max Area of Island
- [ ] LC #785 - Is Graph Bipartite?
- [ ] LC #1091 - Shortest Path in Binary Matrix

