# Graphs Basics - Practice List

## Problems by Pattern

### BFS (Shortest Path, Level Order)
- LC 994: Rotting Oranges (Medium)
- LC 542: 01 Matrix (Medium)
- LC 1091: Shortest Path in Binary Matrix (Medium)
- LC 127: Word Ladder (Hard)
- LC 752: Open the Lock (Medium)
- LC 773: Sliding Puzzle (Hard)

### DFS (Traversal, Connected Components)
- LC 200: Number of Islands (Medium)
- LC 695: Max Area of Island (Medium)
- LC 733: Flood Fill (Easy)
- LC 130: Surrounded Regions (Medium)
- LC 417: Pacific Atlantic Water Flow (Medium)
- LC 547: Number of Provinces (Medium)

### Clone/Copy Graph
- LC 133: Clone Graph (Medium)
- LC 138: Copy List with Random Pointer (Medium)

### Cycle Detection
- LC 207: Course Schedule (Medium)
- LC 210: Course Schedule II (Medium)
- LC 802: Find Eventual Safe States (Medium)

### Bipartite
- LC 785: Is Graph Bipartite (Medium)
- LC 886: Possible Bipartition (Medium)

## Templates

```python
from collections import deque

# BFS Template
def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

# DFS Template (Iterative)
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)

# DFS Template (Recursive)
def dfs_recursive(graph, node, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

# Number of Islands
def numIslands(grid):
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'
        dfs(r+1, c); dfs(r-1, c); dfs(r, c+1); dfs(r, c-1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1
    return count

# Cycle Detection (Course Schedule)
def canFinish(numCourses, prerequisites):
    graph = [[] for _ in range(numCourses)]
    for course, prereq in prerequisites:
        graph[prereq].append(course)

    # 0: unvisited, 1: visiting, 2: visited
    state = [0] * numCourses

    def has_cycle(node):
        if state[node] == 1:
            return True
        if state[node] == 2:
            return False
        state[node] = 1
        for neighbor in graph[node]:
            if has_cycle(neighbor):
                return True
        state[node] = 2
        return False

    return not any(has_cycle(i) for i in range(numCourses))
```

## Key Insights
- BFS for shortest path in unweighted graphs
- DFS for exploring all paths, cycle detection
- Mark visited to avoid infinite loops
- Grid problems: treat each cell as a node

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      GRAPH TRAVERSAL PATTERNS                               │
│                                                                             │
│  BFS (Level-by-Level):                                                      │
│                                                                             │
│       0 ─── 1           Level 0: [0]                                        │
│       │     │           Level 1: [1, 2]                                     │
│       2 ─── 3           Level 2: [3]                                        │
│                                                                             │
│  Queue progression:                                                         │
│  [0] → process 0, add 1,2 → [1,2]                                           │
│  [1,2] → process 1, add 3 → [2,3]                                           │
│  [2,3] → process 2 (3 already queued) → [3]                                 │
│  [3] → process 3 → []                                                       │
│                                                                             │
│  Order visited: 0 → 1 → 2 → 3 (level order)                                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DFS (Go Deep First):                                                       │
│                                                                             │
│       0 ─── 1           Stack: [0]                                          │
│       │     │           Pop 0, push 2,1 → [2,1]                             │
│       2 ─── 3           Pop 1, push 3 → [2,3]                               │
│                         Pop 3 → [2]                                         │
│                         Pop 2 → []                                          │
│                                                                             │
│  Order visited: 0 → 1 → 3 → 2 (depth first)                                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  GRID TRAVERSAL (Number of Islands):                                        │
│                                                                             │
│  Grid:                    4-Direction Movement:                             │
│  ┌───┬───┬───┬───┬───┐            ↑                                         │
│  │ 1 │ 1 │ 0 │ 0 │ 0 │        ← (r,c) →                                     │
│  ├───┼───┼───┼───┼───┤            ↓                                         │
│  │ 1 │ 1 │ 0 │ 0 │ 0 │                                                      │
│  ├───┼───┼───┼───┼───┤   dirs = [(0,1),(0,-1),(1,0),(-1,0)]                 │
│  │ 0 │ 0 │ 1 │ 0 │ 0 │                                                      │
│  ├───┼───┼───┼───┼───┤   Island 1: cells (0,0),(0,1),(1,0),(1,1)            │
│  │ 0 │ 0 │ 0 │ 1 │ 1 │   Island 2: cell (2,2)                               │
│  └───┴───┴───┴───┴───┘   Island 3: cells (3,3),(3,4)                        │
│                                                                             │
│  DFS marks visited by changing '1' to '0'                                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MULTI-SOURCE BFS (Rotting Oranges):                                        │
│                                                                             │
│  Initial:    Minute 1:    Minute 2:    Minute 3:                            │
│  [2,1,1]     [2,2,1]      [2,2,2]      [2,2,2]                               │
│  [1,1,0]     [2,1,0]      [2,2,0]      [2,2,0]                               │
│  [0,1,1]     [0,1,1]      [0,2,1]      [0,2,2]                               │
│                                                                             │
│  Start BFS from ALL rotten oranges (2s) simultaneously                      │
│  Each level = 1 minute of spreading                                         │
│  Answer = number of BFS levels = 3 minutes                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CYCLE DETECTION (3-State DFS):                                             │
│                                                                             │
│  0 → 1 → 2                State colors:                                     │
│      ↑   ↓                WHITE (0): unvisited                              │
│      └── 3                GRAY  (1): currently visiting                     │
│                           BLACK (2): finished                               │
│                                                                             │
│  DFS from 0:                                                                │
│  0(GRAY) → 1(GRAY) → 2(GRAY) → 3(GRAY)                                      │
│                               ↓                                             │
│                           3 → 1 (1 is GRAY = CYCLE!)                        │
│                                                                             │
│  If we reach a GRAY node during DFS = cycle exists                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: Grid DFS
- [ ] LC 733: Flood Fill (Easy)
- [ ] LC 200: Number of Islands (Medium)
- [ ] LC 695: Max Area of Island (Medium)
- [ ] LC 130: Surrounded Regions (Medium)
- [ ] LC 417: Pacific Atlantic Water Flow (Medium)

### Week 2: BFS Fundamentals
- [ ] LC 994: Rotting Oranges (Medium)
- [ ] LC 542: 01 Matrix (Medium)
- [ ] LC 1091: Shortest Path in Binary Matrix (Medium)
- [ ] LC 752: Open the Lock (Medium)
- [ ] LC 127: Word Ladder (Hard)

### Week 3: Graph Properties
- [ ] LC 133: Clone Graph (Medium)
- [ ] LC 207: Course Schedule (Medium)
- [ ] LC 210: Course Schedule II (Medium)
- [ ] LC 785: Is Graph Bipartite (Medium)
- [ ] LC 547: Number of Provinces (Medium)

---

## Common Mistakes

### 1. Not Marking Visited Before Adding to Queue
```python
# WRONG - may add same node multiple times
queue = deque([start])
while queue:
    node = queue.popleft()
    visited.add(node)  # Too late!
    for neighbor in graph[node]:
        queue.append(neighbor)  # Duplicate entries!

# CORRECT - mark visited when adding
visited = {start}
queue = deque([start])
while queue:
    node = queue.popleft()
    for neighbor in graph[node]:
        if neighbor not in visited:
            visited.add(neighbor)  # Mark immediately
            queue.append(neighbor)
```

### 2. Wrong Cycle Detection Logic
```python
# WRONG - only checks if visited
def has_cycle(node):
    if node in visited:
        return True  # Wrong! Could be from different path
    visited.add(node)
    ...

# CORRECT - use 3 states
def has_cycle(node):
    if state[node] == 1:  # Currently visiting = cycle
        return True
    if state[node] == 2:  # Already processed = no cycle
        return False
    state[node] = 1  # Mark as visiting
    for neighbor in graph[node]:
        if has_cycle(neighbor):
            return True
    state[node] = 2  # Mark as processed
    return False
```

### 3. Grid Bounds Check After Access
```python
# WRONG - access before bounds check
def dfs(r, c):
    if grid[r][c] != '1':  # May crash!
        return
    if r < 0 or r >= rows:  # Too late
        return

# CORRECT - bounds check first
def dfs(r, c):
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return
    if grid[r][c] != '1':
        return
```

### 4. Modifying Grid Without Restoration
```python
# WRONG for some problems - permanent modification
def dfs(r, c):
    grid[r][c] = '0'  # Modified permanently
    dfs(r+1, c)
    # No restoration - grid is changed!

# CORRECT for backtracking problems
def dfs(r, c):
    temp = grid[r][c]
    grid[r][c] = '#'  # Mark visited
    dfs(r+1, c)
    grid[r][c] = temp  # Restore
```

### 5. BFS Level Tracking Error
```python
# WRONG - not tracking levels properly
level = 0
while queue:
    node = queue.popleft()
    level += 1  # Wrong! Increments per node, not per level

# CORRECT - process entire level at once
level = 0
while queue:
    level_size = len(queue)
    for _ in range(level_size):
        node = queue.popleft()
        # process node
    level += 1  # Increment after processing entire level
```

---

## Complexity Reference

| Pattern | Time | Space | Notes |
|---------|------|-------|-------|
| BFS | O(V + E) | O(V) | Queue holds at most one level |
| DFS | O(V + E) | O(V) | Stack/recursion depth |
| Grid BFS | O(m * n) | O(m * n) | Each cell visited once |
| Grid DFS | O(m * n) | O(m * n) | Recursion stack |
| Cycle Detection | O(V + E) | O(V) | 3-state coloring |
| Topological Sort | O(V + E) | O(V) | Kahn's or DFS |

---

## Pattern Recognition

| See This | Think This |
|----------|------------|
| "Shortest path" (unweighted) | BFS |
| "Count connected components" | DFS/BFS + counter |
| "Can reach from A to B" | DFS/BFS |
| "Minimum steps/moves" | BFS with level tracking |
| "Detect cycle" | DFS with 3 states |
| "Order of dependencies" | Topological sort |
| "Grid traversal" | DFS with 4 directions |
| "Spread/infection" | Multi-source BFS |
| "Clone/copy graph" | BFS/DFS with hashmap |
