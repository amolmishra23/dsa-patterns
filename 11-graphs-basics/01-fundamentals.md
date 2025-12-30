# Graphs - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE GRAPH ALGORITHMS                             │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Connected components"                                                   │
│  ✓ "Shortest path"                                                          │
│  ✓ "Number of islands"                                                      │
│  ✓ "Course schedule" / "Prerequisites"                                      │
│  ✓ "Network" / "Connections"                                                │
│  ✓ "Cycle detection"                                                        │
│  ✓ "Reachability"                                                           │
│                                                                             │
│  Key insight: Many problems can be modeled as graphs!                       │
│  - Grid → graph (cells are nodes, adjacent cells are edges)                 │
│  - Dependencies → directed graph                                            │
│  - Social network → undirected graph                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Graph terminology (vertices, edges, directed/undirected)
- [ ] Adjacency list representation
- [ ] Recursion and iteration

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GRAPHS BASICS MEMORY MAP                                 │
│                                                                             │
│                    ┌─────────────┐                                          │
│         ┌─────────│   GRAPHS    │─────────┐                                 │
│         │         └─────────────┘         │                                 │
│         ▼                                 ▼                                 │
│  ┌─────────────┐                   ┌─────────────┐                          │
│  │    DFS      │                   │    BFS      │                          │
│  └──────┬──────┘                   └──────┬──────┘                          │
│         │                                 │                                 │
│    ┌────┴────┐                      ┌─────┴─────┐                           │
│    ▼         ▼                      ▼           ▼                           │
│ ┌──────┐ ┌──────┐               ┌──────┐   ┌──────┐                        │
│ │Path  │ │Cycle │               │Short-│   │Level │                        │
│ │Finding│ │Detect│              │est   │   │Order │                        │
│ └──────┘ └──────┘               └──────┘   └──────┘                        │
│                                                                             │
│  Related Patterns:                                                          │
│  • Trees - Trees are special graphs (connected, no cycles)                  │
│  • Union-Find - For dynamic connectivity                                    │
│  • Topological Sort - For DAG ordering                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GRAPH TRAVERSAL DECISION TREE                            │
│                                                                             │
│  Need shortest path (unweighted)?                                           │
│       │                                                                     │
│       ├── YES → Use BFS                                                     │
│       │                                                                     │
│       └── NO → Need to explore all paths?                                   │
│                    │                                                        │
│                    ├── YES → Use DFS                                        │
│                    │                                                        │
│                    └── NO → Need to detect cycle?                           │
│                                 │                                           │
│                                 ├── Undirected → DFS with parent tracking   │
│                                 │                                           │
│                                 └── Directed → DFS with coloring            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### Graph Representation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GRAPH REPRESENTATIONS                                    │
│                                                                             │
│  Graph:    0 ─── 1                                                          │
│            │     │                                                          │
│            │     │                                                          │
│            3 ─── 2                                                          │
│                                                                             │
│  ADJACENCY LIST (Most Common):                                              │
│  graph = {                                                                  │
│      0: [1, 3],                                                             │
│      1: [0, 2],                                                             │
│      2: [1, 3],                                                             │
│      3: [0, 2]                                                              │
│  }                                                                          │
│  Space: O(V + E)                                                            │
│  Check edge: O(degree)                                                      │
│  Get neighbors: O(1)                                                        │
│                                                                             │
│  ADJACENCY MATRIX:                                                          │
│       0  1  2  3                                                            │
│    0 [0, 1, 0, 1]                                                           │
│    1 [1, 0, 1, 0]                                                           │
│    2 [0, 1, 0, 1]                                                           │
│    3 [1, 0, 1, 0]                                                           │
│  Space: O(V²)                                                               │
│  Check edge: O(1)                                                           │
│  Get neighbors: O(V)                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Building a Graph

```python
from collections import defaultdict

def build_graph_undirected(edges: list[list[int]]) -> dict:
    """
    Build undirected graph from edge list.

    edges = [[0,1], [1,2], [2,3], [3,0]]
    """
    graph = defaultdict(list)

    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)  # Undirected: add both directions

    return graph


def build_graph_directed(edges: list[list[int]]) -> dict:
    """
    Build directed graph from edge list.

    edges = [[0,1], [1,2]] means 0→1, 1→2
    """
    graph = defaultdict(list)

    for u, v in edges:
        graph[u].append(v)  # Only one direction

    return graph
```

---

## Graph Traversals

### DFS (Depth-First Search)

```python
def dfs_recursive(graph: dict, start: int, visited: set = None) -> list[int]:
    """
    DFS traversal using recursion.

    Strategy: Go as deep as possible, then backtrack.

    Time: O(V + E)
    Space: O(V) for visited set + O(V) recursion stack
    """
    if visited is None:
        visited = set()

    result = []

    def dfs(node: int):
        # Mark as visited
        visited.add(node)
        result.append(node)

        # Visit all unvisited neighbors
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return result


def dfs_iterative(graph: dict, start: int) -> list[int]:
    """
    DFS traversal using explicit stack.

    Time: O(V + E)
    Space: O(V)
    """
    visited = set()
    stack = [start]
    result = []

    while stack:
        node = stack.pop()

        if node in visited:
            continue

        visited.add(node)
        result.append(node)

        # Add neighbors to stack (in reverse for left-to-right order)
        for neighbor in reversed(graph[node]):
            if neighbor not in visited:
                stack.append(neighbor)

    return result
```

### BFS (Breadth-First Search)

```python
from collections import deque

def bfs(graph: dict, start: int) -> list[int]:
    """
    BFS traversal using queue.

    Strategy: Visit all neighbors at current level before going deeper.

    Time: O(V + E)
    Space: O(V)
    """
    visited = set([start])
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        # Add unvisited neighbors to queue
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result


def bfs_with_level(graph: dict, start: int) -> list[list[int]]:
    """
    BFS with level tracking.

    Returns nodes grouped by their distance from start.
    """
    visited = set([start])
    queue = deque([start])
    levels = []

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        levels.append(current_level)

    return levels
```

---

## Visual: DFS vs BFS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DFS vs BFS TRAVERSAL                                     │
│                                                                             │
│  Graph:        1                                                            │
│               /|\                                                           │
│              2 3 4                                                          │
│             /|   |\                                                         │
│            5 6   7 8                                                        │
│                                                                             │
│  DFS (using stack):                                                         │
│  Visit order: 1 → 2 → 5 → 6 → 3 → 4 → 7 → 8                                │
│  Goes DEEP first, then backtracks                                           │
│                                                                             │
│  BFS (using queue):                                                         │
│  Visit order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8                                │
│  Level 0: [1]                                                               │
│  Level 1: [2, 3, 4]                                                         │
│  Level 2: [5, 6, 7, 8]                                                      │
│  Visits level by level                                                      │
│                                                                             │
│  When to use which?                                                         │
│  DFS: Find any path, cycle detection, topological sort                      │
│  BFS: Shortest path (unweighted), level-order traversal                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Graph Patterns

### Pattern 1: Connected Components

```python
def count_components(n: int, edges: list[list[int]]) -> int:
    """
    Count connected components in undirected graph.

    Strategy: DFS/BFS from each unvisited node.
    Each DFS explores one component.

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

    def dfs(node: int):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    # Try starting DFS from each node
    for node in range(n):
        if node not in visited:
            dfs(node)
            components += 1  # Found a new component

    return components
```

### Pattern 2: Cycle Detection

```python
def has_cycle_undirected(n: int, edges: list[list[int]]) -> bool:
    """
    Detect cycle in undirected graph.

    Strategy: During DFS, if we visit a node that's already visited
    and it's not our parent, there's a cycle.

    Time: O(V + E)
    Space: O(V)
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    visited = set()

    def dfs(node: int, parent: int) -> bool:
        visited.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                if dfs(neighbor, node):
                    return True
            elif neighbor != parent:
                # Visited node that's not parent = cycle!
                return True

        return False

    # Check all components
    for node in range(n):
        if node not in visited:
            if dfs(node, -1):
                return True

    return False


def has_cycle_directed(n: int, edges: list[list[int]]) -> bool:
    """
    Detect cycle in directed graph.

    Strategy: Use three states - unvisited, in current path, fully processed.
    If we visit a node that's in current path, there's a cycle.

    Time: O(V + E)
    Space: O(V)
    """
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)

    # 0 = unvisited, 1 = in current path, 2 = processed
    state = [0] * n

    def dfs(node: int) -> bool:
        state[node] = 1  # Mark as in current path

        for neighbor in graph[node]:
            if state[neighbor] == 1:
                return True  # Back edge = cycle!
            if state[neighbor] == 0:
                if dfs(neighbor):
                    return True

        state[node] = 2  # Mark as fully processed
        return False

    for node in range(n):
        if state[node] == 0:
            if dfs(node):
                return True

    return False
```

### Pattern 3: Grid as Graph

```python
def num_islands(grid: list[list[str]]) -> int:
    """
    Count islands in a grid (LC #200).

    Grid is a graph where:
    - Each cell is a node
    - Adjacent cells (up/down/left/right) are connected

    Time: O(rows * cols)
    Space: O(rows * cols) for recursion stack
    """
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    islands = 0

    def dfs(r: int, c: int):
        """Sink the island by marking visited cells."""
        # Check bounds and if it's land
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return

        # Mark as visited (sink it)
        grid[r][c] = '0'

        # Explore all 4 directions
        dfs(r + 1, c)  # Down
        dfs(r - 1, c)  # Up
        dfs(r, c + 1)  # Right
        dfs(r, c - 1)  # Left

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                islands += 1

    return islands
```

---

## Complexity Analysis

| Algorithm | Time | Space | Use Case |
|-----------|------|-------|----------|
| DFS | O(V + E) | O(V) | Paths, cycles, components |
| BFS | O(V + E) | O(V) | Shortest path (unweighted) |
| Topological Sort | O(V + E) | O(V) | Dependencies, scheduling |
| Dijkstra | O((V+E) log V) | O(V) | Shortest path (weighted) |
| Union-Find | O(α(n)) per op | O(V) | Connected components, MST |

---

## Common Mistakes

```python
# ❌ WRONG: Not marking visited before adding to queue
def bfs_wrong(graph, start):
    visited = set()
    queue = deque([start])

    while queue:
        node = queue.popleft()
        visited.add(node)  # Too late! Node might be added multiple times

        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append(neighbor)

# ✅ CORRECT: Mark visited when adding to queue
def bfs_correct(graph, start):
    visited = set([start])  # Mark start as visited immediately
    queue = deque([start])

    while queue:
        node = queue.popleft()

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)  # Mark before adding to queue
                queue.append(neighbor)


# ❌ WRONG: Forgetting to check all components
def count_components_wrong(n, edges):
    graph = build_graph(edges)
    visited = set()
    dfs(0, visited)  # Only explores component containing node 0!
    return 1

# ✅ CORRECT: Start DFS from each unvisited node
def count_components_correct(n, edges):
    graph = build_graph(edges)
    visited = set()
    components = 0

    for node in range(n):
        if node not in visited:
            dfs(node, visited)
            components += 1

    return components
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"I'll model this as a graph problem. Each [entity] is a node, and
[relationships] are edges. Then I'll use DFS/BFS to traverse and
find [what we're looking for]. Time is O(V+E)."
```

### 2. What Interviewers Look For
- **Problem modeling**: Can you convert the problem to a graph?
- **Traversal choice**: BFS for shortest path, DFS for exploration
- **Visited tracking**: Prevent infinite loops

### 3. Common Follow-up Questions
- "What if graph is disconnected?" → Check all nodes, not just one
- "Can you do it iteratively?" → Use explicit stack for DFS
- "What about weighted edges?" → Use Dijkstra or Bellman-Ford

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
