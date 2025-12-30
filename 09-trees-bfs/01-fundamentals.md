# Trees & Graphs BFS - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE BFS                                          │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Shortest path" (unweighted graph)                                       │
│  ✓ "Minimum steps/moves"                                                    │
│  ✓ "Level by level" / "Level order"                                         │
│  ✓ "Nearest" / "Closest"                                                    │
│  ✓ "Spread" / "Propagate" (multi-source)                                    │
│  ✓ "Transform A to B in minimum steps"                                      │
│                                                                             │
│  Key insight: BFS explores all nodes at distance d before distance d+1      │
│               This guarantees shortest path in unweighted graphs!           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning BFS, ensure you understand:
- [ ] Queue data structure (FIFO)
- [ ] Basic tree structure (nodes, children)
- [ ] Graph representation (adjacency list/matrix)
- [ ] Hash set for visited tracking

---

## Core Concept

BFS explores nodes **level by level**, using a queue to process nodes in order of their distance from the start.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BFS VISUALIZATION                                        │
│                                                                             │
│  Tree BFS (Level Order):                                                    │
│                                                                             │
│           1          Level 0: [1]                                           │
│          / \                                                                │
│         2   3        Level 1: [2, 3]                                        │
│        / \   \                                                              │
│       4   5   6      Level 2: [4, 5, 6]                                     │
│                                                                             │
│  Queue progression:                                                         │
│  [1] → process 1, add children → [2, 3]                                     │
│  [2, 3] → process 2, add children → [3, 4, 5]                               │
│  [3, 4, 5] → process 3, add child → [4, 5, 6]                               │
│  [4, 5, 6] → process all (no children)                                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Graph BFS (Shortest Path):                                                 │
│                                                                             │
│  A ─── B ─── C        Find shortest A → E                                   │
│  │     │                                                                    │
│  D ─── E              BFS from A:                                           │
│                       Distance 0: {A}                                       │
│                       Distance 1: {B, D}                                    │
│                       Distance 2: {C, E} ← Found E! Distance = 2            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BFS PATTERN MEMORY MAP                                   │
│                                                                             │
│                         ┌─────────┐                                         │
│              ┌──────────│   BFS   │──────────┐                              │
│              │          └─────────┘          │                              │
│              ▼                               ▼                              │
│     ┌─────────────────┐             ┌─────────────────┐                     │
│     │    Tree BFS     │             │   Graph BFS     │                     │
│     │  (Level Order)  │             │ (Shortest Path) │                     │
│     └────────┬────────┘             └────────┬────────┘                     │
│              │                               │                              │
│     ┌────────┴────────┐             ┌────────┴────────┐                     │
│     ▼                 ▼             ▼                 ▼                     │
│ ┌────────┐      ┌──────────┐  ┌──────────┐     ┌───────────┐               │
│ │Zigzag  │      │Right View│  │Grid BFS  │     │Multi-src  │               │
│ │Traverse│      │Bottom Up │  │(Islands) │     │(Rotting)  │               │
│ └────────┘      └──────────┘  └──────────┘     └───────────┘               │
│                                                                             │
│  Related Patterns:                                                          │
│  • DFS - When you need to explore deeply first                              │
│  • Dijkstra - When edges have different weights                             │
│  • Topological Sort - When you need ordering with dependencies              │
│                                                                             │
│  When to combine:                                                           │
│  • BFS + Hash Map: Track visited states in state-space search               │
│  • BFS + Level tracking: Group results by distance                          │
│  • Multi-source BFS: Start from multiple nodes simultaneously               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BFS vs DFS DECISION TREE                                 │
│                                                                             │
│  Need shortest path in unweighted graph?                                    │
│       │                                                                     │
│       ├── YES → Use BFS                                                     │
│       │                                                                     │
│       └── NO → Need level-by-level processing?                              │
│                    │                                                        │
│                    ├── YES → Use BFS                                        │
│                    │                                                        │
│                    └── NO → Need to explore all paths?                      │
│                                 │                                           │
│                                 ├── YES → Use DFS (backtracking)            │
│                                 │                                           │
│                                 └── NO → Need to find any path?             │
│                                              │                              │
│                                              ├── YES → Either works         │
│                                              │         (DFS uses less mem)  │
│                                              │                              │
│                                              └── NO → Check specific need   │
│                                                                             │
│  QUICK REFERENCE:                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ BFS                          │ DFS                                 │     │
│  ├──────────────────────────────┼─────────────────────────────────────┤     │
│  │ Shortest path (unweighted)   │ All paths / permutations            │     │
│  │ Level order traversal        │ Path existence                      │     │
│  │ Nearest neighbor             │ Cycle detection                     │     │
│  │ Multi-source spread          │ Topological sort                    │     │
│  │ O(V+E) time, O(V) space      │ O(V+E) time, O(H) space             │     │
│  └──────────────────────────────┴─────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Essential Templates

### Template 1: Basic BFS (Tree Level Order)

```python
from collections import deque

def level_order_traversal(root) -> list[list[int]]:
    """
    Process tree level by level.

    Time: O(n) - visit each node once
    Space: O(w) - max width of tree (worst case n/2 for complete tree)
    """
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        level_size = len(queue)  # Key: snapshot current level size

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP-BY-STEP: Level Order Traversal                                        │
│                                                                             │
│  Tree:      3                                                               │
│            / \                                                              │
│           9  20                                                             │
│             /  \                                                            │
│            15   7                                                           │
│                                                                             │
│  Step 1: queue = [3], level_size = 1                                        │
│          Process 3, add 9, 20                                               │
│          level = [3], result = [[3]]                                        │
│                                                                             │
│  Step 2: queue = [9, 20], level_size = 2                                    │
│          Process 9 (no children)                                            │
│          Process 20, add 15, 7                                              │
│          level = [9, 20], result = [[3], [9, 20]]                           │
│                                                                             │
│  Step 3: queue = [15, 7], level_size = 2                                    │
│          Process 15, 7 (no children)                                        │
│          level = [15, 7], result = [[3], [9, 20], [15, 7]]                  │
│                                                                             │
│  Final: [[3], [9, 20], [15, 7]]                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Template 2: Graph BFS (Shortest Path)

```python
from collections import deque

def shortest_path(graph: dict, start: str, end: str) -> int:
    """
    Find shortest path in unweighted graph.

    Time: O(V + E)
    Space: O(V)
    """
    if start == end:
        return 0

    queue = deque([(start, 0)])  # (node, distance)
    visited = {start}

    while queue:
        node, dist = queue.popleft()

        for neighbor in graph.get(node, []):
            if neighbor == end:
                return dist + 1

            if neighbor not in visited:
                visited.add(neighbor)  # Mark when ADDING, not when processing
                queue.append((neighbor, dist + 1))

    return -1  # No path found
```

### Template 3: Grid BFS

```python
from collections import deque

def grid_bfs(grid: list[list[int]], start: tuple, end: tuple) -> int:
    """
    Find shortest path in grid (4-directional).

    Time: O(m * n)
    Space: O(m * n)
    """
    m, n = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    queue = deque([(start[0], start[1], 0)])
    visited = {start}

    while queue:
        r, c, dist = queue.popleft()

        if (r, c) == end:
            return dist

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if (0 <= nr < m and 0 <= nc < n and
                grid[nr][nc] == 0 and (nr, nc) not in visited):
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))

    return -1
```

### Template 4: Multi-Source BFS

```python
from collections import deque

def multi_source_bfs(grid: list[list[int]], sources: list[tuple]) -> int:
    """
    BFS from multiple starting points simultaneously.
    Used for: Rotting Oranges, 01 Matrix, Walls and Gates

    Time: O(m * n)
    Space: O(m * n)
    """
    m, n = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Add ALL sources to queue initially
    queue = deque(sources)
    visited = set(sources)
    distance = 0

    while queue:
        # Process entire level (all nodes at current distance)
        for _ in range(len(queue)):
            r, c = queue.popleft()

            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                if (0 <= nr < m and 0 <= nc < n and
                    (nr, nc) not in visited and grid[nr][nc] != -1):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        distance += 1

    return distance - 1  # Subtract 1 because we count after last level
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MULTI-SOURCE BFS VISUALIZATION (Rotting Oranges)                           │
│                                                                             │
│  Initial:     Minute 1:     Minute 2:     Minute 3:     Minute 4:           │
│  2 1 1        2 2 1         2 2 2         2 2 2         2 2 2               │
│  1 1 0   →    2 1 0    →    2 2 0    →    2 2 0    →    2 2 0               │
│  0 1 1        0 1 1         0 1 1         0 2 1         0 2 2               │
│                                                                             │
│  Queue: [(0,0)]  [(0,1),(1,0)]  [(0,2),(1,1)]  [(2,1)]    [(2,2)]           │
│                                                                             │
│  Key: Start with ALL rotten oranges in queue!                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common BFS Patterns

### Pattern 1: State-Space BFS

```python
from collections import deque

def open_lock(deadends: list[str], target: str) -> int:
    """
    Find minimum turns to reach target combination.
    Each state is a 4-digit lock combination.

    Time: O(10^4 * 4) = O(40000)
    Space: O(10^4)
    """
    dead = set(deadends)
    if "0000" in dead:
        return -1

    queue = deque([("0000", 0)])
    visited = {"0000"}

    while queue:
        state, turns = queue.popleft()

        if state == target:
            return turns

        # Generate all possible next states
        for i in range(4):
            digit = int(state[i])
            for delta in [-1, 1]:
                new_digit = (digit + delta) % 10
                new_state = state[:i] + str(new_digit) + state[i+1:]

                if new_state not in visited and new_state not in dead:
                    visited.add(new_state)
                    queue.append((new_state, turns + 1))

    return -1
```

### Pattern 2: Bidirectional BFS

```python
from collections import deque

def bidirectional_bfs(begin: str, end: str, word_list: list[str]) -> int:
    """
    Search from both ends, meeting in the middle.
    Reduces time complexity from O(b^d) to O(b^(d/2))

    Time: O(n * m) where n = words, m = word length
    Space: O(n)
    """
    word_set = set(word_list)
    if end not in word_set:
        return 0

    front = {begin}
    back = {end}
    visited = {begin, end}
    steps = 1

    while front and back:
        # Always expand smaller set
        if len(front) > len(back):
            front, back = back, front

        next_front = set()

        for word in front:
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_word = word[:i] + c + word[i+1:]

                    if new_word in back:
                        return steps + 1

                    if new_word in word_set and new_word not in visited:
                        visited.add(new_word)
                        next_front.add(new_word)

        front = next_front
        steps += 1

    return 0
```

---

## Complexity Analysis

| Operation | Time | Space | Notes |
|-----------|------|-------|-------|
| Tree BFS | O(n) | O(w) | w = max width |
| Graph BFS | O(V + E) | O(V) | V = vertices, E = edges |
| Grid BFS | O(m × n) | O(m × n) | m × n grid |
| Multi-source BFS | O(m × n) | O(k + m × n) | k = sources |
| State-space BFS | O(states × transitions) | O(states) | Depends on problem |

**Why BFS guarantees shortest path:**
- BFS explores nodes in order of distance from start
- First time we reach a node = shortest path to that node
- This only works for **unweighted** graphs (all edges cost 1)

---

## Common Mistakes

```python
# ❌ WRONG: Marking visited when PROCESSING instead of when ADDING
def bfs_wrong(graph, start):
    queue = deque([start])
    visited = set()

    while queue:
        node = queue.popleft()
        if node in visited:  # Too late! Node may be added multiple times
            continue
        visited.add(node)

        for neighbor in graph[node]:
            queue.append(neighbor)  # Same node can be added many times!

# ✅ CORRECT: Mark visited when ADDING to queue
def bfs_correct(graph, start):
    queue = deque([start])
    visited = {start}  # Mark start as visited immediately

    while queue:
        node = queue.popleft()

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)  # Mark BEFORE adding
                queue.append(neighbor)


# ❌ WRONG: Not tracking level size for level-order problems
def level_order_wrong(root):
    queue = deque([root])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node.val)  # All nodes in flat list, no levels!
        # ...

# ✅ CORRECT: Snapshot level size before processing
def level_order_correct(root):
    queue = deque([root])
    result = []

    while queue:
        level_size = len(queue)  # Snapshot!
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            # ...
        result.append(level)


# ❌ WRONG: Using list instead of deque (O(n) popleft)
queue = []
queue.append(node)
node = queue.pop(0)  # O(n) operation!

# ✅ CORRECT: Use deque for O(1) operations
from collections import deque
queue = deque()
queue.append(node)
node = queue.popleft()  # O(1) operation
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"I'll use BFS because we need the shortest path in an unweighted graph.
BFS guarantees that the first time we reach a node, it's via the
shortest path, since it explores nodes level by level."
```

### 2. What Interviewers Look For
- **Pattern recognition**: Quickly identify BFS problems
- **Correct visited tracking**: Mark when adding, not processing
- **Edge cases**: Empty input, unreachable target, cycles
- **Space optimization**: Can we modify input instead of visited set?

### 3. Common Follow-up Questions
- "Can you do this with DFS?" → Usually yes, but BFS is better for shortest path
- "What if edges have weights?" → Use Dijkstra's algorithm
- "Can you optimize space?" → Modify grid in-place if allowed
- "What's the time complexity?" → O(V + E) or O(m × n) for grids

### 4. Edge Cases to Always Consider
```python
# Empty input
if not grid or not grid[0]:
    return -1

# Start equals end
if start == end:
    return 0

# Start or end is blocked
if grid[start[0]][start[1]] == 1 or grid[end[0]][end[1]] == 1:
    return -1

# No path exists
# BFS will exhaust queue without finding target → return -1
```

---

## Practice Problems Checklist

| # | Problem | Difficulty | Pattern | Status |
|---|---------|------------|---------|--------|
| 1 | Level Order (LC 102) | Medium | Tree BFS | ☐ |
| 2 | Number of Islands (LC 200) | Medium | Grid BFS | ☐ |
| 3 | Rotting Oranges (LC 994) | Medium | Multi-source | ☐ |
| 4 | Word Ladder (LC 127) | Hard | State-space | ☐ |
| 5 | Shortest Binary Matrix (LC 1091) | Medium | Grid BFS | ☐ |
| 6 | 01 Matrix (LC 542) | Medium | Multi-source | ☐ |
| 7 | Open the Lock (LC 752) | Medium | State-space | ☐ |
| 8 | Walls and Gates (LC 286) | Medium | Multi-source | ☐ |
| 9 | Pacific Atlantic (LC 417) | Medium | Reverse BFS | ☐ |
| 10 | Clone Graph (LC 133) | Medium | Graph BFS | ☐ |

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
