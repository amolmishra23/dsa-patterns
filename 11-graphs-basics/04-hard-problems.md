# Graphs Basics - Hard Problems

## Problem 1: Alien Dictionary (LC #269) - Hard

- [LeetCode](https://leetcode.com/problems/alien-dictionary/)

### Video Explanation
- [NeetCode - Alien Dictionary](https://www.youtube.com/watch?v=6kTZYvNNyps)

### Problem Statement
Derive order of letters in alien language from sorted dictionary.


### Visual Intuition
```
Alien Dictionary
words = ["wrt","wrf","er","ett","rftt"]

Extract ordering from adjacent words:
  wrt vs wrf: t < f
  wrf vs er:  w < e
  er vs ett:  r < t
  ett vs rftt: e < r

Build graph: w→e, e→r, r→t, t→f

Topological sort (Kahn's algorithm):
  indegree: w=0, e=1, r=1, t=1, f=1
  
  Queue: [w]
  Process w → result="w", reduce e's indegree
  Queue: [e]
  Process e → result="we", reduce r's indegree
  ...
  
Answer: "wertf"
```

### Solution
```python
from collections import defaultdict, deque

def alienOrder(words: list[str]) -> str:
    """
    Topological sort to find character order.

    Time: O(C) where C = total chars
    Space: O(U) where U = unique chars
    """
    # Build graph
    graph = defaultdict(set)
    in_degree = {c: 0 for word in words for c in word}

    # Compare adjacent words
    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]

        # Check invalid case: prefix longer than word
        if len(w1) > len(w2) and w1.startswith(w2):
            return ""

        # Find first different character
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break

    # Topological sort using BFS
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []

    while queue:
        c = queue.popleft()
        result.append(c)

        for neighbor in graph[c]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Check for cycle
    if len(result) != len(in_degree):
        return ""

    return "".join(result)
```

### Edge Cases
- Single word → return that word's characters
- Invalid order (cycle) → return ""
- Prefix issue → return ""
- Multiple valid orders → any topological order

---

## Problem 2: Reconstruct Itinerary (LC #332) - Hard

- [LeetCode](https://leetcode.com/problems/reconstruct-itinerary/)

### Video Explanation
- [NeetCode - Reconstruct Itinerary](https://www.youtube.com/watch?v=ZyB_gQ8vqGA)

### Problem Statement
Reconstruct itinerary in lexical order using all tickets.


### Visual Intuition
```
Reconstruct Itinerary (Eulerian Path)
tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]

Build adjacency list (sorted):
  JFK: [MUC]
  MUC: [LHR]
  LHR: [SFO]
  SFO: [SJC]

Hierholzer's algorithm (DFS post-order):
  Start JFK → MUC → LHR → SFO → SJC (dead end)
  Backtrack, add to result: [SJC]
  SFO done, add: [SFO, SJC]
  LHR done, add: [LHR, SFO, SJC]
  ...
  
Result (reversed): ["JFK","MUC","LHR","SFO","SJC"]
```

### Solution
```python
from collections import defaultdict

def findItinerary(tickets: list[list[str]]) -> list[str]:
    """
    Hierholzer's algorithm for Eulerian path.

    Time: O(E log E) for sorting
    Space: O(E)
    """
    graph = defaultdict(list)

    # Sort in reverse for efficient pop
    for src, dst in sorted(tickets, reverse=True):
        graph[src].append(dst)

    result = []

    def dfs(airport):
        while graph[airport]:
            dfs(graph[airport].pop())
        result.append(airport)

    dfs("JFK")
    return result[::-1]
```

### Edge Cases
- Single ticket → return [JFK, destination]
- Multiple same destinations → lexical order
- All tickets used → guaranteed per problem
- Stuck airport → backtrack needed

---

## Problem 3: Critical Connections (LC #1192) - Hard

- [LeetCode](https://leetcode.com/problems/critical-connections-in-a-network/)

### Video Explanation
- [NeetCode - Critical Connections](https://www.youtube.com/watch?v=mKUsbABiwBI)

### Problem Statement
Find all critical connections (bridges) in a network.


### Visual Intuition
```
Critical Connections (Bridges in Graph)
n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]

    0 --- 1 --- 3
     \   /
      \ /
       2

Tarjan's algorithm - find bridges:
  DFS with discovery time (disc) and low-link (low)
  
  disc: [0, 1, 2, 3]
  low:  [0, 0, 0, 3]
  
  Edge (u,v) is bridge if low[v] > disc[u]
  
  Check (1,3): low[3]=3 > disc[1]=1 → BRIDGE!
  Check (0,1): low[1]=0 ≤ disc[0]=0 → not bridge
  
Critical connection: [[1,3]]
```

### Solution
```python
from collections import defaultdict

def criticalConnections(n: int, connections: list[list[int]]) -> list[list[int]]:
    """
    Tarjan's algorithm to find bridges.

    Time: O(V + E)
    Space: O(V + E)
    """
    graph = defaultdict(list)
    for u, v in connections:
        graph[u].append(v)
        graph[v].append(u)

    disc = [0] * n      # Discovery time
    low = [0] * n       # Lowest reachable
    visited = [False] * n
    bridges = []
    time = [0]

    def dfs(node, parent):
        visited[node] = True
        disc[node] = low[node] = time[0]
        time[0] += 1

        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor, node)
                low[node] = min(low[node], low[neighbor])

                # Bridge condition
                if low[neighbor] > disc[node]:
                    bridges.append([node, neighbor])
            elif neighbor != parent:
                low[node] = min(low[node], disc[neighbor])

    for i in range(n):
        if not visited[i]:
            dfs(i, -1)

    return bridges
```

### Edge Cases
- No bridges → return []
- All bridges → return all edges
- Single node → return []
- Tree → all edges are bridges

---

## Problem 4: Shortest Path with Alternating Colors (LC #1129) - Hard

- [LeetCode](https://leetcode.com/problems/shortest-path-with-alternating-colors/)

### Video Explanation
- [NeetCode - Shortest Path with Alternating Colors](https://www.youtube.com/watch?v=69rcy5BQNLQ)

### Problem Statement
Find shortest path from node 0 to each node using alternating red/blue edges.

### Visual Intuition
```
Shortest Path with Alternating Colors
edges: red[(0,1),(1,2)], blue[(1,2),(2,3)]

BFS with state = (node, last_color)
Start: (0, none) dist=0
       
  0 --red--> 1 --blue--> 2 --red--> ?
  0 --blue-> ? (no blue from 0)

(0,_) → (1,red) dist=1 → (2,blue) dist=2 → (3,red) dist=3

Answer: [0, 1, 2, 3] (shortest to each node)
Track visited by (node, color) to allow both colors
```


### Intuition
```
Graph with red and blue edges:
0 --red--> 1 --blue--> 2 --red--> 3

Valid path: must alternate colors.
0 →(red)→ 1 →(blue)→ 2 ✓
0 →(red)→ 1 →(red)→ 2 ✗ (same color)

BFS with state = (node, last_color)
```

### Solution
```python
from collections import defaultdict, deque

def shortestAlternatingPaths(n: int, redEdges: list[list[int]],
                             blueEdges: list[list[int]]) -> list[int]:
    """
    BFS with color state tracking.

    Strategy:
    - State = (node, last_edge_color)
    - BFS from node 0 with both colors as starting options
    - Track visited (node, color) pairs

    Time: O(V + E)
    Space: O(V + E)
    """
    # Build adjacency lists by color
    # 0 = red, 1 = blue
    graph = defaultdict(lambda: defaultdict(list))

    for u, v in redEdges:
        graph[u][0].append(v)
    for u, v in blueEdges:
        graph[u][1].append(v)

    # Result array, -1 means unreachable
    result = [-1] * n
    result[0] = 0

    # BFS: (node, last_color, distance)
    # Start with both colors from node 0
    queue = deque([(0, 0, 0), (0, 1, 0)])  # (node, color, dist)
    visited = {(0, 0), (0, 1)}

    while queue:
        node, last_color, dist = queue.popleft()

        # Try opposite color edges
        next_color = 1 - last_color

        for neighbor in graph[node][next_color]:
            if (neighbor, next_color) not in visited:
                visited.add((neighbor, next_color))
                queue.append((neighbor, next_color, dist + 1))

                # Update result if first time reaching this node
                if result[neighbor] == -1:
                    result[neighbor] = dist + 1

    return result
```

### Complexity
- **Time**: O(V + E)
- **Space**: O(V + E)

### Edge Cases
- Node 0 → always 0
- No path with alternating → -1
- Self-loops → may provide color options
- Both colors reach same node → take minimum

---

## Problem 5: Minimum Knight Moves (LC #1197) - Hard

- [LeetCode](https://leetcode.com/problems/minimum-knight-moves/)

### Video Explanation
- [NeetCode - Minimum Knight Moves](https://www.youtube.com/watch?v=pjyJ-FqPtqE)

### Problem Statement
Find minimum moves for a knight to reach (x, y) from (0, 0) on infinite chessboard.

### Visual Intuition
```
Minimum Knight Moves to (x,y) from (0,0)

Knight moves: 8 L-shaped jumps
    . 2 . 2 .
    2 . . . 2
    . . K . .
    2 . . . 2
    . 2 . 2 .

BFS from (0,0), exploit symmetry: work in quadrant x≥0, y≥0
To reach (2,1): (0,0)→(1,2)→(2,0)→... or (0,0)→(2,1) = 1 move!

Optimization: A* or bidirectional BFS for large coordinates
```


### Intuition
```
Knight moves: 8 possible L-shaped moves
(+2,+1), (+2,-1), (-2,+1), (-2,-1)
(+1,+2), (+1,-2), (-1,+2), (-1,-2)

BFS from (0,0) to (x,y).
Optimization: Use symmetry - only search in first quadrant.
```

### Solution
```python
from collections import deque

def minKnightMoves(x: int, y: int) -> int:
    """
    BFS with symmetry optimization.

    Strategy:
    - Use absolute values (symmetry)
    - BFS from origin
    - Prune search space with bounds

    Time: O(max(x,y)²)
    Space: O(max(x,y)²)
    """
    # Use symmetry - work in first quadrant
    x, y = abs(x), abs(y)

    # Special cases
    if x == 0 and y == 0:
        return 0

    # Knight moves
    moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
             (1, 2), (1, -2), (-1, 2), (-1, -2)]

    queue = deque([(0, 0, 0)])  # (x, y, steps)
    visited = {(0, 0)}

    while queue:
        cx, cy, steps = queue.popleft()

        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy

            if (nx, ny) == (x, y):
                return steps + 1

            # Prune: don't go too far from target
            # Allow some negative to handle edge cases near origin
            if (nx, ny) not in visited and -2 <= nx <= x + 2 and -2 <= ny <= y + 2:
                visited.add((nx, ny))
                queue.append((nx, ny, steps + 1))

    return -1
```

### Bidirectional BFS (Optimized)
```python
def minKnightMoves(x: int, y: int) -> int:
    """
    Bidirectional BFS for faster convergence.
    """
    x, y = abs(x), abs(y)

    if x == 0 and y == 0:
        return 0

    moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
             (1, 2), (1, -2), (-1, 2), (-1, -2)]

    # Two frontiers
    front = {(0, 0)}
    back = {(x, y)}
    visited = {(0, 0), (x, y)}
    steps = 0

    while front and back:
        # Always expand smaller frontier
        if len(front) > len(back):
            front, back = back, front

        next_front = set()

        for cx, cy in front:
            for dx, dy in moves:
                nx, ny = cx + dx, cy + dy

                if (nx, ny) in back:
                    return steps + 1

                if (nx, ny) not in visited and -2 <= nx <= x + 2 and -2 <= ny <= y + 2:
                    visited.add((nx, ny))
                    next_front.add((nx, ny))

        front = next_front
        steps += 1

    return -1
```

### Complexity
- **Time**: O(max(x,y)²) for BFS, O(√(x² + y²)) for bidirectional
- **Space**: O(max(x,y)²)

### Edge Cases
- Target is (0,0) → return 0
- Target is (1,1) → return 2
- Negative coordinates → use absolute values
- Large coordinates → bidirectional BFS helps

---

## Problem 6: Word Ladder II (LC #126) - Hard

- [LeetCode](https://leetcode.com/problems/word-ladder-ii/)

### Video Explanation
- [NeetCode - Word Ladder II](https://www.youtube.com/watch?v=AD4SFl7tu7I)

### Problem Statement
Find all shortest transformation sequences from beginWord to endWord.

### Visual Intuition
```
Word Ladder II - Find ALL shortest transformation paths
begin="hit", end="cog", words=["hot","dot","dog","lot","log","cog"]

BFS builds level graph:
Level 0: hit
Level 1: hot (hit→hot)
Level 2: dot, lot (hot→dot, hot→lot)  
Level 3: dog, log (dot→dog, lot→log)
Level 4: cog (dog→cog, log→cog)

Backtrack from end to find all paths:
cog←dog←dot←hot←hit
cog←log←lot←hot←hit
```


### Intuition
```
beginWord = "hit", endWord = "cog"
wordList = ["hot","dot","dog","lot","log","cog"]

BFS for shortest path length, then DFS to collect all paths.

hit → hot → dot → dog → cog
hit → hot → lot → log → cog
```

### Solution
```python
from collections import defaultdict, deque

def findLadders(beginWord: str, endWord: str,
                wordList: list[str]) -> list[list[str]]:
    """
    BFS to build graph, DFS to find all paths.

    Strategy:
    1. BFS from beginWord, track parent pointers
    2. DFS backwards from endWord to collect paths

    Time: O(N * L² + paths)
    Space: O(N * L)
    """
    word_set = set(wordList)
    if endWord not in word_set:
        return []

    # BFS to find shortest path and build parent graph
    # parent[word] = set of words that can transform to it
    parent = defaultdict(set)

    # Level tracking for BFS
    current_level = {beginWord}
    found = False

    while current_level and not found:
        # Remove current level from word_set to prevent cycles
        word_set -= current_level
        next_level = set()

        for word in current_level:
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + c + word[i+1:]

                    if next_word in word_set:
                        next_level.add(next_word)
                        parent[next_word].add(word)

                        if next_word == endWord:
                            found = True

        current_level = next_level

    # DFS to collect all paths
    result = []

    def dfs(word, path):
        if word == beginWord:
            result.append(path[::-1])
            return

        for prev_word in parent[word]:
            dfs(prev_word, path + [prev_word])

    if found:
        dfs(endWord, [endWord])

    return result
```

### Complexity
- **Time**: O(N * L² + number of paths * L)
- **Space**: O(N * L) for parent graph

### Edge Cases
- endWord not in list → return []
- beginWord == endWord → return [[beginWord]]
- No path exists → return []
- Multiple shortest paths → return all

---

## Summary

| # | Problem | Key Technique |
|---|---------|---------------|
| 1 | Alien Dictionary | Topological sort |
| 2 | Reconstruct Itinerary | Hierholzer's algorithm |
| 3 | Critical Connections | Tarjan's bridges |
| 4 | Alternating Colors | BFS with color state |
| 5 | Minimum Knight Moves | BFS + symmetry/bidirectional |
| 6 | Word Ladder II | BFS + DFS path collection |
