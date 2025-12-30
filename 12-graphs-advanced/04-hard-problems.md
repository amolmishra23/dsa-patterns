# Union-Find - Advanced Problems

## Union-Find Fundamentals

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNION-FIND DATA STRUCTURE                                │
│                                                                             │
│  Also called: Disjoint Set Union (DSU)                                      │
│                                                                             │
│  Operations:                                                                │
│  • find(x): Find root of x's set - O(α(n)) ≈ O(1)                          │
│  • union(x, y): Merge sets containing x and y - O(α(n)) ≈ O(1)             │
│                                                                             │
│  Optimizations:                                                             │
│  • Path Compression: Make nodes point directly to root                      │
│  • Union by Rank/Size: Attach smaller tree under larger                     │
│                                                                             │
│  Use Cases:                                                                 │
│  • Connected components                                                     │
│  • Cycle detection                                                          │
│  • Kruskal's MST                                                            │
│  • Dynamic connectivity                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Union-Find Template

```python
class UnionFind:
    """
    Union-Find with path compression and union by rank.

    Time: O(α(n)) per operation, where α is inverse Ackermann
    Space: O(n)
    """

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n  # Number of components

    def find(self, x: int) -> int:
        """Find root with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """Union by rank. Returns True if merged, False if already same set."""
        px, py = self.find(x), self.find(y)

        if px == py:
            return False

        # Attach smaller tree under larger
        if self.rank[px] < self.rank[py]:
            px, py = py, px

        self.parent[py] = px

        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        self.count -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """Check if x and y are in same set."""
        return self.find(x) == self.find(y)

    def get_count(self) -> int:
        """Return number of disjoint sets."""
        return self.count
```

---

## Problem 1: Number of Connected Components (LC #323) - Medium

- [LeetCode](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

### Video Explanation
- [NeetCode - Number of Connected Components](https://www.youtube.com/watch?v=8f1XPm4WOUc)

### Problem Statement
Count connected components in undirected graph.


### Visual Intuition
```
Number of Connected Components (Union-Find)
n = 5, edges = [[0,1], [1,2], [3,4]]

Initial: each node is its own component
  0   1   2   3   4  (5 components)

Union(0,1):
  0-1   2   3   4  (4 components)

Union(1,2):
  0-1-2   3   4  (3 components)

Union(3,4):
  0-1-2   3-4  (2 components)

parent: [0, 0, 0, 3, 3]
        0→0, 1→0, 2→0, 3→3, 4→3

Answer: 2 components
```

### Solution
```python
def countComponents(n: int, edges: list[list[int]]) -> int:
    """
    Count connected components using Union-Find.

    Time: O(E * α(V))
    Space: O(V)
    """
    uf = UnionFind(n)

    for a, b in edges:
        uf.union(a, b)

    return uf.get_count()
```

### Edge Cases
- No edges → n components
- All nodes connected → 1 component
- Self-loops → don't change count
- Disconnected graph → multiple components

---

## Problem 2: Redundant Connection (LC #684) - Medium

- [LeetCode](https://leetcode.com/problems/redundant-connection/)

### Video Explanation
- [NeetCode - Redundant Connection](https://www.youtube.com/watch?v=FXWRE67PLL0)

### Problem Statement
Find edge that creates a cycle in undirected graph.


### Visual Intuition
```
Redundant Connection (find edge creating cycle)
edges = [[1,2], [1,3], [2,3]]

Union-Find: add edges until cycle detected

Add [1,2]: union(1,2)
  1-2   3

Add [1,3]: union(1,3)
  1-2
   \
    3

Add [2,3]: find(2)=1, find(3)=1 → SAME ROOT!
  Adding this edge creates cycle!

    1
   / \
  2---3  ← redundant edge [2,3]

Answer: [2,3]
```

### Solution
```python
def findRedundantConnection(edges: list[list[int]]) -> list[int]:
    """
    Find edge that creates cycle.

    Strategy:
    - Process edges one by one
    - If union returns False (already connected), this edge creates cycle

    Time: O(E * α(V))
    Space: O(V)
    """
    n = len(edges)
    uf = UnionFind(n + 1)  # 1-indexed nodes

    for a, b in edges:
        if not uf.union(a, b):
            return [a, b]

    return []
```

### Edge Cases
- Tree (no cycle) → return [] (shouldn't happen per problem)
- Multiple cycles → return last edge creating cycle
- Single edge → that edge if creates cycle
- Self-loop → that edge

---

## Problem 3: Accounts Merge (LC #721) - Medium

- [LeetCode](https://leetcode.com/problems/accounts-merge/)

### Video Explanation
- [NeetCode - Accounts Merge](https://www.youtube.com/watch?v=6st4IxEF-90)

### Problem Statement
Merge accounts that share common emails.


### Visual Intuition
```
Accounts Merge
accounts = [["John","john@mail.com","john_work@mail.com"],
            ["John","john@mail.com","john_home@mail.com"],
            ["Mary","mary@mail.com"]]

Union-Find on emails:
Account 0: union(john@mail, john_work@)
Account 1: union(john@mail, john_home@)
  john@mail already seen → same person!

Groups:
  {john@mail, john_work@, john_home@} → John
  {mary@mail} → Mary

Result: [["John","john@mail.com","john_home@mail.com","john_work@mail.com"],
         ["Mary","mary@mail.com"]]
```

### Solution
```python
from collections import defaultdict

def accountsMerge(accounts: list[list[str]]) -> list[list[str]]:
    """
    Merge accounts using Union-Find on emails.

    Strategy:
    - Map each email to an index
    - Union emails belonging to same account
    - Group emails by their root

    Time: O(n * k * α(nk)) where k = avg emails per account
    Space: O(nk)
    """
    email_to_idx = {}
    email_to_name = {}
    idx = 0

    # Assign index to each unique email
    for account in accounts:
        name = account[0]
        for email in account[1:]:
            if email not in email_to_idx:
                email_to_idx[email] = idx
                email_to_name[email] = name
                idx += 1

    # Union-Find on email indices
    uf = UnionFind(idx)

    for account in accounts:
        first_idx = email_to_idx[account[1]]
        for email in account[2:]:
            uf.union(first_idx, email_to_idx[email])

    # Group emails by root
    root_to_emails = defaultdict(list)
    for email, i in email_to_idx.items():
        root = uf.find(i)
        root_to_emails[root].append(email)

    # Build result
    result = []
    for emails in root_to_emails.values():
        name = email_to_name[emails[0]]
        result.append([name] + sorted(emails))

    return result
```

### Edge Cases
- Single account → return as-is
- No shared emails → separate accounts
- All share one email → merge all
- Duplicate emails in same account → handle correctly

---

## Problem 4: Satisfiability of Equality Equations (LC #990) - Medium

- [LeetCode](https://leetcode.com/problems/satisfiability-of-equality-equations/)

### Video Explanation
- [NeetCode - Satisfiability of Equality Equations](https://www.youtube.com/watch?v=gxdFcsNLPRg)

### Problem Statement
Check if equality/inequality equations are satisfiable.


### Visual Intuition
```
Satisfiability of Equality Equations
equations = ["a==b", "b!=a"]

Step 1: Process "==" (union equal variables)
  a==b: union(a, b) → a-b same group

Step 2: Check "!=" (must be different groups)
  b!=a: find(b)=a, find(a)=a → SAME GROUP!
  Contradiction! Return false

Example that works: ["a==b", "b!=c"]
  union(a,b): a-b
  check b!=c: find(b)=a, find(c)=c → different ✓
  Return true
```

### Solution
```python
def equationsPossible(equations: list[str]) -> bool:
    """
    Check equation satisfiability using Union-Find.

    Strategy:
    - First process all == equations (union variables)
    - Then check != equations (should not be in same set)

    Time: O(n * α(26))
    Space: O(26)
    """
    uf = UnionFind(26)

    # Process equalities first
    for eq in equations:
        if eq[1] == '=':
            x = ord(eq[0]) - ord('a')
            y = ord(eq[3]) - ord('a')
            uf.union(x, y)

    # Check inequalities
    for eq in equations:
        if eq[1] == '!':
            x = ord(eq[0]) - ord('a')
            y = ord(eq[3]) - ord('a')
            if uf.connected(x, y):
                return False

    return True
```

### Edge Cases
- No equations → return True
- Only equalities → always True
- a!=a → return False
- Transitive equality → a==b, b==c implies a==c

---

## Problem 5: Number of Islands II (LC #305) - Hard

- [LeetCode](https://leetcode.com/problems/number-of-islands-ii/)

### Video Explanation
- [NeetCode - Number of Islands II](https://www.youtube.com/watch?v=_Vkqk8n4NJA)

### Problem Statement
Count islands after each land addition.


### Visual Intuition
```
Number of Islands II (dynamic island creation)
m=3, n=3, positions = [[0,0],[0,1],[1,2],[2,1]]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Union-Find tracks connected components dynamically
             Add land → union with neighbors → update count
═══════════════════════════════════════════════════════════════

Step 1: Add land at (0,0)
─────────────────────────
  Before:          After:
  . . .            █ . .
  . . .    →       . . .
  . . .            . . .

  Check neighbors: none are land
  count = 0 + 1 = 1
  parent[(0,0)] = (0,0)

  Result: [1]

Step 2: Add land at (0,1)
─────────────────────────
  Before:          After:
  █ . .            █ █ .
  . . .    →       . . .
  . . .            . . .

  Check neighbors:
    ← (0,0) is land! → UNION

  Union-Find state:
    parent[(0,0)] = (0,0)
    parent[(0,1)] = (0,0) ← merged!

  count = 1 + 1 - 1 = 1 (added 1, merged 1)

  Result: [1, 1]

Step 3: Add land at (1,2)
─────────────────────────
  Before:          After:
  █ █ .            █ █ .
  . . .    →       . . █
  . . .            . . .

  Check neighbors:
    ↑ (0,2) is water
    ← (1,1) is water
  No land neighbors → new island!

  count = 1 + 1 = 2

  Result: [1, 1, 2]

Step 4: Add land at (2,1)
─────────────────────────
  Before:          After:
  █ █ .            █ █ .
  . . █    →       . . █
  . . .            . █ .

  Check neighbors:
    ↑ (1,1) is water
    ← (2,0) is water
    → (2,2) is water
  No land neighbors → new island!

  count = 2 + 1 = 3

  Final Grid:
  █ █ .    Island 1: {(0,0), (0,1)}
  . . █    Island 2: {(1,2)}
  . █ .    Island 3: {(2,1)}

  Result: [1, 1, 2, 3]

WHY THIS WORKS:
════════════════
● Union-Find tracks connected components efficiently
● Adding land: +1 island initially
● Union with neighbor: -1 island (merging)
● Final count = additions - merges
● α(mn) amortized time per operation
```

### Solution
```python
def numIslands2(m: int, n: int, positions: list[list[int]]) -> list[int]:
    """
    Dynamic island counting using Union-Find.

    Strategy:
    - Add land one by one
    - Union with adjacent land cells
    - Track component count

    Time: O(k * α(mn)) where k = positions
    Space: O(mn)
    """
    def get_index(r, c):
        return r * n + c

    parent = {}
    rank = {}
    count = 0
    result = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        nonlocal count
        px, py = find(x), find(y)
        if px == py:
            return

        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        count -= 1

    for r, c in positions:
        idx = get_index(r, c)

        if idx in parent:
            # Already land
            result.append(count)
            continue

        # Add new land
        parent[idx] = idx
        rank[idx] = 0
        count += 1

        # Union with adjacent land
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            n_idx = get_index(nr, nc)

            if 0 <= nr < m and 0 <= nc < n and n_idx in parent:
                union(idx, n_idx)

        result.append(count)

    return result
```

### Edge Cases
- Duplicate positions → count stays same
- All same position → count stays 1
- No adjacent land → count increases
- All positions adjacent → final count is 1

---

## Problem 6: Longest Consecutive Sequence (LC #128) - Medium

- [LeetCode](https://leetcode.com/problems/longest-consecutive-sequence/)

### Video Explanation
- [NeetCode - Longest Consecutive Sequence](https://www.youtube.com/watch?v=P6RZZMu_maU)

### Problem Statement
Find longest consecutive sequence in O(n).


### Visual Intuition
```
Longest Consecutive Sequence
nums = [100, 4, 200, 1, 3, 2]

Union-Find: union consecutive numbers
  For each num, check if num-1 or num+1 exists

Process 100: no 99 or 101
Process 4: no 3 yet, no 5
Process 200: no 199 or 201
Process 1: no 0, no 2 yet
Process 3: union(3,4) → {3,4}
Process 2: union(2,1), union(2,3) → {1,2,3,4}

Groups: {100}, {200}, {1,2,3,4}

Largest component size = 4
Answer: 4
```

### Solution
```python
def longestConsecutive(nums: list[int]) -> int:
    """
    Find longest consecutive sequence using Union-Find.

    Alternative to hash set approach.

    Time: O(n * α(n))
    Space: O(n)
    """
    if not nums:
        return 0

    num_to_idx = {num: i for i, num in enumerate(nums)}
    uf = UnionFind(len(nums))

    for num in nums:
        # Union with consecutive numbers if they exist
        if num - 1 in num_to_idx:
            uf.union(num_to_idx[num], num_to_idx[num - 1])
        if num + 1 in num_to_idx:
            uf.union(num_to_idx[num], num_to_idx[num + 1])

    # Count elements in each component
    from collections import Counter
    root_count = Counter(uf.find(i) for i in range(len(nums)))

    return max(root_count.values())
```

### Edge Cases
- Empty array → return 0
- Single element → return 1
- All same number → return 1
- Duplicates → handle via map

---

## Problem 7: Most Stones Removed (LC #947) - Medium

- [LeetCode](https://leetcode.com/problems/most-stones-removed/)

### Video Explanation
- [NeetCode - Most Stones Removed](https://www.youtube.com/watch?v=5en2esX_Fgc)

### Problem Statement
Remove maximum stones where each removal shares row/col with another.


### Visual Intuition
```
Most Stones Removed (same row/col = connected)
stones = [[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]]

Grid:
  0 1 2
0 X X .
1 X . X
2 . X X

Union stones sharing row or column:
  (0,0)-(0,1) same row
  (0,0)-(1,0) same col
  (1,2)-(2,2) same col
  (2,1)-(2,2) same row
  ...

All connected → 1 component
Removable = total - components = 6 - 1 = 5

Keep 1 stone per component, remove rest
```

### Solution
```python
def removeStones(stones: list[list[int]]) -> int:
    """
    Maximum stones removed = total - number of connected components.

    Strategy:
    - Stones sharing row or column are in same component
    - Use Union-Find with row/col as nodes

    Time: O(n * α(n))
    Space: O(n)
    """
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for r, c in stones:
        # Use ~c to distinguish row from column
        union(r, ~c)

    # Count unique roots
    roots = set(find(r) for r, c in stones)

    return len(stones) - len(roots)
```

### Edge Cases
- Single stone → return 0
- No shared row/col → return 0
- All share row or col → remove all but 1
- Grid pattern → depends on connectivity

---

## Problem 8: Smallest String With Swaps (LC #1202) - Medium

- [LeetCode](https://leetcode.com/problems/smallest-string-with-swaps/)

### Video Explanation
- [NeetCode - Smallest String With Swaps](https://www.youtube.com/watch?v=8kptxaOALo0)

### Problem Statement
Find lexicographically smallest string after allowed swaps.


### Visual Intuition
```
Smallest String With Swaps
s = "dcab", pairs = [[0,3],[1,2]]

Union indices that can swap:
  Group 1: {0,3} → chars 'd','b'
  Group 2: {1,2} → chars 'c','a'

Sort chars within each group:
  Group 1: ['b','d']
  Group 2: ['a','c']

Assign sorted chars back to positions:
  pos 0 (group 1): 'b'
  pos 1 (group 2): 'a'
  pos 2 (group 2): 'c'
  pos 3 (group 1): 'd'

Result: "bacd"
```

### Solution
```python
def smallestStringWithSwaps(s: str, pairs: list[list[int]]) -> str:
    """
    Smallest string using Union-Find.

    Strategy:
    - Characters in same component can be rearranged freely
    - Sort characters within each component

    Time: O(n log n)
    Space: O(n)
    """
    n = len(s)
    uf = UnionFind(n)

    for a, b in pairs:
        uf.union(a, b)

    # Group indices by root
    from collections import defaultdict
    groups = defaultdict(list)

    for i in range(n):
        groups[uf.find(i)].append(i)

    # Build result
    result = list(s)

    for indices in groups.values():
        # Sort characters at these indices
        chars = sorted(result[i] for i in indices)

        # Place sorted characters back
        for i, idx in enumerate(sorted(indices)):
            result[idx] = chars[i]

    return ''.join(result)
```

### Edge Cases
- No pairs → return original string
- All indices connected → sort entire string
- Single character → return as-is
- Already smallest → no change needed

---

## Problem 9: Regions Cut By Slashes (LC #959) - Medium

- [LeetCode](https://leetcode.com/problems/regions-cut-by-slashes/)

### Video Explanation
- [NeetCode - Regions Cut By Slashes](https://www.youtube.com/watch?v=n3s9Q7GtfB4)

### Problem Statement
Count regions in grid with '/', '\', ' ' characters.


### Visual Intuition
```
Regions Cut By Slashes
grid = [" /","/ "]

Upscale 1x1 to 3x3:
  " /" →  . . X     "/" →  X . .
          . X .            . X .
          X . .            . . X

Combined 6x6 grid:
  . . X X . .
  . X . . X .
  X . . . . X
  X . . . . X
  . X . . X .
  . . X X . .

Count connected regions of '.' using Union-Find or DFS
Answer: 2 regions
```

### Solution
```python
def regionsBySlashes(grid: list[str]) -> int:
    """
    Count regions using Union-Find on upscaled grid.

    Strategy:
    - Scale each cell to 3x3
    - '/' and '\' become diagonal lines
    - Count connected components of 0s

    Time: O(n² * α(n²))
    Space: O(n²)
    """
    n = len(grid)
    size = n * 3

    # Create upscaled grid
    upscaled = [[0] * size for _ in range(size)]

    for r in range(n):
        for c in range(n):
            if grid[r][c] == '/':
                upscaled[r * 3][c * 3 + 2] = 1
                upscaled[r * 3 + 1][c * 3 + 1] = 1
                upscaled[r * 3 + 2][c * 3] = 1
            elif grid[r][c] == '\\':
                upscaled[r * 3][c * 3] = 1
                upscaled[r * 3 + 1][c * 3 + 1] = 1
                upscaled[r * 3 + 2][c * 3 + 2] = 1

    # Count connected components of 0s using DFS
    def dfs(r, c):
        if r < 0 or r >= size or c < 0 or c >= size or upscaled[r][c] == 1:
            return
        upscaled[r][c] = 1
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    regions = 0
    for r in range(size):
        for c in range(size):
            if upscaled[r][c] == 0:
                dfs(r, c)
                regions += 1

    return regions
```

### Edge Cases
- All spaces → 1 region
- All slashes → maximum regions
- 1x1 grid → 1 or 2 regions
- Mixed slashes → complex regions

---

## Problem 10: Making A Large Island (LC #827) - Hard

- [LeetCode](https://leetcode.com/problems/making-a-large-island/)

### Video Explanation
- [NeetCode - Making A Large Island](https://www.youtube.com/watch?v=lgiz0x0U9eE)

### Problem Statement
Find largest island after changing at most one 0 to 1.


### Visual Intuition
```
Making A Large Island (flip one 0 to 1)
grid = [[1,0],[0,1]]

Step 1: Label islands and compute sizes
  1₁ 0       sizes: {1: 1, 2: 1}
  0  1₂

Step 2: For each 0, check unique adjacent islands
  (0,1): neighbors = {1₁, 1₂}
         potential = 1 + size[1] + size[2] = 1+1+1 = 3

  (1,0): neighbors = {1₁, 1₂}
         potential = 1 + 1 + 1 = 3

Max = 3 (flip either 0)

Result grid (flipping (0,1)):
  1 1
  0 1  → size = 3
```

### Solution
```python
def largestIsland(grid: list[list[int]]) -> int:
    """
    Find largest island after one flip using Union-Find.

    Strategy:
    - Label each island with unique ID
    - Track size of each island
    - For each 0, calculate potential size by merging adjacent islands

    Time: O(n²)
    Space: O(n²)
    """
    n = len(grid)

    # Label islands and track sizes
    island_id = 2  # Start from 2 to distinguish from 0 and 1
    island_size = {0: 0}

    def dfs(r, c, id):
        if r < 0 or r >= n or c < 0 or c >= n or grid[r][c] != 1:
            return 0

        grid[r][c] = id
        size = 1
        size += dfs(r + 1, c, id)
        size += dfs(r - 1, c, id)
        size += dfs(r, c + 1, id)
        size += dfs(r, c - 1, id)
        return size

    # Label all islands
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                island_size[island_id] = dfs(r, c, island_id)
                island_id += 1

    # Find maximum island (without flipping)
    max_size = max(island_size.values()) if island_size else 0

    # Try flipping each 0
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for r in range(n):
        for c in range(n):
            if grid[r][c] == 0:
                # Get unique adjacent islands
                adjacent = set()
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n and 0 <= nc < n:
                        adjacent.add(grid[nr][nc])

                # Calculate potential size
                potential = 1 + sum(island_size.get(id, 0) for id in adjacent)
                max_size = max(max_size, potential)

    return max_size
```

### Edge Cases
- All 1s → return n²
- All 0s → return 1 (flip one)
- Single island → flip adjacent 0
- Multiple islands → merge via flip

---

## Summary: Union-Find Problems

| # | Problem | Key Insight | Time |
|---|---------|-------------|------|
| 1 | Connected Components | Basic counting | O(E * α(V)) |
| 2 | Redundant Connection | Cycle detection | O(E * α(V)) |
| 3 | Accounts Merge | Group by root | O(nk * α(nk)) |
| 4 | Equation Satisfiability | Two-pass (== then !=) | O(n) |
| 5 | Islands II | Dynamic connectivity | O(k * α(mn)) |
| 6 | Longest Consecutive | Union neighbors | O(n) |
| 7 | Most Stones Removed | Row/col as nodes | O(n * α(n)) |
| 8 | Smallest String Swaps | Sort within component | O(n log n) |
| 9 | Regions By Slashes | Upscale grid | O(n²) |
| 10 | Large Island | Label + merge | O(n²) |

---

## Practice More Problems

- [ ] LC #547 - Number of Provinces
- [ ] LC #685 - Redundant Connection II
- [ ] LC #765 - Couples Holding Hands
- [ ] LC #839 - Similar String Groups
- [ ] LC #1319 - Number of Operations to Make Network Connected

