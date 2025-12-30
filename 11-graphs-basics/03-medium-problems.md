# Graph BFS - Medium Problems

## When to Use BFS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BFS VS DFS FOR GRAPHS                                    │
│                                                                             │
│  USE BFS when:                                                              │
│  • Finding shortest path (unweighted)                                       │
│  • Level-by-level traversal needed                                          │
│  • Finding minimum steps/moves                                              │
│  • Exploring neighbors before going deeper                                  │
│                                                                             │
│  USE DFS when:                                                              │
│  • Finding any path (not necessarily shortest)                              │
│  • Detecting cycles                                                         │
│  • Topological sorting                                                      │
│  • Exploring all possibilities (backtracking)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem 1: Shortest Path in Binary Matrix (LC #1091) - Medium

- [LeetCode](https://leetcode.com/problems/shortest-path-in-binary-matrix/)

### Problem Statement
Given an `n x n` binary matrix `grid`, return the length of the shortest **clear path** from top-left to bottom-right. A clear path consists only of cells with value 0, and you can move in **8 directions** (including diagonals). Return -1 if no such path exists.

### Video Explanation
- [NeetCode - Shortest Path in Binary Matrix](https://www.youtube.com/watch?v=YnxUdAO7TAo)

### Examples
```
Input: grid = [[0,1],[1,0]]
Output: 2
Explanation: Path (0,0) → (1,1)

Input: grid = [[0,0,0],[1,1,0],[1,1,0]]
Output: 4
Explanation: (0,0) → (0,1) → (0,2) → (1,2) → (2,2)

Input: grid = [[1,0,0],[1,1,0],[1,1,0]]
Output: -1
Explanation: Start cell is blocked
```

### Intuition Development
```
BFS guarantees SHORTEST path in unweighted graphs!

grid:
  0 0 0
  1 1 0
  1 1 0

┌─────────────────────────────────────────────────────────────────┐
│ BFS explores level by level (distance by distance):            │
│                                                                  │
│ Level 0: (0,0)                                                  │
│ Level 1: (0,1), (1,0)❌ blocked                                 │
│ Level 2: (0,2), (1,1)❌ blocked                                 │
│ Level 3: (1,2)                                                  │
│ Level 4: (2,2) ✓ GOAL!                                          │
│                                                                  │
│ 8 directions allowed:                                            │
│   (-1,-1) (-1,0) (-1,1)                                         │
│   (0,-1)   (X)   (0,1)                                          │
│   (1,-1)  (1,0)  (1,1)                                          │
│                                                                  │
│ First time we reach goal = shortest path!                        │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
from collections import deque

def shortestPathBinaryMatrix(grid: list[list[int]]) -> int:
    """
    Find shortest path using BFS.

    Strategy:
    - BFS from (0,0) to (n-1,n-1)
    - 8 directions allowed
    - Track distance with each cell

    Time: O(n²)
    Space: O(n²)
    """
    n = len(grid)

    # Check start and end
    if grid[0][0] == 1 or grid[n-1][n-1] == 1:
        return -1

    # 8 directions
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    queue = deque([(0, 0, 1)])  # (row, col, distance)
    grid[0][0] = 1  # Mark visited

    while queue:
        r, c, dist = queue.popleft()

        if r == n - 1 and c == n - 1:
            return dist

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                grid[nr][nc] = 1  # Mark visited
                queue.append((nr, nc, dist + 1))

    return -1
```

### Complexity
- **Time**: O(n²) - Each cell visited once, 8 neighbors checked
- **Space**: O(n²) - Queue can hold all cells

### Edge Cases
- Start or end blocked: Return -1
- 1x1 grid with 0: Return 1
- No path exists: BFS exhausts all options, return -1

---

## Problem 2: Word Ladder (LC #127) - Hard

- [LeetCode](https://leetcode.com/problems/word-ladder/)

### Problem Statement
A **transformation sequence** from word `beginWord` to word `endWord` is a sequence where each adjacent pair differs by exactly one letter. Given a dictionary `wordList`, return the number of words in the **shortest** transformation sequence, or 0 if no such sequence exists. Note: `beginWord` does not need to be in `wordList`.

### Video Explanation
- [NeetCode - Word Ladder](https://www.youtube.com/watch?v=h9iTnkgv05E)

### Examples
```
Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
Output: 5
Explanation: hit → hot → dot → dog → cog

Input: beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"]
Output: 0
Explanation: "cog" is not in wordList

Input: beginWord = "a", endWord = "c", wordList = ["a","b","c"]
Output: 2
```

### Intuition Development
```
Model as graph: words are nodes, edges connect 1-letter-different words!

           hit
            ↓
           hot
          /   \
        dot   lot
         |     |
        dog   log
          \   /
           cog

BFS from "hit" to "cog" = shortest path!

┌─────────────────────────────────────────────────────────────────┐
│ Finding neighbors efficiently:                                   │
│                                                                  │
│ Naive: Compare each word with all others → O(N × M)             │
│ Better: For each position, try a-z → O(M × 26) per word         │
│                                                                  │
│ "hot" neighbors:                                                 │
│   *ot: aot, bot, cot, dot ✓, eot, ...                           │
│   h*t: hat, hbt, hct, hdt, het, hit ✓, ...                      │
│   ho*: hoa, hob, hoc, hod, hoe, ..., hop, hoq, hor, hos, hot    │
│                                                                  │
│ Bidirectional BFS: Start from both ends, meet in middle!        │
│ Much faster in practice (exponential to square root)            │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:
    """
    Find shortest transformation using BFS.

    Strategy:
    - BFS where each word is a node
    - Edge exists if words differ by one letter
    - Use set for O(1) word lookup

    Time: O(M² × N) where M = word length, N = word count
    Space: O(M × N)
    """
    word_set = set(wordList)

    if endWord not in word_set:
        return 0

    queue = deque([(beginWord, 1)])
    visited = {beginWord}

    while queue:
        word, length = queue.popleft()

        if word == endWord:
            return length

        # Try changing each character
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]

                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, length + 1))

    return 0


def ladderLength_bidirectional(beginWord: str, endWord: str, wordList: list[str]) -> int:
    """
    Optimized: Bidirectional BFS.

    Time: O(M² × N) but faster in practice
    """
    word_set = set(wordList)

    if endWord not in word_set:
        return 0

    front = {beginWord}
    back = {endWord}
    visited = set()
    length = 1

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
                        return length + 1

                    if new_word in word_set and new_word not in visited:
                        visited.add(new_word)
                        next_front.add(new_word)

        front = next_front
        length += 1

    return 0
```

### Complexity
- **Time**: O(M² × N) where M = word length, N = number of words
- **Space**: O(M × N) for visited set and queue

### Edge Cases
- `endWord` not in wordList: Return 0
- `beginWord` equals `endWord`: Return 1
- No valid transformation: Return 0

---

## Problem 3: Rotting Oranges (LC #994) - Medium

- [LeetCode](https://leetcode.com/problems/rotting-oranges/)

### Problem Statement
You are given an `m x n` grid where each cell can have one of three values: 0 (empty), 1 (fresh orange), or 2 (rotten orange). Every minute, any fresh orange adjacent (4-directionally) to a rotten orange becomes rotten. Return the minimum number of minutes until no fresh orange remains. Return -1 if impossible.

### Video Explanation
- [NeetCode - Rotting Oranges](https://www.youtube.com/watch?v=y704fEOx0s0)

### Examples
```
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
Explanation:
  Minute 0: [2,1,1]    Minute 1: [2,2,1]    Minute 2: [2,2,2]
            [1,1,0]              [2,1,0]              [2,2,0]
            [0,1,1]              [0,1,1]              [0,1,1]

  Minute 3: [2,2,2]    Minute 4: [2,2,2]
            [2,2,0]              [2,2,0]
            [0,2,1]              [0,2,2]

Input: grid = [[2,1,1],[0,1,1],[1,0,1]]
Output: -1
Explanation: Orange at (2,0) can never be reached

Input: grid = [[0,2]]
Output: 0
Explanation: No fresh oranges
```

### Intuition Development
```
Multi-source BFS: All rotten oranges spread simultaneously!

Unlike single-source BFS, we start with ALL rotten oranges in queue.

grid:
  2 1 1
  1 1 0
  0 1 1

┌─────────────────────────────────────────────────────────────────┐
│ Initial queue: [(0,0)]  (all rotten oranges)                    │
│ Fresh count: 6                                                   │
│                                                                  │
│ Minute 1: Process (0,0)                                         │
│   Rot neighbors: (0,1), (1,0)                                   │
│   Queue: [(0,1,1), (1,0,1)]                                     │
│   Fresh: 4                                                       │
│                                                                  │
│ Minute 2: Process (0,1), (1,0)                                  │
│   Rot neighbors: (0,2), (1,1)                                   │
│   Queue: [(0,2,2), (1,1,2)]                                     │
│   Fresh: 2                                                       │
│                                                                  │
│ Continue until fresh = 0 or queue empty...                      │
│ If fresh > 0 at end → return -1                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def orangesRotting(grid: list[list[int]]) -> int:
    """
    Multi-source BFS from all rotten oranges.

    Time: O(m × n)
    Space: O(m × n)
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
                grid[nr][nc] = 2
                fresh -= 1
                queue.append((nr, nc, time + 1))

    return minutes if fresh == 0 else -1
```

### Complexity
- **Time**: O(m × n) - Each cell visited at most once
- **Space**: O(m × n) - Queue can hold all cells

### Edge Cases
- No fresh oranges: Return 0
- No rotten oranges: Return -1 if fresh exist, else 0
- Isolated fresh orange: Return -1

---

## Problem 4: 01 Matrix (LC #542) - Medium

- [LeetCode](https://leetcode.com/problems/01-matrix/)

### Problem Statement
Given an `m x n` binary matrix `mat`, return the distance of the nearest 0 for each cell. The distance between two adjacent cells is 1.

### Video Explanation
- [NeetCode - 01 Matrix](https://www.youtube.com/watch?v=Ezj3VDOfd5c)

### Examples
```
Input: mat = [[0,0,0],[0,1,0],[0,0,0]]
Output: [[0,0,0],[0,1,0],[0,0,0]]

Input: mat = [[0,0,0],[0,1,0],[1,1,1]]
Output: [[0,0,0],[0,1,0],[1,2,1]]
Explanation: The 1 at (2,1) is distance 2 from nearest 0
```

### Intuition Development
```
Multi-source BFS from ALL 0s simultaneously!

mat:                   distances:
  0 0 0                  0 0 0
  0 1 0        →         0 1 0
  1 1 1                  1 2 1

┌─────────────────────────────────────────────────────────────────┐
│ Key insight: BFS from 0s outward!                               │
│   (Not from each 1 toward nearest 0 - that's inefficient)       │
│                                                                  │
│ Algorithm:                                                       │
│   1. Add all 0s to queue (distance 0)                           │
│   2. Mark all 1s as infinity (or very large)                    │
│   3. BFS: for each cell, update neighbors if shorter path       │
│                                                                  │
│ Why multi-source works:                                         │
│   All 0s expand at same rate                                    │
│   First 0 to reach a 1 gives the shortest distance!             │
│                                                                  │
│ Step by step:                                                    │
│   Level 0: All 0s have distance 0                               │
│   Level 1: All cells adjacent to 0 get distance 1               │
│   Level 2: All cells adjacent to level 1 get distance 2         │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def updateMatrix(mat: list[list[int]]) -> list[list[int]]:
    """
    Multi-source BFS from all 0s.

    Time: O(m × n)
    Space: O(m × n)
    """
    rows, cols = len(mat), len(mat[0])
    queue = deque()

    # Initialize: 0s have distance 0, others are infinity
    for r in range(rows):
        for c in range(cols):
            if mat[r][c] == 0:
                queue.append((r, c))
            else:
                mat[r][c] = float('inf')

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        r, c = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                if mat[nr][nc] > mat[r][c] + 1:
                    mat[nr][nc] = mat[r][c] + 1
                    queue.append((nr, nc))

    return mat
```

### Complexity
- **Time**: O(m × n) - Each cell processed once
- **Space**: O(m × n) - Queue size

### Edge Cases
- All 0s: Return matrix of 0s
- Single cell: Return [[0]] or [[0]] (0 is always present)
- Large matrix: Multi-source BFS scales efficiently

---

## Problem 5: Open the Lock (LC #752) - Medium

- [LeetCode](https://leetcode.com/problems/open-the-lock/)

### Problem Statement
You have a lock in front of you with 4 circular wheels, each with 10 slots: '0' through '9'. The lock initially starts at '0000'. You are given a list of `deadends`, combinations that will lock you out. Return the minimum number of turns required to open the lock (reach `target`), or -1 if impossible.

### Video Explanation
- [NeetCode - Open the Lock](https://www.youtube.com/watch?v=Pzg3bCDY87w)

### Examples
```
Input: deadends = ["0201","0101","0102","1212","2002"], target = "0202"
Output: 6
Explanation: "0000" → "1000" → "1100" → "1200" → "1201" → "1202" → "0202"

Input: deadends = ["8888"], target = "0009"
Output: 1
Explanation: "0000" → "0009"

Input: deadends = ["0000"], target = "8888"
Output: -1
Explanation: Start position is a deadend
```

### Intuition Development
```
State space BFS: Each lock combination is a node!

State graph: 10^4 = 10,000 possible states
Each state has 8 neighbors (4 wheels × 2 directions)

┌─────────────────────────────────────────────────────────────────┐
│ "0000" neighbors (each wheel can turn up or down):             │
│                                                                  │
│   Wheel 0: "1000", "9000"  (0→1, 0→9)                          │
│   Wheel 1: "0100", "0900"                                       │
│   Wheel 2: "0010", "0090"                                       │
│   Wheel 3: "0001", "0009"                                       │
│                                                                  │
│ BFS ensures shortest path (minimum turns)!                      │
│                                                                  │
│ Skip deadends and already visited states.                       │
│                                                                  │
│ Special case: "0000" in deadends → return -1 immediately        │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def openLock(deadends: list[str], target: str) -> int:
    """
    BFS on state space (4-digit combinations).

    Time: O(10^4 × 4 × 2) = O(1)
    Space: O(10^4)
    """
    dead = set(deadends)

    if '0000' in dead:
        return -1
    if target == '0000':
        return 0

    queue = deque([('0000', 0)])
    visited = {'0000'}

    while queue:
        state, turns = queue.popleft()

        # Generate all neighbors (turn each wheel up/down)
        for i in range(4):
            digit = int(state[i])

            for delta in [-1, 1]:
                new_digit = (digit + delta) % 10
                new_state = state[:i] + str(new_digit) + state[i+1:]

                if new_state == target:
                    return turns + 1

                if new_state not in visited and new_state not in dead:
                    visited.add(new_state)
                    queue.append((new_state, turns + 1))

    return -1
```

### Complexity
- **Time**: O(10^4 × 4 × 2) = O(1) - Fixed state space
- **Space**: O(10^4) - Visited set

### Edge Cases
- Start is deadend: Return -1
- Target is start: Return 0
- Target is deadend: Impossible, but BFS handles it

---

## Problem 6: Minimum Knight Moves (LC #1197) - Medium

- [LeetCode](https://leetcode.com/problems/minimum-knight-moves/)

### Problem Statement
In an infinite chess board with coordinates from -infinity to +infinity, you have a knight at square `[0, 0]`. A knight has 8 possible moves. Return the minimum number of steps needed to move the knight to the square `[x, y]`.

### Video Explanation
- [NeetCode - Minimum Knight Moves](https://www.youtube.com/watch?v=dDkgpVzJlmA)

### Examples
```
Input: x = 2, y = 1
Output: 1
Explanation: [0, 0] → [2, 1] (one L-shaped move)

Input: x = 5, y = 5
Output: 4

Input: x = 0, y = 0
Output: 0
```

### Intuition Development
```
BFS on infinite board with optimization!

Knight moves (L-shaped):
  (+2,+1), (+2,-1), (-2,+1), (-2,-1)
  (+1,+2), (+1,-2), (-1,+2), (-1,-2)

┌─────────────────────────────────────────────────────────────────┐
│ Optimization: Use symmetry!                                     │
│                                                                  │
│ Knight moves are symmetric in all quadrants.                    │
│ (x, y) requires same moves as (|x|, |y|)                        │
│                                                                  │
│ Bound the search:                                               │
│   Don't go too far past target                                  │
│   Allow small negative (for edge cases like (1,0))              │
│   Bound: -2 ≤ coord ≤ target + 2                                │
│                                                                  │
│ Example reaching (2,1):                                         │
│   Start: (0,0)                                                   │
│   One move: (2,1) ✓                                             │
│                                                                  │
│ Example reaching (1,0):                                         │
│   (0,0) → (2,1) → (0,2) → (1,0)? No!                           │
│   (0,0) → (2,-1) → (1,1) → ... need negative coords!           │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def minKnightMoves(x: int, y: int) -> int:
    """
    BFS with symmetry optimization.

    Use first quadrant only (symmetry).

    Time: O(|x| × |y|)
    Space: O(|x| × |y|)
    """
    # Use symmetry - work in first quadrant
    x, y = abs(x), abs(y)

    # Knight moves
    moves = [(2,1), (1,2), (-1,2), (-2,1), (-2,-1), (-1,-2), (1,-2), (2,-1)]

    queue = deque([(0, 0, 0)])
    visited = {(0, 0)}

    while queue:
        cx, cy, steps = queue.popleft()

        if cx == x and cy == y:
            return steps

        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy

            # Stay in reasonable bounds (allow small negative for edge cases)
            if (nx, ny) not in visited and -2 <= nx <= x + 2 and -2 <= ny <= y + 2:
                visited.add((nx, ny))
                queue.append((nx, ny, steps + 1))

    return -1
```

### Complexity
- **Time**: O(|x| × |y|) - Bounded search area
- **Space**: O(|x| × |y|) - Visited set

### Edge Cases
- Target (0,0): Return 0
- Target (1,0): Requires going negative briefly (3 moves)
- Very distant target: BFS scales with distance

---

## Problem 7: Snakes and Ladders (LC #909) - Medium

- [LeetCode](https://leetcode.com/problems/snakes-and-ladders/)

### Problem Statement
You are given an `n x n` board where cells are labeled from 1 to n² in a **Boustrophedon style** (alternating left-right, right-left). A move consists of rolling a die (1-6). If you land on a snake or ladder, you must take it. Return the minimum number of moves to reach square n².

### Video Explanation
- [NeetCode - Snakes and Ladders](https://www.youtube.com/watch?v=6lH4nO3JfLk)

### Examples
```
Input: board = [[-1,-1,-1,-1,-1,-1],
                [-1,-1,-1,-1,-1,-1],
                [-1,-1,-1,-1,-1,-1],
                [-1,35,-1,-1,13,-1],
                [-1,-1,-1,-1,-1,-1],
                [-1,15,-1,-1,-1,-1]]
Output: 4
Explanation: Square 2 has ladder to 15, square 15 has ladder to 35...

Input: board = [[-1,-1],[-1,3]]
Output: 1
```

### Intuition Development
```
BFS on board positions with snake/ladder jumps!

Boustrophedon numbering (n=4):
  13 14 15 16    ← left to right
  12 11 10  9    ← right to left
   5  6  7  8    ← left to right
   1  2  3  4    ← right to left (start)

┌─────────────────────────────────────────────────────────────────┐
│ Converting square number to (row, col):                         │
│                                                                  │
│   row = (square - 1) // n                                       │
│   col = (square - 1) % n                                        │
│   if row is odd: col = n - 1 - col  (flip direction)           │
│   actual_row = n - 1 - row  (board is bottom-to-top)            │
│                                                                  │
│ BFS from square 1 to square n²:                                 │
│   For each die roll 1-6:                                        │
│     next_square = current + roll                                │
│     If snake/ladder: next_square = board value                  │
│     Add to queue if not visited                                 │
│                                                                  │
│ Key: MUST take snake/ladder if you land on it!                  │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def snakesAndLadders(board: list[list[int]]) -> int:
    """
    BFS on board positions.

    Time: O(n²)
    Space: O(n²)
    """
    n = len(board)

    def get_position(square):
        """Convert square number to (row, col)."""
        row = (square - 1) // n
        col = (square - 1) % n

        # Alternate direction (Boustrophedon)
        if row % 2 == 1:
            col = n - 1 - col

        return n - 1 - row, col

    queue = deque([(1, 0)])  # (square, moves)
    visited = {1}

    while queue:
        square, moves = queue.popleft()

        # Try dice rolls 1-6
        for dice in range(1, 7):
            next_square = square + dice

            if next_square > n * n:
                continue

            r, c = get_position(next_square)

            # Check for snake/ladder
            if board[r][c] != -1:
                next_square = board[r][c]

            if next_square == n * n:
                return moves + 1

            if next_square not in visited:
                visited.add(next_square)
                queue.append((next_square, moves + 1))

    return -1
```

### Complexity
- **Time**: O(n²) - Visit each square at most once
- **Space**: O(n²) - Visited set and queue

### Edge Cases
- 2x2 board: Simple case with possible ladder
- Start has ladder: Must take it
- Ladder leads backward: Still must take it

---

## Problem 8: Shortest Bridge (LC #934) - Medium

- [LeetCode](https://leetcode.com/problems/shortest-bridge/)

### Problem Statement
You are given an `n x n` binary matrix `grid` where 1 represents land and 0 represents water. An island is a group of 1s connected 4-directionally. There are exactly two islands. Return the smallest number of 0s you must flip to connect the two islands.

### Video Explanation
- [NeetCode - Shortest Bridge](https://www.youtube.com/watch?v=gkINMhbbIbU)

### Examples
```
Input: grid = [[0,1],[1,0]]
Output: 1
Explanation: Flip grid[0][0] or grid[1][1]

Input: grid = [[0,1,0],[0,0,0],[0,0,1]]
Output: 2
Explanation: Flip grid[1][1] and grid[2,1]

Input: grid = [[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]
Output: 1
```

### Intuition Development
```
Two-phase approach: DFS to find island, BFS to expand!

grid:
  0 1 0
  0 0 0
  0 0 1

┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: DFS to find and mark first island                     │
│                                                                  │
│   Find first 1 at (0,1)                                         │
│   DFS marks all connected 1s as 2 (visited)                     │
│   Add all island cells to BFS queue                             │
│                                                                  │
│   After Phase 1:                                                 │
│   0 2 0                                                         │
│   0 0 0                                                         │
│   0 0 1                                                         │
│                                                                  │
│ Phase 2: BFS to expand until we hit second island               │
│                                                                  │
│   Level 0: (0,1) with dist=0                                    │
│   Level 1: (0,0), (0,2), (1,1) with dist=1                      │
│   Level 2: (1,0), (1,2), (2,1) with dist=2                      │
│     (2,1) is adjacent to (2,2) which is 1 → FOUND!              │
│                                                                  │
│   Return dist = 2 (flipped 2 zeros to connect)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def shortestBridge(grid: list[list[int]]) -> int:
    """
    DFS to find first island, BFS to expand to second.

    Time: O(n²)
    Space: O(n²)
    """
    n = len(grid)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Find and mark first island using DFS
    queue = deque()
    found = False

    def dfs(r, c):
        if r < 0 or r >= n or c < 0 or c >= n or grid[r][c] != 1:
            return

        grid[r][c] = 2  # Mark as visited
        queue.append((r, c, 0))

        for dr, dc in directions:
            dfs(r + dr, c + dc)

    # Find first island
    for r in range(n):
        if found:
            break
        for c in range(n):
            if grid[r][c] == 1:
                dfs(r, c)
                found = True
                break

    # BFS to find shortest path to second island
    while queue:
        r, c, dist = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < n and 0 <= nc < n:
                if grid[nr][nc] == 1:
                    return dist

                if grid[nr][nc] == 0:
                    grid[nr][nc] = 2
                    queue.append((nr, nc, dist + 1))

    return -1
```

### Complexity
- **Time**: O(n²) - DFS + BFS each visit cells once
- **Space**: O(n²) - Queue for BFS

### Edge Cases
- Adjacent islands: Return 0
- Islands at corners: Longest possible distance
- Single cell islands: Works correctly

---

## Problem 9: Walls and Gates (LC #286) - Medium

- [LeetCode](https://leetcode.com/problems/walls-and-gates/)

### Problem Statement
You are given an `m x n` grid with three types of cells: -1 (wall), 0 (gate), or INF (empty room, represented as 2^31 - 1). Fill each empty room with the distance to its nearest gate. If impossible to reach a gate, leave it as INF.

### Video Explanation
- [NeetCode - Walls and Gates](https://www.youtube.com/watch?v=e69C6xhiSQE)

### Examples
```
Input: rooms = [[INF, -1,  0, INF],
                [INF,INF,INF, -1],
                [INF, -1,INF, -1],
                [  0, -1,INF,INF]]
Output:       [[  3, -1,  0,  1],
               [  2,  2,  1, -1],
               [  1, -1,  2, -1],
               [  0, -1,  3,  4]]
```

### Intuition Development
```
Multi-source BFS from all gates!

rooms:
  INF  -1   0  INF
  INF INF INF  -1
  INF  -1 INF  -1
    0  -1 INF INF

┌─────────────────────────────────────────────────────────────────┐
│ Similar to 01 Matrix - BFS from all 0s (gates)!                │
│                                                                  │
│ Initial queue: [(0,2), (3,0)]  (all gates)                      │
│                                                                  │
│ Level 1: All rooms adjacent to gates get distance 1             │
│   (0,3), (1,2), (2,0) get distance 1                            │
│                                                                  │
│ Level 2: All rooms adjacent to level 1 get distance 2           │
│   (1,1), (1,0), (2,2) get distance 2                            │
│                                                                  │
│ Continue until all reachable rooms have distances...            │
│                                                                  │
│ Walls (-1) are skipped, unreachable rooms stay INF              │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def wallsAndGates(rooms: list[list[int]]) -> None:
    """
    Multi-source BFS from all gates.

    Time: O(m × n)
    Space: O(m × n)
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

### Complexity
- **Time**: O(m × n) - Each cell visited once
- **Space**: O(m × n) - Queue size

### Edge Cases
- No gates: All rooms stay INF
- No empty rooms: Nothing to fill
- Room surrounded by walls: Stays INF

---

## Problem 10: As Far from Land as Possible (LC #1162) - Medium

- [LeetCode](https://leetcode.com/problems/as-far-from-land-as-possible/)

### Problem Statement
Given an `n x n` grid containing only values 0 (water) and 1 (land), find a water cell such that its distance to the nearest land cell is maximized, and return the distance. If no land or no water exists, return -1.

### Video Explanation
- [NeetCode - As Far from Land as Possible](https://www.youtube.com/watch?v=fjxb5mLmu7A)

### Examples
```
Input: grid = [[1,0,1],[0,0,0],[1,0,1]]
Output: 2
Explanation: The water cell at (1,1) has distance 2 to all land cells

Input: grid = [[1,0,0],[0,0,0],[0,0,0]]
Output: 4
Explanation: The water cell at (2,2) is farthest from land

Input: grid = [[1,1,1],[1,1,1],[1,1,1]]
Output: -1
Explanation: No water cells
```

### Intuition Development
```
Multi-source BFS from all land cells!

grid:
  1 0 1
  0 0 0
  1 0 1

┌─────────────────────────────────────────────────────────────────┐
│ Expand from ALL land cells simultaneously.                      │
│ Track maximum distance reached.                                 │
│                                                                  │
│ Initial queue: [(0,0), (0,2), (2,0), (2,2)]                     │
│                                                                  │
│ Level 1: (0,1), (1,0), (1,2), (2,1)  distance=1                 │
│ Level 2: (1,1)  distance=2                                      │
│                                                                  │
│ Maximum distance = 2 ✓                                          │
│                                                                  │
│ BFS ensures we find distances level by level.                   │
│ Last cell to be reached has the maximum distance!               │
│                                                                  │
│ Edge cases:                                                      │
│   All land: No water → return -1                                │
│   All water: No land → return -1                                │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def maxDistance(grid: list[list[int]]) -> int:
    """
    Multi-source BFS from all land cells.

    Time: O(n²)
    Space: O(n²)
    """
    n = len(grid)
    queue = deque()

    # Find all land cells
    for r in range(n):
        for c in range(n):
            if grid[r][c] == 1:
                queue.append((r, c))

    # All land or all water
    if len(queue) == 0 or len(queue) == n * n:
        return -1

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    max_dist = -1

    while queue:
        r, c = queue.popleft()

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 0:
                grid[nr][nc] = grid[r][c] + 1
                max_dist = max(max_dist, grid[nr][nc] - 1)
                queue.append((nr, nc))

    return max_dist
```

### Complexity
- **Time**: O(n²) - Each cell visited once
- **Space**: O(n²) - Queue and grid modification

### Edge Cases
- All land: Return -1
- All water: Return -1
- Single water cell surrounded by land: Return 1
- Corners are farthest: Common pattern

---

## Summary: Graph BFS Medium Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Shortest Path Binary Matrix | Standard BFS | O(n²) |
| 2 | Word Ladder | BFS on words | O(M² × N) |
| 3 | Rotting Oranges | Multi-source BFS | O(mn) |
| 4 | 01 Matrix | Multi-source BFS | O(mn) |
| 5 | Open the Lock | State space BFS | O(1) |
| 6 | Knight Moves | BFS with symmetry | O(xy) |
| 7 | Snakes and Ladders | BFS with jumps | O(n²) |
| 8 | Shortest Bridge | DFS + BFS | O(n²) |
| 9 | Walls and Gates | Multi-source BFS | O(mn) |
| 10 | Far from Land | Multi-source BFS | O(n²) |

---

## BFS Template

```python
from collections import deque

def bfs_template(start, is_goal, get_neighbors):
    """Generic BFS template."""
    queue = deque([(start, 0)])  # (state, distance)
    visited = {start}

    while queue:
        state, dist = queue.popleft()

        if is_goal(state):
            return dist

        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return -1  # No path found
```

---

## Practice More Problems

- [ ] LC #126 - Word Ladder II
- [ ] LC #317 - Shortest Distance from All Buildings
- [ ] LC #490 - The Maze
- [ ] LC #773 - Sliding Puzzle
- [ ] LC #847 - Shortest Path Visiting All Nodes

