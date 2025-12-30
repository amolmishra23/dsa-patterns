# Backtracking - Hard Problems

## Problem 1: N-Queens (LC #51) - Hard

- [LeetCode](https://leetcode.com/problems/n-queens/)

### Video Explanation
- [NeetCode - N-Queens](https://www.youtube.com/watch?v=Ph95IHmRp5M)

### Problem Statement
Place n queens on n×n chessboard so no two attack each other.


### Visual Intuition
```
N-Queens
n = 4, place 4 queens with no attacks

    0 1 2 3
  0 . Q . .
  1 . . . Q
  2 Q . . .
  3 . . Q .

Backtrack row by row:
Row 0: try cols → col 1 valid (no conflicts)
Row 1: try cols → col 3 valid
Row 2: try cols → col 0 valid
Row 3: try cols → col 2 valid ✓

Track: cols set, diagonals (r-c), anti-diagonals (r+c)
If queen at (r,c): cols.add(c), diag.add(r-c), anti.add(r+c)
```

### Solution
```python
def solveNQueens(n: int) -> list[list[str]]:
    """
    N-Queens using backtracking.

    Strategy:
    - Place queens row by row
    - Track columns, diagonals, anti-diagonals
    - Backtrack when no valid position

    Time: O(n!)
    Space: O(n²)
    """
    result = []
    board = [["."] * n for _ in range(n)]
    cols = set()
    diag = set()      # row - col
    anti_diag = set() # row + col

    def backtrack(row):
        if row == n:
            result.append(["".join(r) for r in board])
            return

        for col in range(n):
            if col in cols or (row - col) in diag or (row + col) in anti_diag:
                continue

            # Place queen
            board[row][col] = "Q"
            cols.add(col)
            diag.add(row - col)
            anti_diag.add(row + col)

            backtrack(row + 1)

            # Remove queen
            board[row][col] = "."
            cols.remove(col)
            diag.remove(row - col)
            anti_diag.remove(row + col)

    backtrack(0)
    return result
```

### Edge Cases
- n = 1 → single solution [["Q"]]
- n = 2 or n = 3 → no solution []
- n = 4 → 2 solutions
- Large n → exponential but pruning helps

---

## Problem 2: Sudoku Solver (LC #37) - Hard

- [LeetCode](https://leetcode.com/problems/sudoku-solver/)

### Video Explanation
- [NeetCode - Sudoku Solver](https://www.youtube.com/watch?v=FWAIf_EVUKE)

### Problem Statement
Solve a Sudoku puzzle by filling empty cells.


### Visual Intuition
```
Sudoku Solver
Fill 9x9 grid: each row, col, 3x3 box has 1-9

  5 3 . | . 7 . | . . .
  6 . . | 1 9 5 | . . .
  . 9 8 | . . . | . 6 .
  ------+-------+------
  8 . . | . 6 . | . . 3
  ...

Backtrack cell by cell:
  Find empty cell (0,2)
  Try 1: check row0, col2, box0 → invalid (conflict)
  Try 2: check → invalid
  Try 4: check → valid! Place and recurse

  If stuck, backtrack and try next number

Optimization: Choose cell with fewest candidates (MRV)
```

### Solution
```python
def solveSudoku(board: list[list[str]]) -> None:
    """
    Solve Sudoku using backtracking.

    Time: O(9^(empty cells))
    Space: O(81)
    """
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]

    # Initialize sets with existing numbers
    for i in range(9):
        for j in range(9):
            if board[i][j] != ".":
                num = board[i][j]
                rows[i].add(num)
                cols[j].add(num)
                boxes[(i // 3) * 3 + j // 3].add(num)

    def backtrack(pos):
        if pos == 81:
            return True

        i, j = pos // 9, pos % 9

        if board[i][j] != ".":
            return backtrack(pos + 1)

        box_idx = (i // 3) * 3 + j // 3

        for num in "123456789":
            if num in rows[i] or num in cols[j] or num in boxes[box_idx]:
                continue

            board[i][j] = num
            rows[i].add(num)
            cols[j].add(num)
            boxes[box_idx].add(num)

            if backtrack(pos + 1):
                return True

            board[i][j] = "."
            rows[i].remove(num)
            cols[j].remove(num)
            boxes[box_idx].remove(num)

        return False

    backtrack(0)
```

### Edge Cases
- Already solved → no changes needed
- Multiple solutions → any valid one
- Invalid input → shouldn't happen per problem
- Empty cells only → fill all

---

## Problem 3: Word Search II (LC #212) - Hard

- [LeetCode](https://leetcode.com/problems/word-search-ii/)

### Video Explanation
- [NeetCode - Word Search II](https://www.youtube.com/watch?v=asbcE9mZz_U)

### Problem Statement
Find all words from dictionary that exist in the board.


### Visual Intuition
```
Word Search II (Trie + Backtracking)
board = [["o","a","a","n"],     words = ["oath","pea","eat","rain"]
         ["e","t","a","e"],
         ["i","h","k","r"],
         ["i","f","l","v"]]

Build Trie from words:
      root
     / | \
    o  p  e  r
    |  |  |  |
    a  e  a  a
    |  |  |  |
    t  a* t* i
    |        |
    h*       n*

DFS from each cell, follow Trie:
  Start (0,0)='o': Trie has 'o' → continue
  Move to (1,0)='e': no 'e' under 'o' → backtrack
  Move to (0,1)='a': Trie has 'a' → continue
  ...find "oath" ✓

Found: ["oath", "eat"]
```

### Solution
```python
def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    """
    Find words using Trie + DFS backtracking.

    Time: O(m * n * 4^L) where L = max word length
    Space: O(total chars in words)
    """
    # Build Trie
    trie = {}
    for word in words:
        node = trie
        for c in word:
            node = node.setdefault(c, {})
        node["$"] = word

    m, n = len(board), len(board[0])
    result = []

    def dfs(i, j, node):
        char = board[i][j]
        if char not in node:
            return

        next_node = node[char]

        if "$" in next_node:
            result.append(next_node["$"])
            del next_node["$"]  # Avoid duplicates

        board[i][j] = "#"  # Mark visited

        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and board[ni][nj] != "#":
                dfs(ni, nj, next_node)

        board[i][j] = char  # Restore

    for i in range(m):
        for j in range(n):
            dfs(i, j, trie)

    return result
```

### Edge Cases
- No words found → return []
- All words found → return all
- Overlapping words → Trie handles
- Single cell board → check single char words

---

## Problem 4: Expression Add Operators (LC #282) - Hard

- [LeetCode](https://leetcode.com/problems/expression-add-operators/)

### Video Explanation
- [NeetCode - Expression Add Operators](https://www.youtube.com/watch?v=v05R1OIIg08)

### Problem Statement
Add binary operators (+, -, *) between digits to reach target value.

### Visual Intuition
```
Expression Add Operators - num="123", target=6
Insert +, -, * between digits

Decision tree:
    "123"
   /  |  \
  1   12  123
 /|\
1+2  1-2  1*2
 |
1+2+3=6 ✓  1+2-3=0  1+2*3=7  1-2+3=2  1*2+3=5  1*2*3=6 ✓

Track: current value, previous operand (for * precedence)
1*2+3: prev=2, cur=5 → for *3: cur = 5-2+2*3 = 9
```


### Intuition
```
num = "123", target = 6

Possibilities:
- "1+2+3" = 6 ✓
- "1*2*3" = 6 ✓
- "12-3" = 9 ✗
- "1-2+3" = 2 ✗

Key insight: Track 'prev' operand for multiplication precedence.
"1+2*3" = 1 + (2*3) = 7, not (1+2)*3 = 9
```

### Solution
```python
def addOperators(num: str, target: int) -> list[str]:
    """
    Backtracking with multiplication precedence handling.

    Strategy:
    - Try all ways to split string into numbers
    - Track current value and previous operand (for * precedence)
    - For multiplication: undo previous add, then multiply

    Time: O(4^n) - each position has 4 choices
    Space: O(n) for recursion
    """
    result = []
    n = len(num)

    def backtrack(idx, path, value, prev):
        """
        idx: current position in num
        path: expression built so far
        value: current evaluated value
        prev: previous operand (for undoing in multiplication)
        """
        if idx == n:
            if value == target:
                result.append(path)
            return

        for i in range(idx, n):
            # Skip numbers with leading zeros (except "0" itself)
            if i > idx and num[idx] == '0':
                break

            curr_str = num[idx:i+1]
            curr_num = int(curr_str)

            if idx == 0:
                # First number - no operator
                backtrack(i + 1, curr_str, curr_num, curr_num)
            else:
                # Try addition
                backtrack(i + 1, path + '+' + curr_str,
                         value + curr_num, curr_num)

                # Try subtraction
                backtrack(i + 1, path + '-' + curr_str,
                         value - curr_num, -curr_num)

                # Try multiplication (undo prev, apply *)
                backtrack(i + 1, path + '*' + curr_str,
                         value - prev + prev * curr_num, prev * curr_num)

    backtrack(0, "", 0, 0)
    return result
```

### Complexity
- **Time**: O(4^n * n) - 4 choices per position, n for string ops
- **Space**: O(n) recursion depth

### Edge Cases
- Single digit → check if equals target
- Leading zeros → skip "01", "02", etc.
- target = 0 → may have solutions
- Large numbers → watch for overflow

---

## Problem 5: Remove Invalid Parentheses (LC #301) - Hard

- [LeetCode](https://leetcode.com/problems/remove-invalid-parentheses/)

### Video Explanation
- [NeetCode - Remove Invalid Parentheses](https://www.youtube.com/watch?v=Cbbf5qe5stw)

### Problem Statement
Remove minimum number of invalid parentheses to make string valid.

### Visual Intuition
```
Remove Invalid Parentheses - s="()())()"
Find minimum removals for valid string

═══════════════════════════════════════════════════════════════
KEY INSIGHT: BFS by removal count guarantees minimum removals
             First level with valid strings = optimal answer
═══════════════════════════════════════════════════════════════

Input Analysis:
───────────────
  s = "()())()"
       0123456

  Scan for mismatches:
    '(' at 0: count=1
    ')' at 1: count=0
    '(' at 2: count=1
    ')' at 3: count=0
    ')' at 4: count=-1 ← INVALID! Extra ')'
    '(' at 5: count=0
    ')' at 6: count=-1 ← INVALID!

  Need to remove at least 1 ')' to balance

BFS Level-by-Level:
───────────────────
Level 0: Original string
  "()()()" → is_valid? ✗ (extra ')' at index 4)

Level 1: Remove 1 character (try each position)
  ┌──────────────────────────────────────────────┐
  │ Remove idx 0: ")())()" → ✗ starts with ')'  │
  │ Remove idx 1: "(())()" → ✓ VALID!           │
  │ Remove idx 2: "())()"  → ✗ unbalanced       │
  │ Remove idx 3: "()()()" → ✓ VALID!           │
  │ Remove idx 4: "()()()" → ✓ VALID! (same)    │
  │ Remove idx 5: "()()()" → ✗ unbalanced       │
  │ Remove idx 6: "()()()" → ✗ unbalanced       │
  └──────────────────────────────────────────────┘

  Found valid strings at Level 1 → STOP!
  (No need to check Level 2)

Validation Function:
────────────────────
  "()()()" check:
    ( → count=1
    ) → count=0
    ( → count=1
    ) → count=0
    ( → count=1
    ) → count=0 ✓ (ends at 0, never negative)

Result: ["(())()", "()()()"]

WHY THIS WORKS:
════════════════
● BFS explores by "distance" (number of removals)
● First valid strings found = minimum removals
● Use set to avoid checking duplicate strings
● Stop immediately when valid strings found at any level
```


### Intuition
```
s = "()())()"

Count mismatches:
- Extra ')' at index 4
- Need to remove 1 ')' to balance

BFS by removal level ensures minimum removals.
```

### Solution
```python
from collections import deque

def removeInvalidParentheses(s: str) -> list[str]:
    """
    BFS to find all valid strings with minimum removals.

    Strategy:
    - BFS level = number of removals
    - First level with valid strings = minimum removals
    - Use set to avoid duplicates

    Time: O(2^n) worst case
    Space: O(2^n)
    """
    def is_valid(string):
        """Check if parentheses are balanced."""
        count = 0
        for c in string:
            if c == '(':
                count += 1
            elif c == ')':
                count -= 1
                if count < 0:
                    return False
        return count == 0

    result = []
    visited = {s}
    queue = deque([s])
    found = False

    while queue:
        # Process current level
        level_size = len(queue)

        for _ in range(level_size):
            curr = queue.popleft()

            if is_valid(curr):
                result.append(curr)
                found = True

            # Don't generate more if we found valid at this level
            if found:
                continue

            # Try removing each parenthesis
            for i in range(len(curr)):
                if curr[i] not in '()':
                    continue

                next_str = curr[:i] + curr[i+1:]

                if next_str not in visited:
                    visited.add(next_str)
                    queue.append(next_str)

        # Stop if we found valid strings at this level
        if found:
            break

    return result if result else [""]
```

### Optimized DFS Solution
```python
def removeInvalidParentheses(s: str) -> list[str]:
    """
    DFS with pruning - count exact removals needed.
    """
    def count_invalid(string):
        """Count minimum removals needed."""
        left = right = 0
        for c in string:
            if c == '(':
                left += 1
            elif c == ')':
                if left > 0:
                    left -= 1
                else:
                    right += 1
        return left, right

    result = []
    left_rem, right_rem = count_invalid(s)

    def dfs(idx, left_count, right_count, left_rem, right_rem, path):
        if idx == len(s):
            if left_rem == 0 and right_rem == 0:
                result.append(path)
            return

        char = s[idx]

        if char == '(':
            # Option 1: Remove this '('
            if left_rem > 0:
                dfs(idx + 1, left_count, right_count,
                    left_rem - 1, right_rem, path)
            # Option 2: Keep this '('
            dfs(idx + 1, left_count + 1, right_count,
                left_rem, right_rem, path + char)

        elif char == ')':
            # Option 1: Remove this ')'
            if right_rem > 0:
                dfs(idx + 1, left_count, right_count,
                    left_rem, right_rem - 1, path)
            # Option 2: Keep this ')' (only if valid)
            if left_count > right_count:
                dfs(idx + 1, left_count, right_count + 1,
                    left_rem, right_rem, path + char)

        else:
            # Non-parenthesis character - always keep
            dfs(idx + 1, left_count, right_count,
                left_rem, right_rem, path + char)

    dfs(0, 0, 0, left_rem, right_rem, "")
    return list(set(result))
```

### Complexity
- **Time**: O(2^n) worst case
- **Space**: O(n) for recursion

### Edge Cases
- Already valid → return [s]
- All invalid → return [""]
- Only parentheses → simplest case
- Mixed with letters → letters always kept

---

## Problem 6: Palindrome Partitioning II (LC #132) - Hard

- [LeetCode](https://leetcode.com/problems/palindrome-partitioning-ii/)

### Video Explanation
- [NeetCode - Palindrome Partitioning II](https://www.youtube.com/watch?v=_H8V5hJUGd0)

### Problem Statement
Minimum cuts to partition string into palindromes.

### Visual Intuition
```
Palindrome Partitioning II - s="aab"
Minimum cuts for all palindrome substrings

DP approach:
  a | a | b  → 2 cuts (each char is palindrome)
  aa | b     → 1 cut  ("aa" palindrome, "b" palindrome)

dp[i] = min cuts for s[0:i+1]
isPalin[i][j] = true if s[i:j+1] is palindrome

For "aab": dp = [0, 0, 1]
  dp[0] = 0 (single 'a')
  dp[1] = 0 ("aa" is palindrome, no cut needed)
  dp[2] = 1 ("aab" not palindrome, best: "aa"|"b")
```


### Intuition
```
s = "aab"

Partitions:
- ["a", "a", "b"] → 2 cuts
- ["aa", "b"] → 1 cut ✓

DP: dp[i] = min cuts for s[0:i]
```

### Solution
```python
def minCut(s: str) -> int:
    """
    DP with palindrome precomputation.

    Strategy:
    - Precompute all palindrome substrings
    - dp[i] = min cuts for s[0:i]
    - For each position, try all valid palindrome endings

    Time: O(n²)
    Space: O(n²)
    """
    n = len(s)

    # Precompute palindromes: is_pal[i][j] = s[i:j+1] is palindrome
    is_pal = [[False] * n for _ in range(n)]

    for i in range(n):
        is_pal[i][i] = True

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                is_pal[i][j] = (length == 2) or is_pal[i+1][j-1]

    # DP for minimum cuts
    dp = [float('inf')] * n

    for i in range(n):
        if is_pal[0][i]:
            dp[i] = 0  # Entire prefix is palindrome
        else:
            for j in range(i):
                if is_pal[j+1][i]:
                    dp[i] = min(dp[i], dp[j] + 1)

    return dp[n-1]
```

### Space-Optimized Solution
```python
def minCut(s: str) -> int:
    """
    Expand around center approach - O(n²) time, O(n) space.
    """
    n = len(s)
    dp = list(range(n))  # dp[i] = min cuts, worst case = i cuts

    def expand(left, right):
        """Expand palindrome and update dp."""
        while left >= 0 and right < n and s[left] == s[right]:
            # s[left:right+1] is palindrome
            if left == 0:
                dp[right] = 0
            else:
                dp[right] = min(dp[right], dp[left-1] + 1)
            left -= 1
            right += 1

    for i in range(n):
        expand(i, i)      # Odd length palindromes
        expand(i, i + 1)  # Even length palindromes

    return dp[n-1]
```

### Complexity
- **Time**: O(n²)
- **Space**: O(n) for optimized version

### Edge Cases
- Single character → 0 cuts
- Already palindrome → 0 cuts
- All same characters → 0 cuts
- All different characters → n-1 cuts

---

## Summary

| # | Problem | Key Technique |
|---|---------|---------------|
| 1 | N-Queens | Track cols, diagonals |
| 2 | Sudoku Solver | Track rows, cols, boxes |
| 3 | Word Search II | Trie + DFS backtracking |
| 4 | Expression Add Operators | Track prev for * precedence |
| 5 | Remove Invalid Parens | BFS by removal level |
| 6 | Palindrome Partition II | DP + palindrome precompute |
