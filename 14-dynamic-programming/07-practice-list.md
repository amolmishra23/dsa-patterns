# Dynamic Programming - Practice List

## Problems by Pattern

### 1D DP
- LC 70: Climbing Stairs (Easy)
- LC 198: House Robber (Medium)
- LC 213: House Robber II (Medium)
- LC 746: Min Cost Climbing Stairs (Easy)
- LC 139: Word Break (Medium)
- LC 300: Longest Increasing Subsequence (Medium)

### 2D DP (Grid)
- LC 62: Unique Paths (Medium)
- LC 63: Unique Paths II (Medium)
- LC 64: Minimum Path Sum (Medium)
- LC 120: Triangle (Medium)
- LC 221: Maximal Square (Medium)

### String DP
- LC 5: Longest Palindromic Substring (Medium)
- LC 516: Longest Palindromic Subsequence (Medium)
- LC 1143: Longest Common Subsequence (Medium)
- LC 72: Edit Distance (Medium)
- LC 10: Regular Expression Matching (Hard)
- LC 44: Wildcard Matching (Hard)

### Knapsack
- LC 416: Partition Equal Subset Sum (Medium)
- LC 494: Target Sum (Medium)
- LC 518: Coin Change II (Medium)
- LC 322: Coin Change (Medium)

### Interval DP
- LC 312: Burst Balloons (Hard)
- LC 1039: Minimum Score Triangulation (Medium)

### State Machine
- LC 121: Best Time to Buy and Sell Stock (Easy)
- LC 122: Best Time to Buy and Sell Stock II (Medium)
- LC 123: Best Time to Buy and Sell Stock III (Hard)
- LC 188: Best Time to Buy and Sell Stock IV (Hard)
- LC 309: Best Time with Cooldown (Medium)
- LC 714: Best Time with Transaction Fee (Medium)

## Templates

```python
# 1D DP Template
def solve_1d(n):
    dp = [0] * (n + 1)
    dp[0] = base_case
    for i in range(1, n + 1):
        dp[i] = transition(dp[i-1], dp[i-2], ...)
    return dp[n]

# 2D DP Template (Grid)
def solve_grid(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(m):
        for j in range(n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
    return dp[m-1][n-1]

# LCS Template
def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

# 0/1 Knapsack Template
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])
    return dp[n][capacity]
```

## Key Patterns
1. Identify subproblems
2. Define state clearly
3. Find recurrence relation
4. Determine base cases
5. Choose top-down or bottom-up

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DP PATTERNS                                         │
│                                                                             │
│  1D DP (Fibonacci/Climbing Stairs):                                         │
│  dp[i] = dp[i-1] + dp[i-2]                                                  │
│                                                                             │
│  [1] → [1,1] → [1,1,2] → [1,1,2,3] → [1,1,2,3,5]                           │
│   ↑      ↑        ↑          ↑                                              │
│  base  base    1+1=2      1+2=3                                             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  2D DP (Grid - Unique Paths):                                               │
│                                                                             │
│  [1] [1] [1] [1]     dp[i][j] = dp[i-1][j] + dp[i][j-1]                     │
│  [1] [2] [3] [4]                                                            │
│  [1] [3] [6] [10]    Answer: dp[2][3] = 10                                  │
│                                                                             │
│  Each cell = sum of top + left                                              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LCS (Longest Common Subsequence):                                          │
│  s1 = "ABCD", s2 = "AEBD"                                                   │
│                                                                             │
│      ""  A  E  B  D                                                         │
│  ""   0  0  0  0  0                                                         │
│   A   0  1  1  1  1    Match: dp[i][j] = dp[i-1][j-1] + 1                   │
│   B   0  1  1  2  2    No match: dp[i][j] = max(dp[i-1][j], dp[i][j-1])     │
│   C   0  1  1  2  2                                                         │
│   D   0  1  1  2  3    LCS = "ABD", length = 3                              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  KNAPSACK (0/1):                                                            │
│  Items: [(wt=1,val=1), (wt=3,val=4), (wt=4,val=5)], capacity=7              │
│                                                                             │
│      Cap: 0  1  2  3  4  5  6  7                                            │
│  Item 0:  0  0  0  0  0  0  0  0                                            │
│  Item 1:  0  1  1  1  1  1  1  1    (wt=1, val=1)                           │
│  Item 2:  0  1  1  4  5  5  5  5    (wt=3, val=4)                           │
│  Item 3:  0  1  1  4  5  6  6  9    (wt=4, val=5)                           │
│                                                                             │
│  Answer: 9 (items 2+3)                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: 1D DP Fundamentals
- [ ] LC 70: Climbing Stairs (Easy)
- [ ] LC 746: Min Cost Climbing Stairs (Easy)
- [ ] LC 198: House Robber (Medium)
- [ ] LC 213: House Robber II (Medium)
- [ ] LC 300: Longest Increasing Subsequence (Medium)

### Week 2: 2D DP & Grid
- [ ] LC 62: Unique Paths (Medium)
- [ ] LC 63: Unique Paths II (Medium)
- [ ] LC 64: Minimum Path Sum (Medium)
- [ ] LC 120: Triangle (Medium)
- [ ] LC 221: Maximal Square (Medium)

### Week 3: String DP
- [ ] LC 5: Longest Palindromic Substring (Medium)
- [ ] LC 516: Longest Palindromic Subsequence (Medium)
- [ ] LC 1143: Longest Common Subsequence (Medium)
- [ ] LC 72: Edit Distance (Medium)
- [ ] LC 139: Word Break (Medium)

### Week 4: Knapsack & Advanced
- [ ] LC 322: Coin Change (Medium)
- [ ] LC 518: Coin Change II (Medium)
- [ ] LC 416: Partition Equal Subset Sum (Medium)
- [ ] LC 494: Target Sum (Medium)
- [ ] LC 312: Burst Balloons (Hard)

---

## Common Mistakes

### 1. Wrong Base Case
```python
# WRONG - forgot base case
dp = [0] * (n + 1)
for i in range(2, n + 1):
    dp[i] = dp[i-1] + dp[i-2]  # dp[0], dp[1] are 0!

# CORRECT
dp = [0] * (n + 1)
dp[0] = 1  # or appropriate base
dp[1] = 1
for i in range(2, n + 1):
    dp[i] = dp[i-1] + dp[i-2]
```

### 2. Off-by-One Index Errors
```python
# WRONG - accessing s1[i] when dp is 1-indexed
for i in range(1, m + 1):
    if s1[i] == s2[j]:  # IndexError at i=m!

# CORRECT - use i-1 for string access
for i in range(1, m + 1):
    if s1[i-1] == s2[j-1]:  # Correct indexing
```

### 3. Wrong Recurrence Direction
```python
# WRONG - using future values
for i in range(n):
    dp[i] = dp[i+1] + dp[i+2]  # dp[i+1] not computed yet!

# CORRECT - use past values (bottom-up)
for i in range(2, n + 1):
    dp[i] = dp[i-1] + dp[i-2]  # Uses already computed values
```

### 4. Knapsack: Using Same Item Multiple Times
```python
# WRONG - 0/1 knapsack but allows reuse
for i in range(n):
    for w in range(weights[i], capacity + 1):
        dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

# CORRECT - iterate capacity in reverse
for i in range(n):
    for w in range(capacity, weights[i] - 1, -1):  # Reverse!
        dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
```

### 5. Not Handling Empty Input
```python
# WRONG - crashes on empty input
def solve(nums):
    dp = [0] * len(nums)  # Empty array!
    dp[0] = nums[0]  # IndexError!

# CORRECT - handle edge cases
def solve(nums):
    if not nums:
        return 0
    dp = [0] * len(nums)
    dp[0] = nums[0]
```

---

## Complexity Reference

| Pattern | Time | Space | Space Optimized |
|---------|------|-------|-----------------|
| 1D DP | O(n) | O(n) | O(1) with 2 vars |
| 2D Grid | O(m*n) | O(m*n) | O(n) with 1 row |
| LCS | O(m*n) | O(m*n) | O(min(m,n)) |
| Edit Distance | O(m*n) | O(m*n) | O(n) |
| 0/1 Knapsack | O(n*W) | O(n*W) | O(W) |
| LIS | O(n²) | O(n) | O(n log n) with binary search |

---

## Pattern Recognition

| See This | Think This |
|----------|------------|
| "Number of ways" | DP with addition |
| "Minimum/Maximum" | DP with min/max |
| "Can we achieve X?" | DP with boolean |
| "Subsequence" | Usually 2D DP |
| "Substring" | 2D DP or expand around center |
| "Partition" | Knapsack variant |
| "Buy/sell stock" | State machine DP |
| "Choices at each step" | Decision tree → DP |
