# Dynamic Programming - Fundamentals

> **📁 Extended Module Structure**: Due to the breadth and complexity of Dynamic Programming, this module contains 7 files instead of the standard 5:
> - `01-fundamentals.md` - Core concepts and templates
> - `02-easy-problems.md` - Easy problems (Fibonacci, Climbing Stairs, etc.)
> - `03-medium-problems.md` - Medium problems (House Robber, Coin Change, etc.)
> - `04-hard-problems.md` - Hard problems (Edit Distance, Burst Balloons, etc.)
> - `05-dp-on-strings.md` - Specialized: String DP problems (LCS, Edit Distance, etc.)
> - `06-dp-on-trees.md` - Specialized: Tree DP problems (House Robber III, etc.)
> - `07-practice-list.md` - Complete practice list with all categories

---

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE DYNAMIC PROGRAMMING                          │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Count the number of ways"                                               │
│  ✓ "Minimum/Maximum cost/path/sum"                                          │
│  ✓ "Is it possible to..."                                                   │
│  ✓ "Longest/Shortest sequence"                                              │
│  ✓ "Optimal solution"                                                       │
│  ✓ "Partition into..."                                                      │
│                                                                             │
│  Two key properties:                                                        │
│  1. OVERLAPPING SUBPROBLEMS - Same subproblems solved multiple times        │
│  2. OPTIMAL SUBSTRUCTURE - Optimal solution built from optimal subsolutions │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning DP, ensure you understand:
- [ ] Recursion and recursive thinking
- [ ] Time/Space complexity analysis
- [ ] Arrays and 2D arrays
- [ ] Hash maps (for memoization)

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DYNAMIC PROGRAMMING MEMORY MAP                           │
│                                                                             │
│                    ┌─────────────────────┐                                  │
│         ┌─────────│ DYNAMIC PROGRAMMING │─────────┐                         │
│         │         └─────────────────────┘         │                         │
│         ▼                                         ▼                         │
│  ┌─────────────┐                           ┌─────────────┐                  │
│  │  1D DP      │                           │  2D DP      │                  │
│  └──────┬──────┘                           └──────┬──────┘                  │
│         │                                         │                         │
│    ┌────┴────┬────────┐              ┌───────────┼───────────┐              │
│    ▼         ▼        ▼              ▼           ▼           ▼              │
│ ┌──────┐ ┌──────┐ ┌──────┐    ┌──────────┐ ┌──────────┐ ┌────────┐         │
│ │Linear│ │Knap- │ │State │    │  Grid    │ │  String  │ │Interval│         │
│ │DP    │ │sack  │ │Machine│   │  DP      │ │   DP     │ │   DP   │         │
│ └──────┘ └──────┘ └──────┘    └──────────┘ └──────────┘ └────────┘         │
│                                                                             │
│  Examples by Category:                                                      │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Category     │ Problems                                            │    │
│  ├──────────────┼─────────────────────────────────────────────────────┤    │
│  │ Linear       │ Climbing Stairs, House Robber, Max Subarray         │    │
│  │ Knapsack     │ Coin Change, Partition Equal Subset, Target Sum     │    │
│  │ State Machine│ Buy/Sell Stock series                               │    │
│  │ Grid         │ Unique Paths, Min Path Sum, Dungeon Game            │    │
│  │ String       │ LCS, Edit Distance, Palindrome Subsequence          │    │
│  │ Interval     │ Matrix Chain, Burst Balloons                        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Related Patterns:                                                          │
│  • Recursion - DP is optimized recursion                                    │
│  • Greedy - When local optimal = global optimal (no DP needed)              │
│  • Divide & Conquer - Non-overlapping subproblems                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DP PROBLEM DECISION TREE                                 │
│                                                                             │
│  Can the problem be broken into subproblems?                                │
│       │                                                                     │
│       ├── NO → Not DP, try other approaches                                 │
│       │                                                                     │
│       └── YES → Are subproblems overlapping?                                │
│                    │                                                        │
│                    ├── NO → Divide & Conquer (no memoization needed)        │
│                    │                                                        │
│                    └── YES → DP! Now choose approach:                       │
│                                                                             │
│  DP Approach Selection:                                                     │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                                                                    │    │
│  │  Top-Down (Memoization)          vs    Bottom-Up (Tabulation)      │    │
│  │  ├─ More intuitive                     ├─ Usually faster           │    │
│  │  ├─ Only computes needed states        ├─ No recursion overhead    │    │
│  │  ├─ Risk of stack overflow             ├─ Easier space optimization│    │
│  │  └─ Good for complex transitions       └─ Good for simple patterns │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Dimension Selection:                                                       │
│       │                                                                     │
│       ├── Single sequence/value → 1D DP (dp[i])                             │
│       │                                                                     │
│       ├── Two sequences → 2D DP (dp[i][j])                                  │
│       │                                                                     │
│       ├── Grid problem → 2D DP (dp[row][col])                               │
│       │                                                                     │
│       └── Multiple states → Multi-dimensional (dp[i][j][k]...)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concept

DP = Recursion + Memoization (or Bottom-Up Iteration)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FIBONACCI EXAMPLE                                        │
│                                                                             │
│  Naive Recursion (O(2^n)):                                                  │
│                                                                             │
│                    fib(5)                                                   │
│                   /      \                                                  │
│              fib(4)      fib(3)     ← fib(3) computed twice!               │
│             /    \       /    \                                             │
│         fib(3) fib(2) fib(2) fib(1)  ← fib(2) computed 3 times!            │
│         /   \                                                               │
│     fib(2) fib(1)                                                           │
│                                                                             │
│  With Memoization (O(n)):                                                   │
│  Store results! Each fib(k) computed only ONCE.                             │
│                                                                             │
│  Bottom-Up (O(n)):                                                          │
│  Build from base cases: fib(0)=0, fib(1)=1, fib(2)=1, fib(3)=2, ...        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The DP Framework

### Step 1: Define the State
What information do we need to describe a subproblem?

```python
# Example: Climbing Stairs
# State: dp[i] = number of ways to reach step i

# Example: Coin Change
# State: dp[amount] = minimum coins needed for this amount

# Example: Longest Common Subsequence
# State: dp[i][j] = LCS of first i chars of s1 and first j chars of s2
```

### Step 2: Define the Recurrence Relation
How does the current state relate to previous states?

```python
# Climbing Stairs: dp[i] = dp[i-1] + dp[i-2]
# (can reach step i from step i-1 or step i-2)

# Coin Change: dp[amount] = min(dp[amount - coin] + 1) for each coin
# (try each coin, take minimum)

# LCS: dp[i][j] = dp[i-1][j-1] + 1 if match, else max(dp[i-1][j], dp[i][j-1])
```

### Step 3: Define Base Cases
What are the trivial subproblems?

```python
# Climbing Stairs: dp[0] = 1, dp[1] = 1
# Coin Change: dp[0] = 0 (0 coins for amount 0)
# LCS: dp[0][j] = 0, dp[i][0] = 0 (empty string has LCS 0)
```

### Step 4: Determine Computation Order
Bottom-up: Compute smaller subproblems first.

---

## Two Approaches

### Approach 1: Top-Down (Memoization)

```python
def fibonacci_memo(n: int, memo: dict = None) -> int:
    """
    Top-down DP with memoization.

    Strategy:
    - Start from the target problem
    - Recursively solve subproblems
    - Cache results to avoid recomputation

    Time: O(n) - each subproblem solved once
    Space: O(n) - memoization table + recursion stack
    """
    if memo is None:
        memo = {}

    # Check if already computed
    if n in memo:
        return memo[n]

    # Base cases
    if n <= 1:
        return n

    # Recursive case with memoization
    memo[n] = fibonacci_memo(n - 1, memo) + fibonacci_memo(n - 2, memo)
    return memo[n]


# Using @lru_cache decorator (cleaner)
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cached(n: int) -> int:
    """Fibonacci with automatic memoization."""
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)
```

### Approach 2: Bottom-Up (Tabulation)

```python
def fibonacci_bottom_up(n: int) -> int:
    """
    Bottom-up DP with tabulation.

    Strategy:
    - Start from base cases
    - Build up to target problem
    - Store results in table

    Time: O(n)
    Space: O(n)
    """
    if n <= 1:
        return n

    # Create DP table
    dp = [0] * (n + 1)

    # Base cases
    dp[0] = 0
    dp[1] = 1

    # Fill table bottom-up
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    return dp[n]


def fibonacci_optimized(n: int) -> int:
    """
    Space-optimized bottom-up DP.

    Only need last 2 values, not entire table!

    Time: O(n)
    Space: O(1)
    """
    if n <= 1:
        return n

    prev2 = 0  # fib(i-2)
    prev1 = 1  # fib(i-1)

    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current

    return prev1
```

---

## Common DP Patterns

### Pattern 1: Linear DP (1D)

```python
def climbing_stairs(n: int) -> int:
    """
    Count ways to climb n stairs (1 or 2 steps at a time).

    State: dp[i] = ways to reach step i
    Recurrence: dp[i] = dp[i-1] + dp[i-2]
    Base: dp[0] = 1, dp[1] = 1

    Time: O(n), Space: O(1)
    """
    if n <= 1:
        return 1

    prev2, prev1 = 1, 1

    for i in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current

    return prev1
```

### Pattern 2: Grid DP (2D)

```python
def unique_paths(m: int, n: int) -> int:
    """
    Count unique paths in m×n grid from top-left to bottom-right.
    Can only move right or down.

    State: dp[i][j] = ways to reach cell (i, j)
    Recurrence: dp[i][j] = dp[i-1][j] + dp[i][j-1]
    Base: dp[0][j] = 1, dp[i][0] = 1 (only one way along edges)

    Time: O(m*n), Space: O(n)
    """
    # Space optimization: only need previous row
    dp = [1] * n

    for i in range(1, m):
        for j in range(1, n):
            dp[j] = dp[j] + dp[j - 1]  # dp[j] is from above, dp[j-1] is from left

    return dp[n - 1]
```

### Pattern 3: String DP

```python
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    Find length of longest common subsequence.

    State: dp[i][j] = LCS of text1[:i] and text2[:j]
    Recurrence:
    - If text1[i-1] == text2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
    - Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    Time: O(m*n), Space: O(m*n) or O(min(m,n))
    """
    m, n = len(text1), len(text2)

    # dp[i][j] = LCS of text1[:i] and text2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                # Characters match, extend LCS
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # Take max of excluding one character
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]
```

### Pattern 4: Decision DP (Take or Skip)

```python
def house_robber(nums: list[int]) -> int:
    """
    Maximum money from non-adjacent houses.

    State: dp[i] = max money from first i houses
    Decision: For house i, either rob it or skip it
    Recurrence: dp[i] = max(dp[i-1], dp[i-2] + nums[i])

    Time: O(n), Space: O(1)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    prev2 = 0          # dp[i-2]
    prev1 = nums[0]    # dp[i-1]

    for i in range(1, len(nums)):
        # Decision: skip house i (prev1) or rob it (prev2 + nums[i])
        current = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = current

    return prev1
```

### Pattern 5: Knapsack DP

```python
def knapsack_01(weights: list[int], values: list[int], capacity: int) -> int:
    """
    0/1 Knapsack: Maximize value with weight limit.
    Each item can be taken at most once.

    State: dp[i][w] = max value using first i items with capacity w
    Recurrence:
    - Skip item i: dp[i][w] = dp[i-1][w]
    - Take item i: dp[i][w] = dp[i-1][w-weight[i]] + value[i]
    - dp[i][w] = max of above two

    Time: O(n*W), Space: O(W)
    """
    n = len(weights)

    # Space-optimized: only need previous row
    dp = [0] * (capacity + 1)

    for i in range(n):
        # Traverse backwards to avoid using updated values
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]
```

---

## Visual: DP Table for LCS

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LCS("ABCD", "AEBD")                                       │
│                                                                             │
│        ""  A  E  B  D                                                       │
│    ""   0  0  0  0  0                                                       │
│    A    0  1  1  1  1    ← A matches A                                      │
│    B    0  1  1  2  2    ← B matches B                                      │
│    C    0  1  1  2  2    ← C doesn't match                                  │
│    D    0  1  1  2  3    ← D matches D                                      │
│                     ↑                                                       │
│              Answer: LCS = 3 ("ABD")                                        │
│                                                                             │
│  Fill order: left-to-right, top-to-bottom                                   │
│  Each cell depends on: top, left, and top-left                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Mistakes

```python
# ❌ WRONG: Not handling base cases
def climb_stairs_wrong(n):
    dp = [0] * (n + 1)
    for i in range(2, n + 1):  # Missing base cases!
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# ✅ CORRECT: Handle base cases
def climb_stairs_correct(n):
    if n <= 1:
        return 1
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1  # Base cases
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]


# ❌ WRONG: Wrong iteration order for space optimization
def knapsack_wrong(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for w in range(weights[i], capacity + 1):  # Forward iteration
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
            # Bug! dp[w - weights[i]] might already be updated
    return dp[capacity]

# ✅ CORRECT: Backward iteration for 0/1 knapsack
def knapsack_correct(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):  # Backward!
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]
```

---

## DP Problem Categories

| Category | Example Problems |
|----------|------------------|
| Linear | Climbing Stairs, House Robber, Max Subarray |
| Grid | Unique Paths, Minimum Path Sum, Dungeon Game |
| String | LCS, Edit Distance, Palindrome |
| Knapsack | 0/1 Knapsack, Coin Change, Partition Equal Subset |
| Interval | Matrix Chain, Burst Balloons, Palindrome Partitioning |
| Tree | House Robber III, Binary Tree Maximum Path Sum |
| State Machine | Best Time to Buy and Sell Stock series |

---

## Complexity Analysis

| Problem Type | Time | Space | Space Optimized |
|--------------|------|-------|-----------------|
| Linear 1D (Fibonacci, Stairs) | O(n) | O(n) | O(1) |
| Knapsack (0/1) | O(n × W) | O(n × W) | O(W) |
| Knapsack (Unbounded) | O(n × W) | O(W) | O(W) |
| Grid (Unique Paths) | O(m × n) | O(m × n) | O(n) |
| String (LCS, Edit Distance) | O(m × n) | O(m × n) | O(min(m,n)) |
| Interval (Matrix Chain) | O(n³) | O(n²) | - |
| Bitmask DP | O(n² × 2^n) | O(n × 2^n) | - |

**Key Insight**: Many 2D DP problems can be space-optimized to 1D by only keeping the previous row.

---

## Interview Tips

### 1. How to Explain Your Approach
```
"I'll use dynamic programming because this problem has:
1. Overlapping subproblems - we'll compute the same states multiple times
2. Optimal substructure - optimal solution uses optimal sub-solutions

My approach:
- State: dp[i] represents [what it means]
- Transition: dp[i] = [recurrence relation]
- Base case: dp[0] = [initial value]
- Answer: dp[n] or max(dp)"
```

### 2. What Interviewers Look For
- **State definition**: Clear, complete, and minimal
- **Recurrence relation**: Correct transition logic
- **Base cases**: All edge cases handled
- **Complexity analysis**: Time and space
- **Space optimization**: Can you reduce space?

### 3. Common Follow-up Questions
- "Can you optimize space?" → Use rolling array or variables
- "Can you reconstruct the solution?" → Track choices or backtrack
- "What if constraints are larger?" → Consider different state representation
- "Can you do this iteratively?" → Convert top-down to bottom-up

### 4. DP Problem-Solving Template
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DP PROBLEM-SOLVING STEPS                                                   │
│                                                                             │
│  1. IDENTIFY DP                                                             │
│     • Optimization (min/max) or counting problem?                           │
│     • Overlapping subproblems?                                              │
│     • Optimal substructure?                                                 │
│                                                                             │
│  2. DEFINE STATE                                                            │
│     • What information defines a subproblem?                                │
│     • dp[i] = ? or dp[i][j] = ?                                             │
│     • State should be minimal but complete                                  │
│                                                                             │
│  3. FIND RECURRENCE                                                         │
│     • How does dp[i] relate to smaller subproblems?                         │
│     • Consider all choices at current step                                  │
│     • dp[i] = f(dp[i-1], dp[i-2], ...)                                      │
│                                                                             │
│  4. IDENTIFY BASE CASES                                                     │
│     • What are the smallest subproblems?                                    │
│     • dp[0] = ?, dp[1] = ?                                                  │
│                                                                             │
│  5. DETERMINE ORDER                                                         │
│     • Which states must be computed first?                                  │
│     • Usually: smaller indices before larger                                │
│                                                                             │
│  6. OPTIMIZE (if needed)                                                    │
│     • Can we use rolling array?                                             │
│     • Can we use O(1) variables?                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Related Patterns

- **Recursion**: DP is optimized recursion with memoization
- **Greedy**: When local optimal leads to global optimal, greedy is simpler
- **Divide and Conquer**: DP handles overlapping subproblems; D&C doesn't

### When to Combine

- **DP + Binary Search**: Optimize DP transitions (e.g., LIS in O(n log n))
- **DP + Bitmask**: Track subset states (e.g., Traveling Salesman)
- **DP on Trees**: Combine tree traversal with DP states

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
- [05-dp-on-strings.md](./05-dp-on-strings.md) - String DP patterns
- [06-dp-on-trees.md](./06-dp-on-trees.md) - Tree DP patterns
