# Greedy Algorithms - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE GREEDY                                       │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Maximum/Minimum" with local choices                                     │
│  ✓ "Interval scheduling"                                                    │
│  ✓ "Activity selection"                                                     │
│  ✓ "Jump game"                                                              │
│  ✓ "Gas station"                                                            │
│  ✓ "Assign cookies"                                                         │
│                                                                             │
│  Key insight: Make locally optimal choice at each step                      │
│               hoping to find global optimum                                 │
│                                                                             │
│  WARNING: Greedy doesn't always work! Must prove correctness.               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Sorting algorithms
- [ ] Proof techniques (exchange argument)
- [ ] When greedy fails (coin change with arbitrary coins)

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GREEDY MEMORY MAP                                        │
│                                                                             │
│                    ┌─────────────┐                                          │
│         ┌─────────│   GREEDY    │─────────┐                                 │
│         │         └─────────────┘         │                                 │
│         ▼                                 ▼                                 │
│  ┌─────────────┐                   ┌─────────────┐                          │
│  │  INTERVAL   │                   │   ARRAY     │                          │
│  │  PROBLEMS   │                   │  PROBLEMS   │                          │
│  └──────┬──────┘                   └──────┬──────┘                          │
│         │                                 │                                 │
│    ┌────┴────┐                      ┌─────┴─────┐                           │
│    ▼         ▼                      ▼           ▼                           │
│ ┌──────┐ ┌──────┐               ┌──────┐   ┌──────┐                        │
│ │Activ-│ │Meeting│              │Jump  │   │Gas   │                        │
│ │ity   │ │Rooms │               │Game  │   │Station│                        │
│ └──────┘ └──────┘               └──────┘   └──────┘                        │
│                                                                             │
│  Related Patterns:                                                          │
│  • DP - When greedy doesn't work, try DP                                    │
│  • Sorting - Often need to sort first                                       │
│  • Intervals - Many interval problems are greedy                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GREEDY vs DP DECISION TREE                               │
│                                                                             │
│  Does local optimal lead to global optimal?                                 │
│       │                                                                     │
│       ├── YES (can prove) → Use Greedy                                      │
│       │                                                                     │
│       ├── UNSURE → Try greedy, verify with examples                         │
│       │            If counterexample found → Use DP                         │
│       │                                                                     │
│       └── NO → Use DP                                                       │
│                                                                             │
│  GREEDY WORKS:                     GREEDY FAILS:                            │
│  • Activity selection              • Coin change (arbitrary coins)          │
│  • Fractional knapsack             • 0/1 Knapsack                           │
│  • Huffman coding                  • Longest path in graph                  │
│  • Jump game (can reach end?)      • Traveling salesman                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concept

Greedy algorithms make the locally optimal choice at each step, hoping this leads to a globally optimal solution.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GREEDY VS DYNAMIC PROGRAMMING                            │
│                                                                             │
│  GREEDY:                                                                    │
│  - Make one choice, never look back                                         │
│  - Faster (usually O(n) or O(n log n))                                      │
│  - Doesn't always give optimal solution                                     │
│  - Need to PROVE it works                                                   │
│                                                                             │
│  DYNAMIC PROGRAMMING:                                                       │
│  - Consider all choices, pick best                                          │
│  - Slower (often O(n²) or worse)                                            │
│  - Always gives optimal solution                                            │
│  - Works when greedy fails                                                  │
│                                                                             │
│  Example: Coin Change                                                       │
│  Coins: [1, 3, 4], Amount: 6                                                │
│                                                                             │
│  Greedy: 4 + 1 + 1 = 3 coins ❌                                             │
│  Optimal: 3 + 3 = 2 coins ✓                                                 │
│                                                                             │
│  Greedy fails here! Need DP.                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Greedy Problems

### Problem 1: Jump Game (LC #55)

```python
def canJump(nums: list[int]) -> bool:
    """
    Can we reach the last index?

    Greedy insight: Track the farthest position we can reach.
    If current position > farthest, we're stuck.

    Time: O(n)
    Space: O(1)
    """
    farthest = 0  # Farthest index we can reach

    for i in range(len(nums)):
        # If we can't reach current position, we're stuck
        if i > farthest:
            return False

        # Update farthest we can reach from here
        farthest = max(farthest, i + nums[i])

        # Early exit if we can reach the end
        if farthest >= len(nums) - 1:
            return True

    return True
```

### Problem 2: Jump Game II (LC #45)

```python
def jump(nums: list[int]) -> int:
    """
    Minimum jumps to reach end.

    Greedy insight: At each "level", find the farthest we can reach.
    When we reach the end of current level, increment jumps.

    Time: O(n)
    Space: O(1)
    """
    if len(nums) <= 1:
        return 0

    jumps = 0
    current_end = 0    # End of current jump range
    farthest = 0       # Farthest we can reach

    for i in range(len(nums) - 1):  # Don't need to jump from last
        # Update farthest reachable
        farthest = max(farthest, i + nums[i])

        # If we've reached end of current jump range
        if i == current_end:
            jumps += 1
            current_end = farthest

            # Early exit if we can reach the end
            if current_end >= len(nums) - 1:
                break

    return jumps
```

### Problem 3: Gas Station (LC #134)

```python
def canCompleteCircuit(gas: list[int], cost: list[int]) -> int:
    """
    Find starting station to complete circuit.

    Greedy insight:
    1. If total gas >= total cost, solution exists
    2. If we run out of gas at station i, start must be after i

    Time: O(n)
    Space: O(1)
    """
    total_tank = 0   # Total gas - cost for entire circuit
    current_tank = 0 # Current tank from start point
    start = 0        # Candidate starting station

    for i in range(len(gas)):
        diff = gas[i] - cost[i]
        total_tank += diff
        current_tank += diff

        # If we run out of gas, start from next station
        if current_tank < 0:
            start = i + 1
            current_tank = 0

    # If total is non-negative, we can complete from 'start'
    return start if total_tank >= 0 else -1
```

### Problem 4: Interval Scheduling (LC #435)

```python
def eraseOverlapIntervals(intervals: list[list[int]]) -> int:
    """
    Minimum intervals to remove for non-overlapping.

    Greedy insight: Sort by end time, always keep interval
    that ends earliest (leaves most room for others).

    Time: O(n log n)
    Space: O(1)
    """
    if not intervals:
        return 0

    # Sort by end time
    intervals.sort(key=lambda x: x[1])

    count = 0          # Intervals to remove
    prev_end = float('-inf')

    for start, end in intervals:
        if start >= prev_end:
            # No overlap, keep this interval
            prev_end = end
        else:
            # Overlap, remove this interval (keep previous)
            count += 1

    return count
```

### Problem 5: Assign Cookies (LC #455)

```python
def findContentChildren(g: list[int], s: list[int]) -> int:
    """
    Maximize children with cookies.
    g[i] = greed of child i, s[j] = size of cookie j

    Greedy insight: Give smallest sufficient cookie to each child.
    Sort both, match greedily.

    Time: O(n log n + m log m)
    Space: O(1)
    """
    g.sort()  # Sort children by greed
    s.sort()  # Sort cookies by size

    child = 0
    cookie = 0

    while child < len(g) and cookie < len(s):
        if s[cookie] >= g[child]:
            # Cookie satisfies child
            child += 1
        # Move to next cookie regardless
        cookie += 1

    return child
```

---

## When Greedy Works

Greedy works when the problem has:

1. **Greedy Choice Property**: A locally optimal choice leads to a globally optimal solution.

2. **Optimal Substructure**: An optimal solution contains optimal solutions to subproblems.

### Proving Greedy Correctness

```
Common proof techniques:

1. EXCHANGE ARGUMENT:
   - Assume optimal solution differs from greedy
   - Show we can exchange to match greedy without losing optimality

2. STAYING AHEAD:
   - Show greedy is always at least as good as any other solution
   - At each step, greedy is "ahead" or equal

3. CONTRADICTION:
   - Assume greedy is not optimal
   - Derive a contradiction
```

---

## Visual: Interval Scheduling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTERVAL SCHEDULING GREEDY                               │
│                                                                             │
│  Intervals (sorted by end time):                                            │
│                                                                             │
│  [1,3]  ████                                                                │
│  [2,4]   ████                                                               │
│  [3,5]    ████                                                              │
│  [4,6]     ████                                                             │
│  [5,7]      ████                                                            │
│                                                                             │
│  Greedy selection:                                                          │
│  1. Take [1,3] (ends earliest)                                              │
│  2. Skip [2,4] (overlaps)                                                   │
│  3. Skip [3,5] (overlaps)                                                   │
│  4. Take [4,6] (doesn't overlap with [1,3])                                 │
│  5. Skip [5,7] (overlaps)                                                   │
│                                                                             │
│  Result: 2 intervals selected, 3 removed                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Greedy Patterns

| Pattern | Strategy | Example |
|---------|----------|---------|
| Interval | Sort by end, pick non-overlapping | Activity Selection |
| Fractional | Take most valuable per unit | Fractional Knapsack |
| Huffman | Combine smallest frequencies | Huffman Coding |
| MST | Pick smallest edge | Kruskal's, Prim's |
| Scheduling | Earliest deadline first | Job Scheduling |

---

## Common Mistakes

```python
# ❌ WRONG: Using greedy when DP is needed
def coinChange_greedy(coins, amount):
    coins.sort(reverse=True)
    count = 0
    for coin in coins:
        count += amount // coin
        amount %= coin
    return count if amount == 0 else -1
# Fails for coins=[1,3,4], amount=6 (gives 3, optimal is 2)

# ✅ CORRECT: Use DP for coin change
def coinChange_dp(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i:
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

---

## Complexity Analysis

Most greedy algorithms:
- **Time**: O(n) or O(n log n) if sorting needed
- **Space**: O(1) or O(n) for storing result

---

## Interview Tips

### 1. How to Explain Your Approach
```
"I believe greedy works here because making the locally optimal choice
at each step leads to global optimum. The key insight is [explain why].
Let me verify with an example..."
```

### 2. What Interviewers Look For
- **Correctness proof**: Can you justify why greedy works?
- **Counterexample awareness**: Know when greedy fails
- **Sorting strategy**: What to sort by and why

### 3. Common Follow-up Questions
- "How do you know greedy works?" → Provide proof or counterexample
- "What if greedy doesn't work?" → Fall back to DP
- "Can you optimize?" → Greedy is often already optimal

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
