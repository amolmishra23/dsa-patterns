# Greedy - Complete Practice List

## Organized by Pattern and Difficulty

### Pattern 1: Interval Scheduling

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 435 | [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/) | Medium | Sort by end, count kept |
| 452 | [Min Arrows to Burst Balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/) | Medium | Sort by end, greedy shoot |
| 56 | [Merge Intervals](https://leetcode.com/problems/merge-intervals/) | Medium | Sort by start, merge |
| 253 | [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/) | Medium | Sort + heap or sweep |
| 1353 | [Max Events That Can Be Attended](https://leetcode.com/problems/maximum-number-of-events-that-can-be-attended/) | Medium | Sort + heap |

### Pattern 2: Jump/Reach Problems

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 55 | [Jump Game](https://leetcode.com/problems/jump-game/) | Medium | Track max reach |
| 45 | [Jump Game II](https://leetcode.com/problems/jump-game-ii/) | Medium | BFS-like levels |
| 1024 | [Video Stitching](https://leetcode.com/problems/video-stitching/) | Medium | Sort + extend |
| 1326 | [Min Taps to Water Garden](https://leetcode.com/problems/minimum-number-of-taps-to-open-to-water-a-garden/) | Hard | Same as video stitching |

### Pattern 3: Task Scheduling

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 621 | [Task Scheduler](https://leetcode.com/problems/task-scheduler/) | Medium | Max frequency formula |
| 767 | [Reorganize String](https://leetcode.com/problems/reorganize-string/) | Medium | Max heap |
| 358 | [Rearrange String k Distance Apart](https://leetcode.com/problems/rearrange-string-k-distance-apart/) | Hard | Heap + cooldown |
| 984 | [String Without AAA or BBB](https://leetcode.com/problems/string-without-aaa-or-bbb/) | Medium | Greedy placement |

### Pattern 4: Stock Problems

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 121 | [Best Time to Buy and Sell Stock](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) | Easy | Track min, max profit |
| 122 | [Best Time II (Multiple)](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/) | Medium | Sum all increases |
| 714 | [Best Time with Fee](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/) | Medium | DP or greedy |
| 309 | [Best Time with Cooldown](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/) | Medium | State machine DP |

### Pattern 5: Array Manipulation

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 134 | [Gas Station](https://leetcode.com/problems/gas-station/) | Medium | Track deficit |
| 135 | [Candy](https://leetcode.com/problems/candy/) | Hard | Two-pass |
| 330 | [Patching Array](https://leetcode.com/problems/patching-array/) | Hard | Track reachable sum |
| 406 | [Queue by Height](https://leetcode.com/problems/queue-reconstruction-by-height/) | Medium | Sort + insert |
| 763 | [Partition Labels](https://leetcode.com/problems/partition-labels/) | Medium | Last occurrence |

### Pattern 6: String Manipulation

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 316 | [Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/) | Medium | Monotonic stack |
| 402 | [Remove K Digits](https://leetcode.com/problems/remove-k-digits/) | Medium | Monotonic stack |
| 321 | [Create Maximum Number](https://leetcode.com/problems/create-maximum-number/) | Hard | Merge two arrays |
| 1081 | [Smallest Subsequence](https://leetcode.com/problems/smallest-subsequence-of-distinct-characters/) | Medium | Same as 316 |

---

## Essential Templates

### 1. Interval Scheduling (Maximum Non-overlapping)
```python
def max_non_overlapping(intervals):
    """Sort by END time, greedily pick non-overlapping."""
    intervals.sort(key=lambda x: x[1])

    count = 0
    end = float('-inf')

    for start, finish in intervals:
        if start >= end:
            count += 1
            end = finish

    return count
```

### 2. Jump Game Pattern
```python
def can_jump(nums):
    """Track maximum reachable index."""
    max_reach = 0

    for i in range(len(nums)):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + nums[i])

    return True

def min_jumps(nums):
    """BFS-like level tracking."""
    jumps = 0
    current_end = 0
    farthest = 0

    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])

        if i == current_end:
            jumps += 1
            current_end = farthest

    return jumps
```

### 3. Gas Station Pattern
```python
def can_complete_circuit(gas, cost):
    """Find starting station for circular trip."""
    total_tank = 0
    current_tank = 0
    start = 0

    for i in range(len(gas)):
        diff = gas[i] - cost[i]
        total_tank += diff
        current_tank += diff

        if current_tank < 0:
            start = i + 1
            current_tank = 0

    return start if total_tank >= 0 else -1
```

### 4. Two-Pass Pattern (Candy)
```python
def candy(ratings):
    """Two-pass: left-to-right, then right-to-left."""
    n = len(ratings)
    candies = [1] * n

    # Left to right
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1

    # Right to left
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)

    return sum(candies)
```

### 5. Partition Labels Pattern
```python
def partition_labels(s):
    """Track last occurrence, extend partition."""
    last = {c: i for i, c in enumerate(s)}

    result = []
    start = end = 0

    for i, c in enumerate(s):
        end = max(end, last[c])

        if i == end:
            result.append(end - start + 1)
            start = i + 1

    return result
```

---

## When to Use Greedy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GREEDY DECISION GUIDE                                    │
│                                                                             │
│  USE GREEDY when:                                                           │
│  1. Local optimal choice leads to global optimal                            │
│  2. Problem has optimal substructure                                        │
│  3. Making a choice doesn't affect future choices' validity                 │
│                                                                             │
│  COMMON GREEDY STRATEGIES:                                                  │
│  • Sort by some criteria, then process                                      │
│  • Always pick the best available option                                    │
│  • Track running state (max reach, current sum, etc.)                       │
│                                                                             │
│  GREEDY vs DP:                                                              │
│  • Greedy: Make irrevocable choices, O(n) or O(n log n)                    │
│  • DP: Consider all choices, often O(n²) or O(n * state)                   │
│                                                                             │
│  If greedy doesn't work, try:                                               │
│  • Dynamic Programming                                                      │
│  • Backtracking                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Proof Techniques for Greedy

### 1. Exchange Argument
```
Show that swapping any two elements in the optimal solution
to match the greedy choice doesn't make it worse.
```

### 2. Greedy Stays Ahead
```
Show that at each step, greedy solution is at least as good
as any other solution.
```

### 3. Contradiction
```
Assume greedy is not optimal, show this leads to contradiction.
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] Jump Game
- [ ] Best Time to Buy and Sell Stock
- [ ] Best Time to Buy and Sell Stock II
- [ ] Gas Station
- [ ] Partition Labels

### Week 2: Intervals
- [ ] Non-overlapping Intervals
- [ ] Merge Intervals
- [ ] Min Arrows to Burst Balloons
- [ ] Meeting Rooms II

### Week 3: Advanced
- [ ] Jump Game II
- [ ] Candy
- [ ] Task Scheduler
- [ ] Remove K Digits
- [ ] Queue Reconstruction by Height

---

## Common Mistakes

1. **Assuming greedy works without proof**
   - Always verify with examples
   - Consider counterexamples

2. **Wrong sorting criteria**
   - Interval scheduling: sort by END, not start
   - Think about what ordering enables greedy choice

3. **Not handling edge cases**
   - Empty input
   - Single element
   - All same values

4. **Forgetting to track state**
   - Max reach in jump problems
   - Current deficit in gas station

---

## Complexity Reference

| Problem Type | Time | Space |
|--------------|------|-------|
| Interval scheduling | O(n log n) | O(1) |
| Jump game | O(n) | O(1) |
| Task scheduler | O(n) | O(1) |
| Two-pass (candy) | O(n) | O(n) |
| Monotonic stack | O(n) | O(n) |

