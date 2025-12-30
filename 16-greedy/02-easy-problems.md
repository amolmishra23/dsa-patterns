# Greedy - Practice Problems

## Problem 1: Jump Game (LC #55) - Medium

- [LeetCode](https://leetcode.com/problems/jump-game/)

### Problem Statement
Given array where `nums[i]` is max jump length at position i. Determine if you can reach the last index.

### Examples
```
Input: nums = [2,3,1,1,4]
Output: true (0→1→4 or 0→2→3→4)

Input: nums = [3,2,1,0,4]
Output: false (stuck at index 3)
```

### Video Explanation
- [NeetCode - Jump Game](https://www.youtube.com/watch?v=Yan0cv2cLy8)

### Intuition
```
Greedy: Track the FARTHEST reachable position!

At each index, update how far we can reach.
If current index > farthest, we're stuck!

Visual: nums = [2,3,1,1,4]

        Index:    0   1   2   3   4
        Value:    2   3   1   1   4

        i=0: farthest = max(0, 0+2) = 2
        i=1: farthest = max(2, 1+3) = 4  ← can reach end!

        Early exit: farthest >= 4, return true

Visual: nums = [3,2,1,0,4]

        Index:    0   1   2   3   4
        Value:    3   2   1   0   4

        i=0: farthest = 3
        i=1: farthest = max(3, 1+2) = 3
        i=2: farthest = max(3, 2+1) = 3
        i=3: farthest = max(3, 3+0) = 3
        i=4: 4 > 3, can't reach index 4!

        Return false
```

### Solution
```python
def canJump(nums: list[int]) -> bool:
    """
    Check if we can reach the last index.

    Greedy insight: Track the farthest reachable position.
    If current position > farthest, we're stuck.

    Time: O(n)
    Space: O(1)
    """
    farthest = 0  # Farthest index we can reach

    for i in range(len(nums)):
        # If current position is beyond what we can reach, fail
        if i > farthest:
            return False

        # Update farthest reachable from current position
        farthest = max(farthest, i + nums[i])

        # Early exit if we can already reach the end
        if farthest >= len(nums) - 1:
            return True

    return True
```

### Edge Cases
- Single element → always True (already at end)
- First element is 0 → False (can't move)
- All zeros except first → depends on first value
- Large jump at start → might reach end immediately
- Array of all same values → depends on value

---

## Problem 2: Jump Game II (LC #45) - Medium

- [LeetCode](https://leetcode.com/problems/jump-game-ii/)

### Problem Statement
Minimum number of jumps to reach the last index.

### Examples
```
Input: nums = [2,3,1,1,4]
Output: 2 (0→1→4)
```

### Video Explanation
- [NeetCode - Jump Game II](https://www.youtube.com/watch?v=dJ7sWiOoK7g)

### Intuition
```
Think of it as BFS levels!

Level 0: Starting position (index 0)
Level 1: All positions reachable in 1 jump
Level 2: All positions reachable in 2 jumps
...

Visual: nums = [2,3,1,1,4]

        Index:    0   1   2   3   4
        Value:    2   3   1   1   4

        Level 0: [0]           (start)
        Level 1: [1, 2]        (reachable from 0)
        Level 2: [3, 4]        (reachable from level 1)

        Reached index 4 at level 2 → 2 jumps

Algorithm:
- Track current level's end and farthest reachable
- When we pass current level's end, increment jumps
- Update level's end to farthest reachable
```

### Solution
```python
def jump(nums: list[int]) -> int:
    """
    Minimum jumps to reach end.

    Greedy insight: At each "level", find the farthest we can reach.
    When we exhaust current level, we must jump.

    Think of it as BFS levels:
    - Level 0: index 0
    - Level 1: all indices reachable from level 0
    - Level 2: all indices reachable from level 1
    - etc.

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

### Edge Cases
- Single element → 0 jumps
- Two elements → 1 jump (guaranteed reachable)
- Can reach end in one jump → return 1
- Need to jump from every position → n-1 jumps
- Large values allow skipping many positions

---

## Problem 3: Gas Station (LC #134) - Medium

- [LeetCode](https://leetcode.com/problems/gas-station/)

### Problem Statement
Circular route with gas stations. Can you complete the circuit?

### Examples
```
Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3 (start at station 3)
```

### Video Explanation
- [NeetCode - Gas Station](https://www.youtube.com/watch?v=lJwbPZGo05A)

### Intuition
```
Two key insights:
1. If total gas >= total cost, a solution EXISTS
2. If we fail at station i, we must start AFTER i

Visual: gas = [1,2,3,4,5], cost = [3,4,5,1,2]

        Station:   0    1    2    3    4
        Gas:       1    2    3    4    5
        Cost:      3    4    5    1    2
        Net:      -2   -2   -2   +3   +3

        Total net = -2-2-2+3+3 = 0 ≥ 0, solution exists!

        Try starting at 0: tank = -2 < 0, fail
        Try starting at 1: tank = -2 < 0, fail
        Try starting at 2: tank = -2 < 0, fail
        Try starting at 3:
          - Station 3: tank = 4-1 = 3
          - Station 4: tank = 3+5-2 = 6
          - Station 0: tank = 6+1-3 = 4
          - Station 1: tank = 4+2-4 = 2
          - Station 2: tank = 2+3-5 = 0 ≥ 0 ✓

        Answer: 3
```

### Solution
```python
def canCompleteCircuit(gas: list[int], cost: list[int]) -> int:
    """
    Find starting station to complete circular route.

    Greedy insights:
    1. If total gas >= total cost, solution exists
    2. If we run out of gas at station i, start must be after i
       (any station before i would also fail at i)

    Time: O(n)
    Space: O(1)
    """
    total_tank = 0    # Total gas - cost for entire trip
    current_tank = 0  # Current tank from candidate start
    start = 0         # Candidate starting station

    for i in range(len(gas)):
        diff = gas[i] - cost[i]
        total_tank += diff
        current_tank += diff

        # If tank goes negative, can't start from 'start'
        # Try starting from next station
        if current_tank < 0:
            start = i + 1
            current_tank = 0

    # If total is non-negative, we can complete from 'start'
    return start if total_tank >= 0 else -1
```

### Edge Cases
- Single station → check if gas[0] >= cost[0]
- All gas equals all cost → any start works
- Total gas < total cost → return -1
- Only one valid starting point → return it
- All same gas and cost values → depends on individual values

---

## Problem 4: Assign Cookies (LC #455) - Easy

- [LeetCode](https://leetcode.com/problems/assign-cookies/)

### Problem Statement
Maximize children satisfied with cookies. Child i needs cookie of size >= g[i].

### Examples
```
Input: g = [1,2,3], s = [1,1]
Output: 1 (only child with greed 1 satisfied)

Input: g = [1,2], s = [1,2,3]
Output: 2 (both children satisfied)
```

### Video Explanation
- [NeetCode - Assign Cookies](https://www.youtube.com/watch?v=DIX2p7vb9co)

### Intuition
```
Greedy: Give smallest sufficient cookie to least greedy child!

Sort both arrays. Match smallest cookie that satisfies each child.

Visual: g = [1,2,3], s = [1,1]

        Children (sorted): [1, 2, 3]
        Cookies (sorted):  [1, 1]

        Child 1 (greed=1): Cookie 1 (size=1) works! ✓
        Child 2 (greed=2): Cookie 2 (size=1) too small ✗
        No more cookies.

        Answer: 1 child satisfied

Visual: g = [1,2], s = [1,2,3]

        Children (sorted): [1, 2]
        Cookies (sorted):  [1, 2, 3]

        Child 1 (greed=1): Cookie 1 (size=1) works! ✓
        Child 2 (greed=2): Cookie 2 (size=2) works! ✓

        Answer: 2 children satisfied
```

### Solution
```python
def findContentChildren(g: list[int], s: list[int]) -> int:
    """
    Maximize children satisfied with cookies.

    Greedy insight: Give smallest sufficient cookie to each child.
    Sort both, match greedily.

    Time: O(n log n + m log m)
    Space: O(1) - sorting in place
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

### Edge Cases
- No children → return 0
- No cookies → return 0
- All cookies smaller than all children → return 0
- All cookies larger than all children → return min(children, cookies)
- Exact matches for all → return min(children, cookies)

---

## Problem 5: Lemonade Change (LC #860) - Easy

- [LeetCode](https://leetcode.com/problems/lemonade-change/)

### Problem Statement
Customers pay $5, $10, or $20 for $5 lemonade. Can you give correct change?

### Video Explanation
- [NeetCode - Lemonade Change](https://www.youtube.com/watch?v=n_tmibEhO6Q)

### Intuition
```
Greedy: For $20, prefer giving $10+$5 over $5+$5+$5!

Why? $5 bills are more flexible (can make change for $10 or $20).

Visual: bills = [5,5,5,10,20]

        Customer 1 pays $5:  five=1, ten=0
        Customer 2 pays $5:  five=2, ten=0
        Customer 3 pays $5:  five=3, ten=0
        Customer 4 pays $10: give $5, five=2, ten=1
        Customer 5 pays $20: give $10+$5, five=1, ten=0 ✓

        All customers served!

Counter-example: bills = [5,5,10,10,20]

        Customer 1 pays $5:  five=1, ten=0
        Customer 2 pays $5:  five=2, ten=0
        Customer 3 pays $10: give $5, five=1, ten=1
        Customer 4 pays $10: give $5, five=0, ten=2
        Customer 5 pays $20: need $15, have $20 in tens, no $5! ✗
```

### Solution
```python
def lemonadeChange(bills: list[int]) -> bool:
    """
    Check if we can give correct change to all customers.

    Greedy insight: For $20, prefer giving $10+$5 over $5+$5+$5
    (save $5 bills for flexibility)

    Time: O(n)
    Space: O(1)
    """
    five = ten = 0

    for bill in bills:
        if bill == 5:
            # No change needed
            five += 1

        elif bill == 10:
            # Need $5 change
            if five == 0:
                return False
            five -= 1
            ten += 1

        else:  # bill == 20
            # Need $15 change
            # Prefer $10 + $5 over $5 + $5 + $5
            if ten > 0 and five > 0:
                ten -= 1
                five -= 1
            elif five >= 3:
                five -= 3
            else:
                return False

    return True
```

### Edge Cases
- First customer pays $5 → always works
- First customer pays $10 or $20 → False (no change)
- All $5 bills → always True
- Many $20 bills in a row → need enough $5s and $10s
- Alternating $5 and $10 → works

---

## Problem 6: Best Time to Buy and Sell Stock II (LC #122) - Medium

- [LeetCode](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

### Problem Statement
Multiple transactions allowed. Find maximum profit.

### Video Explanation
- [NeetCode - Best Time to Buy and Sell Stock II](https://www.youtube.com/watch?v=3SJ3pUkPQMc)

### Intuition
```
Greedy: Capture EVERY upward movement!

If tomorrow's price > today's price, buy today and sell tomorrow.
(Or equivalently: add every positive difference)

Visual: prices = [7,1,5,3,6,4]

        7   1   5   3   6   4
        │   │   │   │   │   │
        ▼   ▼   ▲   ▼   ▲   ▼

        Buy at 1, sell at 5: profit = 4
        Buy at 3, sell at 6: profit = 3

        Total: 4 + 3 = 7

Simplified view: Just sum all positive differences!
        1→5: +4
        5→3: -2 (skip)
        3→6: +3
        6→4: -2 (skip)

        Total: 4 + 3 = 7
```

### Solution
```python
def maxProfit(prices: list[int]) -> int:
    """
    Maximum profit with unlimited transactions.

    Greedy insight: Capture every upward movement.
    If tomorrow > today, buy today and sell tomorrow.

    Time: O(n)
    Space: O(1)
    """
    profit = 0

    for i in range(1, len(prices)):
        # If price goes up, capture the gain
        if prices[i] > prices[i - 1]:
            profit += prices[i] - prices[i - 1]

    return profit
```

### Edge Cases
- Single day → return 0 (no transaction)
- Monotonically increasing → sum all differences
- Monotonically decreasing → return 0
- All same prices → return 0
- Alternating up/down → capture all upward movements

---

## Problem 7: Partition Labels (LC #763) - Medium

- [LeetCode](https://leetcode.com/problems/partition-labels/)

### Problem Statement
Partition string so each letter appears in at most one part. Maximize number of parts.

### Examples
```
Input: s = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation: "ababcbaca", "defegde", "hijhklij"
```

### Video Explanation
- [NeetCode - Partition Labels](https://www.youtube.com/watch?v=B7m8UmZE-vw)

### Intuition
```
Greedy: Find last occurrence of each char, extend partition!

Key insight: If char 'a' appears at index 0 and last at index 8,
the first partition must include at least indices 0-8.

Visual: s = "ababcbacadefegdehijhklij"

        Last occurrences:
        a:8, b:5, c:7, d:14, e:15, f:11, g:13, h:19, i:22, j:23, k:20, l:21

        i=0 (a): end = max(0, 8) = 8
        i=1 (b): end = max(8, 5) = 8
        ...
        i=8 (a): end = 8, i == end → partition! length = 9

        i=9 (d): end = max(9, 14) = 14
        ...
        i=15 (e): end = 15, i == end → partition! length = 7

        Continue for last partition...

        Result: [9, 7, 8]
```

### Solution
```python
def partitionLabels(s: str) -> list[int]:
    """
    Partition string so each letter appears in at most one part.

    Greedy insight:
    1. Find last occurrence of each character
    2. Extend partition until we've included all last occurrences

    Time: O(n)
    Space: O(26) = O(1)
    """
    # Find last occurrence of each character
    last_occurrence = {char: i for i, char in enumerate(s)}

    result = []
    start = 0
    end = 0

    for i, char in enumerate(s):
        # Extend end to include last occurrence of current char
        end = max(end, last_occurrence[char])

        # If we've reached the end of current partition
        if i == end:
            result.append(end - start + 1)
            start = end + 1

    return result
```

### Edge Cases
- Single character → return [1]
- All same characters → return [len(s)]
- All unique characters → return [1] * len(s)
- Two characters interleaved → depends on last occurrences
- Empty string → return []

---

## Problem 8: Task Scheduler (LC #621) - Medium

- [LeetCode](https://leetcode.com/problems/task-scheduler/)

### Problem Statement
Schedule tasks with cooldown period n between same tasks. Find minimum time.

### Video Explanation
- [NeetCode - Task Scheduler](https://www.youtube.com/watch?v=s8p8ukTyA2I)

### Intuition
```
Greedy: Most frequent task determines minimum time!

Formula: (max_freq - 1) * (n + 1) + count_of_max_freq

Visual: tasks = [A,A,A,B,B,B], n = 2

        A appears 3 times (max_freq = 3)
        B appears 3 times (count_of_max_freq = 2)

        Pattern:
        A B _ | A B _ | A B
        └──┬──┘ └──┬──┘
         n+1=3    n+1=3

        (3-1) * 3 + 2 = 8 time units

        But if we have many tasks, no idle time needed:
        tasks = [A,A,A,B,B,B,C,C,C,D,D,D], n = 2

        A B C D | A B C D | A B C D

        Total = 12 = len(tasks)

Answer = max(len(tasks), formula)
```

### Solution
```python
from collections import Counter

def leastInterval(tasks: list[str], n: int) -> int:
    """
    Minimum time to complete all tasks with cooldown.

    Greedy insight: Most frequent task determines minimum time.

    If max_freq task appears f times, we need at least:
    (f - 1) * (n + 1) + count_of_max_freq_tasks

    Example: tasks = [A,A,A,B,B,B], n = 2
    Pattern: A B _ A B _ A B
    Time: (3-1) * (2+1) + 2 = 8

    Time: O(n)
    Space: O(26) = O(1)
    """
    freq = Counter(tasks)
    max_freq = max(freq.values())

    # Count how many tasks have max frequency
    max_count = sum(1 for f in freq.values() if f == max_freq)

    # Calculate minimum time based on most frequent task
    # (max_freq - 1) intervals of size (n + 1), plus final batch
    min_time = (max_freq - 1) * (n + 1) + max_count

    # If we have more tasks than idle slots, no idle time needed
    return max(min_time, len(tasks))
```

### Edge Cases
- Single task → return 1
- n = 0 → no idle time needed
- All different tasks → return len(tasks)
- All same task → (count-1) * (n+1) + 1
- Empty tasks → return 0

---

## Problem 9: Queue Reconstruction by Height (LC #406) - Medium

- [LeetCode](https://leetcode.com/problems/queue-reconstruction-by-height/)

### Problem Statement
Reconstruct queue from (height, k) pairs where k = people taller in front.

### Video Explanation
- [NeetCode - Queue Reconstruction by Height](https://www.youtube.com/watch?v=khddrw6Bfyw)

### Solution
```python
def reconstructQueue(people: list[list[int]]) -> list[list[int]]:
    """
    Reconstruct queue from height and k-value.

    Greedy insight:
    1. Sort by height descending, then k ascending
    2. Insert each person at index k

    Why this works:
    - Taller people are placed first
    - When placing shorter person, taller ones don't affect their k
    - Insert at index k ensures exactly k taller people in front

    Time: O(n²) - n insertions, each O(n)
    Space: O(n)
    """
    # Sort: taller first, then by k
    people.sort(key=lambda x: (-x[0], x[1]))

    result = []
    for person in people:
        # Insert at index k
        result.insert(person[1], person)

    return result
```

### Edge Cases
- Single person → return [[h, k]]
- All same height → sort by k, insert at k
- All k = 0 → sort by height descending
- Two people same height different k → sort by k
- Empty input → return []

---

## Problem 10: Minimum Number of Arrows (LC #452) - Medium

- [LeetCode](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)

### Problem Statement
Minimum arrows to burst all balloons (intervals).

### Video Explanation
- [NeetCode - Minimum Number of Arrows to Burst Balloons](https://www.youtube.com/watch?v=lPmkKnvNPrw)

### Solution
```python
def findMinArrowShots(points: list[list[int]]) -> int:
    """
    Minimum arrows to burst all balloons.

    Greedy insight: Sort by end, shoot at end of first balloon.
    Skip all balloons hit by this arrow.

    Time: O(n log n)
    Space: O(1)
    """
    if not points:
        return 0

    # Sort by end position
    points.sort(key=lambda x: x[1])

    arrows = 1
    arrow_pos = points[0][1]

    for start, end in points[1:]:
        if start > arrow_pos:
            # Balloon not hit, need new arrow
            arrows += 1
            arrow_pos = end

    return arrows
```

### Edge Cases
- Empty points → return 0
- Single balloon → return 1
- All balloons overlap → return 1
- No balloons overlap → return n
- Balloons touching at one point → one arrow hits both

---

## Summary: Greedy Problems

| # | Problem | Key Insight | Time |
|---|---------|-------------|------|
| 1 | Jump Game | Track farthest reachable | O(n) |
| 2 | Jump Game II | BFS-like levels | O(n) |
| 3 | Gas Station | If total >= 0, solution exists | O(n) |
| 4 | Assign Cookies | Smallest sufficient | O(n log n) |
| 5 | Lemonade Change | Prefer $10 over $5s | O(n) |
| 6 | Buy/Sell Stock II | Capture all gains | O(n) |
| 7 | Partition Labels | Track last occurrence | O(n) |
| 8 | Task Scheduler | Max freq determines time | O(n) |
| 9 | Queue Reconstruction | Sort tall first, insert at k | O(n²) |
| 10 | Burst Balloons | Sort by end, greedy shoot | O(n log n) |

---

## Practice More Problems

- [ ] LC #135 - Candy
- [ ] LC #376 - Wiggle Subsequence
- [ ] LC #402 - Remove K Digits
- [ ] LC #621 - Task Scheduler
- [ ] LC #767 - Reorganize String

