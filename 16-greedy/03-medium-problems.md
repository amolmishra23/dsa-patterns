# Greedy - Advanced Problems

## When Does Greedy Work?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GREEDY ALGORITHM CONDITIONS                              │
│                                                                             │
│  Greedy works when:                                                         │
│  1. GREEDY CHOICE PROPERTY: Local optimal leads to global optimal           │
│  2. OPTIMAL SUBSTRUCTURE: Optimal solution contains optimal sub-solutions   │
│                                                                             │
│  Common greedy strategies:                                                  │
│  • Sort by some criteria, then process                                      │
│  • Always pick the best available option                                    │
│  • Interval scheduling: sort by end time                                    │
│  • Huffman coding: combine smallest first                                   │
│                                                                             │
│  If greedy doesn't work, consider:                                          │
│  • Dynamic Programming                                                      │
│  • Backtracking                                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem 1: Candy (LC #135) - Hard

- [LeetCode](https://leetcode.com/problems/candy/)

### Problem Statement
Distribute candy to children with ratings. Each child gets at least 1. Higher rating than neighbor means more candy.

### Examples
```
Input: ratings = [1,0,2]
Output: 5 (candies: [2,1,2])

Input: ratings = [1,2,2]
Output: 4 (candies: [1,2,1])
```

### Intuition Development
```
TWO-PASS APPROACH:
Need to satisfy both left and right neighbors.

Pass 1 (Left to Right):
  ratings: [1, 0, 2]
  candies: [1, 1, 1] → initial

  i=1: rating[1]=0 < rating[0]=1 → no change
  i=2: rating[2]=2 > rating[1]=0 → candy[2] = candy[1]+1 = 2
  candies: [1, 1, 2]

Pass 2 (Right to Left):
  i=1: rating[1]=0 < rating[2]=2 → no change
  i=0: rating[0]=1 > rating[1]=0 → candy[0] = max(1, candy[1]+1) = 2
  candies: [2, 1, 2]

Total: 2 + 1 + 2 = 5
```

### Video Explanation
- [NeetCode - Candy](https://www.youtube.com/watch?v=1IzCRCcK17A)

### Solution
```python
def candy(ratings: list[int]) -> int:
    """
    Distribute minimum candies satisfying rating constraints.

    Strategy (Two Pass):
    1. Left to right: ensure right neighbor constraint
    2. Right to left: ensure left neighbor constraint
    """
    n = len(ratings)
    candies = [1] * n

    # Left to right: if rating[i] > rating[i-1], give more than left
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1

    # Right to left: if rating[i] > rating[i+1], give more than right
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)

    return sum(candies)


def candy_one_pass(ratings: list[int]) -> int:
    """
    Alternative: One pass with tracking.

    Track increasing/decreasing sequences.
    """
    if len(ratings) <= 1:
        return len(ratings)

    total = 1
    up = down = peak = 0

    for i in range(1, len(ratings)):
        if ratings[i] > ratings[i - 1]:
            # Increasing
            up += 1
            down = 0
            peak = up
            total += up + 1
        elif ratings[i] < ratings[i - 1]:
            # Decreasing
            down += 1
            up = 0
            # Add 1 for each in decreasing sequence
            # If decreasing sequence exceeds peak, add 1 more
            total += down + (1 if down > peak else 0)
        else:
            # Equal
            up = down = peak = 0
            total += 1

    return total
```

### Complexity
- **Time**: O(n)
- **Space**: O(n) for two-pass, O(1) for one-pass

### Edge Cases
- All same ratings → n candies (1 each)
- Strictly increasing → 1+2+3+...+n
- Strictly decreasing → n+...+2+1
- Single child → 1 candy

---

## Problem 2: Wiggle Subsequence (LC #376) - Medium

- [LeetCode](https://leetcode.com/problems/wiggle-subsequence/)

### Problem Statement
Find length of longest wiggle subsequence (alternating up/down).

### Examples
```
Input: nums = [1,7,4,9,2,5]
Output: 6 (entire array is wiggle)

Input: nums = [1,17,5,10,13,15,10,5,16,8]
Output: 7 ([1,17,5,15,5,16,8])
```

### Intuition Development
```
PEAK AND VALLEY COUNTING:
Wiggle = alternating peaks and valleys.

nums = [1, 17, 5, 10, 13, 15, 10, 5, 16, 8]

Visual:
    17          15
   /  \        /  \      16
  /    \      /    \    /  \
 1      5   10      10 /    8
              \    /  5
               13

Peaks/Valleys: 1, 17, 5, 15, 5, 16, 8 → 7 elements

GREEDY APPROACH:
Track up (ends going up) and down (ends going down).
  up = longest ending with up move
  down = longest ending with down move

For each element:
  If going up: up = down + 1
  If going down: down = up + 1
```

### Video Explanation
- [NeetCode - Wiggle Subsequence](https://www.youtube.com/watch?v=FLbqgyJ-70I)

### Solution
```python
def wiggleMaxLength(nums: list[int]) -> int:
    """
    Find longest wiggle subsequence length.

    Strategy:
    - Track peaks and valleys
    - Count direction changes
    """
    if len(nums) < 2:
        return len(nums)

    # Count peaks and valleys
    up = down = 1

    for i in range(1, len(nums)):
        if nums[i] > nums[i - 1]:
            # Going up: extend from previous down
            up = down + 1
        elif nums[i] < nums[i - 1]:
            # Going down: extend from previous up
            down = up + 1
        # If equal, don't update (skip duplicates)

    return max(up, down)
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- All same elements → 1
- Two elements → 2 if different, 1 if same
- Already a wiggle → return length
- All increasing or decreasing → 2

---

## Problem 3: Remove Duplicate Letters (LC #316) - Medium

- [LeetCode](https://leetcode.com/problems/remove-duplicate-letters/)

### Problem Statement
Remove duplicate letters so each appears once, result is smallest lexicographically.

### Examples
```
Input: s = "bcabc"
Output: "abc"

Input: s = "cbacdcbc"
Output: "acdb"
```

### Intuition Development
```
MONOTONIC STACK + LAST OCCURRENCE:

s = "cbacdcbc"
last_occurrence: {c:7, b:6, a:2, d:4}

Process each character:
  'c': stack = ['c']
  'b': b < c, c appears later → pop c
       stack = ['b']
  'a': a < b, b appears later → pop b
       stack = ['a']
  'c': c > a, stack = ['a', 'c']
  'd': d > c, stack = ['a', 'c', 'd']
  'c': c already in stack, skip
  'b': b < d, but d doesn't appear later → keep d
       stack = ['a', 'c', 'd', 'b']
  'c': c already in stack, skip

Result: "acdb"
```

### Video Explanation
- [NeetCode - Remove Duplicate Letters](https://www.youtube.com/watch?v=zhU7JmGK_nY)

### Solution
```python
def removeDuplicateLetters(s: str) -> str:
    """
    Remove duplicates for smallest lexicographic result.

    Strategy (Monotonic Stack):
    - Track last occurrence of each character
    - Use stack to build result
    - Pop larger characters if they appear later
    """
    # Last occurrence of each character
    last_occurrence = {c: i for i, c in enumerate(s)}

    stack = []
    in_stack = set()

    for i, char in enumerate(s):
        # Skip if already in result
        if char in in_stack:
            continue

        # Pop larger characters that appear later
        while stack and char < stack[-1] and i < last_occurrence[stack[-1]]:
            removed = stack.pop()
            in_stack.remove(removed)

        stack.append(char)
        in_stack.add(char)

    return ''.join(stack)
```

### Complexity
- **Time**: O(n)
- **Space**: O(26) = O(1)

### Edge Cases
- All same character → single character
- Already sorted → return as-is
- Already no duplicates → return as-is
- Reverse sorted → may need many pops

---

## Problem 4: Reorganize String (LC #767) - Medium

- [LeetCode](https://leetcode.com/problems/reorganize-string/)

### Problem Statement
Rearrange string so no two adjacent characters are same.

### Examples
```
Input: s = "aab"
Output: "aba"

Input: s = "aaab"
Output: "" (impossible)
```

### Intuition Development
```
FEASIBILITY CHECK:
If any character appears more than (n+1)/2 times → impossible.

For "aaab" (n=4): max allowed = (4+1)/2 = 2, but 'a' appears 3 times → impossible

GREEDY WITH HEAP:
Always place most frequent character.
After placing, save it and place next most frequent.

s = "aab"
freq: {a:2, b:1}
heap: [(-2,'a'), (-1,'b')]

Step 1: pop 'a', result = "a", push back (-1,'a')
        heap: [(-1,'a'), (-1,'b')]
        prev = (-1, 'a')

Step 2: pop 'b', result = "ab", push back prev (-1,'a')
        heap: [(-1,'a')]
        prev = (0, 'b') → count=0, don't push back

Step 3: pop 'a', result = "aba"
        Done!
```

### Video Explanation
- [NeetCode - Reorganize String](https://www.youtube.com/watch?v=2g_b1aYTHeg)

### Solution
```python
import heapq
from collections import Counter

def reorganizeString(s: str) -> str:
    """
    Rearrange string with no adjacent duplicates.

    Strategy:
    - Use max heap to always place most frequent char
    - After placing, save it and place next most frequent
    - Restore saved char for next iteration
    """
    freq = Counter(s)

    # Check if possible: max frequency <= (n + 1) // 2
    max_freq = max(freq.values())
    if max_freq > (len(s) + 1) // 2:
        return ""

    # Max heap: (-count, char)
    heap = [(-count, char) for char, count in freq.items()]
    heapq.heapify(heap)

    result = []
    prev_count, prev_char = 0, ''

    while heap:
        count, char = heapq.heappop(heap)
        result.append(char)

        # Add previous char back if it still has count
        if prev_count < 0:
            heapq.heappush(heap, (prev_count, prev_char))

        # Save current for next iteration
        prev_count = count + 1  # Decrease count (it's negative)
        prev_char = char

    return ''.join(result)
```

### Complexity
- **Time**: O(n log 26) = O(n)
- **Space**: O(26) = O(1)

### Edge Cases
- Single character → return it
- Two different characters → alternate them
- Empty string → return empty
- All same character with n>1 → impossible

---

## Problem 5: IPO (LC #502) - Hard

- [LeetCode](https://leetcode.com/problems/ipo/)

### Problem Statement
Maximize capital after k projects. Each project has profit and minimum capital required.

### Examples
```
Input: k = 2, w = 0, profits = [1,2,3], capital = [0,1,1]
Output: 4

With capital 0:
  Can afford project 0 (capital=0, profit=1)
  New capital = 0 + 1 = 1

With capital 1:
  Can afford projects 1 and 2
  Pick project 2 (higher profit=3)
  New capital = 1 + 3 = 4

Final capital: 4
```

### Intuition Development
```
TWO-PHASE GREEDY:
1. Which projects can we afford? (capital ≤ w)
2. Of affordable projects, pick highest profit

Use two heaps:
  - Min-heap: projects sorted by capital requirement
  - Max-heap: affordable projects sorted by profit

Process:
  Move affordable projects from min-heap to max-heap
  Pick highest profit from max-heap
  Repeat k times
```

### Video Explanation
- [NeetCode - IPO](https://www.youtube.com/watch?v=1IUzNJ6TPEM)

### Solution
```python
def findMaximizedCapital(k: int, w: int, profits: list[int], capital: list[int]) -> int:
    """
    Maximize capital after k projects.

    Strategy:
    - Sort projects by capital requirement
    - Use max heap for available projects (by profit)
    - Greedily pick highest profit project we can afford
    """
    n = len(profits)

    # Pair and sort by capital requirement
    projects = sorted(zip(capital, profits))

    available = []  # Max heap of profits (negative for max)
    i = 0

    for _ in range(k):
        # Add all projects we can now afford
        while i < n and projects[i][0] <= w:
            heapq.heappush(available, -projects[i][1])
            i += 1

        # If no project available, stop
        if not available:
            break

        # Pick most profitable available project
        w += -heapq.heappop(available)

    return w
```

### Complexity
- **Time**: O(n log n)
- **Space**: O(n)

### Edge Cases
- No affordable projects → return initial capital
- k > n → do all projects
- All projects affordable initially → pick k highest profits
- k = 0 → return w

---

## Problem 6: Minimum Number of Refueling Stops (LC #871) - Hard

- [LeetCode](https://leetcode.com/problems/minimum-number-of-refueling-stops/)

### Problem Statement
Minimum refueling stops to reach target. Car has limited fuel capacity.

### Examples
```
Input: target = 100, startFuel = 10, stations = [[10,60],[20,30],[30,30],[60,40]]
Output: 2

Start with 10 fuel, drive to station at 10, refuel 60 → 70 fuel left
Drive to station at 60, refuel 40 → 50 fuel left
Drive to target at 100 → exactly enough!
```

### Intuition Development
```
GREEDY WITH HINDSIGHT:
Drive as far as possible.
When stuck, use the largest fuel station we passed (but didn't use).

Simulation:
  fuel = 10, pos = 0

  Drive to station at 10: fuel = 10 - 10 = 0
  Can't continue! Use passed station (60 fuel).
  fuel = 0 + 60 = 60, stops = 1

  Drive to station at 60: fuel = 60 - 50 = 10
  Drive to target at 100: fuel = 10 - 40 = -30 ← stuck!
  Use largest passed station (40 fuel from station 60).
  fuel = 10 + 40 = 50, stops = 2

  Now fuel = 50 - 40 = 10, reach target!
```

### Video Explanation
- [NeetCode - Minimum Refueling Stops](https://www.youtube.com/watch?v=TXlex1RUWE4)

### Solution
```python
def minRefuelStops(target: int, startFuel: int, stations: list[list[int]]) -> int:
    """
    Minimum refueling stops to reach target.

    Strategy:
    - Drive as far as possible
    - When stuck, use the largest fuel station we passed
    - Max heap tracks passed stations by fuel amount
    """
    # Max heap of fuel amounts at passed stations
    passed = []

    fuel = startFuel
    stops = 0
    prev_pos = 0

    # Add target as final "station"
    stations.append([target, 0])

    for pos, fuel_amount in stations:
        # Use fuel to reach this station
        fuel -= (pos - prev_pos)

        # If can't reach, refuel from passed stations
        while fuel < 0 and passed:
            fuel += -heapq.heappop(passed)
            stops += 1

        # If still can't reach, impossible
        if fuel < 0:
            return -1

        # Add this station to passed
        heapq.heappush(passed, -fuel_amount)
        prev_pos = pos

    return stops
```

### Complexity
- **Time**: O(n log n)
- **Space**: O(n)

### Edge Cases
- startFuel ≥ target → 0 stops
- No stations and not enough fuel → -1
- Stations not sorted → need to sort first
- Empty stations list

---

## Problem 7: Minimum Cost to Hire K Workers (LC #857) - Hard

- [LeetCode](https://leetcode.com/problems/minimum-cost-to-hire-k-workers/)

### Problem Statement
Hire k workers with minimum cost. Each worker has quality and minimum wage.

### Examples
```
Input: quality = [10,20,5], wage = [70,50,30], k = 2
Output: 105.0

Hire workers 1 and 2 at ratio 50/20 = 2.5:
  Worker 1: 20 × 2.5 = 50 ≥ 50 ✓
  Worker 2: 5 × 2.5 = 12.5 < 30 ✗

Hire workers 0 and 2 at ratio 70/10 = 7:
  Worker 0: 10 × 7 = 70 ≥ 70 ✓
  Worker 2: 5 × 7 = 35 ≥ 30 ✓
  Cost = 70 + 35 = 105 ✓
```

### Intuition Development
```
KEY INSIGHT:
All workers paid at same ratio (wage/quality).
Higher ratio worker sets the pay rate.

STRATEGY:
Sort workers by ratio. For each worker as the maximum ratio:
  - All previous workers will accept
  - Cost = ratio × sum of k smallest qualities

Example sorted by ratio:
  (ratio=2.5, q=20), (ratio=6, q=5), (ratio=7, q=10)

Process (ratio=2.5, q=20):
  heap: [20], sum=20
  Can't form k=2 group yet

Process (ratio=6, q=5):
  heap: [5,20], sum=25
  cost = 6 × 25 = 150

Process (ratio=7, q=10):
  heap: [5,10,20], remove max(20), sum=15
  heap: [5,10], sum=15
  cost = 7 × 15 = 105 ← minimum!
```

### Video Explanation
- [NeetCode - Minimum Cost to Hire K Workers](https://www.youtube.com/watch?v=o8emK4ehhq0)

### Solution
```python
def mincostToHireWorkers(quality: list[int], wage: list[int], k: int) -> float:
    """
    Minimum cost to hire k workers.

    Key insight: If we pay worker i their minimum wage,
    all workers must be paid at ratio wage[i]/quality[i].

    Strategy:
    - Sort workers by wage/quality ratio
    - Use max heap to track k smallest qualities
    - For each ratio, calculate cost = ratio * sum(qualities)
    """
    n = len(quality)

    # (wage/quality ratio, quality)
    workers = sorted([(wage[i] / quality[i], quality[i]) for i in range(n)])

    # Max heap of qualities (negative for max)
    quality_heap = []
    quality_sum = 0
    min_cost = float('inf')

    for ratio, q in workers:
        # Add this worker
        heapq.heappush(quality_heap, -q)
        quality_sum += q

        # If more than k workers, remove highest quality
        if len(quality_heap) > k:
            quality_sum += heapq.heappop(quality_heap)  # Add negative = subtract

        # If exactly k workers, calculate cost
        if len(quality_heap) == k:
            cost = ratio * quality_sum
            min_cost = min(min_cost, cost)

    return min_cost
```

### Complexity
- **Time**: O(n log n)
- **Space**: O(n)

### Edge Cases
- k = 1 → minimum wage worker
- k = n → must pay at highest ratio
- All same ratio → any k workers
- All same quality → minimize sum of wages

---

## Problem 8: Split Array into Consecutive Subsequences (LC #659) - Medium

- [LeetCode](https://leetcode.com/problems/split-array-into-consecutive-subsequences/)

### Problem Statement
Check if array can be split into subsequences of length >= 3 with consecutive integers.

### Examples
```
Input: nums = [1,2,3,3,4,5]
Output: true
Explanation: [1,2,3] and [3,4,5]

Input: nums = [1,2,3,3,4,4,5,5]
Output: true
Explanation: [1,2,3,4,5] and [3,4,5]

Input: nums = [1,2,3,4,4,5]
Output: false
```

### Intuition Development
```
GREEDY CHOICE:
For each number, prefer extending existing sequence over starting new one.

nums = [1,2,3,3,4,5]

Process 1:
  No sequence ends at 0 → start new sequence [1]
  Need 2 and 3 to make valid

Process 2:
  Sequence ends at 1 → extend to [1,2]
  Still need 3

Process 3:
  Sequence ends at 2 → extend to [1,2,3] ✓ valid!

Process 3 (second):
  No sequence ends at 2 → start new [3]
  Need 4 and 5

Process 4:
  Sequence ends at 3 (the new one) → extend to [3,4]

Process 5:
  Sequence ends at 4 → extend to [3,4,5] ✓ valid!

Both sequences valid → true
```

### Video Explanation
- [NeetCode - Split Array into Consecutive Subsequences](https://www.youtube.com/watch?v=uJ8BAQ8lASE)

### Solution
```python
from collections import defaultdict

def isPossible(nums: list[int]) -> bool:
    """
    Check if array can be split into consecutive subsequences.

    Strategy:
    - Count frequency of each number
    - Track tails: how many subsequences end at each number
    - For each number, either extend existing sequence or start new one
    """
    freq = defaultdict(int)
    tails = defaultdict(int)  # tails[x] = subsequences ending at x

    for num in nums:
        freq[num] += 1

    for num in nums:
        if freq[num] == 0:
            continue

        if tails[num - 1] > 0:
            # Extend existing sequence
            tails[num - 1] -= 1
            tails[num] += 1
        elif freq[num + 1] > 0 and freq[num + 2] > 0:
            # Start new sequence of length 3
            freq[num + 1] -= 1
            freq[num + 2] -= 1
            tails[num + 2] += 1
        else:
            # Can't use this number
            return False

        freq[num] -= 1

    return True
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- Less than 3 elements → false
- All same number → false
- Single long consecutive sequence → true
- Gaps in sequence → false

---

## Summary: Advanced Greedy Problems

| # | Problem | Key Insight | Time |
|---|---------|-------------|------|
| 1 | Candy | Two-pass left/right | O(n) |
| 2 | Wiggle Subsequence | Track up/down counts | O(n) |
| 3 | Remove Duplicate Letters | Monotonic stack + last occurrence | O(n) |
| 4 | Reorganize String | Max heap for frequencies | O(n) |
| 5 | IPO | Sort + max heap | O(n log n) |
| 6 | Refueling Stops | Max heap of passed stations | O(n log n) |
| 7 | Hire K Workers | Sort by ratio + min quality sum | O(n log n) |
| 8 | Consecutive Subsequences | Frequency + tails tracking | O(n) |

---

## Greedy vs DP Decision Guide

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GREEDY VS DYNAMIC PROGRAMMING                            │
│                                                                             │
│  Use GREEDY when:                                                           │
│  • Local optimal choice leads to global optimal                             │
│  • No need to reconsider previous choices                                   │
│  • Problem has greedy choice property                                       │
│                                                                             │
│  Use DP when:                                                               │
│  • Need to consider all possibilities                                       │
│  • Optimal solution depends on subproblem solutions                         │
│  • Greedy gives wrong answer                                                │
│                                                                             │
│  Examples:                                                                  │
│  • Activity Selection: Greedy (sort by end time)                            │
│  • 0/1 Knapsack: DP (can't split items)                                     │
│  • Fractional Knapsack: Greedy (can split items)                            │
│  • Coin Change (arbitrary coins): DP                                        │
│  • Coin Change (canonical coins): Greedy                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practice More Problems

- [ ] LC #45 - Jump Game II
- [ ] LC #134 - Gas Station
- [ ] LC #321 - Create Maximum Number
- [ ] LC #330 - Patching Array
- [ ] LC #1353 - Maximum Number of Events That Can Be Attended
