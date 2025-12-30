# Greedy - Hard Problems

## Problem 1: Candy (LC #135) - Hard

- [LeetCode](https://leetcode.com/problems/candy/)

### Video Explanation
- [NeetCode - Candy](https://www.youtube.com/watch?v=1IzCRCcK17A)

### Problem Statement
Distribute minimum candies to children based on ratings.


### Visual Intuition
```
Candy Distribution
ratings = [1, 0, 2]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Two passes - left neighbors, then right neighbors
             Take max to satisfy both directions
═══════════════════════════════════════════════════════════════

Initial State:
──────────────
  Index:   0   1   2
  Rating: [1] [0] [2]
  Candy:  [1] [1] [1]  (everyone starts with 1)

Pass 1: Left to Right (compare with LEFT neighbor)
──────────────────────────────────────────────────
  i=1: rating[1]=0 < rating[0]=1
       Child 1 has LOWER rating → no extra candy needed
       candy[1] stays 1

  i=2: rating[2]=2 > rating[1]=0
       Child 2 has HIGHER rating → needs more candy!
       candy[2] = candy[1] + 1 = 2

  After Pass 1:
    Rating: [1] [0] [2]
    Candy:  [1] [1] [2]
            ↑       ↑
            OK      2 > 0 satisfied ✓

Pass 2: Right to Left (compare with RIGHT neighbor)
───────────────────────────────────────────────────
  i=1: rating[1]=0 < rating[2]=2
       Child 1 has LOWER rating → no change needed

  i=0: rating[0]=1 > rating[1]=0
       Child 0 has HIGHER rating → needs more than child 1!
       candy[0] = max(candy[0], candy[1]+1) = max(1, 2) = 2

  After Pass 2:
    Rating: [1] [0] [2]
    Candy:  [2] [1] [2]
            ↑       ↑
         1>0 ✓   2>0 ✓

Visual Distribution:
────────────────────
  Rating:    1       0       2
             ★       ★       ★★
  Candy:    🍬🍬     🍬     🍬🍬
             2       1       2

  Total = 2 + 1 + 2 = 5 candies (minimum!)

WHY THIS WORKS:
════════════════
● Pass 1: Ensures higher-rated child has more than LEFT neighbor
● Pass 2: Ensures higher-rated child has more than RIGHT neighbor
● max() prevents undoing Pass 1's work
● Two passes = O(n), both directions satisfied
```

### Solution
```python
def candy(ratings: list[int]) -> int:
    """
    Two-pass greedy: left-to-right, then right-to-left.

    Time: O(n)
    Space: O(n)
    """
    n = len(ratings)
    candies = [1] * n

    # Left to right: higher rating than left gets more
    for i in range(1, n):
        if ratings[i] > ratings[i - 1]:
            candies[i] = candies[i - 1] + 1

    # Right to left: higher rating than right gets more
    for i in range(n - 2, -1, -1):
        if ratings[i] > ratings[i + 1]:
            candies[i] = max(candies[i], candies[i + 1] + 1)

    return sum(candies)
```

### Edge Cases
- All same ratings → n candies (1 each)
- Strictly increasing → 1+2+...+n candies
- Strictly decreasing → n+...+2+1 candies
- Single child → 1 candy

---

## Problem 2: Create Maximum Number (LC #321) - Hard

- [LeetCode](https://leetcode.com/problems/create-maximum-number/)

### Video Explanation
- [NeetCode - Create Maximum Number](https://www.youtube.com/watch?v=YYpvLLgG8EM)

### Problem Statement
Create maximum number of length k from two arrays.


### Visual Intuition
```
Create Maximum Number
nums1 = [3,4,6,5], nums2 = [9,1,2,5,8,3], k = 5

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Try all splits (i from nums1, k-i from nums2)
             For each split: get max subsequence, then merge
═══════════════════════════════════════════════════════════════

Step 1: Get Max Subsequence (using monotonic stack)
───────────────────────────────────────────────────
  max_subseq([3,4,6,5], length=2):
    can_drop = 4-2 = 2

    Process 3: stack=[3]
    Process 4: 4>3, pop 3, drop=1, stack=[4]
    Process 6: 6>4, pop 4, drop=0, stack=[6]
    Process 5: can't drop, stack=[6,5]

    Result: [6,5]

  max_subseq([9,1,2,5,8,3], length=3):
    can_drop = 6-3 = 3

    Process 9: stack=[9]
    Process 1: 1<9, stack=[9,1]
    Process 2: 2>1, pop 1, drop=2, stack=[9,2]
    Process 5: 5>2, pop 2, drop=1, stack=[9,5]
    Process 8: 8>5, pop 5, drop=0, stack=[9,8]
    Process 3: can't drop, stack=[9,8,3]

    Result: [9,8,3]

Step 2: Merge Two Arrays (lexicographically largest)
────────────────────────────────────────────────────
  arr1 = [6,5], arr2 = [9,8,3]

  Compare: [6,5] vs [9,8,3]
           [9,8,3] > [6,5] → take 9
           result = [9]

  Compare: [6,5] vs [8,3]
           [8,3] > [6,5] → take 8
           result = [9,8]

  Compare: [6,5] vs [3]
           [6,5] > [3] → take 6
           result = [9,8,6]

  Compare: [5] vs [3]
           [5] > [3] → take 5
           result = [9,8,6,5]

  Take remaining: 3
  result = [9,8,6,5,3]

Step 3: Try All Splits
──────────────────────
  i=0: [] from nums1, [9,8,5,8,3][:5] from nums2 → [9,5,8,3,?]
  i=1: [6] + [9,8,5,3] → merge → [9,8,6,5,3]
  i=2: [6,5] + [9,8,3] → merge → [9,8,6,5,3] ★
  i=3: [6,5,?] + [9,8] → ...
  i=4: [6,5,4,3] + [9] → ...

  Compare all, return maximum: [9,8,6,5,3]

WHY THIS WORKS:
════════════════
● Monotonic stack gives lexicographically largest subsequence
● Merge comparison uses full suffix (not just single element)
● Try all splits ensures we find global optimum
```

### Solution
```python
def maxNumber(nums1: list[int], nums2: list[int], k: int) -> list[int]:
    """
    Try all splits, get max subsequence from each, merge.

    Time: O(k² * (m + n))
    Space: O(k)
    """
    def max_subsequence(nums, length):
        """Get max subsequence of given length using monotonic stack."""
        drop = len(nums) - length
        stack = []
        for num in nums:
            while drop and stack and stack[-1] < num:
                stack.pop()
                drop -= 1
            stack.append(num)
        return stack[:length]

    def merge(arr1, arr2):
        """Merge two arrays to form largest number."""
        result = []
        i = j = 0
        while i < len(arr1) or j < len(arr2):
            if arr1[i:] > arr2[j:]:
                result.append(arr1[i])
                i += 1
            else:
                result.append(arr2[j])
                j += 1
        return result

    result = []

    for i in range(k + 1):
        if i <= len(nums1) and k - i <= len(nums2):
            sub1 = max_subsequence(nums1, i)
            sub2 = max_subsequence(nums2, k - i)
            merged = merge(sub1, sub2)
            result = max(result, merged)

    return result
```

### Edge Cases
- k = 0 → return []
- One array empty → use other array only
- k > len(nums1) + len(nums2) → use all
- All same digits → any order

---

## Problem 3: Minimum Number of Refueling Stops (LC #871) - Hard

- [LeetCode](https://leetcode.com/problems/minimum-number-of-refueling-stops/)

### Video Explanation
- [NeetCode - Minimum Number of Refueling Stops](https://www.youtube.com/watch?v=HBdZ2aGCJG0)

### Problem Statement
Minimum refueling stops to reach target.


### Visual Intuition
```
Minimum Number of Refueling Stops
target = 100, startFuel = 10
stations = [[10,60],[20,30],[30,30],[60,40]]
           [position, fuel_available]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Use max-heap of passed stations' fuel
             Only refuel when you MUST (greedy: use largest tank)
═══════════════════════════════════════════════════════════════

Road Visualization:
───────────────────
  0        10       20       30       60       100
  ├────────┼────────┼────────┼────────┼────────┤
  START   ⛽60     ⛽30     ⛽30     ⛽40     TARGET

Step-by-Step Simulation:
────────────────────────
Initial: position=0, fuel=10, heap=[], stops=0

Step 1: Can we reach target? 0+10=10 < 100 ✗
        Add reachable stations to heap:
        station[10] reachable (10 ≤ 10) → heap=[60]

        Out of fuel! Must refuel.
        Take max from heap: 60
        fuel = 0 + 60 = 60, stops = 1

        Now at position 10 with 60 fuel
        Can reach: 10 + 60 = 70

Step 2: Can we reach target? 70 < 100 ✗
        Add reachable stations:
        station[20] reachable (20 ≤ 70) → heap=[60,30]
        station[30] reachable (30 ≤ 70) → heap=[60,30,30]
        station[60] reachable (60 ≤ 70) → heap=[60,40,30,30]

        Out of fuel at position 70!
        Take max from heap: 40 (from station 60)
        fuel = 0 + 40 = 40, stops = 2

        Can reach: 70 + 40 = 110 ≥ 100 ✓

Step 3: Can we reach target? 110 ≥ 100 ✓ DONE!

Timeline:
─────────
  0 ──10──→ 10 ──60──→ 70 ──30──→ 100
       ↑         ↑
    refuel 1  refuel 2
    (60 fuel) (40 fuel)

Answer: 2 stops

WHY THIS WORKS:
════════════════
● We "collect" fuel from stations we pass (add to heap)
● Only actually use fuel when we run out
● Always use largest available tank (greedy optimal)
● This minimizes total stops needed
```

### Solution
```python
import heapq

def minRefuelStops(target: int, startFuel: int, stations: list[list[int]]) -> int:
    """
    Greedy with max heap: always use largest fuel when needed.

    Time: O(n log n)
    Space: O(n)
    """
    fuel = startFuel
    stops = 0
    heap = []  # Max heap of fuel amounts (negated)
    i = 0

    while fuel < target:
        # Add all reachable stations to heap
        while i < len(stations) and stations[i][0] <= fuel:
            heapq.heappush(heap, -stations[i][1])
            i += 1

        # If no fuel available, can't reach target
        if not heap:
            return -1

        # Use largest fuel tank
        fuel += -heapq.heappop(heap)
        stops += 1

    return stops
```

### Edge Cases
- Start fuel >= target → return 0
- No stations → check if start fuel enough
- Can't reach first station → return -1
- Stations at same position → take both fuel

---

## Problem 4: Task Scheduler (LC #621) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/task-scheduler/)

### Video Explanation
- [NeetCode - Task Scheduler](https://www.youtube.com/watch?v=s8p8ukTyA2I)

### Problem Statement
Find minimum time to execute all tasks with cooldown period between same tasks.

### Visual Intuition
```
Task Scheduler with Cooldown
tasks = [A,A,A,B,B,B], n = 2

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Most frequent task creates "frames" of size (n+1)
             Fill frames with other tasks to minimize idle time
═══════════════════════════════════════════════════════════════

Task Analysis:
──────────────
  tasks = [A,A,A,B,B,B]
  freq(A) = 3, freq(B) = 3
  max_freq = 3
  tasks_with_max_freq = 2 (both A and B)

Frame Structure:
────────────────
  n = 2 means same task needs 2 slots gap
  Frame size = n + 1 = 3

  We need (max_freq - 1) = 2 full frames + 1 partial frame

  Frame 1: │ A │ B │ _ │  (slots 1-3)
  Frame 2: │ A │ B │ _ │  (slots 4-6)
  Frame 3: │ A │ B │      (slots 7-8, partial)
           └───┴───┴───┘

  Timeline: A B _ A B _ A B
            1 2 3 4 5 6 7 8
                ↑     ↑
              idle  idle

Formula Derivation:
───────────────────
  Full frames: (max_freq - 1) = 2
  Slots per frame: (n + 1) = 3
  Extra slots: tasks_with_max_freq = 2

  Total = (max_freq - 1) × (n + 1) + tasks_with_max_freq
        = (3 - 1) × 3 + 2
        = 6 + 2 = 8

When Formula Doesn't Apply:
───────────────────────────
  If we have MORE tasks than formula result:
  tasks = [A,A,A,B,B,B,C,C,C,D,D,D,E,E,E]

  No idle time needed! Answer = len(tasks) = 15

  Final formula: max(formula_result, len(tasks))

Answer: 8 time units

WHY THIS WORKS:
════════════════
● Most frequent task MUST have (n+1) gap between executions
● This creates mandatory "frames" structure
● Fill frames with other tasks to reduce idle time
● If too many tasks, no idle needed
```


### Intuition
```
tasks = ["A","A","A","B","B","B"], n = 2

Most frequent task determines structure:
A _ _ A _ _ A

Fill gaps with other tasks:
A B _ A B _ A B

If gaps remain, they become idle time.
```

### Solution
```python
from collections import Counter

def leastInterval(tasks: list[str], n: int) -> int:
    """
    Math-based greedy approach.

    Strategy:
    - Most frequent task creates (max_count - 1) gaps of size (n + 1)
    - Fill gaps with other tasks
    - If more tasks than gaps, no idle time needed

    Time: O(n)
    Space: O(1) - at most 26 letters
    """
    freq = Counter(tasks)
    max_freq = max(freq.values())

    # Count tasks with max frequency
    max_count = sum(1 for f in freq.values() if f == max_freq)

    # Minimum slots needed
    # (max_freq - 1) full cycles + final partial cycle
    min_slots = (max_freq - 1) * (n + 1) + max_count

    # Answer is max of calculated slots or total tasks
    # (if we have more tasks than gaps, no idle needed)
    return max(min_slots, len(tasks))
```

### Heap-based Solution
```python
import heapq
from collections import Counter, deque

def leastInterval(tasks: list[str], n: int) -> int:
    """
    Simulation with max heap and cooldown queue.

    Time: O(n * m) where m = unique tasks
    Space: O(m)
    """
    freq = Counter(tasks)

    # Max heap of frequencies (negated)
    heap = [-f for f in freq.values()]
    heapq.heapify(heap)

    # Queue of (available_time, count)
    cooldown = deque()
    time = 0

    while heap or cooldown:
        time += 1

        if heap:
            count = heapq.heappop(heap) + 1  # Execute one task
            if count < 0:
                cooldown.append((time + n, count))

        # Check if any task is ready
        if cooldown and cooldown[0][0] == time:
            heapq.heappush(heap, cooldown.popleft()[1])

    return time
```

### Complexity
- **Time**: O(n) for math, O(n * m) for simulation
- **Space**: O(1) or O(m)

### Edge Cases
- n = 0 → return 0
- All same tasks → (count-1)*(n+1) + 1
- n = 0 → return len(tasks)
- Single task type → no idle needed

---

## Problem 5: Jump Game II (LC #45) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/jump-game-ii/)

### Video Explanation
- [NeetCode - Jump Game II](https://www.youtube.com/watch?v=dJ7sWiOoK7g)

### Problem Statement
Minimum jumps to reach the last index.

### Visual Intuition
```
Jump Game II - Minimum Jumps
nums = [2,3,1,1,4]

BFS-like levels (greedy):
Level 0: index 0, can reach 0+2=2
Level 1: indices 1,2, can reach max(1+3,2+1)=4 ✓

  [2, 3, 1, 1, 4]
   0  1  2  3  4
   └──┴──┘     Level 1 boundary
      └──────┘ Level 2 boundary (reaches end!)

Track: current_end, farthest, jumps
i=0: farthest=2, i==current_end → jump! current_end=2
i=1: farthest=4
i=2: i==current_end → jump! current_end=4 (reached!)

Answer: 2 jumps
```


### Intuition
```
nums = [2,3,1,1,4]

Position 0: can reach 1,2 → jump to 1 (reaches farthest: 4)
Position 1: can reach 2,3,4 → reach end!

Greedy: always jump to position that lets us reach farthest.
```

### Solution
```python
def jump(nums: list[int]) -> int:
    """
    Greedy BFS-like approach.

    Strategy:
    - Track current range [start, end]
    - Find farthest reachable from current range
    - Jump count = number of range expansions

    Time: O(n)
    Space: O(1)
    """
    n = len(nums)
    if n <= 1:
        return 0

    jumps = 0
    current_end = 0    # End of current jump range
    farthest = 0       # Farthest we can reach

    for i in range(n - 1):
        farthest = max(farthest, i + nums[i])

        # Reached end of current range - must jump
        if i == current_end:
            jumps += 1
            current_end = farthest

            # Early termination
            if current_end >= n - 1:
                break

    return jumps
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- Single element → 0 jumps
- All zeros except first → can't reach (but guaranteed reachable)
- First element >= n-1 → 1 jump
- All ones → n-1 jumps

---

## Problem 6: Course Schedule III (LC #630) - Hard

- [LeetCode](https://leetcode.com/problems/course-schedule-iii/)

### Video Explanation
- [NeetCode - Course Schedule III](https://www.youtube.com/watch?v=ey8FxYsFAMU)

### Problem Statement
Maximum number of courses that can be taken given durations and deadlines.

### Visual Intuition
```
Course Schedule III - Maximum Courses
courses = [[100,200],[200,1300],[1000,1250],[2000,3200]]
         [duration, deadline]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Sort by deadline, greedily take courses
             If deadline exceeded, replace longest course taken
═══════════════════════════════════════════════════════════════

Sorted by deadline:
───────────────────
  Course 1: duration=100,  deadline=200
  Course 2: duration=200,  deadline=1300
  Course 3: duration=1000, deadline=1250
  Course 4: duration=2000, deadline=3200

Step-by-Step:
─────────────
Step 1: Course [100, 200]
        time = 0 + 100 = 100
        100 ≤ 200 (deadline)? ✓ TAKE IT
        heap = [100]
        courses_taken = 1

Step 2: Course [200, 1300]
        time = 100 + 200 = 300
        300 ≤ 1300 (deadline)? ✓ TAKE IT
        heap = [200, 100]  (max-heap: 200 at top)
        courses_taken = 2

Step 3: Course [1000, 1250]
        time = 300 + 1000 = 1300
        1300 ≤ 1250 (deadline)? ✗ EXCEEDS!

        Replace longest course (200):
        time = 1300 - 200 = 1100
        1100 ≤ 1250? ✓

        heap = [1000, 100]  (replaced 200 with 1000)
        courses_taken = 2 (same count, better for future)

Step 4: Course [2000, 3200]
        time = 1100 + 2000 = 3100
        3100 ≤ 3200 (deadline)? ✓ TAKE IT
        heap = [2000, 1000, 100]
        courses_taken = 3

Final: 3 courses can be taken

Timeline Visualization:
───────────────────────
  0        1100      3100  3200
  ├─────────┼──────────┼────┤
  │  100    │   2000   │
  │  1000   │          │
  └─────────┴──────────┘

  Courses: [100,200], [1000,1250], [2000,3200]

WHY THIS WORKS:
════════════════
● Sort by deadline: consider "urgent" courses first
● Greedy take: maximize courses taken
● Replace longest: if can't fit, swapping for shorter helps future
● Max-heap: O(log n) to find/remove longest course
```


### Intuition
```
courses = [[100,200],[200,1300],[1000,1250],[2000,3200]]
         (duration, deadline)

Sort by deadline.
Greedily take courses, replace longest if deadline exceeded.
```

### Solution
```python
import heapq

def scheduleCourse(courses: list[list[int]]) -> int:
    """
    Greedy with max heap for replacement.

    Strategy:
    - Sort courses by deadline
    - Try to take each course
    - If deadline exceeded, replace longest course taken

    Time: O(n log n)
    Space: O(n)
    """
    # Sort by deadline
    courses.sort(key=lambda x: x[1])

    heap = []  # Max heap of durations (negated)
    total_time = 0

    for duration, deadline in courses:
        # Try to take this course
        total_time += duration
        heapq.heappush(heap, -duration)

        # If deadline exceeded, remove longest course
        if total_time > deadline:
            total_time += heapq.heappop(heap)  # Add negative = subtract

    return len(heap)
```

### Why This Works
```
Proof intuition:
- Sorting by deadline ensures we consider "urgent" courses first
- If we can't fit a course, replacing the longest one we've taken
  gives the best chance of fitting future courses
- We never make the situation worse by replacing
```

### Complexity
- **Time**: O(n log n)
- **Space**: O(n)

### Edge Cases
- No courses fit → return 0
- All courses fit → return n
- Duration > deadline → skip that course
- Same deadline → take shorter first

---

## Summary

| # | Problem | Key Technique |
|---|---------|---------------|
| 1 | Candy | Two-pass greedy |
| 2 | Create Max Number | Subsequence + merge |
| 3 | Min Refueling Stops | Max heap greedy |
| 4 | Task Scheduler | Math or heap simulation |
| 5 | Jump Game II | Range expansion greedy |
| 6 | Course Schedule III | Sort + heap replacement |
