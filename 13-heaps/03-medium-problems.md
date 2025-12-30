# Heaps - Advanced Problems

## Advanced Heap Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED HEAP TECHNIQUES                                 │
│                                                                             │
│  1. TWO HEAPS:                                                              │
│     - Max-heap for smaller half, min-heap for larger half                   │
│     - Used for: median, sliding window problems                             │
│                                                                             │
│  2. LAZY DELETION:                                                          │
│     - Mark elements as deleted instead of removing                          │
│     - Clean up when accessing top                                           │
│                                                                             │
│  3. HEAP + HASH MAP:                                                        │
│     - Track positions for updates                                           │
│     - Used for: priority queue with decrease-key                            │
│                                                                             │
│  4. K-WAY MERGE:                                                            │
│     - Merge k sorted sequences                                              │
│     - Heap of (value, source_index, element_index)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem 1: Sliding Window Median (LC #480) - Hard

- [LeetCode](https://leetcode.com/problems/sliding-window-median/)

### Problem Statement
Find median of each sliding window of size k.

### Examples
```
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [1.0,-1.0,-1.0,3.0,5.0,6.0]

Window [1,3,-1]: sorted = [-1,1,3], median = 1.0
Window [3,-1,-3]: sorted = [-3,-1,3], median = -1.0
...
```

### Intuition Development
```
TWO HEAPS APPROACH:
  small (max-heap): stores smaller half
  large (min-heap): stores larger half

Median = small[0] if k is odd, else (small[0] + large[0]) / 2

LAZY DELETION for sliding window:
  Can't easily remove from middle of heap.
  Mark for deletion, clean up when accessing top.

Example: [1,3,-1,-3,5,3,6,7], k=3

Window [1,3,-1]:
  small: [-1, 1] (as max-heap: -1)
  large: [3]
  median = 1.0

Slide: remove 1, add -3
  to_remove[1] = 1
  Add -3 to small
  Balance...

When accessing heap top, skip marked elements.
```

### Video Explanation
- [NeetCode - Sliding Window Median](https://www.youtube.com/watch?v=pV3Xn5mRHkA)

### Solution
```python
import heapq
from collections import defaultdict

def medianSlidingWindow(nums: list[int], k: int) -> list[float]:
    """
    Sliding window median using two heaps with lazy deletion.

    Strategy:
    - Max-heap for smaller half, min-heap for larger half
    - Use lazy deletion (track elements to remove)
    - Rebalance heaps after each operation
    """
    # Max-heap (negate values), min-heap
    small = []  # Max-heap of smaller half
    large = []  # Min-heap of larger half

    # Track elements to be removed
    to_remove = defaultdict(int)

    def add(num):
        """Add number to appropriate heap."""
        if not small or num <= -small[0]:
            heapq.heappush(small, -num)
        else:
            heapq.heappush(large, num)

    def remove(num):
        """Mark number for lazy deletion."""
        to_remove[num] += 1

    def balance():
        """Balance heaps so small has equal or one more element."""
        while len(small) > len(large) + 1:
            heapq.heappush(large, -heapq.heappop(small))
        while len(large) > len(small):
            heapq.heappush(small, -heapq.heappop(large))

    def prune(heap, is_max=False):
        """Remove marked elements from heap top."""
        while heap:
            top = -heap[0] if is_max else heap[0]
            if to_remove[top] > 0:
                to_remove[top] -= 1
                heapq.heappop(heap)
            else:
                break

    def get_median():
        """Get current median."""
        if k % 2 == 1:
            return float(-small[0])
        return (-small[0] + large[0]) / 2

    result = []

    # Initialize first window
    for i in range(k):
        add(nums[i])
        balance()

    result.append(get_median())

    # Slide window
    for i in range(k, len(nums)):
        # Add new element
        add(nums[i])

        # Remove old element (lazy)
        remove(nums[i - k])

        # Balance and prune
        balance()
        prune(small, is_max=True)
        prune(large, is_max=False)

        result.append(get_median())

    return result
```

### Complexity
- **Time**: O(n log k)
- **Space**: O(k)

### Edge Cases
- k = 1 → each element is its own median
- k = n → single median for entire array
- All same elements
- Negative numbers

---

## Problem 2: Smallest Range Covering K Lists (LC #632) - Hard

- [LeetCode](https://leetcode.com/problems/smallest-range-covering-elements-from-k-lists/)

### Problem Statement
Find smallest range that includes at least one number from each list.

### Examples
```
Input: nums = [[4,10,15,24,26],[0,9,12,20],[5,18,22,30]]
Output: [20,24]

Range [20,24] contains:
  - 24 from list 0
  - 20 from list 1
  - 22 from list 2
```

### Intuition Development
```
K-WAY MERGE PATTERN:
Track one element from each list at a time.
Range = [min element, max element]

Strategy:
  - Use min-heap to track current elements
  - Track max separately
  - Always advance minimum to try shrinking range

Initial: [4, 0, 5]
  min = 0, max = 5, range = [0, 5], size = 5

Advance 0 → 9: [4, 9, 5]
  min = 4, max = 9, range = [4, 9], size = 5

Continue until one list exhausted...
```

### Video Explanation
- [NeetCode - Smallest Range Covering K Lists](https://www.youtube.com/watch?v=Fqal25ZgEDo)

### Solution
```python
def smallestRange(nums: list[list[int]]) -> list[int]:
    """
    Find smallest range covering all k lists.

    Strategy:
    - Use min-heap to track current elements from each list
    - Track max element to know current range
    - Move minimum element forward to shrink range
    """
    k = len(nums)

    # Min-heap: (value, list_index, element_index)
    heap = []
    current_max = float('-inf')

    # Initialize with first element of each list
    for i in range(k):
        heapq.heappush(heap, (nums[i][0], i, 0))
        current_max = max(current_max, nums[i][0])

    result = [float('-inf'), float('inf')]

    while True:
        current_min, list_idx, elem_idx = heapq.heappop(heap)

        # Update result if better range found
        if current_max - current_min < result[1] - result[0]:
            result = [current_min, current_max]

        # Move to next element in this list
        if elem_idx + 1 < len(nums[list_idx]):
            next_val = nums[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
            current_max = max(current_max, next_val)
        else:
            # One list exhausted, can't cover all lists anymore
            break

    return result
```

### Complexity
- **Time**: O(n × log k) where n = total elements
- **Space**: O(k)

### Edge Cases
- Single list → [min, max] of that list
- Lists with single element each
- Overlapping ranges
- Very large values

---

## Problem 3: Trapping Rain Water II (LC #407) - Hard

- [LeetCode](https://leetcode.com/problems/trapping-rain-water-ii/)

### Problem Statement
Calculate water trapped in 2D elevation map.

### Examples
```
Input: heightMap = [
  [1,4,3,1,3,2],
  [3,2,1,3,2,4],
  [2,3,3,2,3,1]
]
Output: 4

Water fills in low areas bounded by higher boundaries.
```

### Intuition Development
```
3D EXTENSION OF 1D PROBLEM:
Water level at cell = min of all paths to boundary.

BFS FROM BOUNDARY:
  - Boundary cells can't hold water
  - Process cells by height (min-heap)
  - Water at cell = max(0, boundary_height - cell_height)

Visualization (cross-section):
    Boundary → inner cells ← Boundary
       4         2           4
       |   Water fills to 4  |
       |    (4-2 = 2 water)  |

Process from lowest boundary, flood inward.
```

### Video Explanation
- [NeetCode - Trapping Rain Water II](https://www.youtube.com/watch?v=bS3bGwXoqFM)

### Solution
```python
def trapRainWater(heightMap: list[list[int]]) -> int:
    """
    Trap rain water in 2D using min-heap BFS.

    Strategy:
    - Start from boundary (water can't be held at boundary)
    - Use min-heap to process cells by height
    - Water at cell = max(boundary_height - cell_height, 0)
    """
    if not heightMap or not heightMap[0]:
        return 0

    rows, cols = len(heightMap), len(heightMap[0])
    visited = [[False] * cols for _ in range(rows)]

    # Min-heap: (height, row, col)
    heap = []

    # Add boundary cells
    for r in range(rows):
        for c in [0, cols - 1]:
            heapq.heappush(heap, (heightMap[r][c], r, c))
            visited[r][c] = True

    for c in range(cols):
        for r in [0, rows - 1]:
            if not visited[r][c]:
                heapq.heappush(heap, (heightMap[r][c], r, c))
                visited[r][c] = True

    water = 0
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while heap:
        height, r, c = heapq.heappop(heap)

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                visited[nr][nc] = True

                # Water trapped = boundary height - cell height
                water += max(0, height - heightMap[nr][nc])

                # New boundary height is max of current and cell
                new_height = max(height, heightMap[nr][nc])
                heapq.heappush(heap, (new_height, nr, nc))

    return water
```

### Complexity
- **Time**: O(mn × log(mn))
- **Space**: O(mn)

### Edge Cases
- 1×1 or 2×2 grid → no water
- All same height → no water
- Bowl shape → fills completely
- Multiple pools at different levels

---

## Problem 4: Maximum Performance of a Team (LC #1383) - Hard

- [LeetCode](https://leetcode.com/problems/maximum-performance-of-a-team/)

### Problem Statement
Select at most k engineers to maximize sum(speed) × min(efficiency).

### Examples
```
Input: n = 6, speed = [2,10,3,1,5,8], efficiency = [5,4,3,9,7,2], k = 2
Output: 60

Choose engineers 0 and 2:
  speed_sum = 2 + 3 = 5 (wait, that's not right)

Actually choose engineers 1 and 4:
  speed_sum = 10 + 5 = 15
  min_efficiency = min(4, 7) = 4
  performance = 15 × 4 = 60
```

### Intuition Development
```
KEY INSIGHT:
If we fix the minimum efficiency, we want maximum speed sum.

SORT BY EFFICIENCY (descending):
As we iterate, current engineer has minimum efficiency.
Keep track of k fastest engineers seen so far.

Engineers sorted by efficiency:
  (eff=9, spd=1), (eff=7, spd=5), (eff=5, spd=2),
  (eff=4, spd=10), (eff=3, spd=3), (eff=2, spd=8)

Process each as potential minimum efficiency:
  eff=9: speeds=[1], sum=1, perf=9×1=9
  eff=7: speeds=[1,5], sum=6, perf=7×6=42
  eff=5: speeds=[1,5,2], keep top 2: [5,2], sum=7, perf=5×7=35
  eff=4: speeds=[5,10], sum=15, perf=4×15=60 ← MAX
  ...
```

### Video Explanation
- [NeetCode - Maximum Performance of a Team](https://www.youtube.com/watch?v=Y7UTvogADH0)

### Solution
```python
def maxPerformance(n: int, speed: list[int], efficiency: list[int], k: int) -> int:
    """
    Maximize team performance.

    Strategy:
    - Sort engineers by efficiency (descending)
    - For each engineer as minimum efficiency
    - Use min-heap to track k fastest engineers
    """
    MOD = 10**9 + 7

    # Pair and sort by efficiency descending
    engineers = sorted(zip(efficiency, speed), reverse=True)

    speed_heap = []  # Min-heap of speeds
    speed_sum = 0
    max_perf = 0

    for eff, spd in engineers:
        # Add current engineer
        heapq.heappush(speed_heap, spd)
        speed_sum += spd

        # If more than k engineers, remove slowest
        if len(speed_heap) > k:
            speed_sum -= heapq.heappop(speed_heap)

        # Calculate performance with current engineer's efficiency as minimum
        max_perf = max(max_perf, speed_sum * eff)

    return max_perf % MOD
```

### Complexity
- **Time**: O(n log n)
- **Space**: O(k)

### Edge Cases
- k = 1 → maximize single engineer's speed × efficiency
- k = n → use all engineers
- All same efficiency → maximize speed sum
- All same speed → maximize minimum efficiency

---

## Problem 5: Course Schedule III (LC #630) - Hard

- [LeetCode](https://leetcode.com/problems/course-schedule-iii/)

### Problem Statement
Maximum courses that can be taken (each has duration and deadline).

### Examples
```
Input: courses = [[100,200],[200,1300],[1000,1250],[2000,3200]]
Output: 3

Take courses:
  Course 0: duration=100, ends at 100 (deadline 200) ✓
  Course 2: duration=1000, ends at 1100 (deadline 1250) ✓
  Course 1: duration=200, ends at 1300 (deadline 1300) ✓
  Course 3: would end at 3300 > 3200 ✗
```

### Intuition Development
```
GREEDY + HEAP:
Sort by deadline → process courses in order of urgency.

For each course:
  If fits: take it
  If doesn't fit: swap with longest taken course if beneficial

Example:
  Sorted: [[100,200],[1000,1250],[200,1300],[2000,3200]]

  Take [100,200]: time=100, taken=[100]
  Take [1000,1250]: time=1100, taken=[100,1000]
  Try [200,1300]: time would be 1300 ≤ 1300 ✓
    taken=[100,1000,200]
  Try [2000,3200]: time would be 3300 > 3200 ✗
    Longest taken = 1000, duration 2000 > 1000
    Swap would make it worse, skip

  Result: 3 courses
```

### Video Explanation
- [NeetCode - Course Schedule III](https://www.youtube.com/watch?v=ey8FxYsFAMU)

### Solution
```python
def scheduleCourse(courses: list[list[int]]) -> int:
    """
    Maximum courses using greedy + heap.

    Strategy:
    - Sort by deadline
    - Greedily take courses
    - If can't fit, swap with longest taken course if beneficial
    """
    # Sort by deadline
    courses.sort(key=lambda x: x[1])

    # Max-heap of course durations (negate for max)
    taken = []
    current_time = 0

    for duration, deadline in courses:
        if current_time + duration <= deadline:
            # Can take this course
            heapq.heappush(taken, -duration)
            current_time += duration
        elif taken and -taken[0] > duration:
            # Swap with longest course if current is shorter
            current_time += duration + heapq.heappop(taken)
            heapq.heappush(taken, -duration)

    return len(taken)
```

### Complexity
- **Time**: O(n log n)
- **Space**: O(n)

### Edge Cases
- All courses fit → take all
- No courses fit → return 0
- Single course → 1 if fits, 0 otherwise
- Duration > deadline for all → return 0

---

## Problem 6: Find K Pairs with Smallest Sums (LC #373) - Medium

- [LeetCode](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/)

### Problem Statement
Find k pairs with smallest sums from two sorted arrays.

### Examples
```
Input: nums1 = [1,7,11], nums2 = [2,4,6], k = 3
Output: [[1,2],[1,4],[1,6]]

Pair sums sorted:
  (1,2)=3, (1,4)=5, (1,6)=7, (7,2)=9, ...
```

### Intuition Development
```
BFS-LIKE EXPANSION:
Start with (0,0) → smallest pair.
Next candidates: (0,1) and (1,0).

Visualization as grid:
        nums2
        2   4   6
nums1  ┌───┬───┬───┐
  1    │ 3 │ 5 │ 7 │
       ├───┼───┼───┤
  7    │ 9 │11 │13 │
       ├───┼───┼───┤
  11   │13 │15 │17 │
       └───┴───┴───┘

Start at (0,0)=3
Next: (0,1)=5, (1,0)=9
Pop 5, add (0,2)=7
Pop 7, add nothing new (or (1,1) if not visited)
...
```

### Video Explanation
- [NeetCode - Find K Pairs with Smallest Sums](https://www.youtube.com/watch?v=Hq4VtA6nJy0)

### Solution
```python
def kSmallestPairs(nums1: list[int], nums2: list[int], k: int) -> list[list[int]]:
    """
    Find k smallest sum pairs using min-heap.

    Strategy:
    - Start with (nums1[0], nums2[0])
    - For each pair (i, j), next candidates are (i+1, j) and (i, j+1)
    - Use set to avoid duplicates
    """
    if not nums1 or not nums2:
        return []

    # Min-heap: (sum, i, j)
    heap = [(nums1[0] + nums2[0], 0, 0)]
    visited = {(0, 0)}
    result = []

    while heap and len(result) < k:
        _, i, j = heapq.heappop(heap)
        result.append([nums1[i], nums2[j]])

        # Add next candidates
        if i + 1 < len(nums1) and (i + 1, j) not in visited:
            heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
            visited.add((i + 1, j))

        if j + 1 < len(nums2) and (i, j + 1) not in visited:
            heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
            visited.add((i, j + 1))

    return result
```

### Complexity
- **Time**: O(k log k)
- **Space**: O(k)

### Edge Cases
- k > m×n → return all pairs
- Single element arrays
- Arrays with duplicates
- Very large k

---

## Problem 7: Minimum Cost to Hire K Workers (LC #857) - Hard

- [LeetCode](https://leetcode.com/problems/minimum-cost-to-hire-k-workers/)

### Problem Statement
Hire k workers with minimum cost. Each worker has quality and minimum wage.

### Examples
```
Input: quality = [10,20,5], wage = [70,50,30], k = 2
Output: 105.0

Worker ratios (wage/quality):
  Worker 0: 70/10 = 7
  Worker 1: 50/20 = 2.5
  Worker 2: 30/5 = 6

If we pay at ratio 6 (worker 2's ratio):
  Worker 0: 10 × 6 = 60 < 70 (doesn't meet minimum)
  Worker 1: 20 × 6 = 120 > 50 ✓
  Worker 2: 5 × 6 = 30 ≥ 30 ✓

Cost = 120 + 30 = 150? Let's check ratio 7...
```

### Intuition Development
```
KEY INSIGHT:
All workers must be paid at the same ratio wage/quality.
If ratio = r, worker with quality q gets paid q × r.

For worker i to accept: q_i × r ≥ wage_i
So: r ≥ wage_i / q_i

STRATEGY:
Sort by ratio. For each worker as the maximum ratio:
  - All previous workers will accept (lower ratios)
  - Cost = ratio × sum of qualities
  - Want minimum quality sum → use min-heap to track k smallest

Wait, we want k TOTAL workers with minimum cost.
Actually, use max-heap to keep k smallest qualities.
```

### Video Explanation
- [NeetCode - Minimum Cost to Hire K Workers](https://www.youtube.com/watch?v=o8emK4ehhq0)

### Solution
```python
def mincostToHireWorkers(quality: list[int], wage: list[int], k: int) -> float:
    """
    Minimize hiring cost.

    Key insight: If we pay worker i their minimum wage,
    all workers must be paid at ratio wage[i]/quality[i].

    Strategy:
    - Sort by wage/quality ratio
    - Use max-heap to track k smallest qualities
    """
    n = len(quality)

    # (ratio, quality)
    workers = sorted([(wage[i] / quality[i], quality[i]) for i in range(n)])

    # Max-heap of qualities
    quality_heap = []
    quality_sum = 0
    min_cost = float('inf')

    for ratio, q in workers:
        heapq.heappush(quality_heap, -q)
        quality_sum += q

        if len(quality_heap) > k:
            quality_sum += heapq.heappop(quality_heap)  # Remove largest

        if len(quality_heap) == k:
            min_cost = min(min_cost, ratio * quality_sum)

    return min_cost
```

### Complexity
- **Time**: O(n log n)
- **Space**: O(n)

### Edge Cases
- k = 1 → hire worker with minimum wage
- k = n → must hire all workers
- All same quality → sort by wage
- All same wage → sort by quality

---

## Problem 8: Super Ugly Number (LC #313) - Medium

- [LeetCode](https://leetcode.com/problems/super-ugly-number/)

### Problem Statement
Find nth super ugly number (factors only from given primes).

### Examples
```
Input: n = 12, primes = [2,7,13,19]
Output: 32

Super ugly numbers: 1, 2, 4, 7, 8, 13, 14, 16, 19, 26, 28, 32
  1 × 2 = 2
  2 × 2 = 4
  1 × 7 = 7
  4 × 2 = 8
  ...
```

### Intuition Development
```
MULTI-POINTER WITH HEAP:
Each prime has a pointer into ugly numbers.
Next ugly = min of (prime[i] × ugly[pointer[i]]) for all i.

primes = [2, 7, 13, 19]
ugly = [1]

Step 1: candidates = [2×1, 7×1, 13×1, 19×1] = [2, 7, 13, 19]
  min = 2, ugly = [1, 2], advance pointer for 2

Step 2: candidates = [2×2, 7×1, 13×1, 19×1] = [4, 7, 13, 19]
  min = 4, ugly = [1, 2, 4], advance pointer for 2

Continue...
```

### Video Explanation
- [NeetCode - Super Ugly Number](https://www.youtube.com/watch?v=Lj68VJ1wu84)

### Solution
```python
def nthSuperUglyNumber(n: int, primes: list[int]) -> int:
    """
    Find nth super ugly number using heap.
    """
    # Min-heap: (value, prime, index in ugly array)
    heap = [(p, p, 0) for p in primes]
    heapq.heapify(heap)

    ugly = [1] * n

    for i in range(1, n):
        ugly[i] = heap[0][0]

        # Update all entries that match current ugly number
        while heap[0][0] == ugly[i]:
            val, prime, idx = heapq.heappop(heap)
            heapq.heappush(heap, (prime * ugly[idx + 1], prime, idx + 1))

    return ugly[n - 1]
```

### Complexity
- **Time**: O(n × k × log(nk)) where k = len(primes)
- **Space**: O(n + k)

### Edge Cases
- n = 1 → return 1
- Single prime → powers of that prime
- Large primes → sparse sequence
- Duplicate primes in input (shouldn't happen per problem)

---

## Summary: Advanced Heap Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Sliding Window Median | Two heaps + lazy delete | O(n log k) |
| 2 | Smallest Range K Lists | Track max + min-heap | O(n log k) |
| 3 | Trapping Rain Water II | BFS from boundary | O(mn log mn) |
| 4 | Maximum Performance | Sort + min-heap | O(n log n) |
| 5 | Course Schedule III | Greedy swap | O(n log n) |
| 6 | K Smallest Pairs | BFS-like expansion | O(k log k) |
| 7 | Hire K Workers | Sort by ratio | O(n log n) |
| 8 | Super Ugly Number | Multi-pointer heap | O(nk log k) |

---

## Heap Implementation Tips

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PYTHON HEAPQ TIPS                                        │
│                                                                             │
│  1. MIN-HEAP only: For max-heap, negate values                              │
│     heapq.heappush(heap, -val)                                              │
│     max_val = -heapq.heappop(heap)                                          │
│                                                                             │
│  2. HEAPIFY existing list: O(n)                                             │
│     heapq.heapify(list)                                                     │
│                                                                             │
│  3. PUSH and POP in one operation:                                          │
│     heapq.heappushpop(heap, val)  # Push then pop                           │
│     heapq.heapreplace(heap, val)  # Pop then push                           │
│                                                                             │
│  4. N LARGEST/SMALLEST:                                                     │
│     heapq.nlargest(k, iterable)                                             │
│     heapq.nsmallest(k, iterable)                                            │
│                                                                             │
│  5. CUSTOM COMPARISON: Use tuples                                           │
│     heapq.heappush(heap, (priority, index, value))                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practice More Problems

- [ ] LC #218 - The Skyline Problem
- [ ] LC #239 - Sliding Window Maximum
- [ ] LC #358 - Rearrange String k Distance Apart
- [ ] LC #502 - IPO
- [ ] LC #778 - Swim in Rising Water
