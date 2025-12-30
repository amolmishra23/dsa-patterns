# Heaps - Hard Problems

## Problem 1: Find Median from Data Stream (LC #295) - Hard

- [LeetCode](https://leetcode.com/problems/find-median-from-data-stream/)

### Video Explanation
- [NeetCode - Find Median from Data Stream](https://www.youtube.com/watch?v=itmhHWaHupI)

### Problem Statement
Design data structure for continuous median finding.


### Visual Intuition
```
Find Median from Data Stream
addNum sequence: [2, 3, 4]

Pattern: Two Heaps (Max-Heap for left, Min-Heap for right)
Why: Median is at boundary between two halves

Step 0 (Data Structure):
  ┌─────────────────────────────────────────────────┐
  │   maxHeap (small)     │     minHeap (large)     │
  │   ← smaller half →    │    ← larger half →      │
  │                       │                         │
  │   [... x x x MAX]     │     [MIN x x x ...]     │
  │              ↑        │       ↑                 │
  │           top         │      top                │
  │                       │                         │
  │   Median = MAX or (MAX + MIN) / 2               │
  └─────────────────────────────────────────────────┘

Step 1: Add 2
  small: [2]    large: []
          ↑
         max

  Sizes: 1, 0 (balanced)
  Median = 2 (odd count, return max of small)

Step 2: Add 3
  Add to small first: small = [2, 3] → max = 3

  Balance check: max(small) ≤ min(large)?
  3 > empty ✗ → move 3 to large

  small: [2]    large: [3]
          ↑            ↑
         max          min

  Sizes: 1, 1 (balanced)
  Median = (2 + 3) / 2 = 2.5

Step 3: Add 4
  4 > max(small)=2? Yes → add to large

  small: [2]    large: [3, 4]

  Balance check: |small| can be at most |large| + 1
  Sizes: 1, 2 → rebalance! Move min(large) to small

  small: [2, 3]    large: [4]
             ↑            ↑
            max          min

  Sizes: 2, 1 (balanced)
  Median = 3 (odd count, return max of small)

Invariants:
  1. max(small) ≤ min(large)  (left half ≤ right half)
  2. |small| - |large| ∈ {0, 1}  (balanced sizes)

Python Implementation Note:
  Python has min-heap only, so negate values for max-heap:
  small = [-2, -3]  → max = -small[0] = 3
  large = [4]       → min = large[0] = 4

Key Insight:
- Two heaps partition stream into halves
- Tops of heaps give median instantly
- O(log n) add, O(1) median
```

### Solution
```python
import heapq

class MedianFinder:
    """
    Two heaps: max-heap for lower half, min-heap for upper half.

    Time: O(log n) per add, O(1) for median
    Space: O(n)
    """

    def __init__(self):
        self.small = []  # Max heap (negated)
        self.large = []  # Min heap

    def addNum(self, num: int) -> None:
        # Add to max heap (small)
        heapq.heappush(self.small, -num)

        # Ensure max of small <= min of large
        if self.small and self.large and -self.small[0] > self.large[0]:
            heapq.heappush(self.large, -heapq.heappop(self.small))

        # Balance sizes (small can have at most 1 more)
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        elif len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

### Edge Cases
- Single element → return that element
- Two elements → return average
- All same elements → return that element
- Alternating add pattern → rebalance each time

---

## Problem 2: Sliding Window Median (LC #480) - Hard

- [LeetCode](https://leetcode.com/problems/sliding-window-median/)

### Video Explanation
- [NeetCode - Sliding Window Median](https://www.youtube.com/watch?v=HwLJnIl8D2o)

### Problem Statement
Find median of each sliding window.


### Visual Intuition
```
Sliding Window Median (k=3)
nums = [1, 3, -1, -3, 5, 3, 6, 7]

Pattern: Two Heaps + Lazy Deletion
Why: Can't efficiently remove from middle of heap

Step 0 (Challenge):
  Regular median finder + sliding window
  Problem: need to REMOVE elements leaving window
  Heap doesn't support efficient arbitrary removal!

  Solution: Lazy deletion with hashmap

Step 1 (Initialize Window [1, 3, -1]):

  nums: [1, 3, -1, -3, 5, 3, 6, 7]
        【─────────】

  Sorted: [-1, 1, 3]

  small (max-heap): [-1, 1]  → max = 1
  large (min-heap): [3]      → min = 3

  Median = 1 (odd k, return max of small)
  Result: [1]

Step 2 (Slide to [3, -1, -3]):

  nums: [1, 3, -1, -3, 5, 3, 6, 7]
           【──────────】

  Remove: 1, Add: -3

  Lazy deletion: to_remove[1] = 1

  Add -3: -3 ≤ max(small)=1? Yes → add to small

  small: [-1, 1✗, -3]  (1 marked for removal)
  large: [3]

  Balance & prune:
  - When accessing top, skip if marked
  - small top = 1✗ → remove → small top = -1

  Median = -1
  Result: [1, -1]

Step 3 (Slide to [-1, -3, 5]):

  Remove: 3, Add: 5

  to_remove[3] = 1

  Add 5: 5 > max(small)=-1? Yes → add to large

  small: [-1, -3]  → max = -1
  large: [3✗, 5]   → prune: 3✗ removed → min = 5

  Median = -1
  Result: [1, -1, -1]

Lazy Deletion Visualization:
  ┌─────────────────────────────────────────────────┐
  │ to_remove = {1: 1, 3: 1}  (hashmap of counts)   │
  │                                                 │
  │ When accessing heap top:                        │
  │   while top in to_remove and count > 0:         │
  │       pop and decrement count                   │
  │                                                 │
  │ This delays removal until element reaches top   │
  └─────────────────────────────────────────────────┘

Continue sliding...
  Window [-3, 5, 3]: median = 3
  Window [5, 3, 6]:  median = 5
  Window [3, 6, 7]:  median = 6

Final Result: [1, -1, -1, 3, 5, 6]

Key Insight:
- Can't remove from heap middle efficiently
- Mark for lazy deletion, remove when it becomes top
- Maintain balance considering "virtual" sizes
- O(n log k) time overall
```

### Solution
```python
import heapq
from collections import defaultdict

def medianSlidingWindow(nums: list[int], k: int) -> list[float]:
    """
    Two heaps with lazy deletion.

    Time: O(n log k)
    Space: O(k)
    """
    small = []  # Max heap (negated)
    large = []  # Min heap
    to_remove = defaultdict(int)
    result = []

    def get_median():
        if k % 2:
            return -small[0]
        return (-small[0] + large[0]) / 2

    def balance():
        # Balance heap sizes
        while len(small) > len(large) + 1:
            heapq.heappush(large, -heapq.heappop(small))
        while len(large) > len(small):
            heapq.heappush(small, -heapq.heappop(large))

    def prune(heap, is_max):
        # Remove invalid top elements
        while heap:
            val = -heap[0] if is_max else heap[0]
            if to_remove[val]:
                to_remove[val] -= 1
                heapq.heappop(heap)
            else:
                break

    # Initialize first window
    for i in range(k):
        heapq.heappush(small, -nums[i])

    for _ in range(k // 2):
        heapq.heappush(large, -heapq.heappop(small))

    result.append(get_median())

    for i in range(k, len(nums)):
        out_num = nums[i - k]
        in_num = nums[i]

        to_remove[out_num] += 1

        # Add new number
        if in_num <= -small[0]:
            heapq.heappush(small, -in_num)
        else:
            heapq.heappush(large, in_num)

        balance()
        prune(small, True)
        prune(large, False)

        result.append(get_median())

    return result
```

### Edge Cases
- k = 1 → single median
- k = n → just that element
- All same elements → return that element
- Duplicate values → handle in lazy deletion

---

## Problem 3: IPO (LC #502) - Hard

- [LeetCode](https://leetcode.com/problems/ipo/)

### Video Explanation
- [NeetCode - IPO](https://www.youtube.com/watch?v=1IUzNJ6TPEM)

### Problem Statement
Maximize capital after k projects given capital requirements.


### Visual Intuition
```
IPO (Maximize Capital)
k = 2, w = 0 (initial capital)
profits = [1, 2, 3], capital = [0, 1, 1]

Pattern: Two Heaps (Min by Capital, Max by Profit)
Why: Find affordable projects, pick most profitable

Step 0 (Setup Projects):

  Project A: profit=1, needs capital=0
  Project B: profit=2, needs capital=1
  Project C: profit=3, needs capital=1

  min-heap by capital: [(0,1), (1,2), (1,3)]
                        cap,profit pairs

Step 1 (Capital = 0, pick project 1 of 2):

  Move affordable projects to profit max-heap:

  capital_heap: [(0,1), (1,2), (1,3)]
                  ↑
                 0 ≤ 0 ✓ → move to available

  available (max-heap by profit): [(1)]

  Pick highest profit: 1

  Capital: 0 + 1 = 1

  ┌────────────────────────────────────────┐
  │ Before: w = 0                          │
  │ After:  w = 1  (+1 from project A)     │
  └────────────────────────────────────────┘

Step 2 (Capital = 1, pick project 2 of 2):

  Move newly affordable projects:

  capital_heap: [(1,2), (1,3)]
                  ↑       ↑
                 1 ≤ 1 ✓  1 ≤ 1 ✓

  available: [2, 3]  (both now affordable!)

  Pick highest profit: 3

  Capital: 1 + 3 = 4

  ┌────────────────────────────────────────┐
  │ Before: w = 1                          │
  │ After:  w = 4  (+3 from project C)     │
  └────────────────────────────────────────┘

Answer: 4 (final capital after k=2 projects)

Greedy Strategy Visualization:

  Round 1 (w=0):
    Affordable: [A(profit=1)]
    Pick: A ★
    w → 1

  Round 2 (w=1):
    Affordable: [B(profit=2), C(profit=3)]
    Pick: C ★ (highest profit)
    w → 4

  Why greedy works:
  - More capital unlocks more projects
  - Picking highest profit maximizes future options
  - Never regret picking highest available profit

Key Insight:
- min-heap: quickly find newly affordable projects
- max-heap: quickly find most profitable among affordable
- Greedy: always pick highest profit we can afford
- O(n log n) for n projects
```

### Solution
```python
import heapq

def findMaximizedCapital(k: int, w: int, profits: list[int], capital: list[int]) -> int:
    """
    Greedy with two heaps.

    Strategy:
    - Min heap for capital requirements
    - Max heap for available profits
    - Always pick highest profit affordable project

    Time: O(n log n)
    Space: O(n)
    """
    n = len(profits)

    # (capital_needed, profit)
    projects = [(capital[i], profits[i]) for i in range(n)]
    heapq.heapify(projects)

    available = []  # Max heap of profits (negated)

    for _ in range(k):
        # Move all affordable projects to available heap
        while projects and projects[0][0] <= w:
            cap, profit = heapq.heappop(projects)
            heapq.heappush(available, -profit)

        if not available:
            break

        # Pick most profitable
        w += -heapq.heappop(available)

    return w
```

### Edge Cases
- No affordable projects → return initial capital
- k > projects → do all affordable
- All projects affordable → pick k highest profit
- Single project → check if affordable

---

## Problem 4: Merge K Sorted Lists (LC #23) - Hard

- [LeetCode](https://leetcode.com/problems/merge-k-sorted-lists/)

### Video Explanation
- [NeetCode - Merge K Sorted Lists](https://www.youtube.com/watch?v=q5a5OiGbT6Q)

### Problem Statement
Merge k sorted linked lists into one sorted list.

### Visual Intuition
```
Merge K Sorted Lists
lists = [[1,4,5], [1,3,4], [2,6]]

Pattern: Min-Heap of K Pointers (one per list)
Why: Always need smallest among k current elements

Step 0 (Initialize - One Element from Each List):

  List 0: 1 → 4 → 5
          ↑
  List 1: 1 → 3 → 4
          ↑
  List 2: 2 → 6
          ↑

  Heap: [(1, list0), (1, list1), (2, list2)]
         ↑
        min

Step 1 (Pop min, push next from same list):

  Pop: (1, list0) → result = [1]
  Push: 4 from list0

  List 0: 1 → 4 → 5
              ↑ (moved)

  Heap: [(1, list1), (2, list2), (4, list0)]

Step 2:
  Pop: (1, list1) → result = [1, 1]
  Push: 3 from list1

  Heap: [(2, list2), (3, list1), (4, list0)]

Step 3:
  Pop: (2, list2) → result = [1, 1, 2]
  Push: 6 from list2

  Heap: [(3, list1), (4, list0), (6, list2)]

Step 4:
  Pop: (3, list1) → result = [1, 1, 2, 3]
  Push: 4 from list1

  Heap: [(4, list0), (4, list1), (6, list2)]

Continue...
  Pop 4 (list0) → push 5
  Pop 4 (list1) → list1 exhausted, no push
  Pop 5 (list0) → list0 exhausted
  Pop 6 (list2) → list2 exhausted

Final Result: [1, 1, 2, 3, 4, 4, 5, 6]

Heap State Trace:
  ┌─────────────────────────────────────────────────┐
  │ Step │ Heap Contents      │ Result So Far       │
  ├─────────────────────────────────────────────────┤
  │  0   │ [1, 1, 2]          │ []                  │
  │  1   │ [1, 2, 4]          │ [1]                 │
  │  2   │ [2, 3, 4]          │ [1,1]               │
  │  3   │ [3, 4, 6]          │ [1,1,2]             │
  │  4   │ [4, 4, 6]          │ [1,1,2,3]           │
  │  5   │ [4, 5, 6]          │ [1,1,2,3,4]         │
  │  6   │ [5, 6]             │ [1,1,2,3,4,4]       │
  │  7   │ [6]                │ [1,1,2,3,4,4,5]     │
  │  8   │ []                 │ [1,1,2,3,4,4,5,6]   │
  └─────────────────────────────────────────────────┘

Key Insight:
- Heap always has at most K elements (one per list)
- Each element pushed/popped once → O(N log K)
- Alternative: Divide & Conquer (merge pairs)
```


### Intuition
```
Lists:
  1 → 4 → 5
  1 → 3 → 4
  2 → 6

Min heap always contains smallest unprocessed element from each list.
Pop minimum, add its next element to heap.

Heap: [1, 1, 2] → pop 1 → [1, 2, 4] → pop 1 → [2, 3, 4] → ...
```

### Solution
```python
import heapq

def mergeKLists(lists: list) -> 'ListNode':
    """
    Min heap to always get smallest element.

    Strategy:
    - Add first element of each list to heap
    - Pop smallest, add its next to heap
    - Repeat until heap empty

    Time: O(N log k) where N = total elements
    Space: O(k) for heap
    """
    # Handle edge cases
    if not lists:
        return None

    # Min heap: (value, list_index, node)
    # list_index breaks ties (nodes aren't comparable)
    heap = []

    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))

    dummy = ListNode(0)
    curr = dummy

    while heap:
        val, idx, node = heapq.heappop(heap)

        # Add to result
        curr.next = node
        curr = curr.next

        # Add next node from same list
        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))

    return dummy.next
```

### Divide and Conquer Alternative
```python
def mergeKLists(lists: list) -> 'ListNode':
    """
    Divide and conquer - merge pairs of lists.

    Time: O(N log k)
    Space: O(log k) for recursion
    """
    def merge_two(l1, l2):
        dummy = ListNode(0)
        curr = dummy

        while l1 and l2:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next

        curr.next = l1 or l2
        return dummy.next

    if not lists:
        return None

    # Merge pairs until one list remains
    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i + 1] if i + 1 < len(lists) else None
            merged.append(merge_two(l1, l2))
        lists = merged

    return lists[0]
```

### Complexity
- **Time**: O(N log k)
- **Space**: O(k) for heap, O(log k) for divide & conquer

### Edge Cases
- Empty list → return None
- Single list → return that list
- All empty lists → return None
- Lists of different lengths → heap handles

---

## Problem 5: Trapping Rain Water II (LC #407) - Hard

- [LeetCode](https://leetcode.com/problems/trapping-rain-water-ii/)

### Video Explanation
- [NeetCode - Trapping Rain Water II](https://www.youtube.com/watch?v=cJayBq38VYw)

### Problem Statement
Calculate trapped water on a 2D elevation map.

### Visual Intuition
```
Trapping Rain Water II (3D version)
heightMap = [[1,4,3,1,3,2],
             [3,2,1,3,2,4],
             [2,3,3,2,3,1]]

Pattern: BFS from Boundary with Min-Heap
Why: Water level = lowest path to boundary (water escapes there)

Step 0 (Understand the Problem):

  3D elevation map (bird's eye view):
  ┌───┬───┬───┬───┬───┬───┐
  │ 1 │ 4 │ 3 │ 1 │ 3 │ 2 │  ← boundary
  ├───┼───┼───┼───┼───┼───┤
  │ 3 │ 2 │ 1 │ 3 │ 2 │ 4 │
  ├───┼───┼───┼───┼───┼───┤
  │ 2 │ 3 │ 3 │ 2 │ 3 │ 1 │  ← boundary
  └───┴───┴───┴───┴───┴───┘
    ↑                   ↑
  boundary           boundary

  Water fills from lowest boundary inward

Step 1 (Initialize - Add Boundary to Min-Heap):

  Heap (by height): all boundary cells

  ┌───┬───┬───┬───┬───┬───┐
  │ ● │ ● │ ● │ ● │ ● │ ● │  ● = in heap
  ├───┼───┼───┼───┼───┼───┤
  │ ● │   │   │   │   │ ● │
  ├───┼───┼───┼───┼───┼───┤
  │ ● │ ● │ ● │ ● │ ● │ ● │
  └───┴───┴───┴───┴───┴───┘

  Heap sorted by height: [1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4]

Step 2 (Process Lowest Boundary First):

  Pop height=1 at (0,0), max_height=1
  Neighbor (1,0) height=3: no water (3 > 1)

  Pop height=1 at (0,3), max_height=1
  Neighbor (1,3) height=3: no water

  Pop height=1 at (2,5), max_height=1
  Neighbor (1,5) height=4: no water

Step 3 (Find Water Trap):

  Pop height=2 at (0,5), max_height=2
  Pop height=2 at (2,0), max_height=2
  Pop height=2 at (2,3), max_height=2

  Eventually reach interior cell (1,2) with height=1:

  ┌───┬───┬───┬───┬───┬───┐
  │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │
  ├───┼───┼───┼───┼───┼───┤
  │ ✓ │ ✓ │[1]│   │   │ ✓ │  ← height=1, max_height=3
  ├───┼───┼───┼───┼───┼───┤       water = 3-1 = 2
  │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │ ✓ │
  └───┴───┴───┴───┴───┴───┘

  max_height tracks highest boundary seen on path
  Water trapped = max_height - cell_height

Step 4 (Continue BFS):

  For each interior cell:
  - Pop from heap (lowest unvisited)
  - Update max_height = max(max_height, cell_height)
  - Water at cell = max_height - cell_height (if positive)
  - Add unvisited neighbors to heap

Total Water Calculation:
  Cell (1,2): height=1, max_height=3 → water=2
  Cell (1,4): height=2, max_height=3 → water=1
  ...

  Total = sum of all water trapped

Key Insight:
- Water escapes through lowest path to boundary
- Process cells by height (min-heap)
- max_height = water level at current cell
- BFS ensures we find optimal escape path
```


### Intuition
```
Elevation map:
[[1,4,3,1,3,2],
 [3,2,1,3,2,4],
 [2,3,3,2,3,1]]

Water level at any cell = min height of surrounding boundary.
Process from lowest boundary cells inward using min heap.
```

### Solution
```python
import heapq

def trapRainWater(heightMap: list[list[int]]) -> int:
    """
    Min heap BFS from boundaries inward.

    Strategy:
    - Start with all boundary cells in min heap
    - Process lowest cell first
    - Water trapped = max_height_seen - cell_height
    - Add unvisited neighbors to heap

    Time: O(mn log(mn))
    Space: O(mn)
    """
    if not heightMap or not heightMap[0]:
        return 0

    m, n = len(heightMap), len(heightMap[0])
    visited = [[False] * n for _ in range(m)]
    heap = []  # (height, row, col)

    # Add all boundary cells to heap
    for i in range(m):
        for j in range(n):
            if i == 0 or i == m - 1 or j == 0 or j == n - 1:
                heapq.heappush(heap, (heightMap[i][j], i, j))
                visited[i][j] = True

    water = 0
    max_height = 0

    while heap:
        height, row, col = heapq.heappop(heap)

        # Update max boundary height seen
        max_height = max(max_height, height)

        # Water trapped at this cell
        water += max_height - height

        # Process neighbors
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = row + dr, col + dc

            if 0 <= nr < m and 0 <= nc < n and not visited[nr][nc]:
                visited[nr][nc] = True
                heapq.heappush(heap, (heightMap[nr][nc], nr, nc))

    return water
```

### Complexity
- **Time**: O(mn log(mn))
- **Space**: O(mn)

### Edge Cases
- Single cell → return 0
- All same height → return 0
- Boundary is highest → no water trapped
- Valley in middle → water fills valley

---

## Problem 6: Smallest Range Covering Elements from K Lists (LC #632) - Hard

- [LeetCode](https://leetcode.com/problems/smallest-range-covering-elements-from-k-lists/)

### Video Explanation
- [NeetCode - Smallest Range Covering Elements from K Lists](https://www.youtube.com/watch?v=Fqal25ZgEDo)

### Problem Statement
Find smallest range that includes at least one number from each of k lists.

### Visual Intuition
```
Smallest Range Covering Elements from K Lists
nums = [[4,10,15,24,26], [0,9,12,20,28], [5,18,22,30]]

Pattern: Min-Heap + Track Current Max
Why: Range = [heap_min, current_max], minimize this

Step 0 (Understand the Goal):

  Need one element from EACH list
  Find smallest range [a, b] containing one from each

  List 0: [4, 10, 15, 24, 26]
  List 1: [0, 9, 12, 20, 28]
  List 2: [5, 18, 22, 30]

  Example valid ranges:
  [0, 5]: has 4, 0, 5 ✓ size=5
  [20, 24]: has 24, 20, 22 ✓ size=4 ★ smaller!

Step 1 (Initialize - First Element from Each):

  Heap: [(0, list1), (4, list0), (5, list2)]
         ↑
        min=0

  current_max = 5
  Range = [0, 5], size = 5

Step 2 (Pop min, push next from same list):

  Pop: (0, list1) → push (9, list1)

  Heap: [(4, list0), (5, list2), (9, list1)]
         ↑
        min=4

  current_max = max(5, 9) = 9
  Range = [4, 9], size = 5 (not better)

Step 3:
  Pop: (4, list0) → push (10, list0)

  Heap: [(5, list2), (9, list1), (10, list0)]

  current_max = 10
  Range = [5, 10], size = 5

Step 4:
  Pop: (5, list2) → push (18, list2)

  Heap: [(9, list1), (10, list0), (18, list2)]

  current_max = 18
  Range = [9, 18], size = 9 (worse!)

Continue...
  ┌──────┬────────────────┬─────────────────┐
  │ Step │ Heap Min       │ Range           │
  ├──────┼────────────────┼─────────────────┤
  │  5   │ 10             │ [10, 18] = 8    │
  │  6   │ 12             │ [12, 18] = 6    │
  │  7   │ 15             │ [15, 18] = 3 ★  │
  │  8   │ 18             │ [18, 20] = 2 ★★ │
  │  9   │ 20             │ [20, 24] = 4    │
  │ ...  │ ...            │ ...             │
  └──────┴────────────────┴─────────────────┘

Wait, let me trace more carefully...

  At some point: heap has [20, 22, 24]
  Range = [20, 24], size = 4

  Pop 20, push 28: heap = [22, 24, 28]
  Range = [22, 28], size = 6 (worse)

  Pop 22, push 30: heap = [24, 28, 30]
  Range = [24, 30], size = 6

  Pop 24, push 26: heap = [26, 28, 30]
  Range = [26, 30], size = 4

  Pop 26, list0 exhausted → STOP

Best found: [20, 24] with size = 4

Key Insight:
- Heap always has exactly K elements (one per list)
- Range = [min (heap top), max (tracked separately)]
- Advance min pointer to try smaller range
- Stop when any list exhausted (can't include all)
- O(N log K) where N = total elements
```


### Intuition
```
Lists:
[4, 10, 15, 24, 26]
[0, 9, 12, 20]
[5, 18, 22, 30]

Need range containing one from each list.
Track: current_min (heap top), current_max (tracked separately)
Range = [current_min, current_max]
```

### Solution
```python
import heapq

def smallestRange(nums: list[list[int]]) -> list[int]:
    """
    Min heap tracking one element from each list.

    Strategy:
    - Initialize heap with first element from each list
    - Track current maximum
    - Range = [heap_min, current_max]
    - Pop min, push its next element, update max
    - Stop when any list is exhausted

    Time: O(N log k) where N = total elements
    Space: O(k)
    """
    # (value, list_index, element_index)
    heap = []
    current_max = float('-inf')

    # Initialize with first element from each list
    for i, lst in enumerate(nums):
        heapq.heappush(heap, (lst[0], i, 0))
        current_max = max(current_max, lst[0])

    result = [float('-inf'), float('inf')]

    while heap:
        current_min, list_idx, elem_idx = heapq.heappop(heap)

        # Update result if current range is smaller
        if current_max - current_min < result[1] - result[0]:
            result = [current_min, current_max]

        # Move to next element in the list
        if elem_idx + 1 < len(nums[list_idx]):
            next_val = nums[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
            current_max = max(current_max, next_val)
        else:
            # One list exhausted - can't include all lists anymore
            break

    return result
```

### Complexity
- **Time**: O(N log k)
- **Space**: O(k)

### Edge Cases
- Single list → return [min, max] of that list
- All same elements → return [elem, elem]
- Lists of length 1 → range spans all elements
- Overlapping ranges → find smallest

---

## Summary

| # | Problem | Key Technique |
|---|---------|---------------|
| 1 | Find Median | Two heaps balanced |
| 2 | Sliding Window Median | Two heaps + lazy deletion |
| 3 | IPO | Greedy + two heaps |
| 4 | Merge K Sorted Lists | Min heap or divide & conquer |
| 5 | Trapping Rain Water II | BFS from boundary with min heap |
| 6 | Smallest Range K Lists | Min heap + max tracking |
