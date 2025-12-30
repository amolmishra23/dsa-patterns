# Heaps - Practice Problems

## Problem 1: Kth Largest Element (LC #215) - Medium

- [LeetCode](https://leetcode.com/problems/kth-largest-element-in-an-array/)

### Problem Statement
Find the kth largest element in unsorted array.

### Examples
```
Input: nums = [3,2,1,5,6,4], k = 2
Output: 5
```

### Video Explanation
- [NeetCode - Kth Largest Element](https://www.youtube.com/watch?v=XEmy13g1Qxc)

### Intuition
```
Two approaches: Min-Heap of size k OR Quickselect

MIN-HEAP APPROACH:
- Keep a heap of the k LARGEST elements seen so far
- Use MIN-heap so smallest of k largest is at root
- Root = kth largest!

Visual: nums = [3,2,1,5,6,4], k = 2

        Process 3: heap = [3]
        Process 2: heap = [2,3]
        Process 1: heap = [2,3], 1 < 2 so ignored (heap full)
        Process 5: heap = [3,5], pop 2, push 5
        Process 6: heap = [5,6], pop 3, push 6
        Process 4: heap = [5,6], 4 < 5 so ignored

        Root = 5 = 2nd largest ✓

QUICKSELECT: O(n) average, based on quicksort partition
```

### Solution
```python
import heapq

def findKthLargest(nums: list[int], k: int) -> int:
    """
    Find kth largest using min-heap of size k.

    Strategy:
    - Maintain min-heap of k largest elements
    - Root is the kth largest

    Time: O(n log k)
    Space: O(k)
    """
    # Min-heap of k largest elements
    heap = []

    for num in nums:
        heapq.heappush(heap, num)

        # Keep only k elements
        if len(heap) > k:
            heapq.heappop(heap)

    # Root is kth largest
    return heap[0]


def findKthLargest_quickselect(nums: list[int], k: int) -> int:
    """
    Alternative: Quickselect algorithm.

    Time: O(n) average, O(n²) worst
    Space: O(1)
    """
    import random

    def partition(left, right, pivot_idx):
        pivot = nums[pivot_idx]
        # Move pivot to end
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]

        store_idx = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[store_idx], nums[i] = nums[i], nums[store_idx]
                store_idx += 1

        # Move pivot to final position
        nums[store_idx], nums[right] = nums[right], nums[store_idx]
        return store_idx

    # Convert to index of kth largest
    target = len(nums) - k
    left, right = 0, len(nums) - 1

    while left <= right:
        pivot_idx = random.randint(left, right)
        pivot_idx = partition(left, right, pivot_idx)

        if pivot_idx == target:
            return nums[pivot_idx]
        elif pivot_idx < target:
            left = pivot_idx + 1
        else:
            right = pivot_idx - 1

    return -1
```

### Edge Cases
- k = 1 → return max element
- k = n → return min element (sorted)
- All elements same → return that element
- Array already sorted → quickselect still O(n) avg
- k larger than array → invalid input

---

## Problem 2: Top K Frequent Elements (LC #347) - Medium

- [LeetCode](https://leetcode.com/problems/top-k-frequent-elements/)

### Problem Statement
Return k most frequent elements.

### Examples
```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

### Video Explanation
- [NeetCode - Top K Frequent Elements](https://www.youtube.com/watch?v=YPTqKIgVk-k)

### Intuition
```
Two approaches: Heap or Bucket Sort

HEAP APPROACH (O(n log k)):
- Count frequencies using hash map
- Use min-heap of size k by frequency
- Heap contains k most frequent elements

BUCKET SORT (O(n)):
- Create buckets where index = frequency
- Collect from highest frequency buckets

Visual: nums = [1,1,1,2,2,3], k = 2

        Frequencies: {1: 3, 2: 2, 3: 1}

        Bucket approach:
        Index:  0    1    2    3
        Bucket: []  [3]  [2]  [1]

        Collect from right: [1, 2] ← top 2 frequent
```

### Solution
```python
from collections import Counter

def topKFrequent(nums: list[int], k: int) -> list[int]:
    """
    Find k most frequent elements using heap.

    Strategy:
    - Count frequencies
    - Use min-heap of size k by frequency

    Time: O(n log k)
    Space: O(n)
    """
    freq = Counter(nums)

    # Min-heap: (frequency, element)
    heap = []

    for num, count in freq.items():
        heapq.heappush(heap, (count, num))

        if len(heap) > k:
            heapq.heappop(heap)

    return [num for count, num in heap]


def topKFrequent_bucket(nums: list[int], k: int) -> list[int]:
    """
    Alternative: Bucket sort approach.

    Time: O(n)
    Space: O(n)
    """
    freq = Counter(nums)

    # Bucket: index = frequency, value = list of elements
    buckets = [[] for _ in range(len(nums) + 1)]

    for num, count in freq.items():
        buckets[count].append(num)

    # Collect top k from highest frequency
    result = []
    for i in range(len(buckets) - 1, -1, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result

    return result
```

### Edge Cases
- k = 1 → return most frequent element
- All elements same frequency → return any k
- k equals unique elements → return all unique
- Single element array → return that element
- Large k → handle with bucket sort for O(n)

---

## Problem 3: Find Median from Data Stream (LC #295) - Hard

- [LeetCode](https://leetcode.com/problems/find-median-from-data-stream/)

### Problem Statement
Design data structure for streaming median.

### Video Explanation
- [NeetCode - Find Median from Data Stream](https://www.youtube.com/watch?v=itmhHWaHupI)

### Intuition
```
Use TWO HEAPS to maintain sorted halves!

max_heap: smaller half (left side)
min_heap: larger half (right side)

Visual:
        Stream: 1, 5, 2, 8, 3

        Add 1: max_heap=[1], min_heap=[]
               median = 1

        Add 5: max_heap=[1], min_heap=[5]
               median = (1+5)/2 = 3

        Add 2: max_heap=[2,1], min_heap=[5]
               median = 2

        Add 8: max_heap=[2,1], min_heap=[5,8]
               median = (2+5)/2 = 3.5

        Add 3: max_heap=[3,2,1], min_heap=[5,8]
               median = 3

Key invariants:
1. All elements in max_heap ≤ all elements in min_heap
2. Size difference ≤ 1
```

### Solution
```python
class MedianFinder:
    """
    Find median from data stream using two heaps.

    Strategy:
    - max_heap: smaller half (store negatives)
    - min_heap: larger half
    - Keep heaps balanced (size difference <= 1)

    Time: O(log n) addNum, O(1) findMedian
    Space: O(n)
    """

    def __init__(self):
        # Max heap for smaller half (negate for max behavior)
        self.max_heap = []
        # Min heap for larger half
        self.min_heap = []

    def addNum(self, num: int) -> None:
        """Add number to data structure."""
        # Add to max_heap first
        heapq.heappush(self.max_heap, -num)

        # Balance: move largest from max_heap to min_heap
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))

        # Ensure max_heap has equal or one more element
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def findMedian(self) -> float:
        """Return median of all elements."""
        if len(self.max_heap) > len(self.min_heap):
            return -self.max_heap[0]
        else:
            return (-self.max_heap[0] + self.min_heap[0]) / 2
```

### Edge Cases
- Single element → return that element
- Two elements → return average
- All same values → return that value
- Alternating large/small → heaps rebalance
- Integer overflow → use float division

---

## Problem 4: Merge K Sorted Lists (LC #23) - Hard

- [LeetCode](https://leetcode.com/problems/merge-k-sorted-lists/)

### Problem Statement
Merge k sorted linked lists.

### Video Explanation
- [NeetCode - Merge K Sorted Lists](https://www.youtube.com/watch?v=q5a5OiGbT6Q)

### Intuition
```
Use MIN-HEAP to always get the smallest element!

Visual: lists = [[1,4,5], [1,3,4], [2,6]]

        Heap: [(1, list0), (1, list1), (2, list2)]

        Pop 1 (list0): result = [1], add 4 from list0
        Heap: [(1, list1), (2, list2), (4, list0)]

        Pop 1 (list1): result = [1,1], add 3 from list1
        Heap: [(2, list2), (3, list1), (4, list0)]

        Pop 2 (list2): result = [1,1,2], add 6 from list2
        ...continue...

        Final: [1,1,2,3,4,4,5,6]

Why heap? Always O(log k) to find minimum across k lists!
```

### Solution
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def mergeKLists(lists: list[ListNode]) -> ListNode:
    """
    Merge k sorted lists using min-heap.

    Strategy:
    - Add first node of each list to heap
    - Pop smallest, add its next to heap

    Time: O(N log k) where N = total nodes
    Space: O(k)
    """
    # Min-heap: (value, list_index, node)
    heap = []

    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))

    dummy = ListNode(0)
    current = dummy

    while heap:
        val, idx, node = heapq.heappop(heap)

        current.next = node
        current = current.next

        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))

    return dummy.next
```

### Edge Cases
- Empty list of lists → return None
- Single list → return that list
- All empty lists → return None
- Lists of different lengths → heap handles
- Duplicate values across lists → handle with index

---

## Problem 5: K Closest Points to Origin (LC #973) - Medium

- [LeetCode](https://leetcode.com/problems/k-closest-points-to-origin/)

### Problem Statement
Find k closest points to origin.

### Examples
```
Input: points = [[1,3],[-2,2]], k = 1
Output: [[-2,2]]
```

### Video Explanation
- [NeetCode - K Closest Points](https://www.youtube.com/watch?v=rI2EBUEMfTk)

### Intuition
```
Same pattern as "Kth Largest" - use heap of size k!

Use MAX-HEAP of size k (furthest of k closest at root).
If new point is closer than root, replace it.

Visual: points = [[1,3],[-2,2],[3,3]], k = 2

        Distance from origin:
        [1,3]:  √(1² + 3²) = √10 ≈ 3.16
        [-2,2]: √(2² + 2²) = √8  ≈ 2.83  ← closest
        [3,3]:  √(3² + 3²) = √18 ≈ 4.24

        Max-heap of size 2 (store negative distance for max behavior):
        Process [1,3]:  heap = [(-10, [1,3])]
        Process [-2,2]: heap = [(-10, [1,3]), (-8, [-2,2])]
        Process [3,3]:  -18 < -10, so [3,3] not added

        Result: [[1,3], [-2,2]]
```

### Solution
```python
def kClosest(points: list[list[int]], k: int) -> list[list[int]]:
    """
    Find k closest points using max-heap.

    Strategy:
    - Use max-heap of size k (negate distances)
    - Keep k smallest distances

    Time: O(n log k)
    Space: O(k)
    """
    # Max-heap: (-distance², point)
    heap = []

    for x, y in points:
        dist = x * x + y * y

        heapq.heappush(heap, (-dist, [x, y]))

        if len(heap) > k:
            heapq.heappop(heap)

    return [point for dist, point in heap]
```

### Edge Cases
- k = 1 → return closest point
- k = n → return all points
- Points at same distance → return any k
- Point at origin → distance 0
- Negative coordinates → squared distance still positive

---

## Problem 6: Task Scheduler (LC #621) - Medium

- [LeetCode](https://leetcode.com/problems/task-scheduler/)

### Problem Statement
Find minimum time to complete all tasks with cooldown.

### Examples
```
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8 (A -> B -> idle -> A -> B -> idle -> A -> B)
```

### Video Explanation
- [NeetCode - Task Scheduler](https://www.youtube.com/watch?v=s8p8ukTyA2I)

### Intuition
```
Greedy: Always schedule the MOST FREQUENT task!

Why? Most frequent task creates the most constraints.
By scheduling it first, we minimize idle time.

Visual: tasks = [A,A,A,B,B,B], n = 2

        Frequencies: A=3, B=3

        Schedule:
        Time 1: A (A=2 left, cooldown until time 4)
        Time 2: B (B=2 left, cooldown until time 5)
        Time 3: idle (no available tasks)
        Time 4: A (A=1 left, cooldown until time 7)
        Time 5: B (B=1 left, cooldown until time 8)
        Time 6: idle
        Time 7: A (done!)
        Time 8: B (done!)

        Total: 8 time units

Math formula: (max_freq - 1) * (n + 1) + count_of_max_freq
```

### Solution
```python
from collections import Counter

def leastInterval(tasks: list[str], n: int) -> int:
    """
    Minimum time with cooldown using max-heap.

    Strategy:
    - Always execute most frequent task
    - Track cooldown with queue

    Time: O(total_tasks)
    Space: O(26) = O(1)
    """
    freq = Counter(tasks)

    # Max-heap of frequencies (negate for max)
    heap = [-count for count in freq.values()]
    heapq.heapify(heap)

    time = 0
    cooldown = []  # (available_time, count)

    while heap or cooldown:
        time += 1

        if heap:
            count = heapq.heappop(heap) + 1  # Execute task (increment negative)

            if count < 0:
                cooldown.append((time + n, count))

        # Check if any task is available
        if cooldown and cooldown[0][0] == time:
            _, count = cooldown.pop(0)
            heapq.heappush(heap, count)

    return time


def leastInterval_math(tasks: list[str], n: int) -> int:
    """
    Alternative: Mathematical approach.

    Time: O(n)
    Space: O(26)
    """
    freq = Counter(tasks)
    max_freq = max(freq.values())
    max_count = sum(1 for f in freq.values() if f == max_freq)

    # Minimum time = (max_freq - 1) * (n + 1) + max_count
    # But can't be less than total tasks
    return max(len(tasks), (max_freq - 1) * (n + 1) + max_count)
```

### Edge Cases
- n = 0 → no cooldown, just count tasks
- Single task type → (count-1) * (n+1) + 1
- All different tasks → just len(tasks)
- Empty tasks → return 0
- Many task types → fewer idle slots needed

---

## Problem 7: Reorganize String (LC #767) - Medium

- [LeetCode](https://leetcode.com/problems/reorganize-string/)

### Video Explanation
- [NeetCode - Reorganize String](https://www.youtube.com/watch?v=2g_b1aYTHeg)

### Problem Statement
Rearrange string so no adjacent characters are same.

### Solution
```python
def reorganizeString(s: str) -> str:
    """
    Reorganize string using max-heap.

    Strategy:
    - Always place most frequent character
    - Save previous character to avoid adjacent duplicates

    Time: O(n log 26) = O(n)
    Space: O(26) = O(1)
    """
    freq = Counter(s)

    # Check if possible
    max_freq = max(freq.values())
    if max_freq > (len(s) + 1) // 2:
        return ""

    # Max-heap: (-count, char)
    heap = [(-count, char) for char, count in freq.items()]
    heapq.heapify(heap)

    result = []
    prev_count, prev_char = 0, ''

    while heap:
        count, char = heapq.heappop(heap)
        result.append(char)

        # Add previous back if still has count
        if prev_count < 0:
            heapq.heappush(heap, (prev_count, prev_char))

        # Save current for next iteration
        prev_count = count + 1  # Decrement (it's negative)
        prev_char = char

    return ''.join(result)
```

### Edge Cases
- Single character → return it
- Two same characters → return "" if len > 2
- All different characters → return any arrangement
- One char dominates (> (n+1)/2) → return ""
- Empty string → return ""

---

## Problem 8: Kth Smallest Element in Sorted Matrix (LC #378) - Medium

- [LeetCode](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

### Problem Statement
Find kth smallest element in row/column sorted matrix.

### Video Explanation
- [NeetCode - Kth Smallest in Sorted Matrix](https://www.youtube.com/watch?v=v3d6WFCpPmU)

### Intuition
```
Similar to "Merge K Sorted Lists" - each row is a sorted list!

Use min-heap to always get next smallest element.

Visual: matrix = [[1,5,9],[10,11,13],[12,13,15]], k=8

        Think of it as 3 sorted lists:
        Row 0: [1, 5, 9]
        Row 1: [10, 11, 13]
        Row 2: [12, 13, 15]

        Heap starts with first element of each row:
        [(1,0,0), (10,1,0), (12,2,0)]

        Pop 1 → add 5 from row 0
        Pop 5 → add 9 from row 0
        Pop 9 → row 0 exhausted
        Pop 10 → add 11 from row 1
        ...

        8th element = 13
```

### Solution
```python
def kthSmallest(matrix: list[list[int]], k: int) -> int:
    """
    Find kth smallest in sorted matrix using min-heap.

    Strategy:
    - Start with first element of each row
    - Pop smallest, add next from same row

    Time: O(k log n)
    Space: O(n)
    """
    n = len(matrix)

    # Min-heap: (value, row, col)
    heap = [(matrix[i][0], i, 0) for i in range(min(n, k))]
    heapq.heapify(heap)

    for _ in range(k):
        val, row, col = heapq.heappop(heap)

        if col + 1 < n:
            heapq.heappush(heap, (matrix[row][col + 1], row, col + 1))

    return val


def kthSmallest_binary_search(matrix: list[list[int]], k: int) -> int:
    """
    Alternative: Binary search on value range.

    Time: O(n log(max - min))
    Space: O(1)
    """
    n = len(matrix)

    def count_less_equal(target):
        """Count elements <= target."""
        count = 0
        row, col = n - 1, 0

        while row >= 0 and col < n:
            if matrix[row][col] <= target:
                count += row + 1
                col += 1
            else:
                row -= 1

        return count

    left, right = matrix[0][0], matrix[n-1][n-1]

    while left < right:
        mid = (left + right) // 2

        if count_less_equal(mid) < k:
            left = mid + 1
        else:
            right = mid

    return left
```

### Edge Cases
- k = 1 → return top-left element
- k = n*n → return bottom-right element
- 1x1 matrix → return single element
- All same values → return that value
- k larger than n*n → invalid

---

## Problem 9: Ugly Number II (LC #264) - Medium

- [LeetCode](https://leetcode.com/problems/ugly-number-ii/)

### Problem Statement
Find nth ugly number (factors only 2, 3, 5).

### Video Explanation
- [NeetCode - Ugly Number II](https://www.youtube.com/watch?v=Lj68VJ1wu84)

### Solution
```python
def nthUglyNumber(n: int) -> int:
    """
    Find nth ugly number using min-heap.

    Strategy:
    - Start with 1
    - For each ugly number, generate next by multiplying 2, 3, 5
    - Use set to avoid duplicates

    Time: O(n log n)
    Space: O(n)
    """
    heap = [1]
    seen = {1}
    factors = [2, 3, 5]

    ugly = 1

    for _ in range(n):
        ugly = heapq.heappop(heap)

        for f in factors:
            next_ugly = ugly * f
            if next_ugly not in seen:
                seen.add(next_ugly)
                heapq.heappush(heap, next_ugly)

    return ugly


def nthUglyNumber_dp(n: int) -> int:
    """
    Alternative: DP with three pointers.

    Time: O(n)
    Space: O(n)
    """
    ugly = [1] * n
    p2 = p3 = p5 = 0

    for i in range(1, n):
        next2 = ugly[p2] * 2
        next3 = ugly[p3] * 3
        next5 = ugly[p5] * 5

        ugly[i] = min(next2, next3, next5)

        if ugly[i] == next2:
            p2 += 1
        if ugly[i] == next3:
            p3 += 1
        if ugly[i] == next5:
            p5 += 1

    return ugly[n - 1]
```

### Edge Cases
- n = 1 → return 1
- n = 2 → return 2
- Large n → DP approach more efficient
- Duplicates in sequence → pointers skip together
- Overflow for large n → use appropriate data type

---

## Problem 10: IPO (LC #502) - Hard

- [LeetCode](https://leetcode.com/problems/ipo/)

### Problem Statement
Maximize capital after completing at most k projects.

### Video Explanation
- [NeetCode - IPO](https://www.youtube.com/watch?v=1IUzNJ6TPEM)

### Solution
```python
def findMaximizedCapital(k: int, w: int, profits: list[int], capital: list[int]) -> int:
    """
    Maximize capital using two heaps.

    Strategy:
    - Sort projects by capital requirement
    - Use max-heap for available projects (by profit)
    - Greedily pick highest profit we can afford

    Time: O(n log n)
    Space: O(n)
    """
    n = len(profits)

    # (capital, profit) sorted by capital
    projects = sorted(zip(capital, profits))

    available = []  # Max-heap of profits (negative)
    i = 0

    for _ in range(k):
        # Add all projects we can now afford
        while i < n and projects[i][0] <= w:
            heapq.heappush(available, -projects[i][1])
            i += 1

        if not available:
            break

        # Take most profitable
        w += -heapq.heappop(available)

    return w
```

### Edge Cases
- k = 0 → return initial capital w
- No affordable projects → return w
- All projects affordable → pick k highest profit
- Single project → either take it or not
- k larger than projects → take all affordable

---

## Summary: Heap Problems

| # | Problem | Heap Type | Time |
|---|---------|-----------|------|
| 1 | Kth Largest | Min-heap size k | O(n log k) |
| 2 | Top K Frequent | Min-heap size k | O(n log k) |
| 3 | Streaming Median | Two heaps | O(log n) add |
| 4 | Merge K Lists | Min-heap | O(N log k) |
| 5 | K Closest Points | Max-heap size k | O(n log k) |
| 6 | Task Scheduler | Max-heap + cooldown | O(n) |
| 7 | Reorganize String | Max-heap | O(n) |
| 8 | Kth in Matrix | Min-heap | O(k log n) |
| 9 | Ugly Number | Min-heap + seen | O(n log n) |
| 10 | IPO | Sort + max-heap | O(n log n) |

---

## Practice More Problems

- [ ] LC #253 - Meeting Rooms II
- [ ] LC #355 - Design Twitter
- [ ] LC #407 - Trapping Rain Water II
- [ ] LC #632 - Smallest Range Covering Elements from K Lists
- [ ] LC #857 - Minimum Cost to Hire K Workers

