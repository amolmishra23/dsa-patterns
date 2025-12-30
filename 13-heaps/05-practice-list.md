# Heaps - Practice List

## Problems by Pattern

### Top K Elements
- LC 215: Kth Largest Element in Array (Medium)
- LC 347: Top K Frequent Elements (Medium)
- LC 692: Top K Frequent Words (Medium)
- LC 973: K Closest Points to Origin (Medium)
- LC 703: Kth Largest Element in Stream (Easy)

### Two Heaps (Median)
- LC 295: Find Median from Data Stream (Hard)
- LC 480: Sliding Window Median (Hard)
- LC 502: IPO (Hard)

### K-Way Merge
- LC 23: Merge K Sorted Lists (Hard)
- LC 373: Find K Pairs with Smallest Sums (Medium)
- LC 378: Kth Smallest Element in Sorted Matrix (Medium)
- LC 632: Smallest Range Covering K Lists (Hard)

### Scheduling
- LC 253: Meeting Rooms II (Medium)
- LC 621: Task Scheduler (Medium)
- LC 767: Reorganize String (Medium)
- LC 1353: Maximum Number of Events (Medium)

## Templates

```python
import heapq

# Top K Largest (use min heap of size k)
def top_k_largest(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)
    return heap

# Top K Smallest (use max heap via negation)
def top_k_smallest(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, -num)
        if len(heap) > k:
            heapq.heappop(heap)
    return [-x for x in heap]

# Two Heaps for Median
class MedianFinder:
    def __init__(self):
        self.small = []  # max heap (negated)
        self.large = []  # min heap
    
    def addNum(self, num):
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))
    
    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2

# K-Way Merge
def merge_k_lists(lists):
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    
    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    return result
```

## Key Insights
- Python heapq is min heap by default
- For max heap, negate values
- Top K largest → min heap of size K
- Top K smallest → max heap of size K
- Two heaps for running median

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HEAP PATTERNS                                       │
│                                                                             │
│  TOP K LARGEST (Min Heap of size K):                                        │
│  nums = [3,1,5,12,2,11], k=3                                                │
│                                                                             │
│  Process: Push each, pop if size > k                                        │
│  [3] → [1,3] → [1,3,5] → pop 1, [3,5,12] → [2,3,5,12] pop 2                │
│  → [3,5,12] → [3,5,11,12] pop 3 → [5,11,12]                                 │
│                                                                             │
│  Min Heap (size 3):       Result: Top 3 largest = [5,11,12]                 │
│         5                                                                   │
│        / \                                                                  │
│       11  12                                                                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TWO HEAPS (Median):                                                        │
│  Stream: 1, 5, 2, 8, 3                                                      │
│                                                                             │
│  Max Heap (small)    Min Heap (large)    Median                             │
│  ───────────────────────────────────────────────                            │
│       [1]                 []              1                                 │
│       [1]                [5]              3                                 │
│      [1,2]               [5]              2                                 │
│      [1,2]              [5,8]             3.5                               │
│     [1,2,3]             [5,8]             3                                 │
│                                                                             │
│  Invariant: len(small) >= len(large)                                        │
│  Median = small[0] or (small[0] + large[0]) / 2                             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  K-WAY MERGE:                                                               │
│  Lists: [1,4,5], [1,3,4], [2,6]                                             │
│                                                                             │
│  Heap tracks: (value, list_index, element_index)                            │
│                                                                             │
│  Initial: [(1,0,0), (1,1,0), (2,2,0)]                                       │
│  Pop (1,0,0) → Result: [1], Push (4,0,1)                                    │
│  Pop (1,1,0) → Result: [1,1], Push (3,1,1)                                  │
│  Pop (2,2,0) → Result: [1,1,2], Push (6,2,1)                                │
│  Pop (3,1,1) → Result: [1,1,2,3], Push (4,1,2)                              │
│  ...                                                                        │
│  Final: [1,1,2,3,4,4,5,6]                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: Top K Fundamentals
- [ ] LC 703: Kth Largest Element in Stream (Easy)
- [ ] LC 215: Kth Largest Element in Array (Medium)
- [ ] LC 347: Top K Frequent Elements (Medium)
- [ ] LC 692: Top K Frequent Words (Medium)
- [ ] LC 973: K Closest Points to Origin (Medium)

### Week 2: Two Heaps & Scheduling
- [ ] LC 295: Find Median from Data Stream (Hard)
- [ ] LC 480: Sliding Window Median (Hard)
- [ ] LC 253: Meeting Rooms II (Medium)
- [ ] LC 621: Task Scheduler (Medium)
- [ ] LC 767: Reorganize String (Medium)

### Week 3: K-Way Merge & Advanced
- [ ] LC 23: Merge K Sorted Lists (Hard)
- [ ] LC 378: Kth Smallest Element in Sorted Matrix (Medium)
- [ ] LC 373: Find K Pairs with Smallest Sums (Medium)
- [ ] LC 632: Smallest Range Covering K Lists (Hard)
- [ ] LC 502: IPO (Hard)

---

## Common Mistakes

### 1. Confusing Min/Max Heap for Top K
```python
# WRONG - using max heap for top k largest
# This keeps smallest elements, not largest!
heap = []
for num in nums:
    heapq.heappush(heap, -num)  # Wrong!
    if len(heap) > k:
        heapq.heappop(heap)

# CORRECT - use min heap for top k largest
heap = []
for num in nums:
    heapq.heappush(heap, num)  # Min heap
    if len(heap) > k:
        heapq.heappop(heap)  # Removes smallest, keeps k largest
```

### 2. Forgetting to Negate for Max Heap
```python
# WRONG - Python heapq is min heap
heapq.heappush(heap, value)  # This is min heap!

# CORRECT - negate for max heap behavior
heapq.heappush(heap, -value)  # Push negative
max_val = -heapq.heappop(heap)  # Negate back when popping
```

### 3. Comparing Non-Comparable Items
```python
# WRONG - ListNode objects aren't comparable
heapq.heappush(heap, (node.val, node))  # Fails if vals equal!

# CORRECT - add unique index as tiebreaker
heapq.heappush(heap, (node.val, idx, node))  # idx breaks ties
```

### 4. Not Handling Empty Heap
```python
# WRONG - crashes on empty heap
top = heap[0]  # IndexError if empty!

# CORRECT - check first
if heap:
    top = heap[0]
```

### 5. Two Heaps Balance Error
```python
# WRONG - not maintaining balance
def addNum(self, num):
    heapq.heappush(self.small, -num)
    # Missing: balance between heaps!

# CORRECT - always rebalance
def addNum(self, num):
    heapq.heappush(self.small, -num)
    heapq.heappush(self.large, -heapq.heappop(self.small))
    if len(self.large) > len(self.small):
        heapq.heappush(self.small, -heapq.heappop(self.large))
```

---

## Complexity Reference

| Pattern | Time | Space | Notes |
|---------|------|-------|-------|
| Top K | O(n log k) | O(k) | Heap of size k |
| Kth Largest | O(n log k) | O(k) | Same as top k |
| Two Heaps | O(log n) per op | O(n) | Balanced heaps |
| K-Way Merge | O(N log k) | O(k) | k = number of lists |
| Heap Sort | O(n log n) | O(1) | In-place |

---

## Pattern Recognition

| See This | Think This |
|----------|------------|
| "K largest/smallest" | Min/max heap of size K |
| "Running median" | Two heaps (max + min) |
| "Merge K sorted" | Min heap with (val, list_idx, elem_idx) |
| "Top K frequent" | Count + heap |
| "Meeting rooms" | Min heap for end times |
| "Reorganize/schedule" | Max heap by frequency |
| "K pairs" | Min heap, expand lazily |
