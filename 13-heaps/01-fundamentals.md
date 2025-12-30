# Heaps - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE HEAPS                                        │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Top K" / "K largest" / "K smallest"                                     │
│  ✓ "Kth element"                                                            │
│  ✓ "Median" / "Running median"                                              │
│  ✓ "Merge K sorted"                                                         │
│  ✓ "Priority" / "Schedule"                                                  │
│  ✓ "Streaming data"                                                         │
│                                                                             │
│  Key insight: When you need efficient access to min/max element             │
│               with frequent insertions                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Binary tree structure
- [ ] heapq module in Python
- [ ] Priority queue concept

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HEAPS MEMORY MAP                                         │
│                                                                             │
│                    ┌─────────────┐                                          │
│         ┌─────────│    HEAPS    │─────────┐                                 │
│         │         └─────────────┘         │                                 │
│         ▼                                 ▼                                 │
│  ┌─────────────┐                   ┌─────────────┐                          │
│  │   TOP K     │                   │  TWO HEAPS  │                          │
│  │  PROBLEMS   │                   │   PATTERN   │                          │
│  └──────┬──────┘                   └──────┬──────┘                          │
│         │                                 │                                 │
│    ┌────┴────┐                           │                                  │
│    ▼         ▼                           ▼                                  │
│ ┌──────┐ ┌──────┐                   ┌──────────┐                           │
│ │K Larg│ │K-way │                   │ Running  │                           │
│ │est   │ │Merge │                   │ Median   │                           │
│ └──────┘ └──────┘                   └──────────┘                           │
│                                                                             │
│  Related Patterns:                                                          │
│  • Sorting - Heap sort uses heap                                            │
│  • Dijkstra - Uses min-heap for shortest path                               │
│  • Merge K Lists - Uses heap for efficient merging                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HEAP PROBLEM DECISION TREE                               │
│                                                                             │
│  Need top K elements?                                                       │
│       │                                                                     │
│       ├── K largest → Min-heap of size K                                    │
│       │               (pop when size > K, smallest falls off)               │
│       │                                                                     │
│       ├── K smallest → Max-heap of size K                                   │
│       │                (negate values in Python)                            │
│       │                                                                     │
│       └── NO → Need running median?                                         │
│                    │                                                        │
│                    ├── YES → Two heaps (max-heap for lower half,            │
│                    │         min-heap for upper half)                       │
│                    │                                                        │
│                    └── NO → Need to merge K sorted lists?                   │
│                                 │                                           │
│                                 ├── YES → Min-heap with (val, list_idx)     │
│                                 │                                           │
│                                 └── NO → Consider other patterns            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concept

A heap is a complete binary tree where each parent is smaller (min-heap) or larger (max-heap) than its children.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MIN-HEAP VISUALIZATION                                   │
│                                                                             │
│                         1  ← root (minimum)                                 │
│                        / \                                                  │
│                       3   2                                                 │
│                      / \ / \                                                │
│                     7  4 5  6                                               │
│                                                                             │
│  Property: parent <= children (for min-heap)                                │
│                                                                             │
│  Array representation: [1, 3, 2, 7, 4, 5, 6]                               │
│  Index:                 0  1  2  3  4  5  6                                │
│                                                                             │
│  For index i:                                                               │
│  - Parent: (i - 1) // 2                                                     │
│  - Left child: 2*i + 1                                                      │
│  - Right child: 2*i + 2                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Python heapq Module

```python
import heapq

# ==================== BASIC OPERATIONS ====================

# Create empty heap
heap = []

# Push element - O(log n)
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 4)
# heap = [1, 3, 4] (min-heap)

# Pop minimum - O(log n)
min_val = heapq.heappop(heap)  # Returns 1

# Peek minimum - O(1)
min_val = heap[0]  # Don't pop, just look

# Push and pop in one operation - O(log n)
result = heapq.heappushpop(heap, 2)  # Push 2, pop min

# Pop and push in one operation - O(log n)
result = heapq.heapreplace(heap, 5)  # Pop min, push 5

# ==================== HEAPIFY ====================

# Convert list to heap in-place - O(n)
arr = [3, 1, 4, 1, 5, 9, 2, 6]
heapq.heapify(arr)
# arr is now a valid min-heap

# ==================== MAX-HEAP TRICK ====================
# Python only has min-heap, so negate values for max-heap

max_heap = []
heapq.heappush(max_heap, -5)  # Store -5 instead of 5
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -8)

max_val = -heapq.heappop(max_heap)  # Returns 8 (negated -8)

# ==================== K LARGEST/SMALLEST ====================

arr = [3, 1, 4, 1, 5, 9, 2, 6]

# K smallest - O(n log k)
k_smallest = heapq.nsmallest(3, arr)  # [1, 1, 2]

# K largest - O(n log k)
k_largest = heapq.nlargest(3, arr)  # [9, 6, 5]
```

---

## Common Heap Patterns

### Pattern 1: Top K Elements

```python
def top_k_frequent(nums: list[int], k: int) -> list[int]:
    """
    Find k most frequent elements.

    Strategy:
    - Count frequencies
    - Use min-heap of size k
    - Heap stores (frequency, element)

    Time: O(n log k)
    Space: O(n) for frequency map
    """
    from collections import Counter

    # Count frequencies
    freq = Counter(nums)

    # Min-heap of size k
    # Stores (frequency, element)
    heap = []

    for num, count in freq.items():
        heapq.heappush(heap, (count, num))

        # Keep only k elements
        if len(heap) > k:
            heapq.heappop(heap)  # Remove least frequent

    # Extract elements from heap
    return [num for count, num in heap]
```

### Pattern 2: Kth Largest Element

```python
def findKthLargest(nums: list[int], k: int) -> int:
    """
    Find kth largest element.

    Strategy: Maintain min-heap of size k.
    After processing all elements, root is kth largest.

    Time: O(n log k)
    Space: O(k)
    """
    # Min-heap of size k
    heap = []

    for num in nums:
        heapq.heappush(heap, num)

        # Keep only k largest
        if len(heap) > k:
            heapq.heappop(heap)  # Remove smallest

    # Root is kth largest
    return heap[0]


def findKthLargest_quickselect(nums: list[int], k: int) -> int:
    """
    Alternative: QuickSelect algorithm.

    Time: O(n) average, O(n²) worst
    Space: O(1)
    """
    import random

    k = len(nums) - k  # Convert to kth smallest

    def quickselect(left: int, right: int) -> int:
        # Random pivot to avoid worst case
        pivot_idx = random.randint(left, right)
        nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]
        pivot = nums[right]

        # Partition
        store_idx = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[store_idx], nums[i] = nums[i], nums[store_idx]
                store_idx += 1
        nums[store_idx], nums[right] = nums[right], nums[store_idx]

        # Recurse on correct half
        if store_idx == k:
            return nums[k]
        elif store_idx < k:
            return quickselect(store_idx + 1, right)
        else:
            return quickselect(left, store_idx - 1)

    return quickselect(0, len(nums) - 1)
```

### Pattern 3: Merge K Sorted Lists

```python
def mergeKLists(lists: list) -> 'ListNode':
    """
    Merge k sorted linked lists.

    Strategy:
    - Use min-heap to track smallest element from each list
    - Pop smallest, add its next to heap

    Time: O(n log k) where n = total elements
    Space: O(k) for heap
    """
    # Min-heap: (value, list_index, node)
    heap = []

    # Initialize heap with first element from each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst.val, i, lst))

    dummy = ListNode(0)
    current = dummy

    while heap:
        val, i, node = heapq.heappop(heap)

        # Add to result
        current.next = node
        current = current.next

        # Add next element from same list
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

### Pattern 4: Two Heaps (Median)

```python
class MedianFinder:
    """
    Find median from data stream.

    Strategy:
    - Use two heaps:
      - max_heap: smaller half (store negated for max behavior)
      - min_heap: larger half
    - Keep heaps balanced (size diff <= 1)
    - Median is from heap(s) root(s)

    Time: O(log n) per add, O(1) for median
    Space: O(n)
    """
    def __init__(self):
        # Max-heap for smaller half (negated values)
        self.small = []
        # Min-heap for larger half
        self.large = []

    def addNum(self, num: int) -> None:
        # Add to max-heap (smaller half)
        heapq.heappush(self.small, -num)

        # Balance: move largest from small to large
        heapq.heappush(self.large, -heapq.heappop(self.small))

        # Maintain size: small can have at most 1 more than large
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            # Odd total: median is root of larger heap
            return -self.small[0]
        else:
            # Even total: median is average of both roots
            return (-self.small[0] + self.large[0]) / 2
```

---

## Visual: Two Heaps for Median

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO HEAPS FOR RUNNING MEDIAN                             │
│                                                                             │
│  Numbers: [5, 2, 8, 1, 9]                                                   │
│                                                                             │
│  After adding 5:                                                            │
│  small (max): [5]      large (min): []                                      │
│  Median: 5                                                                  │
│                                                                             │
│  After adding 2:                                                            │
│  small (max): [2]      large (min): [5]                                     │
│  Median: (2 + 5) / 2 = 3.5                                                  │
│                                                                             │
│  After adding 8:                                                            │
│  small (max): [2, 5]   large (min): [8]                                     │
│  Wait, need to balance: move 5 to large                                     │
│  small (max): [2]      large (min): [5, 8]                                  │
│  Rebalance: small should be >= large                                        │
│  small (max): [2, 5]   large (min): [8]                                     │
│  Median: 5                                                                  │
│                                                                             │
│  And so on...                                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complexity Analysis

| Operation | Time |
|-----------|------|
| Push | O(log n) |
| Pop | O(log n) |
| Peek | O(1) |
| Heapify | O(n) |
| nlargest/nsmallest | O(n log k) |

---

## Common Mistakes

```python
# ❌ WRONG: Forgetting Python has min-heap only
max_val = heapq.heappop(heap)  # This gives MINIMUM!

# ✅ CORRECT: Negate for max-heap
heapq.heappush(heap, -val)
max_val = -heapq.heappop(heap)


# ❌ WRONG: Using heap[0] after popping
heapq.heappop(heap)
print(heap[0])  # This is now the NEW minimum!

# ✅ CORRECT: Save the popped value
min_val = heapq.heappop(heap)
print(min_val)


# ❌ WRONG: Modifying heap directly
heap[0] = new_value  # Breaks heap property!

# ✅ CORRECT: Pop and push
heapq.heapreplace(heap, new_value)  # Or heappop + heappush
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"For finding K largest elements, I'll use a min-heap of size K. As I
iterate, I push each element. When heap size exceeds K, I pop the
minimum. At the end, the heap contains the K largest elements."
```

### 2. What Interviewers Look For
- **Heap size optimization**: K-size heap is O(n log k), not O(n log n)
- **Max-heap in Python**: Know to negate values
- **Two heaps pattern**: For median problems

### 3. Common Follow-up Questions
- "Can you do better than O(n log n)?" → Use K-size heap: O(n log k)
- "What if K is close to n?" → Sorting might be simpler
- "How to handle streaming data?" → Heap naturally supports this

---

## Related Patterns

- **Sorting**: Heap sort is O(n log n) with O(1) extra space
- **Graphs**: Dijkstra's algorithm uses min-heap for shortest paths
- **Greedy**: Many greedy problems use heaps for efficient selection

### When to Combine

- **Heap + Sliding Window**: Track max/min in sliding window
- **Two Heaps**: Maintain median with max-heap (left) + min-heap (right)
- **Heap + Merge**: K-way merge of sorted lists

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
