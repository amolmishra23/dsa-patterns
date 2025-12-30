# DSA Patterns Cheat Sheet

> **Comprehensive Reference** - For quick code templates only, see [03-patterns-cheatsheet.md](./03-patterns-cheatsheet.md)

---

## Quick Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PATTERN RECOGNITION QUICK GUIDE                          │
│                                                                             │
│  "Sorted array" + find element ────────────────► Binary Search              │
│  "Sorted array" + find pair ───────────────────► Two Pointers               │
│  "Subarray/substring" + condition ─────────────► Sliding Window             │
│  "Subarray sum" (with negatives) ──────────────► Prefix Sum + Hash Map      │
│  "All combinations/permutations" ──────────────► Backtracking               │
│  "Count ways" / "Min cost" ────────────────────► Dynamic Programming        │
│  "Shortest path" (unweighted) ─────────────────► BFS                        │
│  "Shortest path" (weighted) ───────────────────► Dijkstra                   │
│  "Connected components" ───────────────────────► DFS/BFS or Union-Find      │
│  "Dependencies/ordering" ──────────────────────► Topological Sort           │
│  "Top K elements" ─────────────────────────────► Heap                       │
│  "Next greater/smaller" ───────────────────────► Monotonic Stack            │
│  "Overlapping intervals" ──────────────────────► Sort + Merge               │
│  "Prefix matching" ────────────────────────────► Trie                       │
│  "Linked list cycle" ──────────────────────────► Fast & Slow Pointers       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Time Complexity Reference

### Common Complexities (Best to Worst)
```
O(1) < O(log n) < O(n) < O(n log n) < O(n²) < O(2ⁿ) < O(n!)
```

### Data Structure Operations

| Data Structure | Access | Search | Insert | Delete |
|---------------|--------|--------|--------|--------|
| Array | O(1) | O(n) | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1)* | O(1)* |
| Hash Table | N/A | O(1)† | O(1)† | O(1)† |
| BST (balanced) | O(log n) | O(log n) | O(log n) | O(log n) |
| Heap | O(1)‡ | O(n) | O(log n) | O(log n) |
| Stack | O(n) | O(n) | O(1) | O(1) |
| Queue | O(n) | O(n) | O(1) | O(1) |

*With reference to node, †Average case, ‡Min/Max only

### Algorithm Complexities

| Algorithm | Time | Space |
|-----------|------|-------|
| Binary Search | O(log n) | O(1) |
| Two Pointers | O(n) | O(1) |
| Sliding Window | O(n) | O(k) |
| DFS/BFS | O(V + E) | O(V) |
| Merge Sort | O(n log n) | O(n) |
| Quick Sort | O(n log n)† | O(log n) |
| Heap Sort | O(n log n) | O(1) |
| Dijkstra | O((V+E) log V) | O(V) |
| Topological Sort | O(V + E) | O(V) |

†Average case

---

## Pattern Templates

### Binary Search
```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

### Two Pointers (Opposite Direction)
```python
def two_pointers(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current = arr[left] + arr[right]
        if current == target:
            return [left, right]
        elif current < target:
            left += 1
        else:
            right -= 1
    return []
```

### Sliding Window (Variable)
```python
def sliding_window(arr):
    left = 0
    window_state = {}
    result = 0

    for right in range(len(arr)):
        # Expand: add arr[right] to window

        while not valid(window_state):
            # Contract: remove arr[left] from window
            left += 1

        result = max(result, right - left + 1)

    return result
```

### BFS
```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

### DFS
```python
def dfs(graph, node, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
```

### Backtracking
```python
def backtrack(path, choices):
    if is_solution(path):
        result.append(path[:])
        return

    for choice in choices:
        if is_valid(choice):
            path.append(choice)      # Make choice
            backtrack(path, remaining_choices)
            path.pop()               # Undo choice
```

### Dynamic Programming
```python
# Top-Down (Memoization)
@lru_cache(maxsize=None)
def dp(state):
    if base_case(state):
        return base_value
    return combine(dp(smaller_state))

# Bottom-Up (Tabulation)
def dp_bottom_up(n):
    dp = [0] * (n + 1)
    dp[0] = base_case
    for i in range(1, n + 1):
        dp[i] = recurrence(dp[i-1], ...)
    return dp[n]
```

---

## Python Tips for Interviews

### Collections
```python
from collections import Counter, defaultdict, deque

# Counter - frequency counting
freq = Counter([1, 2, 2, 3])  # {1: 1, 2: 2, 3: 1}

# defaultdict - auto-initialize
graph = defaultdict(list)
graph[0].append(1)  # No KeyError

# deque - efficient queue
queue = deque([1, 2, 3])
queue.append(4)     # Right
queue.appendleft(0) # Left
queue.pop()         # Right
queue.popleft()     # Left
```

### Heap
```python
import heapq

# Min heap
heap = []
heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
min_val = heapq.heappop(heap)  # 1

# Max heap (negate values)
heapq.heappush(heap, -val)
max_val = -heapq.heappop(heap)

# Heapify existing list
heapq.heapify(arr)  # O(n)

# K largest/smallest
heapq.nlargest(k, arr)
heapq.nsmallest(k, arr)
```

### Sorting
```python
# Sort in place
arr.sort()
arr.sort(reverse=True)
arr.sort(key=lambda x: x[1])

# Return new sorted list
sorted_arr = sorted(arr)
sorted_arr = sorted(arr, key=lambda x: -x)

# Custom comparator
from functools import cmp_to_key
arr.sort(key=cmp_to_key(compare_func))
```

### Useful Built-ins
```python
# Enumerate
for i, val in enumerate(arr):
    print(i, val)

# Zip
for a, b in zip(arr1, arr2):
    print(a, b)

# Map
squares = list(map(lambda x: x**2, arr))

# Filter
evens = list(filter(lambda x: x % 2 == 0, arr))

# Any/All
any(x > 0 for x in arr)  # True if any positive
all(x > 0 for x in arr)  # True if all positive
```

---

## Common Edge Cases

### Arrays
- Empty array
- Single element
- All same elements
- Already sorted / reverse sorted
- Negative numbers
- Integer overflow

### Strings
- Empty string
- Single character
- All same characters
- Case sensitivity
- Unicode characters

### Linked Lists
- Empty list (head = None)
- Single node
- Cycle present

### Trees
- Empty tree (root = None)
- Single node
- Skewed tree
- Negative values

### Graphs
- Disconnected components
- Cycles
- Self-loops
- No edges

---

## Interview Checklist

### Before Coding
- [ ] Clarify inputs, outputs, constraints
- [ ] Work through examples
- [ ] Identify the pattern
- [ ] Discuss approach and complexity
- [ ] Consider edge cases

### While Coding
- [ ] Write clean, readable code
- [ ] Use meaningful variable names
- [ ] Handle edge cases
- [ ] Think out loud

### After Coding
- [ ] Trace through with example
- [ ] Test edge cases
- [ ] State time and space complexity
- [ ] Discuss optimizations

---

## Problem Difficulty Guide

Based on constraints:
- n ≤ 10: O(n!) acceptable
- n ≤ 20: O(2ⁿ) acceptable
- n ≤ 500: O(n³) acceptable
- n ≤ 10⁴: O(n²) acceptable
- n ≤ 10⁶: O(n log n) needed
- n ≤ 10⁸: O(n) needed
- n > 10⁸: O(log n) or O(1) needed
