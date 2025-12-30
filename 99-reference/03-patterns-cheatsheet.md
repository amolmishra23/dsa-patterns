# DSA Patterns Quick Reference - Code Templates

> **Quick Reference Card** - Copy-paste ready code templates for all patterns.
> For comprehensive complexity tables, edge cases, and interview tips, see [01-cheat-sheet.md](./01-cheat-sheet.md)

---

## Pattern Recognition Guide

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KEYWORD → PATTERN MAPPING                                │
│                                                                             │
│  "Sorted array" → Binary Search, Two Pointers                               │
│  "Subarray/substring" → Sliding Window, Prefix Sum                          │
│  "All combinations/permutations" → Backtracking                             │
│  "Shortest path" → BFS (unweighted), Dijkstra (weighted)                    │
│  "Optimal/min/max" → DP, Greedy                                             │
│  "Top K" → Heap                                                             │
│  "Tree traversal" → DFS, BFS                                                │
│  "Connected components" → Union-Find, DFS                                   │
│  "Prefix matching" → Trie                                                   │
│  "Intervals/ranges" → Sort + Greedy                                         │
│  "Parentheses/brackets" → Stack                                             │
│  "Next greater/smaller" → Monotonic Stack                                   │
│  "Single/missing number" → Bit Manipulation                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Two Pointers

```python
# Opposite direction (sorted array)
def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        total = nums[left] + nums[right]
        if total == target:
            return [left, right]
        elif total < target:
            left += 1
        else:
            right -= 1

# Same direction (fast-slow)
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

---

## Sliding Window

```python
# Fixed size window
def max_sum_k(nums, k):
    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)
    return max_sum

# Variable size window
def min_subarray_sum(nums, target):
    left = window_sum = 0
    min_len = float('inf')
    for right in range(len(nums)):
        window_sum += nums[right]
        while window_sum >= target:
            min_len = min(min_len, right - left + 1)
            window_sum -= nums[left]
            left += 1
    return min_len if min_len != float('inf') else 0
```

---

## Binary Search

```python
# Standard binary search
def binary_search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Find leftmost (lower bound)
def lower_bound(nums, target):
    left, right = 0, len(nums)
    while left < right:
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left

# Search on answer space
def min_capacity(weights, days):
    left, right = max(weights), sum(weights)
    while left < right:
        mid = (left + right) // 2
        if can_ship(weights, days, mid):
            right = mid
        else:
            left = mid + 1
    return left
```

---

## Prefix Sum

```python
# Build prefix sum
def build_prefix(nums):
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)
    return prefix

# Range sum query
def range_sum(prefix, i, j):
    return prefix[j + 1] - prefix[i]

# Subarray sum equals K
def subarray_sum_k(nums, k):
    count = 0
    prefix_sum = 0
    prefix_count = {0: 1}
    for num in nums:
        prefix_sum += num
        count += prefix_count.get(prefix_sum - k, 0)
        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1
    return count
```

---

## DFS / BFS

```python
# Tree DFS (recursive)
def dfs(node):
    if not node:
        return
    # Process node
    dfs(node.left)
    dfs(node.right)

# Tree BFS (level order)
def bfs(root):
    if not root:
        return []
    queue = deque([root])
    result = []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result

# Graph DFS
def dfs_graph(node, graph, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_graph(neighbor, graph, visited)
```

---

## Backtracking

```python
# Template
def backtrack(path, choices):
    if is_solution(path):
        result.append(path[:])
        return

    for choice in choices:
        if is_valid(choice):
            path.append(choice)      # Make choice
            backtrack(path, choices)  # Recurse
            path.pop()               # Undo choice

# Subsets
def subsets(nums):
    result = []
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return result

# Permutations
def permutations(nums):
    result = []
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False] * len(nums))
    return result
```

---

## Dynamic Programming

```python
# 1D DP (Fibonacci pattern)
def climb_stairs(n):
    if n <= 2:
        return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

# 2D DP (Grid)
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]

# Knapsack
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w],
                              dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]
```

---

## Heap

```python
import heapq

# Top K elements
def top_k_frequent(nums, k):
    freq = Counter(nums)
    return heapq.nlargest(k, freq.keys(), key=freq.get)

# Merge K sorted lists
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
            heapq.heappush(heap, (lists[list_idx][elem_idx + 1], list_idx, elem_idx + 1))
    return result
```

---

## Union-Find

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

---

## Monotonic Stack

```python
# Next Greater Element
def next_greater(nums):
    n = len(nums)
    result = [-1] * n
    stack = []
    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    return result

# Largest Rectangle in Histogram
def largest_rectangle(heights):
    heights = heights + [0]
    stack = []
    max_area = 0
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i - stack[-1] - 1 if stack else i
            max_area = max(max_area, height * width)
        stack.append(i)
    return max_area
```

---

## Trie

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

---

## Bit Manipulation

```python
# Common operations
n & 1           # Check if odd
n & (n - 1)     # Clear lowest set bit
n & (-n)        # Get lowest set bit
n | (1 << i)    # Set i-th bit
n & ~(1 << i)   # Clear i-th bit
n ^ (1 << i)    # Toggle i-th bit

# Single Number (XOR)
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result

# Count set bits
def count_bits(n):
    count = 0
    while n:
        n &= (n - 1)
        count += 1
    return count
```

---

## Complexity Quick Reference

| Pattern | Time | Space |
|---------|------|-------|
| Two Pointers | O(n) | O(1) |
| Sliding Window | O(n) | O(k) |
| Binary Search | O(log n) | O(1) |
| DFS/BFS | O(V + E) | O(V) |
| Backtracking | O(2^n) or O(n!) | O(n) |
| DP | O(n²) typical | O(n) |
| Heap | O(n log k) | O(k) |
| Union-Find | O(α(n)) | O(n) |
| Trie | O(L) | O(AL) |
| Dijkstra | O((V+E) log V) | O(V) |

---

## Problem Solving Framework

```
1. UNDERSTAND
   - Read problem carefully
   - Identify input/output
   - Note constraints

2. PATTERN MATCH
   - Use keyword mapping
   - Consider data structure hints

3. PLAN
   - Choose algorithm/pattern
   - Think through edge cases

4. IMPLEMENT
   - Write clean code
   - Use meaningful names

5. VERIFY
   - Test with examples
   - Check edge cases
   - Analyze complexity
```

