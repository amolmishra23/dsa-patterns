# DSA Foundations - Course Introduction

## Welcome to Pattern-Based Problem Solving

This course teaches you to **recognize patterns**, not memorize solutions. After completing this course, when you see a new problem, your brain will automatically:

1. **Identify keywords** that signal which pattern to use
2. **Recall the template** for that pattern
3. **Adapt the template** to the specific problem


## Prerequisites

Before diving into this course, ensure you have:

### Programming Fundamentals
- **Python basics**: variables, loops, conditionals, functions
- **Object-oriented concepts**: classes, methods, inheritance
- **Basic debugging skills**: reading error messages, using print statements

### Mathematical Foundation
- **Basic math**: arithmetic, modulo operations, integer division
- **Logarithms**: understanding log₂(n) for complexity analysis
- **Combinatorics basics**: permutations, combinations (helpful for backtracking)

### Mindset Requirements
- **Patience**: Some patterns take multiple attempts to internalize
- **Active practice**: Reading is not enough—you must code solutions
- **Pattern thinking**: Focus on "what type of problem is this?" not "what's the trick?"

```
Checklist before starting:
┌────────────────────────────────────────────────────────┐
│ □ Can write basic Python functions                     │
│ □ Understand time complexity basics (O(n), O(n²))      │
│ □ Know what arrays, lists, dictionaries are            │
│ □ Can implement simple recursion                       │
│ □ Have LeetCode/HackerRank account ready               │
│ □ Committed to practicing 1-2 problems daily           │
└────────────────────────────────────────────────────────┘
```

---

## Memory Map

Visual overview of when to use each data structure and pattern:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA STRUCTURE MEMORY MAP                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SEQUENTIAL ACCESS              FAST LOOKUP                HIERARCHICAL     │
│  ┌─────────────┐               ┌─────────────┐            ┌─────────────┐   │
│  │   Array     │               │  Hash Map   │            │    Tree     │   │
│  │  O(1) index │               │  O(1) get   │            │ O(log n)    │   │
│  │  O(n) search│               │  O(1) set   │            │ search      │   │
│  └─────────────┘               └─────────────┘            └─────────────┘   │
│        │                              │                          │          │
│        ▼                              ▼                          ▼          │
│  ┌─────────────┐               ┌─────────────┐            ┌─────────────┐   │
│  │ Two Pointer │               │ Counting    │            │ DFS / BFS   │   │
│  │ Sliding Win │               │ Frequency   │            │ Traversal   │   │
│  └─────────────┘               └─────────────┘            └─────────────┘   │
│                                                                             │
│  ORDERED DATA                  PRIORITY                   RELATIONSHIPS     │
│  ┌─────────────┐               ┌─────────────┐            ┌─────────────┐   │
│  │ Sorted Arr  │               │    Heap     │            │   Graph     │   │
│  │ Binary Srch │               │  O(log n)   │            │ Adjacency   │   │
│  │  O(log n)   │               │  push/pop   │            │    List     │   │
│  └─────────────┘               └─────────────┘            └─────────────┘   │
│        │                              │                          │          │
│        ▼                              ▼                          ▼          │
│  ┌─────────────┐               ┌─────────────┐            ┌─────────────┐   │
│  │ Binary Srch │               │ Top K       │            │ BFS/DFS     │   │
│  │ Pattern     │               │ Median      │            │ Union Find  │   │
│  └─────────────┘               └─────────────┘            └─────────────┘   │
│                                                                             │
│  LIFO/FIFO                     PREFIX COMPUTATION         STRING MATCHING   │
│  ┌─────────────┐               ┌─────────────┐            ┌─────────────┐   │
│  │Stack / Queue│               │ Prefix Sum  │            │    Trie     │   │
│  │  O(1) ops   │               │ O(1) range  │            │ O(m) search │   │
│  └─────────────┘               │   queries   │            └─────────────┘   │
│        │                       └─────────────┘                              │
│        ▼                                                                    │
│  ┌─────────────┐                                                            │
│  │ Monotonic   │                                                            │
│  │ Next Greater│                                                            │
│  └─────────────┘                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

Use this flowchart to identify which pattern to apply:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PATTERN SELECTION DECISION TREE                     │
└─────────────────────────────────────────────────────────────────────────────┘

START: What does the problem ask for?
       │
       ├─► "Find/search in sorted data"
       │         │
       │         └──► BINARY SEARCH
       │
       ├─► "Subarray/substring with condition"
       │         │
       │         ├─► Fixed size window? ──► SLIDING WINDOW (fixed)
       │         ├─► Variable size? ──► SLIDING WINDOW (variable)
       │         └─► Range sum queries? ──► PREFIX SUM
       │
       ├─► "Two elements satisfying condition"
       │         │
       │         ├─► Array is sorted? ──► TWO POINTERS
       │         └─► Need O(1) lookup? ──► HASH MAP
       │
       ├─► "All combinations/permutations/subsets"
       │         │
       │         └──► BACKTRACKING
       │
       ├─► "Shortest path / minimum steps"
       │         │
       │         ├─► Unweighted graph? ──► BFS
       │         ├─► Weighted graph? ──► DIJKSTRA
       │         └─► Negative weights? ──► BELLMAN-FORD
       │
       ├─► "Count ways / optimal value"
       │         │
       │         ├─► Overlapping subproblems? ──► DYNAMIC PROGRAMMING
       │         └─► Greedy choice works? ──► GREEDY
       │
       ├─► "Next greater/smaller element"
       │         │
       │         └──► MONOTONIC STACK
       │
       ├─► "Top K / Median / Priority"
       │         │
       │         └──► HEAP
       │
       ├─► "Connected components / Union"
       │         │
       │         └──► UNION-FIND
       │
       ├─► "Tree traversal / Path problems"
       │         │
       │         ├─► Level-by-level? ──► BFS
       │         └─► Path/subtree? ──► DFS
       │
       └─► "String prefix matching"
                 │
                 └──► TRIE

Quick Reference Table:
┌──────────────────────┬─────────────────────────────────────────┐
│ KEYWORD              │ LIKELY PATTERN                          │
├──────────────────────┼─────────────────────────────────────────┤
│ "sorted"             │ Binary Search, Two Pointers             │
│ "subarray sum"       │ Prefix Sum, Sliding Window              │
│ "contiguous"         │ Sliding Window, Kadane's                │
│ "all possibilities"  │ Backtracking, DFS                       │
│ "minimum/maximum"    │ DP, Binary Search, Greedy               │
│ "k-th largest"       │ Heap, Quickselect                       │
│ "connected"          │ Union-Find, DFS/BFS                     │
│ "levels"             │ BFS                                     │
│ "palindrome"         │ Two Pointers, DP                        │
│ "intervals"          │ Sorting + Greedy, Sweep Line            │
└──────────────────────┴─────────────────────────────────────────┘
```

---

---

## The Mental Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         THE PATTERN RECOGNITION SYSTEM                       │
│                                                                             │
│    PROBLEM                                                                  │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────┐    ┌──────────────────────────────────────────────────┐   │
│  │  KEYWORDS   │───►│  "sorted" → Binary Search                        │   │
│  │  DETECTOR   │    │  "subarray" → Sliding Window / Prefix Sum        │   │
│  └─────────────┘    │  "all combinations" → Backtracking               │   │
│                     │  "shortest path" → BFS / Dijkstra                │   │
│                     │  "count ways" → Dynamic Programming              │   │
│                     └──────────────────────────────────────────────────┘   │
│                            │                                                │
│                            ▼                                                │
│                     ┌──────────────┐                                        │
│                     │   TEMPLATE   │   Each pattern has 5-10 line template  │
│                     │   LIBRARY    │   that solves 80% of problems          │
│                     └──────────────┘                                        │
│                            │                                                │
│                            ▼                                                │
│                     ┌──────────────┐                                        │
│                     │   SOLUTION   │   Customize template for specifics     │
│                     └──────────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Data Structures You Must Know

### 1. Arrays
- **What**: Contiguous memory, O(1) access by index
- **When to use**: Random access, iteration, when order matters
- **Key operations**: Access O(1), Search O(n), Insert/Delete O(n)

```python
# Array basics
arr = [1, 2, 3, 4, 5]
arr[2]           # O(1) access
arr.append(6)    # O(1) amortized
arr.pop()        # O(1)
arr.insert(0, 0) # O(n)
```

### 2. Hash Map (Dictionary)
- **What**: Key-value store with O(1) average operations
- **When to use**: Frequency counting, lookup tables, caching
- **Key operations**: Insert O(1), Lookup O(1), Delete O(1)

```python
# Hash map basics
d = {}
d['key'] = 'value'          # O(1) insert
val = d.get('key', default) # O(1) lookup
del d['key']                # O(1) delete

# Common patterns
from collections import Counter, defaultdict
freq = Counter([1, 2, 2, 3])  # {1: 1, 2: 2, 3: 1}
graph = defaultdict(list)      # adjacency list
```

### 3. Hash Set
- **What**: Unordered collection of unique elements
- **When to use**: Membership testing, removing duplicates
- **Key operations**: Add O(1), Contains O(1), Remove O(1)

```python
# Set basics
s = set()
s.add(1)        # O(1)
1 in s          # O(1)
s.remove(1)     # O(1)

# Set operations
a | b  # union
a & b  # intersection
a - b  # difference
```

### 4. Linked List
- **What**: Nodes connected by pointers
- **When to use**: Frequent insertions/deletions, unknown size
- **Key operations**: Insert O(1), Delete O(1), Search O(n)

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Traversal
curr = head
while curr:
    print(curr.val)
    curr = curr.next
```

### 5. Stack (LIFO)
- **What**: Last-In-First-Out data structure
- **When to use**: Matching brackets, undo operations, DFS
- **Key operations**: Push O(1), Pop O(1), Peek O(1)

```python
stack = []
stack.append(1)  # push
stack.pop()      # pop
stack[-1]        # peek
```

### 6. Queue (FIFO)
- **What**: First-In-First-Out data structure
- **When to use**: BFS, task scheduling, buffering
- **Key operations**: Enqueue O(1), Dequeue O(1)

```python
from collections import deque
queue = deque()
queue.append(1)    # enqueue (right)
queue.popleft()    # dequeue (left)
```

### 7. Heap (Priority Queue)
- **What**: Tree-based structure with min/max at root
- **When to use**: Top-K problems, scheduling, Dijkstra
- **Key operations**: Insert O(log n), Extract O(log n), Peek O(1)

```python
import heapq
heap = []
heapq.heappush(heap, 3)  # insert
heapq.heappop(heap)       # extract min
heap[0]                   # peek min

# Max heap: negate values
heapq.heappush(heap, -val)
-heapq.heappop(heap)
```

### 8. Binary Tree
- **What**: Hierarchical structure with at most 2 children per node
- **When to use**: Hierarchical data, searching, sorting

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Traversals
def inorder(root):    # Left, Root, Right
    if root:
        inorder(root.left)
        print(root.val)
        inorder(root.right)

def preorder(root):   # Root, Left, Right
    if root:
        print(root.val)
        preorder(root.left)
        preorder(root.right)

def postorder(root):  # Left, Right, Root
    if root:
        postorder(root.left)
        postorder(root.right)
        print(root.val)
```

### 9. Graph
- **What**: Vertices connected by edges
- **When to use**: Networks, relationships, paths

```python
# Adjacency List (most common)
graph = defaultdict(list)
graph[0].append(1)  # edge 0 -> 1
graph[1].append(0)  # undirected: add both

# Adjacency Matrix
matrix = [[0] * n for _ in range(n)]
matrix[0][1] = 1  # edge 0 -> 1
```

---

## Visual: Data Structure Selection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHICH DATA STRUCTURE TO USE?                             │
│                                                                             │
│  Need O(1) lookup by key? ──────────────────────────► HASH MAP             │
│                                                                             │
│  Need O(1) membership check? ───────────────────────► HASH SET             │
│                                                                             │
│  Need O(1) access by index? ────────────────────────► ARRAY                │
│                                                                             │
│  Need to maintain sorted order?                                             │
│    └─► With O(log n) operations ────────────────────► HEAP / BST           │
│    └─► Just need sorted once ───────────────────────► ARRAY + SORT         │
│                                                                             │
│  Need LIFO (last in, first out)? ───────────────────► STACK                │
│                                                                             │
│  Need FIFO (first in, first out)? ──────────────────► QUEUE                │
│                                                                             │
│  Need frequent insert/delete in middle? ────────────► LINKED LIST          │
│                                                                             │
│  Need to track min/max efficiently? ────────────────► HEAP                 │
│                                                                             │
│  Need prefix matching? ─────────────────────────────► TRIE                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem-Solving Framework

For every problem, follow this systematic approach:

### Step 1: Understand
- Read the problem **twice**
- Identify inputs, outputs, constraints
- Work through examples by hand
- Ask clarifying questions

### Step 2: Plan
- What pattern does this match?
- What data structures do I need?
- What's the brute force? Can I optimize?
- What are the edge cases?

### Step 3: Implement
- Write clean, readable code
- Use meaningful variable names
- Add comments for complex logic

### Step 4: Verify
- Test with given examples
- Test edge cases (empty, single element, large)
- Trace through code mentally

### Step 5: Analyze
- Time complexity
- Space complexity
- Can it be optimized further?

---

## Common Edge Cases to Consider

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EDGE CASES CHECKLIST                                │
│                                                                             │
│  Arrays:                                                                    │
│  □ Empty array                                                              │
│  □ Single element                                                           │
│  □ All same elements                                                        │
│  □ Already sorted / reverse sorted                                          │
│  □ Negative numbers                                                         │
│  □ Integer overflow                                                         │
│                                                                             │
│  Strings:                                                                   │
│  □ Empty string                                                             │
│  □ Single character                                                         │
│  □ All same characters                                                      │
│  □ Case sensitivity                                                         │
│  □ Unicode / special characters                                             │
│                                                                             │
│  Linked Lists:                                                              │
│  □ Empty list (head = None)                                                 │
│  □ Single node                                                              │
│  □ Cycle in list                                                            │
│                                                                             │
│  Trees:                                                                     │
│  □ Empty tree (root = None)                                                 │
│  □ Single node                                                              │
│  □ Skewed tree (all left or all right)                                      │
│  □ Negative values                                                          │
│                                                                             │
│  Graphs:                                                                    │
│  □ Disconnected components                                                  │
│  □ Cycles                                                                   │
│  □ Self-loops                                                               │
│  □ No edges                                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Mistakes to Avoid

```python
# ❌ WRONG: Modifying list while iterating
nums = [1, 2, 3, 4, 5]
for num in nums:
    if num % 2 == 0:
        nums.remove(num)  # Skips elements!

# ✅ CORRECT: Create new list or iterate backwards
nums = [num for num in nums if num % 2 != 0]
# OR
for i in range(len(nums) - 1, -1, -1):
    if nums[i] % 2 == 0:
        nums.pop(i)


# ❌ WRONG: Using mutable default argument
def append_to(element, lst=[]):  # lst is shared across calls!
    lst.append(element)
    return lst

# ✅ CORRECT: Use None as default
def append_to(element, lst=None):
    if lst is None:
        lst = []
    lst.append(element)
    return lst


# ❌ WRONG: Integer division truncation
result = 5 / 2    # 2.5 (float)
result = 5 // 2   # 2 (int, but -5 // 2 = -3, not -2!)

# ✅ CORRECT: Be explicit about what you want
result = int(5 / 2)  # 2 (truncates toward zero)


# ❌ WRONG: Shallow copy of nested structures
original = [[1, 2], [3, 4]]
copy = original[:]  # Shallow copy!
copy[0][0] = 99     # Modifies original too!

# ✅ CORRECT: Deep copy for nested structures
import copy
deep = copy.deepcopy(original)


# ❌ WRONG: Off-by-one errors in loops
# Finding last element
arr[-0]  # Same as arr[0], not last element!

# ✅ CORRECT
arr[-1]  # Last element
arr[len(arr) - 1]  # Also last element


# ❌ WRONG: Not handling empty input
def find_max(nums):
    return max(nums)  # Crashes on empty list!

# ✅ CORRECT: Handle edge cases
def find_max(nums):
    if not nums:
        return None  # or raise exception
    return max(nums)


# ❌ WRONG: String concatenation in loop (O(n²))
result = ""
for s in strings:
    result += s  # Creates new string each time!

# ✅ CORRECT: Use join (O(n))
result = "".join(strings)


# ❌ WRONG: Checking None with ==
if node == None:  # Works but not Pythonic
    pass

# ✅ CORRECT: Use 'is' for None
if node is None:
    pass
# OR
if not node:  # Also checks for empty/zero
    pass
```

---

## Interview Tips

### 1. Communication is Key
- **Think out loud**: Explain your thought process
- **Ask clarifying questions**: Input size? Edge cases? Constraints?
- **Discuss trade-offs**: Time vs space, readability vs performance

### 2. Start Simple
- **Brute force first**: Show you understand the problem
- **Then optimize**: Identify bottlenecks and improve
- **Don't over-engineer**: Simple solutions are often best

### 3. Test Your Code
- **Walk through examples**: Trace your code step by step
- **Edge cases**: Empty, single element, duplicates
- **Large inputs**: Will it timeout?

### 4. Know Your Complexities
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPLEXITY QUICK REFERENCE                                                 │
│                                                                             │
│  O(1)       - Hash lookup, array access, stack push/pop                     │
│  O(log n)   - Binary search, heap operations                                │
│  O(n)       - Linear scan, hash table build                                 │
│  O(n log n) - Efficient sorting (merge, quick, heap)                        │
│  O(n²)      - Nested loops, bubble/selection/insertion sort                 │
│  O(2^n)     - Subsets, recursive without memoization                        │
│  O(n!)      - Permutations                                                  │
│                                                                             │
│  For n = 10^6:                                                              │
│  O(n) ≈ 10^6 operations ✓                                                   │
│  O(n²) ≈ 10^12 operations ✗ (too slow!)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Next Steps

1. **Complete**: [Complexity Analysis](./02-complexity-analysis.md) - Master Big O notation
2. **Then**: Start with [Arrays & Hashing](../01-arrays-hashing/) - The foundation of all patterns
3. **Practice**: Solve at least 5 problems per pattern before moving on
