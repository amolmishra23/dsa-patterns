# Complexity Analysis - Big O Notation

## Why Complexity Matters

When solving problems, we need to answer:
1. **How fast is my solution?** → Time Complexity
2. **How much memory does it use?** → Space Complexity

Big O notation describes how runtime/space grows as input size (n) increases.

---

## Time Complexity Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TIME COMPLEXITY COMPARISON                               │
│                                                                             │
│    BEST ◄─────────────────────────────────────────────────────────► WORST   │
│                                                                             │
│    O(1)     O(log n)    O(n)    O(n log n)    O(n²)    O(2ⁿ)    O(n!)      │
│      │          │         │          │          │         │        │        │
│      │          │         │          │          │         │        │        │
│   constant  logarithmic linear  linearithmic quadratic expon.  factorial   │
│                                                                             │
│   n=1000:                                                                   │
│    1         10       1,000     10,000    1,000,000   huge!    huge!       │
│                                                                             │
│   Examples:                                                                 │
│   O(1)      - Array access, hash lookup                                     │
│   O(log n)  - Binary search                                                 │
│   O(n)      - Linear scan                                                   │
│   O(n log n)- Merge sort, heap sort                                         │
│   O(n²)     - Nested loops, bubble sort                                     │
│   O(2ⁿ)     - Subsets, recursive fibonacci                                  │
│   O(n!)     - Permutations                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Visual: Growth Rates

```
Operations
    │
    │                                                    O(n!)
    │                                                 ╱
    │                                              ╱
    │                                           ╱
    │                                        ╱    O(2ⁿ)
    │                                     ╱    ╱
    │                                  ╱    ╱
    │                               ╱    ╱
    │                            ╱    ╱
    │                         ╱    ╱        O(n²)
    │                      ╱    ╱       ╱
    │                   ╱    ╱      ╱
    │                ╱    ╱     ╱
    │             ╱    ╱    ╱           O(n log n)
    │          ╱    ╱   ╱          ╱
    │       ╱    ╱  ╱         ╱
    │    ╱    ╱ ╱        ╱             O(n)
    │ ╱    ╱╱       ╱            ╱
    │   ╱╱     ╱           ╱
    │╱╱   ╱          ╱                  O(log n)
    │ ╱       ╱                    ─────────────── O(1)
    └──────────────────────────────────────────────────► n (input size)
```

---

## Common Time Complexities Explained

### O(1) - Constant Time
Operations that don't depend on input size.

```python
def get_first(arr):
    return arr[0]  # O(1)

def hash_lookup(d, key):
    return d.get(key)  # O(1) average

def push_stack(stack, val):
    stack.append(val)  # O(1) amortized
```

### O(log n) - Logarithmic Time
Halving the search space each step.

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:          # O(log n) iterations
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

**Why log n?** If n = 1,000,000, we only need ~20 comparisons (log₂(1,000,000) ≈ 20)

### O(n) - Linear Time
Touch each element once.

```python
def find_max(arr):
    max_val = arr[0]
    for num in arr:      # O(n)
        max_val = max(max_val, num)
    return max_val

def two_sum_hashmap(arr, target):
    seen = {}
    for num in arr:      # O(n)
        if target - num in seen:  # O(1)
            return True
        seen[num] = True
    return False
```

### O(n log n) - Linearithmic Time
Typical for efficient sorting algorithms.

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # T(n/2)
    right = merge_sort(arr[mid:])   # T(n/2)
    return merge(left, right)       # O(n)
    # Total: O(n log n)

# Python's built-in sort
arr.sort()  # O(n log n) - Timsort
sorted(arr) # O(n log n)
```

### O(n²) - Quadratic Time
Nested loops over input.

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):           # O(n)
        for j in range(n - 1):   # O(n)
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    # Total: O(n²)

def two_sum_brute(arr, target):
    for i in range(len(arr)):        # O(n)
        for j in range(i+1, len(arr)): # O(n)
            if arr[i] + arr[j] == target:
                return [i, j]
    # Total: O(n²)
```

### O(2ⁿ) - Exponential Time
Doubling with each additional element (subsets, naive recursion).

```python
def fibonacci_naive(n):
    if n <= 1:
        return n
    return fibonacci_naive(n-1) + fibonacci_naive(n-2)
    # O(2ⁿ) - each call branches into 2

def all_subsets(arr):
    result = [[]]
    for num in arr:
        result += [subset + [num] for subset in result]
    return result
    # O(2ⁿ) subsets to generate
```

### O(n!) - Factorial Time
All permutations.

```python
def permutations(arr):
    if len(arr) <= 1:
        return [arr]
    result = []
    for i, num in enumerate(arr):
        rest = arr[:i] + arr[i+1:]
        for perm in permutations(rest):
            result.append([num] + perm)
    return result
    # O(n!) permutations
```

---

## Space Complexity

### What Counts as Space?
- Variables
- Data structures (arrays, hash maps, etc.)
- Recursion call stack

### Common Space Complexities

```python
# O(1) - Constant space
def swap(a, b):
    temp = a  # Just one variable
    a = b
    b = temp

# O(n) - Linear space
def copy_array(arr):
    return arr[:]  # New array of size n

# O(n) - Recursion stack
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
    # Call stack depth: n

# O(log n) - Balanced recursion
def binary_search_recursive(arr, target, left, right):
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
    # Call stack depth: log n
```

---

## Analyzing Complexity - Step by Step

### Rule 1: Drop Constants
O(2n) → O(n)
O(n/2) → O(n)
O(100) → O(1)

### Rule 2: Drop Lower Order Terms
O(n² + n) → O(n²)
O(n + log n) → O(n)
O(n³ + n² + n) → O(n³)

### Rule 3: Different Variables
```python
def print_pairs(arr1, arr2):  # arr1 has n elements, arr2 has m elements
    for a in arr1:    # O(n)
        for b in arr2:  # O(m)
            print(a, b)
# Total: O(n * m), NOT O(n²)
```

### Rule 4: Sequential = Add, Nested = Multiply
```python
def example(arr):
    # Sequential operations: ADD
    for x in arr:      # O(n)
        print(x)
    for x in arr:      # O(n)
        print(x)
    # Total: O(n) + O(n) = O(2n) = O(n)

    # Nested operations: MULTIPLY
    for x in arr:          # O(n)
        for y in arr:      # O(n)
            print(x, y)
    # Total: O(n) * O(n) = O(n²)
```

---

## Amortized Analysis

Some operations are expensive occasionally but cheap on average.

### Example: Dynamic Array (Python list)
```python
arr = []
for i in range(n):
    arr.append(i)  # Usually O(1), occasionally O(n) when resizing
# Amortized: O(1) per append
# Total: O(n)
```

**Why?** Array doubles when full. Cost of n insertions:
- 1 + 1 + 1 + ... + 1 (n times) = n
- Plus copying: 1 + 2 + 4 + 8 + ... + n/2 ≈ n
- Total: ~2n = O(n)
- Per operation: O(n)/n = O(1) amortized

---

## Complexity Cheat Sheet

| Data Structure | Access | Search | Insert | Delete |
|---------------|--------|--------|--------|--------|
| Array | O(1) | O(n) | O(n) | O(n) |
| Linked List | O(n) | O(n) | O(1)* | O(1)* |
| Hash Table | N/A | O(1)† | O(1)† | O(1)† |
| BST (balanced) | O(log n) | O(log n) | O(log n) | O(log n) |
| Heap | O(1)‡ | O(n) | O(log n) | O(log n) |

*With reference to node, †Average case, ‡Min/Max only

| Algorithm | Time | Space |
|-----------|------|-------|
| Binary Search | O(log n) | O(1) |
| Merge Sort | O(n log n) | O(n) |
| Quick Sort | O(n log n)† | O(log n) |
| Heap Sort | O(n log n) | O(1) |
| BFS/DFS | O(V + E) | O(V) |
| Dijkstra | O((V+E) log V) | O(V) |

†Average case

---

## Practice: Analyze These

```python
# Problem 1: What's the time complexity?
def mystery1(n):
    i = 1
    while i < n:
        i *= 2
    # Answer: O(log n) - i doubles each iteration

# Problem 2: What's the time complexity?
def mystery2(arr):
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            print(arr[i], arr[j])
    # Answer: O(n²) - n + (n-1) + (n-2) + ... + 1 = n(n+1)/2

# Problem 3: What's the time and space complexity?
def mystery3(n):
    if n <= 0:
        return 0
    return mystery3(n-1) + mystery3(n-1)
    # Answer: Time O(2ⁿ), Space O(n) - call stack depth

# Problem 4: What's the time complexity?
def mystery4(arr):
    arr.sort()  # O(n log n)
    for x in arr:  # O(n)
        binary_search(arr, x)  # O(log n)
    # Answer: O(n log n) - sorting dominates
```

---

## Interview Tips

1. **Always state complexity** after solving a problem
2. **Know the complexity** of built-in operations
3. **Optimize** by reducing time complexity first, then space
4. **Trade-offs**: Often you can trade space for time (hash maps)
5. **Constraints hint at complexity**:
   - n ≤ 10: O(n!) ok
   - n ≤ 20: O(2ⁿ) ok
   - n ≤ 500: O(n³) ok
   - n ≤ 10⁴: O(n²) ok
   - n ≤ 10⁶: O(n log n) needed
   - n ≤ 10⁸: O(n) needed
   - n > 10⁸: O(log n) or O(1) needed

