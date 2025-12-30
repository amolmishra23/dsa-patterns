# Recursion Fundamentals

## What is Recursion?

A function that calls itself to solve smaller subproblems until reaching a base case.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         RECURSION ANATOMY                                   │
│                                                                             │
│   def recursive_function(input):                                            │
│       # 1. BASE CASE - When to stop                                         │
│       if input is trivial:                                                  │
│           return trivial_answer                                             │
│                                                                             │
│       # 2. RECURSIVE CASE - Break down + combine                            │
│       smaller = make_smaller(input)                                         │
│       result = recursive_function(smaller)  # Trust this works!             │
│       return combine(result)                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Three Laws of Recursion

1. **Must have a base case** - A condition that stops the recursion
2. **Must move toward base case** - Each call should get closer to base case
3. **Must call itself** - The function calls itself with modified input

---

## Classic Examples

### Example 1: Factorial

```python
def factorial(n):
    # Base case
    if n <= 1:
        return 1
    # Recursive case
    return n * factorial(n - 1)

# Trace: factorial(5)
# factorial(5) = 5 * factorial(4)
#              = 5 * 4 * factorial(3)
#              = 5 * 4 * 3 * factorial(2)
#              = 5 * 4 * 3 * 2 * factorial(1)
#              = 5 * 4 * 3 * 2 * 1
#              = 120
```

### Example 2: Fibonacci

```python
# Naive O(2^n)
def fib_naive(n):
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)

# Memoized O(n)
def fib_memo(n, memo={}):
    if n <= 1:
        return n
    if n not in memo:
        memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]
```

### Example 3: Sum of Array

```python
def array_sum(arr):
    # Base case: empty array
    if not arr:
        return 0
    # Recursive case: first element + sum of rest
    return arr[0] + array_sum(arr[1:])

# Better: use indices to avoid array copying
def array_sum_optimized(arr, i=0):
    if i == len(arr):
        return 0
    return arr[i] + array_sum_optimized(arr, i + 1)
```

---

## Visualizing Recursion: The Call Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CALL STACK for factorial(4)                              │
│                                                                             │
│   CALL PHASE (going down)              RETURN PHASE (going up)              │
│                                                                             │
│   factorial(4)                         factorial(4) = 4 * 6 = 24 ◄──┐       │
│        │                                    ▲                        │       │
│        ▼                                    │                        │       │
│   factorial(3)                         factorial(3) = 3 * 2 = 6  ◄──┤       │
│        │                                    ▲                        │       │
│        ▼                                    │                        │       │
│   factorial(2)                         factorial(2) = 2 * 1 = 2  ◄──┤       │
│        │                                    ▲                        │       │
│        ▼                                    │                        │       │
│   factorial(1)  ─────► BASE CASE ─────► return 1 ────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Recursion vs Iteration

Most recursion can be converted to iteration and vice versa.

```python
# Recursive
def sum_recursive(n):
    if n <= 0:
        return 0
    return n + sum_recursive(n - 1)

# Iterative
def sum_iterative(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total
```

**When to use recursion:**
- Tree/graph traversals
- Divide and conquer algorithms
- Problems with natural recursive structure
- When code clarity matters more than performance

**When to use iteration:**
- Simple loops
- When stack overflow is a concern
- When performance is critical

---

## Tail Recursion

A recursive call is **tail recursive** if it's the last operation in the function.

```python
# NOT tail recursive - multiplication happens AFTER recursive call
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # Must wait for result, then multiply

# Tail recursive - recursive call is the LAST operation
def factorial_tail(n, accumulator=1):
    if n <= 1:
        return accumulator
    return factorial_tail(n - 1, n * accumulator)  # No work after call
```

Note: Python doesn't optimize tail recursion, but some languages do.

---

## Common Recursion Patterns

### Pattern 1: Linear Recursion
Process one element, recurse on rest.

```python
def linear_search(arr, target, i=0):
    if i == len(arr):
        return -1
    if arr[i] == target:
        return i
    return linear_search(arr, target, i + 1)
```

### Pattern 2: Binary Recursion (Divide and Conquer)
Split into two halves, combine results.

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### Pattern 3: Tree Recursion
Multiple recursive calls (like Fibonacci).

```python
def tree_sum(root):
    if not root:
        return 0
    return root.val + tree_sum(root.left) + tree_sum(root.right)
```

### Pattern 4: Backtracking
Try choices, undo if wrong.

```python
def permutations(arr, path=[], result=[]):
    if not arr:
        result.append(path[:])
        return
    for i in range(len(arr)):
        # Choose
        path.append(arr[i])
        # Explore
        permutations(arr[:i] + arr[i+1:], path, result)
        # Unchoose (backtrack)
        path.pop()
    return result
```

---

## Recursion Tree Visualization

For `fib(5)`:

```
                              fib(5)
                           /         \
                      fib(4)          fib(3)
                     /     \          /     \
                fib(3)    fib(2)   fib(2)  fib(1)
               /    \     /    \   /    \
           fib(2) fib(1) fib(1) fib(0) fib(1) fib(0)
           /    \
       fib(1) fib(0)

Notice: fib(3) computed twice, fib(2) computed three times!
This is why naive fibonacci is O(2^n)
With memoization: each fib(k) computed once → O(n)
```

---

## Common Mistakes

### 1. Missing Base Case
```python
# BAD - infinite recursion!
def countdown(n):
    print(n)
    countdown(n - 1)  # Never stops

# GOOD
def countdown(n):
    if n <= 0:  # Base case
        return
    print(n)
    countdown(n - 1)
```

### 2. Not Moving Toward Base Case
```python
# BAD - never reaches base case
def bad_sum(n):
    if n == 0:
        return 0
    return n + bad_sum(n)  # n never changes!

# GOOD
def good_sum(n):
    if n == 0:
        return 0
    return n + good_sum(n - 1)  # n decreases
```

### 3. Modifying Shared State Incorrectly
```python
# BAD - result is shared across all calls
def subsets_bad(arr, result=[]):  # Default mutable argument!
    result.append(arr[:])
    # ...

# GOOD
def subsets_good(arr, result=None):
    if result is None:
        result = []
    result.append(arr[:])
    # ...
```

---

## Recursion to Iteration Conversion

### Using a Stack

```python
# Recursive DFS
def dfs_recursive(root):
    if not root:
        return
    print(root.val)
    dfs_recursive(root.left)
    dfs_recursive(root.right)

# Iterative DFS using stack
def dfs_iterative(root):
    if not root:
        return
    stack = [root]
    while stack:
        node = stack.pop()
        print(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
```

---

## Practice Problems

1. **Power Function**: Implement `pow(x, n)` recursively
2. **Reverse String**: Reverse a string using recursion
3. **Count Digits**: Count digits in a number recursively
4. **Binary Search**: Implement recursive binary search
5. **Tower of Hanoi**: Classic recursion problem

```python
# Solutions

def power(x, n):
    if n == 0:
        return 1
    if n < 0:
        return 1 / power(x, -n)
    if n % 2 == 0:
        half = power(x, n // 2)
        return half * half
    return x * power(x, n - 1)

def reverse_string(s):
    if len(s) <= 1:
        return s
    return reverse_string(s[1:]) + s[0]

def count_digits(n):
    if n == 0:
        return 0
    return 1 + count_digits(n // 10)

def binary_search(arr, target, left, right):
    if left > right:
        return -1
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search(arr, target, mid + 1, right)
    else:
        return binary_search(arr, target, left, mid - 1)

def hanoi(n, source, auxiliary, target):
    if n == 1:
        print(f"Move disk 1 from {source} to {target}")
        return
    hanoi(n - 1, source, target, auxiliary)
    print(f"Move disk {n} from {source} to {target}")
    hanoi(n - 1, auxiliary, source, target)
```

