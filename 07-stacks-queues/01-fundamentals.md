# Stacks & Queues - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE STACKS & QUEUES                              │
│                                                                             │
│  STACK (LIFO) - Use when:                                                   │
│  ✓ "Matching brackets/parentheses"                                          │
│  ✓ "Next greater/smaller element"                                           │
│  ✓ "Evaluate expression"                                                    │
│  ✓ "Undo operations"                                                        │
│  ✓ "DFS traversal"                                                          │
│  ✓ "Backtracking"                                                           │
│                                                                             │
│  QUEUE (FIFO) - Use when:                                                   │
│  ✓ "BFS traversal"                                                          │
│  ✓ "Level-order processing"                                                 │
│  ✓ "Sliding window maximum"                                                 │
│  ✓ "Task scheduling"                                                        │
│                                                                             │
│  MONOTONIC STACK - Use when:                                                │
│  ✓ "Next greater element"                                                   │
│  ✓ "Previous smaller element"                                               │
│  ✓ "Largest rectangle in histogram"                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] LIFO (Last In First Out) concept
- [ ] FIFO (First In First Out) concept
- [ ] Python list operations

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STACKS & QUEUES MEMORY MAP                               │
│                                                                             │
│            ┌───────────────────────────────────────────┐                    │
│            │          STACKS & QUEUES                  │                    │
│            └─────────────────┬─────────────────────────┘                    │
│                    ┌─────────┴─────────┐                                    │
│                    ▼                   ▼                                    │
│             ┌─────────────┐     ┌─────────────┐                             │
│             │   STACK     │     │   QUEUE     │                             │
│             │   (LIFO)    │     │   (FIFO)    │                             │
│             └──────┬──────┘     └──────┬──────┘                             │
│                    │                   │                                    │
│          ┌────────┼────────┐          │                                    │
│          ▼        ▼        ▼          ▼                                    │
│      ┌──────┐ ┌──────┐ ┌──────┐  ┌──────┐                                  │
│      │Match │ │Mono- │ │Expr  │  │ BFS  │                                  │
│      │Parens│ │tonic │ │Eval  │  │Deque │                                  │
│      └──────┘ └──────┘ └──────┘  └──────┘                                  │
│                                                                             │
│  Related Patterns:                                                          │
│  • DFS/BFS - Stack for DFS, Queue for BFS                                   │
│  • Sliding Window - Monotonic deque for max/min                             │
│  • Recursion - Call stack is implicit stack                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STACK vs QUEUE DECISION TREE                             │
│                                                                             │
│  Need to process in reverse order (LIFO)?                                   │
│       │                                                                     │
│       ├── YES → Use STACK                                                   │
│       │         Examples: Undo, matching brackets, DFS                      │
│       │                                                                     │
│       └── NO → Need to process in order received (FIFO)?                    │
│                    │                                                        │
│                    ├── YES → Use QUEUE (deque)                              │
│                    │         Examples: BFS, task scheduling                 │
│                    │                                                        │
│                    └── NO → Need next greater/smaller element?              │
│                                 │                                           │
│                                 ├── YES → MONOTONIC STACK                   │
│                                 │                                           │
│                                 └── NO → Need sliding window max/min?       │
│                                              │                              │
│                                              ├── YES → MONOTONIC DEQUE      │
│                                              │                              │
│                                              └── NO → Consider other DS     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### Stack (LIFO - Last In First Out)

```python
# Python list as stack
stack = []

# Push - O(1) amortized
stack.append(1)
stack.append(2)
stack.append(3)
# stack = [1, 2, 3]

# Pop - O(1)
top = stack.pop()  # Returns 3, stack = [1, 2]

# Peek - O(1)
top = stack[-1]  # Returns 2, stack unchanged

# Check empty - O(1)
is_empty = len(stack) == 0
```

### Queue (FIFO - First In First Out)

```python
from collections import deque

# Use deque for O(1) operations on both ends
queue = deque()

# Enqueue (add to back) - O(1)
queue.append(1)
queue.append(2)
queue.append(3)
# queue = [1, 2, 3]

# Dequeue (remove from front) - O(1)
front = queue.popleft()  # Returns 1, queue = [2, 3]

# Peek front - O(1)
front = queue[0]  # Returns 2

# Peek back - O(1)
back = queue[-1]  # Returns 3
```

---

## Visual: Stack vs Queue

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STACK (LIFO)                                             │
│                                                                             │
│    Push 1, 2, 3:              Pop:                                          │
│                                                                             │
│    ┌───┐                      ┌───┐                                         │
│    │ 3 │ ← top                │ 2 │ ← top (3 removed)                       │
│    ├───┤                      ├───┤                                         │
│    │ 2 │                      │ 1 │                                         │
│    ├───┤                      └───┘                                         │
│    │ 1 │                                                                    │
│    └───┘                                                                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                    QUEUE (FIFO)                                             │
│                                                                             │
│    Enqueue 1, 2, 3:           Dequeue:                                      │
│                                                                             │
│    front              back    front              back                       │
│      ↓                 ↓        ↓                 ↓                         │
│    ┌───┬───┬───┐            ┌───┬───┐                                       │
│    │ 1 │ 2 │ 3 │            │ 2 │ 3 │  (1 removed from front)               │
│    └───┴───┴───┘            └───┴───┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Monotonic Stack Pattern

A monotonic stack maintains elements in sorted order (increasing or decreasing).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MONOTONIC STACK                                          │
│                                                                             │
│  MONOTONIC DECREASING (for Next Greater Element):                           │
│  - Stack keeps indices of elements in decreasing order                      │
│  - When we find larger element, pop smaller ones (they found their NGE)     │
│                                                                             │
│  Example: Find Next Greater Element for [2, 1, 2, 4, 3]                     │
│                                                                             │
│  i=0: stack=[], push 0          stack=[0]                                   │
│  i=1: 1 < 2, push 1             stack=[0,1]                                 │
│  i=2: 2 > 1, pop 1 (NGE[1]=2)   stack=[0]                                   │
│       2 == 2, push 2            stack=[0,2]                                 │
│  i=3: 4 > 2, pop 2 (NGE[2]=4)   stack=[0]                                   │
│       4 > 2, pop 0 (NGE[0]=4)   stack=[]                                    │
│       push 3                    stack=[3]                                   │
│  i=4: 3 < 4, push 4             stack=[3,4]                                 │
│                                                                             │
│  Result: NGE = [4, 2, 4, -1, -1]                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Stack Problems

### Problem 1: Valid Parentheses (LC #20)

```python
def isValid(s: str) -> bool:
    """
    Check if parentheses are valid.

    Strategy:
    - Push opening brackets onto stack
    - For closing brackets, check if matches top of stack
    - At end, stack should be empty

    Time: O(n)
    Space: O(n)
    """
    # Map closing to opening brackets
    pairs = {')': '(', '}': '{', ']': '['}
    stack = []

    for char in s:
        if char in pairs:
            # Closing bracket - check for match
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
        else:
            # Opening bracket - push to stack
            stack.append(char)

    # Valid if all brackets matched (stack empty)
    return len(stack) == 0
```

### Problem 2: Daily Temperatures (LC #739)

```python
def dailyTemperatures(temperatures: list[int]) -> list[int]:
    """
    Find days until warmer temperature.

    Strategy (Monotonic Decreasing Stack):
    - Stack stores indices of temperatures we haven't found warmer day for
    - When we find warmer temp, pop all smaller ones

    Time: O(n) - each element pushed and popped at most once
    Space: O(n)
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Stack of indices

    for i in range(n):
        # Pop all temperatures smaller than current
        while stack and temperatures[i] > temperatures[stack[-1]]:
            prev_idx = stack.pop()
            result[prev_idx] = i - prev_idx  # Days until warmer

        # Push current index
        stack.append(i)

    # Remaining indices have no warmer day (result stays 0)
    return result
```

### Problem 3: Next Greater Element I (LC #496)

```python
def nextGreaterElement(nums1: list[int], nums2: list[int]) -> list[int]:
    """
    Find next greater element for each element in nums1 (subset of nums2).

    Strategy:
    1. Build NGE map for all elements in nums2 using monotonic stack
    2. Look up each element of nums1 in the map

    Time: O(n + m)
    Space: O(n)
    """
    # Build NGE map for nums2
    nge_map = {}
    stack = []

    for num in nums2:
        # Pop all smaller elements - current num is their NGE
        while stack and num > stack[-1]:
            smaller = stack.pop()
            nge_map[smaller] = num
        stack.append(num)

    # Remaining elements have no NGE
    for num in stack:
        nge_map[num] = -1

    # Look up each element in nums1
    return [nge_map[num] for num in nums1]
```

### Problem 4: Evaluate Reverse Polish Notation (LC #150)

```python
def evalRPN(tokens: list[str]) -> int:
    """
    Evaluate expression in Reverse Polish Notation.

    Strategy:
    - Numbers: push to stack
    - Operators: pop two operands, compute, push result

    Time: O(n)
    Space: O(n)
    """
    stack = []
    operators = {'+', '-', '*', '/'}

    for token in tokens:
        if token in operators:
            # Pop two operands (note: order matters for - and /)
            b = stack.pop()  # Second operand
            a = stack.pop()  # First operand

            if token == '+':
                result = a + b
            elif token == '-':
                result = a - b
            elif token == '*':
                result = a * b
            else:  # token == '/'
                # Integer division truncating toward zero
                result = int(a / b)

            stack.append(result)
        else:
            # Number - push to stack
            stack.append(int(token))

    return stack[0]
```

### Problem 5: Min Stack (LC #155)

```python
class MinStack:
    """
    Stack that supports push, pop, top, and getMin in O(1).

    Strategy:
    - Use auxiliary stack to track minimum at each level
    - When pushing, also push current minimum
    - When popping, also pop from min stack

    Time: O(1) for all operations
    Space: O(n)
    """

    def __init__(self):
        self.stack = []
        self.min_stack = []  # Parallel stack tracking minimums

    def push(self, val: int) -> None:
        self.stack.append(val)

        # Push current minimum to min_stack
        if not self.min_stack:
            self.min_stack.append(val)
        else:
            self.min_stack.append(min(val, self.min_stack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```

### Problem 6: Largest Rectangle in Histogram (LC #84)

```python
def largestRectangleArea(heights: list[int]) -> int:
    """
    Find largest rectangle in histogram.

    Strategy (Monotonic Increasing Stack):
    - For each bar, find how far left and right it can extend
    - Use stack to track bars in increasing height order
    - When we find smaller bar, calculate area for all taller bars

    Time: O(n)
    Space: O(n)
    """
    stack = []  # Stack of indices
    max_area = 0

    # Add sentinel 0 at end to flush remaining bars
    heights = heights + [0]

    for i, h in enumerate(heights):
        # Pop all taller bars
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]

            # Width: from current position to previous bar in stack
            if stack:
                width = i - stack[-1] - 1
            else:
                width = i  # Extends to the beginning

            max_area = max(max_area, height * width)

        stack.append(i)

    return max_area
```

---

## Queue Problems

### Problem 7: Implement Queue using Stacks (LC #232)

```python
class MyQueue:
    """
    Implement queue using two stacks.

    Strategy:
    - input_stack: for push operations
    - output_stack: for pop/peek operations
    - Transfer from input to output when output is empty

    Time: O(1) amortized for all operations
    Space: O(n)
    """

    def __init__(self):
        self.input_stack = []   # For pushing
        self.output_stack = []  # For popping

    def push(self, x: int) -> None:
        self.input_stack.append(x)

    def pop(self) -> int:
        self._transfer()
        return self.output_stack.pop()

    def peek(self) -> int:
        self._transfer()
        return self.output_stack[-1]

    def empty(self) -> bool:
        return not self.input_stack and not self.output_stack

    def _transfer(self) -> None:
        """Transfer from input to output if output is empty."""
        if not self.output_stack:
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())
```

### Problem 8: Sliding Window Maximum (LC #239)

```python
from collections import deque

def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    """
    Find maximum in each sliding window of size k.

    Strategy (Monotonic Decreasing Deque):
    - Deque stores indices in decreasing order of values
    - Front of deque is always the maximum
    - Remove indices outside current window
    - Remove smaller elements when adding new one

    Time: O(n) - each element added and removed at most once
    Space: O(k)
    """
    result = []
    dq = deque()  # Stores indices

    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements (they can't be maximum)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        # Add current index
        dq.append(i)

        # Add maximum to result (after first k elements)
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

---

## Complexity Summary

| Operation | Stack | Queue | Deque |
|-----------|-------|-------|-------|
| Push/Append | O(1) | O(1) | O(1) |
| Pop | O(1) | O(1) | O(1) |
| Peek | O(1) | O(1) | O(1) |
| Search | O(n) | O(n) | O(n) |

---

## Common Mistakes

```python
# ❌ WRONG: Using list.pop(0) for queue - O(n)!
queue = [1, 2, 3]
front = queue.pop(0)  # O(n) operation!

# ✅ CORRECT: Use deque for O(1) operations
from collections import deque
queue = deque([1, 2, 3])
front = queue.popleft()  # O(1)


# ❌ WRONG: Checking empty stack incorrectly
if stack:  # This checks if stack exists, not if empty
    stack.pop()  # Might fail if stack is []

# ✅ CORRECT: Check length
if len(stack) > 0:
    stack.pop()
# Or simply:
if stack:  # Actually this works too, empty list is falsy
    stack.pop()
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"I'll use a monotonic stack to find the next greater element. As I iterate,
I pop elements smaller than current (they found their answer), then push
current. This gives O(n) because each element is pushed and popped once."
```

### 2. What Interviewers Look For
- **Data structure choice**: Know when stack vs queue vs deque
- **Monotonic stack mastery**: Understand increasing vs decreasing
- **Edge cases**: Empty stack, all same elements

### 3. Common Follow-up Questions
- "Can you do it without extra space?" → Usually no for these problems
- "What about circular array?" → Process array twice or use modulo
- "Time complexity of monotonic stack?" → O(n), each element pushed/popped once

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
