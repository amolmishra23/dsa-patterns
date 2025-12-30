# Stacks & Queues - Practice List

## Problems by Pattern

### Basic Stack
- LC 20: Valid Parentheses (Easy)
- LC 155: Min Stack (Medium)
- LC 232: Implement Queue using Stacks (Easy)
- LC 225: Implement Stack using Queues (Easy)
- LC 394: Decode String (Medium)
- LC 150: Evaluate Reverse Polish Notation (Medium)

### Monotonic Stack
- LC 739: Daily Temperatures (Medium)
- LC 496: Next Greater Element I (Easy)
- LC 503: Next Greater Element II (Medium)
- LC 84: Largest Rectangle in Histogram (Hard)
- LC 85: Maximal Rectangle (Hard)
- LC 42: Trapping Rain Water (Hard)
- LC 901: Online Stock Span (Medium)
- LC 402: Remove K Digits (Medium)
- LC 316: Remove Duplicate Letters (Medium)

### Queue Problems
- LC 622: Design Circular Queue (Medium)
- LC 641: Design Circular Deque (Medium)
- LC 346: Moving Average from Data Stream (Easy)
- LC 362: Design Hit Counter (Medium)
- LC 933: Number of Recent Calls (Easy)

### Monotonic Deque
- LC 239: Sliding Window Maximum (Hard)
- LC 862: Shortest Subarray with Sum at Least K (Hard)

## Templates

```python
from collections import deque

# Valid Parentheses
def isValid(s):
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
        else:
            stack.append(char)
    return len(stack) == 0

# Monotonic Stack (Next Greater Element)
def nextGreaterElement(nums):
    n = len(nums)
    result = [-1] * n
    stack = []  # Store indices
    
    for i in range(n):
        while stack and nums[i] > nums[stack[-1]]:
            result[stack.pop()] = nums[i]
        stack.append(i)
    
    return result

# Largest Rectangle in Histogram
def largestRectangleArea(heights):
    stack = []  # Store indices
    max_area = 0
    heights.append(0)  # Sentinel
    
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    
    return max_area

# Sliding Window Maximum (Monotonic Deque)
def maxSlidingWindow(nums, k):
    dq = deque()  # Store indices, decreasing order
    result = []
    
    for i, num in enumerate(nums):
        # Remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        
        # Remove smaller elements
        while dq and nums[dq[-1]] < num:
            dq.pop()
        
        dq.append(i)
        
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result

# Decode String
def decodeString(s):
    stack = []
    curr_str = ""
    curr_num = 0
    
    for char in s:
        if char.isdigit():
            curr_num = curr_num * 10 + int(char)
        elif char == '[':
            stack.append((curr_str, curr_num))
            curr_str = ""
            curr_num = 0
        elif char == ']':
            prev_str, num = stack.pop()
            curr_str = prev_str + curr_str * num
        else:
            curr_str += char
    
    return curr_str
```

## Key Insights
- Stack: LIFO, good for matching pairs, backtracking
- Queue: FIFO, good for BFS, sliding window
- Monotonic stack: next greater/smaller element
- Monotonic deque: sliding window min/max

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STACK & QUEUE PATTERNS                              │
│                                                                             │
│  VALID PARENTHESES:                                                         │
│  Input: "([{}])"                                                            │
│                                                                             │
│  Stack:  (   →  ([   →  ([{   →  ([   →  (   →  empty ✓                    │
│          push   push    push    pop}   pop]   pop)                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MONOTONIC STACK (Next Greater Element):                                    │
│  Input: [2, 1, 2, 4, 3]                                                     │
│                                                                             │
│  Process:                                                                   │
│  i=0: stack=[0]                     result=[-1,-1,-1,-1,-1]                 │
│  i=1: stack=[0,1]                   (1 < 2, push)                           │
│  i=2: pop 1, result[1]=2            result=[-1, 2,-1,-1,-1]                 │
│       stack=[0,2]                                                           │
│  i=3: pop 2, result[2]=4            result=[-1, 2, 4,-1,-1]                 │
│       pop 0, result[0]=4            result=[ 4, 2, 4,-1,-1]                 │
│       stack=[3]                                                             │
│  i=4: stack=[3,4]                   (3 < 4, push)                           │
│                                                                             │
│  Final: [4, 2, 4, -1, -1]                                                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LARGEST RECTANGLE IN HISTOGRAM:                                            │
│  heights = [2, 1, 5, 6, 2, 3]                                               │
│                                                                             │
│       6                                                                     │
│    5  █                                                                     │
│    █  █        3                                                            │
│    █  █  2     █                                                            │
│  2 █  █  █     █                                                            │
│  █ █  █  █  █  █                                                            │
│  0 1  2  3  4  5                                                            │
│                                                                             │
│  Stack tracks increasing heights.                                           │
│  When height decreases, calculate area for popped heights.                  │
│  Max area = 10 (heights[2:4] = [5,6], width=2, height=5)                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SLIDING WINDOW MAXIMUM (Monotonic Deque):                                  │
│  nums = [1,3,-1,-3,5,3,6,7], k=3                                            │
│                                                                             │
│  Window [1,3,-1]:  deque=[3,-1]     max=3                                   │
│  Window [3,-1,-3]: deque=[3,-1,-3]  max=3                                   │
│  Window [-1,-3,5]: deque=[5]        max=5  (5 > all, clear deque)           │
│  Window [-3,5,3]:  deque=[5,3]      max=5                                   │
│  Window [5,3,6]:   deque=[6]        max=6                                   │
│  Window [3,6,7]:   deque=[7]        max=7                                   │
│                                                                             │
│  Result: [3, 3, 5, 5, 6, 7]                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: Basic Stack
- [ ] LC 20: Valid Parentheses (Easy)
- [ ] LC 155: Min Stack (Medium)
- [ ] LC 150: Evaluate Reverse Polish Notation (Medium)
- [ ] LC 394: Decode String (Medium)
- [ ] LC 232: Implement Queue using Stacks (Easy)

### Week 2: Monotonic Stack
- [ ] LC 496: Next Greater Element I (Easy)
- [ ] LC 739: Daily Temperatures (Medium)
- [ ] LC 503: Next Greater Element II (Medium)
- [ ] LC 901: Online Stock Span (Medium)
- [ ] LC 402: Remove K Digits (Medium)

### Week 3: Advanced Stack & Deque
- [ ] LC 84: Largest Rectangle in Histogram (Hard)
- [ ] LC 85: Maximal Rectangle (Hard)
- [ ] LC 42: Trapping Rain Water (Hard)
- [ ] LC 239: Sliding Window Maximum (Hard)
- [ ] LC 316: Remove Duplicate Letters (Medium)

---

## Common Mistakes

### 1. Forgetting Empty Stack Check
```python
# WRONG - crashes on empty stack
top = stack[-1]  # IndexError!
stack.pop()

# CORRECT - check first
if stack:
    top = stack[-1]
    stack.pop()
```

### 2. Wrong Monotonic Stack Direction
```python
# For NEXT GREATER - use increasing stack (pop when current > top)
while stack and nums[i] > nums[stack[-1]]:
    result[stack.pop()] = nums[i]

# For NEXT SMALLER - use decreasing stack (pop when current < top)
while stack and nums[i] < nums[stack[-1]]:
    result[stack.pop()] = nums[i]
```

### 3. Off-by-One in Histogram Width
```python
# WRONG
width = i - stack[-1]  # Missing -1

# CORRECT
width = i - stack[-1] - 1  # Exclusive boundaries
# Or if stack empty:
width = i
```

### 4. Not Adding Sentinel for Histogram
```python
# WRONG - remaining stack elements not processed
for i, h in enumerate(heights):
    while stack and heights[stack[-1]] > h:
        ...
# Stack may still have elements!

# CORRECT - add sentinel to flush stack
heights.append(0)  # Forces all remaining to pop
for i, h in enumerate(heights):
    while stack and heights[stack[-1]] > h:
        ...
```

---

## Complexity Reference

| Pattern | Time | Space | Notes |
|---------|------|-------|-------|
| Valid Parentheses | O(n) | O(n) | Stack for opening |
| Min Stack | O(1) all ops | O(n) | Extra stack for mins |
| Next Greater | O(n) | O(n) | Each element pushed/popped once |
| Histogram | O(n) | O(n) | Monotonic increasing |
| Sliding Max | O(n) | O(k) | Deque of size k |

---

## Pattern Recognition

| See This | Think This |
|----------|------------|
| "Matching pairs" | Stack |
| "Next greater/smaller" | Monotonic stack |
| "Sliding window min/max" | Monotonic deque |
| "Expression evaluation" | Two stacks (values + operators) |
| "Decode/parse nested" | Stack for context |
| "Remove elements to optimize" | Monotonic stack |
| "Rectangle in histogram" | Monotonic increasing stack |
