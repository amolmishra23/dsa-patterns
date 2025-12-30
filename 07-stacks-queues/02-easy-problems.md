# Stacks & Queues - Easy Problems

## Problem 1: Valid Parentheses (LC #20) - Easy

- [LeetCode](https://leetcode.com/problems/valid-parentheses/)

### Problem Statement
Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid. A string is valid if brackets are closed in the correct order.

### Examples
```
Input: s = "()"
Output: true

Input: s = "()[]{}"
Output: true

Input: s = "(]"
Output: false

Input: s = "([)]"
Output: false

Input: s = "{[]}"
Output: true
```

### Video Explanation
- [NeetCode - Valid Parentheses](https://www.youtube.com/watch?v=WTzjTskDFMg)
- [Take U Forward - Balanced Parentheses](https://www.youtube.com/watch?v=wkDfsKijrZ8)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MATCHING BRACKETS WITH A STACK                                             │
│                                                                             │
│  Input: "{[()]}"                                                           │
│                                                                             │
│  Process each character:                                                    │
│                                                                             │
│  char '{': Opening → Push to stack                                         │
│  Stack: [{]                                                                 │
│                                                                             │
│  char '[': Opening → Push to stack                                         │
│  Stack: [{, []                                                             │
│                                                                             │
│  char '(': Opening → Push to stack                                         │
│  Stack: [{, [, (]                                                          │
│                                                                             │
│  char ')': Closing → Check top: '(' matches! Pop it                        │
│  Stack: [{, []                                                             │
│                                                                             │
│  char ']': Closing → Check top: '[' matches! Pop it                        │
│  Stack: [{]                                                                 │
│                                                                             │
│  char '}': Closing → Check top: '{' matches! Pop it                        │
│  Stack: []                                                                  │
│                                                                             │
│  End: Stack is empty → VALID!                                              │
│                                                                             │
│  Key insight: Most recent opening bracket must match first closing bracket │
│  This is LIFO behavior → perfect for a STACK!                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def isValid(s: str) -> bool:
    """
    Check if parentheses are valid using a stack.

    Strategy:
    - Push opening brackets onto stack
    - For closing brackets, check if it matches the top of stack
    - At end, stack should be empty

    Time: O(n) - single pass through string
    Space: O(n) - stack can hold up to n/2 opening brackets
    """
    # Map each closing bracket to its corresponding opening bracket
    bracket_map = {
        ')': '(',
        ']': '[',
        '}': '{'
    }

    stack = []

    for char in s:
        if char in bracket_map:
            # It's a closing bracket
            # Check if stack is empty or top doesn't match
            if not stack or stack[-1] != bracket_map[char]:
                return False
            stack.pop()  # Remove matching opening bracket
        else:
            # It's an opening bracket - push to stack
            stack.append(char)

    # Valid only if all brackets matched (stack empty)
    return len(stack) == 0
```

### Complexity
- **Time**: O(n) - single pass through string
- **Space**: O(n) - stack can hold up to n/2 brackets

### Edge Cases
- Empty string: Return `True`
- Single bracket: `"("` → `False`
- Nested brackets: `"{[()]}"` → `True`
- Wrong order: `"([)]"` → `False`

### Common Mistakes
- Forgetting to check if stack is empty before popping
- Using wrong mapping direction (closing → opening, not opening → closing)
- Not checking if stack is empty at the end

### Related Problems
- LC #22 Generate Parentheses
- LC #32 Longest Valid Parentheses
- LC #1249 Minimum Remove to Make Valid Parentheses

---

## Problem 2: Min Stack (LC #155) - Medium

- [LeetCode](https://leetcode.com/problems/min-stack/)

### Problem Statement
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

### Examples
```
Input:
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output: [null,null,null,null,-3,null,0,-2]

Explanation:
MinStack minStack = new MinStack();
minStack.push(-2);   // stack: [-2]
minStack.push(0);    // stack: [-2, 0]
minStack.push(-3);   // stack: [-2, 0, -3]
minStack.getMin();   // return -3
minStack.pop();      // stack: [-2, 0]
minStack.top();      // return 0
minStack.getMin();   // return -2
```

### Video Explanation
- [NeetCode - Min Stack](https://www.youtube.com/watch?v=qkLl7nAwDPo)
- [Take U Forward - Min Stack](https://www.youtube.com/watch?v=V09NfaGf2ao)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRACKING MINIMUM AT EACH STATE                                             │
│                                                                             │
│  Challenge: When we pop, how do we know the new minimum?                   │
│                                                                             │
│  Solution: Store minimum WITH each element!                                 │
│                                                                             │
│  Push -2:  Stack: [(-2, -2)]        min at this point = -2                 │
│  Push 0:   Stack: [(-2, -2), (0, -2)]    min still -2                      │
│  Push -3:  Stack: [(-2, -2), (0, -2), (-3, -3)]   new min = -3             │
│                                                                             │
│  Each entry stores (value, minimum_so_far)                                 │
│                                                                             │
│  getMin(): Just return top's minimum → O(1)!                               │
│  pop(): Remove top, previous top has its own minimum → O(1)!               │
│                                                                             │
│  After pop(-3):                                                             │
│  Stack: [(-2, -2), (0, -2)]                                                │
│  getMin() = -2 (from top's stored minimum)                                 │
│                                                                             │
│  Alternative: Use two stacks (one for values, one for mins)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
class MinStack:
    """
    Stack with O(1) minimum retrieval.

    Strategy: Store (value, current_minimum) pairs
    Each element knows the minimum at the time it was pushed.

    Time: O(1) for all operations
    Space: O(n)
    """

    def __init__(self):
        self.stack = []  # Stores (value, min_so_far) tuples

    def push(self, val: int) -> None:
        """Push element onto stack."""
        if not self.stack:
            # First element - it's the minimum
            self.stack.append((val, val))
        else:
            # New minimum is min of current value and previous minimum
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))

    def pop(self) -> None:
        """Remove top element."""
        self.stack.pop()

    def top(self) -> int:
        """Return top element."""
        return self.stack[-1][0]

    def getMin(self) -> int:
        """Return minimum element in O(1)."""
        return self.stack[-1][1]


class MinStack_TwoStacks:
    """Alternative: Use two separate stacks."""

    def __init__(self):
        self.stack = []
        self.min_stack = []  # Tracks minimums

    def push(self, val: int) -> None:
        self.stack.append(val)
        # Push to min_stack if it's a new minimum
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        val = self.stack.pop()
        # Pop from min_stack if we're removing the current minimum
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```

### Complexity
- **Time**: O(1) for all operations
- **Space**: O(n) for storing elements

### Edge Cases
- Single element: That element is the minimum
- Decreasing sequence: Each push updates minimum
- Pop minimum: Previous minimum is restored
- Duplicate minimums: Handled correctly by storing with each element

### Common Mistakes
- Forgetting to update minimum when pushing
- Not handling empty stack edge case
- Using O(n) scan to find minimum (defeats the purpose)

### Related Problems
- LC #716 Max Stack
- LC #895 Maximum Frequency Stack
- LC #1381 Design a Stack With Increment Operation

---

## Problem 3: Implement Queue using Stacks (LC #232) - Easy

- [LeetCode](https://leetcode.com/problems/implement-queue-using-stacks/)

### Problem Statement
Implement a first-in-first-out (FIFO) queue using only two stacks. The queue should support push, pop, peek, and empty operations.

### Examples
```
Input:
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]

Output: [null, null, null, 1, 1, false]

Explanation:
MyQueue myQueue = new MyQueue();
myQueue.push(1);  // queue: [1]
myQueue.push(2);  // queue: [1, 2]
myQueue.peek();   // return 1
myQueue.pop();    // return 1, queue: [2]
myQueue.empty();  // return false
```

### Video Explanation
- [NeetCode - Queue using Stacks](https://www.youtube.com/watch?v=3Et9MrMc02A)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CONVERTING LIFO TO FIFO                                                    │
│                                                                             │
│  Stack is LIFO (Last In First Out)                                         │
│  Queue is FIFO (First In First Out)                                        │
│                                                                             │
│  Trick: Use TWO stacks to reverse order!                                   │
│                                                                             │
│  input_stack: receives new elements                                        │
│  output_stack: provides elements for removal                               │
│                                                                             │
│  push(1), push(2), push(3):                                                │
│  input_stack: [1, 2, 3] (3 on top)                                         │
│  output_stack: []                                                           │
│                                                                             │
│  pop() - need to remove 1 (first in):                                      │
│  Transfer input → output (reverses order!)                                 │
│  input_stack: []                                                            │
│  output_stack: [3, 2, 1] (1 on top now!)                                   │
│                                                                             │
│  Now pop from output_stack → returns 1 ✓                                   │
│  output_stack: [3, 2]                                                       │
│                                                                             │
│  Key insight: Only transfer when output is empty                           │
│  This gives O(1) amortized time!                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
class MyQueue:
    """
    Queue implemented with two stacks.

    Strategy:
    - input_stack: for push operations
    - output_stack: for pop/peek operations
    - Transfer from input to output when output is empty

    Time: O(1) amortized for all operations
    Space: O(n)
    """

    def __init__(self):
        self.input_stack = []   # New elements go here
        self.output_stack = []  # Elements for removal come from here

    def push(self, x: int) -> None:
        """Push element to back of queue. O(1)"""
        self.input_stack.append(x)

    def pop(self) -> int:
        """Remove and return front element. O(1) amortized"""
        self._transfer_if_needed()
        return self.output_stack.pop()

    def peek(self) -> int:
        """Return front element without removing. O(1) amortized"""
        self._transfer_if_needed()
        return self.output_stack[-1]

    def empty(self) -> bool:
        """Check if queue is empty. O(1)"""
        return not self.input_stack and not self.output_stack

    def _transfer_if_needed(self) -> None:
        """Transfer elements from input to output if output is empty."""
        if not self.output_stack:
            # Reverse order by popping from input and pushing to output
            while self.input_stack:
                self.output_stack.append(self.input_stack.pop())
```

### Complexity
- **Time**: O(1) amortized - each element transferred at most once
- **Space**: O(n) for storing elements

### Edge Cases
- Empty queue: Check both stacks for empty()
- Single element: Works correctly
- Alternating push/pop: Efficient due to lazy transfer
- Many pushes then many pops: Single transfer batch

### Common Mistakes
- Transferring elements on every pop (should only when output empty)
- Forgetting to check both stacks for empty()
- Not understanding amortized analysis

### Related Problems
- LC #225 Implement Stack using Queues
- LC #622 Design Circular Queue
- LC #641 Design Circular Deque

---

## Problem 4: Implement Stack using Queues (LC #225) - Easy

- [LeetCode](https://leetcode.com/problems/implement-stack-using-queues/)

### Problem Statement
Implement a last-in-first-out (LIFO) stack using only two queues. The stack should support push, pop, top, and empty operations.

### Examples
```
Input:
["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]

Output: [null, null, null, 2, 2, false]

Explanation:
MyStack myStack = new MyStack();
myStack.push(1);  // stack: [1]
myStack.push(2);  // stack: [1, 2]
myStack.top();    // return 2
myStack.pop();    // return 2, stack: [1]
myStack.empty();  // return false
```

### Video Explanation
- [NeetCode - Stack using Queues](https://www.youtube.com/watch?v=rW4vm0-DLYc)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CONVERTING FIFO TO LIFO                                                    │
│                                                                             │
│  Queue is FIFO (First In First Out)                                        │
│  Stack is LIFO (Last In First Out)                                         │
│                                                                             │
│  Trick: After each push, rotate queue so new element is at front!          │
│                                                                             │
│  push(1):                                                                   │
│  Queue: [1]                                                                 │
│  No rotation needed (only 1 element)                                       │
│                                                                             │
│  push(2):                                                                   │
│  Queue: [1, 2]                                                             │
│  Rotate: move 1 to back                                                    │
│  Queue: [2, 1]  (2 is now at front!)                                       │
│                                                                             │
│  push(3):                                                                   │
│  Queue: [2, 1, 3]                                                          │
│  Rotate: move 2 to back, move 1 to back                                    │
│  Queue: [3, 2, 1]  (3 is now at front!)                                    │
│                                                                             │
│  pop(): Just dequeue front → returns 3 ✓                                   │
│  top(): Just peek front → returns 3 ✓                                      │
│                                                                             │
│  This makes push O(n) but pop O(1)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
from collections import deque

class MyStack:
    """
    Stack implemented with single queue.

    Strategy:
    - After each push, rotate queue so new element is at front
    - This makes pop() O(1) but push() O(n)

    Time: O(n) push, O(1) pop/top
    Space: O(n)
    """

    def __init__(self):
        self.queue = deque()

    def push(self, x: int) -> None:
        """Push element onto stack. O(n)"""
        self.queue.append(x)

        # Rotate queue: move all elements before x to after x
        for _ in range(len(self.queue) - 1):
            self.queue.append(self.queue.popleft())

    def pop(self) -> int:
        """Remove and return top element. O(1)"""
        return self.queue.popleft()

    def top(self) -> int:
        """Return top element without removing. O(1)"""
        return self.queue[0]

    def empty(self) -> bool:
        """Check if stack is empty. O(1)"""
        return len(self.queue) == 0
```

### Complexity
- **Time**: O(n) for push, O(1) for pop/top/empty
- **Space**: O(n) for storing elements

### Edge Cases
- Single element: No rotation needed
- Two elements: Single rotation on push
- Empty stack: Handle in empty() check
- Many pushes: Each push is O(n) rotation

### Common Mistakes
- Rotating wrong number of times
- Using two queues when one suffices
- Making pop O(n) instead of push (both work, but less common)

### Related Problems
- LC #232 Implement Queue using Stacks
- LC #622 Design Circular Queue
- LC #155 Min Stack

---

## Problem 5: Baseball Game (LC #682) - Easy

- [LeetCode](https://leetcode.com/problems/baseball-game/)

### Problem Statement
Calculate the sum of scores in a baseball game with special operations:
- Integer: Record a new score
- "+": Record sum of previous two scores
- "D": Record double of previous score
- "C": Invalidate and remove previous score

### Examples
```
Input: ops = ["5","2","C","D","+"]
Output: 30

Explanation:
"5" - Record 5, stack: [5]
"2" - Record 2, stack: [5, 2]
"C" - Remove 2, stack: [5]
"D" - Record 5*2=10, stack: [5, 10]
"+" - Record 5+10=15, stack: [5, 10, 15]
Sum = 5 + 10 + 15 = 30
```

### Video Explanation
- [NeetCode - Baseball Game](https://www.youtube.com/watch?v=Id_tqGdsZQI)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SIMULATING A GAME WITH STACK                                               │
│                                                                             │
│  ops = ["5", "2", "C", "D", "+"]                                           │
│                                                                             │
│  "5": Push 5                                                                │
│  Stack: [5]                                                                 │
│                                                                             │
│  "2": Push 2                                                                │
│  Stack: [5, 2]                                                             │
│                                                                             │
│  "C": Cancel last score (pop)                                              │
│  Stack: [5]                                                                 │
│                                                                             │
│  "D": Double last score                                                     │
│  Stack: [5, 10]  (5 * 2 = 10)                                              │
│                                                                             │
│  "+": Sum of last two scores                                               │
│  Stack: [5, 10, 15]  (5 + 10 = 15)                                         │
│                                                                             │
│  Final sum: 5 + 10 + 15 = 30                                               │
│                                                                             │
│  Stack is perfect because we always operate on RECENT scores               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def calPoints(ops: list[str]) -> int:
    """
    Calculate baseball game score using stack.

    Time: O(n) - process each operation once
    Space: O(n) - stack stores all valid scores
    """
    stack = []

    for op in ops:
        if op == "C":
            # Cancel (remove) last score
            stack.pop()
        elif op == "D":
            # Double of last score
            stack.append(stack[-1] * 2)
        elif op == "+":
            # Sum of last two scores
            stack.append(stack[-1] + stack[-2])
        else:
            # It's a number - record the score
            stack.append(int(op))

    return sum(stack)
```

### Complexity
- **Time**: O(n) - single pass through operations
- **Space**: O(n) - stack stores scores

### Edge Cases
- Single score: Just that number
- All "C" operations: Could empty the stack
- Negative numbers: Valid scores, handled correctly
- "D" after "C": Uses the score before cancelled one

### Common Mistakes
- Forgetting to convert string to int for number operations
- Wrong order for "+" operation (stack[-1] + stack[-2])
- Not handling negative numbers (they're valid scores)

### Related Problems
- LC #20 Valid Parentheses
- LC #150 Evaluate Reverse Polish Notation
- LC #71 Simplify Path

---

## Problem 6: Next Greater Element I (LC #496) - Easy

- [LeetCode](https://leetcode.com/problems/next-greater-element-i/)

### Problem Statement
Find the next greater element for each element in `nums1` (which is a subset of `nums2`). The next greater element of `x` in `nums2` is the first element to the right of `x` that is greater than `x`.

### Examples
```
Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
Output: [-1,3,-1]

Explanation:
- 4: In nums2 = [1,3,4,2], nothing to right of 4 is greater → -1
- 1: In nums2 = [1,3,4,2], next greater after 1 is 3 → 3
- 2: In nums2 = [1,3,4,2], nothing to right of 2 is greater → -1
```

### Video Explanation
- [NeetCode - Next Greater Element I](https://www.youtube.com/watch?v=68a1Dc_qVq4)
- [Take U Forward - Next Greater Element](https://www.youtube.com/watch?v=Du881K7Zew8)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MONOTONIC STACK FOR NEXT GREATER                                           │
│                                                                             │
│  nums2 = [1, 3, 4, 2]                                                      │
│                                                                             │
│  Process from LEFT to RIGHT, maintain DECREASING stack:                    │
│                                                                             │
│  i=0, num=1: Stack empty, push 1                                           │
│  Stack: [1]                                                                 │
│                                                                             │
│  i=1, num=3: 3 > stack top (1)                                             │
│              Pop 1, NGE[1] = 3                                              │
│              Push 3                                                         │
│  Stack: [3], NGE: {1: 3}                                                   │
│                                                                             │
│  i=2, num=4: 4 > stack top (3)                                             │
│              Pop 3, NGE[3] = 4                                              │
│              Push 4                                                         │
│  Stack: [4], NGE: {1: 3, 3: 4}                                             │
│                                                                             │
│  i=3, num=2: 2 < stack top (4)                                             │
│              Just push 2                                                    │
│  Stack: [4, 2], NGE: {1: 3, 3: 4}                                          │
│                                                                             │
│  End: Remaining in stack have no NGE → -1                                  │
│  NGE: {1: 3, 3: 4, 4: -1, 2: -1}                                           │
│                                                                             │
│  Look up nums1 = [4, 1, 2] → [-1, 3, -1]                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def nextGreaterElement(nums1: list[int], nums2: list[int]) -> list[int]:
    """
    Find next greater element using monotonic stack.

    Strategy:
    1. Build a map of next greater element for all elements in nums2
    2. Look up each element of nums1 in the map

    Time: O(n + m) where n = len(nums1), m = len(nums2)
    Space: O(m) for the map and stack
    """
    # Build next greater element map for nums2
    nge_map = {}  # element -> its next greater element
    stack = []    # Monotonic decreasing stack

    for num in nums2:
        # Pop all smaller elements - current num is their NGE
        while stack and stack[-1] < num:
            smaller = stack.pop()
            nge_map[smaller] = num

        stack.append(num)

    # Elements remaining in stack have no NGE
    for num in stack:
        nge_map[num] = -1

    # Look up each element in nums1
    return [nge_map[num] for num in nums1]
```

### Complexity
- **Time**: O(n + m) - each element pushed/popped once
- **Space**: O(m) - map and stack

### Edge Cases
- nums1 has single element: Look up in map
- All elements have no NGE: All return -1
- Strictly increasing nums2: Each has NGE (next element)
- Strictly decreasing nums2: None have NGE (all -1)

### Common Mistakes
- Processing in wrong direction
- Not understanding monotonic stack invariant
- Forgetting to handle elements with no NGE

### Related Problems
- LC #503 Next Greater Element II (circular)
- LC #739 Daily Temperatures
- LC #84 Largest Rectangle in Histogram

---

## Problem 7: Remove All Adjacent Duplicates In String (LC #1047) - Easy

- [LeetCode](https://leetcode.com/problems/remove-all-adjacent-duplicates-in-string/)

### Problem Statement
Remove all adjacent duplicate characters from a string, repeatedly, until no adjacent duplicates remain.

### Examples
```
Input: s = "abbaca"
Output: "ca"

Explanation:
"abbaca" → "aaca" (remove "bb")
"aaca" → "ca" (remove "aa")

Input: s = "azxxzy"
Output: "ay"

Explanation:
"azxxzy" → "azzy" (remove "xx")
"azzy" → "ay" (remove "zz")
```

### Video Explanation
- [NeetCode - Remove Adjacent Duplicates](https://www.youtube.com/watch?v=nRKZC2JF7LU)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STACK FOR ADJACENT DUPLICATES                                              │
│                                                                             │
│  s = "abbaca"                                                              │
│                                                                             │
│  char 'a': Stack empty, push 'a'                                           │
│  Stack: [a]                                                                 │
│                                                                             │
│  char 'b': Top is 'a' ≠ 'b', push 'b'                                      │
│  Stack: [a, b]                                                             │
│                                                                             │
│  char 'b': Top is 'b' = 'b', MATCH! Pop 'b'                                │
│  Stack: [a]                                                                 │
│                                                                             │
│  char 'a': Top is 'a' = 'a', MATCH! Pop 'a'                                │
│  Stack: []                                                                  │
│                                                                             │
│  char 'c': Stack empty, push 'c'                                           │
│  Stack: [c]                                                                 │
│                                                                             │
│  char 'a': Top is 'c' ≠ 'a', push 'a'                                      │
│  Stack: [c, a]                                                             │
│                                                                             │
│  Result: "ca"                                                               │
│                                                                             │
│  Key insight: Stack naturally handles the "chain reaction" of removals     │
│  When we remove a pair, previously non-adjacent chars become adjacent      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def removeDuplicates(s: str) -> str:
    """
    Remove adjacent duplicates using stack.

    Strategy:
    - For each character, check if it matches stack top
    - If match: pop (remove the pair)
    - If no match: push current character

    Time: O(n) - each character pushed/popped at most once
    Space: O(n) - stack can hold entire string
    """
    stack = []

    for char in s:
        if stack and stack[-1] == char:
            # Adjacent duplicate found - remove both
            stack.pop()
        else:
            # No duplicate - add to stack
            stack.append(char)

    return ''.join(stack)
```

### Complexity
- **Time**: O(n) - single pass
- **Space**: O(n) - stack stores result

### Edge Cases
- No duplicates: Return original string
- All same characters: `"aaaa"` → `""`
- Alternating: `"abab"` → `"abab"` (no adjacent dups)
- Chain reaction: `"abba"` → `""`

### Common Mistakes
- Trying to use string slicing (inefficient)
- Not handling empty stack case
- Forgetting that removal can create new adjacent pairs

### Related Problems
- LC #1209 Remove All Adjacent Duplicates in String II
- LC #20 Valid Parentheses
- LC #844 Backspace String Compare

---

## Problem 8: Backspace String Compare (LC #844) - Easy

- [LeetCode](https://leetcode.com/problems/backspace-string-compare/)

### Problem Statement
Given two strings `s` and `t`, return true if they are equal when both are typed into empty text editors. '#' means a backspace character.

### Examples
```
Input: s = "ab#c", t = "ad#c"
Output: true
Explanation: Both become "ac"

Input: s = "ab##", t = "c#d#"
Output: true
Explanation: Both become ""

Input: s = "a#c", t = "b"
Output: false
Explanation: s becomes "c", t is "b"
```

### Video Explanation
- [NeetCode - Backspace String Compare](https://www.youtube.com/watch?v=k2qrymM_DOo)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SIMULATING BACKSPACE WITH STACK                                            │
│                                                                             │
│  s = "ab#c"                                                                │
│                                                                             │
│  'a': Push 'a'           Stack: [a]                                        │
│  'b': Push 'b'           Stack: [a, b]                                     │
│  '#': Backspace! Pop     Stack: [a]                                        │
│  'c': Push 'c'           Stack: [a, c]                                     │
│                                                                             │
│  Result: "ac"                                                               │
│                                                                             │
│  t = "ad#c"                                                                │
│                                                                             │
│  'a': Push 'a'           Stack: [a]                                        │
│  'd': Push 'd'           Stack: [a, d]                                     │
│  '#': Backspace! Pop     Stack: [a]                                        │
│  'c': Push 'c'           Stack: [a, c]                                     │
│                                                                             │
│  Result: "ac"                                                               │
│                                                                             │
│  "ac" == "ac" → TRUE                                                       │
│                                                                             │
│  Edge case: Backspace on empty stack = do nothing                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def backspaceCompare(s: str, t: str) -> bool:
    """
    Compare strings after processing backspaces.

    Strategy: Build final string using stack, then compare.

    Time: O(n + m)
    Space: O(n + m)
    """
    def process(string: str) -> str:
        """Process string with backspaces using stack."""
        stack = []
        for char in string:
            if char == '#':
                if stack:  # Only pop if stack not empty
                    stack.pop()
            else:
                stack.append(char)
        return ''.join(stack)

    return process(s) == process(t)


def backspaceCompare_optimal(s: str, t: str) -> bool:
    """
    O(1) space solution using two pointers from the end.

    Time: O(n + m)
    Space: O(1)
    """
    def next_valid_char(string: str, index: int) -> int:
        """Find next valid character index, skipping backspaced chars."""
        skip = 0
        while index >= 0:
            if string[index] == '#':
                skip += 1
                index -= 1
            elif skip > 0:
                skip -= 1
                index -= 1
            else:
                break
        return index

    i, j = len(s) - 1, len(t) - 1

    while i >= 0 or j >= 0:
        i = next_valid_char(s, i)
        j = next_valid_char(t, j)

        # Compare characters at valid positions
        if i >= 0 and j >= 0:
            if s[i] != t[j]:
                return False
        elif i >= 0 or j >= 0:
            # One string has characters left, other doesn't
            return False

        i -= 1
        j -= 1

    return True
```

### Complexity
- **Stack Solution**: Time O(n+m), Space O(n+m)
- **Two Pointers**: Time O(n+m), Space O(1)

### Edge Cases
- All backspaces: `"###"` → `""`
- Backspace on empty: `"#a"` → `"a"`
- Equal after processing: `"ab#c"` vs `"ac"` → `True`
- Different lengths but equal: Works correctly

### Common Mistakes
- Trying to pop from empty stack
- Not handling multiple consecutive backspaces
- Processing from left (harder) instead of right

### Related Problems
- LC #1047 Remove All Adjacent Duplicates In String
- LC #71 Simplify Path
- LC #20 Valid Parentheses

---

## Summary: Easy Stack & Queue Problems

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | Valid Parentheses | Stack matching | O(n) | O(n) |
| 2 | Min Stack | Store min with each element | O(1) | O(n) |
| 3 | Queue using Stacks | Two stacks, lazy transfer | O(1)* | O(n) |
| 4 | Stack using Queues | Queue rotation | O(n) push | O(n) |
| 5 | Baseball Game | Simulation with stack | O(n) | O(n) |
| 6 | Next Greater Element I | Monotonic stack + map | O(n) | O(n) |
| 7 | Remove Adjacent Duplicates | Stack for pairs | O(n) | O(n) |
| 8 | Backspace Compare | Stack or two pointers | O(n) | O(n)/O(1) |

*O(1) amortized

---

## Practice More Easy Problems

- [ ] LC #1021 - Remove Outermost Parentheses
- [ ] LC #1614 - Maximum Nesting Depth of the Parentheses
- [ ] LC #1475 - Final Prices With a Special Discount
- [ ] LC #933 - Number of Recent Calls (Queue)
