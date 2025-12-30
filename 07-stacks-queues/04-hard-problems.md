# Stacks & Queues - Hard Problems

## Problem 1: Largest Rectangle in Histogram (LC #84) - Hard

- [LeetCode](https://leetcode.com/problems/largest-rectangle-in-histogram/)

### Video Explanation
- [NeetCode - Largest Rectangle in Histogram](https://www.youtube.com/watch?v=zx5Sw9130L0)

### Problem Statement
Given an array of integers `heights` representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

### Examples
```
Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: Rectangle with height 5 spanning indices 2-3

Input: heights = [2,4]
Output: 4
```

### Visual Intuition
```
heights = [2,1,5,6,2,3]

Pattern: Monotonic Increasing Stack
Why: For each bar, find how far it can extend left/right

Step 0 (Visualize Histogram):

      6
     ┌─┐
   5 │ │
  ┌─┬─┬─┐     3
  │ │ │ │   2┌─┐
  │ │ │ │  ┌─┤ │
2 │ │ │ │  │ │ │
┌─┐1│ │ │  │ │ │
│ │┌─┤ │ │  │ │ │
│ ││ │ │ │  │ │ │
└─┴─┴─┴─┴─┴─┴─┘
 0  1  2  3  4  5

Step 1 (Process with Monotonic Stack):

  Stack stores indices with INCREASING heights
  When we see smaller bar → pop and calculate

  i=0, h=2: stack=[] → push 0
            stack=[0]

  i=1, h=1: h[0]=2 > 1 → pop 0, calculate
            height=2, width=1 (no left bound)
            area = 2×1 = 2
            stack=[1]

  i=2, h=5: 5 > 1 → push 2
            stack=[1, 2]

  i=3, h=6: 6 > 5 → push 3
            stack=[1, 2, 3]

  i=4, h=2: h[3]=6 > 2 → pop 3
            height=6, width=4-2-1=1
            area = 6×1 = 6

            h[2]=5 > 2 → pop 2
            height=5, width=4-1-1=2
            area = 5×2 = 10 ★ MAX!

            stack=[1, 4]

Step 2 (Continue Processing):

  i=5, h=3: 3 > 2 → push 5
            stack=[1, 4, 5]

  End (sentinel h=0):
            pop 5: height=3, width=6-4-1=1, area=3
            pop 4: height=2, width=6-1-1=4, area=8
            pop 1: height=1, width=6, area=6

Answer: 10

Stack State Trace:
  ┌─────┬──────────────┬─────────────────────────┐
  │  i  │ Stack        │ Popped Area             │
  ├─────┼──────────────┼─────────────────────────┤
  │  0  │ [0]          │ -                       │
  │  1  │ [1]          │ h=2, w=1, a=2           │
  │  2  │ [1,2]        │ -                       │
  │  3  │ [1,2,3]      │ -                       │
  │  4  │ [1,4]        │ h=6,a=6; h=5,a=10 ★     │
  │  5  │ [1,4,5]      │ -                       │
  │ end │ []           │ h=3,a=3; h=2,a=8; h=1,a=6│
  └─────┴──────────────┴─────────────────────────┘

Key Insight:
- When bar is popped, we know its right boundary
- Left boundary = previous stack top (or 0)
- Width = right - left - 1
- Each bar pushed/popped exactly once → O(n)
```

### Solution
```python
def largestRectangleArea(heights: list[int]) -> int:
    """
    Find largest rectangle area in histogram using monotonic stack.

    Strategy:
    1. Maintain stack of indices with increasing heights
    2. When we see a smaller height, pop and calculate area
    3. For popped bar: width extends from previous stack top to current index

    Key insight: When we pop index i, we know:
    - Left boundary: previous stack top + 1 (or 0 if stack empty)
    - Right boundary: current index - 1

    Time: O(n) - each bar pushed and popped at most once
    Space: O(n) - stack storage
    """
    stack = []  # Stack of indices with increasing heights
    max_area = 0

    # Add sentinel 0 at end to force processing all remaining bars
    for i, h in enumerate(heights + [0]):
        # While current height is smaller than stack top
        while stack and heights[stack[-1]] > h:
            # Pop and calculate area with popped height
            height = heights[stack.pop()]

            # Width: from previous stack top to current index
            # If stack empty, width extends from index 0
            width = i if not stack else i - stack[-1] - 1

            max_area = max(max_area, height * width)

        stack.append(i)

    return max_area
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- Empty array → return 0
- Single bar → return its height
- All same height → width × height
- Strictly increasing → last bar determines
- Strictly decreasing → first bar determines

---

## Problem 2: Maximal Rectangle (LC #85) - Hard

- [LeetCode](https://leetcode.com/problems/maximal-rectangle/)

### Video Explanation
- [NeetCode - Maximal Rectangle](https://www.youtube.com/watch?v=g8bSdXCG-lA)

### Problem Statement
Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

### Examples
```
Input: matrix = [
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
Output: 6
```

### Visual Intuition
```
Original matrix:
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

Pattern: Build Histogram per Row + Apply LC #84
Why: Reduce 2D problem to multiple 1D problems

Step 0 (Build Histograms Row by Row):

  Row 0: [1,0,1,0,0]  ← heights = matrix values

  Row 1: [2,0,2,1,1]  ← if '1': height++, else: reset to 0
          ↑   ↑ ↑ ↑
          1+1 1 1 1

  Row 2: [3,1,3,2,2]  ← continue building
          ↑ ↑ ↑ ↑ ↑
          2+1 0+1 2+1 1+1 1+1

  Row 3: [4,0,0,3,0]  ← '0' resets height
          ↑ ↑ ↑ ↑ ↑
          3+1 reset reset 2+1 reset

Step 1 (Visualize Each Histogram):

  Row 0:        Row 1:        Row 2:        Row 3:
  █   █         █   █         █   █ █ █     █     █
                █   █ █ █     █ █ █ █ █     █     █
                              █ █ █ █ █     █     █
                                            █     █

  [1,0,1,0,0]   [2,0,2,1,1]   [3,1,3,2,2]   [4,0,0,3,0]

Step 2 (Apply Largest Rectangle in Histogram):

  Row 0: max area = 1
  Row 1: max area = 3 (height 1 × width 3)
  Row 2: max area = 6 ★ (height 2 × width 3)
  Row 3: max area = 4 (height 4 × width 1)

Step 3 (Find Max Rectangle in Row 2):

  heights = [3, 1, 3, 2, 2]

      █     █
      █     █ █ █
    █ █ █ █ █ █ █
    █ █ █ █ █ █ █
    0 1 2 3 4

  Using monotonic stack:
  - Pop height 3 (idx 2): area = 3×1 = 3
  - Pop height 2 (idx 3,4): area = 2×3 = 6 ★
  - etc.

Answer: 6

Rectangle Visualization:

  1 0 1 0 0     . . . . .
  1 0 1 1 1     . . ■ ■ ■   ← max rectangle
  1 1 1 1 1     . . ■ ■ ■   ← height=2, width=3
  1 0 0 1 0     . . . . .

Key Insight:
- Each row creates a histogram
- Heights accumulate from top (reset on '0')
- Apply monotonic stack for each row
- O(rows × cols) total time
```

### Solution
```python
def maximalRectangle(matrix: list[list[str]]) -> int:
    """
    Find maximal rectangle in binary matrix.

    Strategy:
    1. Build histogram heights for each row
    2. Apply largest rectangle in histogram for each row
    3. Track maximum area across all rows

    Time: O(rows × cols)
    Space: O(cols)
    """
    if not matrix or not matrix[0]:
        return 0

    cols = len(matrix[0])
    heights = [0] * cols
    max_area = 0

    def largestRectangleArea(heights):
        stack = []
        max_area = 0

        for i, h in enumerate(heights + [0]):
            while stack and heights[stack[-1]] > h:
                height = heights[stack.pop()]
                width = i if not stack else i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)

        return max_area

    for row in matrix:
        # Update heights: increment if '1', reset to 0 if '0'
        for j in range(cols):
            heights[j] = heights[j] + 1 if row[j] == '1' else 0

        # Calculate max rectangle for this row's histogram
        max_area = max(max_area, largestRectangleArea(heights))

    return max_area
```

### Complexity
- **Time**: O(rows × cols)
- **Space**: O(cols)

### Edge Cases
- Empty matrix → return 0
- All zeros → return 0
- All ones → rows × cols
- Single row/column → simple scan

---

## Problem 3: Trapping Rain Water (LC #42) - Hard

- [LeetCode](https://leetcode.com/problems/trapping-rain-water/)

### Video Explanation
- [NeetCode - Trapping Rain Water](https://www.youtube.com/watch?v=ZI2z5pq0TqA)

### Problem Statement
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

### Examples
```
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

Input: height = [4,2,0,3,2,5]
Output: 9
```

### Visual Intuition
```
height = [0,1,0,2,1,0,1,3,2,1,2,1]

Pattern: Monotonic Decreasing Stack (or Two Pointers)
Why: Water at position = min(maxLeft, maxRight) - height

Step 0 (Visualize Water Trapping):

            3
            █
        2   █   2
        █ ≈ █ █ ≈ █
      1 █ ≈ █ █ ≈ █   1
      █ ≈ █ █ ≈ █ █ █
  0   █ ≈ █ █ ≈ █ █ █
  ─────────────────────
  0 1 2 3 4 5 6 7 8 9 10 11

  ≈ = water trapped

Step 1 (Stack Approach - Process Left to Right):

  Maintain DECREASING stack of indices
  When we see taller bar → water can be trapped

  i=0, h=0: stack=[0]
  i=1, h=1: h[0]=0 < 1 → pop 0 (bottom)
            no left wall, skip
            stack=[1]
  i=2, h=0: 0 < 1 → push 2
            stack=[1, 2]
  i=3, h=2: h[2]=0 < 2 → pop 2 (bottom)
            left=1, right=3, bottom=0
            width = 3-1-1 = 1
            height = min(h[1], h[3]) - h[2] = min(1,2) - 0 = 1
            water += 1×1 = 1

            h[1]=1 < 2 → pop 1
            no left wall, skip
            stack=[3]

Step 2 (Continue Processing):

  i=4, h=1: stack=[3, 4]
  i=5, h=0: stack=[3, 4, 5]
  i=6, h=1: pop 5 (bottom=0)
            left=4, width=1, height=min(1,1)-0=1
            water += 1
            stack=[3, 4, 6]
  i=7, h=3: pop 6 (bottom=1)
            left=4, width=2, height=min(1,3)-1=0
            water += 0

            pop 4 (bottom=1)
            left=3, width=3, height=min(2,3)-1=1
            water += 3

            pop 3 (bottom=2)
            no left wall, skip
            stack=[7]

Step 3 (Two Pointer Alternative - O(1) Space):

  left=0, right=11, leftMax=0, rightMax=0

  ┌─────────────────────────────────────────────────┐
  │ While left < right:                             │
  │   if height[left] < height[right]:              │
  │     if height[left] >= leftMax:                 │
  │       leftMax = height[left]                    │
  │     else:                                       │
  │       water += leftMax - height[left]           │
  │     left++                                      │
  │   else:                                         │
  │     (mirror logic for right side)               │
  │     right--                                     │
  └─────────────────────────────────────────────────┘

  Key: Process from side with smaller max
       (that side determines water level)

Answer: 6

Key Insight:
- Water at position i = min(maxLeft, maxRight) - height[i]
- Stack: O(n) time, O(n) space
- Two pointers: O(n) time, O(1) space
- Process smaller side first (determines water level)
```

### Solution
```python
def trap(height: list[int]) -> int:
    """
    Calculate trapped rain water using monotonic stack.

    Strategy:
    1. Maintain decreasing stack of indices
    2. When we see a taller bar, calculate water trapped
    3. Water is bounded by min of left and right walls

    Time: O(n)
    Space: O(n)
    """
    stack = []  # Decreasing stack of indices
    water = 0

    for i, h in enumerate(height):
        # While current bar is taller than stack top
        while stack and height[stack[-1]] < h:
            # Pop the bottom of the "pool"
            bottom = stack.pop()

            if not stack:
                break  # No left wall

            # Calculate water trapped
            left = stack[-1]
            width = i - left - 1
            bounded_height = min(height[left], h) - height[bottom]
            water += width * bounded_height

        stack.append(i)

    return water


def trap_two_pointer(height: list[int]) -> int:
    """
    Two-pointer approach - O(1) space.

    Strategy:
    - Water at position depends on min(maxLeft, maxRight)
    - Process from side with smaller max
    """
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max = right_max = 0
    water = 0

    while left < right:
        if height[left] < height[right]:
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1

    return water
```

### Complexity
- **Stack**: O(n) time, O(n) space
- **Two-pointer**: O(n) time, O(1) space

### Edge Cases
- Empty array → return 0
- No valleys → return 0
- Single bar → return 0
- Strictly increasing/decreasing → return 0

---

## Problem 4: Basic Calculator (LC #224) - Hard

- [LeetCode](https://leetcode.com/problems/basic-calculator/)

### Video Explanation
- [NeetCode - Basic Calculator](https://www.youtube.com/watch?v=081AqOuasw0)

### Problem Statement
Implement a basic calculator to evaluate a simple expression string containing '+', '-', '(', ')' and non-negative integers.

### Examples
```
Input: s = "1 + 1"
Output: 2

Input: s = " 2-1 + 2 "
Output: 3

Input: s = "(1+(4+5+2)-3)+(6+8)"
Output: 23
```

### Visual Intuition
```
Expression: "(1+(4+5+2)-3)+(6+8)"

Pattern: Stack to Handle Nested Parentheses
Why: Save state before '(', restore after ')'

Step 0 (State to Track):

  ┌─────────────────────────────────────────────────┐
  │ result = running total within current scope     │
  │ sign = +1 or -1 for next number                 │
  │ stack = [(prev_result, sign_before_paren), ...] │
  └─────────────────────────────────────────────────┘

Step 1 (Process "(1+(4+5+2)-3)"):

  "(": push (result=0, sign=+1), reset result=0, sign=+1
       stack = [(0, +1)]

  "1": result = 0 + 1*1 = 1

  "+": sign = +1

  "(": push (result=1, sign=+1), reset result=0, sign=+1
       stack = [(0, +1), (1, +1)]

  "4+5+2": result = 4 + 5 + 2 = 11

  ")": pop (prev=1, sign=+1)
       result = prev + sign * result = 1 + 1*11 = 12
       stack = [(0, +1)]

  "-": sign = -1

  "3": result = 12 + (-1)*3 = 9

  ")": pop (prev=0, sign=+1)
       result = 0 + 1*9 = 9
       stack = []

Step 2 (Process "+(6+8)"):

  "+": sign = +1

  "(": push (result=9, sign=+1), reset result=0
       stack = [(9, +1)]

  "6+8": result = 6 + 8 = 14

  ")": pop (prev=9, sign=+1)
       result = 9 + 1*14 = 23
       stack = []

Answer: 23

Stack State Trace:
  ┌──────────┬────────────────┬─────────┬───────┐
  │ Position │ Stack          │ Result  │ Sign  │
  ├──────────┼────────────────┼─────────┼───────┤
  │ (        │ [(0,+1)]       │ 0       │ +1    │
  │ 1        │ [(0,+1)]       │ 1       │ +1    │
  │ (        │ [(0,+1),(1,+1)]│ 0       │ +1    │
  │ 4+5+2    │ [(0,+1),(1,+1)]│ 11      │ +1    │
  │ )        │ [(0,+1)]       │ 12      │ +1    │
  │ -3       │ [(0,+1)]       │ 9       │ -1    │
  │ )        │ []             │ 9       │ +1    │
  │ (        │ [(9,+1)]       │ 0       │ +1    │
  │ 6+8      │ [(9,+1)]       │ 14      │ +1    │
  │ )        │ []             │ 23      │ +1    │
  └──────────┴────────────────┴─────────┴───────┘

Key Insight:
- '(' : save current state, start fresh
- ')' : restore state, combine with inner result
- Stack depth = nesting level
- O(n) time, O(depth) space
```

### Solution
```python
def calculate(s: str) -> int:
    """
    Basic calculator with +, -, parentheses.

    Strategy:
    1. Use stack to handle nested parentheses
    2. Track current result and sign
    3. On '(': push current state, reset
    4. On ')': pop and combine with saved state

    Time: O(n)
    Space: O(n) for nested parentheses
    """
    stack = []
    result = 0
    num = 0
    sign = 1  # 1 for +, -1 for -

    for c in s:
        if c.isdigit():
            num = num * 10 + int(c)

        elif c == '+':
            result += sign * num
            num = 0
            sign = 1

        elif c == '-':
            result += sign * num
            num = 0
            sign = -1

        elif c == '(':
            # Save current state and reset
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1

        elif c == ')':
            result += sign * num
            num = 0
            # Pop sign and previous result
            result *= stack.pop()  # sign before parenthesis
            result += stack.pop()  # previous result

    return result + sign * num
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- Single number → return that number
- Nested parentheses → stack handles properly
- Leading/trailing spaces → ignored
- Negative result → handled by sign

---

## Problem 5: Basic Calculator II (LC #227) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/basic-calculator-ii/)

### Video Explanation
- [NeetCode - Basic Calculator II](https://www.youtube.com/watch?v=m6cHLP2gDwI)

### Problem Statement
Implement calculator with +, -, *, / (no parentheses). Division truncates toward zero.

### Examples
```
Input: s = "3+2*2"
Output: 7

Input: s = " 3/2 "
Output: 1

Input: s = " 3+5 / 2 "
Output: 5
```

### Visual Intuition
```
Expression: "3+2*2"

Pattern: Stack with Operator Precedence
Why: */ has higher precedence than +/-, handle immediately

Step 0 (Strategy):

  ┌─────────────────────────────────────────────────┐
  │ +/- : Push number (with sign) to stack          │
  │ */ : Operate on stack top immediately           │
  │ End : Sum all values in stack                   │
  └─────────────────────────────────────────────────┘

  This delays +/- until all */ are done!

Step 1 (Process "3+2*2"):

  "3": num = 3
  "+": prev_op = '+' → push +3 to stack
       stack = [3]
       op = '+'

  "2": num = 2
  "*": prev_op = '+' → push +2 to stack
       stack = [3, 2]
       op = '*'

  "2": num = 2
  End: prev_op = '*' → stack[-1] *= 2
       stack = [3, 4]

  Sum: 3 + 4 = 7

Answer: 7

Step 2 (More Complex Example "3+5/2-4*2"):

  "3": num = 3
  "+": push +3, stack = [3]

  "5": num = 5
  "/": push +5, stack = [3, 5]

  "2": num = 2
  "-": prev_op = '/' → stack[-1] /= 2 → stack = [3, 2]
       push -4 next...

  Wait, let me trace more carefully:

  "3": num = 3
  "+": sign = '+', push nothing yet
  "5": num = 5
  "/": prev = '+' → push +3, sign = '/'
       stack = [3]
  "2": num = 2
  "-": prev = '/' → stack[-1] = 5/2 = 2
       Hmm, stack was [3], we need 5 on stack first

  Let me re-trace with correct algorithm:

  sign = '+' (initial)

  "3": num = 3
  "+": sign='+' → push +3, stack=[3], sign='+'
  "5": num = 5
  "/": sign='+' → push +5, stack=[3,5], sign='/'
  "2": num = 2
  "-": sign='/' → pop, divide: 5/2=2, push 2
       stack=[3,2], sign='-'
  "4": num = 4
  "*": sign='-' → push -4, stack=[3,2,-4], sign='*'
  "2": num = 2
  End: sign='*' → pop, multiply: -4*2=-8, push -8
       stack=[3,2,-8]

  Sum: 3 + 2 + (-8) = -3

Operator Precedence Visualization:

  3 + 5 / 2 - 4 * 2
    ↑   ↑       ↑
   low high    high

  Process high precedence immediately
  Delay low precedence (push to stack)
  Sum stack at end for low precedence ops

Key Insight:
- Stack accumulates +/- terms
- */ modifies stack top immediately
- Final sum handles all +/- at once
- No need for parentheses handling
```

### Solution
```python
def calculate(s: str) -> int:
    """
    Calculator with +, -, *, / (no parentheses).

    Strategy:
    1. Use stack for intermediate values
    2. +/- push number with sign to stack
    3. */ operate on stack top immediately
    4. Sum stack at end

    Time: O(n)
    Space: O(n)
    """
    stack = []
    num = 0
    sign = '+'

    for i, c in enumerate(s):
        if c.isdigit():
            num = num * 10 + int(c)

        # Process when we hit an operator or end of string
        if c in '+-*/' or i == len(s) - 1:
            if sign == '+':
                stack.append(num)
            elif sign == '-':
                stack.append(-num)
            elif sign == '*':
                stack.append(stack.pop() * num)
            elif sign == '/':
                # Truncate toward zero
                stack.append(int(stack.pop() / num))

            sign = c
            num = 0

    return sum(stack)
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- Single number → return that number
- Division by larger number → truncates to 0
- Negative intermediate results → handled correctly
- Multiple spaces → ignored

---

## Problem 6: Sliding Window Maximum (LC #239) - Hard

- [LeetCode](https://leetcode.com/problems/sliding-window-maximum/)

### Video Explanation
- [NeetCode - Sliding Window Maximum](https://www.youtube.com/watch?v=DfljaUwZsOk)

### Problem Statement
Given an array `nums` and sliding window size `k`, return the max in each window position.

### Examples
```
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]

Window positions and max:
[1  3  -1] -3  5  3  6  7    max = 3
 1 [3  -1  -3] 5  3  6  7    max = 3
 1  3 [-1  -3  5] 3  6  7    max = 5
 1  3  -1 [-3  5  3] 6  7    max = 5
 1  3  -1  -3 [5  3  6] 7    max = 6
 1  3  -1  -3  5 [3  6  7]   max = 7
```

### Visual Intuition
```
nums = [1,3,-1,-3,5,3,6,7], k = 3

Pattern: Monotonic Decreasing Deque
Why: Front always has maximum for current window

Step 0 (Deque Properties):

  ┌─────────────────────────────────────────────────┐
  │ Deque stores INDICES (not values)               │
  │ Values at indices are DECREASING                │
  │                                                 │
  │ Front: maximum in current window                │
  │ Back:  add new elements, remove smaller ones    │
  │                                                 │
  │ Remove from front: if index outside window      │
  │ Remove from back:  if value ≤ new element       │
  └─────────────────────────────────────────────────┘

Step 1 (Build Initial Window):

  nums: [1, 3, -1, -3, 5, 3, 6, 7]
        【─────────】
         window k=3

  i=0, val=1: deque = [0]
              values: [1]

  i=1, val=3: 3 > 1 → pop 0 from back
              deque = [1]
              values: [3]

  i=2, val=-1: -1 < 3 → append
               deque = [1, 2]
               values: [3, -1]

  Window complete! Output: nums[deque[0]] = 3

Step 2 (Slide Window):

  i=3, val=-3:
    Check front: 1 > 3-3=0? No, still in window
    -3 < -1 → append
    deque = [1, 2, 3]
    values: [3, -1, -3]

    Output: 3

  i=4, val=5:
    Check front: 1 > 4-3=1? No, still in window
    5 > -3 → pop 3
    5 > -1 → pop 2
    5 > 3 → pop 1
    deque = [4]
    values: [5]

    Output: 5

  i=5, val=3:
    Check front: 4 > 5-3=2? Yes, still in window
    3 < 5 → append
    deque = [4, 5]
    values: [5, 3]

    Output: 5

Step 3 (Continue):

  i=6, val=6:
    6 > 3 → pop 5
    6 > 5 → pop 4
    deque = [6]
    Output: 6

  i=7, val=7:
    Check front: 6 > 7-3=4? Yes
    7 > 6 → pop 6
    deque = [7]
    Output: 7

Answer: [3, 3, 5, 5, 6, 7]

Deque State Visualization:
  ┌─────┬────────────────┬────────────────┬────────┐
  │  i  │ Window         │ Deque (indices)│ Output │
  ├─────┼────────────────┼────────────────┼────────┤
  │  2  │ [1, 3, -1]     │ [1, 2]         │ 3      │
  │  3  │ [3, -1, -3]    │ [1, 2, 3]      │ 3      │
  │  4  │ [-1, -3, 5]    │ [4]            │ 5      │
  │  5  │ [-3, 5, 3]     │ [4, 5]         │ 5      │
  │  6  │ [5, 3, 6]      │ [6]            │ 6      │
  │  7  │ [3, 6, 7]      │ [7]            │ 7      │
  └─────┴────────────────┴────────────────┴────────┘

Key Insight:
- Remove smaller elements from back (they'll never be max)
- Remove out-of-window elements from front
- Each element added/removed at most once → O(n)
- Deque size ≤ k
```

### Solution
```python
from collections import deque

def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    """
    Find maximum in each sliding window using monotonic deque.

    Strategy:
    1. Maintain decreasing deque of indices
    2. Front of deque is always the maximum
    3. Remove elements outside window from front
    4. Remove smaller elements from back

    Time: O(n) - each element added/removed once
    Space: O(k) - deque size
    """
    dq = deque()  # Stores indices, values are decreasing
    result = []

    for i, num in enumerate(nums):
        # Remove elements outside window from front
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Remove smaller elements from back
        # They can never be maximum while current element exists
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        # Add to result once we have a full window
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

### Complexity
- **Time**: O(n)
- **Space**: O(k)

### Edge Cases
- k = 1 → return nums itself
- k = n → return single max
- All same elements → all same in result
- Strictly decreasing → first element of each window

---

## Summary

| # | Problem | Key Technique |
|---|---------|---------------|
| 1 | Largest Rectangle in Histogram | Monotonic increasing stack |
| 2 | Maximal Rectangle | Histogram per row + LC #84 |
| 3 | Trapping Rain Water | Monotonic stack or two pointers |
| 4 | Basic Calculator | Stack for parentheses |
| 5 | Basic Calculator II | Stack with operator precedence |
| 6 | Sliding Window Maximum | Monotonic decreasing deque |
