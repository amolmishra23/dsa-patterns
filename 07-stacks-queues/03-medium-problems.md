# Monotonic Stack - Deep Dive

## What is a Monotonic Stack?

A monotonic stack is a stack that maintains elements in sorted order (either increasing or decreasing).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MONOTONIC STACK TYPES                                    │
│                                                                             │
│  MONOTONIC INCREASING (bottom to top):                                      │
│  - Elements increase from bottom to top                                     │
│  - Used for: Next Smaller Element, Previous Smaller Element                 │
│  - Pop when: current < stack top                                            │
│                                                                             │
│  Example: [3, 1, 4, 1, 5]                                                   │
│  Stack states: [3] → [1] → [1,4] → [1,1] → [1,1,5]                         │
│                                                                             │
│  MONOTONIC DECREASING (bottom to top):                                      │
│  - Elements decrease from bottom to top                                     │
│  - Used for: Next Greater Element, Previous Greater Element                 │
│  - Pop when: current > stack top                                            │
│                                                                             │
│  Example: [3, 1, 4, 1, 5]                                                   │
│  Stack states: [3] → [3,1] → [4] → [4,1] → [5]                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Template

```python
def monotonic_stack_template(nums: list[int]) -> list[int]:
    """
    Generic monotonic stack template.

    This template finds the NEXT GREATER element for each position.
    Modify the comparison for different variants.

    Time: O(n) - each element pushed and popped at most once
    Space: O(n) - stack size
    """
    n = len(nums)
    result = [-1] * n  # Default: no answer found
    stack = []  # Stack of indices

    for i in range(n):
        # Pop elements that satisfy the condition
        # For Next Greater: pop while stack top < current
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]  # Current element is the answer

        stack.append(i)

    return result
```

---

## Problem 1: Next Greater Element I (LC #496) - Easy

- [LeetCode](https://leetcode.com/problems/next-greater-element-i/)

### Problem Statement
The **next greater element** of some element `x` in an array is the **first greater** element that is to the **right** of `x` in the same array. You are given two distinct 0-indexed integer arrays `nums1` and `nums2`, where `nums1` is a subset of `nums2`. Return an array of next greater elements for each element in `nums1`.

### Video Explanation
- [NeetCode - Next Greater Element I](https://www.youtube.com/watch?v=68a1Dc_qVq4)

### Examples
```
Input: nums1 = [4,1,2], nums2 = [1,3,4,2]
Output: [-1,3,-1]
Explanation:
  - 4: no element greater to its right in nums2
  - 1: next greater is 3
  - 2: no element greater to its right

Input: nums1 = [2,4], nums2 = [1,2,3,4]
Output: [3,-1]
```

### Intuition Development
```
Build a map of Next Greater Elements for nums2, then look up nums1!

nums2 = [1, 3, 4, 2]

Using monotonic decreasing stack:
┌─────────────────────────────────────────────────────────────────┐
│ i=0: num=1, stack=[]        → push 1, stack=[1]                │
│ i=1: num=3, stack=[1]       → 3>1, pop! NGE[1]=3, push 3       │
│                                stack=[3]                        │
│ i=2: num=4, stack=[3]       → 4>3, pop! NGE[3]=4, push 4       │
│                                stack=[4]                        │
│ i=3: num=2, stack=[4]       → 2<4, push 2, stack=[4,2]         │
│                                                                  │
│ Remaining: NGE[4]=-1, NGE[2]=-1                                 │
│                                                                  │
│ Final map: {1:3, 3:4, 4:-1, 2:-1}                               │
│ Look up nums1=[4,1,2] → [-1, 3, -1] ✓                           │
└─────────────────────────────────────────────────────────────────┘

Key: Monotonic DECREASING stack finds NEXT GREATER element!
```

### Solution
```python
def nextGreaterElement(nums1: list[int], nums2: list[int]) -> list[int]:
    """
    Find next greater element using monotonic decreasing stack.

    Strategy:
    1. Build NGE map for all elements in nums2
    2. Look up each element of nums1

    Time: O(n + m)
    Space: O(n)
    """
    # Build NGE map for nums2
    nge_map = {}
    stack = []  # Monotonic decreasing stack

    for num in nums2:
        # Pop all smaller elements - current is their NGE
        while stack and stack[-1] < num:
            smaller = stack.pop()
            nge_map[smaller] = num
        stack.append(num)

    # Remaining elements have no NGE
    for num in stack:
        nge_map[num] = -1

    # Look up each element in nums1
    return [nge_map[num] for num in nums1]
```

### Complexity
- **Time**: O(n + m) - Process nums2 once, lookup nums1
- **Space**: O(n) - Hash map and stack

### Edge Cases
- No greater element: Return -1
- All elements same: All return -1
- Strictly increasing: Each returns next element

---

## Problem 2: Next Greater Element II (LC #503) - Medium

- [LeetCode](https://leetcode.com/problems/next-greater-element-ii/)

### Problem Statement
Given a circular integer array `nums` (i.e., the next element of `nums[nums.length - 1]` is `nums[0]`), return the next greater number for every element in `nums`. The next greater number is the first greater number to its right (circularly).

### Video Explanation
- [NeetCode - Next Greater Element II](https://www.youtube.com/watch?v=ARN3QpLODV0)

### Examples
```
Input: nums = [1,2,1]
Output: [2,-1,2]
Explanation:
  - 1: next greater is 2
  - 2: no greater element (even circularly)
  - 1: wrapping around, next greater is 2

Input: nums = [1,2,3,4,3]
Output: [2,3,4,-1,4]
```

### Intuition Development
```
Handle circular array by iterating TWICE!

nums = [1, 2, 1] (circular)

Imagine: [1, 2, 1, 1, 2, 1]  ← conceptually doubled
         [0, 1, 2, 0, 1, 2]  ← use i % n for actual index

┌─────────────────────────────────────────────────────────────────┐
│ First pass (i=0 to 2): Build stack                              │
│   i=0: num=1, stack=[]      → push 0, stack=[0]                │
│   i=1: num=2, stack=[0]     → 2>nums[0], result[0]=2, push 1   │
│   i=2: num=1, stack=[1]     → 1<nums[1], push 2, stack=[1,2]   │
│                                                                  │
│ Second pass (i=3 to 5): Handle circular                         │
│   i=3: idx=0, num=1         → 1<nums[2], 1<nums[1], no pop     │
│   i=4: idx=1, num=2         → 2>nums[2], result[2]=2           │
│   i=5: idx=2, num=1         → no change                        │
│                                                                  │
│ Only push during first pass (i < n)!                            │
│ Result: [2, -1, 2] ✓                                            │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def nextGreaterElements(nums: list[int]) -> list[int]:
    """
    Next greater element in circular array.

    Strategy: Iterate twice to handle circular nature.

    Time: O(n)
    Space: O(n)
    """
    n = len(nums)
    result = [-1] * n
    stack = []  # Stack of indices

    # Iterate twice (2n) to handle circular
    for i in range(2 * n):
        idx = i % n

        while stack and nums[stack[-1]] < nums[idx]:
            result[stack.pop()] = nums[idx]

        # Only push indices from first pass
        if i < n:
            stack.append(idx)

    return result
```

### Complexity
- **Time**: O(n) - Each element pushed and popped at most once
- **Space**: O(n) - Stack and result array

### Edge Cases
- All same values: All return -1
- Maximum element: Always returns -1
- Strictly decreasing then increasing: Tests circular nature

---

## Problem 3: Daily Temperatures (LC #739) - Medium

- [LeetCode](https://leetcode.com/problems/daily-temperatures/)

### Problem Statement
Given an array of integers `temperatures` representing the daily temperatures, return an array `answer` such that `answer[i]` is the number of days you have to wait after the `ith` day to get a warmer temperature. If there is no future day with warmer temperature, return 0.

### Video Explanation
- [NeetCode - Daily Temperatures](https://www.youtube.com/watch?v=cTBiBSnjO3c)

### Examples
```
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
Explanation:
  - Day 0 (73): Day 1 (74) is warmer → wait 1 day
  - Day 2 (75): Day 6 (76) is warmer → wait 4 days
  - Day 6 (76): No warmer day → 0

Input: temperatures = [30,40,50,60]
Output: [1,1,1,0]

Input: temperatures = [30,60,90]
Output: [1,1,0]
```

### Intuition Development
```
Find NEXT GREATER element, but return DISTANCE instead of value!

temperatures = [73, 74, 75, 71, 69, 72, 76, 73]

Store INDICES in stack (to calculate distance)!

┌─────────────────────────────────────────────────────────────────┐
│ i=0: temp=73, stack=[]      → push 0, stack=[0]                │
│ i=1: temp=74, 74>73         → pop 0, result[0]=1-0=1, push 1   │
│ i=2: temp=75, 75>74         → pop 1, result[1]=2-1=1, push 2   │
│ i=3: temp=71, 71<75         → push 3, stack=[2,3]              │
│ i=4: temp=69, 69<71         → push 4, stack=[2,3,4]            │
│ i=5: temp=72, 72>69, 72>71  → pop 4, result[4]=5-4=1           │
│                             → pop 3, result[3]=5-3=2           │
│                             → push 5, stack=[2,5]              │
│ i=6: temp=76, 76>72, 76>75  → pop 5, result[5]=6-5=1           │
│                             → pop 2, result[2]=6-2=4           │
│                             → push 6, stack=[6]                │
│ i=7: temp=73, 73<76         → push 7, stack=[6,7]              │
│                                                                  │
│ Remaining: result[6]=0, result[7]=0                             │
│ Result: [1,1,4,2,1,1,0,0] ✓                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def dailyTemperatures(temperatures: list[int]) -> list[int]:
    """
    Find days until warmer temperature.

    Strategy: Monotonic decreasing stack of indices.
    When we find warmer temp, calculate days difference.

    Time: O(n)
    Space: O(n)
    """
    n = len(temperatures)
    result = [0] * n
    stack = []  # Stack of indices

    for i in range(n):
        # Pop all cooler temperatures
        while stack and temperatures[stack[-1]] < temperatures[i]:
            prev_idx = stack.pop()
            result[prev_idx] = i - prev_idx  # Days difference

        stack.append(i)

    return result
```

### Complexity
- **Time**: O(n) - Each element pushed and popped at most once
- **Space**: O(n) - Stack stores indices

### Edge Cases
- All same temperatures: All return 0
- Strictly increasing: All return 1 except last
- Strictly decreasing: All return 0

---

## Problem 4: Largest Rectangle in Histogram (LC #84) - Hard

- [LeetCode](https://leetcode.com/problems/largest-rectangle-in-histogram/)

### Problem Statement
Given an array of integers `heights` representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

### Video Explanation
- [NeetCode - Largest Rectangle in Histogram](https://www.youtube.com/watch?v=zx5Sw9130L0)

### Examples
```
Input: heights = [2,1,5,6,2,3]
Output: 10
Explanation: Rectangle of height 5 spanning bars at index 2-3

Input: heights = [2,4]
Output: 4
```

### Intuition Development
```
For each bar, find how far LEFT and RIGHT it can extend!

heights = [2, 1, 5, 6, 2, 3]

┌─────────────────────────────────────────────────────────────────┐
│ Key insight: Bar can extend until it hits a SHORTER bar        │
│                                                                  │
│ For bar of height 5 at index 2:                                 │
│   Left boundary:  bar 1 (height=1 < 5) at index 1              │
│   Right boundary: bar 2 (height=2 < 5) at index 4              │
│   Width = 4 - 1 - 1 = 2                                        │
│   Area = 5 × 2 = 10 ★                                          │
│                                                                  │
│ Visualization:                                                   │
│          ┌───┐                                                  │
│        ┌─┤   │                                                  │
│        │ │   ├───┐                                              │
│  ┌───┐ │ │▓▓▓│▓▓▓│ ┌───┐   ← Rectangle of height 5             │
│  │   │ │ │▓▓▓│▓▓▓│ │   │                                        │
│  │   │ │ │▓▓▓│▓▓▓│ │   │                                        │
│  └───┴─┴─┴───┴───┴─┴───┘                                        │
│    2   1   5   6   2   3                                        │
│                                                                  │
│ Use monotonic INCREASING stack (pop when shorter bar found)     │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def largestRectangleArea(heights: list[int]) -> int:
    """
    Find largest rectangle using monotonic increasing stack.

    Key insight: For each bar, find how far left and right it can extend.
    A bar can extend until it hits a shorter bar.

    Strategy:
    - Maintain increasing stack of heights
    - When shorter bar found, calculate area for all taller bars
    - Width = current index - previous index in stack - 1

    Time: O(n)
    Space: O(n)
    """
    max_area = 0
    stack = []  # Stack of indices

    # Add sentinel 0 at end to process remaining bars
    heights = heights + [0]

    for i, h in enumerate(heights):
        # Pop all taller bars
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]

            # Calculate width
            if stack:
                width = i - stack[-1] - 1
            else:
                width = i  # Extends to beginning

            max_area = max(max_area, height * width)

        stack.append(i)

    return max_area
```

### Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LARGEST RECTANGLE EXAMPLE                                │
│                                                                             │
│  heights = [2, 1, 5, 6, 2, 3]                                               │
│                                                                             │
│          ┌───┐                                                              │
│        ┌─┤   │                                                              │
│        │ │   ├───┐                                                          │
│  ┌───┐ │ │   │   │ ┌───┐                                                    │
│  │   │ │ │   │   │ │   │                                                    │
│  │   │ │ │   │   │ │   │                                                    │
│  └───┴─┴─┴───┴───┴─┴───┘                                                    │
│    2   1   5   6   2   3                                                    │
│                                                                             │
│  When we reach height 2 at index 4:                                         │
│  - Pop 6: width = 4 - 2 - 1 = 1, area = 6                                   │
│  - Pop 5: width = 4 - 1 - 1 = 2, area = 10 ← Maximum!                       │
│                                                                             │
│  Answer: 10                                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Complexity
- **Time**: O(n) - Each bar pushed and popped at most once
- **Space**: O(n) - Stack size

### Edge Cases
- Single bar: Area = height
- All same height: Area = n × height
- Strictly increasing: Each bar forms its own rectangle
- Empty array: Return 0

---

## Problem 5: Trapping Rain Water (LC #42) - Hard

- [LeetCode](https://leetcode.com/problems/trapping-rain-water/)

### Problem Statement
Given `n` non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

### Video Explanation
- [NeetCode - Trapping Rain Water](https://www.youtube.com/watch?v=ZI2z5pq0TqA)

### Examples
```
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: Water units shown as █

Input: height = [4,2,0,3,2,5]
Output: 9
```

### Intuition Development
```
Water at each position = min(max_left, max_right) - height

height = [0,1,0,2,1,0,1,3,2,1,2,1]

Visual:
        █
    █░░░█░░█
█░░█░░░█░█░█░█
    █ █     █ █ █   █     █
0 1 0 2 1 0 1 3 2 1 2 1
          ░ = water trapped

Two approaches:

APPROACH 1: Monotonic Stack (calculate water layer by layer)
┌─────────────────────────────────────────────────────────────────┐
│ Maintain decreasing stack                                        │
│ When taller bar found, calculate water above shorter bars        │
│                                                                  │
│ Water trapped = width × bounded_height                           │
│ bounded_height = min(left_bar, current_bar) - bottom_bar        │
└─────────────────────────────────────────────────────────────────┘

APPROACH 2: Two Pointers (O(1) space)
┌─────────────────────────────────────────────────────────────────┐
│ Insight: Water depends on SMALLER of (max_left, max_right)      │
│                                                                  │
│ Move pointer from smaller side:                                  │
│   If left_max < right_max: process left, move left++            │
│   Else: process right, move right--                             │
│                                                                  │
│ Water at current = current_max - height                         │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def trap(height: list[int]) -> int:
    """
    Calculate trapped rain water using monotonic stack.

    Strategy:
    - Maintain stack of indices in decreasing height order
    - When taller bar found, calculate water trapped above shorter bars

    Time: O(n)
    Space: O(n)
    """
    water = 0
    stack = []  # Stack of indices

    for i, h in enumerate(height):
        while stack and height[stack[-1]] < h:
            bottom = height[stack.pop()]

            if not stack:
                break  # No left boundary

            # Calculate water above bottom bar
            left_idx = stack[-1]
            width = i - left_idx - 1
            bounded_height = min(height[left_idx], h) - bottom
            water += width * bounded_height

        stack.append(i)

    return water


def trap_two_pointers(height: list[int]) -> int:
    """
    Alternative: Two pointers approach (O(1) space).

    Water at each position = min(max_left, max_right) - height

    Time: O(n)
    Space: O(1)
    """
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
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
- **Time**: O(n) for both approaches
- **Space**: O(n) for stack, O(1) for two pointers

### Edge Cases
- All zeros: No water
- Strictly increasing then decreasing: Water trapped in the middle
- Two bars only: No water if heights same, else min(h1,h2)-0

---

## Problem 6: Maximal Rectangle (LC #85) - Hard

- [LeetCode](https://leetcode.com/problems/maximal-rectangle/)

### Problem Statement
Given a `rows × cols` binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

### Video Explanation
- [NeetCode - Maximal Rectangle](https://www.youtube.com/watch?v=dAVF2NpC3j4)

### Examples
```
Input: matrix = [["1","0","1","0","0"],
                 ["1","0","1","1","1"],
                 ["1","1","1","1","1"],
                 ["1","0","0","1","0"]]
Output: 6
Explanation: The maximal rectangle is shown with X:
  ["1","0","X","X","X"],
  ["1","0","X","X","X"],
  ...

Input: matrix = [["0"]]
Output: 0

Input: matrix = [["1"]]
Output: 1
```

### Intuition Development
```
Transform to "Largest Rectangle in Histogram" for each row!

matrix:        histogram heights:
1 0 1 0 0      Row 0: [1,0,1,0,0] → max_area = 1
1 0 1 1 1      Row 1: [2,0,2,1,1] → max_area = 3
1 1 1 1 1      Row 2: [3,1,3,2,2] → max_area = 6 ★
1 0 0 1 0      Row 3: [4,0,0,3,0] → max_area = 4

┌─────────────────────────────────────────────────────────────────┐
│ Building histogram row by row:                                   │
│                                                                  │
│ For each cell (i, j):                                            │
│   If matrix[i][j] == '1': heights[j] += 1                        │
│   Else: heights[j] = 0   (reset - no continuous 1s above)        │
│                                                                  │
│ Row 2 visualization (heights = [3,1,3,2,2]):                     │
│                                                                  │
│    ┌───┐   ┌───┐                                                 │
│    │   │   │   │ ┌───┐ ┌───┐                                     │
│    │   │   │   │ │   │ │   │                                     │
│    │   │ ┌─┴───┴─┴───┴─┴───┘    Area of height 2 × width 3 = 6  │
│    3   1   3   2   2                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def maximalRectangle(matrix: list[list[str]]) -> int:
    """
    Find largest rectangle of 1s using histogram approach.

    Strategy:
    - Build histogram for each row (height = consecutive 1s above)
    - Apply largest rectangle in histogram for each row

    Time: O(m * n)
    Space: O(n)
    """
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0

    for row in matrix:
        # Update heights
        for j in range(cols):
            if row[j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0

        # Calculate max rectangle for this row's histogram
        max_area = max(max_area, largestRectangleArea(heights[:]))

    return max_area


def largestRectangleArea(heights: list[int]) -> int:
    """Helper: Largest rectangle in histogram."""
    max_area = 0
    stack = []
    heights = heights + [0]

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i - stack[-1] - 1 if stack else i
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area
```

### Complexity
- **Time**: O(m × n) - Process each cell once per row
- **Space**: O(n) - Heights array and stack

### Edge Cases
- All zeros: Return 0
- All ones: Return m × n
- Single cell: Return 1 or 0
- Single row/column: Treat as histogram

---

## Problem 7: 132 Pattern (LC #456) - Medium

- [LeetCode](https://leetcode.com/problems/132-pattern/)

### Problem Statement
Given an array of `n` integers `nums`, a **132 pattern** is a subsequence of three integers `nums[i]`, `nums[j]` and `nums[k]` such that `i < j < k` and `nums[i] < nums[k] < nums[j]`. Return `true` if there is a 132 pattern in `nums`, otherwise return `false`.

### Video Explanation
- [NeetCode - 132 Pattern](https://www.youtube.com/watch?v=q5ANAl8Z458)

### Examples
```
Input: nums = [1,2,3,4]
Output: false
Explanation: No 132 pattern exists

Input: nums = [3,1,4,2]
Output: true
Explanation: [1,4,2] is a 132 pattern (1 < 2 < 4)

Input: nums = [-1,3,2,0]
Output: true
Explanation: [-1,3,2] or [-1,3,0] or [-1,2,0]
```

### Intuition Development
```
We need: nums[i] < nums[k] < nums[j]  where i < j < k
         "1"      "2"       "3"       (positions in name)

Strategy: Traverse RIGHT to LEFT!
┌─────────────────────────────────────────────────────────────────┐
│ - Stack holds potential "3"s (nums[j]) in decreasing order     │
│ - max_k tracks the maximum "2" (nums[k]) popped so far         │
│ - If current < max_k, we found "1"! Pattern exists!            │
│                                                                  │
│ Example: [3, 1, 4, 2]                                           │
│                                                                  │
│ i=3: num=2, stack=[]        → push 2, stack=[2], max_k=-∞      │
│ i=2: num=4, 4>2             → pop 2, max_k=2, push 4           │
│                                stack=[4], max_k=2               │
│ i=1: num=1, 1 < max_k=2     → FOUND! 1 < 2 < 4 ✓               │
│                                                                  │
│ Why right to left?                                               │
│ - We fix "3" and "2" first (in stack and max_k)                 │
│ - Then look for "1" to the left                                 │
│ - max_k is guaranteed to have a "3" to its right               │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def find132pattern(nums: list[int]) -> bool:
    """
    Find 132 pattern using monotonic stack.

    Strategy:
    - Traverse from right to left
    - Maintain stack of potential "3"s (nums[j])
    - Track maximum "2" (nums[k]) that was popped
    - If current < max_k, found pattern (current is "1")

    Time: O(n)
    Space: O(n)
    """
    if len(nums) < 3:
        return False

    stack = []  # Potential "3"s (decreasing from bottom)
    max_k = float('-inf')  # Maximum "2" seen

    # Traverse right to left
    for i in range(len(nums) - 1, -1, -1):
        # Check if current can be "1" (nums[i] < nums[k])
        if nums[i] < max_k:
            return True

        # Pop smaller elements - they become candidates for "2"
        while stack and stack[-1] < nums[i]:
            max_k = stack.pop()

        # Push current as potential "3"
        stack.append(nums[i])

    return False
```

### Complexity
- **Time**: O(n) - Single pass right to left
- **Space**: O(n) - Stack size

### Edge Cases
- Less than 3 elements: Return false
- Strictly increasing: No 132 pattern
- Strictly decreasing: No 132 pattern
- All same values: No 132 pattern

---

## Problem 8: Remove K Digits (LC #402) - Medium

- [LeetCode](https://leetcode.com/problems/remove-k-digits/)

### Problem Statement
Given string `num` representing a non-negative integer `num`, and an integer `k`, return the smallest possible integer after removing `k` digits from `num`. The result should not have leading zeros.

### Video Explanation
- [NeetCode - Remove K Digits](https://www.youtube.com/watch?v=cFabMOnJdate)

### Examples
```
Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove 4, 3, 2 → "1219"

Input: num = "10200", k = 1
Output: "200"
Explanation: Remove 1 → "0200" → "200"

Input: num = "10", k = 2
Output: "0"
Explanation: Remove all digits → "0"
```

### Intuition Development
```
Key insight: Keep LEFTMOST digits as SMALL as possible!

Remove larger digits when a smaller digit comes after.

num = "1432219", k = 3

┌─────────────────────────────────────────────────────────────────┐
│ i=0: digit='1', stack=[]      → push '1', stack=['1']          │
│ i=1: digit='4', '4'>'1'       → push '4', stack=['1','4']      │
│ i=2: digit='3', '3'<'4'       → pop '4', k=2, push '3'         │
│                                  stack=['1','3']                │
│ i=3: digit='2', '2'<'3'       → pop '3', k=1, push '2'         │
│                                  stack=['1','2']                │
│ i=4: digit='2', '2'>='2'      → push '2', stack=['1','2','2']  │
│ i=5: digit='1', '1'<'2'       → pop '2', k=0                   │
│                                  stack=['1','2','1']            │
│ i=6: digit='9', k=0           → push '9', stack=['1','2','1','9']│
│                                                                  │
│ Result: "1219" ✓                                                 │
│                                                                  │
│ Why this works?                                                  │
│ - Larger digits on left hurt more (1xxx vs 4xxx)                │
│ - Removing peak creates smaller number                          │
│ - Monotonic increasing stack keeps smallest possible prefix     │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def removeKdigits(num: str, k: int) -> str:
    """
    Remove k digits to get smallest number.

    Strategy (Monotonic Increasing Stack):
    - Keep digits in increasing order
    - Pop larger digits when smaller digit comes
    - This ensures leftmost digits are as small as possible

    Time: O(n)
    Space: O(n)
    """
    stack = []

    for digit in num:
        # Pop larger digits (if we still need to remove)
        while stack and k > 0 and stack[-1] > digit:
            stack.pop()
            k -= 1

        stack.append(digit)

    # If k > 0, remove from end
    if k > 0:
        stack = stack[:-k]

    # Remove leading zeros and handle empty result
    result = ''.join(stack).lstrip('0')
    return result if result else '0'
```

### Complexity
- **Time**: O(n) - Each digit pushed and popped at most once
- **Space**: O(n) - Stack to build result

### Edge Cases
- `k >= len(num)`: Return "0"
- Leading zeros: Strip with `lstrip('0')`
- All same digits: Remove from end
- Already smallest (increasing): Remove from end

---

## Summary: Monotonic Stack Variants

| Problem | Stack Type | Pop Condition | Answer When |
|---------|------------|---------------|-------------|
| Next Greater | Decreasing | current > top | Pop |
| Next Smaller | Increasing | current < top | Pop |
| Prev Greater | Decreasing | current >= top | Before push |
| Prev Smaller | Increasing | current <= top | Before push |
| Largest Rectangle | Increasing | current < top | Pop |
| Trapping Water | Decreasing | current > top | Pop |

---

## Key Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MONOTONIC STACK DECISION GUIDE                           │
│                                                                             │
│  Q: What are you looking for?                                               │
│                                                                             │
│  → Next GREATER element: Monotonic DECREASING stack                         │
│  → Next SMALLER element: Monotonic INCREASING stack                         │
│  → Previous GREATER: Decreasing, answer before push                         │
│  → Previous SMALLER: Increasing, answer before push                         │
│                                                                             │
│  Q: What direction to traverse?                                             │
│                                                                             │
│  → NEXT element: Left to right                                              │
│  → PREVIOUS element: Right to left (or left to right with modification)     │
│  → CIRCULAR: Two passes (2n iterations)                                     │
│                                                                             │
│  Q: Store values or indices?                                                │
│                                                                             │
│  → Need position/distance: Store INDICES                                    │
│  → Only need values: Store VALUES                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practice More Problems

- [ ] LC #901 - Online Stock Span
- [ ] LC #907 - Sum of Subarray Minimums
- [ ] LC #1019 - Next Greater Node In Linked List
- [ ] LC #1475 - Final Prices With a Special Discount
- [ ] LC #1856 - Maximum Subarray Min-Product

