# Math & Geometry - Hard Problems

## Problem 1: Max Points on a Line (LC #149) - Hard

- [LeetCode](https://leetcode.com/problems/max-points-on-a-line/)

### Video Explanation
- [NeetCode - Max Points on a Line](https://www.youtube.com/watch?v=Bb9lOXUOnFw)

### Problem Statement
Find maximum points that lie on the same line.


### Visual Intuition
```
Max Points on a Line
points = [[1,1],[2,2],[3,3],[1,2],[2,3]]

For each point, count slopes to other points:
  From (1,1):
    to (2,2): slope = 1/1 = 1
    to (3,3): slope = 2/2 = 1
    to (1,2): slope = 1/0 = inf
    to (2,3): slope = 2/1 = 2

  Slopes: {1: 2, inf: 1, 2: 1}
  Max on line through (1,1) = 2 + 1 = 3

Use GCD for exact slope representation:
  slope (dy,dx) → (dy/gcd, dx/gcd)

Handle: same point (duplicates), vertical lines

Answer: 3 points on line y=x
```

### Solution
```python
from collections import defaultdict
from math import gcd

def maxPoints(points: list[list[int]]) -> int:
    """
    For each point, group others by slope.

    Strategy:
    - Use (dx, dy) in reduced form as slope key
    - Handle vertical lines and duplicates

    Time: O(n²)
    Space: O(n)
    """
    if len(points) <= 2:
        return len(points)

    def get_slope(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if dx == 0:
            return (0, 1)  # Vertical line
        if dy == 0:
            return (1, 0)  # Horizontal line

        # Reduce to lowest terms
        g = gcd(abs(dx), abs(dy))
        dx, dy = dx // g, dy // g

        # Normalize sign
        if dx < 0:
            dx, dy = -dx, -dy

        return (dx, dy)

    max_count = 1

    for i in range(len(points)):
        slopes = defaultdict(int)

        for j in range(i + 1, len(points)):
            slope = get_slope(points[i], points[j])
            slopes[slope] += 1
            max_count = max(max_count, slopes[slope] + 1)

    return max_count
```

### Edge Cases
- Less than 3 points → return count
- All same points → return count
- All collinear → return n
- Vertical/horizontal lines → handled specially

---

## Problem 2: Largest Rectangle in Histogram (LC #84) - Hard

- [LeetCode](https://leetcode.com/problems/largest-rectangle-in-histogram/)

### Video Explanation
- [NeetCode - Largest Rectangle in Histogram](https://www.youtube.com/watch?v=zx5Sw9130L0)

### Problem Statement
Find largest rectangle in histogram.


### Visual Intuition
```
Largest Rectangle in Histogram
heights = [2,1,5,6,2,3]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: For each bar, find how far left/right it can extend
             Use monotonic increasing stack to track boundaries
═══════════════════════════════════════════════════════════════

Histogram Visualization:
────────────────────────
     6 |      █
     5 |    █ █
     4 |    █ █
     3 |    █ █   █
     2 | █  █ █ █ █
     1 | █ █ █ █ █ █
       ─────────────
         0 1 2 3 4 5
         2 1 5 6 2 3

Stack-Based Algorithm (stores indices):
───────────────────────────────────────
  i=0, h=2: stack empty → push 0
            stack = [0]

  i=1, h=1: h < heights[0]=2 → POP!
            Pop 0: height=2, width=1 (no left boundary)
            Area = 2 × 1 = 2
            Push 1
            stack = [1]

  i=2, h=5: h > heights[1]=1 → push 2
            stack = [1, 2]

  i=3, h=6: h > heights[2]=5 → push 3
            stack = [1, 2, 3]

  i=4, h=2: h < heights[3]=6 → POP!
            Pop 3: height=6, width=4-2-1=1
            Area = 6 × 1 = 6

            Still h < heights[2]=5 → POP!
            Pop 2: height=5, width=4-1-1=2
            Area = 5 × 2 = 10 ★ MAX!

            Push 4
            stack = [1, 4]

  i=5, h=3: h > heights[4]=2 → push 5
            stack = [1, 4, 5]

  i=6 (sentinel h=0): Pop remaining
            Pop 5: height=3, width=6-4-1=1, area=3
            Pop 4: height=2, width=6-1-1=4, area=8
            Pop 1: height=1, width=6, area=6

Maximum Rectangle:
──────────────────
     6 |
     5 |    ████
     5 |    ████
     4 |    ████
     3 |    ████
     2 |    ████
     1 |
       ─────────────
           2 3
       Area = 5 × 2 = 10

Answer: 10

WHY THIS WORKS:
════════════════
● When we pop bar i, current bar is first shorter bar on right
● Stack top (after pop) is first shorter bar on left
● Width = right_boundary - left_boundary - 1
● Each bar pushed/popped exactly once → O(n)
```

### Solution
```python
def largestRectangleArea(heights: list[int]) -> int:
    """
    Monotonic stack to find left/right boundaries.

    Time: O(n)
    Space: O(n)
    """
    stack = []  # Indices of increasing heights
    max_area = 0

    for i, h in enumerate(heights + [0]):  # Add sentinel
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area
```

### Edge Cases
- Empty array → return 0
- Single bar → return its height
- All same height → width × height
- Strictly increasing/decreasing → handled by sentinel

---

## Problem 3: Basic Calculator (LC #224) - Hard

- [LeetCode](https://leetcode.com/problems/basic-calculator/)

### Video Explanation
- [NeetCode - Basic Calculator](https://www.youtube.com/watch?v=081AqOuasw0)

### Problem Statement
Implement calculator with +, -, parentheses.


### Visual Intuition
```
Basic Calculator
s = "(1+(4+5+2)-3)+(6+8)"

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Use stack to save state before entering parentheses
             '(' = push (result, sign), reset
             ')' = pop and combine with inner result
═══════════════════════════════════════════════════════════════

Step-by-Step Trace:
───────────────────
Variables: result=0, sign=+1, num=0
Stack: []

Process "(1+(4+5+2)-3)+(6+8)":

Step 1: '('
  Push: (result=0, sign=+1) → Stack: [(0, +1)]
  Reset: result=0, sign=+1

Step 2: '1'
  num = 1

Step 3: '+'
  result = 0 + (+1)*1 = 1
  sign = +1, num = 0

Step 4: '('
  Push: (result=1, sign=+1) → Stack: [(0,+1), (1,+1)]
  Reset: result=0, sign=+1

Step 5-9: '4+5+2'
  '4': num=4
  '+': result = 0 + 1*4 = 4, sign=+1
  '5': num=5
  '+': result = 4 + 1*5 = 9, sign=+1
  '2': num=2

Step 10: ')'
  result = 9 + 1*2 = 11  (finalize inner)
  Pop: (1, +1)
  result = 1 + (+1)*11 = 12
  Stack: [(0, +1)]

Step 11: '-'
  sign = -1

Step 12: '3'
  num = 3

Step 13: ')'
  result = 12 + (-1)*3 = 9  (finalize inner)
  Pop: (0, +1)
  result = 0 + (+1)*9 = 9
  Stack: []

Step 14: '+'
  sign = +1

Step 15: '('
  Push: (result=9, sign=+1) → Stack: [(9, +1)]
  Reset: result=0, sign=+1

Step 16-18: '6+8'
  '6': num=6
  '+': result = 0 + 1*6 = 6, sign=+1
  '8': num=8

Step 19: ')'
  result = 6 + 1*8 = 14
  Pop: (9, +1)
  result = 9 + (+1)*14 = 23

Final: result = 23

Stack State Visualization:
──────────────────────────
  (1+(4+5+2)-3)+(6+8)
  ↑
  Stack: [(0,+1)]  result=0

  (1+(4+5+2)-3)+(6+8)
     ↑
  Stack: [(0,+1),(1,+1)]  result=0

  (1+(4+5+2)-3)+(6+8)
           ↑
  Stack: [(0,+1)]  result=12

  (1+(4+5+2)-3)+(6+8)
              ↑
  Stack: []  result=9

Answer: 23

WHY THIS WORKS:
════════════════
● Stack saves "context" before entering parentheses
● Inner expression evaluated independently
● ')' combines inner result with saved context
● Sign before '(' determines how inner result is added
```

### Solution
```python
def calculate(s: str) -> int:
    """
    Stack-based evaluation.

    Time: O(n)
    Space: O(n)
    """
    stack = []
    result = 0
    num = 0
    sign = 1

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
            # Save current state
            stack.append(result)
            stack.append(sign)
            result = 0
            sign = 1
        elif c == ')':
            result += sign * num
            num = 0
            # Apply saved sign and add to saved result
            result *= stack.pop()  # sign
            result += stack.pop()  # previous result

    return result + sign * num
```

### Edge Cases
- Only whitespace → return 0
- Single number → return that number
- Nested parentheses → stack handles properly
- Leading/trailing operators → need validation

---

## Problem 4: Integer to English Words (LC #273) - Hard

- [LeetCode](https://leetcode.com/problems/integer-to-english-words/)

### Video Explanation
- [NeetCode - Integer to English Words](https://www.youtube.com/watch?v=g_l4BQ8JXWQ)

### Problem Statement
Convert integer to English words.


### Visual Intuition
```
Integer to English Words
num = 1234567891

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Process in groups of 3 digits, add scale words
             Handle special cases: teens (11-19), zeros
═══════════════════════════════════════════════════════════════

Break into groups of 3:
───────────────────────
  1,234,567,891
  │ │   │   │
  │ │   │   └── ones group:     891
  │ │   └────── thousands:      567
  │ └────────── millions:       234
  └──────────── billions:       1

Process each group (1-999):
───────────────────────────
  Group 891:
    8 hundreds → "Eight Hundred"
    91 = ninety + one → "Ninety One"
    Result: "Eight Hundred Ninety One"

  Group 567:
    5 hundreds → "Five Hundred"
    67 = sixty + seven → "Sixty Seven"
    + "Thousand"
    Result: "Five Hundred Sixty Seven Thousand"

  Group 234:
    2 hundreds → "Two Hundred"
    34 = thirty + four → "Thirty Four"
    + "Million"
    Result: "Two Hundred Thirty Four Million"

  Group 1:
    1 → "One"
    + "Billion"
    Result: "One Billion"

Special Cases for Teens:
────────────────────────
  10 → "Ten"       15 → "Fifteen"
  11 → "Eleven"    16 → "Sixteen"
  12 → "Twelve"    17 → "Seventeen"
  13 → "Thirteen"  18 → "Eighteen"
  14 → "Fourteen"  19 → "Nineteen"

Helper Function Logic:
──────────────────────
  helper(891):
    891 >= 100: "Eight" + " Hundred " + helper(91)
    91 >= 20:   "Ninety" + " " + helper(1)
    1 < 20:     "One"

    = "Eight Hundred Ninety One"

Final Assembly:
───────────────
  "One Billion" + " " +
  "Two Hundred Thirty Four Million" + " " +
  "Five Hundred Sixty Seven Thousand" + " " +
  "Eight Hundred Ninety One"

Answer: "One Billion Two Hundred Thirty Four Million
         Five Hundred Sixty Seven Thousand
         Eight Hundred Ninety One"

WHY THIS WORKS:
════════════════
● English numbers follow consistent 3-digit grouping
● Scale words: Thousand, Million, Billion cycle every 3 digits
● Teens are special case (not "ten-one" but "eleven")
● Recursion naturally handles the structure
```

### Solution
```python
def numberToWords(num: int) -> str:
    """
    Process in groups of 3 digits.

    Time: O(1) - max 10 groups
    Space: O(1)
    """
    if num == 0:
        return "Zero"

    ones = ["", "One", "Two", "Three", "Four", "Five",
            "Six", "Seven", "Eight", "Nine", "Ten",
            "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen",
            "Sixteen", "Seventeen", "Eighteen", "Nineteen"]

    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty",
            "Sixty", "Seventy", "Eighty", "Ninety"]

    thousands = ["", "Thousand", "Million", "Billion"]

    def helper(n):
        if n == 0:
            return ""
        elif n < 20:
            return ones[n] + " "
        elif n < 100:
            return tens[n // 10] + " " + helper(n % 10)
        else:
            return ones[n // 100] + " Hundred " + helper(n % 100)

    result = ""
    for i, unit in enumerate(thousands):
        if num % 1000 != 0:
            result = helper(num % 1000) + unit + " " + result
        num //= 1000

    return result.strip()
```

### Edge Cases
- Zero → return "Zero"
- Single digit → direct lookup
- Teens (11-19) → special handling
- Billion → max for 32-bit

---

## Problem 5: The Skyline Problem (LC #218) - Hard

- [LeetCode](https://leetcode.com/problems/the-skyline-problem/)

### Video Explanation
- [NeetCode - The Skyline Problem](https://www.youtube.com/watch?v=GSBLe8cKu0s)

### Problem Statement
Compute the skyline formed by buildings.

### Visual Intuition
```
The Skyline Problem
buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]

Events at x-coordinates (entering/leaving):
  x=2: enter h=10
  x=3: enter h=15 → skyline changes to 15
  x=5: enter h=12
  x=7: leave h=15 → skyline drops to 12
  x=9: leave h=10
  x=12: leave h=12 → skyline drops to 0
  ...

      15|    ___
      12|   |   |___
      10|  _|       |    ___
       8|           |   |   |__
       0|__|        |___|      |___
         2 3 5 7 9 12 15 19 20 24

Use max-heap to track active heights
Output key points where height changes
```


### Intuition
```
Buildings: [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]

Events at x-coordinates:
- Building starts: add height to active set
- Building ends: remove height from active set
- Skyline point when max height changes

Use sweep line with max heap for active heights.
```

### Solution
```python
import heapq

def getSkyline(buildings: list[list[int]]) -> list[list[int]]:
    """
    Sweep line with max heap.

    Strategy:
    - Create events for building starts and ends
    - Process events left to right
    - Track max height using heap
    - Record point when max height changes

    Time: O(n log n)
    Space: O(n)
    """
    # Create events: (x, type, height)
    # type: 0 = start (process first), 1 = end
    events = []

    for left, right, height in buildings:
        events.append((left, 0, height, right))   # Start
        events.append((right, 1, height, right))  # End

    # Sort: by x, then starts before ends, then taller starts first
    events.sort(key=lambda e: (e[0], e[1], -e[2] if e[1] == 0 else e[2]))

    result = []
    # Max heap: (-height, end_x)
    heap = [(0, float('inf'))]  # Ground level

    for x, event_type, height, end_x in events:
        if event_type == 0:  # Building start
            heapq.heappush(heap, (-height, end_x))

        # Remove expired buildings
        while heap[0][1] <= x:
            heapq.heappop(heap)

        # Current max height
        max_height = -heap[0][0]

        # Add to result if height changed
        if not result or result[-1][1] != max_height:
            result.append([x, max_height])

    return result
```

### Complexity
- **Time**: O(n log n)
- **Space**: O(n)

### Edge Cases
- Single building → two points
- Same height buildings → merged output
- Overlapping buildings → max height wins
- Adjacent buildings → shared edge

---

## Problem 6: Perfect Rectangle (LC #391) - Hard

- [LeetCode](https://leetcode.com/problems/perfect-rectangle/)

### Video Explanation
- [NeetCode - Perfect Rectangle](https://www.youtube.com/watch?v=8JM_dyOu_JY)

### Problem Statement
Determine if rectangles form an exact cover of a rectangular region.

### Visual Intuition
```
Perfect Rectangle - Check if rectangles form exact cover
rectangles = [[1,1,3,3],[3,1,4,2],[3,2,4,4],[1,3,2,4],[2,3,3,4]]

    4 +--+--+
      |  |  |
    3 +--+--+
      |     |
    2 |  +--+
      |  |  |
    1 +--+--+
      1  2  3  4

Conditions for perfect rectangle:
1. Total area = bounding box area
2. Each corner appears even times (cancels out)
   EXCEPT 4 corners of bounding box (appear once)

Count corners with XOR set:
- Add corner → appears odd times
- Add again → removed (even)
Final set should have exactly 4 bounding corners
```


### Intuition
```
For perfect cover:
1. Total area = bounding rectangle area
2. Corner points appear odd times only at 4 corners
3. All other points appear even times (2 or 4)

Use XOR of points to track odd occurrences.
```

### Solution
```python
def isRectangleCover(rectangles: list[list[int]]) -> bool:
    """
    Check area and corner conditions.

    Strategy:
    - Calculate total area
    - Track corner points using XOR (odd occurrences)
    - Verify only 4 corners of bounding box remain

    Time: O(n)
    Space: O(n)
    """
    # Find bounding box and total area
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    total_area = 0
    corners = set()

    for x1, y1, x2, y2 in rectangles:
        # Update bounding box
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

        # Add area
        total_area += (x2 - x1) * (y2 - y1)

        # Toggle corners (XOR simulation with set)
        for corner in [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]:
            if corner in corners:
                corners.remove(corner)
            else:
                corners.add(corner)

    # Check 1: Area must match
    expected_area = (max_x - min_x) * (max_y - min_y)
    if total_area != expected_area:
        return False

    # Check 2: Only 4 corners of bounding box should remain
    expected_corners = {
        (min_x, min_y), (min_x, max_y),
        (max_x, min_y), (max_x, max_y)
    }

    return corners == expected_corners
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- Single rectangle → always perfect
- Overlapping rectangles → area mismatch
- Gap in coverage → corner count mismatch
- Same area but wrong shape → corner check catches

---

## Summary

| # | Problem | Key Technique |
|---|---------|---------------|
| 1 | Max Points on Line | Slope grouping |
| 2 | Largest Rectangle | Monotonic stack |
| 3 | Basic Calculator | Stack evaluation |
| 4 | Integer to Words | Group processing |
| 5 | Skyline Problem | Sweep line + max heap |
| 6 | Perfect Rectangle | Area + corner counting |
