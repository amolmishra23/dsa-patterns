# Math & Geometry - Advanced Problems

## Advanced Math Concepts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED MATH TECHNIQUES                                 │
│                                                                             │
│  MODULAR ARITHMETIC:                                                        │
│  • (a + b) % m = ((a % m) + (b % m)) % m                                   │
│  • (a * b) % m = ((a % m) * (b % m)) % m                                   │
│  • (a - b) % m = ((a % m) - (b % m) + m) % m                               │
│  • a^(-1) mod p = a^(p-2) mod p (Fermat's little theorem)                  │
│                                                                             │
│  NUMBER THEORY:                                                             │
│  • GCD(a, b) = GCD(b, a % b) - Euclidean algorithm                         │
│  • LCM(a, b) = a * b / GCD(a, b)                                           │
│  • Sieve of Eratosthenes for primes                                        │
│                                                                             │
│  COMBINATORICS:                                                             │
│  • C(n, k) = n! / (k! * (n-k)!)                                            │
│  • Pascal's triangle: C(n, k) = C(n-1, k-1) + C(n-1, k)                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem 1: Fraction to Recurring Decimal (LC #166) - Medium

- [LeetCode](https://leetcode.com/problems/fraction-to-recurring-decimal/)

### Problem Statement
Convert fraction to decimal string, with repeating part in parentheses.

### Examples
```
Input: numerator = 1, denominator = 3
Output: "0.(3)"

Input: numerator = 4, denominator = 333
Output: "0.(012)"
```

### Intuition Development
```
LONG DIVISION WITH REMAINDER TRACKING:
When remainder repeats, we've found the cycle.

1 / 3:
  1 ÷ 3 = 0 remainder 1
  10 ÷ 3 = 3 remainder 1 ← remainder 1 seen before!

  Position when 1 first seen: after decimal
  Insert ( there, ) at end
  Result: "0.(3)"

4 / 333:
  4 ÷ 333 = 0 remainder 4
  40 ÷ 333 = 0 remainder 40
  400 ÷ 333 = 1 remainder 67
  670 ÷ 333 = 2 remainder 4 ← remainder 4 seen before!

  Result: "0.(012)"
```

### Video Explanation
- [NeetCode - Fraction to Recurring Decimal](https://www.youtube.com/watch?v=a-62yK1S1O4)

### Solution
```python
def fractionToDecimal(numerator: int, denominator: int) -> str:
    """
    Convert fraction to decimal with repeating part.

    Strategy:
    - Track remainders to detect cycle
    - When remainder repeats, we found the repeating part
    """
    if numerator == 0:
        return "0"

    result = []

    # Handle sign
    if (numerator < 0) ^ (denominator < 0):
        result.append('-')

    numerator, denominator = abs(numerator), abs(denominator)

    # Integer part
    result.append(str(numerator // denominator))
    remainder = numerator % denominator

    if remainder == 0:
        return ''.join(result)

    result.append('.')

    # Decimal part
    remainder_index = {}  # remainder -> index in result

    while remainder != 0:
        # Check for repeating
        if remainder in remainder_index:
            # Insert parentheses
            idx = remainder_index[remainder]
            result.insert(idx, '(')
            result.append(')')
            break

        # Record remainder position
        remainder_index[remainder] = len(result)

        # Long division step
        remainder *= 10
        result.append(str(remainder // denominator))
        remainder %= denominator

    return ''.join(result)
```

### Complexity
- **Time**: O(denominator) - at most denominator unique remainders
- **Space**: O(denominator)

### Edge Cases
- numerator = 0 → "0"
- Exact division → no decimal or repeating
- Negative numbers → handle sign
- Large numbers → remainders can get large

---

## Problem 2: Basic Calculator (LC #224) - Hard

- [LeetCode](https://leetcode.com/problems/basic-calculator/)

### Problem Statement
Evaluate expression with +, -, and parentheses.

### Examples
```
Input: s = "(1+(4+5+2)-3)+(6+8)"
Output: 23

Input: s = "- (3 + (4 + 5))"
Output: -12
```

### Intuition Development
```
STACK FOR NESTED EXPRESSIONS:
Save state before (, restore after ).

s = "(1+(4+5+2)-3)+(6+8)"

Process:
  '(' → push state (0, +1), reset
  '1' → result = 1
  '+' → sign = +1
  '(' → push state (1, +1), reset
  '4+5+2' → result = 11
  ')' → pop, result = 1 + 1*11 = 12
  '-' → sign = -1
  '3' → result = 12 - 3 = 9
  ')' → pop, result = 0 + 1*9 = 9
  '+' → sign = +1
  '(' → push state (9, +1), reset
  '6+8' → result = 14
  ')' → pop, result = 9 + 1*14 = 23

Final: 23
```

### Video Explanation
- [NeetCode - Basic Calculator](https://www.youtube.com/watch?v=081AqOuasw0)

### Solution
```python
def calculate(s: str) -> int:
    """
    Evaluate expression with +, -, and parentheses.

    Strategy:
    - Use stack to handle nested parentheses
    - Track current number, operator, and result
    """
    stack = []
    result = 0
    number = 0
    sign = 1  # 1 for +, -1 for -

    for char in s:
        if char.isdigit():
            number = number * 10 + int(char)

        elif char == '+':
            result += sign * number
            number = 0
            sign = 1

        elif char == '-':
            result += sign * number
            number = 0
            sign = -1

        elif char == '(':
            # Save current state
            stack.append(result)
            stack.append(sign)
            # Reset for new expression
            result = 0
            sign = 1

        elif char == ')':
            # Finish current expression
            result += sign * number
            number = 0
            # Apply saved sign and add to saved result
            result *= stack.pop()  # sign
            result += stack.pop()  # previous result

    # Don't forget last number
    result += sign * number

    return result
```

### Complexity
- **Time**: O(n)
- **Space**: O(n) for stack depth

### Edge Cases
- Leading negative sign: "-1+2"
- Deeply nested: "(((1)))"
- Spaces in expression
- Empty parentheses "()"

---

## Problem 3: Integer to English Words (LC #273) - Hard

- [LeetCode](https://leetcode.com/problems/integer-to-english-words/)

### Problem Statement
Convert integer to English words.

### Examples
```
Input: num = 1234567891
Output: "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven Thousand Eight Hundred Ninety One"
```

### Intuition Development
```
GROUP BY THOUSANDS:
Split number into groups of 3 digits.
Convert each group, add scale (Thousand, Million, Billion).

1,234,567,891
  1 → "One Billion"
  234 → "Two Hundred Thirty Four Million"
  567 → "Five Hundred Sixty Seven Thousand"
  891 → "Eight Hundred Ninety One"

THREE-DIGIT CONVERSION:
  234:
    2 hundreds → "Two Hundred"
    34 → "Thirty Four"

  15:
    0 hundreds
    15 → "Fifteen" (special case 11-19)
```

### Video Explanation
- [NeetCode - Integer to English Words](https://www.youtube.com/watch?v=g_l4G0nt_kM)

### Solution
```python
def numberToWords(num: int) -> str:
    """
    Convert number to English words.

    Strategy:
    - Process in groups of 3 digits
    - Handle billions, millions, thousands, ones
    """
    if num == 0:
        return "Zero"

    ones = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven",
            "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen",
            "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tens = ["", "", "Twenty", "Thirty", "Forty", "Fifty",
            "Sixty", "Seventy", "Eighty", "Ninety"]
    thousands = ["", "Thousand", "Million", "Billion"]

    def three_digits(n):
        """Convert 3-digit number to words."""
        if n == 0:
            return ""

        result = []

        if n >= 100:
            result.append(ones[n // 100])
            result.append("Hundred")
            n %= 100

        if n >= 20:
            result.append(tens[n // 10])
            n %= 10

        if n > 0:
            result.append(ones[n])

        return ' '.join(result)

    result = []
    i = 0

    while num > 0:
        if num % 1000 != 0:
            words = three_digits(num % 1000)
            if thousands[i]:
                words += ' ' + thousands[i]
            result.append(words)

        num //= 1000
        i += 1

    return ' '.join(reversed(result))
```

### Complexity
- **Time**: O(1) - bounded by max int
- **Space**: O(1)

### Edge Cases
- num = 0 → "Zero"
- Single digit
- Numbers 11-19 (special words)
- Trailing zeros in groups (1000000 → "One Million")

---

## Problem 4: Max Points on a Line (LC #149) - Hard

- [LeetCode](https://leetcode.com/problems/max-points-on-a-line/)

### Problem Statement
Find maximum points on a single line.

### Examples
```
Input: points = [[1,1],[2,2],[3,3]]
Output: 3 (all on same line)

Input: points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
Output: 4
```

### Intuition Development
```
SLOPE-BASED COUNTING:
For each point, calculate slope to all others.
Points with same slope from same origin are collinear.

AVOIDING FLOATING POINT:
Represent slope as reduced fraction (dx, dy).

points = [[1,1], [2,2], [3,3]]

From [1,1]:
  to [2,2]: slope = (1,1) after reduction
  to [3,3]: slope = (2,2) → (1,1) after reduction

  slope (1,1): 2 points → 2+1 = 3 on line

Max = 3
```

### Video Explanation
- [NeetCode - Max Points on a Line](https://www.youtube.com/watch?v=Bb9lOXUOnFw)

### Solution
```python
from collections import defaultdict
from math import gcd

def maxPoints(points: list[list[int]]) -> int:
    """
    Find maximum points on a single line.

    Strategy:
    - For each point, calculate slope to all other points
    - Use GCD to represent slope as fraction (avoid floating point)
    - Count points with same slope
    """
    if len(points) <= 2:
        return len(points)

    def get_slope(p1, p2):
        """Get slope as reduced fraction (dx, dy)."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if dx == 0:
            return (0, 1)  # Vertical line
        if dy == 0:
            return (1, 0)  # Horizontal line

        # Reduce fraction
        g = gcd(abs(dx), abs(dy))
        dx //= g
        dy //= g

        # Normalize sign
        if dx < 0:
            dx, dy = -dx, -dy

        return (dx, dy)

    max_count = 0

    for i, p1 in enumerate(points):
        slopes = defaultdict(int)

        for j, p2 in enumerate(points):
            if i != j:
                slope = get_slope(p1, p2)
                slopes[slope] += 1

        if slopes:
            max_count = max(max_count, max(slopes.values()) + 1)

    return max_count
```

### Complexity
- **Time**: O(n²)
- **Space**: O(n)

### Edge Cases
- Single point → 1
- Two points → 2
- All same point → return count
- Vertical/horizontal lines

---

## Problem 5: Rectangle Area (LC #223) - Medium

- [LeetCode](https://leetcode.com/problems/rectangle-area/)

### Problem Statement
Find total area covered by two rectangles.

### Examples
```
Input: ax1=-3, ay1=0, ax2=3, ay2=4, bx1=0, by1=-1, bx2=9, by2=2
Output: 45

Area = Area1 + Area2 - Overlap
```

### Intuition Development
```
INCLUSION-EXCLUSION:
Total = Area1 + Area2 - Overlap

Rectangle 1: (-3,0) to (3,4) → 6 × 4 = 24
Rectangle 2: (0,-1) to (9,2) → 9 × 3 = 27

OVERLAP CALCULATION:
  x_overlap = min(ax2, bx2) - max(ax1, bx1)
            = min(3, 9) - max(-3, 0) = 3 - 0 = 3
  y_overlap = min(ay2, by2) - max(ay1, by1)
            = min(4, 2) - max(0, -1) = 2 - 0 = 2

  If either ≤ 0, no overlap
  Overlap area = 3 × 2 = 6

Total = 24 + 27 - 6 = 45
```

### Video Explanation
- [NeetCode - Rectangle Area](https://www.youtube.com/watch?v=JUN67qY1Eqo)

### Solution
```python
def computeArea(ax1: int, ay1: int, ax2: int, ay2: int,
                bx1: int, by1: int, bx2: int, by2: int) -> int:
    """
    Calculate total area of two rectangles.

    Total = Area1 + Area2 - Overlap
    """
    # Calculate individual areas
    area1 = (ax2 - ax1) * (ay2 - ay1)
    area2 = (bx2 - bx1) * (by2 - by1)

    # Calculate overlap
    overlap_x = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_y = max(0, min(ay2, by2) - max(ay1, by1))
    overlap = overlap_x * overlap_y

    return area1 + area2 - overlap
```

### Complexity
- **Time**: O(1)
- **Space**: O(1)

### Edge Cases
- No overlap → sum of areas
- One inside other → area of larger
- Edge touching only → no overlap (line has no area)
- Identical rectangles → single rectangle area

---

## Problem 6: Largest Number (LC #179) - Medium

- [LeetCode](https://leetcode.com/problems/largest-number/)

### Problem Statement
Arrange numbers to form largest number.

### Examples
```
Input: nums = [3,30,34,5,9]
Output: "9534330"
```

### Intuition Development
```
CUSTOM COMPARATOR:
Compare a+b vs b+a as strings.

nums = [3, 30, 34, 5, 9]

Compare 3 and 30:
  "3" + "30" = "330"
  "30" + "3" = "303"
  "330" > "303" → 3 comes first

Compare 3 and 34:
  "3" + "34" = "334"
  "34" + "3" = "343"
  "343" > "334" → 34 comes first

Sorted order: [9, 5, 34, 3, 30]
Result: "9534330"
```

### Video Explanation
- [NeetCode - Largest Number](https://www.youtube.com/watch?v=WDx6Y4i4xJ8)

### Solution
```python
from functools import cmp_to_key

def largestNumber(nums: list[int]) -> str:
    """
    Arrange numbers to form largest number.

    Strategy:
    - Custom comparator: compare a+b vs b+a as strings
    - Sort in descending order of this comparison
    """
    # Convert to strings
    strs = [str(num) for num in nums]

    # Custom comparator
    def compare(a, b):
        if a + b > b + a:
            return -1
        elif a + b < b + a:
            return 1
        else:
            return 0

    strs.sort(key=cmp_to_key(compare))

    # Handle all zeros case
    if strs[0] == '0':
        return '0'

    return ''.join(strs)
```

### Complexity
- **Time**: O(n log n × k) where k = avg digit count
- **Space**: O(n)

### Edge Cases
- All zeros → "0"
- Single number → that number as string
- Same digits different lengths (3 vs 30)
- Already optimal order

---

## Problem 7: Valid Number (LC #65) - Hard

- [LeetCode](https://leetcode.com/problems/valid-number/)

### Problem Statement
Check if string is valid number.

### Examples
```
Valid: "2", "0089", "-0.1", "+3.14", "4.", ".9", "2e10", "-90E3"
Invalid: "abc", "1a", "e3", "99e2.5", "--6", "95a54e53"
```

### Intuition Development
```
STATE MACHINE APPROACH:
Track what we've seen: digit, dot, e/E, sign

VALID PATTERNS:
- Integer: [sign] digits
- Decimal: [sign] digits . [digits] OR [sign] . digits
- Exponent: (integer OR decimal) e/E integer

Rules:
- Sign: only at start or right after e/E
- Dot: only before e/E, only once
- e/E: must have digit before, must have digit after
- At least one digit required
```

### Video Explanation
- [NeetCode - Valid Number](https://www.youtube.com/watch?v=chwsRCGDyNg)

### Solution
```python
def isNumber(s: str) -> bool:
    """
    Validate number string using state machine.

    Valid patterns:
    - Integer: [sign] digits
    - Decimal: [sign] digits . [digits] OR [sign] . digits
    - Exponent: (integer OR decimal) e/E integer
    """
    s = s.strip()
    if not s:
        return False

    seen_digit = False
    seen_dot = False
    seen_e = False

    for i, char in enumerate(s):
        if char.isdigit():
            seen_digit = True

        elif char in '+-':
            # Sign must be first or right after e/E
            if i > 0 and s[i - 1].lower() != 'e':
                return False

        elif char == '.':
            # Dot can't appear after e or another dot
            if seen_dot or seen_e:
                return False
            seen_dot = True

        elif char.lower() == 'e':
            # e must come after digit, and only once
            if seen_e or not seen_digit:
                return False
            seen_e = True
            seen_digit = False  # Need digit after e

        else:
            return False

    return seen_digit
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- Leading/trailing spaces
- Just a dot "."
- Just a sign "+" or "-"
- Just "e" or "E"
- Multiple e's

---

## Problem 8: Multiply Strings (LC #43) - Medium

- [LeetCode](https://leetcode.com/problems/multiply-strings/)

### Problem Statement
Multiply two numbers represented as strings.

### Examples
```
Input: num1 = "123", num2 = "456"
Output: "56088"
```

### Intuition Development
```
GRADE SCHOOL MULTIPLICATION:
      1 2 3
    ×   4 5 6
    ---------
      7 3 8   (123 × 6)
    6 1 5     (123 × 5, shifted)
  4 9 2       (123 × 4, shifted)
  -----------
  5 6 0 8 8

POSITION FORMULA:
digit1[i] × digit2[j] contributes to result[i + j + 1] and result[i + j]

i=2, j=2: 3 × 6 = 18
  result[5] += 8, result[4] += 1 (carry)
```

### Video Explanation
- [NeetCode - Multiply Strings](https://www.youtube.com/watch?v=1vZswirL8Y8)

### Solution
```python
def multiply(num1: str, num2: str) -> str:
    """
    Multiply two numbers as strings.

    Strategy:
    - Simulate grade school multiplication
    - Result[i + j + 1] += digit1[i] * digit2[j]
    """
    if num1 == "0" or num2 == "0":
        return "0"

    m, n = len(num1), len(num2)
    result = [0] * (m + n)

    # Multiply each digit
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            product = int(num1[i]) * int(num2[j])

            # Position in result
            p1, p2 = i + j, i + j + 1

            # Add to result
            total = product + result[p2]
            result[p2] = total % 10
            result[p1] += total // 10

    # Convert to string, skip leading zeros
    result_str = ''.join(map(str, result))
    return result_str.lstrip('0') or '0'
```

### Complexity
- **Time**: O(m × n)
- **Space**: O(m + n)

### Edge Cases
- Either number is "0" → "0"
- Single digit numbers
- Large numbers (hundreds of digits)
- Leading zeros in input

---

## Problem 9: Convex Hull (LC #587) - Hard

- [LeetCode](https://leetcode.com/problems/convex-hull/)

### Problem Statement
Find convex hull of points (outermost boundary).

### Examples
```
Input: trees = [[1,1],[2,2],[2,0],[2,4],[3,3],[4,2]]
Output: [[1,1],[2,0],[4,2],[3,3],[2,4]]
```

### Intuition Development
```
ANDREW'S MONOTONE CHAIN:
1. Sort points by x, then y
2. Build lower hull (left to right)
3. Build upper hull (right to left)
4. Combine

CROSS PRODUCT for turn direction:
  cross(O, A, B) = (A-O) × (B-O)
  > 0: left turn (counter-clockwise)
  < 0: right turn (clockwise)
  = 0: collinear

Lower hull: keep only right turns
Upper hull: keep only right turns (going backwards)

Points sorted: [(1,1), (2,0), (2,2), (2,4), (3,3), (4,2)]

Lower hull:
  Add (1,1), (2,0) → right turn ✓
  Add (2,2) → left turn, pop (2,0), add (2,2)
  Wait, this is wrong direction...

Actually: keep removing points that make left turn
```

### Video Explanation
- [NeetCode - Erect the Fence (Convex Hull)](https://www.youtube.com/watch?v=B2AJoQSZf4M)

### Solution
```python
def outerTrees(trees: list[list[int]]) -> list[list[int]]:
    """
    Find convex hull using Andrew's monotone chain algorithm.

    Strategy:
    - Sort points by x, then y
    - Build lower hull (left to right)
    - Build upper hull (right to left)
    - Combine, removing duplicates
    """
    def cross(o, a, b):
        """Cross product of vectors OA and OB."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Sort points
    points = sorted(map(tuple, trees))

    if len(points) <= 1:
        return [list(p) for p in points]

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) < 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) < 0:
            upper.pop()
        upper.append(p)

    # Combine (remove last point of each half to avoid duplication)
    hull = set(lower[:-1] + upper[:-1])

    return [list(p) for p in hull]
```

### Complexity
- **Time**: O(n log n)
- **Space**: O(n)

### Edge Cases
- Less than 3 points → return all
- All collinear → return all
- Duplicate points
- Square or regular polygon shapes

---

## Summary: Advanced Math Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Fraction to Decimal | Remainder tracking | O(d) |
| 2 | Basic Calculator | Stack for parentheses | O(n) |
| 3 | Integer to Words | Group by thousands | O(1) |
| 4 | Max Points on Line | Slope as GCD fraction | O(n²) |
| 5 | Rectangle Area | Inclusion-exclusion | O(1) |
| 6 | Largest Number | Custom comparator | O(n log n) |
| 7 | Valid Number | State machine | O(n) |
| 8 | Multiply Strings | Grade school algorithm | O(mn) |
| 9 | Convex Hull | Andrew's algorithm | O(n log n) |

---

## Practice More Problems

- [ ] LC #50 - Pow(x, n)
- [ ] LC #60 - Permutation Sequence
- [ ] LC #172 - Factorial Trailing Zeroes
- [ ] LC #233 - Number of Digit One
- [ ] LC #335 - Self Crossing
