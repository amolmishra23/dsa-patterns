# Math & Geometry - Practice Problems

## Problem 1: Pow(x, n) (LC #50) - Medium

- [LeetCode](https://leetcode.com/problems/powx-n/)

### Problem Statement
Implement `pow(x, n)`, which calculates x raised to the power n.

### Examples
```
Input: x = 2.0, n = 10
Output: 1024.0

Input: x = 2.0, n = -2
Output: 0.25
```

### Video Explanation
- [NeetCode - Pow(x, n)](https://www.youtube.com/watch?v=g9YQyYi4IQQ)

### Intuition
```
Binary Exponentiation: O(log n) instead of O(n)!

Key insight:
- x^n = (x^2)^(n/2)     if n is even
- x^n = x * x^(n-1)     if n is odd

Visual: 2^10

        2^10 = (2^2)^5 = 4^5
        4^5 = 4 * 4^4 = 4 * (4^2)^2 = 4 * 16^2
        16^2 = (16^2)^1 = 256

        Result = 4 * 256 = 1024

Binary view of exponent:
        10 = 1010 (binary)
        2^10 = 2^8 * 2^2 = 256 * 4 = 1024

        We multiply when bit is 1, square always.
```

### Solution
```python
def myPow(x: float, n: int) -> float:
    """
    Calculate x^n using binary exponentiation.

    Key insight: x^n = (x^2)^(n/2) if n is even
                 x^n = x * x^(n-1) if n is odd

    Example: 2^10
    - 2^10 = (2^2)^5 = 4^5
    - 4^5 = 4 * 4^4 = 4 * (4^2)^2 = 4 * 16^2 = 4 * 256 = 1024

    Time: O(log n) - halving n each iteration
    Space: O(1)
    """
    # Handle negative exponent
    if n < 0:
        x = 1 / x
        n = -n

    result = 1

    while n > 0:
        # If n is odd, multiply result by x
        if n & 1:  # n % 2 == 1
            result *= x

        # Square x and halve n
        x *= x
        n >>= 1  # n //= 2

    return result
```

### Edge Cases
- n = 0 → return 1
- n < 0 → return 1/x^|n|
- x = 0 → return 0 (unless n < 0, then undefined)
- x = 1 → return 1
- Large n → binary exponentiation handles efficiently

---

## Problem 2: Sqrt(x) (LC #69) - Easy

- [LeetCode](https://leetcode.com/problems/sqrtx/)

### Problem Statement
Compute integer square root of x.

### Examples
```
Input: x = 8
Output: 2 (sqrt(8) ≈ 2.82, truncated to 2)

Input: x = 4
Output: 2
```

### Video Explanation
- [NeetCode - Sqrt(x)](https://www.youtube.com/watch?v=zdMhGxRWutQ)

### Intuition
```
Binary Search on answer space!

Find largest m where m² ≤ x.

Visual: x = 8

        Search space: [0, 8]

        mid = 4: 4² = 16 > 8 → search [0, 3]
        mid = 1: 1² = 1 ≤ 8 → ans = 1, search [2, 3]
        mid = 2: 2² = 4 ≤ 8 → ans = 2, search [3, 3]
        mid = 3: 3² = 9 > 8 → search [3, 2] (empty)

        Answer: 2

Alternative: Newton's Method (faster convergence)
        x_{n+1} = (x_n + a/x_n) / 2
```

### Solution
```python
def mySqrt(x: int) -> int:
    """
    Integer square root using binary search.

    Find largest integer m where m^2 <= x.

    Time: O(log x)
    Space: O(1)
    """
    if x < 2:
        return x

    left, right = 1, x // 2

    while left <= right:
        mid = (left + right) // 2
        square = mid * mid

        if square == x:
            return mid
        elif square < x:
            left = mid + 1
        else:
            right = mid - 1

    # right is the largest value where right^2 <= x
    return right


def mySqrt_newton(x: int) -> int:
    """
    Alternative: Newton's method.

    Iteratively improve guess: r = (r + x/r) / 2

    Time: O(log x)
    Space: O(1)
    """
    if x < 2:
        return x

    r = x
    while r * r > x:
        r = (r + x // r) // 2

    return r
```

### Edge Cases
- x = 0 → return 0
- x = 1 → return 1
- Perfect square → return exact root
- Large x → binary search handles
- x < 0 → invalid (imaginary result)

---

## Problem 3: Rotate Image (LC #48) - Medium

- [LeetCode](https://leetcode.com/problems/rotate-image/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Rotate n×n matrix 90 degrees clockwise in-place.

### Examples
```
Input: [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
```


### Intuition
```
Key insight: 90° clockwise rotation = Transpose + Reverse each row.

Original → Transpose → Reverse rows:
1 2 3    1 4 7    7 4 1
4 5 6 →  2 5 8 →  8 5 2
7 8 9    3 6 9    9 6 3

Why it works: Transpose swaps rows↔columns. Reversing rows
then mirrors horizontally, completing the 90° rotation.

Alternative: Rotate 4 elements at a time, layer by layer.
```

### Solution
```python
def rotate(matrix: list[list[int]]) -> None:
    """
    Rotate matrix 90° clockwise in-place.

    Strategy: Transpose + Reverse each row

    Original:     Transpose:    Reverse rows:
    1 2 3         1 4 7         7 4 1
    4 5 6    →    2 5 8    →    8 5 2
    7 8 9         3 6 9         9 6 3

    Time: O(n²)
    Space: O(1)
    """
    n = len(matrix)

    # Step 1: Transpose (swap matrix[i][j] with matrix[j][i])
    for i in range(n):
        for j in range(i + 1, n):  # Only upper triangle
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: Reverse each row
    for row in matrix:
        row.reverse()


def rotate_layer(matrix: list[list[int]]) -> None:
    """
    Alternative: Rotate layer by layer.

    For each layer, rotate 4 elements at a time.
    """
    n = len(matrix)

    # Process each layer from outside to inside
    for layer in range(n // 2):
        first = layer
        last = n - 1 - layer

        for i in range(first, last):
            offset = i - first

            # Save top
            top = matrix[first][i]

            # Left → Top
            matrix[first][i] = matrix[last - offset][first]

            # Bottom → Left
            matrix[last - offset][first] = matrix[last][last - offset]

            # Right → Bottom
            matrix[last][last - offset] = matrix[i][last]

            # Top → Right
            matrix[i][last] = top
```

### Edge Cases
- 1x1 matrix → no change needed
- 2x2 matrix → single swap cycle
- Empty matrix → no change
- Non-square matrix → not applicable (problem requires square)
- Large matrix → O(n²) operations

---

## Problem 4: Spiral Matrix (LC #54) - Medium

- [LeetCode](https://leetcode.com/problems/spiral-matrix/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Return elements of matrix in spiral order.

### Examples
```
Input: [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]
```


### Intuition
```
Key insight: Define 4 boundaries, shrink them after each traversal.

Boundaries: top, bottom, left, right
1. Go right along top row, then top++
2. Go down along right column, then right--
3. Go left along bottom row, then bottom--
4. Go up along left column, then left++

Repeat until boundaries cross. Handle edge cases where
matrix becomes single row or column mid-traversal.
```

### Solution
```python
def spiralOrder(matrix: list[list[int]]) -> list[int]:
    """
    Return elements in spiral order.

    Strategy: Define boundaries and shrink them.

    Boundaries:
    - top, bottom: row boundaries
    - left, right: column boundaries

    Direction order: right → down → left → up

    Time: O(m × n)
    Space: O(1) excluding output
    """
    if not matrix:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Traverse right along top row
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1

        # Traverse down along right column
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1

        # Traverse left along bottom row (if rows remain)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1

        # Traverse up along left column (if columns remain)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

    return result
```

### Edge Cases
- Empty matrix → return []
- 1x1 matrix → return single element
- Single row → return row as is
- Single column → return column as list
- Wide vs tall matrix → both work

---

## Problem 5: Set Matrix Zeroes (LC #73) - Medium

- [LeetCode](https://leetcode.com/problems/set-matrix-zeroes/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
If element is 0, set its entire row and column to 0.

### Examples
```
Input: [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
```


### Intuition
```
Key insight: Use first row/column as markers to achieve O(1) space.

Problem: Can't zero immediately (would affect other cells).
Solution: Mark which rows/columns need zeroing, then apply.

1. Check if first row/column originally have zeros
2. Use first row/column to mark other rows/columns
3. Zero cells based on markers (skip first row/col)
4. Finally zero first row/column if needed

This avoids O(m+n) extra space for separate marker arrays.
```

### Solution
```python
def setZeroes(matrix: list[list[int]]) -> None:
    """
    Set rows and columns to 0 where element is 0.

    Strategy: Use first row/column as markers.

    1. Check if first row/column need zeroing
    2. Use first row/column to mark other rows/columns
    3. Zero out based on markers
    4. Zero first row/column if needed

    Time: O(m × n)
    Space: O(1)
    """
    m, n = len(matrix), len(matrix[0])

    # Step 1: Check if first row/column should be zeroed
    first_row_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))

    # Step 2: Use first row/column as markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0  # Mark row
                matrix[0][j] = 0  # Mark column

    # Step 3: Zero out cells based on markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0

    # Step 4: Zero first row if needed
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0

    # Step 5: Zero first column if needed
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0
```

### Edge Cases
- No zeros → matrix unchanged
- All zeros → all stay zero
- Zero in first row/column → handle separately
- Single zero → affects one row and column
- Multiple zeros → union of affected rows/columns

---

## Problem 6: Happy Number (LC #202) - Easy

- [LeetCode](https://leetcode.com/problems/happy-number/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
A happy number eventually reaches 1 when replacing with sum of squares of digits.

### Examples
```
Input: n = 19
Output: true
19 → 1² + 9² = 82 → 8² + 2² = 68 → 6² + 8² = 100 → 1² + 0² + 0² = 1
```


### Intuition
```
Key insight: Either reaches 1, or enters a cycle. Use cycle detection.

The sequence of sum-of-squares is bounded (any number eventually
becomes small), so it must either reach 1 or cycle.

Use Floyd's cycle detection (fast/slow pointers):
- Slow: one step at a time
- Fast: two steps at a time
- If they meet at 1 → happy
- If they meet elsewhere → not happy

Alternative: Use HashSet to detect cycle (O(log n) space).
```

### Solution
```python
def isHappy(n: int) -> bool:
    """
    Check if n is a happy number using cycle detection.

    Strategy: Use Floyd's algorithm (fast/slow pointers).
    If cycle detected and not at 1, it's not happy.

    Time: O(log n) - sum of squares reduces number quickly
    Space: O(1)
    """
    def get_next(num: int) -> int:
        """Calculate sum of squares of digits."""
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total

    # Floyd's cycle detection
    slow = n
    fast = get_next(n)

    while fast != 1 and slow != fast:
        slow = get_next(slow)
        fast = get_next(get_next(fast))

    return fast == 1


def isHappy_set(n: int) -> bool:
    """
    Alternative: Use set to detect cycle.

    Time: O(log n)
    Space: O(log n) for seen set
    """
    def get_next(num):
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total

    seen = set()

    while n != 1 and n not in seen:
        seen.add(n)
        n = get_next(n)

    return n == 1
```

### Edge Cases
- n = 1 → True (1 → 1)
- n = 7 → True (7 → 49 → 97 → ... → 1)
- n = 2 → False (enters cycle)
- Large n → converges quickly
- Single digit → limited outcomes

---

## Problem 7: Count Primes (LC #204) - Medium

- [LeetCode](https://leetcode.com/problems/count-primes/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Count primes less than n.

### Examples
```
Input: n = 10
Output: 4 (primes: 2, 3, 5, 7)
```


### Intuition
```
Key insight: Sieve of Eratosthenes - mark multiples of each prime.

Instead of checking each number for primality O(√n), mark
all composite numbers in one pass.

For each prime p found:
- Mark p², p²+p, p²+2p, ... as composite
- Start from p² because smaller multiples already marked

Only check up to √n (larger primes' multiples already covered).
Time: O(n log log n), much faster than O(n√n) naive approach.
```

### Solution
```python
def countPrimes(n: int) -> int:
    """
    Count primes using Sieve of Eratosthenes.

    Strategy:
    1. Create boolean array is_prime[0..n-1]
    2. Mark 0, 1 as not prime
    3. For each prime p, mark multiples p², p²+p, ... as not prime
    4. Count remaining primes

    Optimization: Only check up to √n

    Time: O(n log log n)
    Space: O(n)
    """
    if n < 2:
        return 0

    # Initialize all as prime
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False

    # Sieve: mark multiples of each prime
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Mark all multiples starting from i²
            # (smaller multiples already marked by smaller primes)
            for j in range(i * i, n, i):
                is_prime[j] = False

    return sum(is_prime)
```

### Edge Cases
- n = 0 → return 0
- n = 1 → return 0
- n = 2 → return 0 (primes less than 2)
- n = 3 → return 1 (only 2)
- Large n → sieve is efficient

---

## Problem 8: Fizz Buzz (LC #412) - Easy

- [LeetCode](https://leetcode.com/problems/fizz-buzz/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Return array where: "FizzBuzz" if divisible by 3 and 5, "Fizz" if by 3, "Buzz" if by 5.


### Intuition
```
Key insight: Check divisibility in correct order (15 before 3 or 5).

Order matters:
- Check 15 first (divisible by both 3 AND 5)
- Then check 3 alone
- Then check 5 alone
- Otherwise, the number itself

For extensibility: use list of (divisor, string) pairs,
build string by concatenating matching rules.
```

### Solution
```python
def fizzBuzz(n: int) -> list[str]:
    """
    Classic FizzBuzz problem.

    Time: O(n)
    Space: O(1) excluding output
    """
    result = []

    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))

    return result


def fizzBuzz_extensible(n: int) -> list[str]:
    """
    More extensible version for adding more rules.
    """
    result = []

    # Rules: (divisor, string)
    rules = [(3, "Fizz"), (5, "Buzz")]

    for i in range(1, n + 1):
        s = ""
        for divisor, word in rules:
            if i % divisor == 0:
                s += word

        result.append(s if s else str(i))

    return result
```

### Edge Cases
- n = 0 → return []
- n = 1 → return ["1"]
- n = 15 → first FizzBuzz
- Large n → linear time
- Multiples of 3 and 5 → handle 15 first

---

## Problem 9: Plus One (LC #66) - Easy

- [LeetCode](https://leetcode.com/problems/plus-one/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Add one to number represented as array of digits.

### Examples
```
Input: digits = [1,2,3]
Output: [1,2,4]

Input: digits = [9,9,9]
Output: [1,0,0,0]
```


### Intuition
```
Key insight: Handle carry from right to left, stop when no carry.

Most cases: rightmost digit < 9, just increment and return.
Edge case: digit = 9 becomes 0, carry propagates left.

Optimization: If digit < 9, increment and return immediately.
No need to process remaining digits.

Special case: All 9s (999 → 1000) - prepend 1 to result.
```

### Solution
```python
def plusOne(digits: list[int]) -> list[int]:
    """
    Add one to number represented as digit array.

    Strategy: Start from rightmost digit, handle carry.

    Time: O(n)
    Space: O(1) or O(n) if new digit needed
    """
    n = len(digits)

    # Start from rightmost digit
    for i in range(n - 1, -1, -1):
        if digits[i] < 9:
            # No carry needed
            digits[i] += 1
            return digits

        # Digit is 9, becomes 0, continue with carry
        digits[i] = 0

    # All digits were 9 (e.g., 999 → 1000)
    return [1] + digits
```

### Edge Cases
- [0] → [1]
- [9] → [1, 0]
- [9, 9, 9] → [1, 0, 0, 0]
- [1, 2, 3] → [1, 2, 4]
- Single digit < 9 → just increment

---

## Problem 10: Excel Sheet Column Number (LC #171) - Easy

- [LeetCode](https://leetcode.com/problems/excel-sheet-column-number/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Convert Excel column title to number.

### Examples
```
Input: columnTitle = "A"
Output: 1

Input: columnTitle = "AB"
Output: 28

Input: columnTitle = "ZY"
Output: 701
```


### Intuition
```
Key insight: Base-26 conversion where A=1, B=2, ..., Z=26.

Like reading a base-26 number left to right:
- "AB" = A*26 + B = 1*26 + 2 = 28
- "ZY" = Z*26 + Y = 26*26 + 25 = 701

For each character: result = result * 26 + char_value
where char_value = ord(char) - ord('A') + 1

Note: This is 1-indexed (A=1 not 0), which affects
the reverse conversion (number → title).
```

### Solution
```python
def titleToNumber(columnTitle: str) -> int:
    """
    Convert Excel column title to number.

    This is base-26 conversion where A=1, B=2, ..., Z=26.

    Example: "AB" = 1*26 + 2 = 28

    Time: O(n)
    Space: O(1)
    """
    result = 0

    for char in columnTitle:
        # Convert char to value (A=1, B=2, ..., Z=26)
        value = ord(char) - ord('A') + 1

        # Shift previous result and add new value
        result = result * 26 + value

    return result


def convertToTitle(columnNumber: int) -> str:
    """
    Reverse: Convert number to Excel column title (LC #168).

    Note: This is 1-indexed base-26, so we subtract 1 before mod.
    """
    result = []

    while columnNumber > 0:
        columnNumber -= 1  # Adjust for 1-indexed
        result.append(chr(columnNumber % 26 + ord('A')))
        columnNumber //= 26

    return ''.join(reversed(result))
```

### Edge Cases
- "A" → 1
- "Z" → 26
- "AA" → 27
- "AZ" → 52
- "ZZ" → 702

---

## Problem 11: Rectangle Overlap (LC #836) - Easy

- [LeetCode](https://leetcode.com/problems/rectangle-overlap/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Check if two axis-aligned rectangles overlap.


### Intuition
```
Key insight: Check for NON-overlap, then negate.

Two rectangles DON'T overlap if:
- One is completely left of the other (rec1.x2 <= rec2.x1)
- One is completely right of the other (rec1.x1 >= rec2.x2)
- One is completely above the other (rec1.y1 >= rec2.y2)
- One is completely below the other (rec1.y2 <= rec2.y1)

Overlap = NOT (any of the above)

Note: Use < not <= for strict overlap (touching edges don't count).
```

### Solution
```python
def isRectangleOverlap(rec1: list[int], rec2: list[int]) -> bool:
    """
    Check if two rectangles overlap.

    Rectangles given as [x1, y1, x2, y2] (bottom-left, top-right).

    Strategy: Check if NOT overlapping, then negate.

    No overlap if:
    - rec1 is completely left of rec2: rec1[2] <= rec2[0]
    - rec1 is completely right of rec2: rec1[0] >= rec2[2]
    - rec1 is completely above rec2: rec1[1] >= rec2[3]
    - rec1 is completely below rec2: rec1[3] <= rec2[1]

    Time: O(1)
    Space: O(1)
    """
    # Check for no overlap conditions
    if rec1[2] <= rec2[0]:  # rec1 left of rec2
        return False
    if rec1[0] >= rec2[2]:  # rec1 right of rec2
        return False
    if rec1[1] >= rec2[3]:  # rec1 above rec2
        return False
    if rec1[3] <= rec2[1]:  # rec1 below rec2
        return False

    return True


def isRectangleOverlap_compact(rec1: list[int], rec2: list[int]) -> bool:
    """Compact version using overlap condition directly."""
    # Overlap exists if both x and y ranges overlap
    x_overlap = rec1[0] < rec2[2] and rec2[0] < rec1[2]
    y_overlap = rec1[1] < rec2[3] and rec2[1] < rec1[3]

    return x_overlap and y_overlap
```

### Edge Cases
- Same rectangle → True (overlaps with itself)
- Touching edges only → False (not overlapping)
- One inside other → True
- Zero-area rectangle (line) → False
- Negative coordinates → still works

---

## Summary: Math & Geometry Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Pow(x, n) | Binary exponentiation | O(log n) |
| 2 | Sqrt(x) | Binary search | O(log x) |
| 3 | Rotate Image | Transpose + reverse | O(n²) |
| 4 | Spiral Matrix | Boundary shrinking | O(mn) |
| 5 | Set Matrix Zeroes | First row/col markers | O(mn) |
| 6 | Happy Number | Floyd's cycle detection | O(log n) |
| 7 | Count Primes | Sieve of Eratosthenes | O(n log log n) |
| 8 | Fizz Buzz | Modulo operations | O(n) |
| 9 | Plus One | Right-to-left carry | O(n) |
| 10 | Excel Column | Base-26 conversion | O(n) |
| 11 | Rectangle Overlap | Boundary comparison | O(1) |

---

## Practice More Problems

- [ ] LC #7 - Reverse Integer
- [ ] LC #9 - Palindrome Number
- [ ] LC #12 - Integer to Roman
- [ ] LC #13 - Roman to Integer
- [ ] LC #223 - Rectangle Area

