# Math & Geometry - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE MATH & GEOMETRY                              │
│                                                                             │
│  MATH:                                                                      │
│  ✓ "Prime numbers"                                                          │
│  ✓ "GCD / LCM"                                                              │
│  ✓ "Factorial / Combinatorics"                                              │
│  ✓ "Modular arithmetic"                                                     │
│  ✓ "Number theory"                                                          │
│  ✓ "Power / Exponentiation"                                                 │
│  ✓ "Digit manipulation"                                                     │
│                                                                             │
│  GEOMETRY:                                                                  │
│  ✓ "Points on a plane"                                                      │
│  ✓ "Distance / Area"                                                        │
│  ✓ "Rectangle overlap"                                                      │
│  ✓ "Rotate matrix"                                                          │
│  ✓ "Spiral traversal"                                                       │
│  ✓ "Convex hull"                                                            │
│                                                                             │
│  Key insight: Many "array" problems have elegant O(1) math solutions!       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Basic arithmetic operations
- [ ] Modular arithmetic basics (a % b)
- [ ] 2D array traversal
- [ ] Coordinate systems (x, y)

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MATH & GEOMETRY MEMORY MAP                               │
│                                                                             │
│                    ┌─────────────────┐                                      │
│         ┌─────────│ MATH & GEOMETRY │─────────┐                             │
│         │         └─────────────────┘         │                             │
│         ▼                                     ▼                             │
│  ┌─────────────┐                       ┌─────────────┐                      │
│  │    MATH     │                       │  GEOMETRY   │                      │
│  └──────┬──────┘                       └──────┬──────┘                      │
│         │                                     │                             │
│    ┌────┴────┬────────┐              ┌────────┼────────┐                    │
│    ▼         ▼        ▼              ▼        ▼        ▼                    │
│ ┌──────┐ ┌──────┐ ┌──────┐    ┌──────┐ ┌──────┐ ┌──────┐                   │
│ │Number│ │Primes│ │Modular│   │Matrix│ │Points│ │Shapes│                   │
│ │Theory│ │ GCD  │ │ Arith │   │ Ops  │ │Lines │ │ Area │                   │
│ └──────┘ └──────┘ └──────┘    └──────┘ └──────┘ └──────┘                   │
│                                                                             │
│  Related Patterns:                                                          │
│  • Binary Search - For sqrt, finding values                                 │
│  • Bit Manipulation - For power of 2, parity                                │
│  • Arrays - Matrix problems overlap                                         │
│                                                                             │
│  When to combine:                                                           │
│  • Math + Binary Search: Finding sqrt, kth element                          │
│  • Math + DP: Counting problems, combinatorics                              │
│  • Geometry + Hash Map: Points on a line, duplicate detection               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MATH PROBLEM DECISION TREE                               │
│                                                                             │
│  Is it about prime numbers?                                                 │
│       │                                                                     │
│       ├── YES → Single prime check? → O(√n) trial division                  │
│       │         Multiple primes? → Sieve of Eratosthenes                    │
│       │                                                                     │
│       └── NO → Is it about GCD/LCM?                                         │
│                    │                                                        │
│                    ├── YES → Euclidean algorithm O(log n)                   │
│                    │                                                        │
│                    └── NO → Is it about exponentiation?                     │
│                                 │                                           │
│                                 ├── YES → Binary exponentiation O(log n)    │
│                                 │                                           │
│                                 └── NO → Is it about counting/combinations? │
│                                              │                              │
│                                              ├── YES → Use math formulas    │
│                                              │         or DP                │
│                                              │                              │
│                                              └── NO → Check for patterns    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                    GEOMETRY PROBLEM DECISION TREE                           │
│                                                                             │
│  Is it a matrix problem?                                                    │
│       │                                                                     │
│       ├── YES → Rotation? → Transpose + Reverse                             │
│       │         Spiral? → Boundary shrinking                                │
│       │         Set zeros? → Use first row/col as markers                   │
│       │                                                                     │
│       └── NO → Is it about points/lines?                                    │
│                    │                                                        │
│                    ├── YES → Distance? → Euclidean formula                  │
│                    │         Collinear? → Cross product = 0                 │
│                    │         Line through points? → Slope with GCD          │
│                    │                                                        │
│                    └── NO → Is it about rectangles/shapes?                  │
│                                 │                                           │
│                                 ├── YES → Overlap? → Check separation       │
│                                 │         Area? → Use formulas              │
│                                 │                                           │
│                                 └── NO → Analyze specific geometry          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Essential Math Functions

```python
import math

# GCD (Greatest Common Divisor)
def gcd(a: int, b: int) -> int:
    """
    Euclidean algorithm for GCD.

    Time: O(log(min(a, b)))
    """
    while b:
        a, b = b, a % b
    return a

# Or use built-in
math.gcd(12, 8)  # 4

# LCM (Least Common Multiple)
def lcm(a: int, b: int) -> int:
    """LCM = (a * b) / GCD(a, b)"""
    return a * b // math.gcd(a, b)

# Or use built-in (Python 3.9+)
math.lcm(12, 8)  # 24

# Check if prime
def is_prime(n: int) -> bool:
    """
    Check if n is prime.

    Time: O(√n)
    """
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Sieve of Eratosthenes
def sieve(n: int) -> list[bool]:
    """
    Find all primes up to n.

    Time: O(n log log n)
    Space: O(n)
    """
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Mark all multiples as non-prime
            for j in range(i*i, n + 1, i):
                is_prime[j] = False

    return is_prime

# Modular exponentiation
def mod_pow(base: int, exp: int, mod: int) -> int:
    """
    Calculate (base^exp) % mod efficiently.

    Time: O(log exp)
    """
    result = 1
    base %= mod

    while exp > 0:
        if exp & 1:  # If exp is odd
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod

    return result

# Or use built-in
pow(2, 10, 1000000007)  # 1024
```

---

## Common Math Problems

### Problem 1: Pow(x, n) (LC #50)

```python
def myPow(x: float, n: int) -> float:
    """
    Calculate x^n.

    Strategy: Binary exponentiation
    x^n = (x^2)^(n/2) if n is even
    x^n = x * x^(n-1) if n is odd

    Time: O(log n)
    Space: O(1)
    """
    if n < 0:
        x = 1 / x
        n = -n

    result = 1

    while n > 0:
        if n & 1:  # n is odd
            result *= x
        x *= x
        n >>= 1

    return result
```

### Problem 2: Sqrt(x) (LC #69)

```python
def mySqrt(x: int) -> int:
    """
    Integer square root using binary search.

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

    return right
```

### Problem 3: Count Primes (LC #204)

```python
def countPrimes(n: int) -> int:
    """
    Count primes less than n using Sieve of Eratosthenes.

    Time: O(n log log n)
    Space: O(n)
    """
    if n < 2:
        return 0

    # is_prime[i] = True if i is prime
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            # Mark multiples as non-prime
            for j in range(i*i, n, i):
                is_prime[j] = False

    return sum(is_prime)
```

### Problem 4: Happy Number (LC #202)

```python
def isHappy(n: int) -> bool:
    """
    Check if n is a happy number.

    Happy: Sum of squares of digits eventually reaches 1.
    Unhappy: Gets stuck in a cycle.

    Strategy: Floyd's cycle detection

    Time: O(log n)
    Space: O(1)
    """
    def get_next(num: int) -> int:
        """Sum of squares of digits."""
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total

    slow = n
    fast = get_next(n)

    while fast != 1 and slow != fast:
        slow = get_next(slow)
        fast = get_next(get_next(fast))

    return fast == 1
```

### Problem 5: Plus One (LC #66)

```python
def plusOne(digits: list[int]) -> list[int]:
    """
    Add one to number represented as array of digits.

    Time: O(n)
    Space: O(1) or O(n) if new digit needed
    """
    n = len(digits)

    # Start from rightmost digit
    for i in range(n - 1, -1, -1):
        if digits[i] < 9:
            digits[i] += 1
            return digits

        # Digit is 9, becomes 0, carry continues
        digits[i] = 0

    # All digits were 9 (e.g., 999 -> 1000)
    return [1] + digits
```

---

## Geometry Problems

### Problem 6: Rotate Image (LC #48)

```python
def rotate(matrix: list[list[int]]) -> None:
    """
    Rotate matrix 90 degrees clockwise in-place.

    Strategy: Transpose + Reverse each row

    Time: O(n²)
    Space: O(1)
    """
    n = len(matrix)

    # Step 1: Transpose (swap matrix[i][j] with matrix[j][i])
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Step 2: Reverse each row
    for row in matrix:
        row.reverse()
```

### Problem 7: Spiral Matrix (LC #54)

```python
def spiralOrder(matrix: list[list[int]]) -> list[int]:
    """
    Return elements in spiral order.

    Strategy: Define boundaries and shrink them.

    Time: O(m*n)
    Space: O(1) excluding output
    """
    if not matrix:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Traverse right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1

        # Traverse down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1

        # Traverse left (if rows remaining)
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1

        # Traverse up (if columns remaining)
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

    return result
```

### Problem 8: Set Matrix Zeroes (LC #73)

```python
def setZeroes(matrix: list[list[int]]) -> None:
    """
    Set entire row and column to 0 if element is 0.

    Strategy: Use first row and column as markers.

    Time: O(m*n)
    Space: O(1)
    """
    m, n = len(matrix), len(matrix[0])

    # Check if first row/column should be zeroed
    first_row_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))

    # Use first row/column as markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0

    # Zero out cells based on markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0

    # Zero out first row if needed
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0

    # Zero out first column if needed
    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0
```

### Problem 9: Rectangle Overlap (LC #836)

```python
def isRectangleOverlap(rec1: list[int], rec2: list[int]) -> bool:
    """
    Check if two rectangles overlap.

    Rectangles: [x1, y1, x2, y2] (bottom-left, top-right)

    Overlap if NOT (separated horizontally OR separated vertically)

    Time: O(1)
    Space: O(1)
    """
    # Check for no overlap conditions
    # rec1 is to the left of rec2
    if rec1[2] <= rec2[0]:
        return False
    # rec1 is to the right of rec2
    if rec1[0] >= rec2[2]:
        return False
    # rec1 is above rec2
    if rec1[1] >= rec2[3]:
        return False
    # rec1 is below rec2
    if rec1[3] <= rec2[1]:
        return False

    return True
```

### Problem 10: Valid Square (LC #593)

```python
def validSquare(p1: list[int], p2: list[int], p3: list[int], p4: list[int]) -> bool:
    """
    Check if four points form a valid square.

    Strategy: Calculate all distances.
    Square has 4 equal sides and 2 equal diagonals.

    Time: O(1)
    Space: O(1)
    """
    def dist_sq(a: list[int], b: list[int]) -> int:
        """Squared distance between two points."""
        return (a[0] - b[0])**2 + (a[1] - b[1])**2

    points = [p1, p2, p3, p4]
    distances = set()

    # Calculate all pairwise distances
    for i in range(4):
        for j in range(i + 1, 4):
            d = dist_sq(points[i], points[j])
            if d == 0:
                return False  # Two points are same
            distances.add(d)

    # Valid square: exactly 2 distinct distances (side and diagonal)
    # And 4 sides + 2 diagonals
    return len(distances) == 2
```

---

## Useful Math Formulas

```python
# Sum of first n natural numbers
def sum_n(n):
    return n * (n + 1) // 2

# Sum of squares
def sum_squares(n):
    return n * (n + 1) * (2*n + 1) // 6

# Combinations C(n, k)
from math import comb
comb(5, 2)  # 10

# Factorial
from math import factorial
factorial(5)  # 120

# Distance between points
def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

# Area of triangle (using coordinates)
def triangle_area(p1, p2, p3):
    return abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2)
```

---

---

## Complexity Analysis

| Algorithm | Time | Space | Use Case |
|-----------|------|-------|----------|
| GCD (Euclidean) | O(log min(a,b)) | O(1) | Find GCD of two numbers |
| Sieve of Eratosthenes | O(n log log n) | O(n) | Find all primes up to n |
| Prime check | O(√n) | O(1) | Check single number |
| Binary exponentiation | O(log n) | O(1) | Calculate x^n efficiently |
| Modular inverse | O(log m) | O(1) | Find a^(-1) mod m |
| Matrix rotation | O(n²) | O(1) | Rotate in-place |
| Spiral traversal | O(m×n) | O(1) | Visit all cells |
| Point distance | O(1) | O(1) | Euclidean distance |
| Triangle area | O(1) | O(1) | Shoelace formula |

---

## Common Mistakes

```python
# ❌ WRONG: Integer overflow in multiplication
def lcm_wrong(a, b):
    return (a * b) // gcd(a, b)  # a * b might overflow in other languages

# ✅ CORRECT: Divide first to avoid overflow
def lcm_correct(a, b):
    return a // gcd(a, b) * b


# ❌ WRONG: Not handling negative numbers in modular arithmetic
result = (-5) % 3  # In Python: 1 (correct)
# In C++/Java: -2 (wrong for many problems!)

# ✅ CORRECT: Ensure positive modulo
def safe_mod(a, m):
    return ((a % m) + m) % m


# ❌ WRONG: Using floating point for exact comparisons
def is_on_line_wrong(p1, p2, p3):
    slope1 = (p2[1] - p1[1]) / (p2[0] - p1[0])  # Division by zero risk!
    slope2 = (p3[1] - p1[1]) / (p3[0] - p1[0])  # Floating point errors!
    return slope1 == slope2

# ✅ CORRECT: Use cross product (integer math)
def is_on_line_correct(p1, p2, p3):
    # Cross product: (p2-p1) × (p3-p1) = 0 means collinear
    return (p2[0]-p1[0]) * (p3[1]-p1[1]) == (p2[1]-p1[1]) * (p3[0]-p1[0])


# ❌ WRONG: Checking prime with even numbers
def is_prime_wrong(n):
    for i in range(2, int(n**0.5) + 1):  # Checks even numbers unnecessarily
        if n % i == 0:
            return False
    return True

# ✅ CORRECT: Skip even numbers after 2
def is_prime_correct(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):  # Only odd numbers
        if n % i == 0:
            return False
    return True


# ❌ WRONG: Sieve starting from wrong index
def sieve_wrong(n):
    is_prime = [True] * (n + 1)
    for i in range(2, n + 1):  # Checking all numbers
        if is_prime[i]:
            for j in range(i + i, n + 1, i):  # Starting from 2*i
                is_prime[j] = False

# ✅ CORRECT: Start from i*i (smaller multiples already marked)
def sieve_correct(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):  # Only up to sqrt(n)
        if is_prime[i]:
            for j in range(i*i, n + 1, i):  # Start from i*i
                is_prime[j] = False
    return is_prime


# ❌ WRONG: Matrix rotation creating new matrix
def rotate_wrong(matrix):
    n = len(matrix)
    result = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            result[j][n-1-i] = matrix[i][j]
    return result  # O(n²) extra space!

# ✅ CORRECT: In-place rotation
def rotate_correct(matrix):
    n = len(matrix)
    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
    # Reverse rows
    for row in matrix:
        row.reverse()
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"For this prime counting problem, I'll use the Sieve of Eratosthenes.
The key insight is that we can mark all multiples of each prime as
composite, starting from i² since smaller multiples were already
marked by smaller primes. This gives us O(n log log n) time."
```

### 2. What Interviewers Look For
- **Mathematical insight**: Can you find the O(1) formula?
- **Edge cases**: Zero, negative numbers, overflow
- **Optimization**: Using integer math over floating point
- **Space efficiency**: In-place operations when possible

### 3. Common Follow-up Questions
- "Can you do this without extra space?" → In-place matrix operations
- "What about very large numbers?" → Modular arithmetic
- "Can you optimize further?" → Look for mathematical formulas
- "What if precision matters?" → Use integer math, avoid floats

### 4. Key Formulas to Memorize
```python
# Sum of 1 to n
sum_n = n * (n + 1) // 2

# Sum of squares
sum_sq = n * (n + 1) * (2*n + 1) // 6

# Arithmetic progression sum
ap_sum = n * (first + last) // 2

# Geometric progression sum (r != 1)
gp_sum = first * (r**n - 1) // (r - 1)

# Combinations C(n, k)
from math import comb
comb(n, k)

# Distance between points
dist = ((x2-x1)**2 + (y2-y1)**2)**0.5

# Triangle area (Shoelace)
area = abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)) / 2
```

---

## Summary

| Problem | Key Technique | Time |
|---------|---------------|------|
| Pow(x, n) | Binary exponentiation | O(log n) |
| Sqrt(x) | Binary search | O(log x) |
| Count Primes | Sieve of Eratosthenes | O(n log log n) |
| Happy Number | Floyd's cycle detection | O(log n) |
| Rotate Image | Transpose + reverse | O(n²) |
| Spiral Matrix | Boundary shrinking | O(mn) |
| Set Matrix Zeroes | First row/col as markers | O(mn) |

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
