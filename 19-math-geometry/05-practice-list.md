# Math & Geometry - Complete Practice List

## Organized by Topic

### Topic 1: Basic Math

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 7 | [Reverse Integer](https://leetcode.com/problems/reverse-integer/) | Medium | Digit extraction |
| 9 | [Palindrome Number](https://leetcode.com/problems/palindrome-number/) | Easy | Reverse half |
| 66 | [Plus One](https://leetcode.com/problems/plus-one/) | Easy | Carry handling |
| 67 | [Add Binary](https://leetcode.com/problems/add-binary/) | Easy | Bit addition |
| 258 | [Add Digits](https://leetcode.com/problems/add-digits/) | Easy | Digital root |
| 415 | [Add Strings](https://leetcode.com/problems/add-strings/) | Easy | Digit by digit |

### Topic 2: Number Theory

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 204 | [Count Primes](https://leetcode.com/problems/count-primes/) | Medium | Sieve of Eratosthenes |
| 1492 | [The kth Factor of n](https://leetcode.com/problems/the-kth-factor-of-n/) | Medium | Factor enumeration |
| 1201 | [Ugly Number III](https://leetcode.com/problems/ugly-number-iii/) | Medium | Binary search + LCM |
| 372 | [Super Pow](https://leetcode.com/problems/super-pow/) | Medium | Modular exponentiation |

### Topic 3: GCD/LCM

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 1071 | [GCD of Strings](https://leetcode.com/problems/greatest-common-divisor-of-strings/) | Easy | Euclidean algorithm |
| 914 | [X of a Kind in a Deck](https://leetcode.com/problems/x-of-a-kind-in-a-deck-of-cards/) | Easy | GCD of counts |
| 365 | [Water and Jug Problem](https://leetcode.com/problems/water-and-jug-problem/) | Medium | Bezout's identity |

### Topic 4: Matrix Operations

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 48 | [Rotate Image](https://leetcode.com/problems/rotate-image/) | Medium | Transpose + reverse |
| 54 | [Spiral Matrix](https://leetcode.com/problems/spiral-matrix/) | Medium | Layer by layer |
| 59 | [Spiral Matrix II](https://leetcode.com/problems/spiral-matrix-ii/) | Medium | Fill spiral |
| 73 | [Set Matrix Zeroes](https://leetcode.com/problems/set-matrix-zeroes/) | Medium | First row/col as markers |
| 289 | [Game of Life](https://leetcode.com/problems/game-of-life/) | Medium | In-place with encoding |

### Topic 5: Geometry

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 149 | [Max Points on a Line](https://leetcode.com/problems/max-points-on-a-line/) | Hard | Slope counting |
| 223 | [Rectangle Area](https://leetcode.com/problems/rectangle-area/) | Medium | Inclusion-exclusion |
| 836 | [Rectangle Overlap](https://leetcode.com/problems/rectangle-overlap/) | Easy | Interval intersection |
| 593 | [Valid Square](https://leetcode.com/problems/valid-square/) | Medium | Distance checking |

### Topic 6: Random/Probability

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 384 | [Shuffle an Array](https://leetcode.com/problems/shuffle-an-array/) | Medium | Fisher-Yates |
| 398 | [Random Pick Index](https://leetcode.com/problems/random-pick-index/) | Medium | Reservoir sampling |
| 470 | [Implement Rand10() Using Rand7()](https://leetcode.com/problems/implement-rand10-using-rand7/) | Medium | Rejection sampling |
| 528 | [Random Pick with Weight](https://leetcode.com/problems/random-pick-with-weight/) | Medium | Prefix sum + binary search |

### Topic 7: Combinatorics

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 62 | [Unique Paths](https://leetcode.com/problems/unique-paths/) | Medium | C(m+n-2, m-1) or DP |
| 96 | [Unique BSTs](https://leetcode.com/problems/unique-binary-search-trees/) | Medium | Catalan numbers |
| 172 | [Factorial Trailing Zeroes](https://leetcode.com/problems/factorial-trailing-zeroes/) | Medium | Count factors of 5 |

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MATH & GEOMETRY PATTERNS                               │
│                                                                             │
│  GCD (Euclidean Algorithm):                                                 │
│  gcd(48, 18):                                                               │
│                                                                             │
│  48 = 18 × 2 + 12    →  gcd(48, 18) = gcd(18, 12)                          │
│  18 = 12 × 1 + 6     →  gcd(18, 12) = gcd(12, 6)                           │
│  12 = 6 × 2 + 0      →  gcd(12, 6) = 6  ✓                                  │
│                                                                             │
│  When remainder = 0, the divisor is the GCD                                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MATRIX ROTATION (90° Clockwise):                                           │
│                                                                             │
│  Original:     Transpose:      Reverse Rows:                                │
│  [1, 2, 3]     [1, 4, 7]       [7, 4, 1]                                    │
│  [4, 5, 6]  →  [2, 5, 8]   →   [8, 5, 2]                                    │
│  [7, 8, 9]     [3, 6, 9]       [9, 6, 3]                                    │
│                                                                             │
│  Step 1: Swap matrix[i][j] with matrix[j][i]                                │
│  Step 2: Reverse each row                                                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SPIRAL MATRIX TRAVERSAL:                                                   │
│                                                                             │
│  ┌───┬───┬───┬───┐                                                          │
│  │ 1 │ 2 │ 3 │ 4 │  →  Right: 1,2,3,4                                       │
│  ├───┼───┼───┼───┤                                                          │
│  │12 │   │   │ 5 │  ↓  Down: 5,6,7,8                                        │
│  ├───┼───┼───┼───┤                                                          │
│  │11 │   │   │ 6 │  ←  Left: 9,10,11,12                                     │
│  ├───┼───┼───┼───┤                                                          │
│  │10 │ 9 │ 8 │ 7 │  ↑  Up: (shrink boundaries, repeat)                      │
│  └───┴───┴───┴───┘                                                          │
│                                                                             │
│  Maintain: top, bottom, left, right boundaries                              │
│  Shrink after each direction traversal                                      │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SIEVE OF ERATOSTHENES (Count Primes < 30):                                 │
│                                                                             │
│  [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, ...]                              │
│      ↓                                                                      │
│  Mark multiples of 2: [2, 3, X, 5, X, 7, X, 9, X, 11, X, 13, ...]           │
│      ↓                                                                      │
│  Mark multiples of 3: [2, 3, X, 5, X, 7, X, X, X, 11, X, 13, ...]           │
│      ↓                                                                      │
│  Continue until √n...                                                       │
│                                                                             │
│  Remaining unmarked = primes: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  POINTS ON A LINE (Slope Counting):                                         │
│                                                                             │
│       •(4,4)                                                                │
│      /                                                                      │
│     •(3,3)        slope = dy/dx = (3-1)/(3-1) = 1                           │
│    /                                                                        │
│   •(1,1)          Use GCD to normalize: (2,2) → (1,1)                       │
│                                                                             │
│  For each point, count other points with same slope                         │
│  Store slope as (dx/gcd, dy/gcd) to avoid floating point                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RECTANGLE OVERLAP:                                                         │
│                                                                             │
│  Rectangle 1: (x1, y1) to (x2, y2)                                          │
│  Rectangle 2: (x3, y3) to (x4, y4)                                          │
│                                                                             │
│  ┌─────────┐                                                                │
│  │    ┌────┼────┐    Overlap exists if:                                     │
│  │    │////│    │    x1 < x4 AND x3 < x2 (x-overlap)                        │
│  └────┼────┘    │    y1 < y4 AND y3 < y2 (y-overlap)                        │
│       └─────────┘                                                           │
│                                                                             │
│  No overlap if: x1 >= x4 OR x3 >= x2 OR y1 >= y4 OR y3 >= y2                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Essential Formulas

```python
# GCD (Euclidean Algorithm)
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# LCM
def lcm(a, b):
    return a * b // gcd(a, b)

# Modular Exponentiation
def mod_pow(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result

# Sieve of Eratosthenes
def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(n + 1) if is_prime[i]]

# Combinations C(n, k)
def combinations(n, k):
    if k > n - k:
        k = n - k

    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)

    return result
```

---

## Matrix Templates

### Rotate 90° Clockwise
```python
def rotate(matrix):
    """Rotate matrix 90° clockwise in-place."""
    n = len(matrix)

    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Reverse each row
    for row in matrix:
        row.reverse()
```

### Spiral Traversal
```python
def spiralOrder(matrix):
    """Return elements in spiral order."""
    if not matrix:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1

        # Down
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1

        # Left
        if top <= bottom:
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1

        # Up
        if left <= right:
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

    return result
```

### Set Matrix Zeroes (O(1) Space)
```python
def setZeroes(matrix):
    """Set row/col to zero if cell is zero."""
    m, n = len(matrix), len(matrix[0])
    first_row_zero = any(matrix[0][j] == 0 for j in range(n))
    first_col_zero = any(matrix[i][0] == 0 for i in range(m))

    # Use first row/col as markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][j] == 0:
                matrix[i][0] = 0
                matrix[0][j] = 0

    # Zero out based on markers
    for i in range(1, m):
        for j in range(1, n):
            if matrix[i][0] == 0 or matrix[0][j] == 0:
                matrix[i][j] = 0

    # Handle first row/col
    if first_row_zero:
        for j in range(n):
            matrix[0][j] = 0

    if first_col_zero:
        for i in range(m):
            matrix[i][0] = 0
```

---

## Geometry Helpers

```python
# Distance between two points
def distance(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# Cross product (for collinearity)
def cross_product(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

# Check if three points are collinear
def collinear(p1, p2, p3):
    return cross_product(p1, p2, p3) == 0

# Slope as fraction (avoid floating point)
def slope(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    if dx == 0:
        return (0, 1)  # Vertical
    if dy == 0:
        return (1, 0)  # Horizontal

    g = gcd(abs(dx), abs(dy))
    dx, dy = dx // g, dy // g

    if dx < 0:
        dx, dy = -dx, -dy

    return (dx, dy)
```

---

## Study Plan

### Week 1: Basic Math
- [ ] Reverse Integer
- [ ] Palindrome Number
- [ ] Plus One
- [ ] Add Binary
- [ ] Factorial Trailing Zeroes

### Week 2: Matrix
- [ ] Rotate Image
- [ ] Spiral Matrix
- [ ] Set Matrix Zeroes
- [ ] Game of Life

### Week 3: Advanced
- [ ] Count Primes
- [ ] Max Points on a Line
- [ ] Random Pick with Weight
- [ ] Unique Paths

---

## Common Mistakes

1. **Integer overflow**
   - Check bounds before operations
   - Use long or arbitrary precision

2. **Floating point precision**
   - Use fractions/GCD for exact comparison
   - Epsilon comparison when needed

3. **Off-by-one in matrix indices**
   - Careful with boundaries
   - Test with small examples

4. **Division by zero**
   - Check denominators
   - Handle vertical lines in slope

---

## Complexity Reference

| Algorithm | Time | Space |
|-----------|------|-------|
| GCD | O(log min(a,b)) | O(1) |
| Sieve | O(n log log n) | O(n) |
| Matrix rotation | O(n²) | O(1) |
| Spiral traversal | O(mn) | O(1) |
| Mod exponentiation | O(log exp) | O(1) |

