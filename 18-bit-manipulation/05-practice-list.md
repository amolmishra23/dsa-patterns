# Bit Manipulation - Complete Practice List

## Organized by Pattern and Difficulty

### Pattern 1: XOR Properties

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 136 | [Single Number](https://leetcode.com/problems/single-number/) | Easy | XOR all elements |
| 137 | [Single Number II](https://leetcode.com/problems/single-number-ii/) | Medium | Bit counting |
| 260 | [Single Number III](https://leetcode.com/problems/single-number-iii/) | Medium | XOR + rightmost bit |
| 268 | [Missing Number](https://leetcode.com/problems/missing-number/) | Easy | XOR with indices |
| 287 | [Find the Duplicate](https://leetcode.com/problems/find-the-duplicate-number/) | Medium | Floyd's or bit count |
| 421 | [Maximum XOR of Two Numbers](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/) | Medium | Trie or prefix set |

### Pattern 2: Bit Counting

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 191 | [Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/) | Easy | n & (n-1) trick |
| 338 | [Counting Bits](https://leetcode.com/problems/counting-bits/) | Easy | DP with bit pattern |
| 461 | [Hamming Distance](https://leetcode.com/problems/hamming-distance/) | Easy | XOR + count bits |
| 477 | [Total Hamming Distance](https://leetcode.com/problems/total-hamming-distance/) | Medium | Count per position |

### Pattern 3: Bit Manipulation Basics

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 190 | [Reverse Bits](https://leetcode.com/problems/reverse-bits/) | Easy | Shift and OR |
| 231 | [Power of Two](https://leetcode.com/problems/power-of-two/) | Easy | n & (n-1) == 0 |
| 342 | [Power of Four](https://leetcode.com/problems/power-of-four/) | Easy | Power of 2 + mask |
| 371 | [Sum of Two Integers](https://leetcode.com/problems/sum-of-two-integers/) | Medium | XOR + AND carry |

### Pattern 4: Subsets/Combinations

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 78 | [Subsets](https://leetcode.com/problems/subsets/) | Medium | Bitmask enumeration |
| 784 | [Letter Case Permutation](https://leetcode.com/problems/letter-case-permutation/) | Medium | Bitmask for cases |
| 1239 | [Max Length of Concatenated String](https://leetcode.com/problems/maximum-length-of-a-concatenated-string-with-unique-characters/) | Medium | Bitmask + backtrack |

### Pattern 5: Advanced

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 201 | [Bitwise AND of Numbers Range](https://leetcode.com/problems/bitwise-and-of-numbers-range/) | Medium | Common prefix |
| 318 | [Maximum Product of Word Lengths](https://leetcode.com/problems/maximum-product-of-word-lengths/) | Medium | Char bitmask |
| 393 | [UTF-8 Validation](https://leetcode.com/problems/utf-8-validation/) | Medium | Bit pattern check |
| 1310 | [XOR Queries of a Subarray](https://leetcode.com/problems/xor-queries-of-a-subarray/) | Medium | Prefix XOR |

---

## Essential Bit Operations

```python
# Get i-th bit
def get_bit(n, i):
    return (n >> i) & 1

# Set i-th bit to 1
def set_bit(n, i):
    return n | (1 << i)

# Clear i-th bit (set to 0)
def clear_bit(n, i):
    return n & ~(1 << i)

# Toggle i-th bit
def toggle_bit(n, i):
    return n ^ (1 << i)

# Check if power of 2
def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

# Count set bits (Brian Kernighan)
def count_bits(n):
    count = 0
    while n:
        n &= (n - 1)  # Clear lowest set bit
        count += 1
    return count

# Get lowest set bit
def lowest_set_bit(n):
    return n & (-n)

# Clear lowest set bit
def clear_lowest_bit(n):
    return n & (n - 1)
```

---

## Common Bit Tricks

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BIT MANIPULATION TRICKS                                  │
│                                                                             │
│  BASIC OPERATIONS:                                                          │
│  x & 1           → Check if odd                                             │
│  x & (x - 1)     → Clear lowest set bit                                     │
│  x & (-x)        → Isolate lowest set bit                                   │
│  x | (x + 1)     → Set lowest unset bit                                     │
│  x ^ x           → Always 0                                                 │
│  x ^ 0           → Always x                                                 │
│                                                                             │
│  USEFUL IDENTITIES:                                                         │
│  a ^ b ^ b = a   → XOR is self-inverse                                      │
│  a ^ b ^ a = b   → Swap without temp                                        │
│  ~x + 1 = -x     → Two's complement                                         │
│                                                                             │
│  POWER OF 2:                                                                │
│  x & (x - 1) == 0  → x is power of 2 (x > 0)                               │
│  x & (-x) == x     → x is power of 2 (x > 0)                               │
│                                                                             │
│  MASKING:                                                                   │
│  (1 << n) - 1    → Mask with n lowest bits set                             │
│  ~((1 << n) - 1) → Mask with n lowest bits clear                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem Templates

### 1. Single Number (XOR)
```python
def singleNumber(nums):
    """Find element appearing once (others appear twice)."""
    result = 0
    for num in nums:
        result ^= num
    return result
```

### 2. Single Number III (Two Unique)
```python
def singleNumber3(nums):
    """Find two elements appearing once."""
    # XOR all to get a ^ b
    xor_all = 0
    for num in nums:
        xor_all ^= num

    # Find rightmost set bit (a and b differ here)
    rightmost = xor_all & (-xor_all)

    # Divide into two groups
    a = b = 0
    for num in nums:
        if num & rightmost:
            a ^= num
        else:
            b ^= num

    return [a, b]
```

### 3. Counting Bits (DP)
```python
def countBits(n):
    """Count bits for 0 to n."""
    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        # dp[i] = dp[i with lowest bit cleared] + 1
        dp[i] = dp[i & (i - 1)] + 1

    return dp
```

### 4. Subsets using Bitmask
```python
def subsets(nums):
    """Generate all subsets using bitmask."""
    n = len(nums)
    result = []

    for mask in range(1 << n):
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)

    return result
```

### 5. Sum Without + (Bit Addition)
```python
def getSum(a, b):
    """Add two integers without + operator."""
    mask = 0xFFFFFFFF

    while b != 0:
        carry = (a & b) << 1
        a = (a ^ b) & mask
        b = carry & mask

    # Handle negative numbers in Python
    if a > 0x7FFFFFFF:
        a = ~(a ^ mask)

    return a
```

### 6. Prefix XOR for Range Queries
```python
def xorQueries(arr, queries):
    """Answer XOR queries on ranges."""
    # Build prefix XOR
    prefix = [0]
    for num in arr:
        prefix.append(prefix[-1] ^ num)

    # Answer queries
    result = []
    for left, right in queries:
        result.append(prefix[right + 1] ^ prefix[left])

    return result
```

---

## Binary Number Representation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BINARY REPRESENTATION                                    │
│                                                                             │
│  Positive numbers: Standard binary                                          │
│    5 = 0101                                                                 │
│   13 = 1101                                                                 │
│                                                                             │
│  Negative numbers: Two's complement                                         │
│   -1 = 1111...1111 (all 1s)                                                │
│   -5 = ~5 + 1 = 1111...1011                                                │
│                                                                             │
│  In Python:                                                                 │
│  - Integers have arbitrary precision                                        │
│  - Use mask (0xFFFFFFFF) for 32-bit behavior                               │
│  - bin(n) shows binary representation                                       │
│  - n.bit_length() gives number of bits needed                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] Single Number
- [ ] Number of 1 Bits
- [ ] Power of Two
- [ ] Reverse Bits
- [ ] Missing Number

### Week 2: Intermediate
- [ ] Counting Bits
- [ ] Hamming Distance
- [ ] Single Number II
- [ ] Single Number III
- [ ] Sum of Two Integers

### Week 3: Advanced
- [ ] Maximum XOR of Two Numbers
- [ ] Total Hamming Distance
- [ ] Bitwise AND of Numbers Range
- [ ] UTF-8 Validation

---

## Common Mistakes

1. **Integer overflow in other languages**
   - Python handles arbitrary precision
   - Use mask for 32-bit behavior when needed

2. **Negative number handling**
   - Two's complement representation
   - Check sign bit for negative

3. **Off-by-one in bit positions**
   - Bits are 0-indexed from right
   - 1 << 0 = 1, 1 << 1 = 2

4. **Forgetting parentheses**
   - Bitwise operators have low precedence
   - (n >> i) & 1, not n >> i & 1

---

## Complexity Reference

| Operation | Time | Space |
|-----------|------|-------|
| Get/Set/Clear bit | O(1) | O(1) |
| Count bits | O(log n) | O(1) |
| XOR all elements | O(n) | O(1) |
| Generate subsets | O(2^n) | O(2^n) |
| Prefix XOR | O(n) build, O(1) query | O(n) |

