# Bit Manipulation - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE BIT MANIPULATION                             │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Single number" / "Missing number"                                       │
│  ✓ "Power of two"                                                           │
│  ✓ "Count bits"                                                             │
│  ✓ "Subsets" (can use bitmask)                                              │
│  ✓ "XOR"                                                                    │
│  ✓ "Binary representation"                                                  │
│                                                                             │
│  Key insight: Bit operations are O(1) and can replace expensive operations  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Binary number system
- [ ] Bitwise operators (&, |, ^, ~, <<, >>)
- [ ] Two's complement for negative numbers

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BIT MANIPULATION MEMORY MAP                              │
│                                                                             │
│                    ┌───────────────┐                                        │
│         ┌─────────│BIT MANIPULATION│─────────┐                              │
│         │         └───────────────┘          │                              │
│         ▼                                    ▼                              │
│  ┌─────────────┐                      ┌─────────────┐                       │
│  │    XOR      │                      │  BITMASK    │                       │
│  │  PATTERNS   │                      │  PATTERNS   │                       │
│  └──────┬──────┘                      └──────┬──────┘                       │
│         │                                    │                              │
│    ┌────┴────┐                         ┌─────┴─────┐                        │
│    ▼         ▼                         ▼           ▼                        │
│ ┌──────┐ ┌──────┐                  ┌──────┐   ┌──────┐                     │
│ │Single│ │Missing│                 │Subsets│  │State │                     │
│ │Number│ │Number │                 │       │  │DP    │                     │
│ └──────┘ └──────┘                  └──────┘   └──────┘                     │
│                                                                             │
│  Key XOR Properties:                                                        │
│  • a ^ a = 0 (same numbers cancel)                                          │
│  • a ^ 0 = a (identity)                                                     │
│  • a ^ b ^ a = b (find unique)                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BIT MANIPULATION DECISION TREE                           │
│                                                                             │
│  Problem involves finding unique/missing element?                           │
│       │                                                                     │
│       ├── YES → XOR all elements                                            │
│       │         Single number: XOR gives the unique one                     │
│       │         Missing number: XOR [0,n] with array                        │
│       │                                                                     │
│       └── NO → Need to check/set/clear specific bits?                       │
│                    │                                                        │
│                    ├── Check bit i: (n >> i) & 1                            │
│                    │   Set bit i:   n | (1 << i)                            │
│                    │   Clear bit i: n & ~(1 << i)                           │
│                    │                                                        │
│                    └── Need to generate all subsets?                        │
│                                 │                                           │
│                                 ├── YES → Iterate 0 to 2^n - 1              │
│                                 │         Each number is a bitmask          │
│                                 │                                           │
│                                 └── Power of 2 check?                       │
│                                     n & (n-1) == 0 means power of 2         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### Bit Operators

```python
# AND (&): Both bits must be 1
5 & 3  # 101 & 011 = 001 = 1

# OR (|): At least one bit must be 1
5 | 3  # 101 | 011 = 111 = 7

# XOR (^): Bits must be different
5 ^ 3  # 101 ^ 011 = 110 = 6

# NOT (~): Flip all bits
~5     # ~101 = ...11111010 = -6 (two's complement)

# Left Shift (<<): Multiply by 2^n
5 << 1  # 101 << 1 = 1010 = 10

# Right Shift (>>): Divide by 2^n
5 >> 1  # 101 >> 1 = 10 = 2
```

### XOR Properties

```python
# XOR is the key to many bit manipulation problems!

# 1. a ^ a = 0 (number XOR itself = 0)
5 ^ 5  # 0

# 2. a ^ 0 = a (number XOR 0 = itself)
5 ^ 0  # 5

# 3. a ^ b ^ a = b (XOR is associative and commutative)
5 ^ 3 ^ 5  # 3

# 4. XOR of all numbers finds the "odd one out"
1 ^ 2 ^ 1  # 2 (1 appears twice, cancels out)
```

---

## Common Bit Tricks

```python
# Check if number is even
def is_even(n: int) -> bool:
    return (n & 1) == 0  # Last bit is 0 for even numbers

# Check if number is power of 2
def is_power_of_two(n: int) -> bool:
    # Power of 2 has exactly one 1-bit
    # n & (n-1) removes the lowest 1-bit
    return n > 0 and (n & (n - 1)) == 0

# Get i-th bit (0-indexed from right)
def get_bit(n: int, i: int) -> int:
    return (n >> i) & 1

# Set i-th bit to 1
def set_bit(n: int, i: int) -> int:
    return n | (1 << i)

# Clear i-th bit (set to 0)
def clear_bit(n: int, i: int) -> int:
    return n & ~(1 << i)

# Toggle i-th bit
def toggle_bit(n: int, i: int) -> int:
    return n ^ (1 << i)

# Get lowest set bit
def lowest_set_bit(n: int) -> int:
    return n & (-n)

# Clear lowest set bit
def clear_lowest_bit(n: int) -> int:
    return n & (n - 1)

# Count set bits (Brian Kernighan's algorithm)
def count_bits(n: int) -> int:
    count = 0
    while n:
        n &= (n - 1)  # Clear lowest set bit
        count += 1
    return count
```

---

## Visual: Bit Operations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BIT OPERATIONS VISUALIZATION                             │
│                                                                             │
│  n = 12 = 1100 in binary                                                    │
│                                                                             │
│  n & (n-1) = Clear lowest set bit                                           │
│    1100                                                                     │
│  & 1011 (n-1)                                                               │
│  ──────                                                                     │
│    1000 = 8                                                                 │
│                                                                             │
│  n & (-n) = Get lowest set bit                                              │
│    1100                                                                     │
│  & 0100 (-n in two's complement)                                            │
│  ──────                                                                     │
│    0100 = 4                                                                 │
│                                                                             │
│  Power of 2 check: n & (n-1) == 0                                           │
│    8 = 1000                                                                 │
│  & 7 = 0111                                                                 │
│  ──────                                                                     │
│        0000 = 0 ✓ (8 is power of 2)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Bit Problems

### Problem 1: Single Number (LC #136)

```python
def singleNumber(nums: list[int]) -> int:
    """
    Find element that appears once (others appear twice).

    Strategy: XOR all numbers
    - Pairs cancel out (a ^ a = 0)
    - Single number remains (a ^ 0 = a)

    Time: O(n)
    Space: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result
```

### Problem 2: Single Number II (LC #137)

```python
def singleNumber2(nums: list[int]) -> int:
    """
    Find element that appears once (others appear three times).

    Strategy: Count bits at each position
    - If count % 3 != 0, single number has 1 at that position

    Time: O(32n) = O(n)
    Space: O(1)
    """
    result = 0

    for i in range(32):
        # Count 1s at position i
        count = sum((num >> i) & 1 for num in nums)

        # If count % 3 != 0, single number has 1 here
        if count % 3:
            result |= (1 << i)

    # Handle negative numbers (Python's int is arbitrary precision)
    if result >= 2**31:
        result -= 2**32

    return result
```

### Problem 3: Single Number III (LC #260)

```python
def singleNumber3(nums: list[int]) -> list[int]:
    """
    Find two elements that appear once (others appear twice).

    Strategy:
    1. XOR all numbers → get xor of the two singles
    2. Find any bit where they differ (rightmost set bit)
    3. Divide numbers into two groups by that bit
    4. XOR each group to find each single

    Time: O(n)
    Space: O(1)
    """
    # Step 1: XOR all numbers
    xor_all = 0
    for num in nums:
        xor_all ^= num
    # xor_all = a ^ b where a, b are the two singles

    # Step 2: Find rightmost set bit (where a and b differ)
    diff_bit = xor_all & (-xor_all)

    # Step 3 & 4: Divide and XOR
    a = b = 0
    for num in nums:
        if num & diff_bit:
            a ^= num  # Group with bit set
        else:
            b ^= num  # Group without bit set

    return [a, b]
```

### Problem 4: Number of 1 Bits (LC #191)

```python
def hammingWeight(n: int) -> int:
    """
    Count number of 1 bits (Hamming weight).

    Strategy: Clear lowest set bit until n becomes 0

    Time: O(number of 1 bits)
    Space: O(1)
    """
    count = 0
    while n:
        n &= (n - 1)  # Clear lowest set bit
        count += 1
    return count


def hammingWeight_builtin(n: int) -> int:
    """Using Python's built-in."""
    return bin(n).count('1')
```

### Problem 5: Counting Bits (LC #338)

```python
def countBits(n: int) -> list[int]:
    """
    Count 1 bits for all numbers from 0 to n.

    Strategy (DP):
    - dp[i] = dp[i >> 1] + (i & 1)
    - i >> 1 removes last bit, (i & 1) checks if last bit is 1

    Time: O(n)
    Space: O(n)
    """
    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        # Number of 1s = number of 1s in i//2 + last bit
        dp[i] = dp[i >> 1] + (i & 1)

    return dp
```

### Problem 6: Reverse Bits (LC #190)

```python
def reverseBits(n: int) -> int:
    """
    Reverse bits of 32-bit unsigned integer.

    Strategy: Build result bit by bit

    Time: O(32) = O(1)
    Space: O(1)
    """
    result = 0

    for i in range(32):
        # Get i-th bit from right
        bit = (n >> i) & 1

        # Place it at (31-i) position from right
        result |= bit << (31 - i)

    return result
```

### Problem 7: Missing Number (LC #268)

```python
def missingNumber(nums: list[int]) -> int:
    """
    Find missing number in [0, n].

    Strategy: XOR all indices and values
    - Each number appears twice (once as index, once as value)
    - Except missing number (only as index)

    Time: O(n)
    Space: O(1)
    """
    n = len(nums)
    result = n  # Start with n (not in indices)

    for i, num in enumerate(nums):
        result ^= i ^ num

    return result


def missingNumber_math(nums: list[int]) -> int:
    """Alternative: Sum formula."""
    n = len(nums)
    expected = n * (n + 1) // 2
    actual = sum(nums)
    return expected - actual
```

### Problem 8: Power of Two (LC #231)

```python
def isPowerOfTwo(n: int) -> bool:
    """
    Check if n is power of 2.

    Power of 2 has exactly one 1-bit.
    n & (n-1) clears the lowest 1-bit.
    If result is 0, n had only one 1-bit.

    Time: O(1)
    Space: O(1)
    """
    return n > 0 and (n & (n - 1)) == 0
```

---

## Bitmask for Subsets

```python
def subsets_bitmask(nums: list[int]) -> list[list[int]]:
    """
    Generate all subsets using bitmask.

    For n elements, there are 2^n subsets.
    Each subset can be represented by n-bit number.
    Bit i = 1 means include nums[i].

    Time: O(n * 2^n)
    Space: O(n) per subset
    """
    n = len(nums)
    result = []

    # Iterate through all 2^n possible subsets
    for mask in range(1 << n):  # 0 to 2^n - 1
        subset = []

        for i in range(n):
            # Check if i-th bit is set
            if mask & (1 << i):
                subset.append(nums[i])

        result.append(subset)

    return result
```

---

## Complexity Analysis

All basic bit operations are O(1):
- AND, OR, XOR, NOT
- Left shift, Right shift
- Get/Set/Clear/Toggle bit

Counting bits: O(number of 1 bits) with Brian Kernighan

---

## Common Mistakes

```python
# ❌ WRONG: Forgetting Python's arbitrary precision integers
def reverse_bits_wrong(n):
    result = 0
    while n:
        result = (result << 1) | (n & 1)
        n >>= 1
    return result
    # Doesn't work for 32-bit constraint!

# ✅ CORRECT: Handle 32-bit explicitly
def reverse_bits_correct(n):
    result = 0
    for i in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


# ❌ WRONG: Not handling negative numbers
def single_number_2_wrong(nums):
    result = 0
    for i in range(32):
        count = sum((num >> i) & 1 for num in nums)
        if count % 3:
            result |= (1 << i)
    return result  # Wrong for negative result!

# ✅ CORRECT: Handle two's complement
def single_number_2_correct(nums):
    result = 0
    for i in range(32):
        count = sum((num >> i) & 1 for num in nums)
        if count % 3:
            result |= (1 << i)
    # Convert from unsigned to signed
    if result >= 2**31:
        result -= 2**32
    return result
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"I'll use XOR because it has the property that a ^ a = 0 and a ^ 0 = a.
When we XOR all elements, duplicates cancel out, leaving only the
unique element. This gives O(n) time and O(1) space."
```

### 2. What Interviewers Look For
- **XOR properties**: Know the key properties
- **Bit operations**: Check, set, clear, toggle bits
- **Edge cases**: Negative numbers, overflow

### 3. Common Follow-up Questions
- "What if there are two unique numbers?" → XOR + find differing bit
- "Can you do this without extra space?" → Bit manipulation often can
- "How to handle negative numbers?" → Be careful with right shift

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
