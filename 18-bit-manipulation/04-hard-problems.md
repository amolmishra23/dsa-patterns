# Bit Manipulation - Hard Problems

## Problem 1: Maximum XOR of Two Numbers (LC #421) - Hard

- [LeetCode](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/)

### Video Explanation
- [NeetCode - Maximum XOR of Two Numbers](https://www.youtube.com/watch?v=EIhAwfHubE8)

### Problem Statement
Find maximum XOR of any two numbers in array.


### Visual Intuition
```
Maximum XOR of Two Numbers
nums = [3, 10, 5, 25, 2, 8]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: To maximize XOR, we want OPPOSITE bits at each position
             Build Trie of binary representations, greedily pick opposites
═══════════════════════════════════════════════════════════════

Binary representations (5 bits):
  3  = 00011
  10 = 01010
  5  = 00101
  25 = 11001
  2  = 00010
  8  = 01000

Build Trie (MSB to LSB):
────────────────────────
              root
             /    \
            0      1
           / \      \
          0   1      1
         /|   |       \
        0 1   0        0
        | |   |         \
        1 1   1          0
        | |   |           \
        0 0   0            1
        ↓ ↓   ↓            ↓
        2 3   10          25

        (8 and 5 also in tree...)

Find Max XOR for 25 (11001):
────────────────────────────
  Start at root, greedily pick OPPOSITE bit at each level:

  Bit 4 (MSB): 25 has 1 → want 0 → go LEFT (0 branch) ✓
  Bit 3:       25 has 1 → want 0 → go LEFT (0 branch) ✓
  Bit 2:       25 has 0 → want 1 → go RIGHT if exists...
               Only 0 available → go LEFT (0 branch)
  Bit 1:       25 has 0 → want 1 → go RIGHT (1 branch) ✓
  Bit 0:       25 has 1 → want 0 → go LEFT (0 branch) ✓

  Path leads to: 00010 = 2
  XOR = 25 ^ 2 = 11001 ^ 00010 = 11011 = 27

  But wait, let's try 5:
  25 ^ 5 = 11001 ^ 00101 = 11100 = 28 ← MAXIMUM!

XOR Maximization:
─────────────────
  25 = 1 1 0 0 1
   5 = 0 0 1 0 1
  ─────────────
  XOR= 1 1 1 0 0 = 28 ★

Answer: 28

WHY THIS WORKS:
════════════════
● XOR gives 1 when bits differ, 0 when same
● Higher bits contribute more to value (2^4 > 2^0)
● Greedily picking opposite bits from MSB maximizes result
● Trie allows O(32) lookup for best partner
```

### Solution
```python
def findMaximumXOR(nums: list[int]) -> int:
    """
    Build Trie of binary representations.

    Strategy:
    - For each number, find number with most different bits
    - Use Trie to efficiently find opposite bits

    Time: O(n * 32)
    Space: O(n * 32)
    """
    # Build Trie
    root = {}

    for num in nums:
        node = root
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]

    max_xor = 0

    for num in nums:
        node = root
        curr_xor = 0

        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            # Try to go opposite direction
            opposite = 1 - bit

            if opposite in node:
                curr_xor |= (1 << i)
                node = node[opposite]
            else:
                node = node[bit]

        max_xor = max(max_xor, curr_xor)

    return max_xor
```

### Edge Cases
- Single element → XOR with itself = 0
- All same numbers → max XOR = 0
- Two elements → their XOR
- All zeros → max XOR = 0

---

## Problem 2: Minimum Number of K Consecutive Bit Flips (LC #995) - Hard

- [LeetCode](https://leetcode.com/problems/minimum-number-of-k-consecutive-bit-flips/)

### Video Explanation
- [NeetCode - Minimum Number of K Consecutive Bit Flips](https://www.youtube.com/watch?v=Fv3M9uO5ovU)

### Problem Statement
Minimum flips of k consecutive bits to make all 1s.


### Visual Intuition
```
Minimum Number of K Consecutive Bit Flips
nums = [0,1,0,1], k = 2

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Greedy flip at first 0, track flip effects with window
             Current bit = original XOR (flips affecting this position)
═══════════════════════════════════════════════════════════════

Step-by-Step:
─────────────
Initial: [0, 1, 0, 1], k=2
          0  1  2  3

Step 0: i=0, nums[0]=0
        flip_count = 0
        current = 0 XOR 0 = 0 (still 0)

        Current is 0 → MUST FLIP!
        Can flip? i + k = 0 + 2 = 2 ≤ 4 ✓

        Mark flip at position 0
        flip_count = 1, flips = 1

        Effect: [0,1,0,1] → flip [0,1] → [1,0,0,1]
                 ↑↑
                 flipped

Step 1: i=1, nums[1]=1
        Check if flip from step 0 expired: 1 >= k=2? No
        flip_count = 1
        current = 1 XOR 1 = 0 (was 1, now 0 due to flip)

        Current is 0 → MUST FLIP!
        Can flip? i + k = 1 + 2 = 3 ≤ 4 ✓

        Mark flip at position 1
        flip_count = 2, flips = 2

        Effect: [1,0,0,1] → flip [0,0] → [1,1,1,1]
                   ↑↑
                   flipped

Step 2: i=2, nums[2]=0
        Check if flip from step 0 expired: 2 >= 0+k=2? Yes!
        flip_count = 2 - 1 = 1
        Check if flip from step 1 expired: 2 >= 1+k=3? No

        current = 0 XOR 1 = 1 ✓

        Current is 1 → no flip needed

Step 3: i=3, nums[3]=1
        Check expirations: flip from step 1 expired (3 >= 3)
        flip_count = 0

        current = 1 XOR 0 = 1 ✓

        Current is 1 → no flip needed

Final: All bits are 1, flips = 2

Flip Tracking Visualization:
────────────────────────────
  Position:  0   1   2   3
  Original: [0] [1] [0] [1]

  Flip 1 affects: [0,1]
  Flip 2 affects:     [1,2]

  Combined:
    pos 0: 1 flip  → 0 XOR 1 = 1
    pos 1: 2 flips → 1 XOR 0 = 1 (even flips cancel)
    pos 2: 1 flip  → 0 XOR 1 = 1
    pos 3: 0 flips → 1 XOR 0 = 1

Answer: 2 flips

WHY THIS WORKS:
════════════════
● Greedy: flip at first 0 encountered (left to right)
● Order doesn't matter - flips are commutative
● Track flip effects with sliding window counter
● If can't flip (i + k > n), return -1
```

### Solution
```python
def minKBitFlips(nums: list[int], k: int) -> int:
    """
    Greedy with flip tracking.

    Time: O(n)
    Space: O(1)
    """
    n = len(nums)
    flips = 0
    flip_count = 0  # Current flip effect

    for i in range(n):
        # Remove flip effect that's out of window
        if i >= k and nums[i - k] == 2:
            flip_count -= 1

        # Current value after flips
        curr = (nums[i] + flip_count) % 2

        if curr == 0:
            # Need to flip
            if i + k > n:
                return -1

            nums[i] = 2  # Mark as flip starting point
            flip_count += 1
            flips += 1

    return flips
```

### Edge Cases
- All ones → return 0
- All zeros → return -1
- k = 1 → count zeros
- k = n → flip entire array or not

---

## Problem 3: Find XOR Sum of All Pairs Bitwise AND (LC #1835) - Hard

- [LeetCode](https://leetcode.com/problems/find-xor-sum-of-all-pairs-bitwise-and/)

### Video Explanation
- [NeetCode - Find XOR Sum of All Pairs Bitwise AND](https://www.youtube.com/watch?v=Y0_H3LHQG0w)

### Problem Statement
Find XOR of (a AND b) for all pairs from two arrays.


### Visual Intuition
```
Find XOR Sum of All Pairs Bitwise AND
nums1 = [1,2,3], nums2 = [6,5]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: XOR distributes over AND across array pairs
             Result = (XOR of arr1) AND (XOR of arr2)
═══════════════════════════════════════════════════════════════

Brute Force (for understanding):
────────────────────────────────
  All pairs (nums1[i] AND nums2[j]):

  nums1:  1=001  2=010  3=011
  nums2:  6=110  5=101

  1 AND 6 = 001 AND 110 = 000 = 0
  1 AND 5 = 001 AND 101 = 001 = 1
  2 AND 6 = 010 AND 110 = 010 = 2
  2 AND 5 = 010 AND 101 = 000 = 0
  3 AND 6 = 011 AND 110 = 010 = 2
  3 AND 5 = 011 AND 101 = 001 = 1

  XOR all: 0 ^ 1 ^ 2 ^ 0 ^ 2 ^ 1 = 0

Mathematical Derivation:
────────────────────────
  XOR(all pairs) = (1&6)^(1&5)^(2&6)^(2&5)^(3&6)^(3&5)

  Step 1: Group by nums1 element
    = [(1&6)^(1&5)] ^ [(2&6)^(2&5)] ^ [(3&6)^(3&5)]

  Step 2: Apply distributive property: a&b ^ a&c = a & (b^c)
    = [1&(6^5)] ^ [2&(6^5)] ^ [3&(6^5)]
    = [1&3] ^ [2&3] ^ [3&3]
    = 1 ^ 2 ^ 3

  Step 3: Apply again: a&c ^ b&c = (a^b) & c
    = (1^2^3) & (6^5)
    = 0 & 3
    = 0

Verification:
─────────────
  XOR of nums1: 1 ^ 2 ^ 3 = 0
  XOR of nums2: 6 ^ 5 = 3

  Result = 0 AND 3 = 0 ✓

Answer: 0

WHY THIS WORKS:
════════════════
● Property: (a&b) XOR (a&c) = a & (b XOR c)
● Applying repeatedly collapses all pairs to single operation
● O(n+m) instead of O(n*m) brute force
● Bit-level proof: count 1s at each position, check parity
```

### Solution
```python
def getXORSum(arr1: list[int], arr2: list[int]) -> int:
    """
    Use property: XOR of (a AND b) = (XOR of a) AND (XOR of b)

    Proof:
    - For each bit position, count 1s in both arrays
    - AND produces 1 only when both have 1
    - XOR of results: odd count of 1s → 1

    Time: O(n + m)
    Space: O(1)
    """
    xor1 = 0
    for num in arr1:
        xor1 ^= num

    xor2 = 0
    for num in arr2:
        xor2 ^= num

    return xor1 & xor2
```

### Edge Cases
- One array empty → return 0
- All zeros in one array → return 0
- Single element arrays → simple AND
- Large arrays → still O(n+m)

---

## Problem 4: Total Hamming Distance (LC #477) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/total-hamming-distance/)

### Video Explanation
- [NeetCode - Total Hamming Distance](https://www.youtube.com/watch?v=1Gj7p5XKmaw)

### Problem Statement
Sum of Hamming distances between all pairs.


### Visual Intuition
```
Total Hamming Distance
nums = [4, 14, 2]

Binary: 4=0100, 14=1110, 2=0010

Count bit differences at each position:
  Bit 0: [0,0,0] → 0 ones, 3 zeros → 0*3=0 pairs
  Bit 1: [0,1,1] → 2 ones, 1 zero → 2*1=2 pairs
  Bit 2: [1,1,0] → 2 ones, 1 zero → 2*1=2 pairs
  Bit 3: [0,1,0] → 1 one, 2 zeros → 1*2=2 pairs

Total = 0 + 2 + 2 + 2 = 6

Formula: For each bit position, count = ones * zeros
(pairs with different bits at that position)
```

### Solution
```python
def totalHammingDistance(nums: list[int]) -> int:
    """
    Count bits at each position.

    Strategy:
    - For each bit position, count 0s and 1s
    - Pairs with different bits = zeros * ones

    Time: O(32 * n)
    Space: O(1)
    """
    n = len(nums)
    total = 0

    for i in range(32):
        ones = sum((num >> i) & 1 for num in nums)
        zeros = n - ones
        total += ones * zeros

    return total
```

### Edge Cases
- Single element → return 0
- Two elements → their Hamming distance
- All same → return 0
- All zeros → return 0

---

## Problem 5: Gray Code (LC #89) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/gray-code/)

### Video Explanation
- [NeetCode - Gray Code](https://www.youtube.com/watch?v=dwCvHopGJXc)

### Problem Statement
Generate n-bit Gray code sequence where consecutive numbers differ by one bit.

### Visual Intuition
```
Gray Code - n-bit sequence where adjacent differ by 1 bit
n = 2: [0, 1, 3, 2] or [00, 01, 11, 10]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Each adjacent pair differs by exactly 1 bit
             Formula: gray(i) = i XOR (i >> 1)
═══════════════════════════════════════════════════════════════

Reflection Method (build recursively):
──────────────────────────────────────
n=1: [0, 1]
     ↓
n=2: [0, 1] + mirror([0, 1]) with leading 1
     [0, 1] + [1+2, 0+2]
     [0, 1, 3, 2]
     ↓
n=3: [0,1,3,2] + mirror([0,1,3,2]) with leading 1
     [0,1,3,2] + [2+4, 3+4, 1+4, 0+4]
     [0,1,3,2,6,7,5,4]

Visual for n=2:
───────────────
  Binary  Gray   Decimal
    00  →  00  →   0
    01  →  01  →   1
           ↓ (differ by 1 bit)
    10  →  11  →   3
    11  →  10  →   2

  Sequence: 0 → 1 → 3 → 2
            00  01  11  10
             ↑   ↑   ↑
             └─1─┘─1─┘ (each step changes 1 bit)

XOR Formula Derivation:
───────────────────────
  i    i>>1   i ^ (i>>1) = gray(i)

  0    00     00     00 XOR 00 = 00 (0)
  1    01     00     01 XOR 00 = 01 (1)
  2    10     01     10 XOR 01 = 11 (3)
  3    11     01     11 XOR 01 = 10 (2)

  Result: [0, 1, 3, 2] ✓

Why Adjacent Differ by 1 Bit:
─────────────────────────────
  gray(0) = 00
  gray(1) = 01  ← bit 0 changed
  gray(2) = 11  ← bit 1 changed
  gray(3) = 10  ← bit 0 changed
  gray(0) = 00  ← bit 1 changed (wraps around!)

WHY THIS WORKS:
════════════════
● XOR with right-shifted self: leftmost 1 stays, others flip based on neighbor
● Reflection ensures smooth transition at boundary
● Property: exactly 1 bit changes between consecutive codes
● Used in: error correction, rotary encoders, Karnaugh maps
```


### Intuition
```
n=2: 0 → 1 → 3 → 2
Binary: 00 → 01 → 11 → 10

Pattern: Reflect and prefix
n=1: [0, 1]
n=2: [0, 1] + [1+2, 0+2] = [0, 1, 3, 2]
n=3: [0,1,3,2] + [2+4,3+4,1+4,0+4] = [0,1,3,2,6,7,5,4]

Formula: gray(i) = i XOR (i >> 1)
```

### Solution
```python
def grayCode(n: int) -> list[int]:
    """
    Generate Gray code using XOR formula.

    Time: O(2^n)
    Space: O(1) excluding output
    """
    result = []
    for i in range(1 << n):  # 2^n numbers
        # Gray code formula: i XOR (i >> 1)
        result.append(i ^ (i >> 1))
    return result
```

### Reflection Method
```python
def grayCode(n: int) -> list[int]:
    """
    Build Gray code by reflection.

    Strategy:
    - Start with [0]
    - For each bit, reflect current list and add 2^bit

    Time: O(2^n)
    Space: O(2^n)
    """
    result = [0]

    for i in range(n):
        # Reflect: traverse in reverse, add 2^i
        for j in range(len(result) - 1, -1, -1):
            result.append(result[j] | (1 << i))

    return result
```

### Complexity
- **Time**: O(2^n)
- **Space**: O(1) for formula, O(2^n) for reflection

### Edge Cases
- n = 0 → return [0]
- n = 1 → return [0, 1]
- Large n → exponential output size
- First element always 0

---

## Problem 6: Bitwise AND of Numbers Range (LC #201) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/bitwise-and-of-numbers-range/)

### Video Explanation
- [NeetCode - Bitwise AND of Numbers Range](https://www.youtube.com/watch?v=-qrpJykY2gE)

### Problem Statement
Find bitwise AND of all numbers in range [left, right].

### Visual Intuition
```
Bitwise AND of Numbers Range [5, 7]
5 = 101
6 = 110
7 = 111
AND = 100 = 4

Key insight: Result is common prefix of left and right
Bits that differ will become 0 (some number in range has 0 there)

Algorithm: Right shift both until equal
  5=101, 7=111 → different
  2=10,  3=11  → different (shift 1)
  1=1,   1=1   → same! (shift 2)

Result = 1 << 2 = 100 = 4

Common prefix = leftmost bits that don't change in range
```


### Intuition
```
Range [5, 7]:
5 = 101
6 = 110
7 = 111

AND = 100 = 4

Key insight: Result is common prefix of left and right.
Any bit that differs will become 0 (some number in range has 0 there).
```

### Solution
```python
def rangeBitwiseAnd(left: int, right: int) -> int:
    """
    Find common prefix of left and right.

    Strategy:
    - Shift both numbers right until they're equal
    - That's the common prefix
    - Shift back left

    Time: O(log n)
    Space: O(1)
    """
    shift = 0

    # Find common prefix
    while left < right:
        left >>= 1
        right >>= 1
        shift += 1

    # Shift back
    return left << shift
```

### Brian Kernighan's Method
```python
def rangeBitwiseAnd(left: int, right: int) -> int:
    """
    Turn off rightmost 1-bit of right until right <= left.

    Time: O(log n)
    Space: O(1)
    """
    while right > left:
        # Turn off rightmost 1-bit
        right = right & (right - 1)

    return right
```

### Complexity
- **Time**: O(log n) where n = right
- **Space**: O(1)

### Edge Cases
- left == right → return left
- left = 0 → return 0 (any range including 0)
- Adjacent numbers → common prefix
- Large range → quickly converges to 0

---

## Summary

| # | Problem | Key Technique |
|---|---------|---------------|
| 1 | Maximum XOR | Trie of bits |
| 2 | K Bit Flips | Greedy + flip tracking |
| 3 | XOR Sum of AND | Math property |
| 4 | Total Hamming | Bit counting |
| 5 | Gray Code | XOR formula or reflection |
| 6 | Range Bitwise AND | Common prefix |
