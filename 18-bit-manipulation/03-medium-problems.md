# Bit Manipulation - Advanced Problems

## Advanced Bit Techniques

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED BIT TRICKS                                      │
│                                                                             │
│  Swap without temp:     a ^= b; b ^= a; a ^= b;                            │
│  Absolute value:        (n ^ (n >> 31)) - (n >> 31)                        │
│  Check same sign:       (a ^ b) >= 0                                        │
│  Add 1:                 -~n                                                 │
│  Subtract 1:            ~-n                                                 │
│  Multiply by 2^k:       n << k                                              │
│  Divide by 2^k:         n >> k                                              │
│  Check power of 4:      n > 0 && (n & (n-1)) == 0 && (n & 0x55555555)      │
│  Isolate rightmost 0:   ~n & (n + 1)                                        │
│  Turn on rightmost 0:   n | (n + 1)                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem 1: Maximum XOR of Two Numbers (LC #421) - Medium

- [LeetCode](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/)

### Problem Statement
Find maximum XOR of any two numbers in array.

### Examples
```
Input: nums = [3,10,5,25,2,8]
Output: 28

5 XOR 25 = 00101 XOR 11001 = 11100 = 28
```

### Intuition Development
```
BIT-BY-BIT GREEDY:
Build answer from MSB to LSB.
For each bit, check if we can set it to 1.

Key property: if a ^ b = c, then a ^ c = b

For bit position 4 (value 16):
  Prefixes at bit 4: {00, 01, 11}
  Can we get XOR = 1XXXX? Try candidate = 10000
  Check: any prefix ^ 10000 in prefixes?
    00 ^ 10 = 10 ✗ not in set
    01 ^ 10 = 11 ✓ in set!
  Yes! max_xor = 10000

Continue for each bit position...
```

### Video Explanation
- [NeetCode - Maximum XOR of Two Numbers](https://www.youtube.com/watch?v=EIhAwfHubE8)

### Solution
```python
def findMaximumXOR(nums: list[int]) -> int:
    """
    Find maximum XOR using bit-by-bit greedy approach.

    Strategy:
    - Build answer bit by bit from MSB to LSB
    - For each bit, check if we can set it to 1
    - Use property: if a ^ b = c, then a ^ c = b
    """
    max_xor = 0
    mask = 0

    for i in range(31, -1, -1):  # From MSB to LSB
        # Add current bit to mask
        mask |= (1 << i)

        # Get prefixes of all numbers up to current bit
        prefixes = set()
        for num in nums:
            prefixes.add(num & mask)

        # Try to set current bit in answer
        candidate = max_xor | (1 << i)

        # Check if any two prefixes XOR to candidate
        # If a ^ b = candidate, then a ^ candidate = b
        for prefix in prefixes:
            if prefix ^ candidate in prefixes:
                max_xor = candidate
                break

    return max_xor
```

### Complexity
- **Time**: O(32n) = O(n)
- **Space**: O(n)

### Edge Cases
- Single element → 0
- All same elements → 0
- All zeros → 0
- Maximum value numbers

---

## Problem 2: Total Hamming Distance (LC #477) - Medium

- [LeetCode](https://leetcode.com/problems/total-hamming-distance/)

### Problem Statement
Sum of Hamming distances between all pairs.

### Examples
```
Input: nums = [4,14,2]
Output: 6

4  = 0100
14 = 1110
2  = 0010

Hamming(4,14) = 2 (bits 1 and 3 differ)
Hamming(4,2) = 2 (bits 1 and 2 differ)
Hamming(14,2) = 2 (bits 2 and 3 differ)
Total = 2 + 2 + 2 = 6
```

### Intuition Development
```
COUNT BITS PER POSITION:
For each bit position, count 0s and 1s.
Each (0,1) pair contributes 1 to Hamming distance.

nums = [4, 14, 2] = [0100, 1110, 0010]

Bit 0: [0, 0, 0] → 0 ones, 3 zeros → 0 × 3 = 0 pairs
Bit 1: [0, 1, 1] → 2 ones, 1 zero → 2 × 1 = 2 pairs
Bit 2: [1, 1, 0] → 2 ones, 1 zero → 2 × 1 = 2 pairs
Bit 3: [0, 1, 0] → 1 one, 2 zeros → 1 × 2 = 2 pairs

Total = 0 + 2 + 2 + 2 = 6
```

### Video Explanation
- [NeetCode - Total Hamming Distance](https://www.youtube.com/watch?v=XCyuHSJS7XE)

### Solution
```python
def totalHammingDistance(nums: list[int]) -> int:
    """
    Calculate total Hamming distance efficiently.

    Key insight: For each bit position, count 0s and 1s.
    Contribution = count_0 * count_1 (pairs with different bits)
    """
    total = 0
    n = len(nums)

    for i in range(32):
        # Count numbers with bit i set
        count_ones = sum((num >> i) & 1 for num in nums)
        count_zeros = n - count_ones

        # Each pair of (0, 1) contributes 1 to Hamming distance
        total += count_ones * count_zeros

    return total
```

### Complexity
- **Time**: O(32n) = O(n)
- **Space**: O(1)

### Edge Cases
- Single element → 0
- All same elements → 0
- Two elements → regular Hamming distance
- Large numbers with many bits set

---

## Problem 3: Maximum Product of Word Lengths (LC #318) - Medium

- [LeetCode](https://leetcode.com/problems/maximum-product-of-word-lengths/)

### Problem Statement
Find maximum product of lengths of two words with no common letters.

### Examples
```
Input: words = ["abcw","baz","foo","bar","xtfn","abcdef"]
Output: 16

"abcw" and "xtfn" have no common letters
4 × 4 = 16
```

### Intuition Development
```
BITMASK REPRESENTATION:
Each word → 26-bit mask (1 if letter present)

"abcw" → 0...0010111 (a,b,c,w set)
"xtfn" → 0...1100100001000000000000 (n,t,x,f set)

Check common letters: mask1 & mask2 == 0

words = ["abcw", "foo"]
mask_abcw = 00...0010111
mask_foo  = 00...1000000000000100000

mask_abcw & mask_foo = 0 → no common letters!
Product = 4 × 3 = 12
```

### Video Explanation
- [NeetCode - Maximum Product of Word Lengths](https://www.youtube.com/watch?v=by8JLMYbqjc)

### Solution
```python
def maxProduct(words: list[str]) -> int:
    """
    Find max product of word lengths with no common letters.

    Strategy:
    - Represent each word as bitmask of letters
    - Two words share no letters if masks AND to 0
    """
    # Create bitmask for each word
    masks = []
    for word in words:
        mask = 0
        for char in word:
            mask |= (1 << (ord(char) - ord('a')))
        masks.append(mask)

    max_product = 0
    n = len(words)

    for i in range(n):
        for j in range(i + 1, n):
            # Check if no common letters
            if masks[i] & masks[j] == 0:
                product = len(words[i]) * len(words[j])
                max_product = max(max_product, product)

    return max_product
```

### Complexity
- **Time**: O(n² + nL) where L = avg word length
- **Space**: O(n)

### Edge Cases
- All words share a letter → 0
- Single word → 0
- Empty word in list → contributes 0 to product
- Same word multiple times

---

## Problem 4: Gray Code (LC #89) - Medium

- [LeetCode](https://leetcode.com/problems/gray-code/)

### Problem Statement
Generate n-bit Gray code sequence.

### Examples
```
Input: n = 2
Output: [0,1,3,2] or [0,2,3,1]

00 → 01 → 11 → 10 (each differs by 1 bit)
```

### Intuition Development
```
FORMULA: gray(i) = i ^ (i >> 1)

For n=3:
  i=0: 000 ^ 000 = 000 = 0
  i=1: 001 ^ 000 = 001 = 1
  i=2: 010 ^ 001 = 011 = 3
  i=3: 011 ^ 001 = 010 = 2
  i=4: 100 ^ 010 = 110 = 6
  i=5: 101 ^ 010 = 111 = 7
  i=6: 110 ^ 011 = 101 = 5
  i=7: 111 ^ 011 = 100 = 4

Result: [0, 1, 3, 2, 6, 7, 5, 4]

REFLECTION METHOD:
n=1: [0, 1]
n=2: [0, 1] + [1+2, 0+2] = [0, 1, 3, 2]
n=3: [0, 1, 3, 2] + [2+4, 3+4, 1+4, 0+4] = [0, 1, 3, 2, 6, 7, 5, 4]
```

### Video Explanation
- [NeetCode - Gray Code](https://www.youtube.com/watch?v=RUhXLRto-0M)

### Solution
```python
def grayCode(n: int) -> list[int]:
    """
    Generate n-bit Gray code sequence.

    Gray code: consecutive numbers differ by exactly 1 bit.

    Formula: gray(i) = i ^ (i >> 1)
    """
    return [i ^ (i >> 1) for i in range(1 << n)]


def grayCode_reflect(n: int) -> list[int]:
    """
    Alternative: Reflection method.

    Build n-bit from (n-1)-bit by:
    1. Take (n-1)-bit sequence
    2. Reflect it
    3. Prepend 0 to original, 1 to reflected
    """
    result = [0]

    for i in range(n):
        # Reflect and add 2^i
        result += [x | (1 << i) for x in reversed(result)]

    return result
```

### Complexity
- **Time**: O(2^n)
- **Space**: O(1) excluding output

### Edge Cases
- n = 0 → [0]
- n = 1 → [0, 1]
- Large n → exponential size

---

## Problem 5: Bitwise AND of Numbers Range (LC #201) - Medium

- [LeetCode](https://leetcode.com/problems/bitwise-and-of-numbers-range/)

### Problem Statement
Find bitwise AND of all numbers in range [left, right].

### Examples
```
Input: left = 5, right = 7
Output: 4

5 = 101
6 = 110
7 = 111

5 & 6 & 7 = 100 = 4
```

### Intuition Development
```
COMMON PREFIX:
Result = common prefix of left and right.

Why? Between left and right, all bit combinations occur.
Bits that differ will become 0 somewhere in range.

left = 5 (101)
right = 7 (111)

Bit 0: varies (5=1, 6=0, 7=1) → becomes 0
Bit 1: varies (5=0, 6=1, 7=1) → becomes 0
Bit 2: constant (5=1, 6=1, 7=1) → stays 1

Result = 100 = 4

ALGORITHM:
Shift right until left == right (find common prefix)
Shift back left to restore position
```

### Video Explanation
- [NeetCode - Bitwise AND of Numbers Range](https://www.youtube.com/watch?v=R3T0olAhUq0)

### Solution
```python
def rangeBitwiseAnd(left: int, right: int) -> int:
    """
    Find AND of all numbers in range.

    Key insight: Result is common prefix of left and right.
    """
    shift = 0

    # Find common prefix
    while left < right:
        left >>= 1
        right >>= 1
        shift += 1

    # Shift back
    return left << shift


def rangeBitwiseAnd_brian(left: int, right: int) -> int:
    """
    Alternative: Brian Kernighan's approach.

    Keep turning off rightmost 1 bit of right until right <= left.
    """
    while right > left:
        right &= (right - 1)  # Turn off rightmost 1

    return right
```

### Complexity
- **Time**: O(32) = O(1)
- **Space**: O(1)

### Edge Cases
- left == right → return left
- left = 0 → return 0
- Adjacent numbers (right = left + 1) → might be 0
- Large numbers

---

## Problem 6: Divide Two Integers (LC #29) - Medium

- [LeetCode](https://leetcode.com/problems/divide-two-integers/)

### Problem Statement
Divide without multiplication, division, or mod.

### Examples
```
Input: dividend = 10, divisor = 3
Output: 3

10 = 3 × 3 + 1
```

### Intuition Development
```
BIT SHIFTING FOR MULTIPLICATION:
Find largest 2^k × divisor that fits in dividend.
Subtract, repeat with remainder.

dividend = 10, divisor = 3

Step 1: 3 × 2^1 = 6 ≤ 10, 3 × 2^2 = 12 > 10
  Subtract 6: 10 - 6 = 4
  Quotient += 2

Step 2: 3 × 2^0 = 3 ≤ 4, 3 × 2^1 = 6 > 4
  Subtract 3: 4 - 3 = 1
  Quotient += 1

Step 3: 3 > 1, done

Quotient = 2 + 1 = 3
```

### Video Explanation
- [NeetCode - Divide Two Integers](https://www.youtube.com/watch?v=m4L_5qG4vG8)

### Solution
```python
def divide(dividend: int, divisor: int) -> int:
    """
    Divide using bit manipulation.

    Strategy:
    - Use bit shifts for multiplication by powers of 2
    - Find largest power of 2 that fits, subtract, repeat
    """
    MAX_INT = 2**31 - 1
    MIN_INT = -2**31

    # Handle overflow case
    if dividend == MIN_INT and divisor == -1:
        return MAX_INT

    # Determine sign
    negative = (dividend < 0) ^ (divisor < 0)

    # Work with positive numbers
    dividend = abs(dividend)
    divisor = abs(divisor)

    quotient = 0

    while dividend >= divisor:
        # Find largest power of 2 * divisor that fits
        temp = divisor
        power = 1

        while dividend >= (temp << 1):
            temp <<= 1
            power <<= 1

        dividend -= temp
        quotient += power

    return -quotient if negative else quotient
```

### Complexity
- **Time**: O(32) = O(1)
- **Space**: O(1)

### Edge Cases
- Overflow: -2^31 / -1 = 2^31 (exceeds max int)
- dividend = 0 → 0
- divisor = 1 → dividend
- divisor = -1 → -dividend (watch overflow)

---

## Problem 7: UTF-8 Validation (LC #393) - Medium

- [LeetCode](https://leetcode.com/problems/utf-8-validation/)

### Problem Statement
Check if byte array is valid UTF-8 encoding.

### Examples
```
Input: data = [197,130,1]
Output: true

197 = 11000101 → 2-byte char start
130 = 10000010 → continuation byte
1   = 00000001 → 1-byte char

Valid UTF-8!
```

### Intuition Development
```
UTF-8 ENCODING RULES:
1 byte:  0xxxxxxx (0-127)
2 bytes: 110xxxxx 10xxxxxx
3 bytes: 1110xxxx 10xxxxxx 10xxxxxx
4 bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx

VALIDATION:
For each byte:
  If expecting continuation (remaining > 0):
    Must be 10xxxxxx
  Else:
    Count leading 1s to determine sequence length

data = [197, 130, 1]
197 = 11000101 → starts with 110, expect 1 continuation
130 = 10000010 → valid continuation, remaining = 0
1   = 00000001 → starts with 0, 1-byte char

All valid!
```

### Video Explanation
- [NeetCode - UTF-8 Validation](https://www.youtube.com/watch?v=8PEBJDkjcIo)

### Solution
```python
def validUtf8(data: list[int]) -> bool:
    """
    Validate UTF-8 encoding.

    UTF-8 rules:
    - 1 byte:  0xxxxxxx
    - 2 bytes: 110xxxxx 10xxxxxx
    - 3 bytes: 1110xxxx 10xxxxxx 10xxxxxx
    - 4 bytes: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    """
    # Number of continuation bytes expected
    remaining = 0

    for byte in data:
        # Only consider lower 8 bits
        byte &= 0xFF

        if remaining == 0:
            # Determine number of bytes in character
            if (byte >> 7) == 0b0:
                # 1-byte character
                remaining = 0
            elif (byte >> 5) == 0b110:
                # 2-byte character
                remaining = 1
            elif (byte >> 4) == 0b1110:
                # 3-byte character
                remaining = 2
            elif (byte >> 3) == 0b11110:
                # 4-byte character
                remaining = 3
            else:
                return False
        else:
            # Must be continuation byte: 10xxxxxx
            if (byte >> 6) != 0b10:
                return False
            remaining -= 1

    return remaining == 0
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- Empty data → true
- Incomplete multi-byte sequence → false
- Values > 255 → use only lower 8 bits
- 5+ leading 1s → invalid

---

## Problem 8: Find the Duplicate Number (LC #287) - Medium

- [LeetCode](https://leetcode.com/problems/find-the-duplicate-number/)

### Problem Statement
Find duplicate in array of n+1 integers in range [1, n].

### Examples
```
Input: nums = [1,3,4,2,2]
Output: 2
```

### Intuition Development
```
BIT COUNTING APPROACH:
For each bit position, count 1s in nums and in [1, n].
If nums has more 1s, duplicate has 1 at that position.

nums = [1, 3, 4, 2, 2], n = 4

Bit 0 (value 1):
  nums: 1,3,4,2,2 → 1,1,0,0,0 → 2 ones
  [1,4]: 1,2,3,4 → 1,0,1,0 → 2 ones
  Same → duplicate bit 0 = 0

Bit 1 (value 2):
  nums: 1,3,4,2,2 → 0,1,0,1,1 → 3 ones
  [1,4]: 1,2,3,4 → 0,1,1,0 → 2 ones
  More in nums → duplicate bit 1 = 1

Duplicate = 2 (binary 10)
```

### Video Explanation
- [NeetCode - Find the Duplicate Number](https://www.youtube.com/watch?v=wjYnzkAhcNk)

### Solution
```python
def findDuplicate(nums: list[int]) -> int:
    """
    Find duplicate using bit counting.

    Strategy:
    - For each bit position, count 1s in nums and in [1, n]
    - If nums has more 1s, duplicate has 1 at that position
    """
    n = len(nums) - 1
    duplicate = 0

    for i in range(32):
        mask = 1 << i

        # Count 1s at position i in nums
        count_nums = sum(1 for num in nums if num & mask)

        # Count 1s at position i in [1, n]
        count_range = sum(1 for num in range(1, n + 1) if num & mask)

        # If nums has more 1s, duplicate has 1 at this position
        if count_nums > count_range:
            duplicate |= mask

    return duplicate


def findDuplicate_floyd(nums: list[int]) -> int:
    """
    Alternative: Floyd's cycle detection (O(1) space).

    Treat array as linked list where nums[i] points to nums[nums[i]].
    """
    # Find intersection point
    slow = fast = nums[0]
    while True:
        slow = nums[slow]
        fast = nums[nums[fast]]
        if slow == fast:
            break

    # Find cycle start
    slow = nums[0]
    while slow != fast:
        slow = nums[slow]
        fast = nums[fast]

    return slow
```

### Complexity
- **Time**: O(32n) = O(n) for bit counting, O(n) for Floyd's
- **Space**: O(1)

### Edge Cases
- Only two elements → one is duplicate
- Duplicate appears many times
- Duplicate is 1 or n
- Array is sorted

---

## Problem 9: Decode XORed Array (LC #1720) - Easy

- [LeetCode](https://leetcode.com/problems/decode-xored-array/)

### Problem Statement
Decode array where encoded[i] = arr[i] XOR arr[i+1].

### Examples
```
Input: encoded = [1,2,3], first = 1
Output: [1,0,2,1]

Verify: 1^0=1, 0^2=2, 2^1=3 ✓
```

### Intuition Development
```
XOR INVERSE PROPERTY:
If a ^ b = c, then b = a ^ c

encoded[i] = arr[i] ^ arr[i+1]
So: arr[i+1] = encoded[i] ^ arr[i]

Given first = arr[0]:
  arr[1] = encoded[0] ^ arr[0] = 1 ^ 1 = 0
  arr[2] = encoded[1] ^ arr[1] = 2 ^ 0 = 2
  arr[3] = encoded[2] ^ arr[2] = 3 ^ 2 = 1

Result: [1, 0, 2, 1]
```

### Video Explanation
- [NeetCode - Decode XORed Array](https://www.youtube.com/watch?v=GQHJ8c-lXnQ)

### Solution
```python
def decode(encoded: list[int], first: int) -> list[int]:
    """
    Decode XORed array.

    Given: encoded[i] = arr[i] ^ arr[i+1]
    So: arr[i+1] = encoded[i] ^ arr[i]
    """
    arr = [first]

    for enc in encoded:
        arr.append(enc ^ arr[-1])

    return arr
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- Empty encoded → [first]
- Single element encoded → two element result
- All zeros → [first, first, first, ...]
- first = 0 → result is cumulative XOR

---

## Summary: Advanced Bit Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Maximum XOR | Prefix + XOR property | O(32n) |
| 2 | Total Hamming | Count bits per position | O(32n) |
| 3 | Max Word Product | Bitmask for letters | O(n² + nL) |
| 4 | Gray Code | i ^ (i >> 1) | O(2^n) |
| 5 | Range AND | Common prefix | O(32) |
| 6 | Divide | Bit shift subtraction | O(32) |
| 7 | UTF-8 Validation | Bit pattern matching | O(n) |
| 8 | Find Duplicate | Bit counting | O(32n) |
| 9 | Decode XOR | XOR inverse | O(n) |

---

## Bit Manipulation Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BIT MANIPULATION PATTERNS                                │
│                                                                             │
│  1. XOR PATTERNS:                                                           │
│     • Find single/missing number                                            │
│     • Swap values                                                           │
│     • Find two unique numbers                                               │
│                                                                             │
│  2. BIT COUNTING:                                                           │
│     • Count per position across array                                       │
│     • Find duplicate by bit count difference                                │
│                                                                             │
│  3. BITMASK:                                                                │
│     • Represent sets (subsets problem)                                      │
│     • Represent character sets                                              │
│     • State compression in DP                                               │
│                                                                             │
│  4. BIT SHIFTING:                                                           │
│     • Multiply/divide by powers of 2                                        │
│     • Extract/set specific bits                                             │
│     • Binary search on answer                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practice More Problems

- [ ] LC #137 - Single Number II
- [ ] LC #260 - Single Number III
- [ ] LC #371 - Sum of Two Integers
- [ ] LC #1310 - XOR Queries of a Subarray
- [ ] LC #1734 - Decode XORed Permutation
