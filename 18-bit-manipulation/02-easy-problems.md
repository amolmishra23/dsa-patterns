# Bit Manipulation - Practice Problems

## Problem 1: Single Number (LC #136) - Easy

- [LeetCode](https://leetcode.com/problems/single-number/)

### Problem Statement
Every element appears twice except one. Find the single one.

### Examples
```
Input: nums = [2,2,1]
Output: 1

Input: nums = [4,1,2,1,2]
Output: 4
```

### Video Explanation
- [NeetCode - Single Number](https://www.youtube.com/watch?v=qMPX1AOa83k)

### Intuition
```
XOR magic! Key properties:
- a ^ a = 0 (same numbers cancel)
- a ^ 0 = a (XOR with 0 = itself)
- XOR is commutative & associative

Visual: nums = [4,1,2,1,2]

        4 ^ 1 ^ 2 ^ 1 ^ 2
        = 4 ^ (1 ^ 1) ^ (2 ^ 2)   (rearrange)
        = 4 ^ 0 ^ 0               (pairs cancel)
        = 4                        (answer!)

Binary view:
        4 = 100
        1 = 001
        2 = 010
        1 = 001
        2 = 010
        ─────────
        4 = 100   (each bit: odd count of 1s survives)
```

### Solution
```python
def singleNumber(nums: list[int]) -> int:
    """
    Find single element using XOR.

    Key insight: XOR properties
    - a ^ a = 0 (number XOR itself = 0)
    - a ^ 0 = a (number XOR 0 = itself)
    - XOR is associative and commutative

    So: a ^ b ^ a = b (pairs cancel out)

    Time: O(n)
    Space: O(1)
    """
    result = 0

    for num in nums:
        result ^= num

    return result
```

### Edge Cases
- Single element array → return that element
- Two elements → return XOR of both
- All same except one → XOR cancels pairs
- Negative numbers → XOR works the same
- Large array → still O(n) time

---

## Problem 2: Single Number II (LC #137) - Medium

- [LeetCode](https://leetcode.com/problems/single-number-ii/)

### Problem Statement
Every element appears three times except one. Find the single one.

### Examples
```
Input: nums = [2,2,3,2]
Output: 3

Input: nums = [0,1,0,1,0,1,99]
Output: 99
```

### Video Explanation
- [NeetCode - Single Number II](https://www.youtube.com/watch?v=cOFAmaMBVps)

### Intuition
```
XOR doesn't work for triples! Use bit counting instead.

For each bit position, count 1s across all numbers.
If count % 3 != 0, the single number has 1 at that position.

Visual: nums = [2,2,3,2]

        Binary:
        2 = 010
        2 = 010
        3 = 011
        2 = 010

        Bit 0: count = 0+0+1+0 = 1, 1 % 3 = 1 → bit is 1
        Bit 1: count = 1+1+1+1 = 4, 4 % 3 = 1 → bit is 1
        Bit 2: count = 0+0+0+0 = 0, 0 % 3 = 0 → bit is 0

        Result = 011 = 3 ✓
```

### Solution
```python
def singleNumber(nums: list[int]) -> int:
    """
    Find single element when others appear 3 times.

    Strategy: Count bits at each position.
    If count % 3 != 0, single number has 1 at that position.

    Time: O(32n) = O(n)
    Space: O(1)
    """
    result = 0

    # Check each of 32 bit positions
    for i in range(32):
        # Count 1s at position i across all numbers
        bit_count = 0
        for num in nums:
            # Extract i-th bit and add to count
            bit_count += (num >> i) & 1

        # If count % 3 != 0, single number has 1 at position i
        if bit_count % 3:
            result |= (1 << i)

    # Handle negative numbers in Python (arbitrary precision)
    # If bit 31 is set, it's a negative number
    if result >= 2**31:
        result -= 2**32

    return result


def singleNumber_state_machine(nums: list[int]) -> int:
    """
    Alternative: State machine approach.

    Track count mod 3 using two variables (ones, twos).
    - ones: bits that appeared 1 mod 3 times
    - twos: bits that appeared 2 mod 3 times

    Time: O(n)
    Space: O(1)
    """
    ones = twos = 0

    for num in nums:
        # ones tracks bits appearing once (before twos clears them)
        ones = (ones ^ num) & ~twos
        # twos tracks bits appearing twice
        twos = (twos ^ num) & ~ones

    return ones
```

### Edge Cases
- Single element → return that element
- Negative numbers → handle sign bit carefully
- All same except one → bit counting works
- Zero as single → all bits sum to 0 mod 3
- Large values → 32-bit handling needed

---

## Problem 3: Single Number III (LC #260) - Medium

- [LeetCode](https://leetcode.com/problems/single-number-iii/)

### Problem Statement
Two elements appear once, others appear twice. Find both.

### Examples
```
Input: nums = [1,2,1,3,2,5]
Output: [3,5]
```

### Video Explanation
- [NeetCode - Single Number III](https://www.youtube.com/watch?v=faoVORjd-T8)

### Intuition
```
Two singles! XOR all gives us a^b. How to separate a and b?

Key insight: a and b differ in at least one bit.
Use that bit to split numbers into two groups!

Visual: nums = [1,2,1,3,2,5]

        Step 1: XOR all = 1^2^1^3^2^5 = 3^5 = 011^101 = 110

        Step 2: Find differing bit (rightmost 1 in 110)
                110 & -110 = 110 & 010 = 010 (bit 1)

        Step 3: Split by bit 1:
                Bit 1 = 0: [1,1,5]  → XOR = 5
                Bit 1 = 1: [2,3,2]  → XOR = 3

        Result: [3, 5]

Why this works: Pairs go to same group (cancel out).
a and b go to different groups (different at that bit).
```

### Solution
```python
def singleNumber(nums: list[int]) -> list[int]:
    """
    Find two single elements using XOR.

    Strategy:
    1. XOR all numbers → get xor of two singles (a ^ b)
    2. Find any bit where a and b differ
    3. Split numbers into two groups by that bit
    4. XOR each group to find each single

    Time: O(n)
    Space: O(1)
    """
    # Step 1: XOR all numbers to get a ^ b
    xor_all = 0
    for num in nums:
        xor_all ^= num

    # Step 2: Find rightmost set bit (where a and b differ)
    # n & (-n) isolates the rightmost 1 bit
    diff_bit = xor_all & (-xor_all)

    # Step 3 & 4: Split and XOR
    a = b = 0
    for num in nums:
        if num & diff_bit:
            # Group 1: bit is set
            a ^= num
        else:
            # Group 2: bit is not set
            b ^= num

    return [a, b]
```

### Edge Cases
- Two elements only → return both
- Singles are negatives → handle sign
- Singles differ in only one bit → that bit splits
- Singles are 0 and something → 0 goes to "bit not set" group
- Large array → still O(n) time

---

## Problem 4: Number of 1 Bits (LC #191) - Easy

- [LeetCode](https://leetcode.com/problems/number-of-1-bits/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Count the number of 1 bits (Hamming weight).

### Examples
```
Input: n = 11 (binary: 1011)
Output: 3

Input: n = 128 (binary: 10000000)
Output: 1
```


### Intuition
```
Key insight: n & (n-1) clears the rightmost 1-bit.

Brian Kernighan's algorithm: Keep clearing rightmost 1-bit
until n becomes 0. Count how many times we do this.

Example: n = 12 (1100)
- 12 & 11 = 1100 & 1011 = 1000 (cleared one 1-bit)
- 8 & 7 = 1000 & 0111 = 0000 (cleared one 1-bit)
- Count = 2

Why it works: n-1 flips all bits from rightmost 1 to end.
ANDing clears exactly that rightmost 1.
```

### Solution
```python
def hammingWeight(n: int) -> int:
    """
    Count 1 bits using Brian Kernighan's algorithm.

    Key insight: n & (n-1) clears the rightmost 1 bit.

    Example: n = 12 (1100)
    - 12 & 11 = 1100 & 1011 = 1000 = 8 (cleared rightmost 1)
    - 8 & 7 = 1000 & 0111 = 0000 = 0 (cleared rightmost 1)
    - Count = 2

    Time: O(number of 1 bits)
    Space: O(1)
    """
    count = 0

    while n:
        n &= (n - 1)  # Clear rightmost 1 bit
        count += 1

    return count


def hammingWeight_simple(n: int) -> int:
    """
    Alternative: Check each bit.

    Time: O(32) = O(1)
    Space: O(1)
    """
    count = 0

    while n:
        count += n & 1  # Add last bit
        n >>= 1         # Shift right

    return count
```

### Edge Cases
- n = 0 → return 0
- n = 1 → return 1
- Power of 2 → return 1
- All 1s (like 7, 15, 31) → return number of bits
- Large n → still O(number of 1 bits)

---

## Problem 5: Counting Bits (LC #338) - Easy

- [LeetCode](https://leetcode.com/problems/counting-bits/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Count 1 bits for all numbers from 0 to n.

### Examples
```
Input: n = 5
Output: [0,1,1,2,1,2]
Explanation:
0 → 0
1 → 1
2 → 1
3 → 2
4 → 1
5 → 2
```


### Intuition
```
Key insight: dp[i] = dp[i >> 1] + (i & 1)

Any number's bit count = (number with last bit removed) + last bit.
- i >> 1 removes last bit (already computed!)
- i & 1 is the last bit (0 or 1)

Example: i = 5 (101)
- 5 >> 1 = 2 (10), dp[2] = 1
- 5 & 1 = 1
- dp[5] = 1 + 1 = 2 ✓

Process 0 to n in order, each lookup is O(1).
```

### Solution
```python
def countBits(n: int) -> list[int]:
    """
    Count 1 bits for 0 to n using DP.

    Key insight: dp[i] = dp[i >> 1] + (i & 1)

    Explanation:
    - i >> 1 removes the last bit
    - i & 1 checks if last bit is 1
    - Total 1s = 1s in (i without last bit) + last bit

    Example: i = 5 (101)
    - 5 >> 1 = 2 (10)
    - 5 & 1 = 1
    - dp[5] = dp[2] + 1 = 1 + 1 = 2

    Time: O(n)
    Space: O(n) for result
    """
    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        # Number of 1s = 1s in i//2 + last bit
        dp[i] = dp[i >> 1] + (i & 1)

    return dp


def countBits_alternative(n: int) -> list[int]:
    """
    Alternative: dp[i] = dp[i & (i-1)] + 1

    i & (i-1) clears the rightmost 1, so we just add 1.
    """
    dp = [0] * (n + 1)

    for i in range(1, n + 1):
        dp[i] = dp[i & (i - 1)] + 1

    return dp
```

### Edge Cases
- n = 0 → return [0]
- n = 1 → return [0, 1]
- Powers of 2 → always have 1 bit set
- n-1 (all 1s) → has log(n) bits set
- Large n → DP handles efficiently

---

## Problem 6: Reverse Bits (LC #190) - Easy

- [LeetCode](https://leetcode.com/problems/reverse-bits/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Reverse bits of a 32-bit unsigned integer.

### Examples
```
Input: n = 43261596 (00000010100101000001111010011100)
Output: 964176192 (00111001011110000010100101000000)
```


### Intuition
```
Key insight: Extract bits from right, place them from left.

For each bit position i (0 to 31):
- Extract bit i: (n >> i) & 1
- Place at position (31-i): bit << (31 - i)
- OR into result

Alternative: Divide and conquer - swap halves recursively.
Swap 16-bit halves, then 8-bit, then 4-bit, then 2-bit, then 1-bit.
```

### Solution
```python
def reverseBits(n: int) -> int:
    """
    Reverse bits of 32-bit integer.

    Strategy: Extract each bit from right, place it from left.

    Time: O(32) = O(1)
    Space: O(1)
    """
    result = 0

    for i in range(32):
        # Extract i-th bit from right
        bit = (n >> i) & 1

        # Place it at (31-i) position from right
        result |= bit << (31 - i)

    return result


def reverseBits_optimized(n: int) -> int:
    """
    Alternative: Divide and conquer approach.

    Swap halves recursively:
    - Swap 16-bit halves
    - Swap 8-bit halves within each 16-bit
    - Swap 4-bit halves within each 8-bit
    - etc.

    Time: O(1)
    Space: O(1)
    """
    # Swap 16-bit halves
    n = (n >> 16) | (n << 16)

    # Swap 8-bit halves
    n = ((n & 0xFF00FF00) >> 8) | ((n & 0x00FF00FF) << 8)

    # Swap 4-bit halves
    n = ((n & 0xF0F0F0F0) >> 4) | ((n & 0x0F0F0F0F) << 4)

    # Swap 2-bit halves
    n = ((n & 0xCCCCCCCC) >> 2) | ((n & 0x33333333) << 2)

    # Swap 1-bit halves
    n = ((n & 0xAAAAAAAA) >> 1) | ((n & 0x55555555) << 1)

    return n & 0xFFFFFFFF  # Ensure 32-bit
```

### Edge Cases
- n = 0 → return 0
- n = 1 → return 1 (single bit)
- Palindrome (like 0x81) → returns same value
- All 1s → returns all 1s
- All 0s → returns 0

---

## Problem 7: Missing Number (LC #268) - Easy

- [LeetCode](https://leetcode.com/problems/missing-number/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Find missing number in [0, n] array.

### Examples
```
Input: nums = [3,0,1]
Output: 2

Input: nums = [9,6,4,2,3,5,7,0,1]
Output: 8
```


### Intuition
```
Key insight: XOR all indices with all values.

XOR properties: a ^ a = 0, a ^ 0 = a

For array [3,0,1] (missing 2):
- XOR indices: 0 ^ 1 ^ 2 ^ 3 = some value
- XOR values: 3 ^ 0 ^ 1 = some value
- XOR both: each number cancels except missing one

Every number appears twice (once as index, once as value)
except the missing number (only as index). Result = missing.

Alternative: sum(0 to n) - sum(array) = missing.
```

### Solution
```python
def missingNumber(nums: list[int]) -> int:
    """
    Find missing number using XOR.

    Strategy: XOR all indices (0 to n) with all values.
    Each number appears twice (once as index, once as value)
    except the missing one (only as index).

    Time: O(n)
    Space: O(1)
    """
    n = len(nums)
    result = n  # Start with n (not in indices 0 to n-1)

    for i, num in enumerate(nums):
        result ^= i ^ num

    return result


def missingNumber_math(nums: list[int]) -> int:
    """
    Alternative: Math approach.

    Expected sum = n*(n+1)/2
    Missing = Expected - Actual

    Time: O(n)
    Space: O(1)
    """
    n = len(nums)
    expected_sum = n * (n + 1) // 2
    actual_sum = sum(nums)

    return expected_sum - actual_sum
```

### Edge Cases
- Empty array → missing is 0
- [0] → missing is 1
- Missing 0 → XOR result is 0
- Missing n → XOR result is n
- Large n → both XOR and math work

---

## Problem 8: Power of Two (LC #231) - Easy

- [LeetCode](https://leetcode.com/problems/power-of-two/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Check if n is a power of 2.

### Examples
```
Input: n = 16
Output: true (2^4)

Input: n = 3
Output: false
```


### Intuition
```
Key insight: Power of 2 has exactly one 1-bit.

Binary pattern: 1, 10, 100, 1000, 10000...
n & (n-1) clears the rightmost 1-bit.
If result is 0, n had only one 1-bit → power of 2.

Example: n = 8 (1000)
- 8 & 7 = 1000 & 0111 = 0 ✓ (power of 2)

Example: n = 6 (110)
- 6 & 5 = 110 & 101 = 100 ≠ 0 ✗ (not power of 2)

Edge case: n must be > 0 (0 is not a power of 2).
```

### Solution
```python
def isPowerOfTwo(n: int) -> bool:
    """
    Check if n is power of 2.

    Key insight: Power of 2 has exactly one 1-bit.
    n & (n-1) clears the rightmost 1-bit.
    If result is 0, n had only one 1-bit.

    Example: n = 8 (1000)
    - 8 & 7 = 1000 & 0111 = 0 ✓

    Example: n = 6 (110)
    - 6 & 5 = 110 & 101 = 100 ≠ 0 ✗

    Time: O(1)
    Space: O(1)
    """
    return n > 0 and (n & (n - 1)) == 0
```

### Edge Cases
- n = 0 → return False
- n = 1 → return True (2^0)
- n < 0 → return False
- n = 2^31 → return True
- n = 2^31 - 1 → return False (all 1s)

---

## Problem 9: Bitwise AND of Numbers Range (LC #201) - Medium

- [LeetCode](https://leetcode.com/problems/bitwise-and-of-numbers-range/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Find bitwise AND of all numbers in range [left, right].

### Examples
```
Input: left = 5, right = 7
Output: 4

Explanation:
5 = 101
6 = 110
7 = 111
AND = 100 = 4
```


### Intuition
```
Key insight: Result is the common prefix of left and right.

Any bit where left and right differ will become 0, because
there's some number in between with 0 at that position.

Example: left=5 (101), right=7 (111)
- Bit 0: differs (1 vs 1 vs 0 in 6) → 0
- Bit 1: differs → 0
- Bit 2: same (1) → 1
- Result: 100 = 4

Algorithm: Shift both right until equal (find common prefix),
then shift back left to restore position.
```

### Solution
```python
def rangeBitwiseAnd(left: int, right: int) -> int:
    """
    Find AND of all numbers in range.

    Key insight: Result is the common prefix of left and right.

    Any bit position where left and right differ will become 0
    because there's a number in between with 0 at that position.

    Strategy: Shift both right until they're equal (find common prefix),
    then shift back left.

    Time: O(32) = O(1)
    Space: O(1)
    """
    shift = 0

    # Find common prefix by shifting until equal
    while left < right:
        left >>= 1
        right >>= 1
        shift += 1

    # Shift back to original position
    return left << shift
```

### Edge Cases
- left = right → return that value
- left = 0 → return 0
- Range crosses power of 2 → result loses bits
- Single element range → return that element
- Large range → common prefix might be 0

---

## Problem 10: Sum of Two Integers (LC #371) - Medium

- [LeetCode](https://leetcode.com/problems/sum-of-two-integers/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Calculate sum of two integers without using + or -.

### Examples
```
Input: a = 1, b = 2
Output: 3

Input: a = 2, b = 3
Output: 5
```


### Intuition
```
Key insight: XOR gives sum without carry, AND gives carry bits.

Binary addition without +/-:
- a XOR b = sum ignoring carries
- a AND b = positions where carry occurs
- Shift carry left and repeat until no carry

Example: 2 + 3 = 5
- 10 XOR 11 = 01 (sum without carry)
- 10 AND 11 = 10, shift left = 100 (carry)
- 01 XOR 100 = 101 = 5 ✓

Repeat until carry becomes 0.
```

### Solution
```python
def getSum(a: int, b: int) -> int:
    """
    Add two integers using bit manipulation.

    Strategy:
    - XOR gives sum without carry
    - AND gives carry positions
    - Shift carry left and repeat until no carry

    Example: a = 2 (10), b = 3 (11)
    - sum = 10 ^ 11 = 01
    - carry = (10 & 11) << 1 = 10 << 1 = 100
    - sum = 01 ^ 100 = 101
    - carry = (01 & 100) << 1 = 0
    - Result = 101 = 5

    Time: O(32) = O(1)
    Space: O(1)
    """
    # 32-bit mask for handling negative numbers in Python
    MASK = 0xFFFFFFFF
    MAX_INT = 0x7FFFFFFF

    while b != 0:
        # Calculate sum without carry
        sum_without_carry = (a ^ b) & MASK

        # Calculate carry
        carry = ((a & b) << 1) & MASK

        a = sum_without_carry
        b = carry

    # Handle negative result
    if a > MAX_INT:
        a = ~(a ^ MASK)

    return a
```

### Edge Cases
- a = 0 → return b
- b = 0 → return a
- Both positive → straightforward
- One negative → handle with mask
- Both negative → handle sign extension

---

## Problem 11: Subsets (LC #78) - Medium (Bitmask approach)

- [LeetCode](https://leetcode.com/problems/subsets/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Return all possible subsets.

### Examples
```
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```


### Intuition
```
Key insight: Each subset maps to an n-bit binary number.

For n elements, there are 2^n subsets.
Each element is either included (1) or excluded (0).

Example: nums = [1,2,3], mask = 5 (101)
- Bit 0 = 1: include nums[0] = 1
- Bit 1 = 0: exclude nums[1] = 2
- Bit 2 = 1: include nums[2] = 3
- Subset = [1, 3]

Iterate mask from 0 to 2^n - 1, extract elements
where corresponding bit is set.
```

### Solution
```python
def subsets(nums: list[int]) -> list[list[int]]:
    """
    Generate all subsets using bitmask.

    Strategy:
    - For n elements, there are 2^n subsets
    - Each subset can be represented by n-bit number
    - Bit i = 1 means include nums[i]

    Example: nums = [1,2,3], mask = 5 (101)
    - Bit 0 = 1: include nums[0] = 1
    - Bit 1 = 0: exclude nums[1] = 2
    - Bit 2 = 1: include nums[2] = 3
    - Subset = [1, 3]

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

### Edge Cases
- Empty array → return [[]]
- Single element → return [[], [element]]
- Duplicate elements → bitmask treats as different
- Large n → 2^n subsets (exponential)
- Order matters → bitmask gives consistent order

---

## Summary: Bit Manipulation Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Single Number | XOR all | O(n) |
| 2 | Single Number II | Count bits mod 3 | O(n) |
| 3 | Single Number III | XOR + split by diff bit | O(n) |
| 4 | Number of 1 Bits | n & (n-1) loop | O(k) |
| 5 | Counting Bits | DP with bit shift | O(n) |
| 6 | Reverse Bits | Extract and place | O(1) |
| 7 | Missing Number | XOR indices and values | O(n) |
| 8 | Power of Two | n & (n-1) == 0 | O(1) |
| 9 | Range AND | Find common prefix | O(1) |
| 10 | Sum of Two | XOR + carry | O(1) |
| 11 | Subsets | Bitmask enumeration | O(n * 2^n) |

---

## Practice More Problems

- [ ] LC #318 - Maximum Product of Word Lengths
- [ ] LC #342 - Power of Four
- [ ] LC #389 - Find the Difference
- [ ] LC #461 - Hamming Distance
- [ ] LC #477 - Total Hamming Distance

