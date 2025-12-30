# Arrays & Hashing - Easy Problems

## Problem 1: Two Sum (LC #1) - Easy

- [LeetCode](https://leetcode.com/problems/two-sum/)

### Problem Statement
Given an array of integers `nums` and an integer `target`, return indices of the two numbers that add up to `target`. Each input has exactly one solution.

### Video Explanation
- [NeetCode - Two Sum](https://www.youtube.com/watch?v=KLlXCFG5TnA)
- [Take U Forward - Two Sum](https://www.youtube.com/watch?v=UXDSeD9mN-k)

### Examples
```
Input: nums = [2,7,11,15], target = 9
Output: [0,1]  (nums[0] + nums[1] = 2 + 7 = 9)

Input: nums = [3,2,4], target = 6
Output: [1,2]

Input: nums = [3,3], target = 6
Output: [0,1]
```

### Intuition Development
```
Brute Force: Check every pair → O(n²)

Better: For each number, we need (target - num)
        Use hash map to check if complement exists!

nums = [2, 7, 11, 15], target = 9

Step 1: num = 2, need 9-2=7, seen = {} → not found, add {2: 0}
Step 2: num = 7, need 9-7=2, seen = {2: 0} → FOUND! Return [0, 1]
```

### Solution
```python
def twoSum(nums: list[int], target: int) -> list[int]:
    seen = {}  # value -> index

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return []  # No solution found
```

### Complexity
- **Time**: O(n) - Single pass through array
- **Space**: O(n) - Hash map stores at most n elements

### Edge Cases
- Same element used twice: `[3,3], target=6` → Works because we check before adding
- Negative numbers: Works naturally with subtraction

---

## Problem 2: Contains Duplicate (LC #217) - Easy

- [LeetCode](https://leetcode.com/problems/contains-duplicate/)

### Problem Statement
Given an integer array `nums`, return `true` if any value appears at least twice.

### Video Explanation
- [NeetCode - Contains Duplicate](https://www.youtube.com/watch?v=3OamzN90kPg)

### Examples
```
Input: nums = [1,2,3,1]
Output: true

Input: nums = [1,2,3,4]
Output: false

Input: nums = [1,1,1,3,3,4,3,2,4,2]
Output: true
```

### Intuition Development
```
Approach 1: Sort and check adjacent → O(n log n)
Approach 2: Use set to track seen elements → O(n)

As we iterate, if we see an element already in set → duplicate!
```

### Solution
```python
def containsDuplicate(nums: list[int]) -> bool:
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

# One-liner alternative
def containsDuplicate_oneliner(nums: list[int]) -> bool:
    return len(nums) != len(set(nums))
```

### Complexity
- **Time**: O(n) - Single pass
- **Space**: O(n) - Set stores up to n elements

### Edge Cases
- Empty array: `[]` → return `False`
- Single element: `[1]` → return `False` (no duplicate possible)
- All same elements: `[1,1,1]` → return `True`

---

## Problem 3: Valid Anagram (LC #242) - Easy

- [LeetCode](https://leetcode.com/problems/valid-anagram/)

### Problem Statement
Given two strings `s` and `t`, return `true` if `t` is an anagram of `s`.

### Video Explanation
- [NeetCode - Valid Anagram](https://www.youtube.com/watch?v=9UtInBqnCgA)

### Examples
```
Input: s = "anagram", t = "nagaram"
Output: true

Input: s = "rat", t = "car"
Output: false
```

### Intuition Development
```
Anagram = same characters, different order
→ Same frequency count for all characters

s = "anagram"
freq_s = {a: 3, n: 1, g: 1, r: 1, m: 1}

t = "nagaram"
freq_t = {n: 1, a: 3, g: 1, r: 1, m: 1}

freq_s == freq_t → True!
```

### Solution
```python
from collections import Counter

def isAnagram(s: str, t: str) -> bool:
    # Quick length check
    if len(s) != len(t):
        return False

    return Counter(s) == Counter(t)

# Without Counter
def isAnagram_manual(s: str, t: str) -> bool:
    if len(s) != len(t):
        return False

    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1

    for c in t:
        if c not in freq or freq[c] == 0:
            return False
        freq[c] -= 1

    return True
```

### Complexity
- **Time**: O(n) where n = length of strings
- **Space**: O(1) - At most 26 lowercase letters (constant)

### Follow-up: Unicode Characters
```python
def isAnagram_unicode(s: str, t: str) -> bool:
    return Counter(s) == Counter(t)  # Works for any characters
```

### Edge Cases
- Different lengths: Return `False` immediately
- Empty strings: Both empty → `True`
- Single character: `"a"` and `"a"` → `True`
- Case sensitivity: `"Aa"` and `"aA"` → depends on problem requirements

---

## Problem 4: Single Number (LC #136) - Easy

- [LeetCode](https://leetcode.com/problems/single-number/)

### Problem Statement
Given a non-empty array where every element appears twice except one, find that single one.

### Video Explanation
- [NeetCode - Single Number](https://www.youtube.com/watch?v=qMPX1AOa83k)

### Examples
```
Input: nums = [2,2,1]
Output: 1

Input: nums = [4,1,2,1,2]
Output: 4
```

### Intuition Development
```
Approach 1: Hash map to count frequencies → O(n) space
Approach 2: XOR trick → O(1) space!

XOR properties:
- a ^ a = 0 (same numbers cancel)
- a ^ 0 = a (XOR with 0 gives itself)
- XOR is commutative and associative

[4,1,2,1,2] → 4^1^2^1^2 = 4^(1^1)^(2^2) = 4^0^0 = 4
```

### Solution
```python
# Approach 1: Hash map
def singleNumber_hashmap(nums: list[int]) -> int:
    freq = {}
    for num in nums:
        freq[num] = freq.get(num, 0) + 1

    for num, count in freq.items():
        if count == 1:
            return num

# Approach 2: XOR (optimal)
def singleNumber(nums: list[int]) -> int:
    result = 0
    for num in nums:
        result ^= num
    return result

# One-liner with reduce
from functools import reduce
def singleNumber_reduce(nums: list[int]) -> int:
    return reduce(lambda x, y: x ^ y, nums)
```

### Complexity
- **Time**: O(n)
- **Space**: O(1) for XOR approach, O(n) for hash map

### Edge Cases
- Single element: `[5]` → return `5`
- Negative numbers: XOR works the same way
- Large numbers: No overflow issues with XOR

---

## Problem 5: Majority Element (LC #169) - Easy

- [LeetCode](https://leetcode.com/problems/majority-element/)

### Problem Statement
Given an array `nums` of size n, return the majority element (appears more than ⌊n/2⌋ times). Assume majority always exists.

### Video Explanation
- [NeetCode - Majority Element](https://www.youtube.com/watch?v=7pnhv842keE)

### Examples
```
Input: nums = [3,2,3]
Output: 3

Input: nums = [2,2,1,1,1,2,2]
Output: 2
```

### Intuition Development
```
Approach 1: Hash map count → O(n) space
Approach 2: Boyer-Moore Voting → O(1) space!

Boyer-Moore: Maintain a candidate and count
- If count is 0, pick current as candidate
- If current == candidate, increment count
- Else decrement count

The majority element will survive because it has > n/2 votes
```

### Solution
```python
from collections import Counter

# Approach 1: Hash map
def majorityElement_hashmap(nums: list[int]) -> int:
    freq = Counter(nums)
    return max(freq.keys(), key=freq.get)

# Approach 2: Boyer-Moore Voting Algorithm
def majorityElement(nums: list[int]) -> int:
    candidate = None
    count = 0

    for num in nums:
        if count == 0:
            candidate = num
        count += 1 if num == candidate else -1

    return candidate
```

### Complexity
- **Time**: O(n)
- **Space**: O(1) for Boyer-Moore, O(n) for hash map

### Visual: Boyer-Moore
```
nums = [2, 2, 1, 1, 1, 2, 2]

Step | num | candidate | count
-----|-----|-----------|------
  0  |  2  |     2     |   1
  1  |  2  |     2     |   2
  2  |  1  |     2     |   1
  3  |  1  |     2     |   0
  4  |  1  |     1     |   1   ← New candidate
  5  |  2  |     1     |   0
  6  |  2  |     2     |   1   ← Final candidate

Result: 2 ✓
```

### Edge Cases
- Single element: `[1]` → return `1` (majority is guaranteed)
- All same: `[2,2,2]` → return `2`
- Minimum majority: Element appears exactly ⌊n/2⌋ + 1 times

---

## Problem 6: Missing Number (LC #268) - Easy

- [LeetCode](https://leetcode.com/problems/missing-number/)

### Problem Statement
Given an array `nums` containing n distinct numbers in range [0, n], return the missing number.

### Video Explanation
- [NeetCode - Missing Number](https://www.youtube.com/watch?v=WnPLSRLSANE)

### Examples
```
Input: nums = [3,0,1]
Output: 2

Input: nums = [0,1]
Output: 2

Input: nums = [9,6,4,2,3,5,7,0,1]
Output: 8
```

### Intuition Development
```
Approach 1: Set difference → O(n) space
Approach 2: Sum formula → O(1) space
Approach 3: XOR trick → O(1) space

Sum formula: sum(0 to n) - sum(nums) = missing
           = n*(n+1)/2 - sum(nums)

XOR: XOR all indices and values, pairs cancel out
```

### Solution
```python
# Approach 1: Set
def missingNumber_set(nums: list[int]) -> int:
    full = set(range(len(nums) + 1))
    return (full - set(nums)).pop()

# Approach 2: Sum formula
def missingNumber_sum(nums: list[int]) -> int:
    n = len(nums)
    expected = n * (n + 1) // 2
    return expected - sum(nums)

# Approach 3: XOR
def missingNumber(nums: list[int]) -> int:
    result = len(nums)  # Start with n
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result
```

### Complexity
- **Time**: O(n)
- **Space**: O(1) for sum/XOR, O(n) for set

### Edge Cases
- Missing 0: `[1]` → return `0`
- Missing n: `[0,1,2]` → return `3`
- Single element: `[0]` → return `1`

---

## Problem 7: Happy Number (LC #202) - Easy

- [LeetCode](https://leetcode.com/problems/happy-number/)

### Problem Statement
A happy number is defined by: Starting with any positive integer, replace the number by the sum of squares of its digits. Repeat until it equals 1, or loops endlessly. Return `true` if it's happy.

### Video Explanation
- [NeetCode - Happy Number](https://www.youtube.com/watch?v=ljz85bxOYJ0)

### Examples
```
Input: n = 19
Output: true
19 → 1² + 9² = 82
82 → 8² + 2² = 68
68 → 6² + 8² = 100
100 → 1² + 0² + 0² = 1 ✓

Input: n = 2
Output: false (enters cycle)
```

### Solution
```python
def isHappy(n: int) -> bool:
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

# Alternative: Floyd's cycle detection (O(1) space)
def isHappy_floyd(n: int) -> bool:
    def get_next(num):
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

### Complexity
- **Time**: O(log n) - Number of digits decreases
- **Space**: O(log n) for set, O(1) for Floyd's

### Intuition Development
```
Why does this work?
- For any number, sum of digit squares < number (for large numbers)
- Eventually converges to a small cycle or reaches 1

Example cycle for unhappy numbers:
4 → 16 → 37 → 58 → 89 → 145 → 42 → 20 → 4 (cycle!)

n = 19:
19 → 82 → 68 → 100 → 1 ✓ (happy!)
```

### Edge Cases
- n = 1: Already happy, return `True`
- n = 7: Happy number (7 → 49 → 97 → 130 → 10 → 1)
- Large numbers: Will eventually reduce due to digit sum property

---

## Summary: Easy Problems Checklist

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | Two Sum | Hash map complement | O(n) | O(n) |
| 2 | Contains Duplicate | Set membership | O(n) | O(n) |
| 3 | Valid Anagram | Frequency count | O(n) | O(1) |
| 4 | Single Number | XOR | O(n) | O(1) |
| 5 | Majority Element | Boyer-Moore | O(n) | O(1) |
| 6 | Missing Number | Sum/XOR | O(n) | O(1) |
| 7 | Happy Number | Set cycle detection | O(log n) | O(log n) |

---

## Practice More Easy Problems

- [ ] LC #349 - Intersection of Two Arrays
- [ ] LC #350 - Intersection of Two Arrays II
- [ ] LC #387 - First Unique Character
- [ ] LC #448 - Find All Numbers Disappeared
- [ ] LC #1 - Two Sum (if not done)

