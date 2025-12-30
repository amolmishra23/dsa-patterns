# Binary Search - Easy Problems

## Problem 1: Binary Search (LC #704) - Easy

- [LeetCode](https://leetcode.com/problems/binary-search/)

### Problem Statement
Given a sorted array of integers `nums` and a target value, return the index of target if found, otherwise return -1.

### Examples
```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4

Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
```

### Video Explanation
- [NeetCode - Binary Search](https://www.youtube.com/watch?v=s4DPM8ct1pI)
- [Take U Forward - Binary Search](https://www.youtube.com/watch?v=MHf6awe89xw)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  WHY BINARY SEARCH WORKS                                                    │
│                                                                             │
│  Array is SORTED → we can eliminate half each time!                         │
│                                                                             │
│  nums = [-1, 0, 3, 5, 9, 12]    target = 9                                 │
│           L        M        R                                               │
│                                                                             │
│  Step 1: mid = 3, target = 9                                               │
│          3 < 9 → target is in RIGHT half                                   │
│          Eliminate [-1, 0, 3], search [5, 9, 12]                           │
│                                                                             │
│  Step 2: [5, 9, 12]                                                        │
│           L  M   R                                                          │
│          mid = 9 = target → FOUND at index 4!                              │
│                                                                             │
│  Key insight: Each step eliminates HALF the search space                   │
│  n → n/2 → n/4 → n/8 → ... → 1                                             │
│  Steps needed = log₂(n)                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- Single element: `[5], target=5` → return `0`
- Target at start: `[1,2,3], target=1` → return `0`
- Target at end: `[1,2,3], target=3` → return `2`
- Target not found: Return `-1`
- Empty array: Return `-1`

### Solution
```python
def search(nums: list[int], target: int) -> int:
    """
    Classic binary search for exact match.

    Strategy:
    - Maintain search space [left, right]
    - Compare mid element with target
    - Eliminate half the search space each iteration

    Loop invariant: If target exists, it's in nums[left:right+1]

    Time: O(log n) - halve search space each step
    Space: O(1) - only pointers
    """
    left = 0
    right = len(nums) - 1

    while left <= right:
        # Calculate mid index (safe from overflow)
        mid = left + (right - left) // 2

        if nums[mid] == target:
            # Found the target!
            return mid
        elif nums[mid] < target:
            # Target is larger, search right half
            # Eliminate left half including mid
            left = mid + 1
        else:
            # Target is smaller, search left half
            # Eliminate right half including mid
            right = mid - 1

    # Target not found
    return -1
```

### Complexity
- **Time**: O(log n) - halve search space each iteration
- **Space**: O(1) - only use two pointers

### Common Mistakes
- Using `(left + right) / 2` instead of `left + (right - left) // 2` (integer overflow in other languages)
- Using `<` instead of `<=` in while condition (misses single element case)
- Updating `left = mid` or `right = mid` without +1/-1 (infinite loop)

### Related Problems
- LC #35 Search Insert Position
- LC #74 Search a 2D Matrix
- LC #33 Search in Rotated Sorted Array

---

## Problem 2: Search Insert Position (LC #35) - Easy

- [LeetCode](https://leetcode.com/problems/search-insert-position/)

### Problem Statement
Given a sorted array and a target value, return the index if found. If not, return the index where it would be inserted to maintain sorted order.

### Examples
```
Input: nums = [1,3,5,6], target = 5
Output: 2

Input: nums = [1,3,5,6], target = 2
Output: 1 (would insert between 1 and 3)

Input: nums = [1,3,5,6], target = 7
Output: 4 (would insert at end)
```

### Video Explanation
- [NeetCode - Search Insert Position](https://www.youtube.com/watch?v=K-RYzDZkzCI)
- [Take U Forward - Lower Bound](https://www.youtube.com/watch?v=6zhGS79oQ4k)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FINDING THE INSERT POSITION                                                │
│                                                                             │
│  This is finding "lower bound" - first position where nums[i] >= target    │
│                                                                             │
│  nums = [1, 3, 5, 6]    target = 2                                         │
│                                                                             │
│  Where should 2 go?                                                         │
│  [1, 3, 5, 6]                                                              │
│      ↑                                                                      │
│   Insert here! (index 1)                                                    │
│                                                                             │
│  Result: [1, 2, 3, 5, 6]                                                   │
│                                                                             │
│  Algorithm:                                                                 │
│  - Find first element >= target                                            │
│  - If all elements < target, insert at end                                 │
│  - Use left < right (not <=) to converge to single position                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- Target exists: Return its index
- Insert at start: `[2,3,4], target=1` → return `0`
- Insert at end: `[1,2,3], target=5` → return `3`
- Empty array: Return `0`
- Duplicate values: Insert at first occurrence

### Solution
```python
def searchInsert(nums: list[int], target: int) -> int:
    """
    Find index where target is or should be inserted.

    This is essentially finding the "lower bound" -
    the first position where nums[i] >= target.

    Strategy:
    - Binary search for first element >= target
    - If found, return its index
    - If not found, return where it would be inserted

    Time: O(log n)
    Space: O(1)
    """
    left = 0
    right = len(nums)  # Note: can insert at end, so right = len(nums)

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] < target:
            # All elements <= mid are too small
            # Insert position must be after mid
            left = mid + 1
        else:
            # nums[mid] >= target
            # This could be the insert position, or earlier
            right = mid

    # left is the first index where nums[left] >= target
    # This is exactly where target should be inserted
    return left
```

### Complexity
- **Time**: O(log n) - binary search
- **Space**: O(1) - constant extra space

### Common Mistakes
- Setting `right = len(nums) - 1` (can't insert at end)
- Using `<=` instead of `<` with this template
- Returning `mid` instead of `left` at the end

### Related Problems
- LC #704 Binary Search
- LC #278 First Bad Version
- LC #34 Find First and Last Position

---

## Problem 3: First Bad Version (LC #278) - Easy

- [LeetCode](https://leetcode.com/problems/first-bad-version/)

### Problem Statement
You have n versions [1, 2, ..., n]. Find the first bad version. API `isBadVersion(version)` returns True if version is bad. All versions after a bad version are also bad.

### Examples
```
Input: n = 5, bad = 4
Output: 4
Explanation:
- isBadVersion(3) → false
- isBadVersion(5) → true
- isBadVersion(4) → true
First bad version is 4
```

### Video Explanation
- [NeetCode - First Bad Version](https://www.youtube.com/watch?v=SiDMFIMldgg)
- [Take U Forward - Binary Search Variants](https://www.youtube.com/watch?v=6zhGS79oQ4k)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FINDING THE BOUNDARY                                                       │
│                                                                             │
│  Versions: [1, 2, 3, 4, 5]                                                 │
│  Status:   [G, G, G, B, B]   (G=Good, B=Bad)                               │
│                                                                             │
│  We need to find the FIRST bad version (the boundary)                      │
│                                                                             │
│  Binary Search:                                                             │
│  [G, G, G, B, B]                                                           │
│   L     M     R                                                             │
│                                                                             │
│  mid = 3 (Good) → first bad must be AFTER mid                              │
│  [G, G, G, B, B]                                                           │
│            L  R                                                             │
│                                                                             │
│  mid = 4 (Bad) → this COULD be first, or there's an earlier one            │
│                → keep mid in search space, move right = mid                 │
│                                                                             │
│  Eventually left == right → that's the first bad version!                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- First version is bad: `n=5, bad=1` → return `1`
- Last version is bad: `n=5, bad=5` → return `5`
- Single version: `n=1, bad=1` → return `1`
- All bad: First version is the answer

### Solution
```python
def firstBadVersion(n: int) -> int:
    """
    Find first bad version using binary search.

    Versions look like: [good, good, good, BAD, BAD, BAD, ...]
    We need to find the first BAD.

    This is a "find left boundary" problem:
    Find first version where isBadVersion(v) is True.

    Time: O(log n) - binary search
    Space: O(1)
    """
    left = 1      # First version
    right = n     # Last version

    while left < right:
        mid = left + (right - left) // 2

        if isBadVersion(mid):
            # mid is bad, but there might be earlier bad versions
            # Search left half (including mid)
            right = mid
        else:
            # mid is good, first bad must be after mid
            # Search right half (excluding mid)
            left = mid + 1

    # left == right, pointing to first bad version
    return left


# Mock API for testing
def isBadVersion(version: int) -> bool:
    """API provided by problem - returns True if version is bad."""
    pass
```

### Complexity
- **Time**: O(log n) - binary search
- **Space**: O(1) - only pointers

### Common Mistakes
- Using `left <= right` with `right = mid` (infinite loop)
- Calling `isBadVersion` too many times (each call may be expensive)
- Off-by-one error when version numbers start from 1

### Related Problems
- LC #35 Search Insert Position
- LC #34 Find First and Last Position
- LC #162 Find Peak Element

---

## Problem 4: Sqrt(x) (LC #69) - Easy

- [LeetCode](https://leetcode.com/problems/sqrtx/)

### Problem Statement
Compute and return the square root of x (rounded down to nearest integer). You must not use any built-in exponent function or operator.

### Examples
```
Input: x = 4
Output: 2

Input: x = 8
Output: 2
Explanation: sqrt(8) = 2.828..., rounded down is 2
```

### Video Explanation
- [NeetCode - Sqrt(x)](https://www.youtube.com/watch?v=zdMhGxRWutQ)
- [Take U Forward - Square Root](https://www.youtube.com/watch?v=Bsv3FPUX_BA)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BINARY SEARCH ON ANSWER SPACE                                              │
│                                                                             │
│  Find largest n where n² ≤ x                                               │
│                                                                             │
│  Example: x = 8                                                             │
│  Search space: [1, 2, 3, 4, 5, 6, 7, 8]                                    │
│  Squares:      [1, 4, 9, 16, 25, ...]                                      │
│                                                                             │
│  1² = 1  ≤ 8 ✓                                                             │
│  2² = 4  ≤ 8 ✓                                                             │
│  3² = 9  > 8 ✗  ← First one that fails                                     │
│                                                                             │
│  Answer = 2 (largest n where n² ≤ 8)                                       │
│                                                                             │
│  Binary Search:                                                             │
│  [1, 2, 3, 4]  mid = 2, 2² = 4 ≤ 8 → search right                         │
│  [3, 4]        mid = 3, 3² = 9 > 8 → search left                           │
│  [2]           answer = 2                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- x = 0: Return `0`
- x = 1: Return `1`
- Perfect square: `x=16` → return `4`
- Large numbers: Use `mid <= x // mid` to avoid overflow

### Solution
```python
def mySqrt(x: int) -> int:
    """
    Find floor of square root using binary search.

    We're looking for largest integer n where n² <= x.

    Search space: [0, x]
    Condition: n² <= x (find largest n satisfying this)

    Time: O(log x)
    Space: O(1)
    """
    if x < 2:
        return x  # sqrt(0) = 0, sqrt(1) = 1

    left = 1
    right = x // 2  # sqrt(x) <= x/2 for x >= 4

    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid

        if square == x:
            # Perfect square
            return mid
        elif square < x:
            # mid might be answer, but try larger
            left = mid + 1
        else:
            # mid is too large
            right = mid - 1

    # After loop: right is the largest value where right² <= x
    return right
```

### Complexity
- **Time**: O(log x) - binary search on [1, x]
- **Space**: O(1) - constant extra space

### Common Mistakes
- Not handling x = 0 and x = 1 edge cases
- Integer overflow when computing `mid * mid` (use `mid <= x // mid` instead)
- Returning `left` instead of `right` at the end

### Related Problems
- LC #367 Valid Perfect Square
- LC #50 Pow(x, n)
- LC #372 Super Pow

---

## Problem 5: Guess Number Higher or Lower (LC #374) - Easy

- [LeetCode](https://leetcode.com/problems/guess-number-higher-or-lower/)

### Problem Statement
I pick a number from 1 to n. You guess and I tell you if your guess is higher, lower, or correct using `guess(num)` API.

### Examples
```
Input: n = 10, pick = 6
Output: 6

Guesses:
- guess(5) returns 1 (my guess is too low)
- guess(7) returns -1 (my guess is too high)
- guess(6) returns 0 (correct!)
```

### Video Explanation
- [NeetCode - Guess Number](https://www.youtube.com/watch?v=xW4QsTtaCa4)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INTERACTIVE BINARY SEARCH                                                  │
│                                                                             │
│  n = 10, pick = 6                                                          │
│                                                                             │
│  Search: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                                   │
│           L           M              R                                      │
│                                                                             │
│  guess(5) = 1 (too low) → search [6, 7, 8, 9, 10]                          │
│                               L     M        R                              │
│                                                                             │
│  guess(8) = -1 (too high) → search [6, 7]                                  │
│                                 L  M  R                                     │
│                                                                             │
│  guess(6) = 0 → FOUND!                                                     │
│                                                                             │
│  Note: API returns opposite of what you might expect:                       │
│  - Returns 1 if pick > guess (your guess is too LOW)                       │
│  - Returns -1 if pick < guess (your guess is too HIGH)                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- n = 1: Only one number, return `1`
- Pick = 1: First number is the answer
- Pick = n: Last number is the answer
- Large n: Binary search handles efficiently in O(log n)

### Solution
```python
def guessNumber(n: int) -> int:
    """
    Binary search to find the picked number.

    API: guess(num) returns:
    - -1 if num is higher than pick (guess too high)
    - 1 if num is lower than pick (guess too low)
    - 0 if num equals pick

    Time: O(log n)
    Space: O(1)
    """
    left = 1
    right = n

    while left <= right:
        mid = left + (right - left) // 2
        result = guess(mid)

        if result == 0:
            # Found the number!
            return mid
        elif result == 1:
            # My guess is too low, pick is higher
            left = mid + 1
        else:  # result == -1
            # My guess is too high, pick is lower
            right = mid - 1

    # Should never reach here if pick is valid
    return -1


# Mock API for testing
def guess(num: int) -> int:
    """API provided by problem."""
    pass
```

### Complexity
- **Time**: O(log n) - binary search
- **Space**: O(1) - only pointers

### Common Mistakes
- Confusing the API return values (1 means guess too low, -1 means too high)
- Not reading the problem carefully about what the API returns

### Related Problems
- LC #704 Binary Search
- LC #278 First Bad Version
- LC #375 Guess Number Higher or Lower II

---

## Problem 6: Valid Perfect Square (LC #367) - Easy

- [LeetCode](https://leetcode.com/problems/valid-perfect-square/)

### Problem Statement
Given a positive integer num, return true if it is a perfect square, otherwise return false. Do not use any built-in library function.

### Examples
```
Input: num = 16
Output: true (4² = 16)

Input: num = 14
Output: false
```

### Video Explanation
- [NeetCode - Valid Perfect Square](https://www.youtube.com/watch?v=Cg_wWPHJ2Sk)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  IS THERE AN INTEGER SQUARE ROOT?                                           │
│                                                                             │
│  num = 16                                                                   │
│  Search for n where n² = 16                                                │
│                                                                             │
│  [1, 2, 3, 4, 5, 6, 7, 8]                                                  │
│   L        M           R                                                    │
│                                                                             │
│  4² = 16 = num → TRUE, it's a perfect square!                              │
│                                                                             │
│  num = 14                                                                   │
│  3² = 9 < 14                                                               │
│  4² = 16 > 14                                                              │
│  No integer n where n² = 14 → FALSE                                        │
│                                                                             │
│  This is similar to Sqrt(x) but we check for EXACT match                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- num = 1: Perfect square (1² = 1)
- num = 0: Edge case, handle separately
- Large perfect squares: 2147395600 = 46340²
- Non-perfect squares: Return `False`

### Solution
```python
def isPerfectSquare(num: int) -> bool:
    """
    Check if num is a perfect square using binary search.

    Strategy:
    - Binary search for n where n² = num
    - Search space: [1, num]

    Time: O(log num)
    Space: O(1)
    """
    if num < 2:
        return True  # 0 and 1 are perfect squares

    left = 1
    right = num // 2  # sqrt(num) <= num/2 for num >= 4

    while left <= right:
        mid = left + (right - left) // 2
        square = mid * mid

        if square == num:
            return True  # Found perfect square root
        elif square < num:
            left = mid + 1
        else:
            right = mid - 1

    return False  # No integer square root found


def isPerfectSquare_newton(num: int) -> bool:
    """
    Alternative: Newton's method for faster convergence.

    x_{n+1} = (x_n + num/x_n) / 2

    Time: O(log log num) - very fast!
    Space: O(1)
    """
    if num < 2:
        return True

    x = num // 2

    while x * x > num:
        x = (x + num // x) // 2

    return x * x == num
```

### Complexity
- **Binary Search**: Time O(log n), Space O(1)
- **Newton's Method**: Time O(log log n), Space O(1)

### Common Mistakes
- Integer overflow when computing `mid * mid`
- Not handling edge cases (0, 1)
- Forgetting that sqrt(num) ≤ num/2 for num ≥ 4

### Related Problems
- LC #69 Sqrt(x)
- LC #633 Sum of Square Numbers
- LC #279 Perfect Squares

---

## Problem 7: Arranging Coins (LC #441) - Easy

- [LeetCode](https://leetcode.com/problems/arranging-coins/)

### Problem Statement
Build a staircase with n coins. Row i has exactly i coins. Return the number of complete rows you can build.

### Examples
```
Input: n = 5
Output: 2
Explanation:
¤
¤ ¤
¤ ¤ ← incomplete (needs 3, only has 2)

Input: n = 8
Output: 3
Explanation:
¤
¤ ¤
¤ ¤ ¤
¤ ¤ ← incomplete
```

### Video Explanation
- [NeetCode - Arranging Coins](https://www.youtube.com/watch?v=5rHz_6s2Buw)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FIND MAX COMPLETE ROWS                                                     │
│                                                                             │
│  k complete rows need: 1 + 2 + 3 + ... + k = k(k+1)/2 coins                │
│                                                                             │
│  n = 8 coins                                                                │
│  Row 1: 1 coin   (total: 1)                                                │
│  Row 2: 2 coins  (total: 3)                                                │
│  Row 3: 3 coins  (total: 6)                                                │
│  Row 4: 4 coins  (total: 10) ← need 10, only have 8                        │
│                                                                             │
│  Answer: 3 complete rows                                                    │
│                                                                             │
│  Binary Search:                                                             │
│  Find largest k where k(k+1)/2 ≤ n                                         │
│                                                                             │
│  k=1: 1 ≤ 8 ✓                                                              │
│  k=2: 3 ≤ 8 ✓                                                              │
│  k=3: 6 ≤ 8 ✓                                                              │
│  k=4: 10 > 8 ✗                                                             │
│                                                                             │
│  Answer = 3                                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- n = 0: Return `0` (no rows)
- n = 1: Return `1` (one complete row)
- Perfect triangular number: n = 6 → return `3` (exactly 1+2+3)
- Large n: Use math formula for O(1) solution

### Solution
```python
def arrangeCoins(n: int) -> int:
    """
    Find maximum complete rows with n coins.

    Row k needs 1+2+3+...+k = k(k+1)/2 coins total.
    Find largest k where k(k+1)/2 <= n.

    Strategy: Binary search for k

    Time: O(log n)
    Space: O(1)
    """
    left = 0
    right = n

    while left <= right:
        mid = left + (right - left) // 2

        # Coins needed for mid complete rows
        coins_needed = mid * (mid + 1) // 2

        if coins_needed == n:
            return mid  # Exactly enough
        elif coins_needed < n:
            left = mid + 1  # Can build more rows
        else:
            right = mid - 1  # Too many coins needed

    # right is the largest k where k(k+1)/2 <= n
    return right


def arrangeCoins_math(n: int) -> int:
    """
    Mathematical solution using quadratic formula.

    Solve k(k+1)/2 <= n
    k² + k - 2n <= 0
    k = (-1 + sqrt(1 + 8n)) / 2

    Time: O(1)
    Space: O(1)
    """
    import math
    return int((-1 + math.sqrt(1 + 8 * n)) / 2)
```

### Complexity
- **Binary Search**: Time O(log n), Space O(1)
- **Math**: Time O(1), Space O(1)

### Common Mistakes
- Integer overflow in `mid * (mid + 1)` for very large n
- Using wrong formula for sum (should be k(k+1)/2)
- Returning `left` instead of `right`

### Related Problems
- LC #69 Sqrt(x)
- LC #1014 Best Sightseeing Pair
- LC #1281 Subtract the Product and Sum of Digits

---

## Summary: Easy Problems Checklist

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | Binary Search | Standard template | O(log n) | O(1) |
| 2 | Search Insert Position | Lower bound | O(log n) | O(1) |
| 3 | First Bad Version | Left boundary | O(log n) | O(1) |
| 4 | Sqrt(x) | Search for n² ≤ x | O(log x) | O(1) |
| 5 | Guess Number | Standard search | O(log n) | O(1) |
| 6 | Valid Perfect Square | Search for n² = x | O(log n) | O(1) |
| 7 | Arranging Coins | Search for sum formula | O(log n) | O(1) |

---

## Practice More Easy Problems

- [ ] LC #744 - Find Smallest Letter Greater Than Target
- [ ] LC #852 - Peak Index in a Mountain Array
- [ ] LC #1351 - Count Negative Numbers in a Sorted Matrix
- [ ] LC #1539 - Kth Missing Positive Number
