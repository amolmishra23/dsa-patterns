# Prefix Sum - Easy Problems

## Problem 1: Range Sum Query - Immutable (LC #303) - Easy

- [LeetCode](https://leetcode.com/problems/range-sum-query-immutable/)

### Problem Statement
Given an integer array `nums`, handle multiple queries to calculate the sum of elements between indices `left` and `right` inclusive.

### Examples
```
Input: nums = [-2, 0, 3, -5, 2, -1]
sumRange(0, 2) → 1   (-2 + 0 + 3 = 1)
sumRange(2, 5) → -1  (3 + -5 + 2 + -1 = -1)
sumRange(0, 5) → -3  (entire array sum)
```

### Video Explanation
- [NeetCode - Range Sum Query](https://www.youtube.com/watch?v=2pndAmo_sMA)
- [Take U Forward - Prefix Sum](https://www.youtube.com/watch?v=7pJo_rM0z_s)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  THE POWER OF PREFIX SUMS                                                   │
│                                                                             │
│  nums = [-2, 0, 3, -5, 2, -1]                                              │
│                                                                             │
│  Without prefix sum: sumRange(2,5) requires 4 additions each time          │
│  With prefix sum: sumRange(2,5) is just ONE subtraction!                   │
│                                                                             │
│  Build prefix array:                                                        │
│  prefix[i] = sum of nums[0..i-1]                                           │
│                                                                             │
│  Index:     0   1   2   3   4   5   6                                      │
│  prefix = [ 0, -2, -2,  1, -4, -2, -3]                                     │
│            ↑   ↑   ↑   ↑   ↑   ↑   ↑                                       │
│          empty -2  -2  -2  -2  -2  -2                                       │
│                    +0  +0  +0  +0  +0                                       │
│                        +3  +3  +3  +3                                       │
│                            -5  -5  -5                                       │
│                                +2  +2                                       │
│                                    -1                                       │
│                                                                             │
│  sumRange(2, 5) = prefix[6] - prefix[2] = -3 - (-2) = -1 ✓                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- Single element query: `sumRange(0, 0)` → return `nums[0]`
- Full array query: `sumRange(0, n-1)` → return total sum
- Negative numbers: Works correctly with subtraction
- Many queries: O(1) per query after O(n) preprocessing

### Solution
```python
class NumArray:
    """
    Range Sum Query using prefix sums.

    Build once, query many times efficiently.

    Time: O(n) init, O(1) query
    Space: O(n)
    """

    def __init__(self, nums: list[int]):
        """
        Build prefix sum array.

        prefix[i] = sum of nums[0..i-1]
        We add an extra 0 at the start for easier calculation.
        """
        self.prefix = [0]

        for num in nums:
            # Each new prefix = previous prefix + current number
            self.prefix.append(self.prefix[-1] + num)

        # Example: nums = [-2, 0, 3, -5, 2, -1]
        # prefix = [0, -2, -2, 1, -4, -2, -3]

    def sumRange(self, left: int, right: int) -> int:
        """
        Return sum of nums[left..right] inclusive.

        Formula: prefix[right+1] - prefix[left]

        This works because:
        - prefix[right+1] = sum of nums[0..right]
        - prefix[left] = sum of nums[0..left-1]
        - Difference = sum of nums[left..right]
        """
        return self.prefix[right + 1] - self.prefix[left]
```

### Complexity
- **Time**: O(n) for initialization, O(1) for each query
- **Space**: O(n) for prefix array

### Common Mistakes
- Off-by-one errors in prefix array indexing
- Forgetting to add the initial 0 to prefix array
- Using `prefix[right] - prefix[left]` instead of `prefix[right+1] - prefix[left]`

### Related Problems
- LC #304 Range Sum Query 2D
- LC #307 Range Sum Query - Mutable
- LC #724 Find Pivot Index

---

## Problem 2: Find Pivot Index (LC #724) - Easy

- [LeetCode](https://leetcode.com/problems/find-pivot-index/)

### Problem Statement
Find the leftmost pivot index where the sum of elements to the left equals the sum of elements to the right. The pivot element is not included in either sum.

### Examples
```
Input: nums = [1,7,3,6,5,6]
Output: 3
Explanation: Left sum = 1+7+3 = 11, Right sum = 5+6 = 11

Input: nums = [1,2,3]
Output: -1 (no pivot exists)

Input: nums = [2,1,-1]
Output: 0 (left sum = 0, right sum = 1+(-1) = 0)
```

### Video Explanation
- [NeetCode - Find Pivot Index](https://www.youtube.com/watch?v=u89i60lYx8U)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FINDING THE BALANCE POINT                                                  │
│                                                                             │
│  nums = [1, 7, 3, 6, 5, 6]                                                 │
│  total = 28                                                                 │
│                                                                             │
│  At each index i:                                                           │
│    left_sum = sum of elements before i                                     │
│    right_sum = total - left_sum - nums[i]                                  │
│                                                                             │
│  i=0: left=0, right=28-0-1=27    → 0 ≠ 27                                  │
│  i=1: left=1, right=28-1-7=20    → 1 ≠ 20                                  │
│  i=2: left=8, right=28-8-3=17    → 8 ≠ 17                                  │
│  i=3: left=11, right=28-11-6=11  → 11 = 11 ✓ FOUND!                        │
│                                                                             │
│       [1, 7, 3, 6, 5, 6]                                                   │
│        ←─────→  ↑  ←───→                                                   │
│        left=11  P  right=11                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- Pivot at index 0: `[2,1,-1]` → left_sum = 0, right_sum = 0
- Pivot at last index: `[-1,1,2]` → left_sum = 0, right_sum = 0
- No pivot exists: Return `-1`
- Single element: `[1]` → index 0 is pivot (both sums = 0)

### Solution
```python
def pivotIndex(nums: list[int]) -> int:
    """
    Find pivot index where left sum equals right sum.

    At index i:
    - Left sum = prefix[i]
    - Right sum = total - prefix[i] - nums[i]

    We need: left_sum == right_sum
    i.e., prefix[i] == total - prefix[i] - nums[i]
    i.e., 2 * prefix[i] + nums[i] == total

    Time: O(n)
    Space: O(1)
    """
    total = sum(nums)
    left_sum = 0

    for i, num in enumerate(nums):
        # Right sum = total - left_sum - current element
        right_sum = total - left_sum - num

        if left_sum == right_sum:
            return i

        # Update left sum for next iteration
        left_sum += num

    return -1  # No pivot found
```

### Complexity
- **Time**: O(n) - single pass after computing total
- **Space**: O(1) - only tracking running sum

### Common Mistakes
- Including pivot element in left or right sum
- Not handling edge case where pivot is at index 0 (left sum = 0)
- Computing total sum inside the loop (inefficient)

### Related Problems
- LC #1991 Find the Middle Index in Array
- LC #238 Product of Array Except Self
- LC #560 Subarray Sum Equals K

---

## Problem 3: Running Sum of 1d Array (LC #1480) - Easy

- [LeetCode](https://leetcode.com/problems/running-sum-of-1d-array/)

### Problem Statement
Given an array `nums`, return the running sum where `runningSum[i] = sum(nums[0]...nums[i])`.

### Examples
```
Input: nums = [1,2,3,4]
Output: [1,3,6,10]
Explanation: Running sum = [1, 1+2, 1+2+3, 1+2+3+4]

Input: nums = [1,1,1,1,1]
Output: [1,2,3,4,5]
```

### Video Explanation
- [NeetCode - Running Sum](https://www.youtube.com/watch?v=nMF2xLBrkU0)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BUILDING A PREFIX SUM ARRAY                                                │
│                                                                             │
│  nums = [1, 2, 3, 4]                                                       │
│                                                                             │
│  Step by step:                                                              │
│  result[0] = 1                    → [1, _, _, _]                           │
│  result[1] = 1 + 2 = 3            → [1, 3, _, _]                           │
│  result[2] = 3 + 3 = 6            → [1, 3, 6, _]                           │
│  result[3] = 6 + 4 = 10           → [1, 3, 6, 10]                          │
│                                                                             │
│  Key insight: result[i] = result[i-1] + nums[i]                            │
│  Each new sum builds on the previous one!                                   │
│                                                                             │
│  Visual:                                                                    │
│  nums:   [1]  [2]  [3]  [4]                                                │
│           ↓    ↓    ↓    ↓                                                 │
│  sums:   [1]→[3]→[6]→[10]                                                  │
│              +2   +3   +4                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- Single element: `[5]` → `[5]`
- All zeros: `[0,0,0]` → `[0,0,0]`
- Negative numbers: Works correctly
- In-place vs new array: Choose based on whether input can be modified

### Solution
```python
def runningSum(nums: list[int]) -> list[int]:
    """
    Compute running sum (prefix sum) of array.

    Each element becomes sum of all elements up to and including itself.

    Time: O(n)
    Space: O(1) if modifying in place, O(n) for output
    """
    # In-place modification
    for i in range(1, len(nums)):
        nums[i] += nums[i - 1]

    return nums


def runningSum_new_array(nums: list[int]) -> list[int]:
    """
    Version that creates new array (doesn't modify input).
    """
    result = []
    running = 0

    for num in nums:
        running += num
        result.append(running)

    return result
```

### Complexity
- **Time**: O(n) - single pass
- **Space**: O(1) for in-place, O(n) for new array

### Common Mistakes
- Starting loop at index 0 instead of 1 for in-place version
- Creating unnecessary extra space when in-place modification is allowed

### Related Problems
- LC #303 Range Sum Query - Immutable
- LC #724 Find Pivot Index
- LC #1413 Minimum Value to Get Positive Step by Step Sum

---

## Problem 4: Minimum Value to Get Positive Step by Step Sum (LC #1413) - Easy

- [LeetCode](https://leetcode.com/problems/minimum-value-to-get-positive-step-by-step-sum/)

### Problem Statement
Given an array `nums`, find the minimum positive start value such that the running sum is never less than 1.

### Examples
```
Input: nums = [-3,2,-3,4,2]
Output: 5
Explanation: With startValue = 5:
Step 1: 5 + (-3) = 2
Step 2: 2 + 2 = 4
Step 3: 4 + (-3) = 1
Step 4: 1 + 4 = 5
Step 5: 5 + 2 = 7
All steps ≥ 1 ✓

Input: nums = [1,2]
Output: 1 (minimum possible)
```

### Video Explanation
- [LeetCode Discuss - Step by Step Sum](https://leetcode.com/problems/minimum-value-to-get-positive-step-by-step-sum/)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FINDING MINIMUM START VALUE                                                │
│                                                                             │
│  nums = [-3, 2, -3, 4, 2]                                                  │
│                                                                             │
│  First, compute prefix sums (with startValue = 0):                         │
│  prefix = [-3, -1, -4, 0, 2]                                               │
│                                                                             │
│  The minimum prefix sum is -4                                               │
│                                                                             │
│  We need: startValue + min_prefix ≥ 1                                      │
│  So: startValue ≥ 1 - min_prefix = 1 - (-4) = 5                            │
│                                                                             │
│  Verification with startValue = 5:                                          │
│  [5] → [2] → [4] → [1] → [5] → [7]                                         │
│      -3    +2    -3    +4    +2                                             │
│  All values ≥ 1 ✓                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- All positive: `[1,2]` → return `1` (minimum allowed)
- All negative: Find deepest valley
- Single element: `[-3]` → return `4` (1 - (-3))
- Zero in array: Works correctly

### Solution
```python
def minStartValue(nums: list[int]) -> int:
    """
    Find minimum start value so running sum is always ≥ 1.

    Key insight:
    - Compute prefix sums starting from 0
    - Find minimum prefix sum
    - startValue = 1 - min_prefix (but at least 1)

    Time: O(n)
    Space: O(1)
    """
    min_prefix = 0
    prefix_sum = 0

    for num in nums:
        prefix_sum += num
        min_prefix = min(min_prefix, prefix_sum)

    # We need startValue + min_prefix >= 1
    # So startValue >= 1 - min_prefix
    # Also startValue must be positive, so at least 1
    return max(1, 1 - min_prefix)
```

### Complexity
- **Time**: O(n) - single pass
- **Space**: O(1) - only tracking min and current sum

### Common Mistakes
- Forgetting that start value must be at least 1 (positive)
- Computing all prefix sums first (unnecessary space)
- Off-by-one error (need sum ≥ 1, not ≥ 0)

### Related Problems
- LC #1480 Running Sum of 1d Array
- LC #724 Find Pivot Index
- LC #53 Maximum Subarray

---

## Problem 5: Number of Ways to Split Array (LC #2270) - Medium

- [LeetCode](https://leetcode.com/problems/number-of-ways-to-split-array/)

### Problem Statement
Find the number of valid splits where the sum of the first `i+1` elements is greater than or equal to the sum of the remaining elements.

### Examples
```
Input: nums = [10,4,-8,7]
Output: 2
Valid splits:
- i=0: [10] vs [4,-8,7] → 10 ≥ 3 ✓
- i=1: [10,4] vs [-8,7] → 14 ≥ -1 ✓
- i=2: [10,4,-8] vs [7] → 6 < 7 ✗
```

### Video Explanation
- [LeetCode - Number of Ways to Split Array](https://leetcode.com/problems/number-of-ways-to-split-array/)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  COUNTING VALID SPLIT POINTS                                                │
│                                                                             │
│  nums = [10, 4, -8, 7]                                                     │
│  total = 13                                                                 │
│                                                                             │
│  At each split point i:                                                     │
│    left_sum = prefix sum up to i                                           │
│    right_sum = total - left_sum                                            │
│    Valid if left_sum >= right_sum                                          │
│                                                                             │
│  i=0: left=10, right=3   → 10 ≥ 3 ✓                                        │
│       [10 | 4, -8, 7]                                                      │
│                                                                             │
│  i=1: left=14, right=-1  → 14 ≥ -1 ✓                                       │
│       [10, 4 | -8, 7]                                                      │
│                                                                             │
│  i=2: left=6, right=7    → 6 < 7 ✗                                         │
│       [10, 4, -8 | 7]                                                      │
│                                                                             │
│  Answer: 2 valid splits                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- Two elements: `[1,1]` → check only one split point
- All same: `[5,5,5,5]` → multiple valid splits
- Negative numbers: Works correctly with sum comparison
- Can't split after last element (right would be empty)

### Solution
```python
def waysToSplitArray(nums: list[int]) -> int:
    """
    Count valid splits where left sum >= right sum.

    A valid split at index i means:
    - Left part: nums[0..i]
    - Right part: nums[i+1..n-1]
    - Left sum >= Right sum

    Note: Can't split after last element (right part would be empty)

    Time: O(n)
    Space: O(1)
    """
    total = sum(nums)
    left_sum = 0
    count = 0

    # Check all split points except the last one
    for i in range(len(nums) - 1):
        left_sum += nums[i]
        right_sum = total - left_sum

        if left_sum >= right_sum:
            count += 1

    return count
```

### Complexity
- **Time**: O(n) - single pass after computing total
- **Space**: O(1) - only tracking sums

### Common Mistakes
- Including the last index as a valid split point
- Computing total sum inside the loop
- Using `>` instead of `>=` for comparison

### Related Problems
- LC #724 Find Pivot Index
- LC #915 Partition Array into Disjoint Intervals
- LC #1712 Ways to Split Array Into Three Subarrays

---

## Problem 6: Left and Right Sum Differences (LC #2574) - Easy

- [LeetCode](https://leetcode.com/problems/left-and-right-sum-differences/)

### Problem Statement
Given array `nums`, return array `answer` where `answer[i] = |leftSum[i] - rightSum[i]|`.

### Examples
```
Input: nums = [10,4,8,3]
Output: [15,1,11,22]
Explanation:
- i=0: |0 - 15| = 15
- i=1: |10 - 11| = 1
- i=2: |14 - 3| = 11
- i=3: |22 - 0| = 22
```

### Video Explanation
- [LeetCode - Left and Right Sum](https://leetcode.com/problems/left-and-right-sum-differences/)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPUTING LEFT AND RIGHT SUMS                                              │
│                                                                             │
│  nums = [10, 4, 8, 3]                                                      │
│                                                                             │
│  leftSum[i] = sum of elements BEFORE index i                               │
│  rightSum[i] = sum of elements AFTER index i                               │
│                                                                             │
│  Index:     0    1    2    3                                               │
│  nums:     10    4    8    3                                               │
│  leftSum:   0   10   14   22                                               │
│  rightSum: 15   11    3    0                                               │
│                                                                             │
│  answer[i] = |leftSum[i] - rightSum[i]|                                    │
│  answer:   15    1   11   22                                               │
│                                                                             │
│  Efficient: Compute both in single pass!                                    │
│  rightSum = total - leftSum - nums[i]                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- Single element: `[5]` → `[0]` (both sums are 0)
- Two elements: `[1,2]` → `[|0-2|, |1-0|]` = `[2,1]`
- All same: Symmetric differences
- Negative numbers: Absolute value handles correctly

### Solution
```python
def leftRightDifference(nums: list[int]) -> list[int]:
    """
    Compute |leftSum - rightSum| for each index.

    leftSum[i] = sum of nums[0..i-1]
    rightSum[i] = sum of nums[i+1..n-1]

    Time: O(n)
    Space: O(1) extra (output doesn't count)
    """
    total = sum(nums)
    left_sum = 0
    result = []

    for num in nums:
        # rightSum = total - leftSum - current element
        right_sum = total - left_sum - num
        result.append(abs(left_sum - right_sum))
        left_sum += num

    return result
```

### Complexity
- **Time**: O(n) - single pass
- **Space**: O(1) extra space (output array doesn't count)

### Common Mistakes
- Including current element in left or right sum
- Forgetting absolute value
- Computing prefix arrays separately (inefficient)

### Related Problems
- LC #724 Find Pivot Index
- LC #1991 Find the Middle Index in Array
- LC #2270 Number of Ways to Split Array

---

## Problem 7: Find the Highest Altitude (LC #1732) - Easy

- [LeetCode](https://leetcode.com/problems/find-the-highest-altitude/)

### Problem Statement
A biker starts at altitude 0 and travels through points with altitude changes given in `gain`. Return the highest altitude reached.

### Examples
```
Input: gain = [-5,1,5,0,-7]
Output: 1
Explanation: Altitudes are [0,-5,-4,1,1,-6]. Highest = 1

Input: gain = [-4,-3,-2,-1,4,3,2]
Output: 0 (starting point is highest)
```

### Video Explanation
- [LeetCode - Highest Altitude](https://leetcode.com/problems/find-the-highest-altitude/)

### Intuition Development
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRACKING ALTITUDE (PREFIX SUM WITH MAX)                                    │
│                                                                             │
│  gain = [-5, 1, 5, 0, -7]                                                  │
│  Start at altitude 0                                                        │
│                                                                             │
│  Point 0: altitude = 0                                                      │
│  Point 1: altitude = 0 + (-5) = -5                                         │
│  Point 2: altitude = -5 + 1 = -4                                           │
│  Point 3: altitude = -4 + 5 = 1   ← HIGHEST!                               │
│  Point 4: altitude = 1 + 0 = 1                                             │
│  Point 5: altitude = 1 + (-7) = -6                                         │
│                                                                             │
│  Visual:                                                                    │
│       1 ─────────────●───●                                                 │
│       0 ●                                                                   │
│      -4         ●                                                           │
│      -5     ●                                                               │
│      -6                     ●                                               │
│                                                                             │
│  Answer: 1                                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- All negative gains: Starting point (0) is highest
- All positive gains: Last point is highest
- Single gain: Compare 0 with final altitude
- Zero gains: Altitude stays constant

### Solution
```python
def largestAltitude(gain: list[int]) -> int:
    """
    Find highest altitude reached.

    Altitude at point i = sum of gain[0..i-1]
    This is a prefix sum problem where we track max.

    Time: O(n)
    Space: O(1)
    """
    max_altitude = 0  # Starting point could be highest
    current_altitude = 0

    for g in gain:
        current_altitude += g
        max_altitude = max(max_altitude, current_altitude)

    return max_altitude
```

### Complexity
- **Time**: O(n) - single pass
- **Space**: O(1) - only tracking current and max

### Common Mistakes
- Forgetting that starting altitude (0) could be the maximum
- Computing all altitudes first then finding max (unnecessary space)
- Off-by-one: gain has n elements, but there are n+1 altitude points

### Related Problems
- LC #1480 Running Sum of 1d Array
- LC #53 Maximum Subarray
- LC #121 Best Time to Buy and Sell Stock

---

## Summary: Prefix Sum Easy Problems

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | Range Sum Query | Basic prefix sum | O(1) query | O(n) |
| 2 | Find Pivot Index | Left vs right sum | O(n) | O(1) |
| 3 | Running Sum | Build prefix array | O(n) | O(1) |
| 4 | Min Start Value | Track min prefix | O(n) | O(1) |
| 5 | Ways to Split | Count valid splits | O(n) | O(1) |
| 6 | Left Right Diff | Dual prefix sums | O(n) | O(1) |
| 7 | Highest Altitude | Prefix sum + max | O(n) | O(1) |

---

## Practice More Easy Problems

- [ ] LC #1893 - Check if All Integers in Range Are Covered
- [ ] LC #2485 - Find the Pivot Integer
- [ ] LC #2389 - Longest Subsequence With Limited Sum
- [ ] LC #1854 - Maximum Population Year
