# Dynamic Programming - Easy Problems

## Problem 1: Climbing Stairs (LC #70) - Easy

- [LeetCode](https://leetcode.com/problems/climbing-stairs/)

### Problem Statement
You are climbing a staircase with `n` steps. Each time you can climb 1 or 2 steps. How many distinct ways can you climb to the top?

### Examples
```
Input: n = 2
Output: 2 (1+1 or 2)

Input: n = 3
Output: 3 (1+1+1, 1+2, 2+1)

Input: n = 4
Output: 5 (1+1+1+1, 1+1+2, 1+2+1, 2+1+1, 2+2)
```

### Video Explanation
- [NeetCode - Climbing Stairs](https://www.youtube.com/watch?v=Y0lT9Fck7qI)
- [Take U Forward - Climbing Stairs](https://www.youtube.com/watch?v=mLfjzJsN8us)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  THIS IS FIBONACCI IN DISGUISE!                                             │
│                                                                             │
│  To reach step n, you can come from:                                        │
│  - Step n-1 (take 1 step)                                                  │
│  - Step n-2 (take 2 steps)                                                 │
│                                                                             │
│  So: ways(n) = ways(n-1) + ways(n-2)                                       │
│                                                                             │
│  Example for n = 4:                                                         │
│                                                                             │
│  Step 0: 1 way (already there)                                             │
│  Step 1: 1 way (0→1)                                                       │
│  Step 2: 2 ways (0→1→2, 0→2)                                               │
│  Step 3: 3 ways (from step 1: 1 way) + (from step 2: 2 ways) = 3           │
│  Step 4: 5 ways (from step 2: 2 ways) + (from step 3: 3 ways) = 5          │
│                                                                             │
│  Visual:                                                                    │
│       ┌─┐                                                                   │
│     ┌─┤4│  ← 5 ways to reach here                                          │
│   ┌─┤3│                                                                     │
│  ┌┤2│                                                                       │
│  │1│                                                                        │
│  │0│                                                                        │
│  └─┘                                                                        │
│                                                                             │
│  n:    0  1  2  3  4  5  6  7  ...                                         │
│  ways: 1  1  2  3  5  8  13 21 ...  ← Fibonacci!                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def climbStairs(n: int) -> int:
    """
    Count distinct ways to climb n stairs.

    State: dp[i] = number of ways to reach step i
    Recurrence: dp[i] = dp[i-1] + dp[i-2]
    Base: dp[0] = 1, dp[1] = 1

    Time: O(n)
    Space: O(1) - only need last two values
    """
    if n <= 1:
        return 1

    # Only need to track last two values
    prev2 = 1  # Ways to reach step i-2
    prev1 = 1  # Ways to reach step i-1

    for i in range(2, n + 1):
        # Ways to reach step i = ways from i-1 + ways from i-2
        current = prev1 + prev2

        # Shift values for next iteration
        prev2 = prev1
        prev1 = current

    return prev1
```

### Complexity
- **Time**: O(n) - single pass
- **Space**: O(1) - only tracking two values

### Common Mistakes
- Using recursion without memoization (exponential time)
- Off-by-one errors in base cases
- Not recognizing the Fibonacci pattern

### Related Problems
- LC #746 Min Cost Climbing Stairs
- LC #509 Fibonacci Number
- LC #1137 N-th Tribonacci Number
### Edge Cases
- Empty input → handle base case
- Single element → return directly
- Large input → check time complexity
- Boundary values → test edge conditions

---

## Problem 2: Min Cost Climbing Stairs (LC #746) - Easy

- [LeetCode](https://leetcode.com/problems/min-cost-climbing-stairs/)

### Problem Statement
Given array `cost` where `cost[i]` is the cost of step i. You can climb 1 or 2 steps. Find minimum cost to reach the top (beyond the last step). You can start from step 0 or step 1.

### Examples
```
Input: cost = [10,15,20]
Output: 15
Explanation: Start at step 1 (cost 15), climb 2 steps to top

Input: cost = [1,100,1,1,1,100,1,1,100,1]
Output: 6
```

### Video Explanation
- [NeetCode - Min Cost Climbing Stairs](https://www.youtube.com/watch?v=ktmzAZWkEZ0)
- [Take U Forward - Min Cost Stairs](https://www.youtube.com/watch?v=fqUxJmjU4Dk)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MINIMUM COST TO REACH EACH STEP                                            │
│                                                                             │
│  cost = [10, 15, 20]                                                       │
│                                                                             │
│  To reach step i, choose cheaper path:                                     │
│  - From step i-1: dp[i-1] + cost[i]                                        │
│  - From step i-2: dp[i-2] + cost[i]                                        │
│                                                                             │
│  dp[i] = cost[i] + min(dp[i-1], dp[i-2])                                   │
│                                                                             │
│  Step 0: cost = 10 (start here free)                                       │
│  Step 1: cost = 15 (start here free)                                       │
│  Step 2: cost = 20 + min(10, 15) = 30                                      │
│                                                                             │
│  Top (step 3): min(dp[1], dp[2]) = min(15, 30) = 15                        │
│                                                                             │
│  Visual:                                                                    │
│         TOP ←── We want to reach here                                      │
│          │                                                                  │
│    ┌─────┴─────┐                                                           │
│    │           │                                                           │
│  [20]        [20]    Step 2 (cost 20)                                      │
│    │           │                                                           │
│  [15]        [15]    Step 1 (cost 15) ← Start here, pay 15, jump to top   │
│    │           │                                                           │
│  [10]        [10]    Step 0 (cost 10)                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def minCostClimbingStairs(cost: list[int]) -> int:
    """
    Find minimum cost to reach top of stairs.

    State: dp[i] = minimum cost to reach step i
    Recurrence: dp[i] = cost[i] + min(dp[i-1], dp[i-2])
    Base: dp[0] = cost[0], dp[1] = cost[1]
    Answer: min(dp[n-1], dp[n-2]) - can reach top from either

    Time: O(n)
    Space: O(1)
    """
    n = len(cost)

    # Can start from step 0 or step 1
    prev2 = cost[0]  # Cost to reach step 0
    prev1 = cost[1]  # Cost to reach step 1

    for i in range(2, n):
        # Cost to reach step i = cost[i] + min cost to get here
        current = cost[i] + min(prev1, prev2)
        prev2 = prev1
        prev1 = current

    # Can reach top from last or second-to-last step
    return min(prev1, prev2)
```

### Complexity
- **Time**: O(n) - single pass
- **Space**: O(1) - only tracking two values

### Common Mistakes
- Forgetting you can start from step 0 OR step 1
- Not taking min of last two steps for final answer
- Adding cost when reaching top (top has no cost)

### Related Problems
- LC #70 Climbing Stairs
- LC #198 House Robber
- LC #322 Coin Change
### Edge Cases
- Empty input → handle base case
- Single element → return directly
- Large input → check time complexity
- Boundary values → test edge conditions

---

## Problem 3: Maximum Subarray (LC #53) - Easy/Medium

- [LeetCode](https://leetcode.com/problems/maximum-subarray/)

### Problem Statement
Given an integer array `nums`, find the contiguous subarray with the largest sum and return its sum.

### Examples
```
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has sum 6

Input: nums = [1]
Output: 1

Input: nums = [5,4,-1,7,8]
Output: 23 (entire array)
```

### Video Explanation
- [NeetCode - Maximum Subarray](https://www.youtube.com/watch?v=5WZl3MMT0Eg)
- [Take U Forward - Kadane's Algorithm](https://www.youtube.com/watch?v=w_KEocd__20)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  KADANE'S ALGORITHM                                                         │
│                                                                             │
│  nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]                                    │
│                                                                             │
│  Key insight: At each position, decide:                                    │
│  - Start a NEW subarray here, OR                                           │
│  - EXTEND the previous subarray                                            │
│                                                                             │
│  If previous sum is NEGATIVE, starting fresh is better!                    │
│                                                                             │
│  Step by step:                                                              │
│  i=0: num=-2, current=max(-2, 0+(-2))=-2, max=-2                           │
│  i=1: num=1,  current=max(1, -2+1)=1,     max=1                            │
│  i=2: num=-3, current=max(-3, 1+(-3))=-2, max=1                            │
│  i=3: num=4,  current=max(4, -2+4)=4,     max=4   ← Start fresh!           │
│  i=4: num=-1, current=max(-1, 4+(-1))=3,  max=4                            │
│  i=5: num=2,  current=max(2, 3+2)=5,      max=5                            │
│  i=6: num=1,  current=max(1, 5+1)=6,      max=6   ← New maximum!           │
│  i=7: num=-5, current=max(-5, 6+(-5))=1,  max=6                            │
│  i=8: num=4,  current=max(4, 1+4)=5,      max=6                            │
│                                                                             │
│  Answer: 6 (subarray [4, -1, 2, 1])                                        │
│                                                                             │
│  Visual of best subarray:                                                   │
│  [-2, 1, -3, 4, -1, 2, 1, -5, 4]                                           │
│               └──────────┘                                                  │
│                 sum = 6                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def maxSubArray(nums: list[int]) -> int:
    """
    Find maximum sum of contiguous subarray (Kadane's Algorithm).

    State: dp[i] = maximum subarray sum ENDING at index i
    Recurrence: dp[i] = max(nums[i], dp[i-1] + nums[i])
    Answer: max(dp[i]) for all i

    Time: O(n)
    Space: O(1)
    """
    # Track max sum ending at current position
    current_sum = nums[0]

    # Track overall maximum
    max_sum = nums[0]

    for i in range(1, len(nums)):
        # Decision: start fresh at nums[i] or extend previous subarray
        current_sum = max(nums[i], current_sum + nums[i])

        # Update overall maximum
        max_sum = max(max_sum, current_sum)

    return max_sum
```

### Complexity
- **Time**: O(n) - single pass
- **Space**: O(1) - only tracking current and max sum

### Common Mistakes
- Not handling all-negative arrays correctly
- Resetting max_sum to 0 instead of first element
- Forgetting that we need max ending at each position

### Related Problems
- LC #152 Maximum Product Subarray
- LC #918 Maximum Sum Circular Subarray
- LC #1749 Maximum Absolute Sum of Any Subarray
### Edge Cases
- Empty input → handle base case
- Single element → return directly
- Large input → check time complexity
- Boundary values → test edge conditions

---

## Problem 4: House Robber (LC #198) - Easy/Medium

- [LeetCode](https://leetcode.com/problems/house-robber/)

### Problem Statement
You are a robber planning to rob houses along a street. Each house has money, but you cannot rob two adjacent houses (alarm will go off). Find the maximum amount you can rob.

### Examples
```
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 0 (1) + house 2 (3) = 4

Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 0 (2) + house 2 (9) + house 4 (1) = 12
```

### Video Explanation
- [NeetCode - House Robber](https://www.youtube.com/watch?v=73r3KWiEvyk)
- [Take U Forward - House Robber](https://www.youtube.com/watch?v=GrMBfJNk_NY)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ROB OR SKIP EACH HOUSE                                                     │
│                                                                             │
│  nums = [2, 7, 9, 3, 1]                                                    │
│                                                                             │
│  At each house, two choices:                                               │
│  1. SKIP it: keep previous maximum                                         │
│  2. ROB it: add its value to max from 2 houses ago                         │
│                                                                             │
│  dp[i] = max money from houses 0 to i                                      │
│  dp[i] = max(dp[i-1], dp[i-2] + nums[i])                                   │
│              skip     rob                                                   │
│                                                                             │
│  Step by step:                                                              │
│  House 0: dp[0] = 2                                                        │
│  House 1: dp[1] = max(2, 7) = 7       (rob house 1, skip house 0)          │
│  House 2: dp[2] = max(7, 2+9) = 11    (rob houses 0 and 2)                 │
│  House 3: dp[3] = max(11, 7+3) = 11   (skip house 3)                       │
│  House 4: dp[4] = max(11, 11+1) = 12  (rob houses 0, 2, and 4)             │
│                                                                             │
│  Visual:                                                                    │
│  [2]  [7]  [9]  [3]  [1]                                                   │
│   ✓    ✗    ✓    ✗    ✓   = 2 + 9 + 1 = 12                                │
│                                                                             │
│  Note: We can't take 7+9 because they're adjacent!                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def rob(nums: list[int]) -> int:
    """
    Maximum money from non-adjacent houses.

    State: dp[i] = max money from first i houses
    Recurrence: dp[i] = max(dp[i-1], dp[i-2] + nums[i])

    Time: O(n)
    Space: O(1)
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    # dp[i-2] and dp[i-1]
    prev2 = 0
    prev1 = nums[0]

    for i in range(1, len(nums)):
        # Decision: skip house i (prev1) or rob it (prev2 + nums[i])
        current = max(prev1, prev2 + nums[i])

        # Shift for next iteration
        prev2 = prev1
        prev1 = current

    return prev1
```

### Complexity
- **Time**: O(n) - single pass
- **Space**: O(1) - only tracking two values

### Common Mistakes
- Not handling single house case
- Wrong initialization of prev1 and prev2
- Forgetting that we CAN skip multiple houses in a row

### Related Problems
- LC #213 House Robber II (circular)
- LC #337 House Robber III (tree)
- LC #740 Delete and Earn
### Edge Cases
- Empty input → handle base case
- Single element → return directly
- Large input → check time complexity
- Boundary values → test edge conditions

---

## Problem 5: Best Time to Buy and Sell Stock (LC #121) - Easy

- [LeetCode](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

### Problem Statement
Given array `prices` where `prices[i]` is the stock price on day i. You can buy on one day and sell on a later day. Return the maximum profit (or 0 if no profit possible).

### Examples
```
Input: prices = [7,1,5,3,6,4]
Output: 5 (buy at 1, sell at 6)

Input: prices = [7,6,4,3,1]
Output: 0 (prices only decrease, no profit possible)
```

### Video Explanation
- [NeetCode - Best Time to Buy/Sell Stock](https://www.youtube.com/watch?v=1pkOgXD63yU)
- [Take U Forward - Stock Buy Sell](https://www.youtube.com/watch?v=eMSfBgbiEjk)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TRACK MINIMUM PRICE SEEN SO FAR                                            │
│                                                                             │
│  prices = [7, 1, 5, 3, 6, 4]                                               │
│                                                                             │
│  Key insight: At each day, if we sell today, what's the best profit?       │
│  Best profit = today's price - minimum price seen so far                   │
│                                                                             │
│  Day 0: price=7, min=7,  profit=7-7=0,  max_profit=0                       │
│  Day 1: price=1, min=1,  profit=1-1=0,  max_profit=0  ← New minimum!       │
│  Day 2: price=5, min=1,  profit=5-1=4,  max_profit=4                       │
│  Day 3: price=3, min=1,  profit=3-1=2,  max_profit=4                       │
│  Day 4: price=6, min=1,  profit=6-1=5,  max_profit=5  ← Best!              │
│  Day 5: price=4, min=1,  profit=4-1=3,  max_profit=5                       │
│                                                                             │
│  Visual:                                                                    │
│  7 ─●                                                                       │
│  6 ─┼───────────────●                                                       │
│  5 ─┼───────●       │                                                       │
│  4 ─┼───────┼───────┼───●                                                   │
│  3 ─┼───────┼───●   │                                                       │
│  2 ─┼───────┼───────┼                                                       │
│  1 ─┼───●───┼───────┼   Buy here                                           │
│     0   1   2   3   4   5                                                   │
│                                                                             │
│  Buy at day 1 (price 1), sell at day 4 (price 6) = profit 5                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def maxProfit(prices: list[int]) -> int:
    """
    Maximum profit from single buy-sell transaction.

    Strategy: Track minimum price seen so far.
    At each day: potential profit = current_price - min_price_so_far

    Time: O(n)
    Space: O(1)
    """
    if not prices:
        return 0

    min_price = float('inf')  # Minimum price seen so far
    max_profit = 0            # Maximum profit so far

    for price in prices:
        # Update minimum price (best day to buy)
        min_price = min(min_price, price)

        # Calculate profit if we sell today
        profit = price - min_price

        # Update maximum profit
        max_profit = max(max_profit, profit)

    return max_profit
```

### Complexity
- **Time**: O(n) - single pass
- **Space**: O(1) - only tracking min and max

### Common Mistakes
- Trying to find max and min separately (max must come AFTER min)
- Returning negative profit instead of 0
- Using nested loops (O(n²) when O(n) is possible)

### Related Problems
- LC #122 Best Time to Buy and Sell Stock II
- LC #123 Best Time to Buy and Sell Stock III
- LC #188 Best Time to Buy and Sell Stock IV
### Edge Cases
- Empty input → handle base case
- Single element → return directly
- Large input → check time complexity
- Boundary values → test edge conditions

---

## Problem 6: Pascal's Triangle (LC #118) - Easy

- [LeetCode](https://leetcode.com/problems/pascals-triangle/)

### Problem Statement
Given an integer `numRows`, return the first numRows of Pascal's triangle.

### Examples
```
Input: numRows = 5
Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]

Visual:
    1
   1 1
  1 2 1
 1 3 3 1
1 4 6 4 1
```

### Video Explanation
- [NeetCode - Pascal's Triangle](https://www.youtube.com/watch?v=nPVEaB3AjUM)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  EACH ELEMENT = SUM OF TWO ABOVE                                            │
│                                                                             │
│         1           Row 0                                                   │
│        1 1          Row 1                                                   │
│       1 2 1         Row 2: middle = 1+1                                    │
│      1 3 3 1        Row 3: 3 = 1+2, 3 = 2+1                                │
│     1 4 6 4 1       Row 4: 4 = 1+3, 6 = 3+3, 4 = 3+1                       │
│                                                                             │
│  Pattern:                                                                   │
│  - First and last elements are always 1                                    │
│  - Middle elements: triangle[i][j] = triangle[i-1][j-1] + triangle[i-1][j] │
│                                                                             │
│  Building row 4 from row 3 [1, 3, 3, 1]:                                   │
│                                                                             │
│      [1, 3, 3, 1]                                                          │
│       ↓  ↘↓  ↘↓  ↘↓                                                        │
│      [1, 4, 6, 4, 1]                                                       │
│                                                                             │
│  Position 1: 1 + 3 = 4                                                     │
│  Position 2: 3 + 3 = 6                                                     │
│  Position 3: 3 + 1 = 4                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def generate(numRows: int) -> list[list[int]]:
    """
    Generate Pascal's triangle.

    Each element = sum of two elements above it.
    dp[i][j] = dp[i-1][j-1] + dp[i-1][j]

    Time: O(numRows²)
    Space: O(numRows²) for output
    """
    triangle = []

    for row_num in range(numRows):
        # Create row with all 1s (edges are always 1)
        row = [1] * (row_num + 1)

        # Fill in middle values
        for j in range(1, row_num):
            # Sum of two values from previous row
            row[j] = triangle[row_num - 1][j - 1] + triangle[row_num - 1][j]

        triangle.append(row)

    return triangle
```

### Complexity
- **Time**: O(n²) where n = numRows
- **Space**: O(n²) for the triangle

### Common Mistakes
- Off-by-one errors in row/column indices
- Forgetting that edges are always 1
- Not handling numRows = 0 or 1

### Related Problems
- LC #119 Pascal's Triangle II (single row)
- LC #120 Triangle (minimum path sum)
- LC #931 Minimum Falling Path Sum
### Edge Cases
- Empty input → handle base case
- Single element → return directly
- Large input → check time complexity
- Boundary values → test edge conditions

---

## Problem 7: Is Subsequence (LC #392) - Easy

- [LeetCode](https://leetcode.com/problems/is-subsequence/)

### Problem Statement
Given two strings `s` and `t`, return true if `s` is a subsequence of `t`. A subsequence is formed by deleting some (or no) characters without changing the order of remaining characters.

### Examples
```
Input: s = "abc", t = "ahbgdc"
Output: true

Input: s = "axc", t = "ahbgdc"
Output: false (no 'x' in t)
```

### Video Explanation
- [NeetCode - Is Subsequence](https://www.youtube.com/watch?v=99RVfqklbCE)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  TWO POINTER APPROACH                                                       │
│                                                                             │
│  s = "abc", t = "ahbgdc"                                                   │
│                                                                             │
│  Walk through t, try to match characters of s in order:                    │
│                                                                             │
│  t: a h b g d c                                                            │
│     ↑                 s[0]='a' matches! s_ptr=1                            │
│       ↑               s[1]='b'? no, 'h'≠'b'                                │
│         ↑             s[1]='b' matches! s_ptr=2                            │
│           ↑           s[2]='c'? no, 'g'≠'c'                                │
│             ↑         s[2]='c'? no, 'd'≠'c'                                │
│               ↑       s[2]='c' matches! s_ptr=3                            │
│                                                                             │
│  s_ptr reached end of s → TRUE, s is subsequence of t                      │
│                                                                             │
│  Visual:                                                                    │
│  t = "a h b g d c"                                                         │
│       ↑   ↑     ↑                                                          │
│       a   b     c  ← These form subsequence "abc"                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def isSubsequence(s: str, t: str) -> bool:
    """
    Check if s is subsequence of t.

    Two-pointer approach: try to match all characters of s in t.

    Time: O(n) where n = len(t)
    Space: O(1)
    """
    if not s:
        return True

    s_ptr = 0  # Pointer for s

    for char in t:
        if char == s[s_ptr]:
            s_ptr += 1
            if s_ptr == len(s):
                return True  # All characters matched

    return False  # Didn't match all of s
```

### Complexity
- **Time**: O(n) where n = len(t)
- **Space**: O(1) - only using pointers

### Common Mistakes
- Forgetting empty string is subsequence of everything
- Moving s_ptr even when characters don't match
- Not handling case where s is longer than t

### Related Problems
- LC #792 Number of Matching Subsequences
- LC #1143 Longest Common Subsequence
- LC #516 Longest Palindromic Subsequence
### Edge Cases
- Empty input → handle base case
- Single element → return directly
- Large input → check time complexity
- Boundary values → test edge conditions

---

## Summary: Easy DP Problems Checklist

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | Climbing Stairs | Fibonacci-like | O(n) | O(1) |
| 2 | Min Cost Climbing | Min of two choices | O(n) | O(1) |
| 3 | Maximum Subarray | Kadane's algorithm | O(n) | O(1) |
| 4 | House Robber | Take or skip | O(n) | O(1) |
| 5 | Buy/Sell Stock | Track min price | O(n) | O(1) |
| 6 | Pascal's Triangle | Sum from above | O(n²) | O(n²) |
| 7 | Is Subsequence | Two pointers | O(n) | O(1) |

---

## Practice More Easy Problems

- [ ] LC #509 - Fibonacci Number
- [ ] LC #1137 - N-th Tribonacci Number
- [ ] LC #338 - Counting Bits
- [ ] LC #303 - Range Sum Query - Immutable
- [ ] LC #119 - Pascal's Triangle II
