# Dynamic Programming - Medium Problems

## Problem 1: House Robber (LC #198) - Medium

- [LeetCode](https://leetcode.com/problems/house-robber/)

### Problem Statement
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, but adjacent houses have security systems connected. If two adjacent houses are broken into on the same night, it will alert the police. Return the maximum amount of money you can rob without alerting the police.

### Video Explanation
- [NeetCode - House Robber](https://www.youtube.com/watch?v=73r3KWiEvyk)

### Examples
```
Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 0 and 2: 1 + 3 = 4

Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 0, 2, and 4: 2 + 9 + 1 = 12

Input: nums = [2,1,1,2]
Output: 4
Explanation: Rob house 0 and 3: 2 + 2 = 4
```

### Intuition Development
```
Classic DP: At each house, decide to ROB or SKIP!

nums = [2, 7, 9, 3, 1]

┌─────────────────────────────────────────────────────────────────┐
│ State: dp[i] = max money robbing houses 0..i                    │
│                                                                  │
│ Choice at house i:                                               │
│   1. Rob house i:  dp[i-2] + nums[i]  (skip previous)           │
│   2. Skip house i: dp[i-1]            (keep previous best)      │
│                                                                  │
│ Transition: dp[i] = max(dp[i-2] + nums[i], dp[i-1])             │
│                                                                  │
│ Walkthrough:                                                     │
│   House 0: dp[0] = 2                                            │
│   House 1: dp[1] = max(0+7, 2) = 7                              │
│   House 2: dp[2] = max(2+9, 7) = 11                             │
│   House 3: dp[3] = max(7+3, 11) = 11                            │
│   House 4: dp[4] = max(11+1, 11) = 12 ★                         │
│                                                                  │
│ Space optimization: Only need prev2, prev1!                     │
│   prev2 → prev1 → current                                       │
│   Reduces O(n) space to O(1)                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def rob(nums: list[int]) -> int:
    """
    Maximum money from non-adjacent houses.

    Strategy:
    - dp[i] = max money robbing houses 0..i
    - Either rob current + dp[i-2], or skip and take dp[i-1]

    Time: O(n)
    Space: O(1) optimized
    """
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    # Space optimized: only need prev two values
    prev2 = 0  # dp[i-2]
    prev1 = 0  # dp[i-1]

    for num in nums:
        current = max(prev2 + num, prev1)
        prev2 = prev1
        prev1 = current

    return prev1
```

### Complexity
- **Time**: O(n) - Single pass through array
- **Space**: O(1) - Only two variables

### Edge Cases
- Empty array: Return 0
- Single house: Return that value
- Two houses: Return max of two
- All zeros: Return 0

---

## Problem 2: Coin Change (LC #322) - Medium

- [LeetCode](https://leetcode.com/problems/coin-change/)

### Problem Statement
You are given an integer array `coins` representing coins of different denominations and an integer `amount` representing a total amount of money. Return the fewest number of coins needed to make up that amount. If that amount cannot be made up by any combination of the coins, return -1. You may use each coin an unlimited number of times.

### Video Explanation
- [NeetCode - Coin Change](https://www.youtube.com/watch?v=H9bfqozjoqs)

### Examples
```
Input: coins = [1,2,5], amount = 11
Output: 3 (5 + 5 + 1)

Input: coins = [2], amount = 3
Output: -1
```

### Intuition Development
```
Unbounded Knapsack: Use each coin unlimited times!

coins = [1, 2, 5], amount = 11

┌─────────────────────────────────────────────────────────────────┐
│ State: dp[i] = minimum coins to make amount i                   │
│                                                                  │
│ Transition: For each coin, can we improve?                      │
│   dp[i] = min(dp[i], dp[i - coin] + 1)                          │
│                                                                  │
│ Base case: dp[0] = 0 (0 coins for amount 0)                     │
│ Initialize others to infinity (impossible initially)            │
│                                                                  │
│ Filling the table:                                               │
│   Amount:  0  1  2  3  4  5  6  7  8  9  10  11                 │
│   Coins:   0  1  1  2  2  1  2  2  3  3   2   3                 │
│                                                                  │
│ dp[11] = 3: One 5 + one 5 + one 1 = 5+5+1 = 11 ✓                │
│                                                                  │
│ Key insight: For each amount, try ALL coins!                    │
│   dp[11] = min(dp[10]+1, dp[9]+1, dp[6]+1)                      │
│          = min(2+1, 3+1, 2+1) = 3                               │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def coinChange(coins: list[int], amount: int) -> int:
    """
    Minimum coins to make amount.

    Strategy:
    - dp[i] = min coins for amount i
    - For each coin, dp[i] = min(dp[i], dp[i-coin] + 1)

    Time: O(amount * len(coins))
    Space: O(amount)
    """
    # Initialize with impossible value
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
```

### Complexity
- **Time**: O(amount × coins) - Fill array, check each coin
- **Space**: O(amount) - DP array

### Edge Cases
- Amount 0: Return 0 (no coins needed)
- Impossible amount: Return -1 (e.g., [2] for amount 3)
- Coin equals amount: Return 1
- Single coin: Amount must be divisible

---

## Problem 3: Longest Increasing Subsequence (LC #300) - Medium

- [LeetCode](https://leetcode.com/problems/longest-increasing-subsequence/)

### Problem Statement
Given an integer array `nums`, return the length of the longest **strictly increasing** subsequence. A subsequence is an array that can be derived by deleting some or no elements without changing the order of remaining elements.

### Video Explanation
- [NeetCode - Longest Increasing Subsequence](https://www.youtube.com/watch?v=cjWnW0hdF1Y)

### Examples
```
Input: nums = [10,9,2,5,3,7,101,18]
Output: 4 ([2,3,7,101] or [2,3,7,18])

Input: nums = [0,1,0,3,2,3]
Output: 4 ([0,1,2,3])
```

### Intuition Development
```
Two approaches: O(n²) DP vs O(n log n) Binary Search!

nums = [10, 9, 2, 5, 3, 7, 101, 18]

┌─────────────────────────────────────────────────────────────────┐
│ O(n²) Approach: DP ending at each index                        │
│                                                                  │
│ dp[i] = length of LIS ending at index i                         │
│                                                                  │
│ For each i, check all j < i:                                    │
│   If nums[j] < nums[i]: dp[i] = max(dp[i], dp[j] + 1)           │
│                                                                  │
│ nums:  10   9   2   5   3   7  101  18                          │
│ dp:     1   1   1   2   2   3    4   4                          │
│                                                                  │
│ LIS = 4: [2, 3, 7, 101] or [2, 3, 7, 18]                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ O(n log n) Approach: Patience Sorting / Binary Search           │
│                                                                  │
│ Maintain "tails" array:                                          │
│   tails[i] = smallest tail of all LIS of length i+1            │
│                                                                  │
│ For each num:                                                    │
│   Binary search for position in tails                           │
│   Replace or extend                                             │
│                                                                  │
│ nums: [10, 9, 2, 5, 3, 7, 101, 18]                              │
│ tails evolution:                                                 │
│   [10] → [9] → [2] → [2,5] → [2,3] → [2,3,7]                   │
│   → [2,3,7,101] → [2,3,7,18]                                    │
│                                                                  │
│ Final length = 4                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def lengthOfLIS(nums: list[int]) -> int:
    """
    Longest Increasing Subsequence - O(n²) approach.

    Strategy:
    - dp[i] = length of LIS ending at index i
    - For each j < i, if nums[j] < nums[i], dp[i] = max(dp[i], dp[j] + 1)

    Time: O(n²)
    Space: O(n)
    """
    n = len(nums)
    dp = [1] * n  # Each element is LIS of length 1

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


def lengthOfLIS_optimized(nums: list[int]) -> int:
    """
    LIS with O(n log n) using binary search.

    Strategy:
    - Maintain array 'tails' where tails[i] = smallest tail of LIS of length i+1
    - For each num, binary search for position

    Time: O(n log n)
    Space: O(n)
    """
    from bisect import bisect_left

    tails = []

    for num in nums:
        pos = bisect_left(tails, num)

        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num

    return len(tails)
```

### Complexity
- **O(n²)**: Time O(n²), Space O(n)
- **O(n log n)**: Time O(n log n), Space O(n)

### Edge Cases
- Single element: Return 1
- Strictly decreasing: Return 1 (any single element)
- Strictly increasing: Return n
- All same elements: Return 1

---

## Problem 4: Unique Paths (LC #62) - Medium

- [LeetCode](https://leetcode.com/problems/unique-paths/)

### Problem Statement
There is a robot on an `m x n` grid. The robot starts at the top-left corner and can only move either down or right at any point. The robot is trying to reach the bottom-right corner. Return the number of possible unique paths.

### Video Explanation
- [NeetCode - Unique Paths](https://www.youtube.com/watch?v=IlEsdxuD4lY)

### Examples
```
Input: m = 3, n = 7
Output: 28

Input: m = 3, n = 2
Output: 3
```

### Intuition Development
```
Grid DP: Each cell = sum of paths from top + left!

m = 3, n = 7

┌─────────────────────────────────────────────────────────────────┐
│ State: dp[i][j] = number of unique paths to (i, j)              │
│                                                                  │
│ Base case:                                                       │
│   First row: only way is going right → all 1s                  │
│   First col: only way is going down → all 1s                   │
│                                                                  │
│ Transition: dp[i][j] = dp[i-1][j] + dp[i][j-1]                  │
│                                                                  │
│ Grid visualization:                                              │
│   ┌────────────────────────────────┐                            │
│   │  1    1    1    1    1    1  1 │  ← first row (all 1s)      │
│   │  1    2    3    4    5    6  7 │                            │
│   │  1    3    6   10   15   21 28 │  ← answer at (2,6)         │
│   └────────────────────────────────┘                            │
│   ↑                                                              │
│   first col (all 1s)                                            │
│                                                                  │
│ Space optimization: Only need previous row!                     │
│ Math alternative: C(m+n-2, m-1) = choose (m-1) down moves      │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def uniquePaths(m: int, n: int) -> int:
    """
    Count unique paths in grid.

    Strategy:
    - dp[i][j] = paths to reach cell (i,j)
    - dp[i][j] = dp[i-1][j] + dp[i][j-1]

    Time: O(m * n)
    Space: O(n) optimized
    """
    # Space optimized: only need previous row
    dp = [1] * n

    for i in range(1, m):
        for j in range(1, n):
            dp[j] = dp[j] + dp[j - 1]

    return dp[n - 1]


def uniquePaths_math(m: int, n: int) -> int:
    """
    Mathematical solution using combinations.

    Total moves = (m-1) + (n-1) = m+n-2
    Choose (m-1) down moves: C(m+n-2, m-1)

    Time: O(min(m, n))
    Space: O(1)
    """
    from math import comb
    return comb(m + n - 2, m - 1)
```

### Complexity
- **DP**: Time O(m×n), Space O(n)
- **Math**: Time O(min(m,n)), Space O(1)

### Edge Cases
- 1×1 grid: Return 1 (already at destination)
- 1×n or m×1: Return 1 (only one path)
- Large grid: Use math formula to avoid overflow

---

## Problem 5: Jump Game (LC #55) - Medium

- [LeetCode](https://leetcode.com/problems/jump-game/)

### Problem Statement
You are given an integer array `nums`. You are initially positioned at the first index, and each element represents your **maximum jump length** at that position. Return `true` if you can reach the last index, or `false` otherwise.

### Video Explanation
- [NeetCode - Jump Game](https://www.youtube.com/watch?v=Yan0cv2cLy8)

### Examples
```
Input: nums = [2,3,1,1,4]
Output: true (0 → 1 → 4)

Input: nums = [3,2,1,0,4]
Output: false (stuck at index 3)
```

### Intuition Development
```
Greedy is optimal here! Track maximum reachable index.

nums = [2, 3, 1, 1, 4]

┌─────────────────────────────────────────────────────────────────┐
│ Key insight: We can always reach any index ≤ max_reach         │
│                                                                  │
│ Track max_reach as we go:                                        │
│                                                                  │
│ Index 0: nums[0]=2, max_reach = max(0, 0+2) = 2                 │
│ Index 1: 1 ≤ 2 ✓, max_reach = max(2, 1+3) = 4                   │
│ Index 2: 2 ≤ 4 ✓, max_reach = max(4, 2+1) = 4                   │
│ Index 3: 3 ≤ 4 ✓, max_reach = max(4, 3+1) = 4                   │
│ Index 4: 4 ≤ 4 ✓, REACHED LAST! Return true                     │
│                                                                  │
│ Failure case: nums = [3, 2, 1, 0, 4]                            │
│ Index 0: max_reach = 3                                          │
│ Index 1: max_reach = 3                                          │
│ Index 2: max_reach = 3                                          │
│ Index 3: max_reach = 3                                          │
│ Index 4: 4 > 3! Can't reach! Return false                       │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def canJump(nums: list[int]) -> bool:
    """
    Check if we can reach last index.

    Strategy (Greedy):
    - Track maximum reachable index
    - If current index > max_reach, return False

    Time: O(n)
    Space: O(1)
    """
    max_reach = 0

    for i in range(len(nums)):
        if i > max_reach:
            return False

        max_reach = max(max_reach, i + nums[i])

        if max_reach >= len(nums) - 1:
            return True

    return True


def canJump_dp(nums: list[int]) -> bool:
    """
    DP approach (less efficient but educational).

    dp[i] = True if we can reach index i

    Time: O(n²)
    Space: O(n)
    """
    n = len(nums)
    dp = [False] * n
    dp[0] = True

    for i in range(n):
        if not dp[i]:
            continue

        for j in range(1, nums[i] + 1):
            if i + j < n:
                dp[i + j] = True

    return dp[n - 1]
```

### Complexity
- **Greedy**: Time O(n), Space O(1)
- **DP**: Time O(n²), Space O(n)

### Edge Cases
- Single element array: Always reachable
- All zeros: Only reachable if length is 1
- First element is 0: Can't move, return length == 1
- First element ≥ length: Return true

---

## Problem 6: Word Break (LC #139) - Medium

- [LeetCode](https://leetcode.com/problems/word-break/)

### Problem Statement
Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words. The same word in the dictionary may be reused multiple times.

### Video Explanation
- [NeetCode - Word Break](https://www.youtube.com/watch?v=Sx9NNgInc3A)

### Examples
```
Input: s = "leetcode", wordDict = ["leet","code"]
Output: true ("leet" + "code")

Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true ("apple" + "pen" + "apple")
```

### Intuition Development
```
DP: Can we build valid segmentation ending at each position?

s = "leetcode", wordDict = ["leet", "code"]

┌─────────────────────────────────────────────────────────────────┐
│ State: dp[i] = True if s[0:i] can be segmented                  │
│                                                                  │
│ Base case: dp[0] = True (empty string is valid)                 │
│                                                                  │
│ Transition: For each i, check all possible last words:          │
│   For j from 0 to i:                                            │
│     If dp[j] is True AND s[j:i] in wordDict:                    │
│       dp[i] = True                                               │
│                                                                  │
│ Walkthrough for "leetcode":                                      │
│   dp[0] = True  (empty)                                         │
│   dp[1] = "l" in dict? No                                       │
│   dp[2] = "le" in dict? No                                      │
│   dp[3] = "lee" in dict? No                                     │
│   dp[4] = dp[0] AND "leet" in dict? Yes! dp[4] = True           │
│   dp[5] = "c" or "leetc" in dict? No                            │
│   dp[6] = "co" or "leetco" in dict? No                          │
│   dp[7] = "cod" or "leetcod" in dict? No                        │
│   dp[8] = dp[4] AND "code" in dict? Yes! dp[8] = True ★         │
│                                                                  │
│ Optimization: Only check word lengths that exist in dictionary  │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def wordBreak(s: str, wordDict: list[str]) -> bool:
    """
    Check if string can be segmented into dictionary words.

    Strategy:
    - dp[i] = True if s[0:i] can be segmented
    - For each i, check all possible last words

    Time: O(n² * m) where m = max word length
    Space: O(n)
    """
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True  # Empty string

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[n]


def wordBreak_optimized(s: str, wordDict: list[str]) -> bool:
    """
    Optimized: Only check word lengths that exist.

    Time: O(n * m * k) where k = number of words
    Space: O(n)
    """
    word_set = set(wordDict)
    max_len = max(len(w) for w in wordDict) if wordDict else 0
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(max(0, i - max_len), i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[n]
```

### Complexity
- **Time**: O(n² × m) or O(n × m × k) with optimization
- **Space**: O(n) for DP array + O(k) for word set

### Edge Cases
- Empty string: Return true
- Empty dictionary: Return false (can't form any word)
- Single character words: Each letter must be in dictionary
- Repeated words: Same word can be used multiple times

---

## Summary: Medium DP Problems

| # | Problem | Pattern | Key Insight |
|---|---------|---------|-------------|
| 1 | House Robber | 1D DP | max(skip, take + prev2) |
| 2 | Coin Change | Unbounded Knapsack | Try all coins for each amount |
| 3 | LIS | 1D DP or Binary Search | Track LIS ending at each index |
| 4 | Unique Paths | 2D Grid DP | Sum of top and left |
| 5 | Jump Game | Greedy/DP | Track max reachable |
| 6 | Word Break | 1D DP | Check all possible last words |

---

## DP Patterns Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DP PATTERN RECOGNITION                                   │
│                                                                             │
│  1D DP (Linear):                                                            │
│  • House Robber: dp[i] depends on dp[i-1], dp[i-2]                         │
│  • Climbing Stairs: dp[i] = dp[i-1] + dp[i-2]                              │
│  • LIS: dp[i] = max(dp[j] + 1) for all j < i                               │
│                                                                             │
│  2D DP (Grid):                                                              │
│  • Unique Paths: dp[i][j] = dp[i-1][j] + dp[i][j-1]                        │
│  • Minimum Path Sum: dp[i][j] = grid[i][j] + min(top, left)                │
│                                                                             │
│  Knapsack:                                                                  │
│  • 0/1 Knapsack: Each item used once                                        │
│  • Unbounded: Each item used unlimited (Coin Change)                        │
│                                                                             │
│  String DP:                                                                 │
│  • Word Break: Segment into dictionary words                                │
│  • Edit Distance: Transform one string to another                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
