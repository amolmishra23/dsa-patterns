# Prefix Sum - Medium Problems

## Problem 1: Continuous Subarray Sum (LC #523) - Medium

- [LeetCode](https://leetcode.com/problems/continuous-subarray-sum/)

### Problem Statement
Given an integer array `nums` and an integer `k`, return `true` if `nums` has a **good subarray** or `false` otherwise. A good subarray is a subarray where:
- Its length is at least 2, and
- The sum of the elements of the subarray is a multiple of `k`.

### Video Explanation
- [NeetCode - Continuous Subarray Sum](https://www.youtube.com/watch?v=OKcrLfR-8mE)

### Examples
```
Input: nums = [23,2,4,6,7], k = 6
Output: true
Explanation: [2,4] is a continuous subarray of size 2 whose sum = 6, which is a multiple of 6.

Input: nums = [23,2,6,4,7], k = 6
Output: true
Explanation: [23,2,6,4,7] sums to 42 = 7 × 6.

Input: nums = [23,2,6,4,7], k = 13
Output: false
```

### Intuition Development
```
Key Math Insight: Modular Arithmetic for Subarray Sums
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If prefix[j] % k == prefix[i] % k, then (prefix[j] - prefix[i]) % k == 0!

Example: nums = [23, 2, 4, 6, 7], k = 6

Index:      -1    0    1    2    3    4
Prefix:      0   23   25   29   35   42
Prefix % 6:  0    5    1    5    5    0
                  ↑         ↑
                  │         └── Same remainder (5) at index 2!
                  └──────────── First seen at index 0

Subarray [1,2] = nums[1..2] = [2,4], sum = 6, length = 2 ✓

Step-by-step:
┌─────────────────────────────────────────────────────────────┐
│ i=0: prefix=23, rem=5, map={0:-1}     → store {0:-1, 5:0}   │
│ i=1: prefix=25, rem=1, map has no 1   → store {.., 1:1}     │
│ i=2: prefix=29, rem=5, map has 5 at 0 → length=2-0=2 ≥ 2 ✓  │
│ FOUND! Return true                                           │
└─────────────────────────────────────────────────────────────┘

Why store first index only? We want MAXIMUM possible length.
Why initialize {0: -1}? Empty prefix (sum 0) at "virtual index -1".
```

### Solution
```python
def checkSubarraySum(nums: list[int], k: int) -> bool:
    """
    Check for subarray with sum divisible by k.

    Key insight: If prefix[j] % k == prefix[i] % k, then
    (prefix[j] - prefix[i]) % k == 0.

    Strategy:
    - Track remainder -> first index with that remainder
    - If same remainder seen at index >= 2 apart, found valid subarray

    Time: O(n)
    Space: O(min(n, k))
    """
    # Map: remainder -> first index where this remainder occurred
    # Initialize with 0: -1 (empty prefix has sum 0, at "index -1")
    remainder_index = {0: -1}

    prefix_sum = 0

    for i, num in enumerate(nums):
        prefix_sum += num
        remainder = prefix_sum % k

        if remainder in remainder_index:
            # Check if subarray length >= 2
            if i - remainder_index[remainder] >= 2:
                return True
        else:
            # Only store first occurrence
            remainder_index[remainder] = i

    return False
```

### Complexity
- **Time**: O(n) - Single pass through array
- **Space**: O(min(n, k)) - Hash map stores at most min(n, k) distinct remainders

### Edge Cases
- `k = 0`: Not possible per constraints (k ≥ 1)
- All zeros: `[0,0,0]` with any k → true (sum 0 is multiple of any k)
- Single element: Never valid (need length ≥ 2)
- Negative numbers: Not in constraints but would still work with modular arithmetic

---

## Problem 2: Binary Subarrays With Sum (LC #930) - Medium

- [LeetCode](https://leetcode.com/problems/binary-subarrays-with-sum/)

### Problem Statement
Given a binary array `nums` and an integer `goal`, return the number of non-empty subarrays with a sum equal to `goal`. A binary array contains only 0s and 1s.

### Video Explanation
- [NeetCode - Binary Subarrays With Sum](https://www.youtube.com/watch?v=XnMdNUkX6VM)

### Examples
```
Input: nums = [1,0,1,0,1], goal = 2
Output: 4
Explanation: The 4 subarrays are:
  - [1,0,1] at indices [0,2]
  - [1,0,1,0] at indices [0,3]
  - [0,1,0,1] at indices [1,4]
  - [1,0,1] at indices [2,4]

Input: nums = [0,0,0,0,0], goal = 0
Output: 15
Explanation: Every subarray sums to 0.

Input: nums = [1,1,1,1,1], goal = 3
Output: 3
```

### Intuition Development
```
This is "Subarray Sum Equals K" with a binary twist!

nums = [1, 0, 1, 0, 1], goal = 2

Prefix sums: [0, 1, 1, 2, 2, 3]
              ↑
              Virtual index -1

For each position, count how many previous prefixes satisfy:
  prefix[j] - prefix[i] = goal
  prefix[i] = prefix[j] - goal

Step-by-step:
┌──────────────────────────────────────────────────────────────┐
│ i=0: num=1, prefix=1, need=1-2=-1, count+=0, map={0:1,1:1}   │
│ i=1: num=0, prefix=1, need=1-2=-1, count+=0, map={0:1,1:2}   │
│ i=2: num=1, prefix=2, need=2-2=0,  count+=1, map={..,2:1}    │
│ i=3: num=0, prefix=2, need=2-2=0,  count+=1, map={..,2:2}    │
│ i=4: num=1, prefix=3, need=3-2=1,  count+=2, map={..,3:1}    │
│                                                               │
│ Total count = 0 + 0 + 1 + 1 + 2 = 4 ✓                        │
└──────────────────────────────────────────────────────────────┘
```

### Solution
```python
def numSubarraysWithSum(nums: list[int], goal: int) -> int:
    """
    Count subarrays with sum = goal using prefix sum + hash map.

    Same as "Subarray Sum Equals K" pattern.

    Time: O(n)
    Space: O(n)
    """
    count = 0
    prefix_sum = 0

    # Map: prefix_sum -> count of occurrences
    prefix_count = {0: 1}

    for num in nums:
        prefix_sum += num

        # How many previous prefixes have sum = prefix_sum - goal?
        target = prefix_sum - goal
        if target in prefix_count:
            count += prefix_count[target]

        # Record current prefix sum
        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1

    return count


def numSubarraysWithSum_sliding(nums: list[int], goal: int) -> int:
    """
    Alternative: Sliding window approach.

    count(sum <= goal) - count(sum <= goal-1) = count(sum == goal)

    Time: O(n)
    Space: O(1)
    """
    def at_most(k):
        """Count subarrays with sum <= k."""
        if k < 0:
            return 0

        count = 0
        left = 0
        current_sum = 0

        for right in range(len(nums)):
            current_sum += nums[right]

            while current_sum > k:
                current_sum -= nums[left]
                left += 1

            # All subarrays ending at right with sum <= k
            count += right - left + 1

        return count

    return at_most(goal) - at_most(goal - 1)
```

### Complexity
- **Time**: O(n) - Single pass (or two passes for sliding window approach)
- **Space**: O(n) for hash map approach, O(1) for sliding window approach

### Edge Cases
- `goal = 0`: Count subarrays of all zeros
- All zeros with `goal > 0`: Return 0
- Single element array: Return 1 if element equals goal, else 0

---

## Problem 3: Count Number of Nice Subarrays (LC #1248) - Medium

- [LeetCode](https://leetcode.com/problems/count-number-of-nice-subarrays/)

### Problem Statement
Given an array of integers `nums` and an integer `k`, return the number of **nice** subarrays. A nice subarray is a contiguous subarray with exactly `k` odd numbers.

### Video Explanation
- [NeetCode - Count Number of Nice Subarrays](https://www.youtube.com/watch?v=j_QOv9OT9Og)

### Examples
```
Input: nums = [1,1,2,1,1], k = 3
Output: 2
Explanation: The nice subarrays are [1,1,2,1] and [1,2,1,1].

Input: nums = [2,4,6], k = 1
Output: 0
Explanation: No odd numbers, so no nice subarrays.

Input: nums = [2,2,2,1,2,2,1,2,2,2], k = 2
Output: 16
```

### Intuition Development
```
Transform the problem: odd → 1, even → 0
Now it's "Subarray Sum Equals K"!

nums = [1, 1, 2, 1, 1], k = 3
transform: [1, 1, 0, 1, 1]  (odd=1, even=0)

Odd count prefix: [0, 1, 2, 2, 3, 4]
                   ↑
                   Virtual index -1

For count = 3, need previous positions with odd_count = current - 3

┌──────────────────────────────────────────────────────────────┐
│ i=0: odd=1, cnt=1, need=-2, count+=0, map={0:1,1:1}          │
│ i=1: odd=1, cnt=2, need=-1, count+=0, map={..,2:1}           │
│ i=2: odd=0, cnt=2, need=-1, count+=0, map={..,2:2}           │
│ i=3: odd=1, cnt=3, need=0,  count+=1, map={..,3:1}  ← Found! │
│ i=4: odd=1, cnt=4, need=1,  count+=1, map={..,4:1}  ← Found! │
│                                                               │
│ Total = 2 ✓                                                   │
└──────────────────────────────────────────────────────────────┘
```

### Solution
```python
def numberOfSubarrays(nums: list[int], k: int) -> int:
    """
    Count subarrays with exactly k odd numbers.

    Transform: odd -> 1, even -> 0, then find subarrays with sum = k.

    Time: O(n)
    Space: O(n)
    """
    count = 0
    odd_count = 0  # Running count of odd numbers (prefix sum)

    # Map: odd_count -> number of times we've seen this count
    prefix_count = {0: 1}

    for num in nums:
        # Increment if odd
        if num % 2 == 1:
            odd_count += 1

        # How many previous positions had (odd_count - k) odds?
        target = odd_count - k
        if target in prefix_count:
            count += prefix_count[target]

        # Record current odd count
        prefix_count[odd_count] = prefix_count.get(odd_count, 0) + 1

    return count
```

### Complexity
- **Time**: O(n) - Single pass through array
- **Space**: O(n) - Hash map stores prefix counts

### Edge Cases
- No odd numbers with `k > 0`: Return 0
- All odd numbers: Every subarray of length k is nice
- `k = 0`: Count subarrays with no odd numbers (all even)

---

## Problem 4: Maximum Size Subarray Sum Equals k (LC #325) - Medium

- [LeetCode](https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/)

### Problem Statement
Given an integer array `nums` and an integer `k`, return the maximum length of a subarray that sums to `k`. If there is no subarray that sums to `k`, return 0. Note: This problem allows negative numbers!

### Video Explanation
- [LeetCode Premium - Max Size Subarray Sum Equals K](https://www.youtube.com/watch?v=8ygFQQ8n0uo)

### Examples
```
Input: nums = [1,-1,5,-2,3], k = 3
Output: 4
Explanation: [1,-1,5,-2] sums to 3 and has length 4.

Input: nums = [-2,-1,2,1], k = 1
Output: 2
Explanation: [-1,2] sums to 1.

Input: nums = [1,2,3], k = 10
Output: 0
Explanation: No subarray sums to 10.
```

### Intuition Development
```
Key insight: Store FIRST occurrence of each prefix sum!
(First occurrence gives LONGEST subarray)

nums = [1, -1, 5, -2, 3], k = 3

Prefix:  [0,  1,  0,  5,  3,  6]
Index:   -1,  0,  1,  2,  3,  4

For sum = 3 ending at index i, need prefix[j] = prefix[i] - 3

┌──────────────────────────────────────────────────────────────────┐
│ i=0: prefix=1, need=1-3=-2, not found, map={0:-1, 1:0}           │
│ i=1: prefix=0, need=0-3=-3, not found, 0 already in map (skip)   │
│ i=2: prefix=5, need=5-3=2, not found, map={.., 5:2}              │
│ i=3: prefix=3, need=3-3=0, found at -1! len=3-(-1)=4 ★           │
│ i=4: prefix=6, need=6-3=3, found at 3! len=4-3=1                 │
│                                                                   │
│ Maximum length = 4 ✓                                              │
└──────────────────────────────────────────────────────────────────┘

Why first occurrence only?
  If prefix=0 at index -1 and index 1, using -1 gives longer subarray!
```

### Solution
```python
def maxSubArrayLen(nums: list[int], k: int) -> int:
    """
    Find longest subarray with sum = k.

    Strategy:
    - Track first occurrence of each prefix sum
    - For each position, check if (prefix_sum - k) seen before
    - Calculate length and update max

    Time: O(n)
    Space: O(n)
    """
    max_length = 0
    prefix_sum = 0

    # Map: prefix_sum -> first index where this sum occurred
    first_index = {0: -1}  # Empty prefix at "index -1"

    for i, num in enumerate(nums):
        prefix_sum += num

        # Check if we can form subarray with sum k
        target = prefix_sum - k
        if target in first_index:
            length = i - first_index[target]
            max_length = max(max_length, length)

        # Only store first occurrence (for maximum length)
        if prefix_sum not in first_index:
            first_index[prefix_sum] = i

    return max_length
```

### Complexity
- **Time**: O(n) - Single pass through array
- **Space**: O(n) - Hash map stores first index of each prefix sum

### Edge Cases
- Negative numbers: Algorithm handles them correctly
- `k = 0`: Valid - look for subarrays that sum to 0
- No valid subarray: Return 0
- Entire array sums to k: Return n

---

## Problem 5: Subarray Product Less Than K (LC #713) - Medium

- [LeetCode](https://leetcode.com/problems/subarray-product-less-than-k/)

### Problem Statement
Given an array of positive integers `nums` and an integer `k`, return the number of contiguous subarrays where the product of all elements is strictly less than `k`.

### Video Explanation
- [NeetCode - Subarray Product Less Than K](https://www.youtube.com/watch?v=SxtxCSfSGlo)

### Examples
```
Input: nums = [10,5,2,6], k = 100
Output: 8
Explanation: The 8 subarrays are:
  [10], [5], [2], [6], [10,5], [5,2], [2,6], [5,2,6]
  Products: 10, 5, 2, 6, 50, 10, 12, 60 - all < 100

Input: nums = [1,2,3], k = 0
Output: 0
Explanation: No subarray has product < 0.

Input: nums = [1,1,1], k = 2
Output: 6
```

### Intuition Development
```
Note: This uses SLIDING WINDOW, not prefix sum!
(Prefix products can overflow; division is needed)

nums = [10, 5, 2, 6], k = 100

Sliding window: expand right, shrink left when product ≥ k

┌────────────────────────────────────────────────────────────────┐
│ r=0: prod=10 < 100, subarrays ending at 0: [10] → count=1      │
│      window: [10], count += (0-0+1) = 1                        │
│                                                                 │
│ r=1: prod=50 < 100, subarrays: [5], [10,5] → count=3           │
│      window: [10,5], count += (1-0+1) = 2                      │
│                                                                 │
│ r=2: prod=100 ≥ 100, shrink! prod=10, l=1                      │
│      prod=10 < 100, subarrays: [2], [5,2] → count=5            │
│      window: [5,2], count += (2-1+1) = 2                       │
│                                                                 │
│ r=3: prod=60 < 100, subarrays: [6], [2,6], [5,2,6] → count=8   │
│      window: [5,2,6], count += (3-1+1) = 3                     │
│                                                                 │
│ Total = 8 ✓                                                     │
└────────────────────────────────────────────────────────────────┘

Why (right - left + 1)? Each new element creates that many NEW subarrays!
```

### Solution
```python
def numSubarrayProductLessThanK(nums: list[int], k: int) -> int:
    """
    Count subarrays with product < k using sliding window.

    Note: This uses sliding window, not prefix sum, because
    prefix products can overflow and division is needed.

    Strategy:
    - Expand right, multiply product
    - Shrink left when product >= k
    - Count subarrays ending at right

    Time: O(n)
    Space: O(1)
    """
    if k <= 1:
        return 0

    count = 0
    product = 1
    left = 0

    for right in range(len(nums)):
        product *= nums[right]

        # Shrink window while product >= k
        while product >= k:
            product //= nums[left]
            left += 1

        # All subarrays ending at right with start in [left, right]
        count += right - left + 1

    return count
```

### Complexity
- **Time**: O(n) - Each element visited at most twice (once by right, once by left)
- **Space**: O(1) - Only use a few variables

### Edge Cases
- `k ≤ 1`: No positive product can be < 1, return 0
- All 1s: Every subarray is valid
- Single element ≥ k: That element contributes 0 subarrays

---

## Problem 6: Path Sum III (LC #437) - Medium

- [LeetCode](https://leetcode.com/problems/path-sum-iii/)

### Problem Statement
Given the root of a binary tree and an integer `targetSum`, return the number of paths where the sum of the values along the path equals `targetSum`. The path does not need to start at the root or end at a leaf, but must go downwards (from parent to child nodes).

### Video Explanation
- [NeetCode - Path Sum III](https://www.youtube.com/watch?v=uZzvivFkgtM)

### Examples
```
Input: root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
Output: 3
Explanation: Paths that sum to 8:
  - 5 → 3
  - 5 → 2 → 1
  - -3 → 11

Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: 3

Input: root = [1], targetSum = 1
Output: 1
```

### Intuition Development
```
Apply prefix sum to TREE PATHS! Use DFS with backtracking.

        10
       /  \
      5   -3
     / \    \
    3   2   11
   / \   \
  3  -2   1

Path from root: [10, 5, 3] → prefix sums: [0, 10, 15, 18]

For targetSum = 8:
  At node 3 (prefix=18): need 18-8=10, found at root! ✓
  This means path [5,3] sums to 8.

Key insight: Prefix sum works on any root-to-node path!

DFS with backtracking:
┌────────────────────────────────────────────────────────────────┐
│ 1. Go down: Add current prefix_sum to map                      │
│ 2. Check: How many previous prefixes = current - target?       │
│ 3. Recurse: Visit children                                     │
│ 4. Backtrack: REMOVE current prefix_sum from map when returning│
└────────────────────────────────────────────────────────────────┘

Why backtrack? Different branches shouldn't see each other's prefixes!
```

### Solution
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def pathSum(root: TreeNode, targetSum: int) -> int:
    """
    Count paths with sum = targetSum using prefix sum on tree paths.

    Strategy:
    - Use prefix sum along each root-to-leaf path
    - Track prefix sums in hash map during DFS
    - Backtrack when returning from recursion

    Time: O(n)
    Space: O(n) for recursion and hash map
    """
    count = 0

    # Map: prefix_sum -> count of paths with this sum
    prefix_count = {0: 1}

    def dfs(node: TreeNode, current_sum: int):
        nonlocal count

        if not node:
            return

        # Update current prefix sum
        current_sum += node.val

        # Check if we can form path with target sum
        target = current_sum - targetSum
        if target in prefix_count:
            count += prefix_count[target]

        # Add current sum to map
        prefix_count[current_sum] = prefix_count.get(current_sum, 0) + 1

        # Recurse to children
        dfs(node.left, current_sum)
        dfs(node.right, current_sum)

        # Backtrack: remove current sum from map
        prefix_count[current_sum] -= 1

    dfs(root, 0)
    return count
```

### Complexity
- **Time**: O(n) - Visit each node once
- **Space**: O(n) - Recursion stack + hash map

### Edge Cases
- Empty tree: Return 0
- Single node equals target: Return 1
- Negative values: Algorithm handles them correctly
- Target = 0: Count paths that sum to 0

---

## Problem 7: Make Sum Divisible by P (LC #1590) - Medium

- [LeetCode](https://leetcode.com/problems/make-sum-divisible-by-p/)

### Problem Statement
Given an array of positive integers `nums`, remove the **smallest** subarray (possibly empty) such that the sum of the remaining elements is divisible by `p`. Return the length of the smallest subarray to remove, or `-1` if impossible. Note: You cannot remove the entire array.

### Video Explanation
- [NeetCode - Make Sum Divisible by P](https://www.youtube.com/watch?v=fMwFXi_sRAQ)

### Examples
```
Input: nums = [3,1,4,2], p = 6
Output: 1
Explanation: Total sum = 10. Remove [4], remaining = 6, divisible by 6.

Input: nums = [6,3,5,2], p = 9
Output: 2
Explanation: Total = 16, 16 % 9 = 7. Remove [5,2], remaining = 9.

Input: nums = [1,2,3], p = 3
Output: 0
Explanation: Sum = 6, already divisible by 3.

Input: nums = [1,2,3], p = 7
Output: -1
Explanation: Cannot make remaining sum divisible by 7.
```

### Intuition Development
```
Goal: Remove subarray whose sum ≡ (total % p) (mod p)

nums = [3, 1, 4, 2], p = 6
total = 10, target_remainder = 10 % 6 = 4

We need to find SHORTEST subarray with sum ≡ 4 (mod 6)

Prefix sums % 6: [0, 3, 4, 2, 4]
                  ↑
                  Virtual index -1

For each position, we need previous prefix where:
  (prefix[i] - prefix[j]) % p == target
  prefix[j] % p == (prefix[i] - target) % p

┌─────────────────────────────────────────────────────────────────┐
│ target = 4                                                       │
│                                                                  │
│ i=0: prefix=3, need=(3-4)%6=5, not found, map={0:-1, 3:0}       │
│ i=1: prefix=4, need=(4-4)%6=0, found at -1, len=1-(-1)=2        │
│ i=2: prefix=2, need=(2-4)%6=4, not found, map={.., 2:2}         │
│ i=3: prefix=4, need=(4-4)%6=0, found at -1, len=3-(-1)=4        │
│      But also check: prefix=4 at i=1? need=(4-4)=0              │
│                                                                  │
│ Wait, better: i=2, prefix=8%6=2                                  │
│ Actually: nums[2]=4, prefix[2]=8, 8%6=2                          │
│ need=(2-4)%6=(-2)%6=4, found at i=1! len=2-1=1 ★                │
│                                                                  │
│ Minimum length = 1 ✓ (remove [4] at index 2)                     │
└─────────────────────────────────────────────────────────────────┘

Key: Store MOST RECENT index (not first) for shortest length!
```

### Solution
```python
def minSubarray(nums: list[int], p: int) -> int:
    """
    Find shortest subarray to remove for sum divisible by p.

    Key insight:
    - total_sum % p = target (remainder we need to remove)
    - Find shortest subarray with sum % p == target

    Strategy:
    - Use prefix sum modulo p
    - Track most recent index for each remainder
    - Find shortest subarray with remainder = target

    Time: O(n)
    Space: O(min(n, p))
    """
    total = sum(nums)
    target = total % p

    if target == 0:
        return 0  # Already divisible

    # Map: remainder -> most recent index with this remainder
    last_index = {0: -1}

    min_length = len(nums)
    prefix_sum = 0

    for i, num in enumerate(nums):
        prefix_sum = (prefix_sum + num) % p

        # We need subarray with remainder = target
        # prefix[i] - prefix[j] ≡ target (mod p)
        # prefix[j] ≡ prefix[i] - target (mod p)
        need = (prefix_sum - target) % p

        if need in last_index:
            length = i - last_index[need]
            min_length = min(min_length, length)

        last_index[prefix_sum] = i

    return min_length if min_length < len(nums) else -1
```

### Complexity
- **Time**: O(n) - Single pass through array
- **Space**: O(min(n, p)) - Hash map stores at most min(n, p) remainders

### Edge Cases
- Already divisible: Return 0
- Can't make divisible without removing all: Return -1
- Single element array: Either 0 or -1
- All elements same remainder: May need to remove almost entire array

---

## Summary: Medium Prefix Sum Problems

| # | Problem | Key Variation | Time |
|---|---------|---------------|------|
| 1 | Continuous Subarray Sum | Remainder + length >= 2 | O(n) |
| 2 | Binary Subarrays With Sum | Binary array sum | O(n) |
| 3 | Nice Subarrays | Count odds (transform) | O(n) |
| 4 | Max Size Subarray Sum K | First occurrence for max length | O(n) |
| 5 | Product Less Than K | Sliding window (not prefix) | O(n) |
| 6 | Path Sum III | Prefix sum on tree | O(n) |
| 7 | Make Sum Divisible | Remove subarray with remainder | O(n) |

---

## Key Patterns Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREFIX SUM PATTERN VARIATIONS                            │
│                                                                             │
│  1. COUNT subarrays with sum = k:                                           │
│     prefix_count = {0: 1}                                                   │
│     count += prefix_count[prefix_sum - k]                                   │
│                                                                             │
│  2. LONGEST subarray with sum = k:                                          │
│     first_index = {0: -1}                                                   │
│     Store first occurrence only                                             │
│                                                                             │
│  3. DIVISIBILITY by k:                                                      │
│     Use prefix_sum % k as key                                               │
│     Same remainder = divisible subarray                                     │
│                                                                             │
│  4. BINARY/COUNT transformations:                                           │
│     odd -> 1, even -> 0                                                     │
│     char == target -> 1, else -> 0                                          │
│                                                                             │
│  5. TREE paths:                                                             │
│     DFS with backtracking                                                   │
│     Remove prefix_sum when returning                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practice More Problems

- [ ] LC #974 - Subarray Sums Divisible by K
- [ ] LC #1074 - Number of Submatrices That Sum to Target
- [ ] LC #1371 - Find the Longest Substring Containing Vowels in Even Counts
- [ ] LC #1442 - Count Triplets That Can Form Two Arrays of Equal XOR

