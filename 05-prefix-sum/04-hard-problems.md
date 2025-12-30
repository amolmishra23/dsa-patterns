# Prefix Sum - Hard Problems

## Problem 1: Maximum Sum of 3 Non-Overlapping Subarrays (LC #689) - Hard

- [LeetCode](https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/)

### Video Explanation
- [NeetCode - Maximum Sum of 3 Non-Overlapping Subarrays](https://www.youtube.com/watch?v=HyZfPEK3Gts)

### Problem Statement
Find three non-overlapping subarrays of length k with maximum sum.


### Visual Intuition
```
Maximum Sum of 3 Non-Overlapping Subarrays (k=2)
nums = [1, 2, 1, 2, 6, 7, 5, 1]

Pattern: DP with Left/Right Best Tracking
Why: For each middle position, combine best left + middle + best right

Step 0 (Compute Window Sums):
  nums:  [1, 2, 1, 2, 6, 7, 5, 1]
          【──】                    sum = 3, idx = 0
             【──】                 sum = 3, idx = 1
                【──】              sum = 3, idx = 2
                   【──】           sum = 8, idx = 3
                      【──】        sum = 13, idx = 4 ★ max
                         【──】     sum = 12, idx = 5
                            【──】  sum = 6, idx = 6

  window_sum = [3, 3, 3, 8, 13, 12, 6]
  indices:      0  1  2  3   4   5  6

Step 1 (Build best_left - best ending at or before i):
  Scan left → right, track maximum seen so far

  idx:        0  1  2  3  4  5  6
  sum:        3  3  3  8 13 12  6
  best_left: [0, 0, 0, 3, 4, 4, 4]  ← stores INDEX of best
              ↑        ↑  ↑
              3        8  13 (new max at each point)

Step 2 (Build best_right - best starting at or after i):
  Scan right → left, track maximum seen so far

  idx:         0  1  2  3  4  5  6
  sum:         3  3  3  8 13 12  6
  best_right: [4, 4, 4, 4, 4, 5, 6]  ← stores INDEX of best
                        ↑  ↑
                       13  12 (scan from right)

Step 3 (Find Best Middle):
  For each middle position j (must leave room for left and right):

  j=2: left=best_left[0]=0, right=best_right[4]=4
       total = 3 + 3 + 13 = 19
       【1,2】    【1,2】    【6,7】
         ↑          ↑          ↑
        left      middle     right

  j=3: left=best_left[1]=0, right=best_right[5]=5
       total = 3 + 8 + 12 = 23 ★ better!
       【1,2】       【2,6】    【7,5】

  j=4: left=best_left[2]=0, right=best_right[6]=6
       total = 3 + 13 + 6 = 22

Answer: indices [0, 3, 5] with sum = 23

Key Insight:
- Precompute best_left[i] = best subarray in [0, i]
- Precompute best_right[i] = best subarray in [i, n-1]
- For middle at j: best_left[j-k] + sum[j] + best_right[j+k]
- Non-overlapping guaranteed by index gaps of k
```

### Solution
```python
def maxSumOfThreeSubarrays(nums: list[int], k: int) -> list[int]:
    """
    Find indices of three non-overlapping subarrays with max sum.

    Strategy:
    - Compute prefix sums for window sums
    - Track best left subarray ending at each position
    - Track best right subarray starting at each position
    - For each middle position, combine best left + middle + best right

    Time: O(n)
    Space: O(n)
    """
    n = len(nums)

    # Compute window sums using prefix sum
    window_sum = [0] * (n - k + 1)
    curr_sum = sum(nums[:k])
    window_sum[0] = curr_sum

    for i in range(1, n - k + 1):
        curr_sum = curr_sum - nums[i - 1] + nums[i + k - 1]
        window_sum[i] = curr_sum

    # Best left index for each position
    left = [0] * len(window_sum)
    best = 0
    for i in range(len(window_sum)):
        if window_sum[i] > window_sum[best]:
            best = i
        left[i] = best

    # Best right index for each position
    right = [0] * len(window_sum)
    best = len(window_sum) - 1
    for i in range(len(window_sum) - 1, -1, -1):
        if window_sum[i] >= window_sum[best]:
            best = i
        right[i] = best

    # Find best combination
    result = [-1, -1, -1]
    max_sum = 0

    for mid in range(k, len(window_sum) - k):
        l, r = left[mid - k], right[mid + k]
        total = window_sum[l] + window_sum[mid] + window_sum[r]

        if total > max_sum:
            max_sum = total
            result = [l, mid, r]

    return result
```

### Edge Cases
- k > n/3 → not enough elements
- All same elements → multiple valid combinations
- Negative numbers → can still find max sum
- n = 3k → exactly one element per gap

---

## Problem 2: Number of Submatrices That Sum to Target (LC #1074) - Hard

- [LeetCode](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/)

### Video Explanation
- [NeetCode - Number of Submatrices That Sum to Target](https://www.youtube.com/watch?v=43DRBP2DUHs)

### Problem Statement
Count submatrices with sum equal to target.


### Visual Intuition
```
Number of Submatrices That Sum to Target
matrix = [[0,1,0],      target = 0
          [1,1,1],
          [0,1,0]]

Pattern: 2D → 1D Reduction + Prefix Sum HashMap
Why: Fix row boundaries, reduce to "subarray sum = target" problem

Step 0 (Visualize Matrix):
  ┌───┬───┬───┐
  │ 0 │ 1 │ 0 │  row 0
  ├───┼───┼───┤
  │ 1 │ 1 │ 1 │  row 1
  ├───┼───┼───┤
  │ 0 │ 1 │ 0 │  row 2
  └───┴───┴───┘
   c0  c1  c2

Step 1 (Fix top=0, bottom=0 - single row):
  col_sum = [0, 1, 0]  ← just row 0

  Find subarrays with sum = 0:
  prefix:  0 → 0 → 1 → 1
           ↑       ↑
           start   same prefix = subarray sum 0!

  HashMap: {0:1} → {0:1, 0:2} → {0:2, 1:1} → {0:2, 1:2}
                    ↑ found 0 twice

  Subarrays: [0] at idx 0, [0] at idx 2 → count = 2

Step 2 (Fix top=0, bottom=1 - rows 0-1):
  col_sum = [0+1, 1+1, 0+1] = [1, 2, 1]
            ┌───┬───┬───┐
            │ 0 │ 1 │ 0 │ ← compressed
            │ 1 │ 1 │ 1 │   to 1D
            └───┴───┴───┘

  prefix: 0 → 1 → 3 → 4
  No prefix repeats with diff = 0 → count += 0

Step 3 (Fix top=0, bottom=2 - all rows):
  col_sum = [1, 3, 1]
  prefix: 0 → 1 → 4 → 5
  count += 0

Step 4 (Fix top=1, bottom=1):
  col_sum = [1, 1, 1]
  prefix: 0 → 1 → 2 → 3
  count += 0

Step 5 (Fix top=1, bottom=2):
  col_sum = [1, 2, 1]
  count += 0

Step 6 (Fix top=2, bottom=2):
  col_sum = [0, 1, 0]  ← same as row 0!
  count += 2

Total count = 4 submatrices with sum = 0

Visualization of Found Submatrices:
  ┌─┐         ┌─┐       ┌─┐         ┌─┐
  │0│ . .     . . │0│   │0│ . .     . . │0│
  └─┘         └─┘       . . .       . . .
  . . .       . . .     └─┘         └─┘
  (top-left) (top-right) (bottom-left) (bottom-right)

Key Insight:
- O(m²) row pairs × O(n) column scan = O(m² × n)
- HashMap trick: count prefix occurrences
- If prefix[j] - prefix[i] = target, we found a submatrix
```

### Solution
```python
def numSubmatrixSumTarget(matrix: list[list[int]], target: int) -> int:
    """
    Count submatrices summing to target.

    Strategy:
    - Fix top and bottom rows
    - Compress to 1D array (column sums)
    - Use prefix sum + hashmap for 1D subarray sum = target

    Time: O(m² * n)
    Space: O(n)
    """
    from collections import defaultdict

    m, n = len(matrix), len(matrix[0])
    count = 0

    # For each pair of rows
    for top in range(m):
        # Column sums between top and bottom rows
        col_sum = [0] * n

        for bottom in range(top, m):
            # Update column sums
            for col in range(n):
                col_sum[col] += matrix[bottom][col]

            # Count subarrays with sum = target (1D problem)
            prefix_count = defaultdict(int)
            prefix_count[0] = 1
            prefix = 0

            for val in col_sum:
                prefix += val
                count += prefix_count[prefix - target]
                prefix_count[prefix] += 1

    return count
```

### Edge Cases
- target = 0 → count zero-sum submatrices
- Single cell → check if equals target
- All zeros → every submatrix if target = 0
- Negative values → still works with prefix sum

---

## Problem 3: Count of Range Sum (LC #327) - Hard

- [LeetCode](https://leetcode.com/problems/count-of-range-sum/)

### Video Explanation
- [NeetCode - Count of Range Sum](https://www.youtube.com/watch?v=VPVXXlbLk80)

### Problem Statement
Count range sums in [lower, upper] inclusive.


### Visual Intuition
```
Count of Range Sum
nums = [-2, 5, -1], lower = -2, upper = 2

Pattern: Merge Sort + Counting During Merge
Why: Count pairs (i,j) where lower ≤ prefix[j] - prefix[i] ≤ upper

Step 0 (Build Prefix Sums):
  nums:   [-2, 5, -1]
  prefix: [0, -2, 3, 2]
           ↑   ↑  ↑  ↑
           0   1  2  3  (indices)

  Range sum [i,j] = prefix[j+1] - prefix[i]
  Example: sum[-2,5] = prefix[2] - prefix[0] = 3 - 0 = 3

Step 1 (Rearrange Condition):
  Want: lower ≤ prefix[j] - prefix[i] ≤ upper
  Rearrange: prefix[j] - upper ≤ prefix[i] ≤ prefix[j] - lower

  For each j, count valid i's in sorted left half

Step 2 (Merge Sort - Divide):
  prefix: [0, -2, 3, 2]
          └──┬──┘ └─┬─┘
           left   right

  Left:  [0, -2] → sort → [-2, 0]
  Right: [3, 2]  → sort → [2, 3]

Step 3 (Count During Merge):
  For each element in right, count valid elements in left

  Right element = 2:
    Need: 2 - 2 ≤ prefix[i] ≤ 2 - (-2)
          0 ≤ prefix[i] ≤ 4

    Left (sorted): [-2, 0]
                        ↑
                    0 is in [0, 4] ✓
    Count = 1

  Right element = 3:
    Need: 3 - 2 ≤ prefix[i] ≤ 3 - (-2)
          1 ≤ prefix[i] ≤ 5

    Left (sorted): [-2, 0]
                    ✗   ✗  (neither in [1, 5])
    Count = 0

Step 4 (Recursive Counting):
  Also count within left half and right half recursively

  Left half [0, -2]:
    Range [0,0] = prefix[1] - prefix[0] = -2 - 0 = -2
    Is -2 in [-2, 2]? ✓ Count = 1

  Right half [3, 2]:
    Range [1,2] = prefix[3] - prefix[2] = 2 - 3 = -1
    Is -1 in [-2, 2]? ✓ Count = 1

Total count = 1 (merge) + 1 (left) + 1 (right) = 3

Valid Range Sums:
  [-2] = -2 ✓ (in [-2,2])
  [-2,5] = 3 ✗
  [-2,5,-1] = 2 ✓
  [5] = 5 ✗
  [5,-1] = 4 ✗
  [-1] = -1 ✓

  Answer: 3 range sums in [-2, 2]

Key Insight:
- Sorting doesn't change which pairs are valid
- Two pointers find range efficiently in O(n) per level
- Total: O(n log n) like merge sort
```

### Solution
```python
def countRangeSum(nums: list[int], lower: int, upper: int) -> int:
    """
    Count range sums within [lower, upper].

    Strategy:
    - Use prefix sums
    - Merge sort to count valid pairs
    - For each prefix[i], count prefix[j] where lower <= prefix[i] - prefix[j] <= upper

    Time: O(n log n)
    Space: O(n)
    """
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)

    def merge_count(lo, hi):
        if lo >= hi:
            return 0

        mid = (lo + hi) // 2
        count = merge_count(lo, mid) + merge_count(mid + 1, hi)

        # Count valid pairs across left and right
        j = k = mid + 1
        for i in range(lo, mid + 1):
            while j <= hi and prefix[j] - prefix[i] < lower:
                j += 1
            while k <= hi and prefix[k] - prefix[i] <= upper:
                k += 1
            count += k - j

        # Merge sorted halves
        prefix[lo:hi+1] = sorted(prefix[lo:hi+1])

        return count

    return merge_count(0, len(prefix) - 1)
```

### Edge Cases
- lower = upper → count exact sums
- All positive → simpler counting
- All negative → still works
- Single element → check if in range

---

## Problem 4: Split Array Largest Sum (LC #410) - Hard

- [LeetCode](https://leetcode.com/problems/split-array-largest-sum/)

### Video Explanation
- [NeetCode - Split Array Largest Sum](https://www.youtube.com/watch?v=YUF3_eBdzsk)

### Problem Statement
Split array into m non-empty continuous subarrays to minimize the largest sum among them.

### Visual Intuition
```
Split Array Largest Sum - Binary Search on Answer
nums = [7, 2, 5, 10, 8], k = 2 parts

Pattern: Binary Search on Answer + Greedy Validation
Why: If max_sum = X works, any Y > X also works (monotonic)

Step 0 (Define Search Space):
  min = max(nums) = 10  (largest element must fit)
  max = sum(nums) = 32  (all in one part)

  Binary search: [10, 32]
                  L    R

Step 1: mid = 21, can we split with max ≤ 21?

  nums: [7, 2, 5, 10, 8]
        【──────】【────】
         7+2+5    10+8
          =14      =18

  Greedy packing:
  Part 1: 7 → 7+2=9 → 9+5=14 → 14+10=24 > 21 ✗ start new
  Part 2: 10 → 10+8=18 ≤ 21 ✓

  Parts = 2 ≤ k=2 ✓ → try smaller, right = 21

Step 2: mid = 15, can we split with max ≤ 15?

  nums: [7, 2, 5, 10, 8]
        【──────】【──】【─】
           14      10    8

  Part 1: 7+2+5=14 ≤ 15 ✓
  Part 2: 14+10=24 > 15 ✗ start new: 10
  Part 3: 10+8=18 > 15 ✗ start new: 8

  Parts = 3 > k=2 ✗ → need larger, left = 16

Step 3: mid = 18, can we split with max ≤ 18?

  nums: [7, 2, 5, 10, 8]
        【──────】 【────】
           14        18

  Part 1: 7+2+5=14 ≤ 18 ✓
  Part 2: 10+8=18 ≤ 18 ✓

  Parts = 2 ≤ k=2 ✓ → try smaller, right = 18

Step 4: left = 16, mid = 17
  Parts needed = 3 > k ✗ → left = 18

Answer: 18 (left == right)

Visualization of Optimal Split:
  ┌─────────────────┬─────────────────┐
  │  7 + 2 + 5 = 14 │  10 + 8 = 18    │
  │     Part 1      │     Part 2      │
  └─────────────────┴─────────────────┘
                          ↑
                    Largest sum = 18 (minimized)

Key Insight:
- Greedy: pack as much as possible per part
- If we can achieve max_sum = X with k parts,
  we can achieve any Y > X with ≤ k parts
- Binary search finds minimum X that works
```


### Intuition
```
Array: [7, 2, 5, 10, 8], m = 2

Option 1: [7, 2, 5] | [10, 8]  → max(14, 18) = 18
Option 2: [7, 2, 5, 10] | [8] → max(24, 8) = 24
Option 3: [7, 2] | [5, 10, 8] → max(9, 23) = 23
Option 4: [7] | [2, 5, 10, 8] → max(7, 25) = 25

Binary search on answer: Can we split with max sum ≤ X?
```

### Solution
```python
def splitArray(nums: list[int], m: int) -> int:
    """
    Binary search on the answer + greedy validation.

    Strategy:
    - Binary search on possible max sum values
    - For each candidate, greedily check if we can split into ≤ m parts
    - Lower bound = max element, Upper bound = total sum

    Time: O(n * log(sum - max))
    Space: O(1)
    """
    def can_split(max_sum: int) -> bool:
        """Check if we can split array with each part ≤ max_sum."""
        parts = 1
        current_sum = 0

        for num in nums:
            # If adding this number exceeds max_sum, start new part
            if current_sum + num > max_sum:
                parts += 1
                current_sum = num

                # Too many parts needed
                if parts > m:
                    return False
            else:
                current_sum += num

        return True

    # Binary search bounds
    left = max(nums)      # At minimum, max element must fit in one part
    right = sum(nums)     # At maximum, all elements in one part

    while left < right:
        mid = (left + right) // 2

        if can_split(mid):
            right = mid   # Try smaller max sum
        else:
            left = mid + 1  # Need larger max sum

    return left
```

### Complexity
- **Time**: O(n * log(sum - max))
- **Space**: O(1)

### Edge Cases
- m = 1 → return total sum
- m = n → return max element
- All same → return that element
- Single element → return it

---

## Problem 5: Maximum Sum Circular Subarray (LC #918) - Hard

- [LeetCode](https://leetcode.com/problems/maximum-sum-circular-subarray/)

### Video Explanation
- [NeetCode - Maximum Sum Circular Subarray](https://www.youtube.com/watch?v=fxT9KjakYPM)

### Problem Statement
Find the maximum sum of a non-empty subarray in a circular array.

### Visual Intuition
```
Maximum Sum Circular Subarray
nums = [5, -3, 5]

Pattern: Kadane for Max + Kadane for Min (Invert Problem)
Why: Circular max = total - middle_min

Step 0 (Visualize Two Cases):

  Case 1: Normal subarray (no wrap)
  [5, -3, 5]
   ●───────●  contiguous segment

  Case 2: Wrap-around subarray
  [5, -3, 5]
   ●       ●  take ends, skip middle
   └───────┘
     wrap around

Step 1 (Case 1 - Standard Kadane):
  nums: [5, -3, 5]

  i=0: curr_max = max(5, 0+5) = 5, max_sum = 5
  i=1: curr_max = max(-3, 5-3) = 2, max_sum = 5
  i=2: curr_max = max(5, 2+5) = 7, max_sum = 7

  Normal max = 7 (entire array [5,-3,5])

Step 2 (Case 2 - Find Min to Exclude):

  Wrap-around = take prefix + suffix = total - middle

  [5, -3, 5]
      【】    ← find minimum middle part

  Kadane for minimum:
  i=0: curr_min = min(5, 0+5) = 5, min_sum = 5
  i=1: curr_min = min(-3, 5-3) = -3, min_sum = -3 ★
  i=2: curr_min = min(5, -3+5) = 2, min_sum = -3

  Minimum subarray = [-3] with sum = -3

Step 3 (Calculate Circular Max):
  total = 5 + (-3) + 5 = 7
  circular_max = total - min_sum = 7 - (-3) = 10

  Visualization:
  [5, -3, 5]
   ●       ●   take 5 + 5 = 10
      ✗        skip -3
  └────────┘
    wraps around

Step 4 (Compare and Handle Edge Case):
  normal_max = 7
  circular_max = 10

  Answer = max(7, 10) = 10 ✓

Edge Case (All Negative):
  nums = [-3, -2, -5]

  max_sum = -2 (least negative)
  min_sum = -10 (entire array)
  total = -10
  circular = -10 - (-10) = 0 ← WRONG! Can't take empty array

  Solution: if max_sum < 0, return max_sum (all negative)

Key Insight:
- Circular max = total - min_middle
- Run Kadane twice: once for max, once for min
- Edge case: all negative → return normal Kadane max
- O(n) time, O(1) space
```


### Intuition
```
Array: [5, -3, 5]

Case 1: Max subarray is in the middle (normal Kadane)
[5, -3, 5] → max contiguous = 5 + (-3) + 5 = 7

Case 2: Max subarray wraps around
[5, -3, 5] → wrap around: [5] + [5] = 10
           → total - min_middle = 7 - (-3) = 10 ✓

Answer = max(normal_kadane, total - min_kadane)
```

### Solution
```python
def maxSubarraySumCircular(nums: list[int]) -> int:
    """
    Handle two cases: normal subarray OR wrap-around subarray.

    Strategy:
    - Case 1: Max subarray doesn't wrap → standard Kadane
    - Case 2: Max subarray wraps → total_sum - min_subarray
    - Edge case: all negative → return max element

    Time: O(n)
    Space: O(1)
    """
    total_sum = 0

    # Kadane for maximum subarray
    max_sum = float('-inf')
    current_max = 0

    # Kadane for minimum subarray
    min_sum = float('inf')
    current_min = 0

    for num in nums:
        total_sum += num

        # Standard Kadane for max
        current_max = max(current_max + num, num)
        max_sum = max(max_sum, current_max)

        # Kadane for min (to find middle part to exclude)
        current_min = min(current_min + num, num)
        min_sum = min(min_sum, current_min)

    # Edge case: all elements are negative
    # min_sum would be total_sum, making wrap case = 0
    if max_sum < 0:
        return max_sum

    # Return max of non-wrap and wrap cases
    return max(max_sum, total_sum - min_sum)
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- All negative → return max element
- All positive → standard Kadane
- Wrap gives better result → total - min
- Single element → return it

---

## Problem 6: Shortest Subarray with Sum at Least K (LC #862) - Hard

- [LeetCode](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/)

### Video Explanation
- [NeetCode - Shortest Subarray with Sum at Least K](https://www.youtube.com/watch?v=K0NgGYEAkA4)

### Problem Statement
Find the length of the shortest non-empty subarray with sum at least k.

### Visual Intuition
```
Shortest Subarray with Sum ≥ K (handles negatives!)
nums = [2, -1, 2], k = 3

Pattern: Monotonic Deque + Prefix Sum
Why: Negatives break sliding window, need smarter approach

Step 0 (Build Prefix Sums):
  nums:   [2, -1, 2]
  prefix: [0, 2, 1, 3]
           ↑  ↑  ↑  ↑
           0  1  2  3  (indices)

  Subarray sum [i,j) = prefix[j] - prefix[i]

Step 1 (Process with Monotonic Deque):
  Deque stores indices with INCREASING prefix values

  i=0: prefix[0] = 0
       deque = [0]
       No valid subarray yet

  i=1: prefix[1] = 2
       Check front: 2 - 0 = 2 < k=3 ✗
       Add to deque: [0, 1]

       deque: [0, 1]
       prefix: 0  2
               ↑  ↑

  i=2: prefix[2] = 1
       Check front: 1 - 0 = 1 < k=3 ✗

       Pop back while prefix[back] ≥ prefix[i]:
       prefix[1]=2 ≥ 1 → pop 1

       deque: [0, 2]
       prefix: 0  1

       Why pop? Index 1 with prefix=2 will never be better
       than index 2 with prefix=1 for future j's:
       prefix[j] - 2 < prefix[j] - 1 always

  i=3: prefix[3] = 3
       Check front: 3 - 0 = 3 ≥ k=3 ✓
         Found! length = 3 - 0 = 3
         Pop front (used up): deque = [2]

       Check front again: 3 - 1 = 2 < k=3 ✗

       Add to deque: [2, 3]

Answer: 3 (shortest subarray is entire array [2,-1,2])

Deque Invariant Visualization:
  ┌─────────────────────────────────────────────┐
  │ Monotonic INCREASING prefix values          │
  │                                             │
  │ deque: [i₁, i₂, i₃, ...]                    │
  │        prefix[i₁] < prefix[i₂] < prefix[i₃] │
  │                                             │
  │ Front: smallest prefix (best for sum ≥ k)   │
  │ Back:  add new, pop if not smaller          │
  └─────────────────────────────────────────────┘

Why Pop from Front After Finding Valid?
  If prefix[j] - prefix[front] ≥ k,
  any future j' > j will give longer subarray
  So front is "used up" → pop it

Why Pop from Back if prefix[back] ≥ prefix[new]?
  For future j: prefix[j] - prefix[back] ≤ prefix[j] - prefix[new]
  AND back < new (shorter subarray)
  But wait! new gives LARGER sum with SHORTER length
  So back is dominated → pop it

Key Insight:
- Deque maintains potential start indices
- Pop front when valid (found answer, won't get shorter)
- Pop back when dominated (new index is better)
- Each index added/removed once → O(n)
```


### Intuition
```
Array: [2, -1, 2], k = 3

Prefix: [0, 2, 1, 3]

For each prefix[j], find smallest i where prefix[j] - prefix[i] >= k
Use monotonic deque to maintain candidates efficiently.

Deque maintains increasing prefix sums (indices).
```

### Solution
```python
from collections import deque

def shortestSubarray(nums: list[int], k: int) -> int:
    """
    Prefix sum + monotonic deque.

    Strategy:
    - Compute prefix sums
    - Use deque to maintain potential starting indices
    - Deque is monotonically increasing by prefix value
    - For each j, pop from front while prefix[j] - prefix[front] >= k
    - Pop from back while prefix[j] <= prefix[back]

    Time: O(n)
    Space: O(n)
    """
    n = len(nums)

    # Compute prefix sums
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]

    result = float('inf')
    dq = deque()  # Store indices

    for j in range(n + 1):
        # Check if current position gives valid subarray
        # Pop from front while condition is satisfied
        while dq and prefix[j] - prefix[dq[0]] >= k:
            result = min(result, j - dq.popleft())

        # Maintain monotonicity: remove larger prefix sums
        # They can never be better starting points than j
        while dq and prefix[j] <= prefix[dq[-1]]:
            dq.pop()

        dq.append(j)

    return result if result != float('inf') else -1
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- k > total sum → return -1
- Single element >= k → return 1
- All negative → may still have solution
- No valid subarray → return -1

---

## Summary

| # | Problem | Key Technique |
|---|---------|---------------|
| 1 | Max Sum 3 Subarrays | Left/right best tracking |
| 2 | Submatrices Sum Target | 2D to 1D reduction |
| 3 | Count of Range Sum | Merge sort + prefix |
| 4 | Split Array Largest Sum | Binary search on answer |
| 5 | Max Sum Circular Subarray | Kadane + wrap-around |
| 6 | Shortest Subarray Sum ≥ K | Monotonic deque + prefix |
