# Binary Search - Hard Problems

## Problem 1: Median of Two Sorted Arrays (LC #4) - Hard

- [LeetCode](https://leetcode.com/problems/median-of-two-sorted-arrays/)

### Video Explanation
- [NeetCode - Median of Two Sorted Arrays](https://www.youtube.com/watch?v=q6IEA26hvXc)

### Problem Statement
Find median of two sorted arrays in O(log(m+n)).

### Examples
```
Input: nums1 = [1,3], nums2 = [2]
Output: 2.0

Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.5
```


### Visual Intuition
```
Median of Two Sorted Arrays
nums1 = [1, 3, 8], nums2 = [2, 7, 11, 15]

Pattern: Binary Search on Partition Point
Why: Find partition where all left elements ≤ all right elements

Step 0 (Initial Setup):
  nums1: [1, 3, 8]       (smaller array, search here)
  nums2: [2, 7, 11, 15]
  Total = 7 elements → left half needs 4 elements
  Search range: i ∈ [0, 3]

Step 1: Try i=1 (partition nums1 after index 0)
  nums1: [1 │ 3, 8]      i=1, takes 1 element
  nums2: [2, 7, 11 │ 15] j=3, takes 3 elements
         ├─ left ─┤ │ ├─ right ─┤

  Check: max(left) ≤ min(right)?
         max(1, 11) = 11  vs  min(3, 15) = 3
         11 > 3 ✗ → nums2_left too big, need more from nums1
         → Move i right: left = 2

Step 2: Try i=2 (partition nums1 after index 1)
  nums1: [1, 3 │ 8]      i=2, takes 2 elements
  nums2: [2, 7 │ 11, 15] j=2, takes 2 elements
         ├─ left ─┤ │ ├─ right ─┤

  Check: max(left) ≤ min(right)?
         max(3, 7) = 7  vs  min(8, 11) = 8
         7 ≤ 8 ✓ Valid partition!

Step 3 (Calculate Median):
  Left half:  [1, 2, 3, 7]  → max = 7
  Right half: [8, 11, 15]   → min = 8

  Odd total (7): median = max(left) = 7

Key Insight:
- Binary search on SMALLER array for O(log(min(m,n)))
- j = half_len - i (ensures left half has correct count)
- If nums1[i-1] > nums2[j] → i too big, move left
- If nums2[j-1] > nums1[i] → i too small, move right

Before: O(m+n) merge then find median
After:  O(log(min(m,n))) binary search on partition
```

### Solution
```python
def findMedianSortedArrays(nums1: list[int], nums2: list[int]) -> float:
    """
    Find median using binary search on partition.

    Key insight: Partition both arrays such that:
    - Left half has (m + n + 1) // 2 elements
    - All left elements <= all right elements

    Strategy:
    - Binary search on smaller array
    - For each partition, check if valid

    Time: O(log(min(m, n)))
    Space: O(1)
    """
    # Ensure nums1 is smaller
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    left, right = 0, m
    half_len = (m + n + 1) // 2

    while left <= right:
        # Partition index for nums1
        i = (left + right) // 2
        # Partition index for nums2
        j = half_len - i

        # Get boundary elements (use inf for out of bounds)
        nums1_left = nums1[i - 1] if i > 0 else float('-inf')
        nums1_right = nums1[i] if i < m else float('inf')
        nums2_left = nums2[j - 1] if j > 0 else float('-inf')
        nums2_right = nums2[j] if j < n else float('inf')

        # Check if partition is valid
        if nums1_left <= nums2_right and nums2_left <= nums1_right:
            # Found valid partition
            if (m + n) % 2 == 1:
                return max(nums1_left, nums2_left)
            else:
                return (max(nums1_left, nums2_left) +
                        min(nums1_right, nums2_right)) / 2
        elif nums1_left > nums2_right:
            # nums1 partition too far right
            right = i - 1
        else:
            # nums1 partition too far left
            left = i + 1

    return 0.0
```

### Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  nums1 = [1, 3, 8, 9, 15]                                                   │
│  nums2 = [7, 11, 18, 19, 21, 25]                                            │
│                                                                             │
│  Total = 11 elements, median at position 6                                  │
│                                                                             │
│  Partition nums1 at i=2: [1, 3 | 8, 9, 15]                                  │
│  Partition nums2 at j=4: [7, 11, 18, 19 | 21, 25]                           │
│                                                                             │
│  Left half: [1, 3, 7, 11, 18, 19] - 6 elements                              │
│  Right half: [8, 9, 15, 21, 25] - 5 elements                                │
│                                                                             │
│  Check: max(3, 19) <= min(8, 21)? 19 <= 8? NO                               │
│  nums2_left (19) > nums1_right (8), move nums1 partition right              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- One array empty → median of other array
- Both single element → average of two
- All elements same → that element
- Odd vs even total → affects return type

---

## Problem 2: Split Array Largest Sum (LC #410) - Hard

- [LeetCode](https://leetcode.com/problems/split-array-largest-sum/)

### Video Explanation
- [NeetCode - Split Array Largest Sum](https://www.youtube.com/watch?v=YUF3_eBdzsk)

### Problem Statement
Split array into m subarrays to minimize largest sum.

### Examples
```
Input: nums = [7,2,5,10,8], m = 2
Output: 18 ([7,2,5] and [10,8])
```


### Visual Intuition
```
Split Array Largest Sum
nums = [7, 2, 5, 10, 8], k = 2 subarrays

Pattern: Binary Search on Answer (minimize maximum)
Why: If we can achieve max_sum=X, we can achieve any Y > X

Step 0 (Define Search Space):
  min possible = max(nums) = 10  (can't split element)
  max possible = sum(nums) = 32  (one subarray)

  Search: [10, 32]
              L   R

Step 1: mid = 21, can we split with max ≤ 21?
  nums: [7, 2, 5, 10, 8]
        【─────】 【────】
         sum=14    sum=18

  Greedy: add until exceeds limit, then start new part
    7 → 7
    7+2 → 9
    9+5 → 14
    14+10 → 24 > 21 ✗ start new: [10]
    10+8 → 18 ≤ 21 ✓

  Parts: 2 ≤ k=2 ✓ → try smaller, right = 21

Step 2: mid = 15, can we split with max ≤ 15?
  nums: [7, 2, 5, 10, 8]
        【─────】【──】【─】
         sum=14   10    8

    7+2+5 = 14 ≤ 15 ✓
    14+10 = 24 > 15 ✗ start new: [10]
    10+8 = 18 > 15 ✗ start new: [8]

  Parts: 3 > k=2 ✗ → need larger, left = 16

Step 3: mid = 18, can we split with max ≤ 18?
  nums: [7, 2, 5, 10, 8]
        【─────】 【────】
         sum=14    sum=18

    7+2+5 = 14 ≤ 18 ✓
    14+10 = 24 > 18 ✗ start new: [10]
    10+8 = 18 ≤ 18 ✓

  Parts: 2 ≤ k=2 ✓ → try smaller, right = 18

Step 4: left = 16, right = 18, mid = 17
  Parts needed: 3 > k ✗ → left = 18

Answer: 18 (left == right)

Key Insight:
- Greedy assignment: pack as much as possible per subarray
- Monotonic: if X works, X+1 also works
- Search space is [max_element, total_sum]

Visualization of optimal split:
  [7, 2, 5] | [10, 8]
     14    |    18    ← largest sum minimized
```

### Solution
```python
def splitArray(nums: list[int], m: int) -> int:
    """
    Minimize largest subarray sum using binary search.

    Strategy:
    - Binary search on answer (largest sum)
    - For each candidate, check if we can split into <= m parts

    Time: O(n * log(sum - max))
    Space: O(1)
    """
    def can_split(max_sum: int) -> bool:
        """Check if we can split into m parts with max_sum limit."""
        parts = 1
        current_sum = 0

        for num in nums:
            if current_sum + num > max_sum:
                parts += 1
                current_sum = num

                if parts > m:
                    return False
            else:
                current_sum += num

        return True

    # Binary search range: [max element, total sum]
    left = max(nums)
    right = sum(nums)

    while left < right:
        mid = (left + right) // 2

        if can_split(mid):
            right = mid  # Try smaller max sum
        else:
            left = mid + 1  # Need larger max sum

    return left
```

### Edge Cases
- m = 1 → return sum of array
- m = n → return max element
- All same elements → return that element
- Single element → return that element

---

## Problem 3: Find Minimum in Rotated Sorted Array II (LC #154) - Hard

- [LeetCode](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/)

### Video Explanation
- [NeetCode - Find Minimum in Rotated Sorted Array II](https://www.youtube.com/watch?v=K0PGvvSU9Ws)

### Problem Statement
Find minimum in rotated sorted array with duplicates.

### Examples
```
Input: nums = [2,2,2,0,1]
Output: 0
```


### Visual Intuition
```
Find Min in Rotated Sorted Array II (with duplicates)
nums = [2, 2, 2, 0, 1, 2]

Pattern: Binary Search with Duplicate Handling
Why: Duplicates break the comparison, need fallback strategy

Step 0 (Initial):
  nums: [2, 2, 2, 0, 1, 2]
         L        M     R
         ↑        ↑     ↑

  Compare nums[M] vs nums[R]:
  nums[M]=2 == nums[R]=2 → Can't determine which half!

  Fallback: R-- (safe because if M is min, we keep it)

Step 1 (After R--):
  nums: [2, 2, 2, 0, 1]
         L     M     R
         ↑     ↑     ↑

  nums[M]=2 > nums[R]=1
  → Rotation point (minimum) is in RIGHT half
  → L = M + 1

Step 2 (Search right half):
  nums: [2, 2, 2, 0, 1]
                  L  R
                  M

  nums[M]=0 < nums[R]=1
  → Minimum is in LEFT half (including M)
  → R = M

Step 3 (Converged):
  nums: [2, 2, 2, 0, 1]
                  LR
                  ↑
  L == R → Found minimum = 0 ✓

Decision Tree:
  ┌─────────────────────────────────────┐
  │     Compare nums[mid] vs nums[R]    │
  └─────────────────────────────────────┘
            │
    ┌───────┼───────┐
    ↓       ↓       ↓
   >R      ==R      <R
    │       │       │
    ↓       ↓       ↓
  L=M+1   R--     R=M
  (min    (can't  (min in
  right)  decide) left+M)

Key Insight:
- When nums[M] == nums[R], we can't determine side
- Safe to do R-- because:
  • If nums[R] is the only min, nums[M] is also min
  • We don't lose the minimum
- Worst case O(n): [1,1,1,1,1,1,1] all same

Before/After:
  Without duplicates: O(log n) always
  With duplicates:    O(log n) avg, O(n) worst
```

### Solution
```python
def findMin(nums: list[int]) -> int:
    """
    Find minimum in rotated array with duplicates.

    Challenge: Duplicates can make mid == right, can't determine side.
    Solution: When nums[mid] == nums[right], shrink right by 1.

    Time: O(n) worst case, O(log n) average
    Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2

        if nums[mid] > nums[right]:
            # Minimum is in right half
            left = mid + 1
        elif nums[mid] < nums[right]:
            # Minimum is in left half (including mid)
            right = mid
        else:
            # nums[mid] == nums[right], can't determine
            # Safe to shrink right (if mid is min, we still have it)
            right -= 1

    return nums[left]
```

### Edge Cases
- All same elements → return that element
- Not rotated → first element is min
- Single element → return it
- All duplicates → O(n) worst case

---

## Problem 4: Find K-th Smallest Pair Distance (LC #719) - Hard

- [LeetCode](https://leetcode.com/problems/find-k-th-smallest-pair-distance/)

### Video Explanation
- [NeetCode - Find K-th Smallest Pair Distance](https://www.youtube.com/watch?v=WHfljqX_-T4)

### Problem Statement
Find kth smallest distance among all pairs.

### Examples
```
Input: nums = [1,3,1], k = 1
Output: 0 (pairs: (1,1), distance 0)
```


### Visual Intuition
```
K-th Smallest Pair Distance
nums = [1, 6, 1], k = 3

Pattern: Binary Search on Answer + Two Pointers Counting
Why: Don't enumerate all O(n²) pairs, count them efficiently

Step 0 (Setup):
  Sorted: [1, 1, 6]
  All pairs: (1,1)=0, (1,6)=5, (1,6)=5
  Sorted distances: [0, 5, 5]

  Search space: [0, 6-1] = [0, 5]
                 L          R

Step 1: mid = 2, count pairs with distance ≤ 2
  Sorted: [1, 1, 6]
           i  j

  Two-pointer counting:
  j=0: no pairs (j-i = 0)
  j=1: nums[1]-nums[0] = 0 ≤ 2 ✓ count += 1
       【1, 1】
  j=2: nums[2]-nums[0] = 5 > 2, move i
       nums[2]-nums[1] = 5 > 2, move i
       i=2, count += 0

  Total count = 1 < k=3 ✗ → need larger distance
  left = 3

Step 2: mid = 4, count pairs with distance ≤ 4
  j=1: count += 1 (pair 0,1)
  j=2: 6-1=5 > 4, move i until valid
       count += 0

  Total count = 1 < k=3 ✗ → left = 5

Step 3: mid = 5, count pairs with distance ≤ 5
  j=1: 1-1=0 ≤ 5 ✓ count += 1
  j=2: 6-1=5 ≤ 5 ✓ count += 2 (pairs with i=0,1)
       【1, 1, 6】
        ↑  ↑  ↑
        all pairs valid

  Total count = 3 ≥ k=3 ✓ → right = 5

Answer: 5 (left == right)

Two-Pointer Counting Visualization:
  For each right pointer j:
    [1, 1, 6]
     ←──i  j

    Slide i right until nums[j] - nums[i] ≤ mid
    All pairs (i, j), (i+1, j), ..., (j-1, j) are valid
    count += (j - i)

Key Insight:
- Sorting enables two-pointer counting in O(n)
- Binary search on distance value, not on array
- count_pairs(d) is monotonic: more pairs as d increases
- Total: O(n log n) sort + O(n log W) search × O(n) count

Why Two Pointers Work:
  As j increases, valid i can only increase (never decrease)
  → Amortized O(n) for all j iterations
```

### Solution
```python
def smallestDistancePair(nums: list[int], k: int) -> int:
    """
    Find kth smallest pair distance using binary search.

    Strategy:
    - Binary search on distance value
    - Count pairs with distance <= mid
    - Use two pointers for counting

    Time: O(n log n + n log W) where W = max - min
    Space: O(1)
    """
    nums.sort()
    n = len(nums)

    def count_pairs(max_dist: int) -> int:
        """Count pairs with distance <= max_dist."""
        count = 0
        left = 0

        for right in range(n):
            while nums[right] - nums[left] > max_dist:
                left += 1
            count += right - left

        return count

    # Binary search on distance
    left, right = 0, nums[-1] - nums[0]

    while left < right:
        mid = (left + right) // 2

        if count_pairs(mid) < k:
            left = mid + 1
        else:
            right = mid

    return left
```

### Edge Cases
- k = 1 → return 0 (smallest distance)
- All same elements → return 0
- Two elements → return their difference
- k = n*(n-1)/2 → return max distance

---

## Problem 5: Kth Smallest Element in Sorted Matrix (LC #378) - Medium

- [LeetCode](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/)

### Video Explanation
- [NeetCode - Kth Smallest Element in Sorted Matrix](https://www.youtube.com/watch?v=vHqTdBKaFLE)

### Problem Statement
Find kth smallest in row/column sorted matrix.


### Visual Intuition
```
Kth Smallest in Sorted Matrix
matrix = [[1,  5,  9],
          [10, 11, 13],
          [12, 13, 15]], k = 8

Pattern: Binary Search on Value + Staircase Counting
Why: Matrix is sorted row-wise AND column-wise

Step 0 (Setup):
  Search space: [matrix[0][0], matrix[n-1][n-1]] = [1, 15]
  Start counting from bottom-left corner

Step 1: mid = 8, count elements ≤ 8

  ┌─────┬─────┬─────┐
  │  1  │  5  │  9  │  ← row 0
  ├─────┼─────┼─────┤
  │ 10  │ 11  │ 13  │  ← row 1
  ├─────┼─────┼─────┤
  │ 12  │ 13  │ 15  │  ← row 2
  └─────┴─────┴─────┘
     ↑
   start here (bottom-left)

  Staircase walk:
  (2,0): 12 > 8 → go up ↑
  (1,0): 10 > 8 → go up ↑
  (0,0): 1 ≤ 8 → count += 1, go right →
  (0,1): 5 ≤ 8 → count += 1, go right →
  (0,2): 9 > 8 → go up ↑ (out of bounds)

  Count = 2 < k=8 ✗ → left = 9

Step 2: mid = 12, count elements ≤ 12

  ┌─────┬─────┬─────┐
  │ ●1  │ ●5  │ ●9  │  ● = counted
  ├─────┼─────┼─────┤
  │ ●10 │ ●11 │ 13  │
  ├─────┼─────┼─────┤
  │ ●12 │ 13  │ 15  │
  └─────┴─────┴─────┘

  (2,0): 12 ≤ 12 → count += 3 (all above), go right →
  (2,1): 13 > 12 → go up ↑
  (1,1): 11 ≤ 12 → count += 2 (all above), go right →
  (1,2): 13 > 12 → go up ↑
  (0,2): 9 ≤ 12 → count += 1, go right → (done)

  Count = 6 < k=8 ✗ → left = 13

Step 3: mid = 13, count elements ≤ 13

  ┌─────┬─────┬─────┐
  │ ●1  │ ●5  │ ●9  │  All ● counted
  ├─────┼─────┼─────┤
  │ ●10 │ ●11 │ ●13 │
  ├─────┼─────┼─────┤
  │ ●12 │ ●13 │ 15  │
  └─────┴─────┴─────┘

  Count = 8 ≥ k=8 ✓ → right = 13

Answer: 13 (left == right)

Staircase Walk Pattern:
  Start: bottom-left (row=n-1, col=0)

  if matrix[row][col] ≤ target:
      count += (row + 1)  ← all elements above are smaller
      col += 1            ← move right →
  else:
      row -= 1            ← move up ↑

Key Insight:
- O(n) counting per binary search iteration
- Total: O(n × log(max - min))
- Works because matrix sorted both ways
- Each step eliminates a row or column
```

### Solution
```python
def kthSmallest(matrix: list[list[int]], k: int) -> int:
    """
    Find kth smallest using binary search on value.

    Strategy:
    - Binary search on value range
    - Count elements <= mid efficiently

    Time: O(n * log(max - min))
    Space: O(1)
    """
    n = len(matrix)

    def count_less_equal(target: int) -> int:
        """Count elements <= target using staircase search."""
        count = 0
        row, col = n - 1, 0  # Start from bottom-left

        while row >= 0 and col < n:
            if matrix[row][col] <= target:
                count += row + 1  # All elements in this column up to row
                col += 1
            else:
                row -= 1

        return count

    left, right = matrix[0][0], matrix[n - 1][n - 1]

    while left < right:
        mid = (left + right) // 2

        if count_less_equal(mid) < k:
            left = mid + 1
        else:
            right = mid

    return left
```

### Edge Cases
- k = 1 → return top-left element
- k = n*n → return bottom-right element
- 1x1 matrix → return single element
- All same elements → return that element

---

## Problem 6: Aggressive Cows / Magnetic Force (LC #1552) - Medium

- [LeetCode](https://leetcode.com/problems/aggressive-cows-magnetic-force/)

### Video Explanation
- [NeetCode - Magnetic Force Between Two Balls](https://www.youtube.com/watch?v=WGQvfVLuMGQ)

### Problem Statement
Place m balls in n positions to maximize minimum distance.

### Examples
```
Input: position = [1,2,3,4,7], m = 3
Output: 3 (place at 1, 4, 7)
```


### Visual Intuition
```
Aggressive Cows / Magnetic Force
positions = [1, 2, 3, 4, 7], m = 3 balls

Pattern: Binary Search on Answer (maximize minimum)
Why: If we can achieve min_dist=X, we can achieve any Y < X

Step 0 (Setup):
  Sorted positions: [1, 2, 3, 4, 7]
  Search space: [1, 7-1] = [1, 6]
                 L          R

  Number line:
  1   2   3   4   5   6   7
  ●───●───●───●───────────●
  positions to place balls

Step 1: mid = 3, can we place 3 balls with min_dist ≥ 3?

  1   2   3   4   5   6   7
  ●───●───●───●───────────●
  ⚫          ⚫          ⚫
  ↑           ↑           ↑
  ball1     ball2       ball3

  Greedy placement:
  • Place ball1 at position 1
  • Next valid: 1 + 3 = 4, place ball2 at 4
  • Next valid: 4 + 3 = 7, place ball3 at 7

  Placed 3 balls ✓ → try larger, left = 4

Step 2: mid = 5, can we place 3 balls with min_dist ≥ 5?

  1   2   3   4   5   6   7
  ●───●───●───●───────────●
  ⚫                      ⚫
  ↑                       ↑
  ball1                 ball2

  • Place ball1 at 1
  • Next valid: 1 + 5 = 6, place ball2 at 7
  • Next valid: 7 + 5 = 12, no position ✗

  Only 2 balls ✗ → right = 4

Step 3: mid = 4, can we place 3 balls with min_dist ≥ 4?

  1   2   3   4   5   6   7
  ●───●───●───●───────────●
  ⚫                      ⚫

  • Place at 1, next ≥ 5, only 7 works
  • Only 2 balls ✗ → right = 3

Step 4: left = 3, right = 3 → Answer: 3

Optimal Placement:
  1   2   3   4   5   6   7
  ⚫──────────⚫──────────⚫
  ↑    3     ↑    3     ↑
  min distance = 3 (maximized)

Key Insight:
- Use upper-mid for maximization: mid = (L + R + 1) // 2
- Greedy works: always place at first valid position
- Monotonic: if min_dist=X works, X-1 also works
- Placing at first valid position maximizes remaining space

Why Greedy Works:
  Placing earlier never hurts - leaves more room for later balls
  If we skip a valid position, we can only do worse
```

### Solution
```python
def maxDistance(position: list[int], m: int) -> int:
    """
    Maximize minimum distance using binary search.

    Strategy:
    - Binary search on minimum distance
    - Check if we can place m balls with given min distance

    Time: O(n log n + n log W)
    Space: O(1)
    """
    position.sort()

    def can_place(min_dist: int) -> bool:
        """Check if we can place m balls with min_dist apart."""
        count = 1
        last_pos = position[0]

        for pos in position[1:]:
            if pos - last_pos >= min_dist:
                count += 1
                last_pos = pos

                if count >= m:
                    return True

        return False

    # Binary search on minimum distance
    left, right = 1, position[-1] - position[0]

    while left < right:
        mid = (left + right + 1) // 2  # Upper mid for maximization

        if can_place(mid):
            left = mid  # Try larger distance
        else:
            right = mid - 1

    return left
```

### Edge Cases
- m = 2 → place at extremes
- m = n → place at each position
- Evenly spaced → optimal is spacing
- Clustered positions → limited options

---

## Problem 7: Capacity To Ship Packages (LC #1011) - Medium

- [LeetCode](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)

### Video Explanation
- [NeetCode - Capacity To Ship Packages Within D Days](https://www.youtube.com/watch?v=ER_oLmdc-nw)

### Problem Statement
Find minimum ship capacity to ship all packages in D days.


### Visual Intuition
```
Capacity To Ship Packages Within D Days
weights = [1,2,3,4,5,6,7,8,9,10], days = 5

Pattern: Binary Search on Answer (minimize capacity)
Why: If capacity X works, any Y > X also works (monotonic)

Step 0 (Define Search Space):
  min capacity = max(weights) = 10  (must fit largest package)
  max capacity = sum(weights) = 55  (ship all in one day)

  Search: [10, 55]
           L    R

Step 1: mid = 32, can we ship in ≤5 days?

  Day 1: [1,2,3,4,5,6,7] = 28 ≤ 32 ✓
         【──────────────】
  Day 2: [8,9,10] = 27 ≤ 32 ✓
         【───────】

  Days needed: 2 ≤ 5 ✓ → try smaller, right = 32

Step 2: mid = 21, can we ship in ≤5 days?

  Day 1: [1,2,3,4,5,6] = 21 ✓
  Day 2: [7,8] = 15 ✓
  Day 3: [9,10] = 19 ✓

  Days needed: 3 ≤ 5 ✓ → try smaller, right = 21

Step 3: mid = 15, can we ship in ≤5 days?

  weights: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
           【──────────】【───】【─】【─】【──】
              Day 1      Day2  D3  D4  Day5
              1+2+3+4+5  6+7   8   9   10
                 =15     =13   =8  =9  =10

  Days needed: 5 ≤ 5 ✓ → try smaller, right = 15

Step 4: mid = 12, can we ship in ≤5 days?

  Day 1: 1+2+3+4 = 10 (can't add 5, would be 15 > 12)
  Day 2: 5+6 = 11
  Day 3: 7 = 7 (can't add 8, would be 15 > 12)
  Day 4: 8 = 8
  Day 5: 9 = 9
  Day 6: 10 = 10

  Days needed: 6 > 5 ✗ → left = 13

Step 5: mid = 14, days needed = 6 > 5 ✗ → left = 15

Answer: 15 (left == right)

Greedy Loading Visualization:
  Capacity = 15
  ┌─────────────────┐
  │ 1+2+3+4+5 = 15  │ Day 1: full
  ├─────────────────┤
  │ 6+7 = 13        │ Day 2: can't add 8
  ├─────────────────┤
  │ 8 = 8           │ Day 3: can't add 9
  ├─────────────────┤
  │ 9 = 9           │ Day 4: can't add 10
  ├─────────────────┤
  │ 10 = 10         │ Day 5: last package
  └─────────────────┘

Key Insight:
- Greedy: pack until capacity exceeded, start new day
- Order matters: packages must ship in given order
- min = max(weights) ensures every package fits
- Binary search finds minimum capacity meeting deadline
```

### Solution
```python
def shipWithinDays(weights: list[int], days: int) -> int:
    """
    Find minimum capacity using binary search.

    Strategy:
    - Binary search on capacity
    - Check if we can ship within days limit

    Time: O(n * log(sum - max))
    Space: O(1)
    """
    def can_ship(capacity: int) -> bool:
        """Check if we can ship all packages within days."""
        days_needed = 1
        current_load = 0

        for weight in weights:
            if current_load + weight > capacity:
                days_needed += 1
                current_load = weight
            else:
                current_load += weight

        return days_needed <= days

    # Binary search: [max weight, total weight]
    left = max(weights)
    right = sum(weights)

    while left < right:
        mid = (left + right) // 2

        if can_ship(mid):
            right = mid
        else:
            left = mid + 1

    return left
```

### Edge Cases
- days = n → capacity = max weight
- days = 1 → capacity = sum of weights
- Single package → return its weight
- All same weight → simple calculation

---

## Problem 8: Koko Eating Bananas (LC #875) - Medium

- [LeetCode](https://leetcode.com/problems/koko-eating-bananas/)

### Video Explanation
- [NeetCode - Koko Eating Bananas](https://www.youtube.com/watch?v=U2SozAs9RzA)

### Problem Statement
Find minimum eating speed to finish all bananas in h hours.


### Visual Intuition
```
Koko Eating Bananas
piles = [3, 6, 7, 11], h = 8 hours

Pattern: Binary Search on Answer (minimize speed)
Why: If speed X works, any Y > X also works (monotonic)

Step 0 (Setup):
  Search space: [1, max(piles)] = [1, 11]

  piles visualization:
  Pile 1: 🍌🍌🍌         (3 bananas)
  Pile 2: 🍌🍌🍌🍌🍌🍌     (6 bananas)
  Pile 3: 🍌🍌🍌🍌🍌🍌🍌    (7 bananas)
  Pile 4: 🍌🍌🍌🍌🍌🍌🍌🍌🍌🍌🍌 (11 bananas)

Step 1: mid = 6, hours needed at speed 6?

  Pile 3:  ⌈3/6⌉  = 1 hour  🍌🍌🍌
  Pile 6:  ⌈6/6⌉  = 1 hour  🍌🍌🍌🍌🍌🍌
  Pile 7:  ⌈7/6⌉  = 2 hours 🍌🍌🍌🍌🍌🍌|🍌
  Pile 11: ⌈11/6⌉ = 2 hours 🍌🍌🍌🍌🍌🍌|🍌🍌🍌🍌🍌
                            ─────────  ─────────
                             hour 1     hour 2

  Total = 1+1+2+2 = 6 hours ≤ 8 ✓ → try smaller, right = 6

Step 2: mid = 3, hours needed at speed 3?

  Pile 3:  ⌈3/3⌉  = 1 hour
  Pile 6:  ⌈6/3⌉  = 2 hours
  Pile 7:  ⌈7/3⌉  = 3 hours
  Pile 11: ⌈11/3⌉ = 4 hours

  Total = 1+2+3+4 = 10 hours > 8 ✗ → left = 4

Step 3: mid = 5, hours needed at speed 5?

  Pile 3:  ⌈3/5⌉  = 1 hour
  Pile 6:  ⌈6/5⌉  = 2 hours
  Pile 7:  ⌈7/5⌉  = 2 hours
  Pile 11: ⌈11/5⌉ = 3 hours

  Total = 1+2+2+3 = 8 hours ≤ 8 ✓ → right = 5

Step 4: mid = 4, hours needed at speed 4?

  Pile 3:  ⌈3/4⌉  = 1 hour
  Pile 6:  ⌈6/4⌉  = 2 hours
  Pile 7:  ⌈7/4⌉  = 2 hours
  Pile 11: ⌈11/4⌉ = 3 hours

  Total = 1+2+2+3 = 8 hours ≤ 8 ✓ → right = 4

Answer: 4 bananas/hour (left == right)

Timeline at speed 4:
  Hour: 1   2   3   4   5   6   7   8
        ├───┼───┼───┼───┼───┼───┼───┼───┤
  Pile1 │███│   │   │   │   │   │   │   │  (3→0)
  Pile2 │   │███│███│   │   │   │   │   │  (6→2→0)
  Pile3 │   │   │   │███│███│   │   │   │  (7→3→0)
  Pile4 │   │   │   │   │   │███│███│███│  (11→7→3→0)

Key Insight:
- Each pile takes ⌈pile/speed⌉ hours (ceiling division)
- Koko can only eat from one pile per hour
- Even if she finishes early, she waits for next hour
- Monotonic: slower speed → more hours needed
```

### Solution
```python
import math

def minEatingSpeed(piles: list[int], h: int) -> int:
    """
    Find minimum eating speed using binary search.

    Strategy:
    - Binary search on speed
    - Calculate hours needed for each speed

    Time: O(n * log(max))
    Space: O(1)
    """
    def hours_needed(speed: int) -> int:
        """Calculate total hours to eat all bananas at given speed."""
        return sum(math.ceil(pile / speed) for pile in piles)

    # Binary search: [1, max pile]
    left, right = 1, max(piles)

    while left < right:
        mid = (left + right) // 2

        if hours_needed(mid) <= h:
            right = mid  # Try slower speed
        else:
            left = mid + 1  # Need faster speed

    return left
```

### Edge Cases
- h = n → speed = 1 works
- h = sum of piles → speed = 1
- Single pile → return ceil(pile/h)
- All piles size 1 → speed = 1

---

## Problem 9: Find in Mountain Array (LC #1095) - Hard

- [LeetCode](https://leetcode.com/problems/find-in-mountain-array/)

### Video Explanation
- [NeetCode - Find in Mountain Array](https://www.youtube.com/watch?v=pJyzxE7IqkM)

### Problem Statement
Find target in mountain array with minimum API calls.


### Visual Intuition
```
Find in Mountain Array
mountainArr = [1, 2, 3, 4, 5, 3, 1], target = 3

Pattern: Three Binary Searches (Peak + Ascending + Descending)
Why: Mountain array = ascending + descending, search both halves

Step 0 (Visualize Mountain):

       5 ← peak
      /\
     4  3
    /    \
   3      1
  /
 2
/
1

  indices: 0  1  2  3  4  5  6
  values:  1  2  3  4  5  3  1
           ↑──ascending──↑──desc──↑

Step 1: Find Peak (Binary Search #1)

  [1, 2, 3, 4, 5, 3, 1]
   L        M        R

  nums[M]=4 < nums[M+1]=5 → peak is right, L = M+1

  [1, 2, 3, 4, 5, 3, 1]
               L  M  R

  nums[M]=3 > nums[M+1]=1 → peak is left or M, R = M

  [1, 2, 3, 4, 5, 3, 1]
               LR
               ↑
  Peak found at index 4, value = 5

Step 2: Search Ascending Part [0, peak] (Binary Search #2)

  [1, 2, 3, 4, 5]  target = 3
   L     M     R

  nums[M]=3 == target ✓ Found at index 2!

  Return immediately (minimize API calls)

Step 3: (Would run if Step 2 failed)
  Search descending part [peak+1, n-1]

  [3, 1]  (indices 5, 6)

  Note: Binary search reversed for descending:
    if nums[mid] < target: right = mid - 1
    if nums[mid] > target: left = mid + 1

Answer: 2

API Call Optimization:
  ┌────────────────────────────────────┐
  │ Problem: Minimize mountainArr.get()│
  │                                    │
  │ Strategy:                          │
  │ 1. Find peak: O(log n) calls       │
  │ 2. Search ascending: O(log n)      │
  │ 3. Search descending: O(log n)     │
  │                                    │
  │ Total: O(log n) calls              │
  │ Return ASAP when found             │
  └────────────────────────────────────┘

Key Insight:
- Peak finding: if arr[mid] < arr[mid+1], peak is right
- Ascending search: normal binary search
- Descending search: reversed comparison
- Always search ascending first (return smaller index)

Why This Order:
  target = 3 appears at indices 2 AND 5
  We want index 2 (smaller), so search ascending first
```

### Solution
```python
def findInMountainArray(target: int, mountain_arr) -> int:
    """
    Find target in mountain array.

    Strategy:
    1. Find peak using binary search
    2. Binary search in ascending part
    3. If not found, binary search in descending part

    Time: O(log n)
    Space: O(1)
    """
    n = mountain_arr.length()

    # Step 1: Find peak
    left, right = 0, n - 1
    while left < right:
        mid = (left + right) // 2
        if mountain_arr.get(mid) < mountain_arr.get(mid + 1):
            left = mid + 1
        else:
            right = mid
    peak = left

    # Step 2: Search in ascending part (0 to peak)
    left, right = 0, peak
    while left <= right:
        mid = (left + right) // 2
        val = mountain_arr.get(mid)
        if val == target:
            return mid
        elif val < target:
            left = mid + 1
        else:
            right = mid - 1

    # Step 3: Search in descending part (peak+1 to n-1)
    left, right = peak + 1, n - 1
    while left <= right:
        mid = (left + right) // 2
        val = mountain_arr.get(mid)
        if val == target:
            return mid
        elif val < target:
            right = mid - 1  # Descending, so go left
        else:
            left = mid + 1

    return -1
```

### Edge Cases
- Target at peak → found in step 2
- Target in ascending only → found in step 2
- Target in descending only → found in step 3
- Target not in array → return -1
- Multiple occurrences → return leftmost index

---

## Summary: Binary Search Hard Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Median Two Arrays | Partition binary search | O(log min(m,n)) |
| 2 | Split Array | Search on answer | O(n log W) |
| 3 | Rotated Min II | Handle duplicates | O(n) worst |
| 4 | Kth Pair Distance | Search + two pointers | O(n log W) |
| 5 | Kth in Matrix | Staircase counting | O(n log W) |
| 6 | Aggressive Cows | Maximize minimum | O(n log W) |
| 7 | Ship Capacity | Minimize maximum | O(n log W) |
| 8 | Koko Bananas | Search on speed | O(n log max) |
| 9 | Mountain Array | Find peak + search | O(log n) |

---

## Binary Search on Answer Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BINARY SEARCH ON ANSWER                                  │
│                                                                             │
│  When to use:                                                               │
│  • "Minimum/maximum value that satisfies condition"                         │
│  • Condition is monotonic (if X works, all X+1 work OR all X-1 work)       │
│                                                                             │
│  Template for MINIMIZATION:                                                 │
│  while left < right:                                                        │
│      mid = (left + right) // 2                                              │
│      if condition(mid):                                                     │
│          right = mid      # Try smaller                                     │
│      else:                                                                  │
│          left = mid + 1   # Need larger                                     │
│                                                                             │
│  Template for MAXIMIZATION:                                                 │
│  while left < right:                                                        │
│      mid = (left + right + 1) // 2  # Upper mid                            │
│      if condition(mid):                                                     │
│          left = mid       # Try larger                                      │
│      else:                                                                  │
│          right = mid - 1  # Need smaller                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practice More Problems

- [ ] LC #668 - Kth Smallest Number in Multiplication Table
- [ ] LC #774 - Minimize Max Distance to Gas Station
- [ ] LC #786 - K-th Smallest Prime Fraction
- [ ] LC #1231 - Divide Chocolate
- [ ] LC #1482 - Minimum Number of Days to Make m Bouquets

