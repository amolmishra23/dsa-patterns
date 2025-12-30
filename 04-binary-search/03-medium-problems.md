# Binary Search - Medium Problems

## Problem 1: Search in Rotated Sorted Array (LC #33) - Medium

- [LeetCode](https://leetcode.com/problems/search-in-rotated-sorted-array/)

### Problem Statement
There is an integer array `nums` sorted in ascending order (with distinct values). Prior to being passed to your function, `nums` is possibly rotated at an unknown pivot index. Given the array `nums` after the possible rotation and an integer `target`, return the index of `target` if it is in `nums`, or `-1` if it is not.

### Video Explanation
- [NeetCode - Search in Rotated Sorted Array](https://www.youtube.com/watch?v=U8XENwh8Oy8)

### Examples
```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Explanation: 0 is at index 4.

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
Explanation: 3 is not in the array.

Input: nums = [1], target = 0
Output: -1
```

### Intuition Development
```
Key insight: One half is ALWAYS sorted!

Original sorted:  [0, 1, 2, 4, 5, 6, 7]
Rotated at 4:     [4, 5, 6, 7, 0, 1, 2]
                   ↑ sorted ↑  ↑ sorted ↑

How to identify which half is sorted?
┌─────────────────────────────────────────────────────────────────┐
│ Compare nums[left] with nums[mid]:                              │
│                                                                  │
│ nums = [4, 5, 6, 7, 0, 1, 2], target = 0                        │
│         L        M        R                                      │
│                                                                  │
│ nums[left]=4 <= nums[mid]=7 → LEFT half is sorted [4,5,6,7]    │
│                                                                  │
│ Is target in sorted half? 4 <= 0 < 7? NO                        │
│ → Search RIGHT half                                              │
│                                                                  │
│ nums = [4, 5, 6, 7, 0, 1, 2], target = 0                        │
│                     L  M  R                                      │
│                                                                  │
│ nums[left]=0 <= nums[mid]=1 → LEFT half is sorted [0,1]        │
│ Is target in sorted half? 0 <= 0 < 1? YES                       │
│ → Search LEFT half                                               │
│                                                                  │
│ Found at index 4! ✓                                              │
└─────────────────────────────────────────────────────────────────┘

Decision tree:
  If LEFT half sorted (nums[left] <= nums[mid]):
    If target in [nums[left], nums[mid)): search LEFT
    Else: search RIGHT
  Else (RIGHT half sorted):
    If target in (nums[mid], nums[right]]: search RIGHT
    Else: search LEFT
```

### Solution
```python
def search(nums: list[int], target: int) -> int:
    """
    Search in rotated sorted array.

    Strategy:
    1. Find mid point
    2. Determine which half is sorted
    3. Check if target is in sorted half
    4. Narrow search accordingly

    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        # Found target
        if nums[mid] == target:
            return mid

        # Left half is sorted
        if nums[left] <= nums[mid]:
            # Target is in left sorted half
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # Right half is sorted
        else:
            # Target is in right sorted half
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

### Complexity
- **Time**: O(log n) - Standard binary search
- **Space**: O(1) - Only use pointers

### Edge Cases
- Single element array: Check directly
- Target at rotation point: Algorithm handles naturally
- No rotation (already sorted): Works like standard binary search
- Target not in array: Returns -1 after search exhausts

---

## Problem 2: Find Minimum in Rotated Sorted Array (LC #153) - Medium

- [LeetCode](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

### Problem Statement
Suppose an array of length `n` sorted in ascending order is rotated between 1 and n times. Given the sorted rotated array `nums` of unique elements, return the minimum element of this array. You must write an algorithm that runs in O(log n) time.

### Video Explanation
- [NeetCode - Find Minimum in Rotated Sorted Array](https://www.youtube.com/watch?v=nIVW4P8b1VA)

### Examples
```
Input: nums = [3,4,5,1,2]
Output: 1
Explanation: Original array was [1,2,3,4,5] rotated 3 times.

Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: Original array was [0,1,2,4,5,6,7] rotated 4 times.

Input: nums = [11,13,15,17]
Output: 11
Explanation: Array was not rotated (or rotated n times).
```

### Intuition Development
```
The minimum is always at the "rotation point" (inflection)!

[4, 5, 6, 7, 0, 1, 2]
             ↑
          minimum

Key insight: Compare mid with RIGHT element!

┌─────────────────────────────────────────────────────────────────┐
│ Why compare with RIGHT (not left)?                              │
│                                                                  │
│ Case 1: nums[mid] > nums[right]                                 │
│   [4, 5, 6, 7, 0, 1, 2]                                        │
│    L        M        R                                          │
│   7 > 2 → Minimum is in RIGHT half (after mid)                 │
│                                                                  │
│ Case 2: nums[mid] <= nums[right]                                │
│   [4, 5, 6, 7, 0, 1, 2]                                        │
│                  L M R                                          │
│   0 <= 2 → Minimum is in LEFT half (including mid)             │
│                                                                  │
│ Step-by-step for [3, 4, 5, 1, 2]:                              │
│   L=0, R=4, mid=2: nums[2]=5 > nums[4]=2 → L=3                 │
│   L=3, R=4, mid=3: nums[3]=1 <= nums[4]=2 → R=3                │
│   L=3, R=3: Exit loop, return nums[3]=1 ✓                      │
└─────────────────────────────────────────────────────────────────┘

Note: Use left < right (not <=) because we're finding a value, not exact match.
```

### Solution
```python
def findMin(nums: list[int]) -> int:
    """
    Find minimum in rotated sorted array.

    Strategy:
    - Compare mid with right element
    - If mid > right, minimum is on right side
    - Otherwise, minimum is on left side (including mid)

    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] > nums[right]:
            # Minimum is in right half
            left = mid + 1
        else:
            # Minimum is in left half (including mid)
            right = mid

    return nums[left]
```

### Complexity
- **Time**: O(log n) - Standard binary search
- **Space**: O(1) - Only use pointers

### Edge Cases
- No rotation: Minimum is first element
- Single element: Return that element
- Two elements: Return the smaller one
- Rotated n times (full rotation): Same as no rotation

---

## Problem 3: Find Peak Element (LC #162) - Medium

- [LeetCode](https://leetcode.com/problems/find-peak-element/)

### Problem Statement
A peak element is an element that is strictly greater than its neighbors. Given a 0-indexed integer array `nums`, find a peak element, and return its index. If the array contains multiple peaks, return the index to **any** of the peaks. You may imagine that `nums[-1] = nums[n] = -∞`.

### Video Explanation
- [NeetCode - Find Peak Element](https://www.youtube.com/watch?v=kMzJy9es7Hc)

### Examples
```
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element at index 2.

Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: 6 at index 5 is a peak. Index 1 (value 2) is also valid.

Input: nums = [1]
Output: 0
```

### Intuition Development
```
Key insight: Follow the "uphill" slope - guaranteed to find a peak!

Why? Because edges are -∞, we can't "fall off" without finding a peak.

[1, 2, 3, 1]
       ↑ peak (3 > 2 and 3 > 1)

┌─────────────────────────────────────────────────────────────────┐
│ Compare nums[mid] with nums[mid+1]:                             │
│                                                                  │
│ nums = [1, 2, 1, 3, 5, 6, 4]                                    │
│         L        M        R                                      │
│                                                                  │
│ nums[mid]=3 < nums[mid+1]=5 → Peak is on RIGHT                 │
│ (We're on upward slope, peak must be ahead)                     │
│                                                                  │
│ nums = [1, 2, 1, 3, 5, 6, 4]                                    │
│                     L  M  R                                      │
│                                                                  │
│ nums[mid]=6 > nums[mid+1]=4 → Peak is on LEFT (including mid)  │
│ (We're on downward slope, peak is behind or at mid)             │
│                                                                  │
│ Visualization:                                                   │
│       6                                                          │
│     5 ↑ 4                                                        │
│   3 ↑   ↓                                                        │
│ 1 2 ↓                                                            │
│   ↓ 1                                                            │
│                                                                  │
│ Always move toward the higher neighbor!                         │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def findPeakElement(nums: list[int]) -> int:
    """
    Find any peak element using binary search.

    Strategy:
    - Compare mid with mid+1
    - Move toward the higher neighbor
    - Eventually converge to a peak

    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] < nums[mid + 1]:
            # Peak is on the right
            left = mid + 1
        else:
            # Peak is on the left (including mid)
            right = mid

    return left
```

### Complexity
- **Time**: O(log n) - Standard binary search
- **Space**: O(1) - Only use pointers

### Edge Cases
- Single element: That element is the peak (return 0)
- Two elements: Return index of larger element
- Strictly increasing: Last element is peak
- Strictly decreasing: First element is peak
- Multiple peaks: Any peak is valid answer

---

## Problem 4: Koko Eating Bananas (LC #875) - Medium

- [LeetCode](https://leetcode.com/problems/koko-eating-bananas/)

### Problem Statement
Koko loves to eat bananas. There are `n` piles of bananas, the `ith` pile has `piles[i]` bananas. The guards will return in `h` hours. Koko can decide her eating speed `k` (bananas per hour). Each hour, she chooses a pile and eats `k` bananas. If the pile has fewer than `k` bananas, she eats all and won't eat any more that hour. Return the minimum integer `k` such that she can eat all bananas within `h` hours.

### Video Explanation
- [NeetCode - Koko Eating Bananas](https://www.youtube.com/watch?v=U2SozAs9RzA)

### Examples
```
Input: piles = [3,6,7,11], h = 8
Output: 4
Explanation: At speed 4:
  - Pile 3: ceil(3/4) = 1 hour
  - Pile 6: ceil(6/4) = 2 hours
  - Pile 7: ceil(7/4) = 2 hours
  - Pile 11: ceil(11/4) = 3 hours
  Total: 1+2+2+3 = 8 hours ✓

Input: piles = [30,11,23,4,20], h = 5
Output: 30
Explanation: Need to finish 5 piles in 5 hours, so each pile in 1 hour.

Input: piles = [30,11,23,4,20], h = 6
Output: 23
```

### Intuition Development
```
This is "Binary Search on Answer" pattern!

The answer (speed k) has a monotonic property:
  - If speed k works, all speeds > k also work
  - If speed k doesn't work, all speeds < k also don't work

Search space: [1, max(piles)]
┌─────────────────────────────────────────────────────────────────┐
│ piles = [3, 6, 7, 11], h = 8                                    │
│                                                                  │
│ Speed range: 1 to 11 (max pile)                                 │
│                                                                  │
│ Speed 1: ceil(3/1)+ceil(6/1)+ceil(7/1)+ceil(11/1) = 27 hours ✗ │
│ Speed 6: ceil(3/6)+ceil(6/6)+ceil(7/6)+ceil(11/6) = 1+1+2+2=6 ✓│
│ Speed 4: ceil(3/4)+ceil(6/4)+ceil(7/4)+ceil(11/4) = 1+2+2+3=8 ✓│
│ Speed 3: ceil(3/3)+ceil(6/3)+ceil(7/3)+ceil(11/3) = 1+2+3+4=10✗│
│                                                                  │
│ Binary search for minimum speed that finishes in ≤ h hours!     │
│                                                                  │
│ L=1, R=11, mid=6: 6 hours ≤ 8 ✓ → R=6                          │
│ L=1, R=6, mid=3: 10 hours > 8 ✗ → L=4                           │
│ L=4, R=6, mid=5: 7 hours ≤ 8 ✓ → R=5                            │
│ L=4, R=5, mid=4: 8 hours ≤ 8 ✓ → R=4                            │
│ L=4, R=4: Answer = 4 ✓                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def minEatingSpeed(piles: list[int], h: int) -> int:
    """
    Find minimum eating speed to finish in h hours.

    Strategy:
    - Binary search on speed [1, max(piles)]
    - For each speed, calculate total hours needed
    - Find minimum speed where hours <= h

    Time: O(n * log(max_pile))
    Space: O(1)
    """
    def can_finish(speed: int) -> bool:
        """Check if Koko can finish at given speed."""
        hours = 0
        for pile in piles:
            # Ceiling division: (pile + speed - 1) // speed
            hours += (pile + speed - 1) // speed
        return hours <= h

    left, right = 1, max(piles)

    while left < right:
        mid = left + (right - left) // 2

        if can_finish(mid):
            # Can finish, try smaller speed
            right = mid
        else:
            # Can't finish, need faster speed
            left = mid + 1

    return left
```

### Complexity
- **Time**: O(n × log(max_pile)) - Binary search on speed, O(n) check per speed
- **Space**: O(1) - Only use variables

### Edge Cases
- `h == n`: Must eat each pile in 1 hour, answer = max(piles)
- `h` very large: Can eat slowly, answer = 1
- Single pile: Answer = ceil(pile/h)
- All piles same size: Simplified calculation

---

## Problem 5: Capacity To Ship Packages (LC #1011) - Medium

- [LeetCode](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/)

### Problem Statement
A conveyor belt has packages that must be shipped from one port to another within `days` days. The `ith` package has weight `weights[i]`. Each day, we load packages in order (we can't skip packages). We cannot load more weight than the ship's capacity. Return the **least** weight capacity of the ship that will ship all packages within `days` days.

### Video Explanation
- [NeetCode - Capacity To Ship Packages](https://www.youtube.com/watch?v=ER_oLmdc-nw)

### Examples
```
Input: weights = [1,2,3,4,5,6,7,8,9,10], days = 5
Output: 15
Explanation:
  Day 1: 1,2,3,4,5 (sum=15)
  Day 2: 6,7 (sum=13)
  Day 3: 8 (sum=8)
  Day 4: 9 (sum=9)
  Day 5: 10 (sum=10)

Input: weights = [3,2,2,4,1,4], days = 3
Output: 6

Input: weights = [1,2,3,1,1], days = 4
Output: 3
```

### Intuition Development
```
Another "Binary Search on Answer" problem!

Search space: [max(weights), sum(weights)]
┌─────────────────────────────────────────────────────────────────┐
│ Why these bounds?                                                │
│                                                                  │
│ Minimum capacity = max(weights)                                  │
│   Must at least carry the heaviest single package               │
│                                                                  │
│ Maximum capacity = sum(weights)                                  │
│   Can carry everything in one day                               │
│                                                                  │
│ weights = [1,2,3,4,5,6,7,8,9,10], days = 5                      │
│ Search range: [10, 55]                                          │
│                                                                  │
│ Simulation for capacity = 15:                                    │
│   Day 1: load 1,2,3,4,5 (sum=15) → next day                     │
│   Day 2: load 6,7 (sum=13, can't add 8) → next day              │
│   Day 3: load 8 (sum=8, can't add 9) → next day                 │
│   Day 4: load 9 (sum=9, can't add 10) → next day                │
│   Day 5: load 10 (sum=10)                                        │
│   Total: 5 days ✓                                                │
│                                                                  │
│ Binary search: find minimum capacity where days_needed ≤ days   │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def shipWithinDays(weights: list[int], days: int) -> int:
    """
    Find minimum ship capacity to ship within days.

    Strategy:
    - Binary search on capacity
    - For each capacity, simulate and count days needed
    - Find minimum capacity where days_needed <= days

    Time: O(n * log(sum - max))
    Space: O(1)
    """
    def can_ship(capacity: int) -> bool:
        """Check if we can ship within days at given capacity."""
        days_needed = 1
        current_load = 0

        for weight in weights:
            if current_load + weight > capacity:
                # Start new day
                days_needed += 1
                current_load = weight
            else:
                current_load += weight

        return days_needed <= days

    left = max(weights)  # Must carry heaviest
    right = sum(weights)  # Carry all in one day

    while left < right:
        mid = left + (right - left) // 2

        if can_ship(mid):
            right = mid
        else:
            left = mid + 1

    return left
```

### Complexity
- **Time**: O(n × log(sum - max)) - Binary search on capacity, O(n) simulation per capacity
- **Space**: O(1) - Only use variables

### Edge Cases
- `days == n`: Each package on separate day, answer = max(weights)
- `days == 1`: Ship all in one day, answer = sum(weights)
- Single package: Answer = that package's weight
- All same weight: Simplified calculation

---

## Problem 6: Search a 2D Matrix (LC #74) - Medium

- [LeetCode](https://leetcode.com/problems/search-a-2d-matrix/)

### Problem Statement
You are given an `m x n` integer matrix `matrix` with the following properties:
- Each row is sorted in non-decreasing order.
- The first integer of each row is greater than the last integer of the previous row.
Given an integer `target`, return `true` if `target` is in `matrix` or `false` otherwise. You must write a solution in O(log(m * n)) time complexity.

### Video Explanation
- [NeetCode - Search a 2D Matrix](https://www.youtube.com/watch?v=Ber2pi2C0j0)

### Examples
```
Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
Output: true

Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13
Output: false

Input: matrix = [[1]], target = 1
Output: true
```

### Intuition Development
```
The matrix is essentially a sorted 1D array!

matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]

Flatten conceptually:
[1, 3, 5, 7, 10, 11, 16, 20, 23, 30, 34, 60]
 0  1  2  3   4   5   6   7   8   9  10  11

┌─────────────────────────────────────────────────────────────────┐
│ Key: Convert 1D index to 2D coordinates!                        │
│                                                                  │
│ For index i in flattened array:                                 │
│   row = i // cols                                               │
│   col = i % cols                                                │
│                                                                  │
│ Example: matrix is 3×4 (rows=3, cols=4)                         │
│   index 5 → row = 5//4 = 1, col = 5%4 = 1 → matrix[1][1] = 11  │
│   index 10 → row = 10//4 = 2, col = 10%4 = 2 → matrix[2][2]=34 │
│                                                                  │
│ Binary search on [0, m*n-1]:                                    │
│   L=0, R=11, mid=5: matrix[1][1]=11 > 3 → R=4                   │
│   L=0, R=4, mid=2: matrix[0][2]=5 > 3 → R=1                     │
│   L=0, R=1, mid=0: matrix[0][0]=1 < 3 → L=1                     │
│   L=1, R=1, mid=1: matrix[0][1]=3 == 3 → FOUND! ✓               │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def searchMatrix(matrix: list[list[int]], target: int) -> bool:
    """
    Search in sorted 2D matrix.

    Strategy:
    - Treat as 1D sorted array
    - Convert 1D index to 2D: row = idx // cols, col = idx % cols

    Time: O(log(m*n))
    Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False

    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1

    while left <= right:
        mid = left + (right - left) // 2

        # Convert 1D index to 2D
        row = mid // cols
        col = mid % cols
        value = matrix[row][col]

        if value == target:
            return True
        elif value < target:
            left = mid + 1
        else:
            right = mid - 1

    return False
```

### Alternative: Two Binary Searches
```python
def searchMatrix_two_searches(matrix: list[list[int]], target: int) -> bool:
    """
    Alternative: Binary search for row, then for column.

    Time: O(log m + log n)
    Space: O(1)
    """
    if not matrix or not matrix[0]:
        return False

    rows, cols = len(matrix), len(matrix[0])

    # Binary search for correct row
    top, bottom = 0, rows - 1
    while top <= bottom:
        mid_row = (top + bottom) // 2
        if target < matrix[mid_row][0]:
            bottom = mid_row - 1
        elif target > matrix[mid_row][-1]:
            top = mid_row + 1
        else:
            break

    if top > bottom:
        return False

    row = (top + bottom) // 2

    # Binary search within row
    left, right = 0, cols - 1
    while left <= right:
        mid = (left + right) // 2
        if matrix[row][mid] == target:
            return True
        elif matrix[row][mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return False
```

### Complexity
- **Time**: O(log(m × n)) - Single binary search over all elements
- **Space**: O(1) - Only use variables

### Edge Cases
- Single element matrix: Check directly
- Target smaller than all: Return false
- Target larger than all: Return false
- Empty matrix: Return false

---

## Summary: Medium Binary Search Problems

| # | Problem | Pattern | Key Insight |
|---|---------|---------|-------------|
| 1 | Search Rotated Array | Modified BS | One half always sorted |
| 2 | Find Min Rotated | Modified BS | Compare mid with right |
| 3 | Find Peak | Modified BS | Follow upward slope |
| 4 | Koko Bananas | Search on Answer | Binary search on speed |
| 5 | Ship Packages | Search on Answer | Binary search on capacity |
| 6 | Search 2D Matrix | Standard BS | Treat as 1D array |

---

## Binary Search on Answer Template

```python
def search_on_answer_template():
    """
    Template for "Binary Search on Answer" problems.

    Use when:
    - Looking for minimum/maximum value satisfying condition
    - Condition is monotonic (if X works, X+1 also works)
    """
    def is_feasible(value):
        # Return True if 'value' satisfies the condition
        pass

    left = minimum_possible_answer
    right = maximum_possible_answer

    while left < right:
        mid = left + (right - left) // 2

        if is_feasible(mid):
            right = mid  # For minimum, or left = mid for maximum
        else:
            left = mid + 1

    return left
```
