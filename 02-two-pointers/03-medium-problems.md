# Two Pointers - Medium Problems

## Problem 1: 3Sum (LC #15) - Medium

- [LeetCode](https://leetcode.com/problems/3sum/)

### Problem Statement
Find all unique triplets in array that sum to zero. No duplicate triplets allowed.

### Video Explanation
- [NeetCode - 3Sum](https://www.youtube.com/watch?v=jzZsG8n2R9A)

### Examples
```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]

Input: nums = [0,1,1]
Output: []

Input: nums = [0,0,0]
Output: [[0,0,0]]
```

### Intuition Development
```
Key insight: Fix one number, then use Two Sum II on the rest!

1. Sort the array
2. For each number nums[i], find pairs in nums[i+1:] that sum to -nums[i]
3. Skip duplicates to avoid duplicate triplets

nums = [-4, -1, -1, 0, 1, 2] (sorted)

i=0, nums[i]=-4, need pairs summing to 4
    Two pointers on [-1, -1, 0, 1, 2] → no pair sums to 4

i=1, nums[i]=-1, need pairs summing to 1
    Two pointers on [-1, 0, 1, 2]
    L=-1, R=2: sum=1 ✓ Found [-1, -1, 2]

i=2, nums[i]=-1, SKIP (duplicate of i=1)

i=3, nums[i]=0, need pairs summing to 0
    Two pointers on [1, 2]
    L=1, R=2: sum=3 > 0, move R... no more pairs

Wait, we missed [-1, 0, 1]! Let me redo...

Actually at i=1, after finding [-1,-1,2], we continue:
    L=-1, R=2 → found, L++, R--
    L=0, R=1: sum=1 ✓ Found [-1, 0, 1]
```

### Solution
```python
def threeSum(nums: list[int]) -> list[list[int]]:
    """
    Find all unique triplets that sum to zero.

    Strategy:
    1. Sort the array (enables two-pointer technique)
    2. Fix first number, use two pointers for remaining two
    3. Skip duplicates at all levels to avoid duplicate triplets

    Time: O(n²) - O(n log n) sort + O(n²) for nested loops
    Space: O(1) or O(n) depending on sort implementation
    """
    nums.sort()  # Sort to enable two-pointer technique
    result = []
    n = len(nums)

    for i in range(n - 2):  # Leave room for at least 2 more elements
        # ===== OPTIMIZATION: Early termination =====
        # If smallest number > 0, no triplet can sum to 0
        if nums[i] > 0:
            break

        # ===== SKIP DUPLICATES for first number =====
        # If same as previous, skip to avoid duplicate triplets
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # ===== TWO POINTERS for remaining two numbers =====
        # Find pairs in nums[i+1:] that sum to -nums[i]
        target = -nums[i]
        left = i + 1
        right = n - 1

        while left < right:
            current_sum = nums[left] + nums[right]

            if current_sum == target:
                # Found a valid triplet!
                result.append([nums[i], nums[left], nums[right]])

                # Move both pointers and skip duplicates
                left += 1
                right -= 1

                # Skip duplicate values for left pointer
                while left < right and nums[left] == nums[left - 1]:
                    left += 1

                # Skip duplicate values for right pointer
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1

            elif current_sum < target:
                # Sum too small - need larger values
                left += 1
            else:
                # Sum too large - need smaller values
                right -= 1

    return result
```

### Complexity
- **Time**: O(n²)
- **Space**: O(1) extra (excluding output)

### Edge Cases
- All zeros [0,0,0] → [[0,0,0]]
- No valid triplets → []
- All same number (non-zero) → []
- Many duplicates → skip logic prevents duplicate triplets
- Already sorted array

---

## Problem 2: Container With Most Water (LC #11) - Medium

- [LeetCode](https://leetcode.com/problems/container-with-most-water/)

### Problem Statement
Given array `height` where `height[i]` is height of line at position `i`, find two lines that form a container holding the most water.

### Examples
```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49

Visual:
    |         |
    |         |   |
    | |       |   |
    | |   |   |   |
    | |   | | |   |
    | |   | | | | |
    | | | | | | | |
  | | | | | | | | |
  1 8 6 2 5 4 8 3 7
    ^             ^
    L             R   Area = min(8,7) × 7 = 49
```

### Video Explanation
- [NeetCode - Container With Most Water](https://www.youtube.com/watch?v=UuiTKBwPgAo)

### Intuition Development
```
Area = min(height[L], height[R]) × (R - L)

Key insight: Why move the SHORTER line?
- Area is limited by the shorter line
- Moving the taller line can only DECREASE width, can't increase height limit
- Moving the shorter line MIGHT find a taller line

height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
          L                       R   area = min(1,7) × 8 = 8
             L                    R   area = min(8,7) × 7 = 49 ← max so far
             L                 R      area = min(8,3) × 6 = 18
             L              R         area = min(8,8) × 5 = 40
             ...
```

### Solution
```python
def maxArea(height: list[int]) -> int:
    """
    Find maximum water container area.

    Strategy:
    - Two pointers at opposite ends
    - Calculate area at each step
    - Move the pointer at the SHORTER line

    Why move the shorter line?
    - Area = min(left_height, right_height) × width
    - Moving taller line: width decreases, height can't increase beyond shorter
    - Moving shorter line: width decreases, but height MIGHT increase
    - So moving shorter line is the only way to potentially find larger area

    Time: O(n) - each element visited at most once
    Space: O(1) - only two pointers
    """
    left = 0
    right = len(height) - 1
    max_area = 0

    while left < right:
        # Calculate current area
        # Width = distance between lines
        # Height = limited by shorter line
        width = right - left
        current_height = min(height[left], height[right])
        current_area = width * current_height

        # Update maximum area
        max_area = max(max_area, current_area)

        # Move the pointer at the shorter line
        # This is the only way to potentially find a larger area
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- Two elements → single container
- All same height → any pair works
- Decreasing heights → first and last pair
- Single tall line among short ones

---

## Problem 3: 4Sum (LC #18) - Medium

- [LeetCode](https://leetcode.com/problems/4sum/)

### Problem Statement
Find all unique quadruplets that sum to target.

### Video Explanation
- [NeetCode - 4Sum](https://www.youtube.com/watch?v=EYeR-_1NRlQ)

### Examples
```
Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
```

### Intuition Development
```
Extension of 3Sum:
- 3Sum: Fix 1 number, two pointers for remaining 2
- 4Sum: Fix 2 numbers, two pointers for remaining 2

Time: O(n³)
```

### Solution
```python
def fourSum(nums: list[int], target: int) -> list[list[int]]:
    """
    Find all unique quadruplets that sum to target.

    Strategy:
    - Sort array
    - Fix first two numbers with nested loops
    - Use two pointers for remaining two numbers
    - Skip duplicates at all levels

    Generalizes to kSum by recursion (see below).

    Time: O(n³) - two nested loops + two pointers
    Space: O(1) extra (excluding output)
    """
    nums.sort()
    result = []
    n = len(nums)

    for i in range(n - 3):  # First number
        # Skip duplicates for first number
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # Early termination: if smallest 4 numbers > target
        if nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target:
            break

        # Skip if current number with 3 largest can't reach target
        if nums[i] + nums[n-1] + nums[n-2] + nums[n-3] < target:
            continue

        for j in range(i + 1, n - 2):  # Second number
            # Skip duplicates for second number
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue

            # Early termination for inner loop
            if nums[i] + nums[j] + nums[j+1] + nums[j+2] > target:
                break

            if nums[i] + nums[j] + nums[n-1] + nums[n-2] < target:
                continue

            # Two pointers for remaining two numbers
            left = j + 1
            right = n - 1
            remaining_target = target - nums[i] - nums[j]

            while left < right:
                current_sum = nums[left] + nums[right]

                if current_sum == remaining_target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])

                    # Move pointers and skip duplicates
                    left += 1
                    right -= 1
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1

                elif current_sum < remaining_target:
                    left += 1
                else:
                    right -= 1

    return result


def kSum(nums: list[int], target: int, k: int) -> list[list[int]]:
    """
    Generalized k-Sum solution using recursion.

    Base case: 2Sum with two pointers
    Recursive case: Fix one number, solve (k-1)Sum

    Time: O(n^(k-1))
    Space: O(k) for recursion
    """
    nums.sort()
    result = []

    def helper(start: int, k: int, target: int, path: list):
        # Base case: 2Sum
        if k == 2:
            left, right = start, len(nums) - 1
            while left < right:
                current_sum = nums[left] + nums[right]
                if current_sum == target:
                    result.append(path + [nums[left], nums[right]])
                    left += 1
                    right -= 1
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
            return

        # Recursive case: fix one number, solve (k-1)Sum
        for i in range(start, len(nums) - k + 1):
            # Skip duplicates
            if i > start and nums[i] == nums[i - 1]:
                continue
            # Early termination
            if nums[i] * k > target:
                break
            if nums[i] + nums[-1] * (k - 1) < target:
                continue

            helper(i + 1, k - 1, target - nums[i], path + [nums[i]])

    helper(0, k, target, [])
    return result
```

### Complexity
- **Time**: O(n³) for 4Sum, O(n^(k-1)) for kSum
- **Space**: O(k) for recursion

### Edge Cases
- Less than 4 elements → []
- Target requires negative numbers
- Overflow when summing large numbers (use long in some languages)
- All same numbers summing to target
- Empty result (no valid quadruplets)

---

## Problem 4: Sort Colors (LC #75) - Medium (Dutch National Flag)

- [LeetCode](https://leetcode.com/problems/sort-colors/)

### Problem Statement
Sort array containing only 0, 1, and 2 in-place. One-pass, constant space.

### Video Explanation
- [NeetCode - Sort Colors](https://www.youtube.com/watch?v=4xbWSRZHqac)

### Examples
```
Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
```

### Intuition Development
```
Dutch National Flag algorithm:
- Three pointers: low, mid, high
- [0, low): all 0s
- [low, mid): all 1s
- [mid, high]: unknown
- (high, end]: all 2s

nums = [2, 0, 2, 1, 1, 0]
        L  M           H

mid=2: swap with high, high--
nums = [0, 0, 2, 1, 1, 2]
        L  M        H

mid=0: swap with low, low++, mid++
nums = [0, 0, 2, 1, 1, 2]
           L  M     H

mid=2: swap with high, high--
nums = [0, 0, 1, 1, 2, 2]
           L  M  H

mid=1: just mid++
nums = [0, 0, 1, 1, 2, 2]
           L     M
                 H
mid > high: done!
```

### Solution
```python
def sortColors(nums: list[int]) -> None:
    """
    Sort array of 0s, 1s, and 2s in-place (Dutch National Flag).

    Strategy (Three Pointers):
    - low: boundary for 0s (everything before low is 0)
    - mid: current element being examined
    - high: boundary for 2s (everything after high is 2)

    Invariants:
    - [0, low): contains all 0s
    - [low, mid): contains all 1s
    - [mid, high]: unprocessed elements
    - (high, n): contains all 2s

    Actions:
    - If nums[mid] == 0: swap with low, increment both low and mid
    - If nums[mid] == 1: just increment mid (1 is in correct region)
    - If nums[mid] == 2: swap with high, decrement high (don't increment mid!)

    Why not increment mid when swapping with high?
    - The swapped element from high is unprocessed, need to check it

    Time: O(n) - single pass
    Space: O(1) - in-place
    """
    low = 0           # Next position for 0
    mid = 0           # Current element
    high = len(nums) - 1  # Next position for 2

    while mid <= high:
        if nums[mid] == 0:
            # Swap 0 to the low region
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1  # Safe to move mid because swapped element was in [low, mid)

        elif nums[mid] == 1:
            # 1 is already in correct region, just move forward
            mid += 1

        else:  # nums[mid] == 2
            # Swap 2 to the high region
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
            # DON'T increment mid! Swapped element needs to be checked
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- All same color → already sorted
- All 0s, all 1s, or all 2s
- Already sorted
- Reverse sorted
- Single element or two elements

---

## Problem 5: Trapping Rain Water (LC #42) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/trapping-rain-water/)

### Problem Statement
Given `n` non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

### Video Explanation
- [NeetCode - Trapping Rain Water](https://www.youtube.com/watch?v=ZI2z5pq0TqA)

### Examples
```
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: 6 units of rain water (shown as ░) are trapped.

Input: height = [4,2,0,3,2,5]
Output: 9
```

### Intuition Development
```
Water at position i = min(max_left, max_right) - height[i]

Visual: height = [0,1,0,2,1,0,1,3,2,1,2,1]

                    █
            █ ░ ░ ░ █ █ ░ █
        █ ░ █ █ ░ █ █ █ █ █ █
      0 1 0 2 1 0 1 3 2 1 2 1

┌─────────────────────────────────────────────────────────────────┐
│ Two Pointer Insight:                                            │
│                                                                  │
│ Key: Water is bounded by the SMALLER of left_max and right_max  │
│                                                                  │
│ If left_max < right_max:                                        │
│   - Water at left position is bounded by left_max               │
│   - We don't need exact right_max, just that it's higher!       │
│   - Process left side                                           │
│                                                                  │
│ If right_max <= left_max:                                       │
│   - Water at right position is bounded by right_max             │
│   - Process right side                                          │
│                                                                  │
│ Example: height = [4, 2, 0, 3, 2, 5]                            │
│   L=0, R=5: h[0]=4 < h[5]=5, left_max=4, water+=0, L=1         │
│   L=1, R=5: h[1]=2 < h[5]=5, water+=4-2=2, L=2                 │
│   L=2, R=5: h[2]=0 < h[5]=5, water+=4-0=4, L=3                 │
│   L=3, R=5: h[3]=3 < h[5]=5, water+=4-3=1, L=4                 │
│   L=4, R=5: h[4]=2 < h[5]=5, water+=4-2=2, L=5                 │
│   Total: 2+4+1+2 = 9 ✓                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def trap(height: list[int]) -> int:
    """
    Calculate trapped rain water using two pointers.

    Strategy:
    - Water at position i = min(max_left, max_right) - height[i]
    - Use two pointers to track max from both sides
    - Process from the side with smaller max (that's the limiting factor)

    Key insight:
    - Water level at any position is limited by the SMALLER of max heights
    - If left_max < right_max, we know water level at left is determined by left_max
    - We don't need to know exact right_max, just that it's >= left_max

    Time: O(n) - single pass with two pointers
    Space: O(1) - only tracking two max values
    """
    if not height:
        return 0

    left = 0
    right = len(height) - 1
    left_max = 0   # Max height seen from left
    right_max = 0  # Max height seen from right
    water = 0

    while left < right:
        # Process from the side with smaller max
        if height[left] < height[right]:
            # Left side is limiting factor
            if height[left] >= left_max:
                # Current bar is new max - no water trapped here
                left_max = height[left]
            else:
                # Water can be trapped: left_max - current height
                water += left_max - height[left]
            left += 1
        else:
            # Right side is limiting factor
            if height[right] >= right_max:
                # Current bar is new max
                right_max = height[right]
            else:
                # Water can be trapped
                water += right_max - height[right]
            right -= 1

    return water
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- Empty array or single element → 0
- Monotonically increasing/decreasing → 0
- All same height → 0
- Two tall bars at ends with low bars between
- Negative heights (not typically in problem but worth considering)

---

## Summary: Medium Problems Checklist

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | 3Sum | Sort + fix one + two pointers | O(n²) | O(1) |
| 2 | Container With Most Water | Move shorter line | O(n) | O(1) |
| 3 | 4Sum | Fix two + two pointers | O(n³) | O(1) |
| 4 | Sort Colors | Dutch National Flag | O(n) | O(1) |
| 5 | Trapping Rain Water | Two pointers from ends | O(n) | O(1) |

---

## Practice More Medium Problems

- [ ] LC #16 - 3Sum Closest
- [ ] LC #259 - 3Sum Smaller
- [ ] LC #713 - Subarray Product Less Than K
- [ ] LC #986 - Interval List Intersections
- [ ] LC #524 - Longest Word in Dictionary through Deleting

