# Two Pointers - Hard Problems

## Problem 1: Trapping Rain Water (LC #42) - Hard

- [LeetCode](https://leetcode.com/problems/trapping-rain-water/)

### Video Explanation
- [NeetCode - Trapping Rain Water](https://www.youtube.com/watch?v=ZI2z5pq0TqA)

### Problem Statement
Calculate water trapped between bars after raining.

### Examples
```
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

Visual:
       █
   █   ██ █
 █ ██ ██████
```


### Visual Intuition
```
Trapping Rain Water - Two Pointer Approach
height = [0,1,0,2,1,0,1,3,2,1,2,1]

Water at each position = min(maxLeft, maxRight) - height

     3|              █
     2|        █ ≈ ≈ █ █ ≈ █
     1|  █ ≈ █ █ █ ≈ █ █ █ █ █
     0|█ █ █ █ █ █ █ █ █ █ █ █
       0 1 2 3 4 5 6 7 8 9 ...

≈ = trapped water

Two pointers L=0, R=11, track maxL and maxR:
- If maxL < maxR: water[L] = maxL - height[L], move L→
- Else: water[R] = maxR - height[R], move ←R

Total water = 6 units
```

### Solution
```python
def trap(height: list[int]) -> int:
    """
    Calculate trapped rain water using two pointers.

    Key insight: Water at position i = min(max_left, max_right) - height[i]

    Strategy:
    - Two pointers from both ends
    - Track max height seen from each side
    - Process smaller side (guaranteed water level)

    Time: O(n)
    Space: O(1)
    """
    if not height:
        return 0

    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    water = 0

    while left < right:
        if height[left] < height[right]:
            # Process left side
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            # Process right side
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1

    return water


def trap_dp(height: list[int]) -> int:
    """
    Alternative: DP approach with precomputed max arrays.

    Time: O(n)
    Space: O(n)
    """
    if not height:
        return 0

    n = len(height)

    # Precompute max from left
    left_max = [0] * n
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], height[i])

    # Precompute max from right
    right_max = [0] * n
    right_max[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], height[i])

    # Calculate water
    water = 0
    for i in range(n):
        water += min(left_max[i], right_max[i]) - height[i]

    return water
```

### Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  height = [0,1,0,2,1,0,1,3,2,1,2,1]                                         │
│                                                                             │
│  Index:    0 1 2 3 4 5 6 7 8 9 10 11                                        │
│                                                                             │
│                        █                                                    │
│            █  ░░░░░░░░██░█                                                  │
│     █  ░░░██░░██░░░░████████                                                │
│  ─────────────────────────────                                              │
│                                                                             │
│  ░ = water trapped                                                          │
│  █ = bars                                                                   │
│                                                                             │
│  At index 2: left_max=1, right_max=3, water = min(1,3) - 0 = 1              │
│  At index 5: left_max=2, right_max=3, water = min(2,3) - 0 = 2              │
│                                                                             │
│  Total water = 6                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Edge Cases
- Empty array → return 0
- All same height → return 0
- Single bar → return 0
- Monotonically increasing/decreasing → return 0
- Two bars → return 0 (no middle to trap)

---

## Problem 2: Container With Most Water (LC #11) - Medium

- [LeetCode](https://leetcode.com/problems/container-with-most-water/)

### Video Explanation
- [NeetCode - Container With Most Water](https://www.youtube.com/watch?v=UuiTKBwPgAo)

### Problem Statement
Find two lines that form container with most water.

### Examples
```
Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49 (between index 1 and 8)
```


### Visual Intuition
```
Container With Most Water
height = [1,8,6,2,5,4,8,3,7]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Area = min(height[L], height[R]) × (R - L)
             Move the SHORTER pointer (taller can't help)
═══════════════════════════════════════════════════════════════

Histogram:
──────────
  8 |   █           █
  7 |   █           █     █
  6 |   █   █       █     █
  5 |   █   █   █   █     █
  4 |   █   █   █ █ █     █
  3 |   █   █   █ █ █   █ █
  2 |   █   █ █ █ █ █   █ █
  1 | █ █   █ █ █ █ █   █ █
      0 1 2 3 4 5 6 7 8

Step-by-Step:
─────────────
Step 0: L=0, R=8
        height[0]=1, height[8]=7
        Area = min(1,7) × 8 = 8
        Move L (shorter) → L=1

Step 1: L=1, R=8
        height[1]=8, height[8]=7
        Area = min(8,7) × 7 = 49 ★ MAX!
        Move R (shorter) → R=7

Step 2: L=1, R=7
        height[1]=8, height[7]=3
        Area = min(8,3) × 6 = 18
        Move R → R=6

Step 3: L=1, R=6
        height[1]=8, height[6]=8
        Area = min(8,8) × 5 = 40
        Move either → L=2

[Continue until L >= R...]

Why Move Shorter Pointer?
─────────────────────────
  If height[L] < height[R]:
    - Current area limited by height[L]
    - Moving R can only DECREASE width
    - Moving R won't increase height (still limited by L)
    → Moving L might find taller bar, increasing area

Answer: 49 (L=1, R=8)

WHY THIS WORKS:
════════════════
● Start widest → maximum possible width
● Moving shorter pointer: might find taller bar
● Moving taller pointer: can only decrease area
● Greedy choice ensures we don't miss optimal
```

### Solution
```python
def maxArea(height: list[int]) -> int:
    """
    Find maximum water container using two pointers.

    Key insight: Area = min(height[left], height[right]) * (right - left)

    Strategy:
    - Start with widest container (left=0, right=n-1)
    - Move pointer with smaller height (can only increase area)

    Time: O(n)
    Space: O(1)
    """
    left, right = 0, len(height) - 1
    max_water = 0

    while left < right:
        # Calculate current area
        width = right - left
        h = min(height[left], height[right])
        max_water = max(max_water, width * h)

        # Move pointer with smaller height
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water
```

### Edge Cases
- Only 2 bars → calculate single area
- All same height → width determines area
- One very tall bar → limited by shorter partner
- Sorted array → start with widest, shrink optimally

---

## Problem 3: 4Sum (LC #18) - Medium

- [LeetCode](https://leetcode.com/problems/4sum/)

### Video Explanation
- [NeetCode - 4Sum](https://www.youtube.com/watch?v=EYeR-_1NRlQ)

### Problem Statement
Find all unique quadruplets that sum to target.

### Examples
```
Input: nums = [1,0,-1,0,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
```


### Visual Intuition
```
4Sum - Reduce to 3Sum to 2Sum
nums = [-2,-1,0,0,1,2], target = 0

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Fix 2 elements, use two pointers for remaining 2
             Skip duplicates at each level to avoid duplicates
═══════════════════════════════════════════════════════════════

Sorted: [-2, -1, 0, 0, 1, 2]
         0   1   2  3  4  5

Nested Structure:
─────────────────
for i in range(n-3):           # First element
  for j in range(i+1, n-2):    # Second element
    L, R = j+1, n-1            # Two pointers for 3rd & 4th

Trace:
──────
i=0 (val=-2):
  │
  ├─ j=1 (val=-1): need L+R = 0-(-2)-(-1) = 3
  │    [-2, -1, 0, 0, 1, 2]
  │           L        R
  │    L=0, R=2: 0+2=2 < 3 → L++
  │    L=0, R=1: 0+1=1 < 3 → L++
  │    L=1, R=1: L≥R, done. No match.
  │
  ├─ j=2 (val=0): need L+R = 0-(-2)-0 = 2
  │    [-2, -1, 0, 0, 1, 2]
  │              L     R
  │    L=0, R=2: 0+2=2 = 2 ✓ Found! [-2,0,0,2]
  │    Skip duplicates, L++, R--
  │    L=1, R=1: L≥R, done.
  │
  └─ j=3 (val=0): SKIP (duplicate of j=2)

i=1 (val=-1):
  │
  ├─ j=2 (val=0): need L+R = 0-(-1)-0 = 1
  │    [-2, -1, 0, 0, 1, 2]
  │                 L  R
  │    L=0, R=2: 0+2=2 > 1 → R--
  │    L=0, R=1: 0+1=1 = 1 ✓ Found! [-1,0,0,1]
  │
  └─ j=3 (val=0): SKIP (duplicate)

i=2 (val=0): need i+j+L+R = 0, but remaining sum needed is negative
             Early termination possible

Result: [[-2,-1,1,2], [-2,0,0,2], [-1,0,0,1]]

WHY THIS WORKS:
════════════════
● Sorting enables two-pointer technique
● Fix 2 elements → reduces to 2Sum problem
● Skip duplicates at each level to avoid duplicate quadruplets
● Early termination when min/max sum can't reach target
● Time: O(n³) - two nested loops + two pointers
```

### Solution
```python
def fourSum(nums: list[int], target: int) -> list[list[int]]:
    """
    Find all unique quadruplets summing to target.

    Strategy:
    - Sort array
    - Fix first two elements with nested loops
    - Use two pointers for remaining two
    - Skip duplicates at each level

    Time: O(n³)
    Space: O(1) excluding output
    """
    nums.sort()
    n = len(nums)
    result = []

    for i in range(n - 3):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # Early termination
        if nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target:
            break
        if nums[i] + nums[n - 3] + nums[n - 2] + nums[n - 1] < target:
            continue

        for j in range(i + 1, n - 2):
            # Skip duplicates for second element
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue

            # Early termination
            if nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target:
                break
            if nums[i] + nums[j] + nums[n - 2] + nums[n - 1] < target:
                continue

            # Two pointers for remaining two elements
            left, right = j + 1, n - 1

            while left < right:
                total = nums[i] + nums[j] + nums[left] + nums[right]

                if total == target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])

                    # Skip duplicates
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1

                    left += 1
                    right -= 1
                elif total < target:
                    left += 1
                else:
                    right -= 1

    return result
```

### Edge Cases
- Less than 4 elements → return []
- All same numbers → return [] or multiple duplicates
- No valid quadruplet → return []
- Negative numbers → target can be negative
- Integer overflow → use long or check bounds

---

## Problem 4: Minimum Window Sort (LC #581) - Medium

- [LeetCode](https://leetcode.com/problems/shortest-unsorted-continuous-subarray/)

### Video Explanation
- [NeetCode - Shortest Unsorted Continuous Subarray](https://www.youtube.com/watch?v=OpnLzHqHqQg)

### Problem Statement
Find shortest subarray to sort for entire array to be sorted.

### Examples
```
Input: nums = [2,6,4,8,10,9,15]
Output: 5 (sort [6,4,8,10,9])
```


### Visual Intuition
```
Minimum Window Sort
nums = [2, 6, 4, 8, 10, 9, 15]
        0  1  2  3  4   5  6

Find bounds of unsorted region:

Left→Right: Track max_seen, find rightmost < max
  max: 2  6  6  8  10  10  15
       ✓  ✓  ✗  ✓  ✓   ✗   ✓
                        ↑ right_bound = 5

Right←Left: Track min_seen, find leftmost > min
  min: 15 9  9  8  4   4   4
       ✓  ✓  ✓  ✓  ✗   ✗   ✗
              ↑ left_bound = 1

Answer: right - left + 1 = 5 - 1 + 1 = 5
Sort indices [1,5]: [6,4,8,10,9] → [4,6,8,9,10]
```

### Solution
```python
def findUnsortedSubarray(nums: list[int]) -> int:
    """
    Find minimum subarray to sort.

    Strategy:
    - Find leftmost element out of place (greater than something to its right)
    - Find rightmost element out of place (less than something to its left)

    Time: O(n)
    Space: O(1)
    """
    n = len(nums)

    # Find minimum and maximum in unsorted portion
    # Going left to right, find max so far
    # If current < max, it's out of place
    max_seen = float('-inf')
    right_bound = -1

    for i in range(n):
        if nums[i] < max_seen:
            right_bound = i
        else:
            max_seen = nums[i]

    # Going right to left, find min so far
    # If current > min, it's out of place
    min_seen = float('inf')
    left_bound = 0

    for i in range(n - 1, -1, -1):
        if nums[i] > min_seen:
            left_bound = i
        else:
            min_seen = nums[i]

    if right_bound == -1:
        return 0  # Already sorted

    return right_bound - left_bound + 1
```

### Edge Cases
- Already sorted → return 0
- Reverse sorted → return n
- Single element → return 0
- Two elements out of order → return 2
- Duplicates → handle carefully

---

## Problem 5: Subarrays with Product Less Than K (LC #713) - Medium

- [LeetCode](https://leetcode.com/problems/subarray-product-less-than-k/)

### Video Explanation
- [NeetCode - Subarray Product Less Than K](https://www.youtube.com/watch?v=SxtxCSfSGlo)

### Problem Statement
Count subarrays with product < k.

### Examples
```
Input: nums = [10,5,2,6], k = 100
Output: 8
```


### Visual Intuition
```
Subarrays with Product Less Than K
nums = [10, 5, 2, 6], k = 100

Sliding window tracking product:
  L  R  Window      Product  Valid subarrays ending at R
  0  0  [10]        10<100   [10] → count=1
  0  1  [10,5]      50<100   [5],[10,5] → count=2
  0  2  [10,5,2]    100≥100  shrink!
  1  2  [5,2]       10<100   [2],[5,2] → count=2
  1  3  [5,2,6]     60<100   [6],[2,6],[5,2,6] → count=3

Total = 1 + 2 + 2 + 3 = 8

Formula: For valid window [L,R], count += R - L + 1
(all subarrays ending at R with start ≥ L)
```

### Solution
```python
def numSubarrayProductLessThanK(nums: list[int], k: int) -> int:
    """
    Count subarrays with product < k using sliding window.

    Strategy:
    - Expand window, multiply product
    - Shrink when product >= k
    - Each valid window of size n contributes n subarrays

    Time: O(n)
    Space: O(1)
    """
    if k <= 1:
        return 0

    left = 0
    product = 1
    count = 0

    for right in range(len(nums)):
        product *= nums[right]

        # Shrink window while product >= k
        while product >= k:
            product //= nums[left]
            left += 1

        # All subarrays ending at right with start >= left are valid
        count += right - left + 1

    return count
```

### Edge Cases
- k <= 1 → return 0 (no valid subarrays)
- All elements >= k → return 0
- Single element < k → return 1
- All elements = 1 → entire array is valid

---

## Problem 6: Boats to Save People (LC #881) - Medium

- [LeetCode](https://leetcode.com/problems/boats-to-save-people/)

### Video Explanation
- [NeetCode - Boats to Save People](https://www.youtube.com/watch?v=XbaxWuHIWUs)

### Problem Statement
Minimum boats to carry all people (each boat max 2 people, weight limit).

### Examples
```
Input: people = [3,2,2,1], limit = 3
Output: 3
```


### Visual Intuition
```
Boats to Save People (max 2 per boat)
people = [3, 2, 2, 1], limit = 3

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Sort + greedy pairing (heaviest with lightest)
             Heaviest always takes a boat, lightest may join
═══════════════════════════════════════════════════════════════

Sorted: [1, 2, 2, 3]
         L        R

Step-by-Step:
─────────────
Step 1: Try to pair lightest (1) with heaviest (3)
        ┌─┐     ┌─┐
        │1│     │3│
        └─┘     └─┘
         L       R

        1 + 3 = 4 > limit (3) ✗
        Heaviest goes ALONE

        🚤 Boat 1: [3]
        R-- → R points to 2
        boats = 1

Step 2: Try to pair lightest (1) with heaviest (2)
        ┌─┐ ┌─┐
        │1│ │2│ 2
        └─┘ └─┘
         L     R

        1 + 2 = 3 ≤ limit (3) ✓
        They can PAIR!

        🚤 Boat 2: [1, 2]
        L++, R-- → L=R=middle 2
        boats = 2

Step 3: L == R (same person)
        ┌─┐
        │2│
        └─┘
        L=R

        One person left, goes alone

        🚤 Boat 3: [2]
        boats = 3

Final Answer: 3 boats

Visual Summary:
───────────────
  🚤 Boat 1: [3]     (alone, too heavy to pair)
  🚤 Boat 2: [1, 2]  (paired, 1+2=3 ≤ limit)
  🚤 Boat 3: [2]     (alone, last person)

WHY THIS WORKS:
════════════════
● Heaviest person MUST take a boat (no choice)
● Pairing with lightest maximizes chance of fitting
● If lightest can't fit with heaviest, no one can
● Greedy optimal: each boat carries max possible weight
```

### Solution
```python
def numRescueBoats(people: list[int], limit: int) -> int:
    """
    Minimum boats using greedy two pointers.

    Strategy:
    - Sort by weight
    - Try to pair heaviest with lightest
    - If can't pair, heaviest goes alone

    Time: O(n log n)
    Space: O(1)
    """
    people.sort()
    left, right = 0, len(people) - 1
    boats = 0

    while left <= right:
        # Heaviest person always takes a boat
        if left == right:
            boats += 1
            break

        # Try to pair with lightest
        if people[left] + people[right] <= limit:
            left += 1  # Lightest can join

        right -= 1  # Heaviest takes boat
        boats += 1

    return boats
```

### Edge Cases
- Single person → 1 boat
- All same weight → pair if 2*weight <= limit
- Heaviest > limit → impossible (guaranteed weight <= limit)
- All can pair → n/2 boats

---

## Problem 7: 3Sum Closest (LC #16) - Medium

- [LeetCode](https://leetcode.com/problems/3sum-closest/)

### Video Explanation
- [NeetCode - 3Sum Closest](https://www.youtube.com/watch?v=BoHO04xVeU0)

### Problem Statement
Find triplet with sum closest to target.

### Examples
```
Input: nums = [-1,2,1,-4], target = 1
Output: 2 ([-1,2,1])
```


### Visual Intuition
```
3Sum Closest to Target
nums = [-1, 2, 1, -4], target = 1

Sort: [-4, -1, 1, 2]
        i   L     R

Fix i=-4: L=-1, R=2
  sum = -4 + (-1) + 2 = -3, diff=|1-(-3)|=4
  sum < target, move L→
  sum = -4 + 1 + 2 = -1, diff=|1-(-1)|=2
  sum < target, move L→ (L crosses R, done)

Fix i=-1: L=1, R=2
  sum = -1 + 1 + 2 = 2, diff=|1-2|=1 ← closest!
  sum > target, move ←R (done)

Answer: 2 (closest to target 1)
```

### Solution
```python
def threeSumClosest(nums: list[int], target: int) -> int:
    """
    Find triplet sum closest to target.

    Strategy:
    - Sort array
    - Fix first element, use two pointers for rest
    - Track closest sum

    Time: O(n²)
    Space: O(1)
    """
    nums.sort()
    n = len(nums)
    closest = float('inf')

    for i in range(n - 2):
        left, right = i + 1, n - 1

        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]

            # Update closest if better
            if abs(current_sum - target) < abs(closest - target):
                closest = current_sum

            if current_sum == target:
                return target
            elif current_sum < target:
                left += 1
            else:
                right -= 1

    return closest
```

### Edge Cases
- Exactly 3 elements → return their sum
- Exact match exists → return target
- All same numbers → return 3*num
- Large negative/positive numbers → watch for overflow

---

## Problem 8: Sort Colors (LC #75) - Medium

- [LeetCode](https://leetcode.com/problems/sort-colors/)

### Video Explanation
- [NeetCode - Sort Colors](https://www.youtube.com/watch?v=4xbWSRZHqac)

### Problem Statement
Sort array with values 0, 1, 2 in-place (Dutch National Flag).

### Examples
```
Input: nums = [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
```


### Visual Intuition
```
Sort Colors (Dutch National Flag)
nums = [2, 0, 2, 1, 1, 0]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Three regions maintained by three pointers
             [0...low-1]=0s, [low...mid-1]=1s, [high+1...n-1]=2s
═══════════════════════════════════════════════════════════════

Region Invariants:
──────────────────
  [0, low)    → all 0s (red)
  [low, mid)  → all 1s (white)
  [mid, high] → unprocessed
  (high, n)   → all 2s (blue)

Step-by-Step:
─────────────
Initial: low=0, mid=0, high=5
  [2, 0, 2, 1, 1, 0]
   ↑              ↑
  mid            high

Step 1: nums[mid]=2 → swap with high, high--
  [0, 0, 2, 1, 1, 2]
   ↑           ↑
  mid         high
  (Don't move mid - need to check swapped value)

Step 2: nums[mid]=0 → swap with low, low++, mid++
  [0, 0, 2, 1, 1, 2]
      ↑        ↑
     low      high
      mid

Step 3: nums[mid]=0 → swap with low, low++, mid++
  [0, 0, 2, 1, 1, 2]
         ↑     ↑
        low   high
        mid

Step 4: nums[mid]=2 → swap with high, high--
  [0, 0, 1, 1, 2, 2]
         ↑  ↑
        low high
        mid

Step 5: nums[mid]=1 → correct region, mid++
  [0, 0, 1, 1, 2, 2]
         ↑  ↑
        low mid
            high

Step 6: nums[mid]=1 → correct region, mid++
  [0, 0, 1, 1, 2, 2]
         ↑     ↑
        low   mid
            high

Step 7: mid > high → DONE!

Final: [0, 0, 1, 1, 2, 2]
        └─0s─┘└─1s─┘└─2s─┘

WHY THIS WORKS:
════════════════
● Three pointers partition array into 4 regions
● mid scans through unprocessed elements
● 0 → swap to front, 2 → swap to back, 1 → leave in middle
● Single pass O(n), in-place O(1) space
```

### Solution
```python
def sortColors(nums: list[int]) -> None:
    """
    Dutch National Flag algorithm.

    Strategy:
    - Three pointers: low, mid, high
    - 0s go to [0, low), 1s in [low, high], 2s in (high, n)
    - Process mid pointer

    Time: O(n)
    Space: O(1)
    """
    low, mid, high = 0, 0, len(nums) - 1

    while mid <= high:
        if nums[mid] == 0:
            # Swap with low, move both forward
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            # 1 is in correct region, move mid
            mid += 1
        else:  # nums[mid] == 2
            # Swap with high, only move high back
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
            # Don't increment mid - need to check swapped value
```

### Edge Cases
- All same color → already sorted
- Only two colors → simpler partition
- Single element → no change needed
- Already sorted → still works correctly

---

## Problem 9: Partition Labels (LC #763) - Medium

- [LeetCode](https://leetcode.com/problems/partition-labels/)

### Video Explanation
- [NeetCode - Partition Labels](https://www.youtube.com/watch?v=B7m8UmZE-vw)

### Problem Statement
Partition string so each letter appears in at most one part.

### Examples
```
Input: s = "ababcbacadefegdehijhklij"
Output: [9,7,8]
```


### Visual Intuition
```
Partition Labels
s = "ababcbacadefegdehijhklij"

Step 1: Find last occurrence of each char
  a:8, b:5, c:7, d:14, e:15, f:11, g:13, h:19, i:22, j:23, k:20, l:21

Step 2: Extend partition to include all occurrences

  a b a b c b a c a | d e f e g d e | h i j h k l i j
  0 1 2 3 4 5 6 7 8   9 ...     15    16 ...       23

  Start=0, see 'a' → must extend to 8
  See 'b' at 1 → must extend to 5 (already covered)
  See 'c' at 4 → must extend to 7 (already covered)
  Reach 8 = end → partition! size = 9

Result: [9, 7, 8]
```

### Solution
```python
def partitionLabels(s: str) -> list[int]:
    """
    Partition string into maximum parts.

    Strategy:
    - Find last occurrence of each character
    - Extend partition to include all occurrences of seen characters

    Time: O(n)
    Space: O(26) = O(1)
    """
    # Find last occurrence of each character
    last_occurrence = {c: i for i, c in enumerate(s)}

    result = []
    start = 0
    end = 0

    for i, c in enumerate(s):
        # Extend partition to include all occurrences of c
        end = max(end, last_occurrence[c])

        # If we've reached the end of current partition
        if i == end:
            result.append(end - start + 1)
            start = i + 1

    return result
```

### Edge Cases
- Single character → return [1]
- All same character → return [n]
- All unique characters → return [1,1,1,...]
- Two characters alternating → depends on last occurrence

---

## Problem 10: Merge Sorted Array (LC #88) - Easy

- [LeetCode](https://leetcode.com/problems/merge-sorted-array/)

### Video Explanation
- [NeetCode - Merge Sorted Array](https://www.youtube.com/watch?v=P1Ic85RarKY)

### Problem Statement
Merge nums2 into nums1 (nums1 has extra space).

### Examples
```
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]
```


### Visual Intuition
```
Merge Sorted Array (merge from end)
nums1 = [1, 2, 3, 0, 0, 0], m=3
nums2 = [2, 5, 6], n=3

Start from end to avoid overwriting:
  p1=2, p2=2, p=5

  [1, 2, 3, _, _, _]  nums2=[2, 5, 6]
              p1    p         p2

  3 vs 6: 6 wins → [1,2,3,_,_,6], p2--, p--
  3 vs 5: 5 wins → [1,2,3,_,5,6], p2--, p--
  3 vs 2: 3 wins → [1,2,3,3,5,6], p1--, p--
  2 vs 2: 2 wins → [1,2,2,3,5,6], p2--, p--
  p2 < 0: done!

Result: [1, 2, 2, 3, 5, 6]
```

### Solution
```python
def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    """
    Merge sorted arrays in-place.

    Strategy:
    - Start from end to avoid overwriting
    - Place larger element at current position

    Time: O(m + n)
    Space: O(1)
    """
    # Pointers for nums1, nums2, and merged array
    p1 = m - 1
    p2 = n - 1
    p = m + n - 1

    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1

    # Copy remaining elements from nums2
    while p2 >= 0:
        nums1[p] = nums2[p2]
        p2 -= 1
        p -= 1
```

### Edge Cases
- nums2 empty (n=0) → no change needed
- nums1 empty (m=0) → copy all of nums2
- All nums2 smaller → insert at beginning
- All nums2 larger → insert at end

---

## Summary: Two Pointers Hard Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Trapping Rain Water | Left/right max tracking | O(n) |
| 2 | Container Most Water | Move smaller height | O(n) |
| 3 | 4Sum | Nested loops + two pointers | O(n³) |
| 4 | Min Window Sort | Find out-of-place bounds | O(n) |
| 5 | Product < K | Sliding window | O(n) |
| 6 | Boats to Save | Greedy pairing | O(n log n) |
| 7 | 3Sum Closest | Track closest sum | O(n²) |
| 8 | Sort Colors | Dutch National Flag | O(n) |
| 9 | Partition Labels | Last occurrence tracking | O(n) |
| 10 | Merge Sorted Array | Merge from end | O(m+n) |

---

## Practice More Problems

- [ ] LC #259 - 3Sum Smaller
- [ ] LC #360 - Sort Transformed Array
- [ ] LC #457 - Circular Array Loop
- [ ] LC #838 - Push Dominoes
- [ ] LC #844 - Backspace String Compare

