# Two Pointers - Easy Problems

## Problem 1: Valid Palindrome (LC #125) - Easy

- [LeetCode](https://leetcode.com/problems/valid-palindrome/)

### Problem Statement
Given a string `s`, return `true` if it is a palindrome considering only alphanumeric characters and ignoring case.

### Video Explanation
- [NeetCode - Valid Palindrome](https://www.youtube.com/watch?v=jJXJ16kPFWg)
- [Take U Forward - Two Pointers](https://www.youtube.com/watch?v=BHr381Guz3Y)

### Examples
```
Input: s = "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome

Input: s = "race a car"
Output: false

Input: s = " "
Output: true (empty string is palindrome)
```

### Intuition Development
```
Palindrome: reads same forwards and backwards

Strategy: Two pointers from opposite ends
- Skip non-alphanumeric characters
- Compare characters (case-insensitive)
- If mismatch found → not palindrome
- If pointers meet/cross → palindrome

"A man, a plan, a canal: Panama"
 L                            R
 ↓                            ↓
 A                            a  → Match! (case-insensitive)
   L                        R
   ↓                        ↓
   m                        m    → Match!
   ... continue until L >= R
```

### Solution
```python
def isPalindrome(s: str) -> bool:
    """
    Check if string is a valid palindrome (alphanumeric only, case-insensitive).

    Strategy:
    - Use two pointers from opposite ends
    - Skip non-alphanumeric characters
    - Compare characters in lowercase

    Time: O(n) - each character visited at most once
    Space: O(1) - only two pointers, no extra data structures
    """
    left = 0
    right = len(s) - 1

    while left < right:
        # Skip non-alphanumeric characters from left
        # isalnum() returns True for letters and digits
        while left < right and not s[left].isalnum():
            left += 1

        # Skip non-alphanumeric characters from right
        while left < right and not s[right].isalnum():
            right -= 1

        # Compare characters (case-insensitive)
        if s[left].lower() != s[right].lower():
            return False  # Mismatch found - not a palindrome

        # Move both pointers inward
        left += 1
        right -= 1

    # All characters matched - it's a palindrome
    return True


def isPalindrome_clean(s: str) -> bool:
    """
    Alternative: Clean string first, then check.

    Simpler logic but uses O(n) extra space.
    """
    # Remove non-alphanumeric and convert to lowercase
    cleaned = ''.join(char.lower() for char in s if char.isalnum())

    # Check if cleaned string equals its reverse
    return cleaned == cleaned[::-1]
```

### Complexity
- **Time**: O(n) - Single pass through string
- **Space**: O(1) for two-pointer approach, O(n) for clean approach

### Edge Cases
- Empty string: `" "` → `True` (empty after cleaning)
- Single character: `"a"` → `True`
- All non-alphanumeric: `",.!?"` → `True` (empty after cleaning)
- Mixed case: `"Aa"` → `True` (case-insensitive)

---

## Problem 2: Two Sum II - Sorted Array (LC #167) - Easy

- [LeetCode](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

### Problem Statement
Given a 1-indexed sorted array `numbers`, find two numbers that add up to `target`. Return their indices (1-indexed).

### Video Explanation
- [NeetCode - Two Sum II](https://www.youtube.com/watch?v=cQ1Oz4ckceM)

### Examples
```
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: numbers[1] + numbers[2] = 2 + 7 = 9

Input: numbers = [2,3,4], target = 6
Output: [1,3]

Input: numbers = [-1,0], target = -1
Output: [1,2]
```

### Intuition Development
```
Key insight: Array is SORTED!
- If sum too small → need larger numbers → move left pointer right
- If sum too large → need smaller numbers → move right pointer left

numbers = [2, 7, 11, 15], target = 9
           L          R    sum = 2 + 15 = 17 > 9, move R left
           L      R        sum = 2 + 11 = 13 > 9, move R left
           L  R            sum = 2 + 7 = 9 ✓ Found!

Return [1, 2] (1-indexed)
```

### Solution
```python
def twoSum(numbers: list[int], target: int) -> list[int]:
    """
    Find two numbers in sorted array that sum to target.

    Strategy:
    - Two pointers at opposite ends
    - If sum < target: need larger, move left pointer right
    - If sum > target: need smaller, move right pointer left
    - If sum == target: found the answer

    Why this works:
    - Sorted array means moving left pointer right INCREASES sum
    - Moving right pointer left DECREASES sum
    - We systematically eliminate impossible pairs

    Time: O(n) - each element visited at most once
    Space: O(1) - only two pointers
    """
    left = 0
    right = len(numbers) - 1

    while left < right:
        current_sum = numbers[left] + numbers[right]

        if current_sum == target:
            # Found the pair! Return 1-indexed positions
            return [left + 1, right + 1]
        elif current_sum < target:
            # Sum too small - need larger numbers
            # Move left pointer right (to larger values)
            left += 1
        else:
            # Sum too large - need smaller numbers
            # Move right pointer left (to smaller values)
            right -= 1

    # Problem guarantees a solution exists, so we won't reach here
    return []
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- Two elements: `[2,7], target=9` → `[1,2]`
- Negative numbers: `[-1,0], target=-1` → `[1,2]`
- Same elements: `[3,3], target=6` → `[1,2]`
- Target at extremes: First or last pair

---

## Problem 3: Merge Sorted Array (LC #88) - Easy

- [LeetCode](https://leetcode.com/problems/merge-sorted-array/)

### Problem Statement
Merge `nums2` into `nums1` as one sorted array. `nums1` has enough space (size m + n) to hold elements from both arrays.

### Video Explanation
- [NeetCode - Merge Sorted Array](https://www.youtube.com/watch?v=P1Ic85RarKY)

### Examples
```
Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
Output: [1,2,2,3,5,6]

Input: nums1 = [1], m = 1, nums2 = [], n = 0
Output: [1]
```

### Intuition Development
```
Key insight: Merge from the END to avoid overwriting!

If we merge from start, we'd overwrite nums1's elements.
Merge from end - fill the largest elements first.

nums1 = [1, 2, 3, 0, 0, 0], nums2 = [2, 5, 6]
                  ↑                       ↑
                 p1=2                    p2=2
                              write position = 5

Compare nums1[p1] vs nums2[p2]: 3 vs 6 → 6 is larger
nums1 = [1, 2, 3, 0, 0, 6], p2--, write--

Compare: 3 vs 5 → 5 is larger
nums1 = [1, 2, 3, 0, 5, 6], p2--, write--

Compare: 3 vs 2 → 3 is larger
nums1 = [1, 2, 3, 3, 5, 6], p1--, write--

Compare: 2 vs 2 → equal, take either
nums1 = [1, 2, 2, 3, 5, 6], p1--, write--

... and so on
```

### Solution
```python
def merge(nums1: list[int], m: int, nums2: list[int], n: int) -> None:
    """
    Merge nums2 into nums1 in-place (nums1 has space for both).

    Strategy:
    - Merge from the END to avoid overwriting nums1's elements
    - Use three pointers: p1 (end of nums1 data), p2 (end of nums2), write (end of nums1)
    - Compare and place larger element at write position

    Why merge from end?
    - nums1 has extra space at the end (zeros)
    - Filling from end means we never overwrite unprocessed elements

    Time: O(m + n) - process each element once
    Space: O(1) - in-place modification
    """
    # Pointers for the last elements of each array's data
    p1 = m - 1      # Last element of nums1's actual data
    p2 = n - 1      # Last element of nums2
    write = m + n - 1  # Position to write next element (end of nums1)

    # Merge from end to beginning
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            # nums1's element is larger - place it at write position
            nums1[write] = nums1[p1]
            p1 -= 1
        else:
            # nums2's element is larger or equal - place it
            nums1[write] = nums2[p2]
            p2 -= 1
        write -= 1

    # If nums2 has remaining elements, copy them
    # (If nums1 has remaining elements, they're already in place)
    while p2 >= 0:
        nums1[write] = nums2[p2]
        p2 -= 1
        write -= 1

    # Note: No return needed - nums1 is modified in-place
```

### Complexity
- **Time**: O(m + n)
- **Space**: O(1)

### Edge Cases
- nums2 empty: `nums1=[1], m=1, nums2=[], n=0` → `[1]`
- nums1 empty: `nums1=[0,0], m=0, nums2=[1,2], n=2` → `[1,2]`
- No overlap: All nums1 < all nums2 or vice versa
- Interleaved: Elements alternate between arrays

---

## Problem 4: Move Zeroes (LC #283) - Easy

- [LeetCode](https://leetcode.com/problems/move-zeroes/)

### Problem Statement
Move all zeros to the end of array while maintaining relative order of non-zero elements. Must be done in-place.

### Video Explanation
- [NeetCode - Move Zeroes](https://www.youtube.com/watch?v=aayNRwUN3Do)

### Examples
```
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]

Input: nums = [0]
Output: [0]
```

### Intuition Development
```
Two pointers - same direction:
- slow: position for next non-zero element
- fast: scans through array

nums = [0, 1, 0, 3, 12]
        s  f              fast=0 is zero, skip
        s     f           fast=1 is non-zero, swap with slow, slow++
       [1, 0, 0, 3, 12]
           s     f        fast=0 is zero, skip
           s        f     fast=3 is non-zero, swap with slow, slow++
       [1, 3, 0, 0, 12]
              s        f  fast=12 is non-zero, swap with slow, slow++
       [1, 3, 12, 0, 0]   Done!
```

### Solution
```python
def moveZeroes(nums: list[int]) -> None:
    """
    Move all zeros to end while maintaining order of non-zeros.

    Strategy (Two Pointers - Same Direction):
    - slow: marks position for next non-zero element
    - fast: scans through array looking for non-zeros
    - When fast finds non-zero, swap with slow position

    Why this works:
    - Non-zero elements "bubble" to the front
    - Zeros naturally end up at the back
    - Relative order preserved because we process left to right

    Time: O(n) - single pass
    Space: O(1) - in-place
    """
    slow = 0  # Position for next non-zero element

    for fast in range(len(nums)):
        if nums[fast] != 0:
            # Found a non-zero element - swap it to slow position
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1

    # After loop: all non-zeros are at positions 0 to slow-1
    # All zeros are at positions slow to end


def moveZeroes_alternative(nums: list[int]) -> None:
    """
    Alternative approach: First move non-zeros, then fill zeros.

    Slightly more operations but clearer logic.
    """
    slow = 0

    # Step 1: Move all non-zero elements to front
    for fast in range(len(nums)):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1

    # Step 2: Fill remaining positions with zeros
    for i in range(slow, len(nums)):
        nums[i] = 0
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- Single zero: `[0]` → `[0]` (already at end)
- No zeros: `[1,2,3]` → `[1,2,3]` (no change)
- All zeros: `[0,0,0]` → `[0,0,0]` (no change)
- Zeros at start: `[0,0,1,2]` → `[1,2,0,0]`

---

## Problem 5: Remove Duplicates from Sorted Array (LC #26) - Easy

- [LeetCode](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)

### Problem Statement
Remove duplicates from sorted array in-place. Return the number of unique elements.

### Video Explanation
- [NeetCode - Remove Duplicates](https://www.youtube.com/watch?v=DEJAZBq0FDA)

### Examples
```
Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]

Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
```

### Intuition Development
```
Since array is SORTED, duplicates are adjacent!

Two pointers:
- slow: position for next unique element
- fast: scans for new unique elements

nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
        s  f                           nums[f]=0 == nums[s]=0, skip
        s     f                        nums[f]=1 != nums[s]=0, slow++, copy
       [0, 1, 1, 1, 1, 2, 2, 3, 3, 4]
           s     f                     nums[f]=1 == nums[s]=1, skip
           s        f                  nums[f]=1 == nums[s]=1, skip
           s           f               nums[f]=2 != nums[s]=1, slow++, copy
       [0, 1, 2, 1, 1, 2, 2, 3, 3, 4]
              s           f            ... continue
```

### Solution
```python
def removeDuplicates(nums: list[int]) -> int:
    """
    Remove duplicates from sorted array in-place.

    Strategy:
    - Use two pointers (slow and fast)
    - slow: position of last unique element
    - fast: scans for next unique element
    - When fast finds new unique, increment slow and copy

    Key insight: Array is SORTED, so duplicates are ADJACENT

    Time: O(n) - single pass
    Space: O(1) - in-place

    Returns: Number of unique elements
    """
    if not nums:
        return 0

    # slow points to last unique element (start with first element)
    slow = 0

    for fast in range(1, len(nums)):
        # If current element is different from last unique
        if nums[fast] != nums[slow]:
            # Move slow forward and copy the new unique element
            slow += 1
            nums[slow] = nums[fast]

    # slow is index of last unique element
    # Number of unique elements = slow + 1
    return slow + 1


def removeDuplicates_verbose(nums: list[int]) -> int:
    """
    Same solution with more detailed comments for learning.
    """
    if not nums:
        return 0

    # Position 0 always contains first unique element
    # So we start slow at 0 (first unique is already in place)
    slow = 0

    # Start fast at 1 (we'll compare each element with previous unique)
    for fast in range(1, len(nums)):
        # Compare current element with the last unique element
        current = nums[fast]
        last_unique = nums[slow]

        if current != last_unique:
            # Found a new unique element!
            # 1. Move slow forward to next position
            slow += 1
            # 2. Place the new unique element there
            nums[slow] = current
            # Note: We don't need to "delete" duplicates
            # We just overwrite them and return the new length

    # Array now looks like: [unique1, unique2, ..., uniqueN, garbage...]
    # Return count of unique elements
    return slow + 1
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- Single element: `[1]` → return `1`
- All same: `[1,1,1]` → return `1`
- Already unique: `[1,2,3]` → return `3`
- All duplicates: `[1,1,2,2,3,3]` → return `3`

---

## Problem 6: Squares of a Sorted Array (LC #977) - Easy

- [LeetCode](https://leetcode.com/problems/squares-of-a-sorted-array/)

### Problem Statement
Given a sorted array, return array of squares in sorted order.

### Video Explanation
- [NeetCode - Squares of Sorted Array](https://www.youtube.com/watch?v=FPCZsG_AkUg)

### Examples
```
Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]

Input: nums = [-7,-3,2,3,11]
Output: [4,9,9,49,121]
```

### Intuition Development
```
Challenge: Negative numbers! Their squares might be larger than positive numbers.

Key insight: Largest squares are at the ENDS (most negative or most positive)

nums = [-4, -1, 0, 3, 10]
        L              R

Squares: 16, 1, 0, 9, 100
         ^              ^
         L              R

Compare |nums[L]| vs |nums[R]|: 4 vs 10 → 10 is larger
Place 100 at end of result, move R left

Continue comparing ends, filling result from back to front.
```

### Solution
```python
def sortedSquares(nums: list[int]) -> list[int]:
    """
    Return squares of sorted array in sorted order.

    Strategy:
    - Two pointers at opposite ends
    - Compare absolute values (largest squares at ends)
    - Fill result array from back to front

    Why this works:
    - Input is sorted, so largest absolute values are at ends
    - Negative numbers: largest absolute value at left
    - Positive numbers: largest value at right
    - Compare and take the larger square

    Time: O(n) - single pass
    Space: O(n) - result array (required by problem)
    """
    n = len(nums)
    result = [0] * n  # Pre-allocate result array

    left = 0
    right = n - 1
    write = n - 1  # Fill from the end (largest values first)

    while left <= right:
        # Compare absolute values
        left_square = nums[left] ** 2
        right_square = nums[right] ** 2

        if left_square > right_square:
            # Left has larger square
            result[write] = left_square
            left += 1
        else:
            # Right has larger or equal square
            result[write] = right_square
            right -= 1

        write -= 1

    return result


def sortedSquares_simple(nums: list[int]) -> list[int]:
    """
    Simple approach: Square all, then sort.

    Time: O(n log n) due to sorting
    Space: O(n) for result

    Less efficient but simpler to understand.
    """
    return sorted(x ** 2 for x in nums)
```

### Complexity
- **Two Pointers**: Time O(n), Space O(n)
- **Simple**: Time O(n log n), Space O(n)

### Edge Cases
- All positive: `[1,2,3]` → `[1,4,9]`
- All negative: `[-3,-2,-1]` → `[1,4,9]`
- Single element: `[5]` → `[25]`
- Zero in array: `[-2,0,1]` → `[0,1,4]`

---

## Summary: Easy Problems Checklist

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | Valid Palindrome | Opposite pointers, skip non-alnum | O(n) | O(1) |
| 2 | Two Sum II | Opposite pointers on sorted | O(n) | O(1) |
| 3 | Merge Sorted Array | Merge from end | O(m+n) | O(1) |
| 4 | Move Zeroes | Same direction pointers | O(n) | O(1) |
| 5 | Remove Duplicates | Same direction on sorted | O(n) | O(1) |
| 6 | Squares of Sorted | Opposite pointers, fill from end | O(n) | O(n) |

---

## Practice More Easy Problems

- [ ] LC #344 - Reverse String
- [ ] LC #392 - Is Subsequence
- [ ] LC #557 - Reverse Words in a String III
- [ ] LC #680 - Valid Palindrome II
- [ ] LC #27 - Remove Element

