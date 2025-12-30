# Two Pointers - Complete Practice List

## Organized by Pattern and Difficulty

### Pattern 1: Opposite Direction (Sorted Array)

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 167 | [Two Sum II](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/) | Medium | Classic two sum on sorted |
| 15 | [3Sum](https://leetcode.com/problems/3sum/) | Medium | Fix one, two-pointer rest |
| 18 | [4Sum](https://leetcode.com/problems/4sum/) | Medium | Fix two, two-pointer rest |
| 11 | [Container With Most Water](https://leetcode.com/problems/container-with-most-water/) | Medium | Move shorter line |
| 42 | [Trapping Rain Water](https://leetcode.com/problems/trapping-rain-water/) | Hard | Track left/right max |
| 977 | [Squares of a Sorted Array](https://leetcode.com/problems/squares-of-a-sorted-array/) | Easy | Merge from ends |

### Pattern 2: Same Direction (Fast/Slow)

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 26 | [Remove Duplicates](https://leetcode.com/problems/remove-duplicates-from-sorted-array/) | Easy | Slow tracks unique |
| 27 | [Remove Element](https://leetcode.com/problems/remove-element/) | Easy | Slow tracks valid |
| 283 | [Move Zeroes](https://leetcode.com/problems/move-zeroes/) | Easy | Swap non-zeros forward |
| 80 | [Remove Duplicates II](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/) | Medium | Allow 2 duplicates |
| 75 | [Sort Colors](https://leetcode.com/problems/sort-colors/) | Medium | Dutch flag (3 pointers) |

### Pattern 3: Linked List (Cycle Detection)

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 141 | [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/) | Easy | Fast catches slow |
| 142 | [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/) | Medium | Find cycle start |
| 287 | [Find the Duplicate](https://leetcode.com/problems/find-the-duplicate-number/) | Medium | Array as linked list |
| 876 | [Middle of Linked List](https://leetcode.com/problems/middle-of-the-linked-list/) | Easy | Fast moves 2x |
| 234 | [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/) | Easy | Find middle, reverse half |

### Pattern 4: String Problems

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 125 | [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/) | Easy | Skip non-alphanumeric |
| 680 | [Valid Palindrome II](https://leetcode.com/problems/valid-palindrome-ii/) | Easy | Allow one deletion |
| 344 | [Reverse String](https://leetcode.com/problems/reverse-string/) | Easy | Swap from ends |
| 345 | [Reverse Vowels](https://leetcode.com/problems/reverse-vowels-of-a-string/) | Easy | Two pointers on vowels |
| 5 | [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/) | Medium | Expand from center |

### Pattern 5: Partition/Rearrange

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 905 | [Sort Array By Parity](https://leetcode.com/problems/sort-array-by-parity/) | Easy | Partition by odd/even |
| 922 | [Sort Array By Parity II](https://leetcode.com/problems/sort-array-by-parity-ii/) | Easy | Two pointers for positions |
| 88 | [Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/) | Easy | Merge from end |

---

## Essential Templates

### 1. Opposite Direction (Two Sum Pattern)
```python
def two_sum_sorted(nums: list[int], target: int) -> list[int]:
    """
    Find two numbers that sum to target in sorted array.

    Time: O(n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left < right:
        total = nums[left] + nums[right]

        if total == target:
            return [left, right]
        elif total < target:
            left += 1
        else:
            right -= 1

    return []
```

### 2. 3Sum Template
```python
def threeSum(nums: list[int]) -> list[list[int]]:
    """
    Find all unique triplets that sum to zero.

    Time: O(n²)
    Space: O(1) excluding output
    """
    nums.sort()
    result = []
    n = len(nums)

    for i in range(n - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # Two pointer for remaining
        left, right = i + 1, n - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]

            if total == 0:
                result.append([nums[i], nums[left], nums[right]])

                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result
```

### 3. Fast/Slow (Remove Duplicates)
```python
def removeDuplicates(nums: list[int]) -> int:
    """
    Remove duplicates in-place, return new length.

    Time: O(n)
    Space: O(1)
    """
    if not nums:
        return 0

    slow = 0  # Position to write next unique

    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]

    return slow + 1
```

### 4. Container With Most Water
```python
def maxArea(height: list[int]) -> int:
    """
    Find maximum water container.

    Key insight: Move the shorter line inward.

    Time: O(n)
    Space: O(1)
    """
    left, right = 0, len(height) - 1
    max_area = 0

    while left < right:
        # Calculate area
        width = right - left
        h = min(height[left], height[right])
        max_area = max(max_area, width * h)

        # Move shorter line
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_area
```

### 5. Trapping Rain Water
```python
def trap(height: list[int]) -> int:
    """
    Calculate trapped rain water.

    Water at position = min(left_max, right_max) - height

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
            if height[left] >= left_max:
                left_max = height[left]
            else:
                water += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                water += right_max - height[right]
            right -= 1

    return water
```

### 6. Dutch National Flag (3-way Partition)
```python
def sortColors(nums: list[int]) -> None:
    """
    Sort array with 0s, 1s, 2s in-place.

    Three pointers:
    - low: boundary for 0s
    - mid: current element
    - high: boundary for 2s

    Time: O(n)
    Space: O(1)
    """
    low = mid = 0
    high = len(nums) - 1

    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1
```

### 7. Linked List Cycle Detection
```python
def hasCycle(head) -> bool:
    """
    Detect cycle using Floyd's algorithm.

    Time: O(n)
    Space: O(1)
    """
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            return True

    return False


def detectCycle(head):
    """
    Find cycle start node.

    After meeting:
    - Reset one pointer to head
    - Move both at same speed
    - They meet at cycle start
    """
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            # Find cycle start
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow

    return None
```

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO POINTER PATTERNS                                     │
│                                                                             │
│  OPPOSITE DIRECTION (Two Sum):                                              │
│  [1, 2, 3, 4, 6, 8, 9], target = 10                                        │
│   L              R     → 1 + 9 = 10 ✓                                       │
│                                                                             │
│  SAME DIRECTION (Remove Duplicates):                                        │
│  [1, 1, 2, 2, 3]                                                           │
│   S  F           → nums[F] == nums[S], skip                                 │
│   S     F        → nums[F] != nums[S], copy to S+1                          │
│      S     F     → nums[F] == nums[S], skip                                 │
│      S        F  → nums[F] != nums[S], copy to S+1                          │
│  Result: [1, 2, 3, ...]                                                     │
│                                                                             │
│  CYCLE DETECTION (Floyd's):                                                 │
│  1 → 2 → 3 → 4 → 5                                                         │
│              ↑   ↓                                                          │
│              8 ← 7 ← 6                                                      │
│  Slow: 1→2→3→4→5→6→7→8→4→5                                                 │
│  Fast: 1→3→5→7→4→6→8→5                                                     │
│  Meet at 5!                                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] Two Sum II (LC #167)
- [ ] Remove Duplicates (LC #26)
- [ ] Move Zeroes (LC #283)
- [ ] Valid Palindrome (LC #125)
- [ ] Reverse String (LC #344)

### Week 2: Intermediate
- [ ] 3Sum (LC #15)
- [ ] Container With Most Water (LC #11)
- [ ] Sort Colors (LC #75)
- [ ] Linked List Cycle (LC #141)
- [ ] Middle of Linked List (LC #876)

### Week 3: Advanced
- [ ] 4Sum (LC #18)
- [ ] Trapping Rain Water (LC #42)
- [ ] Linked List Cycle II (LC #142)
- [ ] Find the Duplicate Number (LC #287)
- [ ] Longest Palindromic Substring (LC #5)

---

## Common Mistakes

1. **Not handling duplicates in 3Sum/4Sum**
   - Skip duplicate first elements
   - Skip duplicate pairs after finding solution

2. **Wrong pointer movement**
   - In two sum: move based on comparison with target
   - In container: always move shorter line

3. **Infinite loop in cycle detection**
   - Check `fast and fast.next` before moving
   - Ensure termination condition

4. **Off-by-one in remove duplicates**
   - Return `slow + 1` (length, not index)
   - Handle empty array

5. **Not sorting when required**
   - Two sum on sorted array requires sorting
   - 3Sum requires sorting for deduplication

---

## Complexity Reference

| Pattern | Time | Space |
|---------|------|-------|
| Two Sum (sorted) | O(n) | O(1) |
| 3Sum | O(n²) | O(1) |
| 4Sum | O(n³) | O(1) |
| Container | O(n) | O(1) |
| Trapping Water | O(n) | O(1) |
| Remove Duplicates | O(n) | O(1) |
| Sort Colors | O(n) | O(1) |
| Cycle Detection | O(n) | O(1) |

