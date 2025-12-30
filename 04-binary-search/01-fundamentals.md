# Binary Search - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE BINARY SEARCH                                │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Sorted array"                                                           │
│  ✓ "Find element" / "Search"                                                │
│  ✓ "Minimum/Maximum that satisfies condition"                               │
│  ✓ "Kth element"                                                            │
│  ✓ "O(log n) required"                                                      │
│  ✓ "Rotated sorted array"                                                   │
│  ✓ "Search space" can be halved                                             │
│                                                                             │
│  Key insight: If you can eliminate half the search space each step,         │
│               use binary search! O(n) → O(log n)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Array indexing
- [ ] Loop invariants concept
- [ ] Integer division behavior

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BINARY SEARCH MEMORY MAP                                 │
│                                                                             │
│                    ┌───────────────┐                                        │
│         ┌─────────│ BINARY SEARCH │─────────┐                               │
│         │         └───────────────┘         │                               │
│         ▼                                   ▼                               │
│  ┌─────────────┐                     ┌─────────────┐                        │
│  │  ON ARRAY   │                     │  ON ANSWER  │                        │
│  └──────┬──────┘                     └──────┬──────┘                        │
│         │                                   │                               │
│    ┌────┴────┬────────┐              ┌──────┴──────┐                        │
│    ▼         ▼        ▼              ▼             ▼                        │
│ ┌──────┐ ┌──────┐ ┌──────┐      ┌────────┐  ┌──────────┐                   │
│ │Exact │ │Left/ │ │Rotated│     │Min that │  │Capacity/ │                   │
│ │Match │ │Right │ │Array  │     │satisfies│  │Speed     │                   │
│ └──────┘ └──────┘ └──────┘      └────────┘  └──────────┘                   │
│                                                                             │
│  Related Patterns:                                                          │
│  • Two Pointers - For pair finding in sorted arrays                         │
│  • Divide & Conquer - Binary search is a form of D&C                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BINARY SEARCH DECISION TREE                              │
│                                                                             │
│  Is the search space sorted/monotonic?                                      │
│       │                                                                     │
│       ├── NO → Can't use binary search directly                             │
│       │                                                                     │
│       └── YES → What are you searching for?                                 │
│                    │                                                        │
│                    ├── Exact value → Template 1 (standard)                  │
│                    │   while left <= right, return mid when found           │
│                    │                                                        │
│                    ├── First/Last occurrence → Template 2/3 (boundary)      │
│                    │   while left < right, narrow down to boundary          │
│                    │                                                        │
│                    └── Min/Max satisfying condition → Search on Answer      │
│                        Define search space [lo, hi], binary search on it    │
│                                                                             │
│  TEMPLATE SELECTION:                                                        │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ Goal                        │ Template                             │     │
│  ├─────────────────────────────┼──────────────────────────────────────┤     │
│  │ Find exact value            │ while left <= right, mid exact       │     │
│  │ Find first >= target        │ while left < right, bisect_left      │     │
│  │ Find last <= target         │ while left < right, bisect_right - 1 │     │
│  │ Min satisfying condition    │ Search on answer space               │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concept

Binary search works on **sorted** or **monotonic** data by repeatedly halving the search space.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BINARY SEARCH VISUALIZATION                              │
│                                                                             │
│  Find 7 in sorted array:                                                    │
│                                                                             │
│  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                                            │
│   L           M              R     mid=5, target=7 > 5, search right        │
│                                                                             │
│              [6, 7, 8, 9, 10]                                               │
│               L     M     R        mid=8, target=7 < 8, search left         │
│                                                                             │
│              [6, 7]                                                         │
│               L  M  R              mid=6, target=7 > 6, search right        │
│                                                                             │
│                 [7]                                                         │
│                  L                                                          │
│                  M                                                          │
│                  R                 mid=7 = target, FOUND!                   │
│                                                                             │
│  10 elements → 4 comparisons (log₂(10) ≈ 3.3)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Three Binary Search Templates

### Template 1: Standard Binary Search (Find Exact Match)

```python
def binary_search_standard(nums: list[int], target: int) -> int:
    """
    Find exact target in sorted array.

    Returns: Index of target, or -1 if not found

    Loop invariant: If target exists, it's in [left, right]

    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums) - 1

    while left <= right:  # Note: <= because single element is valid
        # Calculate mid (avoid overflow in other languages)
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid  # Found!
        elif nums[mid] < target:
            left = mid + 1  # Target in right half
        else:
            right = mid - 1  # Target in left half

    return -1  # Not found
```

### Template 2: Find Left Boundary (First Occurrence)

```python
def binary_search_left(nums: list[int], target: int) -> int:
    """
    Find leftmost (first) position where target could be inserted.

    Returns: Index of first element >= target

    Use cases:
    - First occurrence of target
    - Count elements less than target
    - Lower bound

    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums)  # Note: right = len(nums), not len-1

    while left < right:  # Note: < not <=
        mid = left + (right - left) // 2

        if nums[mid] < target:
            left = mid + 1  # Move right, target is larger
        else:
            right = mid  # Move left, might be the answer

    return left  # First position where nums[i] >= target


def first_occurrence(nums: list[int], target: int) -> int:
    """Find first occurrence of target, or -1 if not found."""
    idx = binary_search_left(nums, target)
    if idx < len(nums) and nums[idx] == target:
        return idx
    return -1
```

### Template 3: Find Right Boundary (Last Occurrence)

```python
def binary_search_right(nums: list[int], target: int) -> int:
    """
    Find rightmost position where target could be inserted.

    Returns: Index of first element > target

    Use cases:
    - Position after last occurrence of target
    - Count elements <= target
    - Upper bound

    Time: O(log n)
    Space: O(1)
    """
    left, right = 0, len(nums)

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] <= target:  # Note: <= not <
            left = mid + 1  # Move right
        else:
            right = mid  # Move left

    return left  # First position where nums[i] > target


def last_occurrence(nums: list[int], target: int) -> int:
    """Find last occurrence of target, or -1 if not found."""
    idx = binary_search_right(nums, target) - 1
    if idx >= 0 and nums[idx] == target:
        return idx
    return -1
```

---

## Visual: Template Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEMPLATE COMPARISON                                      │
│                                                                             │
│  Array: [1, 2, 2, 2, 3, 4, 5]                                               │
│  Target: 2                                                                  │
│                                                                             │
│  Standard (Template 1):                                                     │
│  Returns: 2 (any index with value 2)                                        │
│                                                                             │
│  Left Boundary (Template 2):                                                │
│  Returns: 1 (first index where value >= 2)                                  │
│           ↓                                                                 │
│  [1, 2, 2, 2, 3, 4, 5]                                                      │
│      ^                                                                      │
│                                                                             │
│  Right Boundary (Template 3):                                               │
│  Returns: 4 (first index where value > 2)                                   │
│                 ↓                                                           │
│  [1, 2, 2, 2, 3, 4, 5]                                                      │
│              ^                                                              │
│                                                                             │
│  Count of 2s = right_boundary - left_boundary = 4 - 1 = 3                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Binary Search on Answer Space

When the answer itself is monotonic, binary search the answer!

```python
def binary_search_on_answer(condition_func, lo: int, hi: int) -> int:
    """
    Find minimum value where condition becomes True.

    Assumes: condition(x) is monotonic (False...False, True...True)

    Use cases:
    - Minimum capacity to ship packages in D days
    - Koko eating bananas
    - Split array largest sum

    Time: O(log(hi-lo) * cost_of_condition)
    Space: O(1)
    """
    while lo < hi:
        mid = lo + (hi - lo) // 2

        if condition_func(mid):
            hi = mid  # mid works, try smaller
        else:
            lo = mid + 1  # mid doesn't work, try larger

    return lo  # Minimum value where condition is True


# Example: Minimum capacity to ship in D days
def shipWithinDays(weights: list[int], days: int) -> int:
    """Find minimum ship capacity to ship all packages in 'days' days."""

    def can_ship(capacity: int) -> bool:
        """Check if we can ship all packages with given capacity in 'days' days."""
        current_load = 0
        days_needed = 1

        for weight in weights:
            if current_load + weight > capacity:
                days_needed += 1
                current_load = weight
            else:
                current_load += weight

        return days_needed <= days

    # Search space: [max_weight, sum_of_weights]
    # Minimum capacity must fit largest package
    # Maximum capacity is shipping everything in one day
    lo = max(weights)
    hi = sum(weights)

    return binary_search_on_answer(can_ship, lo, hi)
```

---

## Common Patterns

### Pattern 1: Rotated Sorted Array

```python
def search_rotated(nums: list[int], target: int) -> int:
    """
    Search in rotated sorted array.

    Key insight: One half is always sorted!

    [4, 5, 6, 7, 0, 1, 2]
     ↑        ↑  ↑     ↑
    left    mid       right

    Left half [4,5,6,7] is sorted (nums[left] <= nums[mid])
    Check if target is in sorted half, otherwise search other half.
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid

        # Check which half is sorted
        if nums[left] <= nums[mid]:
            # Left half is sorted
            if nums[left] <= target < nums[mid]:
                right = mid - 1  # Target in left half
            else:
                left = mid + 1  # Target in right half
        else:
            # Right half is sorted
            if nums[mid] < target <= nums[right]:
                left = mid + 1  # Target in right half
            else:
                right = mid - 1  # Target in left half

    return -1
```

### Pattern 2: Find Peak Element

```python
def findPeakElement(nums: list[int]) -> int:
    """
    Find any peak element (greater than neighbors).

    Key insight: If nums[mid] < nums[mid+1], peak is on the right
                 Otherwise, peak is on the left (or mid is peak)
    """
    left, right = 0, len(nums) - 1

    while left < right:
        mid = left + (right - left) // 2

        if nums[mid] < nums[mid + 1]:
            # Ascending, peak must be on right
            left = mid + 1
        else:
            # Descending or peak, answer is mid or left
            right = mid

    return left
```

---

## Complexity Analysis

| Scenario | Time | Space |
|----------|------|-------|
| Standard search | O(log n) | O(1) |
| Search on answer | O(log(range) × check) | O(1) |
| Recursive | O(log n) | O(log n) stack |

---

## Common Mistakes

```python
# ❌ WRONG: Integer overflow (in other languages)
mid = (left + right) / 2  # Can overflow!

# ✅ CORRECT: Safe mid calculation
mid = left + (right - left) // 2

# ❌ WRONG: Infinite loop with wrong boundary
while left < right:
    mid = left + (right - left) // 2
    if condition:
        right = mid - 1  # Bug! Should be right = mid
    else:
        left = mid  # Bug! Should be left = mid + 1

# ❌ WRONG: Off-by-one in boundary
def search_left(nums, target):
    left, right = 0, len(nums) - 1  # Bug! Should be len(nums)
    while left < right:
        # ...
```

---

## Python's bisect Module

```python
import bisect

# bisect_left: Find leftmost position to insert (lower bound)
# bisect_right: Find rightmost position to insert (upper bound)

nums = [1, 2, 2, 2, 3, 4, 5]

bisect.bisect_left(nums, 2)   # Returns 1 (first 2)
bisect.bisect_right(nums, 2)  # Returns 4 (after last 2)

# Count occurrences
count = bisect.bisect_right(nums, 2) - bisect.bisect_left(nums, 2)  # 3

# Insert maintaining sorted order
bisect.insort_left(nums, 2.5)  # Inserts at correct position
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"Since the array is sorted, I can use binary search to achieve O(log n).
I'll maintain invariant that target is in [left, right]. Each iteration
I check mid: if too small, search right half; if too large, search left."
```

### 2. What Interviewers Look For
- **Template mastery**: Know when to use `<=` vs `<`, `mid` vs `mid±1`
- **Loop invariant**: Can you explain what's true at each iteration?
- **Edge cases**: Empty array, single element, target not found

### 3. Common Follow-up Questions
- "What if there are duplicates?" → Use boundary search templates
- "What if array is rotated?" → Check which half is sorted
- "Can you do this without binary search?" → Yes, but O(n) vs O(log n)

---

## Related Patterns

- **Two Pointers**: When array is sorted, both patterns may apply
- **Sliding Window**: For subarray problems on sorted data
- **Divide and Conquer**: Binary search is a specific form of D&C

### When to Combine

- **Binary Search + Two Pointers**: Find pair in sorted array with target sum
- **Binary Search on Answer**: When checking feasibility is O(n), search answer space in O(log range)

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
