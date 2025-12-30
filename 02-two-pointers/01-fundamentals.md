# Two Pointers - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE TWO POINTERS                                 │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Sorted array" + find pair/triplet                                       │
│  ✓ "In-place" modification                                                  │
│  ✓ "Remove duplicates"                                                      │
│  ✓ "Reverse" array/string                                                   │
│  ✓ "Palindrome" check                                                       │
│  ✓ "Merge sorted arrays"                                                    │
│  ✓ "Partition" array                                                        │
│                                                                             │
│  Key insight: Reduce O(n²) brute force to O(n) using pointer movement       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Array indexing and iteration
- [ ] Sorting algorithms (for sorted array problems)
- [ ] Basic loop constructs (while, for)

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO POINTERS MEMORY MAP                                  │
│                                                                             │
│                    ┌──────────────┐                                         │
│         ┌─────────│ TWO POINTERS │─────────┐                                │
│         │         └──────────────┘         │                                │
│         ▼                                  ▼                                │
│  ┌─────────────┐                    ┌─────────────┐                         │
│  │  OPPOSITE   │                    │    SAME     │                         │
│  │  DIRECTION  │                    │  DIRECTION  │                         │
│  └──────┬──────┘                    └──────┬──────┘                         │
│         │                                  │                                │
│    ┌────┴────┐                       ┌─────┴─────┐                          │
│    ▼         ▼                       ▼           ▼                          │
│ ┌──────┐ ┌──────┐                ┌──────┐   ┌──────┐                       │
│ │Two   │ │Palin-│                │Remove│   │Fast/ │                       │
│ │Sum II│ │drome │                │Dups  │   │Slow  │                       │
│ └──────┘ └──────┘                └──────┘   └──────┘                       │
│                                                                             │
│  Related Patterns:                                                          │
│  • Sliding Window - Variable size window problems                           │
│  • Binary Search - When you need to find specific value                     │
│  • Linked List - Fast/slow pointer for cycle detection                      │
│                                                                             │
│  When to combine:                                                           │
│  • Two Pointers + Sorting: 3Sum, 4Sum problems                              │
│  • Two Pointers + Binary Search: Optimized pair finding                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO POINTERS DECISION TREE                               │
│                                                                             │
│  Is the array sorted (or can be sorted)?                                    │
│       │                                                                     │
│       ├── YES → Need to find pair with target sum?                          │
│       │            │                                                        │
│       │            ├── YES → Opposite direction pointers                    │
│       │            │                                                        │
│       │            └── NO → Need to merge/compare two arrays?               │
│       │                         │                                           │
│       │                         ├── YES → Same direction on both            │
│       │                         │                                           │
│       │                         └── NO → Consider other patterns            │
│       │                                                                     │
│       └── NO → Need in-place modification?                                  │
│                    │                                                        │
│                    ├── YES → Same direction (slow/fast)                     │
│                    │         Examples: Remove duplicates, Move zeros        │
│                    │                                                        │
│                    └── NO → Need to check palindrome/reverse?               │
│                                 │                                           │
│                                 ├── YES → Opposite direction                │
│                                 │                                           │
│                                 └── NO → May need hash map instead          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concept

Two pointers is a technique where we use two indices to traverse data structure(s), typically to find pairs or modify arrays in-place.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO POINTER VARIATIONS                                   │
│                                                                             │
│  1. OPPOSITE DIRECTION (Converging)                                         │
│     ┌───┬───┬───┬───┬───┬───┬───┐                                          │
│     │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │                                          │
│     └───┴───┴───┴───┴───┴───┴───┘                                          │
│       ▲                       ▲                                             │
│      left                   right                                           │
│       └─────────────────────►│◄──── Move towards each other                 │
│                                                                             │
│     Use cases: Two Sum (sorted), Palindrome, Container with Most Water      │
│                                                                             │
│  2. SAME DIRECTION (Fast & Slow)                                            │
│     ┌───┬───┬───┬───┬───┬───┬───┐                                          │
│     │ 0 │ 1 │ 0 │ 2 │ 0 │ 3 │ 4 │                                          │
│     └───┴───┴───┴───┴───┴───┴───┘                                          │
│       ▲   ▲                                                                 │
│      slow fast ────────────────►  Both move forward                         │
│                                                                             │
│     Use cases: Remove duplicates, Move zeros, Linked list cycle             │
│                                                                             │
│  3. TWO ARRAYS (Merging)                                                    │
│     Array 1: [1, 3, 5]    Array 2: [2, 4, 6]                               │
│               ▲                     ▲                                       │
│               p1                    p2                                      │
│                                                                             │
│     Use cases: Merge sorted arrays, Intersection                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Essential Templates

### Template 1: Opposite Direction (Converging Pointers)

```python
def two_pointer_opposite(arr: list, target: int) -> bool:
    """
    Template for two pointers moving towards each other.

    Common use cases:
    - Two Sum in sorted array
    - Valid palindrome
    - Container with most water

    Time: O(n) - each element visited at most once
    Space: O(1) - only two pointers
    """
    left = 0                    # Start from beginning
    right = len(arr) - 1        # Start from end

    while left < right:         # Continue until pointers meet
        # Calculate current state (e.g., sum, comparison)
        current = arr[left] + arr[right]

        if current == target:
            return True         # Found answer
        elif current < target:
            left += 1           # Need larger sum, move left pointer right
        else:
            right -= 1          # Need smaller sum, move right pointer left

    return False                # No solution found
```

### Template 2: Same Direction (Fast & Slow)

```python
def two_pointer_same_direction(arr: list) -> int:
    """
    Template for two pointers moving in same direction.

    Common use cases:
    - Remove duplicates from sorted array
    - Move zeros to end
    - Remove element

    Pattern:
    - slow: marks position for next valid element
    - fast: scans through array

    Time: O(n)
    Space: O(1)
    """
    if not arr:
        return 0

    slow = 0  # Position to place next valid element

    for fast in range(len(arr)):
        # Check if current element should be kept
        if should_keep(arr[fast]):
            arr[slow] = arr[fast]  # Place at slow position
            slow += 1              # Move slow forward

    return slow  # New length of modified array


def should_keep(element) -> bool:
    """Condition to determine if element should be kept."""
    pass  # Implement based on problem
```

### Template 3: Two Arrays (Merging)

```python
def merge_two_arrays(arr1: list, arr2: list) -> list:
    """
    Template for two pointers on different arrays.

    Common use cases:
    - Merge sorted arrays
    - Find intersection
    - Compare strings

    Time: O(n + m) where n, m are array lengths
    Space: O(n + m) for result (O(1) if modifying in-place)
    """
    result = []
    p1 = 0  # Pointer for arr1
    p2 = 0  # Pointer for arr2

    # Process while both arrays have elements
    while p1 < len(arr1) and p2 < len(arr2):
        if arr1[p1] <= arr2[p2]:
            result.append(arr1[p1])
            p1 += 1
        else:
            result.append(arr2[p2])
            p2 += 1

    # Add remaining elements from arr1 (if any)
    while p1 < len(arr1):
        result.append(arr1[p1])
        p1 += 1

    # Add remaining elements from arr2 (if any)
    while p2 < len(arr2):
        result.append(arr2[p2])
        p2 += 1

    return result
```

---

## Visual: Why Two Pointers Works on Sorted Arrays

```
Finding pair with sum = 9 in sorted array:

arr = [1, 2, 4, 5, 7, 8, 9]
       L                 R     sum = 1 + 9 = 10 > 9, move R left
       L              R        sum = 1 + 8 = 9 ✓ FOUND!

Why it works:
- Array is SORTED
- If sum too large → decrease by moving right pointer left
- If sum too small → increase by moving left pointer right
- We never miss the answer because we systematically explore all possibilities

Proof of correctness:
- When we move right pointer left, we're eliminating arr[right] as a candidate
  with ALL elements from arr[left] onwards (because sum was too large)
- When we move left pointer right, we're eliminating arr[left] as a candidate
  with ALL elements from arr[right] backwards (because sum was too small)
```

---

## Complexity Analysis

| Variation | Time | Space | When to Use |
|-----------|------|-------|-------------|
| Opposite Direction | O(n) | O(1) | Sorted array, find pair |
| Same Direction | O(n) | O(1) | In-place modification |
| Two Arrays | O(n+m) | O(1)* | Merge, compare |

*O(1) if modifying in-place, O(n+m) if creating new array

---

## Common Mistakes

```python
# ❌ WRONG: Using < instead of <= when needed
def two_sum_wrong(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:  # Might miss when left == right is valid
        # ...

# ❌ WRONG: Not handling duplicates in 3Sum
def three_sum_wrong(nums):
    # Forgetting to skip duplicate values leads to duplicate triplets
    pass

# ❌ WRONG: Modifying array while iterating with wrong index
def remove_duplicates_wrong(arr):
    for i in range(len(arr)):
        if arr[i] == arr[i-1]:  # Index error when i=0!
            arr.pop(i)          # Modifying while iterating - dangerous!

# ✅ CORRECT: Use two pointers pattern
def remove_duplicates_correct(arr):
    if not arr:
        return 0
    slow = 1  # Start from 1, keep first element
    for fast in range(1, len(arr)):
        if arr[fast] != arr[fast - 1]:  # Different from previous
            arr[slow] = arr[fast]
            slow += 1
    return slow
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"Since the array is sorted, I'll use two pointers starting from opposite
ends. If the sum is too large, I move the right pointer left to decrease
it. If too small, I move the left pointer right. This gives O(n) time."
```

### 2. What Interviewers Look For
- **Pattern recognition**: Quickly identify two pointer opportunities
- **Pointer movement logic**: Clear reasoning for when to move which pointer
- **Edge cases**: Empty array, single element, no valid pair

### 3. Common Follow-up Questions
- "What if array is not sorted?" → Sort first O(n log n), or use hash map
- "Can you find all pairs?" → Continue after finding one, skip duplicates
- "What about 3Sum/4Sum?" → Fix one pointer, use two pointers for rest

---

## Related Patterns

- **Sliding Window**: When you need to track a contiguous range
- **Binary Search**: When you need to find a specific value in sorted array
- **Fast & Slow Pointers**: Specialized two pointers for linked lists/cycles

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
