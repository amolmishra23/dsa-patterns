# Sliding Window - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE SLIDING WINDOW                               │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Contiguous subarray/substring"                                          │
│  ✓ "Maximum/Minimum length"                                                 │
│  ✓ "Longest/Shortest with condition"                                        │
│  ✓ "Sum equals K" / "At most K distinct"                                    │
│  ✓ "Window of size K"                                                       │
│  ✓ "Consecutive elements"                                                   │
│                                                                             │
│  Key insight: Maintain a "window" that slides through data                  │
│               Avoid recomputing from scratch each time                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Array/string iteration
- [ ] Hash maps for tracking frequencies
- [ ] Two pointers concept

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SLIDING WINDOW MEMORY MAP                                │
│                                                                             │
│                    ┌───────────────┐                                        │
│         ┌─────────│SLIDING WINDOW │─────────┐                               │
│         │         └───────────────┘         │                               │
│         ▼                                   ▼                               │
│  ┌─────────────┐                     ┌─────────────┐                        │
│  │   FIXED     │                     │  VARIABLE   │                        │
│  │    SIZE     │                     │    SIZE     │                        │
│  └──────┬──────┘                     └──────┬──────┘                        │
│         │                                   │                               │
│    ┌────┴────┐                        ┌─────┴─────┐                         │
│    ▼         ▼                        ▼           ▼                         │
│ ┌──────┐ ┌──────┐                 ┌──────┐   ┌──────┐                      │
│ │Max   │ │Moving│                 │Longest│  │Min   │                      │
│ │Sum K │ │Avg   │                 │Substr │  │Window│                      │
│ └──────┘ └──────┘                 └──────┘   └──────┘                      │
│                                                                             │
│  Related Patterns:                                                          │
│  • Two Pointers - Sliding window is specialized two pointers                │
│  • Prefix Sum - For range sums with negative numbers                        │
│  • Hash Map - Track window contents efficiently                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SLIDING WINDOW DECISION TREE                             │
│                                                                             │
│  Is window size fixed?                                                      │
│       │                                                                     │
│       ├── YES → Fixed Window Template                                       │
│       │         Initialize window, slide by adding/removing one element     │
│       │                                                                     │
│       └── NO → What condition determines window validity?                   │
│                    │                                                        │
│                    ├── Sum/count constraint → Variable Window               │
│                    │   Expand right, shrink left when invalid               │
│                    │                                                        │
│                    ├── "At most K" problems → Variable + counting trick     │
│                    │   atMost(K) - atMost(K-1) = exactly K                  │
│                    │                                                        │
│                    └── Negative numbers in sum? → Consider Prefix Sum       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concept

Instead of recalculating for every subarray (O(n²)), maintain a window and update incrementally (O(n)).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SLIDING WINDOW VISUALIZATION                             │
│                                                                             │
│  FIXED SIZE WINDOW (size k=3):                                              │
│                                                                             │
│  Array: [1, 3, 2, 6, -1, 4, 1, 8, 2]                                        │
│          └──────┘                      Window 1: sum = 6                    │
│             └──────┘                   Window 2: sum = 11 (add 6, remove 1) │
│                └──────┘                Window 3: sum = 7                    │
│                   └──────┘             Window 4: sum = 9                    │
│                      ...                                                    │
│                                                                             │
│  VARIABLE SIZE WINDOW:                                                      │
│                                                                             │
│  String: "ADOBECODEBANC"                                                    │
│           └───────────┘    Expand until valid                               │
│            └──────────┘    Contract while still valid                       │
│             └─────────┘    Keep contracting...                              │
│                  └────┘    Minimum valid window!                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Two Types of Sliding Window

### Type 1: Fixed Size Window

Window size is given. Slide window one element at a time.

```python
def fixed_window_template(arr: list, k: int):
    """
    Template for fixed-size sliding window.

    Pattern:
    1. Initialize window with first k elements
    2. Slide: add new element, remove old element
    3. Process/update result at each position

    Time: O(n) - each element added and removed once
    Space: O(1) or O(k) depending on what we track
    """
    n = len(arr)
    if n < k:
        return None  # Not enough elements

    # ===== STEP 1: Initialize first window =====
    window_sum = sum(arr[:k])  # Or other initialization
    result = window_sum

    # ===== STEP 2: Slide the window =====
    for i in range(k, n):
        # Add new element (entering window)
        window_sum += arr[i]

        # Remove old element (leaving window)
        window_sum -= arr[i - k]

        # Update result
        result = max(result, window_sum)  # Or other update logic

    return result
```

### Type 2: Variable Size Window

Window size changes based on condition. Expand and contract dynamically.

```python
def variable_window_template(arr: list, condition):
    """
    Template for variable-size sliding window.

    Pattern:
    1. Expand window by moving right pointer
    2. When condition violated, contract by moving left pointer
    3. Track result (usually at valid states)

    Two common variations:
    - Find LONGEST valid window: update result when valid, then expand
    - Find SHORTEST valid window: update result when valid, then contract

    Time: O(n) - each element visited at most twice (once by each pointer)
    Space: O(1) or O(k) depending on what we track
    """
    left = 0
    result = 0  # Or float('inf') for minimum
    window_state = {}  # Track window contents (e.g., character counts)

    for right in range(len(arr)):
        # ===== EXPAND: Add element at right =====
        element = arr[right]
        # Update window state (e.g., add to count)
        window_state[element] = window_state.get(element, 0) + 1

        # ===== CONTRACT: Shrink while condition violated =====
        while not is_valid(window_state, condition):
            # Remove element at left
            left_element = arr[left]
            # Update window state (e.g., decrease count)
            window_state[left_element] -= 1
            if window_state[left_element] == 0:
                del window_state[left_element]
            left += 1

        # ===== UPDATE RESULT =====
        # Current window [left, right] is valid
        window_length = right - left + 1
        result = max(result, window_length)  # For longest
        # result = min(result, window_length)  # For shortest

    return result
```

---

## Visual: Fixed vs Variable Window

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FIXED SIZE WINDOW (k=3)                                  │
│                                                                             │
│  Step 1: [1, 2, 3] 4, 5, 6    Initialize: sum = 6                          │
│  Step 2:  1 [2, 3, 4] 5, 6    Slide: sum = 6 - 1 + 4 = 9                   │
│  Step 3:  1, 2 [3, 4, 5] 6    Slide: sum = 9 - 2 + 5 = 12                  │
│  Step 4:  1, 2, 3 [4, 5, 6]   Slide: sum = 12 - 3 + 6 = 15                 │
│                                                                             │
│  Window size NEVER changes!                                                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                    VARIABLE SIZE WINDOW                                     │
│                                                                             │
│  Find longest substring with at most 2 distinct characters                  │
│  String: "eceba"                                                            │
│                                                                             │
│  [e]ceba         distinct={e:1}         length=1                           │
│  [ec]eba         distinct={e:1,c:1}     length=2                           │
│  [ece]ba         distinct={e:2,c:1}     length=3                           │
│  [eceb]a         distinct={e:2,c:1,b:1} TOO MANY! Contract...              │
│   [ceb]a         distinct={c:1,e:1,b:1} Still 3, contract...               │
│    [eb]a         distinct={e:1,b:1}     length=2                           │
│    [eba]         distinct={e:1,b:1,a:1} TOO MANY! Contract...              │
│     [ba]         distinct={b:1,a:1}     length=2                           │
│                                                                             │
│  Maximum length = 3 ("ece")                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Window State Tracking

```python
# 1. Track sum
window_sum = 0

# 2. Track character/element count
from collections import defaultdict
window_count = defaultdict(int)  # element -> count

# 3. Track number of distinct elements
num_distinct = len(window_count)

# 4. Track whether window is valid
# Example: all required characters present
have = 0  # Characters we have enough of
required = len(target_count)  # Characters we need
is_valid = (have == required)
```

---

## Complexity Analysis

| Window Type | Time | Space | Use Case |
|-------------|------|-------|----------|
| Fixed Size | O(n) | O(1) or O(k) | Max sum of k elements |
| Variable Size | O(n) | O(k) | Longest/shortest with condition |

**Why O(n) for variable window?**
- Each element is added to window at most once (right pointer)
- Each element is removed from window at most once (left pointer)
- Total operations: at most 2n = O(n)

---

## Common Mistakes

```python
# ❌ WRONG: Not handling empty window
def longest_substring_wrong(s, k):
    left = 0
    for right in range(len(s)):
        while condition_violated():
            left += 1
        # Bug: What if left > right? Window is invalid!

# ✅ CORRECT: Check window validity
def longest_substring_correct(s, k):
    left = 0
    max_len = 0
    for right in range(len(s)):
        while left <= right and condition_violated():
            left += 1
        if left <= right:  # Valid window exists
            max_len = max(max_len, right - left + 1)
    return max_len

# ❌ WRONG: Using wrong boundary for contraction
def min_window_wrong(s, t):
    while have == required:  # Should be >= sometimes
        # ...

# ❌ WRONG: Off-by-one in window size
def fixed_window_wrong(arr, k):
    for i in range(k, len(arr)):
        window_sum += arr[i]
        window_sum -= arr[i - k + 1]  # Wrong! Should be arr[i - k]
```

---

## Related Patterns

- **Two Pointers**: Sliding window is a specialized two-pointer technique
- **Prefix Sum**: Alternative for range sum queries (especially with negative numbers)
- **Hash Map**: Often used to track window contents efficiently

---

## Decision Guide

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHICH APPROACH TO USE?                                   │
│                                                                             │
│  Problem mentions "subarray" or "substring"?                                │
│       │                                                                     │
│       ├──► Window size given? ──────────────────► FIXED WINDOW             │
│       │                                                                     │
│       ├──► Find longest/shortest with condition? ──► VARIABLE WINDOW       │
│       │                                                                     │
│       ├──► Sum equals K (with negatives)? ──────► PREFIX SUM + HASH MAP    │
│       │                                                                     │
│       └──► Count subarrays with condition? ─────► Usually VARIABLE WINDOW  │
│            (might need "at most K" trick)                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"I'll use a sliding window to maintain a contiguous subarray. I'll expand
the window by moving the right pointer, and shrink it by moving the left
pointer when the condition is violated. This gives O(n) time complexity."
```

### 2. What Interviewers Look For
- **Template selection**: Fixed vs variable window
- **Window state tracking**: What to track (sum, count, frequency map)
- **Shrinking logic**: When and how to contract the window

### 3. Common Follow-up Questions
- "What if there are negative numbers?" → Prefix sum might be needed
- "Can you count all valid subarrays?" → Use atMost(K) - atMost(K-1) trick
- "What's the space complexity?" → O(1) for sum, O(k) for frequency map

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
