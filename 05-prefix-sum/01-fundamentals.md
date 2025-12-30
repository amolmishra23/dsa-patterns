# Prefix Sum - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE PREFIX SUM                                   │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Subarray sum"                                                           │
│  ✓ "Range sum query"                                                        │
│  ✓ "Sum equals K"                                                           │
│  ✓ "Count subarrays with sum..."                                            │
│  ✓ "Cumulative sum"                                                         │
│  ✓ "Running total"                                                          │
│                                                                             │
│  Key insight: Precompute cumulative sums to answer range queries in O(1)    │
│                                                                             │
│  sum(i, j) = prefix[j+1] - prefix[i]                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Array basics and indexing
- [ ] Hash maps for O(1) lookup
- [ ] Basic math (cumulative sums)

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREFIX SUM MEMORY MAP                                    │
│                                                                             │
│                    ┌────────────┐                                           │
│         ┌─────────│ PREFIX SUM │─────────┐                                  │
│         │         └────────────┘         │                                  │
│         ▼                                ▼                                  │
│  ┌─────────────┐                  ┌─────────────┐                           │
│  │    1D       │                  │    2D       │                           │
│  │ PREFIX SUM  │                  │ PREFIX SUM  │                           │
│  └──────┬──────┘                  └──────┬──────┘                           │
│         │                                │                                  │
│    ┌────┴────┐                     ┌─────┴─────┐                            │
│    ▼         ▼                     ▼           ▼                            │
│ ┌──────┐ ┌──────┐              ┌──────┐   ┌──────┐                         │
│ │Range │ │Subarray│            │Matrix│   │Region │                         │
│ │Sum   │ │Sum=K  │             │Sum   │   │Queries│                         │
│ └──────┘ └──────┘              └──────┘   └──────┘                         │
│                                                                             │
│  Related Patterns:                                                          │
│  • Sliding Window - For contiguous subarrays (positive only)                │
│  • Hash Map - Combined for "subarray sum = K" problems                      │
│  • Difference Array - For range updates                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREFIX SUM DECISION TREE                                 │
│                                                                             │
│  Need to compute range/subarray sums?                                       │
│       │                                                                     │
│       ├── YES → Multiple queries?                                           │
│       │            │                                                        │
│       │            ├── YES → Build prefix sum array O(n), query O(1)        │
│       │            │                                                        │
│       │            └── NO → Direct sum might be simpler                     │
│       │                                                                     │
│       └── NO → Need to find subarray with specific sum?                     │
│                    │                                                        │
│                    ├── YES → Prefix Sum + Hash Map                          │
│                    │         sum[i..j] = K means prefix[j] - prefix[i] = K  │
│                    │         So we look for prefix[j] - K in hash map       │
│                    │                                                        │
│                    └── NO → Consider other patterns                         │
│                                                                             │
│  WHEN TO USE WHICH:                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ Problem Type              │ Approach                               │     │
│  ├───────────────────────────┼────────────────────────────────────────┤     │
│  │ Range sum queries         │ Prefix sum array                       │     │
│  │ Subarray sum = K          │ Prefix sum + Hash map                  │     │
│  │ Max subarray (positive)   │ Sliding window                         │     │
│  │ Max subarray (any)        │ Kadane's algorithm                     │     │
│  │ 2D range sum              │ 2D prefix sum                          │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concept

Prefix sum allows us to compute any subarray sum in O(1) after O(n) preprocessing.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREFIX SUM VISUALIZATION                                 │
│                                                                             │
│  Array:      [3,  1,  2,  4,  6,  2]                                        │
│  Index:       0   1   2   3   4   5                                         │
│                                                                             │
│  Prefix:    [0,  3,  4,  6, 10, 16, 18]                                     │
│  Index:      0   1   2   3   4   5   6                                      │
│                                                                             │
│  prefix[i] = sum of elements from index 0 to i-1                            │
│  prefix[0] = 0 (empty prefix)                                               │
│  prefix[1] = arr[0] = 3                                                     │
│  prefix[2] = arr[0] + arr[1] = 4                                            │
│  prefix[6] = arr[0] + ... + arr[5] = 18                                     │
│                                                                             │
│  Sum of subarray [i, j] = prefix[j+1] - prefix[i]                           │
│                                                                             │
│  Example: sum(1, 3) = sum of arr[1..3] = 1 + 2 + 4 = 7                      │
│           = prefix[4] - prefix[1] = 10 - 3 = 7 ✓                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Building Prefix Sum Array

```python
def build_prefix_sum(arr: list[int]) -> list[int]:
    """
    Build prefix sum array from input array.

    prefix[i] = sum of arr[0] to arr[i-1]
    prefix[0] = 0 (sum of empty array)

    This allows us to compute sum(i, j) = prefix[j+1] - prefix[i]

    Time: O(n) to build
    Space: O(n) for prefix array

    Args:
        arr: Input array of numbers

    Returns:
        Prefix sum array of length n+1
    """
    n = len(arr)

    # prefix[i] = sum of first i elements (arr[0] to arr[i-1])
    prefix = [0] * (n + 1)

    for i in range(n):
        # Each prefix sum = previous prefix sum + current element
        prefix[i + 1] = prefix[i] + arr[i]

    return prefix


def range_sum(prefix: list[int], i: int, j: int) -> int:
    """
    Get sum of elements from index i to j (inclusive).

    Formula: sum(i, j) = prefix[j+1] - prefix[i]

    Why this works:
    - prefix[j+1] = arr[0] + arr[1] + ... + arr[j]
    - prefix[i] = arr[0] + arr[1] + ... + arr[i-1]
    - Difference = arr[i] + arr[i+1] + ... + arr[j]

    Time: O(1) per query
    """
    return prefix[j + 1] - prefix[i]


# Example usage
arr = [3, 1, 2, 4, 6, 2]
prefix = build_prefix_sum(arr)
print(prefix)  # [0, 3, 4, 6, 10, 16, 18]

# Sum from index 1 to 3 (elements: 1, 2, 4)
print(range_sum(prefix, 1, 3))  # 7
```

---

## Common Prefix Sum Patterns

### Pattern 1: Range Sum Query (LC #303)

```python
class NumArray:
    """
    Range Sum Query - Immutable.

    Precompute prefix sums to answer range queries in O(1).

    Time: O(n) to build, O(1) per query
    Space: O(n)
    """

    def __init__(self, nums: list[int]):
        """
        Build prefix sum array during initialization.
        """
        # prefix[i] = sum of nums[0..i-1]
        self.prefix = [0]

        for num in nums:
            self.prefix.append(self.prefix[-1] + num)

    def sumRange(self, left: int, right: int) -> int:
        """
        Return sum of elements from index left to right (inclusive).

        Uses the prefix sum formula: sum(i,j) = prefix[j+1] - prefix[i]
        """
        return self.prefix[right + 1] - self.prefix[left]
```

### Pattern 2: Subarray Sum Equals K (LC #560)

```python
def subarraySum(nums: list[int], k: int) -> int:
    """
    Count subarrays with sum equal to k.

    Key insight: If prefix[j] - prefix[i] = k, then sum(i, j-1) = k

    This means: prefix[j] - k = prefix[i]

    Strategy:
    - Track frequency of each prefix sum seen so far
    - For each position, check how many times (current_prefix - k) appeared
    - That's the number of subarrays ending here with sum k

    Time: O(n)
    Space: O(n)
    """
    count = 0
    current_sum = 0

    # Map: prefix_sum -> frequency
    # Initialize with 0:1 (empty prefix has sum 0)
    prefix_count = {0: 1}

    for num in nums:
        # Update running sum
        current_sum += num

        # Check if (current_sum - k) exists in our prefix sums
        # If so, there are subarrays ending here with sum k
        target = current_sum - k
        if target in prefix_count:
            count += prefix_count[target]

        # Add current prefix sum to map
        prefix_count[current_sum] = prefix_count.get(current_sum, 0) + 1

    return count
```

### Pattern 3: Contiguous Array (LC #525)

```python
def findMaxLength(nums: list[int]) -> int:
    """
    Find longest subarray with equal 0s and 1s.

    Trick: Treat 0 as -1, then find longest subarray with sum 0!

    If prefix[j] - prefix[i] = 0, then sum from i to j-1 is 0,
    meaning equal 0s and 1s in that range.

    Strategy:
    - Convert 0s to -1s (conceptually)
    - Track first occurrence of each prefix sum
    - When we see same prefix sum again, subarray between has sum 0

    Time: O(n)
    Space: O(n)
    """
    max_length = 0
    running_sum = 0

    # Map: prefix_sum -> first index where this sum occurred
    # Initialize with 0: -1 (sum 0 at "index -1", before array)
    first_occurrence = {0: -1}

    for i, num in enumerate(nums):
        # Add 1 for 1, subtract 1 for 0
        running_sum += 1 if num == 1 else -1

        if running_sum in first_occurrence:
            # Same sum seen before - subarray between has equal 0s and 1s
            length = i - first_occurrence[running_sum]
            max_length = max(max_length, length)
        else:
            # First time seeing this sum - record the index
            first_occurrence[running_sum] = i

    return max_length
```

### Pattern 4: 2D Prefix Sum (LC #304)

```python
class NumMatrix:
    """
    Range Sum Query 2D - Immutable.

    Precompute 2D prefix sums for O(1) rectangle sum queries.

    prefix[i][j] = sum of all elements in rectangle (0,0) to (i-1,j-1)

    Time: O(m*n) to build, O(1) per query
    Space: O(m*n)
    """

    def __init__(self, matrix: list[list[int]]):
        """
        Build 2D prefix sum matrix.

        Formula: prefix[i][j] = matrix[i-1][j-1]
                              + prefix[i-1][j]
                              + prefix[i][j-1]
                              - prefix[i-1][j-1]
        """
        if not matrix or not matrix[0]:
            self.prefix = [[]]
            return

        m, n = len(matrix), len(matrix[0])

        # Create prefix matrix with extra row and column of zeros
        self.prefix = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Current cell value
                val = matrix[i - 1][j - 1]

                # Sum = current + left + top - top-left (avoid double counting)
                self.prefix[i][j] = (val
                                    + self.prefix[i - 1][j]
                                    + self.prefix[i][j - 1]
                                    - self.prefix[i - 1][j - 1])

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """
        Return sum of elements in rectangle from (row1,col1) to (row2,col2).

        Formula uses inclusion-exclusion:
        sum = prefix[row2+1][col2+1]
            - prefix[row1][col2+1]      (remove top)
            - prefix[row2+1][col1]      (remove left)
            + prefix[row1][col1]        (add back corner, was removed twice)
        """
        return (self.prefix[row2 + 1][col2 + 1]
                - self.prefix[row1][col2 + 1]
                - self.prefix[row2 + 1][col1]
                + self.prefix[row1][col1])
```

---

## Visual: 2D Prefix Sum

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    2D PREFIX SUM VISUALIZATION                              │
│                                                                             │
│  Matrix:                    Prefix Sum:                                     │
│  ┌───┬───┬───┐             ┌───┬───┬───┬───┐                               │
│  │ 1 │ 2 │ 3 │             │ 0 │ 0 │ 0 │ 0 │                               │
│  ├───┼───┼───┤             ├───┼───┼───┼───┤                               │
│  │ 4 │ 5 │ 6 │             │ 0 │ 1 │ 3 │ 6 │                               │
│  ├───┼───┼───┤             ├───┼───┼───┼───┤                               │
│  │ 7 │ 8 │ 9 │             │ 0 │ 5 │12 │21 │                               │
│  └───┴───┴───┘             ├───┼───┼───┼───┤                               │
│                            │ 0 │12 │27 │45 │                               │
│                            └───┴───┴───┴───┘                               │
│                                                                             │
│  Sum of rectangle (1,1) to (2,2) = elements [5,6,8,9] = 28                 │
│                                                                             │
│  = prefix[3][3] - prefix[1][3] - prefix[3][1] + prefix[1][1]               │
│  = 45 - 6 - 12 + 1 = 28 ✓                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prefix Sum + Hash Map Pattern

```python
def subarray_sum_pattern(nums: list[int], condition) -> int:
    """
    General pattern for counting subarrays with prefix sum + hash map.

    Works for:
    - Sum equals K
    - Sum divisible by K
    - Equal 0s and 1s (convert to sum = 0)

    Template:
    1. Track running prefix sum
    2. Use hash map to count/store prefix sums
    3. For each position, check if complementary prefix sum exists
    """
    count = 0
    prefix_sum = 0
    prefix_map = {0: 1}  # or {0: -1} for length problems

    for i, num in enumerate(nums):
        prefix_sum += num  # Or modified value

        # Check for condition (e.g., prefix_sum - k in map)
        target = compute_target(prefix_sum, condition)
        if target in prefix_map:
            count += prefix_map[target]  # Or compute length

        # Update map
        prefix_map[prefix_sum] = prefix_map.get(prefix_sum, 0) + 1

    return count
```

---

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Build 1D prefix | O(n) | O(n) |
| Build 2D prefix | O(m×n) | O(m×n) |
| Range query 1D | O(1) | O(1) |
| Range query 2D | O(1) | O(1) |
| Subarray sum = K | O(n) | O(n) |

---

## Common Mistakes

```python
# ❌ WRONG: Off-by-one error in range sum
def range_sum_wrong(prefix, i, j):
    return prefix[j] - prefix[i]  # Missing +1!

# ✅ CORRECT: Include element at j
def range_sum_correct(prefix, i, j):
    return prefix[j + 1] - prefix[i]


# ❌ WRONG: Forgetting to initialize prefix_map with {0: 1}
def subarray_sum_wrong(nums, k):
    count = 0
    prefix_sum = 0
    prefix_map = {}  # Missing {0: 1}!

    for num in nums:
        prefix_sum += num
        if prefix_sum - k in prefix_map:
            count += prefix_map[prefix_sum - k]
        prefix_map[prefix_sum] = prefix_map.get(prefix_sum, 0) + 1

    return count
    # Fails when subarray starting from index 0 has sum k

# ✅ CORRECT: Initialize with empty prefix
def subarray_sum_correct(nums, k):
    count = 0
    prefix_sum = 0
    prefix_map = {0: 1}  # Empty prefix has sum 0
    # ... rest is same
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"I'll use prefix sums to precompute cumulative sums. Then any range sum
can be computed in O(1) as prefix[j+1] - prefix[i]. For 'subarray sum = K',
I'll combine prefix sum with a hash map to find complement in O(1)."
```

### 2. What Interviewers Look For
- **Understanding the formula**: sum(i,j) = prefix[j+1] - prefix[i]
- **Hash map combination**: For "count subarrays with sum K" problems
- **Initialization**: Remember {0: 1} for subarrays starting at index 0

### 3. Common Follow-up Questions
- "What about 2D matrices?" → Use 2D prefix sum with inclusion-exclusion
- "Can you update elements?" → Consider Fenwick Tree or Segment Tree
- "What if elements can be negative?" → Prefix sum works, sliding window doesn't

---

## Related Patterns

- **Sliding Window**: When window size is fixed or elements are positive
- **Two Pointers**: When looking for subarrays with positive numbers only
- **Difference Array**: For range update operations

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
