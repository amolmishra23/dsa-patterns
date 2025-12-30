# Prefix Sum - Complete Practice List

## Organized by Pattern and Difficulty

### Pattern 1: Basic Prefix Sum

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 303 | [Range Sum Query - Immutable](https://leetcode.com/problems/range-sum-query-immutable/) | Easy | Basic prefix sum |
| 1480 | [Running Sum of 1d Array](https://leetcode.com/problems/running-sum-of-1d-array/) | Easy | Cumulative sum |
| 724 | [Find Pivot Index](https://leetcode.com/problems/find-pivot-index/) | Easy | Left = Right sum |
| 1991 | [Find Middle Index](https://leetcode.com/problems/find-the-middle-index-in-array/) | Easy | Same as pivot |

### Pattern 2: Prefix Sum + Hash Map

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 560 | [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/) | Medium | Count complement |
| 525 | [Contiguous Array](https://leetcode.com/problems/contiguous-array/) | Medium | Balance (0→-1) |
| 930 | [Binary Subarrays With Sum](https://leetcode.com/problems/binary-subarrays-with-sum/) | Medium | Count exact sum |
| 974 | [Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/) | Medium | Modulo counting |
| 523 | [Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/) | Medium | Modulo + index |
| 1248 | [Count Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/) | Medium | Odd count as sum |

### Pattern 3: 2D Prefix Sum

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 304 | [Range Sum Query 2D](https://leetcode.com/problems/range-sum-query-2d-immutable/) | Medium | 2D prefix sum |
| 1314 | [Matrix Block Sum](https://leetcode.com/problems/matrix-block-sum/) | Medium | 2D range query |
| 221 | [Maximal Square](https://leetcode.com/problems/maximal-square/) | Medium | DP (related) |

### Pattern 4: Prefix Sum on Trees

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 437 | [Path Sum III](https://leetcode.com/problems/path-sum-iii/) | Medium | DFS + prefix map |
| 124 | [Binary Tree Max Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/) | Hard | DFS (related) |

### Pattern 5: Advanced Applications

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 238 | [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/) | Medium | Prefix/suffix product |
| 1588 | [Sum of All Odd Length Subarrays](https://leetcode.com/problems/sum-of-all-odd-length-subarrays/) | Easy | Contribution counting |
| 1074 | [Number of Submatrices That Sum to Target](https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/) | Hard | 2D prefix + hash |

---

## Essential Templates

### 1. Basic Prefix Sum Array
```python
def build_prefix(nums):
    """Build prefix sum array."""
    prefix = [0]  # prefix[i] = sum of nums[0:i]

    for num in nums:
        prefix.append(prefix[-1] + num)

    return prefix

def range_sum(prefix, i, j):
    """Get sum of nums[i:j+1]."""
    return prefix[j + 1] - prefix[i]
```

### 2. Subarray Sum Equals K
```python
def subarray_sum(nums, k):
    """Count subarrays with sum = k."""
    count = 0
    prefix_sum = 0
    prefix_count = {0: 1}  # Initialize with 0 sum

    for num in nums:
        prefix_sum += num

        # Check if complement exists
        count += prefix_count.get(prefix_sum - k, 0)

        # Update count of current prefix sum
        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1

    return count
```

### 3. Contiguous Array (Equal 0s and 1s)
```python
def find_max_length(nums):
    """Find longest subarray with equal 0s and 1s."""
    # Convert 0 to -1, find subarray with sum = 0
    prefix_sum = 0
    first_occurrence = {0: -1}
    max_length = 0

    for i, num in enumerate(nums):
        prefix_sum += 1 if num == 1 else -1

        if prefix_sum in first_occurrence:
            max_length = max(max_length, i - first_occurrence[prefix_sum])
        else:
            first_occurrence[prefix_sum] = i

    return max_length
```

### 4. 2D Prefix Sum
```python
def build_2d_prefix(matrix):
    """Build 2D prefix sum matrix."""
    if not matrix:
        return []

    m, n = len(matrix), len(matrix[0])
    prefix = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            prefix[i][j] = (matrix[i-1][j-1] +
                          prefix[i-1][j] +
                          prefix[i][j-1] -
                          prefix[i-1][j-1])

    return prefix

def region_sum(prefix, r1, c1, r2, c2):
    """Get sum of region [r1:r2+1, c1:c2+1]."""
    return (prefix[r2+1][c2+1] - prefix[r1][c2+1] -
            prefix[r2+1][c1] + prefix[r1][c1])
```

### 5. Prefix Sum with Modulo
```python
def subarrays_div_by_k(nums, k):
    """Count subarrays divisible by k."""
    count = 0
    prefix_sum = 0
    mod_count = {0: 1}

    for num in nums:
        prefix_sum += num
        mod = prefix_sum % k

        # Handle negative modulo
        if mod < 0:
            mod += k

        count += mod_count.get(mod, 0)
        mod_count[mod] = mod_count.get(mod, 0) + 1

    return count
```

---

## Visual Explanations

### Prefix Sum Array
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  nums = [1, 2, 3, 4, 5]                                                     │
│                                                                             │
│  prefix = [0, 1, 3, 6, 10, 15]                                              │
│            ↑  ↑  ↑  ↑   ↑   ↑                                               │
│            │  │  │  │   │   └── sum(nums[0:5]) = 15                         │
│            │  │  │  │   └────── sum(nums[0:4]) = 10                         │
│            │  │  │  └────────── sum(nums[0:3]) = 6                          │
│            │  │  └────────────── sum(nums[0:2]) = 3                          │
│            │  └────────────────── sum(nums[0:1]) = 1                          │
│            └────────────────────── sum(nums[0:0]) = 0                          │
│                                                                             │
│  Range sum [2:4] = prefix[5] - prefix[2] = 15 - 3 = 12                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Subarray Sum = K
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  nums = [1, 1, 1], k = 2                                                    │
│                                                                             │
│  prefix sums: 0 → 1 → 2 → 3                                                 │
│                                                                             │
│  At prefix_sum = 2: look for 2 - 2 = 0 in map → found!                      │
│  At prefix_sum = 3: look for 3 - 2 = 1 in map → found!                      │
│                                                                             │
│  Subarrays: [1,1] (index 0-1), [1,1] (index 1-2)                            │
│  Answer: 2                                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2D Prefix Sum
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Original:           Prefix:                                                │
│  ┌───┬───┬───┐      ┌───┬───┬───┬───┐                                       │
│  │ 1 │ 2 │ 3 │      │ 0 │ 0 │ 0 │ 0 │                                       │
│  ├───┼───┼───┤      ├───┼───┼───┼───┤                                       │
│  │ 4 │ 5 │ 6 │      │ 0 │ 1 │ 3 │ 6 │                                       │
│  └───┴───┴───┘      ├───┼───┼───┼───┤                                       │
│                     │ 0 │ 5 │12 │21 │                                       │
│                     └───┴───┴───┴───┘                                       │
│                                                                             │
│  Region sum [0,0] to [1,1] = 21 - 6 - 0 + 0 = 15? No!                       │
│  Correct: prefix[2][2] - prefix[0][2] - prefix[2][0] + prefix[0][0]         │
│         = 12 - 0 - 0 + 0 = 12 ✓                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] Running Sum of 1d Array
- [ ] Range Sum Query - Immutable
- [ ] Find Pivot Index
- [ ] Product of Array Except Self

### Week 2: Hash Map Combination
- [ ] Subarray Sum Equals K
- [ ] Contiguous Array
- [ ] Binary Subarrays With Sum
- [ ] Subarray Sums Divisible by K

### Week 3: Advanced
- [ ] Range Sum Query 2D
- [ ] Continuous Subarray Sum
- [ ] Path Sum III
- [ ] Number of Submatrices That Sum to Target

---

## Common Mistakes to Avoid

1. **Off-by-one in range queries**
   ```python
   # Wrong: prefix[j] - prefix[i]
   # Correct: prefix[j + 1] - prefix[i]
   ```

2. **Forgetting to initialize prefix_count with {0: 1}**
   ```python
   # This handles subarrays starting from index 0
   prefix_count = {0: 1}
   ```

3. **Negative modulo in Python**
   ```python
   # Python handles this correctly, but be aware
   # -1 % 5 = 4 in Python (correct)
   # -1 % 5 = -1 in some languages (need adjustment)
   ```

4. **2D prefix sum formula**
   ```python
   # Inclusion-exclusion principle
   prefix[i][j] = matrix[i-1][j-1] + prefix[i-1][j] + prefix[i][j-1] - prefix[i-1][j-1]
   ```

---

## Complexity Reference

| Operation | Time | Space |
|-----------|------|-------|
| Build 1D prefix | O(n) | O(n) |
| Range sum query | O(1) | O(1) |
| Build 2D prefix | O(mn) | O(mn) |
| Region sum query | O(1) | O(1) |
| Subarray sum = k | O(n) | O(n) |

