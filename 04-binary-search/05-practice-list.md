# Binary Search - Complete Practice List

## Organized by Pattern and Difficulty

### Pattern 1: Classic Binary Search

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 704 | [Binary Search](https://leetcode.com/problems/binary-search/) | Easy | Basic template |
| 374 | [Guess Number Higher or Lower](https://leetcode.com/problems/guess-number-higher-or-lower/) | Easy | Search space |
| 278 | [First Bad Version](https://leetcode.com/problems/first-bad-version/) | Easy | Find boundary |
| 35 | [Search Insert Position](https://leetcode.com/problems/search-insert-position/) | Easy | Lower bound |
| 69 | [Sqrt(x)](https://leetcode.com/problems/sqrtx/) | Easy | Search on answer |

### Pattern 2: Find Boundary (First/Last)

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 34 | [Find First and Last Position](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/) | Medium | Two binary searches |
| 744 | [Find Smallest Letter Greater](https://leetcode.com/problems/find-smallest-letter-greater-than-target/) | Easy | Upper bound |
| 1150 | [Check Majority Element](https://leetcode.com/problems/check-if-a-number-is-majority-element-in-a-sorted-array/) | Easy | Count with bounds |
| 2089 | [Find Target Indices](https://leetcode.com/problems/find-target-indices-after-sorting-array/) | Easy | Count approach |

### Pattern 3: Rotated/Modified Arrays

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 33 | [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/) | Medium | Identify sorted half |
| 81 | [Search Rotated Array II](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/) | Medium | Handle duplicates |
| 153 | [Find Minimum in Rotated Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/) | Medium | Compare with right |
| 154 | [Find Minimum Rotated II](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array-ii/) | Hard | Handle duplicates |
| 162 | [Find Peak Element](https://leetcode.com/problems/find-peak-element/) | Medium | Follow upward slope |
| 852 | [Peak Index in Mountain](https://leetcode.com/problems/peak-index-in-a-mountain-array/) | Medium | Find peak |

### Pattern 4: Search on Answer

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 875 | [Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/) | Medium | Min speed to finish |
| 1011 | [Capacity To Ship Packages](https://leetcode.com/problems/capacity-to-ship-packages-within-d-days/) | Medium | Min capacity |
| 410 | [Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/) | Hard | Min max sum |
| 1482 | [Min Days for Bouquets](https://leetcode.com/problems/minimum-number-of-days-to-make-m-bouquets/) | Medium | Min days |
| 1283 | [Find Smallest Divisor](https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/) | Medium | Min divisor |
| 774 | [Min Max Distance Gas Station](https://leetcode.com/problems/minimize-max-distance-to-gas-station/) | Hard | Binary on distance |

### Pattern 5: 2D Matrix Search

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 74 | [Search a 2D Matrix](https://leetcode.com/problems/search-a-2d-matrix/) | Medium | Treat as 1D |
| 240 | [Search 2D Matrix II](https://leetcode.com/problems/search-a-2d-matrix-ii/) | Medium | Start from corner |
| 378 | [Kth Smallest in Sorted Matrix](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/) | Medium | BS + count |

### Pattern 6: Advanced

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 4 | [Median of Two Sorted Arrays](https://leetcode.com/problems/median-of-two-sorted-arrays/) | Hard | Partition arrays |
| 719 | [Find K-th Smallest Pair Distance](https://leetcode.com/problems/find-k-th-smallest-pair-distance/) | Hard | BS + count pairs |
| 668 | [Kth Smallest in Multiplication Table](https://leetcode.com/problems/kth-smallest-number-in-multiplication-table/) | Hard | BS + count |

---

## Essential Templates

### 1. Basic Binary Search
```python
def binary_search(nums, target):
    """Find exact target. Returns index or -1."""
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
```

### 2. Lower Bound (First >= target)
```python
def lower_bound(nums, target):
    """Find first index where nums[i] >= target."""
    left, right = 0, len(nums)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    
    return left
```

### 3. Upper Bound (First > target)
```python
def upper_bound(nums, target):
    """Find first index where nums[i] > target."""
    left, right = 0, len(nums)
    
    while left < right:
        mid = left + (right - left) // 2
        
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid
    
    return left
```

### 4. Search on Answer (Minimize)
```python
def search_on_answer_min(lo, hi, is_feasible):
    """Find minimum value where is_feasible returns True."""
    while lo < hi:
        mid = lo + (hi - lo) // 2
        
        if is_feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    
    return lo
```

### 5. Search on Answer (Maximize)
```python
def search_on_answer_max(lo, hi, is_feasible):
    """Find maximum value where is_feasible returns True."""
    while lo < hi:
        mid = lo + (hi - lo + 1) // 2  # Note: round up
        
        if is_feasible(mid):
            lo = mid
        else:
            hi = mid - 1
    
    return lo
```

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BINARY SEARCH PATTERNS                                   │
│                                                                             │
│  BASIC SEARCH (find exact):                                                 │
│  [1, 3, 5, 7, 9, 11, 13], target = 7                                       │
│   L        M          R   → 7 == 7 ✓ return mid                             │
│                                                                             │
│  LOWER BOUND (first >= 6):                                                  │
│  [1, 3, 5, 7, 9, 11, 13]                                                   │
│   L        M          R   → 7 >= 6, right = mid                             │
│   L     M  R              → 5 < 6, left = mid + 1                           │
│         L=R               → return 3 (index of 7)                           │
│                                                                             │
│  ROTATED ARRAY:                                                             │
│  [4, 5, 6, 7, 0, 1, 2], target = 0                                         │
│   L        M        R   → Left half [4,5,6,7] sorted                        │
│                         → 0 not in [4,7], go right                          │
│            L  M     R   → Found!                                            │
│                                                                             │
│  SEARCH ON ANSWER (Koko Bananas):                                           │
│  piles = [3, 6, 7, 11], h = 8                                              │
│  Speed range: [1, 11]                                                       │
│  mid=6: 1+1+2+2 = 6 hrs ≤ 8 ✓ → try smaller                                │
│  mid=3: 1+2+3+4 = 10 hrs > 8 ✗ → need larger                               │
│  mid=4: 1+2+2+3 = 8 hrs ≤ 8 ✓ → answer = 4                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## When to Use Each Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BINARY SEARCH DECISION GUIDE                             │
│                                                                             │
│  SORTED ARRAY:                                                              │
│  • Find exact value → Basic binary search                                   │
│  • Find first/last occurrence → Lower/upper bound                           │
│  • Find insert position → Lower bound                                       │
│                                                                             │
│  ROTATED ARRAY:                                                             │
│  • No duplicates → Identify sorted half                                     │
│  • With duplicates → Handle equal case carefully                            │
│  • Find minimum → Compare with rightmost element                            │
│                                                                             │
│  SEARCH ON ANSWER:                                                          │
│  • "Minimum X such that..." → Binary search, check feasibility              │
│  • "Maximum X such that..." → Binary search, check feasibility              │
│  • Key: Feasibility must be monotonic                                       │
│                                                                             │
│  2D MATRIX:                                                                 │
│  • Fully sorted → Treat as 1D array                                        │
│  • Rows/cols sorted separately → Start from corner                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] Binary Search (LC #704)
- [ ] Search Insert Position (LC #35)
- [ ] First Bad Version (LC #278)
- [ ] Sqrt(x) (LC #69)
- [ ] Find First and Last Position (LC #34)

### Week 2: Modified Arrays
- [ ] Search in Rotated Sorted Array (LC #33)
- [ ] Find Minimum in Rotated Sorted Array (LC #153)
- [ ] Find Peak Element (LC #162)
- [ ] Search a 2D Matrix (LC #74)
- [ ] Search a 2D Matrix II (LC #240)

### Week 3: Search on Answer
- [ ] Koko Eating Bananas (LC #875)
- [ ] Capacity To Ship Packages (LC #1011)
- [ ] Split Array Largest Sum (LC #410)
- [ ] Minimum Days for Bouquets (LC #1482)

### Week 4: Advanced
- [ ] Median of Two Sorted Arrays (LC #4)
- [ ] Kth Smallest in Sorted Matrix (LC #378)
- [ ] Find K-th Smallest Pair Distance (LC #719)

---

## Common Mistakes

1. **Integer overflow in mid calculation**
   ```python
   # Wrong (can overflow in other languages)
   mid = (left + right) // 2
   
   # Correct
   mid = left + (right - left) // 2
   ```

2. **Infinite loop with wrong boundary**
   ```python
   # If using left < right:
   # - right = mid (not mid - 1) when going left
   # - left = mid + 1 when going right
   
   # If using left <= right:
   # - right = mid - 1 when going left
   # - left = mid + 1 when going right
   ```

3. **Off-by-one in rotated array**
   ```python
   # Use <= not < for left half check
   if nums[left] <= nums[mid]:  # Left half sorted
   ```

4. **Wrong search space bounds**
   ```python
   # Search on answer: think carefully about min/max
   # Koko: [1, max(piles)] not [0, max(piles)]
   # Ship: [max(weights), sum(weights)]
   ```

---

## Complexity Reference

| Pattern | Time | Space |
|---------|------|-------|
| Basic binary search | O(log n) | O(1) |
| Lower/upper bound | O(log n) | O(1) |
| Rotated array | O(log n) | O(1) |
| Rotated with duplicates | O(n) worst | O(1) |
| Search on answer | O(n log S) | O(1) |
| 2D matrix (1D treat) | O(log(mn)) | O(1) |
| 2D matrix (corner) | O(m + n) | O(1) |
| Median two arrays | O(log min(m,n)) | O(1) |
