# Arrays & Hashing - Complete Practice List

## Organized by Difficulty and Pattern

### Easy Problems

| # | Problem | Pattern | Key Technique |
|---|---------|---------|---------------|
| 1 | [Two Sum](https://leetcode.com/problems/two-sum/) | Hash Map | Complement lookup |
| 217 | [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/) | Hash Set | Existence check |
| 242 | [Valid Anagram](https://leetcode.com/problems/valid-anagram/) | Frequency Count | Character count |
| 169 | [Majority Element](https://leetcode.com/problems/majority-element/) | Boyer-Moore | Vote counting |
| 268 | [Missing Number](https://leetcode.com/problems/missing-number/) | Math/XOR | Sum formula or XOR |
| 448 | [Find All Numbers Disappeared](https://leetcode.com/problems/find-all-numbers-disappeared-in-an-array/) | Index Marking | In-place modification |
| 136 | [Single Number](https://leetcode.com/problems/single-number/) | XOR | Bit manipulation |
| 1480 | [Running Sum of 1d Array](https://leetcode.com/problems/running-sum-of-1d-array/) | Prefix Sum | Cumulative sum |
| 724 | [Find Pivot Index](https://leetcode.com/problems/find-pivot-index/) | Prefix Sum | Left/right sum |

### Medium Problems

| # | Problem | Pattern | Key Technique |
|---|---------|---------|---------------|
| 49 | [Group Anagrams](https://leetcode.com/problems/group-anagrams/) | Hash Map | Sorted key or count key |
| 347 | [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/) | Bucket Sort/Heap | Frequency counting |
| 238 | [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/) | Prefix/Suffix | Left/right products |
| 128 | [Longest Consecutive Sequence](https://leetcode.com/problems/longest-consecutive-sequence/) | Hash Set | Start of sequence |
| 560 | [Subarray Sum Equals K](https://leetcode.com/problems/subarray-sum-equals-k/) | Prefix Sum + Hash | Complement count |
| 36 | [Valid Sudoku](https://leetcode.com/problems/valid-sudoku/) | Hash Set | Row/col/box tracking |
| 271 | [Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/) | String | Length prefix |
| 442 | [Find All Duplicates](https://leetcode.com/problems/find-all-duplicates-in-an-array/) | Index Marking | Sign flipping |
| 287 | [Find the Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/) | Floyd's Cycle | Fast/slow pointers |
| 380 | [Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/) | Hash + Array | Index mapping |
| 525 | [Contiguous Array](https://leetcode.com/problems/contiguous-array/) | Prefix Sum | Balance counting |
| 523 | [Continuous Subarray Sum](https://leetcode.com/problems/continuous-subarray-sum/) | Prefix Sum | Modulo |
| 974 | [Subarray Sums Divisible by K](https://leetcode.com/problems/subarray-sums-divisible-by-k/) | Prefix Sum | Modulo counting |
| 454 | [4Sum II](https://leetcode.com/problems/4sum-ii/) | Hash Map | Two-pair grouping |
| 299 | [Bulls and Cows](https://leetcode.com/problems/bulls-and-cows/) | Frequency Count | Position matching |

### Hard Problems

| # | Problem | Pattern | Key Technique |
|---|---------|---------|---------------|
| 41 | [First Missing Positive](https://leetcode.com/problems/first-missing-positive/) | Index Marking | Cyclic sort |
| 76 | [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) | Sliding Window + Hash | Character tracking |
| 239 | [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) | Monotonic Deque | Max tracking |
| 30 | [Substring with Concatenation](https://leetcode.com/problems/substring-with-concatenation-of-all-words/) | Hash Map | Word window |
| 149 | [Max Points on a Line](https://leetcode.com/problems/max-points-on-a-line/) | Hash Map | Slope counting |
| 895 | [Maximum Frequency Stack](https://leetcode.com/problems/maximum-frequency-stack/) | Hash + Stack | Frequency buckets |
| 381 | [Insert Delete GetRandom O(1) - Duplicates](https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/) | Hash + Array | Index set |

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ARRAYS & HASHING PATTERNS                              │
│                                                                             │
│  TWO SUM (Hash Map Lookup):                                                 │
│  nums = [2, 7, 11, 15], target = 9                                          │
│                                                                             │
│  Step 1: seen = {}                                                          │
│  Step 2: num=2, complement=7, not in seen → seen = {2: 0}                   │
│  Step 3: num=7, complement=2, FOUND in seen! → return [0, 1]                │
│                                                                             │
│  Hash Map State:                                                            │
│  ┌─────────┬───────┐                                                        │
│  │   Key   │ Index │                                                        │
│  ├─────────┼───────┤                                                        │
│  │    2    │   0   │  ← complement of 7 found here!                         │
│  └─────────┴───────┘                                                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FREQUENCY COUNTING (Anagram Check):                                        │
│  s = "anagram", t = "nagaram"                                               │
│                                                                             │
│  Count for s:           Count for t:                                        │
│  ┌───┬───┐              ┌───┬───┐                                           │
│  │ a │ 3 │              │ a │ 3 │                                           │
│  │ n │ 1 │              │ n │ 1 │                                           │
│  │ g │ 1 │    ═══       │ g │ 1 │   ← Counts match = Anagram!               │
│  │ r │ 1 │              │ r │ 1 │                                           │
│  │ m │ 1 │              │ m │ 1 │                                           │
│  └───┴───┘              └───┴───┘                                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  CONTAINS DUPLICATE (Hash Set):                                             │
│  nums = [1, 2, 3, 1]                                                        │
│                                                                             │
│  Process:                                                                   │
│  seen = {}                                                                  │
│  1 → not in seen → seen = {1}                                               │
│  2 → not in seen → seen = {1, 2}                                            │
│  3 → not in seen → seen = {1, 2, 3}                                         │
│  1 → IN SEEN! → return True (duplicate found)                               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PREFIX SUM + HASH (Subarray Sum = K):                                      │
│  nums = [1, 2, 3], k = 3                                                    │
│                                                                             │
│  Array:     [1]  [2]  [3]                                                   │
│  Prefix:     1    3    6                                                    │
│                                                                             │
│  Hash: {0: 1}  ← Important! Empty prefix has sum 0                          │
│                                                                             │
│  i=0: prefix=1, need 1-3=-2, not found → hash={0:1, 1:1}                    │
│  i=1: prefix=3, need 3-3=0, FOUND! count=1 → hash={0:1, 1:1, 3:1}           │
│  i=2: prefix=6, need 6-3=3, FOUND! count=2 → hash={0:1, 1:1, 3:1, 6:1}      │
│                                                                             │
│  Subarrays with sum=3: [1,2] and [3]                                        │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INDEX MARKING (Find Duplicates In-Place):                                  │
│  nums = [4, 3, 2, 7, 8, 2, 3, 1]                                            │
│                                                                             │
│  Mark index (num-1) as negative when seen:                                  │
│  [4, 3, 2, 7, 8, 2, 3, 1]                                                   │
│   ↓                                                                         │
│  [-4, 3, 2,-7, 8, 2,-3,-1]  (marked indices 3,6,0,2,7)                      │
│                                                                             │
│  When we see 2 again, index 1 is already negative → duplicate!              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Patterns Summary

### 1. Hash Map for Lookup
```python
# Two Sum pattern
seen = {}
for i, num in enumerate(nums):
    complement = target - num
    if complement in seen:
        return [seen[complement], i]
    seen[num] = i
```

### 2. Frequency Counting
```python
from collections import Counter
freq = Counter(nums)
# or
freq = {}
for num in nums:
    freq[num] = freq.get(num, 0) + 1
```

### 3. Index Marking (In-place)
```python
# Mark seen numbers by negating at index
for num in nums:
    idx = abs(num) - 1
    nums[idx] = -abs(nums[idx])
```

### 4. Prefix Sum with Hash
```python
prefix_sum = 0
count = {0: 1}
for num in nums:
    prefix_sum += num
    # Check for target sum
    result += count.get(prefix_sum - target, 0)
    count[prefix_sum] = count.get(prefix_sum, 0) + 1
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] Contains Duplicate
- [ ] Valid Anagram
- [ ] Two Sum
- [ ] Group Anagrams
- [ ] Top K Frequent

### Week 2: Prefix Sum
- [ ] Running Sum
- [ ] Find Pivot Index
- [ ] Subarray Sum Equals K
- [ ] Contiguous Array
- [ ] Product Except Self

### Week 3: Advanced
- [ ] Longest Consecutive Sequence
- [ ] Insert Delete GetRandom
- [ ] First Missing Positive
- [ ] 4Sum II
- [ ] Maximum Frequency Stack

---

## Common Mistakes to Avoid

1. **Off-by-one errors** in index calculations
2. **Forgetting edge cases**: empty array, single element
3. **Not handling duplicates** properly
4. **Integer overflow** in sum calculations
5. **Modifying array** while iterating

---

## Time Complexity Reference

| Operation | Array | Hash Set | Hash Map |
|-----------|-------|----------|----------|
| Access | O(1) | N/A | O(1) avg |
| Search | O(n) | O(1) avg | O(1) avg |
| Insert | O(n) | O(1) avg | O(1) avg |
| Delete | O(n) | O(1) avg | O(1) avg |

