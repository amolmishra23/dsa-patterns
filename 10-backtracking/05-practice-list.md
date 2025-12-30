# Backtracking - Complete Practice List

## Organized by Pattern and Difficulty

### Pattern 1: Subsets/Combinations

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 78 | [Subsets](https://leetcode.com/problems/subsets/) | Medium | Include/exclude each element |
| 90 | [Subsets II](https://leetcode.com/problems/subsets-ii/) | Medium | Skip duplicates after sorting |
| 77 | [Combinations](https://leetcode.com/problems/combinations/) | Medium | Choose k from n |
| 39 | [Combination Sum](https://leetcode.com/problems/combination-sum/) | Medium | Reuse allowed |
| 40 | [Combination Sum II](https://leetcode.com/problems/combination-sum-ii/) | Medium | No reuse, skip duplicates |
| 216 | [Combination Sum III](https://leetcode.com/problems/combination-sum-iii/) | Medium | Fixed count, no reuse |

### Pattern 2: Permutations

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 46 | [Permutations](https://leetcode.com/problems/permutations/) | Medium | Track used elements |
| 47 | [Permutations II](https://leetcode.com/problems/permutations-ii/) | Medium | Skip duplicates |
| 31 | [Next Permutation](https://leetcode.com/problems/next-permutation/) | Medium | Find next lexicographic |

### Pattern 3: Grid Search

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 79 | [Word Search](https://leetcode.com/problems/word-search/) | Medium | DFS with visited tracking |
| 212 | [Word Search II](https://leetcode.com/problems/word-search-ii/) | Hard | Trie + DFS optimization |

### Pattern 4: Classic Constraint Problems

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 51 | [N-Queens](https://leetcode.com/problems/n-queens/) | Hard | Track cols, diagonals |
| 37 | [Sudoku Solver](https://leetcode.com/problems/sudoku-solver/) | Hard | Row/col/box constraints |
| 131 | [Palindrome Partitioning](https://leetcode.com/problems/palindrome-partitioning/) | Medium | Partition + palindrome check |
| 93 | [Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/) | Medium | 4 parts, valid range |
| 17 | [Letter Combinations of Phone Number](https://leetcode.com/problems/letter-combinations-of-a-phone-number/) | Medium | Map digits to letters |
| 22 | [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/) | Medium | Track open/close count |

## Templates

```python
# Subsets Template
def subsets(nums):
    result = []
    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()
    backtrack(0, [])
    return result

# Permutations Template
def permutations(nums):
    result = []
    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return
        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False
    backtrack([], [False] * len(nums))
    return result

# Combination Sum Template
def combination_sum(candidates, target):
    result = []
    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        if remaining < 0:
            return
        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])
            path.pop()
    backtrack(0, [], target)
    return result
```

## Key Insights
- Subsets: include/exclude each element
- Permutations: try all unused elements
- Combinations: start from current index to avoid duplicates
- Pruning: skip invalid states early

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKTRACKING DECISION TREE                               │
│                                                                             │
│  SUBSETS [1,2,3]:                                                           │
│                        []                                                   │
│                    /        \                                               │
│               [1]            []                                             │
│              /    \        /    \                                           │
│          [1,2]   [1]    [2]     []                                          │
│          /  \    /  \   /  \   /  \                                         │
│      [1,2,3][1,2][1,3][1][2,3][2][3][]                                      │
│                                                                             │
│  Result: [],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PERMUTATIONS [1,2,3]:                                                      │
│                        start                                                │
│                    /     |     \                                            │
│                  1       2       3     (pick first)                         │
│                / \     / \     / \                                          │
│               2   3   1   3   1   2    (pick second)                        │
│               |   |   |   |   |   |                                         │
│               3   2   3   1   2   1    (pick third)                         │
│                                                                             │
│  Result: [1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  COMBINATION SUM [2,3,6,7], target=7:                                       │
│                                                                             │
│                      start(target=7)                                        │
│                    /    |    \    \                                         │
│                  2      3     6    7                                        │
│               (t=5)  (t=4) (t=1) (t=0)✓                                     │
│              / | \    / \    X                                              │
│             2  3  6  3   6                                                  │
│          (t=3)(t=2)(X)(t=1)(X)                                              │
│           /|\  |                                                            │
│          2 3 6 3                                                            │
│        (1)(0)✓                                                              │
│         |                                                                   │
│         2                                                                   │
│        (X)                                                                  │
│                                                                             │
│  Result: [2,2,3], [7]                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] LC 78: Subsets - Core pattern
- [ ] LC 46: Permutations - Core pattern
- [ ] LC 77: Combinations - Core pattern
- [ ] LC 39: Combination Sum - With repetition
- [ ] LC 17: Letter Combinations of Phone Number

### Week 2: Handling Duplicates
- [ ] LC 90: Subsets II - Skip duplicates
- [ ] LC 47: Permutations II - Skip duplicates
- [ ] LC 40: Combination Sum II - No reuse + skip duplicates
- [ ] LC 131: Palindrome Partitioning
- [ ] LC 93: Restore IP Addresses

### Week 3: Advanced & Grid
- [ ] LC 79: Word Search - Grid backtracking
- [ ] LC 212: Word Search II - Trie optimization
- [ ] LC 51: N-Queens - Constraint tracking
- [ ] LC 37: Sudoku Solver - Multiple constraints
- [ ] LC 282: Expression Add Operators

---

## Common Mistakes

### 1. Forgetting to Undo State (Backtrack)
```python
# WRONG - state not restored
def backtrack(path):
    path.append(num)
    backtrack(path)
    # Missing: path.pop()

# CORRECT
def backtrack(path):
    path.append(num)
    backtrack(path)
    path.pop()  # Restore state
```

### 2. Not Handling Duplicates
```python
# WRONG - generates duplicate subsets for [1,1,2]
def subsets(nums):
    for i in range(start, len(nums)):
        backtrack(i + 1, path + [nums[i]])

# CORRECT - skip duplicates
def subsets(nums):
    nums.sort()  # Sort first!
    for i in range(start, len(nums)):
        if i > start and nums[i] == nums[i-1]:
            continue  # Skip duplicate
        backtrack(i + 1, path + [nums[i]])
```

### 3. Wrong Base Case
```python
# WRONG - missing base case for permutations
def permute(path):
    for num in nums:
        permute(path + [num])  # Infinite recursion!

# CORRECT
def permute(path):
    if len(path) == len(nums):  # Base case
        result.append(path[:])
        return
    for num in nums:
        if num not in path:
            permute(path + [num])
```

### 4. Modifying List While Iterating
```python
# WRONG
def backtrack(candidates):
    for c in candidates:
        candidates.remove(c)  # Don't modify during iteration!
        backtrack(candidates)

# CORRECT - use index or copy
def backtrack(start, candidates):
    for i in range(start, len(candidates)):
        backtrack(i + 1, candidates)  # Use index
```

### 5. Not Copying Path to Result
```python
# WRONG - all results point to same list
result.append(path)  # path will change!

# CORRECT - make a copy
result.append(path[:])  # Shallow copy
result.append(list(path))  # Alternative
```

---

## Complexity Reference

| Pattern | Time | Space | Key Factor |
|---------|------|-------|------------|
| Subsets | O(n * 2^n) | O(n) | 2 choices per element |
| Permutations | O(n * n!) | O(n) | n! arrangements |
| Combinations | O(k * C(n,k)) | O(k) | Choose k from n |
| N-Queens | O(n!) | O(n) | Pruning reduces |
| Word Search | O(m*n * 4^L) | O(L) | L = word length |

---

## Pattern Recognition

| See This | Think This |
|----------|------------|
| "All subsets" | Subset pattern, include/exclude |
| "All permutations" | Permutation pattern, used array |
| "All combinations of size k" | Combination pattern, start index |
| "Sum equals target" | Combination sum, track remaining |
| "Partition into parts" | Palindrome partition pattern |
| "Place items on grid" | N-Queens, constraint sets |
| "Find word in grid" | Word search, 4-direction DFS |
