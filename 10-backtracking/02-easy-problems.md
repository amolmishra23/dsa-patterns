# Backtracking - Easy/Medium Problems

## Problem 1: Subsets (LC #78) - Medium

- [LeetCode](https://leetcode.com/problems/subsets/)

### Problem Statement
Return all possible subsets (power set) of an array with distinct integers.

### Video Explanation
- [NeetCode - Subsets](https://www.youtube.com/watch?v=REOH22Xwdkk)
- [Take U Forward - Subsets](https://www.youtube.com/watch?v=AxNNVECce8c)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BACKTRACKING DECISION TREE                                                 │
│                                                                             │
│  nums = [1, 2, 3]                                                          │
│                                                                             │
│  At each element, we have 2 choices: INCLUDE or EXCLUDE                    │
│                                                                             │
│                        []                                                   │
│                       /  \                                                  │
│               include 1   exclude 1                                         │
│                  /           \                                              │
│                [1]           []                                             │
│               /   \         /   \                                           │
│           +2      skip   +2     skip                                        │
│           /         \     /        \                                        │
│        [1,2]       [1]  [2]        []                                      │
│        /   \       / \  / \       / \                                       │
│      +3   skip  +3  skip...                                                │
│      /       \                                                              │
│   [1,2,3]  [1,2]                                                           │
│                                                                             │
│  All leaf nodes (and intermediate nodes) are valid subsets!                │
│                                                                             │
│  Result: [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]                 │
│                                                                             │
│  Pattern: For n elements → 2^n subsets                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Examples
```
Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

### Solution
```python
def subsets(nums: list[int]) -> list[list[int]]:
    """
    Generate all subsets using backtracking.

    Strategy:
    - At each position, choose to include or exclude element
    - Collect all paths as valid subsets

    Time: O(n * 2^n) - 2^n subsets, O(n) to copy each
    Space: O(n) for recursion depth
    """
    result = []

    def backtrack(start: int, path: list[int]):
        # Every path is a valid subset
        result.append(path[:])

        # Try adding each remaining element
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)  # Move to next index
            path.pop()  # Backtrack

    backtrack(0, [])
    return result


def subsets_iterative(nums: list[int]) -> list[list[int]]:
    """
    Iterative approach: build subsets incrementally.

    For each number, add it to all existing subsets.
    """
    result = [[]]

    for num in nums:
        # Add num to each existing subset
        result += [subset + [num] for subset in result]

    return result
```

### Edge Cases
- Empty input → return [[]]
- Single element → return [[element]]
- Duplicates → handle based on problem
- Large input → watch for TLE

---

## Problem 2: Subsets II (LC #90) - Medium

- [LeetCode](https://leetcode.com/problems/subsets-ii/)

### Problem Statement
Return all subsets, array may contain duplicates. Result must not contain duplicate subsets.

### Video Explanation
- [NeetCode - Subsets II](https://www.youtube.com/watch?v=Vn2v6ajA7U0)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  HANDLING DUPLICATES IN BACKTRACKING                                        │
│                                                                             │
│  nums = [1, 2, 2]                                                          │
│                                                                             │
│  Problem: Without handling, we get duplicate subsets:                      │
│  [1,2] appears twice (using 1st 2 and 2nd 2)                               │
│                                                                             │
│  Solution: SORT + SKIP duplicates at same level                            │
│                                                                             │
│  After sorting: [1, 2, 2]                                                  │
│                                                                             │
│  At index 1 (first 2): Include it → continue                               │
│  At index 2 (second 2): Skip if previous same value was skipped            │
│                                                                             │
│  Key rule: if nums[i] == nums[i-1] AND i > start, SKIP                     │
│                                                                             │
│  Valid subsets: [[], [1], [1,2], [1,2,2], [2], [2,2]]                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Examples
```
Input: nums = [1,2,2]
Output: [[],[1],[1,2],[1,2,2],[2],[2,2]]
```

### Solution
```python
def subsetsWithDup(nums: list[int]) -> list[list[int]]:
    """
    Subsets with duplicates handling.

    Strategy:
    - Sort to group duplicates
    - Skip duplicate elements at same level

    Time: O(n * 2^n)
    Space: O(n)
    """
    nums.sort()  # Sort to handle duplicates
    result = []

    def backtrack(start: int, path: list[int]):
        result.append(path[:])

        for i in range(start, len(nums)):
            # Skip duplicates at same level
            if i > start and nums[i] == nums[i - 1]:
                continue

            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

### Edge Cases
- Empty string → return [""]
- Single char → return [char]
- All same chars → single permutation
- Case sensitivity → per problem

---

## Problem 3: Permutations (LC #46) - Medium

- [LeetCode](https://leetcode.com/problems/permutations/)

### Problem Statement
Return all permutations of distinct integers.

### Video Explanation
- [NeetCode - Permutations](https://www.youtube.com/watch?v=s7AvT7cGdSo)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PERMUTATION = ALL ORDERINGS                                                │
│                                                                             │
│  nums = [1, 2, 3]                                                          │
│                                                                             │
│  Unlike subsets, ORDER matters and we use ALL elements                     │
│                                                                             │
│                        []                                                   │
│                    /   |   \                                                │
│                  1     2     3      (pick first element)                   │
│                / \   / \   / \                                              │
│              2   3  1   3  1   2    (pick second element)                  │
│              |   |  |   |  |   |                                           │
│              3   2  3   1  2   1    (pick third element)                   │
│                                                                             │
│  Result: [1,2,3] [1,3,2] [2,1,3] [2,3,1] [3,1,2] [3,2,1]                   │
│                                                                             │
│  Key: Track which elements are USED with a boolean array                   │
│  n elements → n! permutations                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Examples
```
Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

### Solution
```python
def permute(nums: list[int]) -> list[list[int]]:
    """
    Generate all permutations using backtracking.

    Strategy:
    - Track used elements
    - When path length equals nums length, we have a permutation

    Time: O(n * n!) - n! permutations, O(n) to copy each
    Space: O(n) for recursion and used set
    """
    result = []
    used = [False] * len(nums)

    def backtrack(path: list[int]):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            if used[i]:
                continue

            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False

    backtrack([])
    return result


def permute_swap(nums: list[int]) -> list[list[int]]:
    """
    Alternative: Swap-based approach.

    Swap each element to current position, recurse, swap back.
    """
    result = []

    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])
            return

        for i in range(start, len(nums)):
            nums[start], nums[i] = nums[i], nums[start]
            backtrack(start + 1)
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result
```

### Edge Cases
- n=0 → return []
- n=1 → return [[1]]
- Large n → factorial growth
- Duplicates not possible with 1..n

---

## Problem 4: Permutations II (LC #47) - Medium

- [LeetCode](https://leetcode.com/problems/permutations-ii/)

### Problem Statement
Return all unique permutations (array may have duplicates).

### Video Explanation
- [NeetCode - Permutations II](https://www.youtube.com/watch?v=qhBVWf0YafA)

### Examples
```
Input: nums = [1,1,2]
Output: [[1,1,2],[1,2,1],[2,1,1]]
```

### Solution
```python
def permuteUnique(nums: list[int]) -> list[list[int]]:
    """
    Unique permutations with duplicates.

    Strategy:
    - Sort to group duplicates
    - Only use duplicate if previous duplicate was used

    Time: O(n * n!)
    Space: O(n)
    """
    nums.sort()
    result = []
    used = [False] * len(nums)

    def backtrack(path: list[int]):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            if used[i]:
                continue

            # Skip duplicate if previous duplicate not used
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue

            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False

    backtrack([])
    return result
```

### Edge Cases
- Empty set → return [[]]
- Single element → [[],[el]]
- All same → depends on uniqueness
- Large set → 2^n subsets

---

## Problem 5: Combination Sum (LC #39) - Medium

- [LeetCode](https://leetcode.com/problems/combination-sum/)

### Problem Statement
Find combinations that sum to target (can reuse elements).

### Video Explanation
- [NeetCode - Combination Sum](https://www.youtube.com/watch?v=GBKI9VSKdGg)

### Examples
```
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
```

### Solution
```python
def combinationSum(candidates: list[int], target: int) -> list[list[int]]:
    """
    Find combinations summing to target (with repetition).

    Strategy:
    - Try each candidate, can use same candidate again
    - Stop when sum exceeds target

    Time: O(n^(target/min)) - branching factor n, depth target/min
    Space: O(target/min) for recursion depth
    """
    result = []

    def backtrack(start: int, path: list[int], remaining: int):
        if remaining == 0:
            result.append(path[:])
            return

        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            path.append(candidates[i])
            # Same index i allowed (can reuse)
            backtrack(i, path, remaining - candidates[i])
            path.pop()

    backtrack(0, [], target)
    return result
```

### Edge Cases
- target=0 → return [[]]
- No valid combination → return []
- Single candidate equals target → [[candidate]]
- Negative numbers → adjust logic

---

## Problem 6: Combination Sum II (LC #40) - Medium

- [LeetCode](https://leetcode.com/problems/combination-sum-ii/)

### Problem Statement
Find combinations summing to target (each element used once).

### Video Explanation
- [NeetCode - Combination Sum II](https://www.youtube.com/watch?v=rSA3t6BDDwg)

### Examples
```
Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: [[1,1,6],[1,2,5],[1,7],[2,6]]
```

### Solution
```python
def combinationSum2(candidates: list[int], target: int) -> list[list[int]]:
    """
    Combinations without repetition, handling duplicates.

    Strategy:
    - Sort to handle duplicates
    - Move to next index (no reuse)
    - Skip duplicates at same level

    Time: O(2^n)
    Space: O(n)
    """
    candidates.sort()
    result = []

    def backtrack(start: int, path: list[int], remaining: int):
        if remaining == 0:
            result.append(path[:])
            return

        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            # Skip duplicates at same level
            if i > start and candidates[i] == candidates[i - 1]:
                continue

            # Pruning: if current > remaining, all following are too
            if candidates[i] > remaining:
                break

            path.append(candidates[i])
            backtrack(i + 1, path, remaining - candidates[i])
            path.pop()

    backtrack(0, [], target)
    return result
```

### Edge Cases
- k > n → return []
- k = 0 → return [[]]
- k = n → return [[1..n]]
- Large n, small k → manageable

---

## Problem 7: Combination Sum III (LC #216) - Medium

- [LeetCode](https://leetcode.com/problems/combination-sum-iii/)

### Problem Statement
Find k numbers from 1-9 that sum to n (each used once).

### Video Explanation
- [NeetCode - Combination Sum III](https://www.youtube.com/watch?v=Bj8KHFbKDFw)

### Examples
```
Input: k = 3, n = 7
Output: [[1,2,4]]

Input: k = 3, n = 9
Output: [[1,2,6],[1,3,5],[2,3,4]]
```

### Solution
```python
def combinationSum3(k: int, n: int) -> list[list[int]]:
    """
    Find k numbers from 1-9 summing to n.

    Time: O(C(9,k) * k)
    Space: O(k)
    """
    result = []

    def backtrack(start: int, path: list[int], remaining: int):
        if len(path) == k:
            if remaining == 0:
                result.append(path[:])
            return

        # Pruning
        if remaining <= 0:
            return

        for num in range(start, 10):
            # Pruning: not enough numbers left
            if 10 - num < k - len(path):
                break

            path.append(num)
            backtrack(num + 1, path, remaining - num)
            path.pop()

    backtrack(1, [], n)
    return result
```

### Edge Cases
- Empty board → return []
- No valid path → return []
- Single cell word → check cell
- Word longer than cells → impossible

---

## Problem 8: Letter Combinations of Phone Number (LC #17) - Medium

- [LeetCode](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)

### Problem Statement
Return all letter combinations for phone number digits.

### Video Explanation
- [NeetCode - Letter Combinations of Phone Number](https://www.youtube.com/watch?v=0snEunUacZY)

### Examples
```
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

### Solution
```python
def letterCombinations(digits: str) -> list[str]:
    """
    Generate all letter combinations for phone digits.

    Time: O(4^n * n) - up to 4 letters per digit
    Space: O(n) for recursion
    """
    if not digits:
        return []

    phone_map = {
        '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
        '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'
    }

    result = []

    def backtrack(index: int, path: list[str]):
        if index == len(digits):
            result.append(''.join(path))
            return

        for letter in phone_map[digits[index]]:
            path.append(letter)
            backtrack(index + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

### Edge Cases
- digits="" → return []
- Single digit → return its letters
- digits="1" → return [] (no letters)
- Max 4 digits → bounded

---

## Problem 9: Generate Parentheses (LC #22) - Medium

- [LeetCode](https://leetcode.com/problems/generate-parentheses/)

### Problem Statement
Generate all valid combinations of n pairs of parentheses.

### Video Explanation
- [NeetCode - Generate Parentheses](https://www.youtube.com/watch?v=s9fokUqJ76A)

### Examples
```
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
```

### Solution
```python
def generateParenthesis(n: int) -> list[str]:
    """
    Generate valid parentheses combinations.

    Strategy:
    - Track open and close counts
    - Can add '(' if open < n
    - Can add ')' if close < open

    Time: O(4^n / sqrt(n)) - Catalan number
    Space: O(n)
    """
    result = []

    def backtrack(path: list[str], open_count: int, close_count: int):
        if len(path) == 2 * n:
            result.append(''.join(path))
            return

        # Can add '(' if we haven't used all
        if open_count < n:
            path.append('(')
            backtrack(path, open_count + 1, close_count)
            path.pop()

        # Can add ')' if it won't make invalid
        if close_count < open_count:
            path.append(')')
            backtrack(path, open_count, close_count + 1)
            path.pop()

    backtrack([], 0, 0)
    return result
```

### Edge Cases
- Empty string → return [""]
- Single char → return [char]
- All same → one result
- Palindrome input → multiple valid

---

## Problem 10: N-Queens (LC #51) - Hard

- [LeetCode](https://leetcode.com/problems/n-queens/)

### Problem Statement
Place n queens on n×n board so no two attack each other.

### Video Explanation
- [NeetCode - N-Queens](https://www.youtube.com/watch?v=Ph95IHmRp5M)

### Solution
```python
def solveNQueens(n: int) -> list[list[str]]:
    """
    Solve N-Queens using backtracking.

    Strategy:
    - Place queens row by row
    - Track columns and diagonals that are attacked

    Time: O(n!)
    Space: O(n)
    """
    result = []

    # Track attacked columns and diagonals
    cols = set()
    diag1 = set()  # row - col (top-left to bottom-right)
    diag2 = set()  # row + col (top-right to bottom-left)

    def backtrack(row: int, queens: list[int]):
        if row == n:
            # Build board representation
            board = []
            for col in queens:
                board.append('.' * col + 'Q' + '.' * (n - col - 1))
            result.append(board)
            return

        for col in range(n):
            if col in cols or (row - col) in diag1 or (row + col) in diag2:
                continue

            # Place queen
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            queens.append(col)

            backtrack(row + 1, queens)

            # Remove queen
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)
            queens.pop()

    backtrack(0, [])
    return result
```

### Edge Cases
- n=0 → return [""]
- n=1 → return ["()"] 
- Large n → Catalan number growth
- Always n open, n close

---

## Problem 11: Sudoku Solver (LC #37) - Hard

- [LeetCode](https://leetcode.com/problems/sudoku-solver/)

### Problem Statement
Solve a Sudoku puzzle.

### Video Explanation
- [NeetCode - Sudoku Solver](https://www.youtube.com/watch?v=FWAIf_EVUKE)

### Solution
```python
def solveSudoku(board: list[list[str]]) -> None:
    """
    Solve Sudoku using backtracking.

    Strategy:
    - Find empty cell
    - Try digits 1-9
    - Check validity, recurse
    - Backtrack if no solution

    Time: O(9^(empty cells))
    Space: O(81) for recursion
    """
    def is_valid(row: int, col: int, num: str) -> bool:
        # Check row
        if num in board[row]:
            return False

        # Check column
        if num in [board[i][col] for i in range(9)]:
            return False

        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if board[i][j] == num:
                    return False

        return True

    def solve() -> bool:
        # Find empty cell
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    for num in '123456789':
                        if is_valid(i, j, num):
                            board[i][j] = num

                            if solve():
                                return True

                            board[i][j] = '.'  # Backtrack

                    return False  # No valid number

        return True  # All cells filled

    solve()
```

### Edge Cases
- Empty input → return [[]]
- All same → single combination
- Target 0 → return [[]]
- No valid → return []

---

## Summary: Backtracking Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Subsets | Include/exclude each | O(2^n) |
| 2 | Subsets II | Sort + skip duplicates | O(2^n) |
| 3 | Permutations | Used array tracking | O(n!) |
| 4 | Permutations II | Sort + skip unused dup | O(n!) |
| 5 | Combination Sum | Reuse allowed | O(n^(t/m)) |
| 6 | Combination Sum II | No reuse + skip dups | O(2^n) |
| 7 | Combination Sum III | Fixed k, range 1-9 | O(C(9,k)) |
| 8 | Letter Combinations | Multiple choices per position | O(4^n) |
| 9 | Generate Parentheses | Open/close count tracking | O(4^n/√n) |
| 10 | N-Queens | Column + diagonal tracking | O(n!) |
| 11 | Sudoku | Constraint checking | O(9^cells) |

---

## Practice More Problems

- [ ] LC #79 - Word Search
- [ ] LC #93 - Restore IP Addresses
- [ ] LC #131 - Palindrome Partitioning
- [ ] LC #212 - Word Search II
- [ ] LC #291 - Word Pattern II

