# Backtracking - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE BACKTRACKING                                 │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "All combinations"                                                       │
│  ✓ "All permutations"                                                       │
│  ✓ "All subsets"                                                            │
│  ✓ "Generate all valid..."                                                  │
│  ✓ "N-Queens" / "Sudoku"                                                    │
│  ✓ "Word search"                                                            │
│  ✓ "Partition into..."                                                      │
│                                                                             │
│  Key insight: Systematically explore all possibilities,                     │
│               prune invalid paths early                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Recursion basics
- [ ] Tree traversal (backtracking explores a decision tree)
- [ ] Time complexity of exponential algorithms

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKTRACKING MEMORY MAP                                  │
│                                                                             │
│                    ┌─────────────┐                                          │
│         ┌─────────│BACKTRACKING │─────────┐                                 │
│         │         └─────────────┘         │                                 │
│         ▼                                 ▼                                 │
│  ┌─────────────┐                   ┌─────────────┐                          │
│  │ ENUMERATION │                   │ CONSTRAINT  │                          │
│  │  PROBLEMS   │                   │ SATISFACTION│                          │
│  └──────┬──────┘                   └──────┬──────┘                          │
│         │                                 │                                 │
│    ┌────┴────┐                      ┌─────┴─────┐                           │
│    ▼         ▼                      ▼           ▼                           │
│ ┌──────┐ ┌──────┐               ┌──────┐   ┌──────┐                        │
│ │Subsets│ │Permu-│              │N-Queen│  │Sudoku│                        │
│ │Combos│ │tations│              │       │  │      │                        │
│ └──────┘ └──────┘               └──────┘   └──────┘                        │
│                                                                             │
│  Related Patterns:                                                          │
│  • DFS - Backtracking is DFS on decision tree                               │
│  • Recursion - Core technique for backtracking                              │
│  • DP - Sometimes backtracking + memoization = DP                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKTRACKING DECISION TREE                               │
│                                                                             │
│  Need to generate ALL solutions?                                            │
│       │                                                                     │
│       ├── YES → Backtracking is likely needed                               │
│       │         Use pruning to optimize                                     │
│       │                                                                     │
│       └── NO → Need just one solution or count?                             │
│                    │                                                        │
│                    ├── Count → Consider DP if overlapping subproblems       │
│                    │                                                        │
│                    └── One solution → Backtracking with early return        │
│                                                                             │
│  TEMPLATE SELECTION:                                                        │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ Problem Type        │ Key Difference                               │     │
│  ├─────────────────────┼──────────────────────────────────────────────┤     │
│  │ Subsets             │ Include or exclude each element              │     │
│  │ Combinations        │ Choose k elements, order doesn't matter      │     │
│  │ Permutations        │ All orderings, order matters                 │     │
│  │ Constraint problems │ Add validity check before recursing          │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concept

Backtracking = DFS + Pruning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKTRACKING VISUALIZATION                               │
│                                                                             │
│  Generate all subsets of [1, 2, 3]:                                         │
│                                                                             │
│                         []                                                  │
│                        /  \                                                 │
│                 include 1  exclude 1                                        │
│                      /       \                                              │
│                    [1]        []                                            │
│                   /   \      /   \                                          │
│              inc 2  exc 2  inc 2  exc 2                                     │
│               /       \     /       \                                       │
│           [1,2]      [1]  [2]       []                                      │
│           /   \      / \   / \      / \                                     │
│      [1,2,3] [1,2] [1,3] [1] [2,3] [2] [3] []                              │
│                                                                             │
│  At each node: Make a choice, explore, undo choice (backtrack)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Backtracking Template

```python
def backtrack(path: list, choices: list) -> None:
    """
    Universal backtracking template.

    1. Check if current path is a valid solution
    2. If yes, add to results
    3. For each available choice:
       a. Make the choice (add to path)
       b. Recurse with updated state
       c. Undo the choice (backtrack)
    """
    # BASE CASE: Check if we have a complete solution
    if is_solution(path):
        result.append(path[:])  # Add COPY of path
        return

    # RECURSIVE CASE: Try each available choice
    for choice in choices:
        # Pruning: Skip invalid choices early
        if not is_valid(choice, path):
            continue

        # 1. MAKE CHOICE
        path.append(choice)

        # 2. EXPLORE (recurse)
        backtrack(path, remaining_choices(choice))

        # 3. UNDO CHOICE (backtrack)
        path.pop()
```

---

## Common Backtracking Problems

### Problem 1: Subsets (LC #78)

```python
def subsets(nums: list[int]) -> list[list[int]]:
    """
    Generate all subsets of nums.

    For each element, we have two choices:
    - Include it in the current subset
    - Exclude it from the current subset

    Time: O(n * 2^n) - 2^n subsets, O(n) to copy each
    Space: O(n) - recursion depth
    """
    result = []

    def backtrack(index: int, current: list[int]):
        """
        Build subsets starting from index.

        Args:
            index: Current position in nums
            current: Current subset being built
        """
        # Every path is a valid subset
        result.append(current[:])

        # Try adding each remaining element
        for i in range(index, len(nums)):
            # Include nums[i]
            current.append(nums[i])

            # Recurse with next index
            backtrack(i + 1, current)

            # Backtrack: remove nums[i]
            current.pop()

    backtrack(0, [])
    return result
```

### Problem 2: Permutations (LC #46)

```python
def permutations(nums: list[int]) -> list[list[int]]:
    """
    Generate all permutations of nums.

    At each position, try each unused element.

    Time: O(n * n!) - n! permutations, O(n) to copy each
    Space: O(n) - recursion depth
    """
    result = []
    used = [False] * len(nums)  # Track which elements are used

    def backtrack(current: list[int]):
        """Build permutations by choosing unused elements."""
        # Base case: permutation is complete
        if len(current) == len(nums):
            result.append(current[:])
            return

        # Try each unused element
        for i in range(len(nums)):
            if used[i]:
                continue  # Skip used elements

            # Choose
            used[i] = True
            current.append(nums[i])

            # Explore
            backtrack(current)

            # Unchoose (backtrack)
            used[i] = False
            current.pop()

    backtrack([])
    return result


def permutations_swap(nums: list[int]) -> list[list[int]]:
    """
    Alternative using swapping (more space efficient).
    """
    result = []

    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])
            return

        for i in range(start, len(nums)):
            # Swap to put nums[i] at position start
            nums[start], nums[i] = nums[i], nums[start]

            # Recurse for next position
            backtrack(start + 1)

            # Swap back
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result
```

### Problem 3: Combinations (LC #77)

```python
def combinations(n: int, k: int) -> list[list[int]]:
    """
    Generate all combinations of k numbers from 1 to n.

    Time: O(k * C(n,k))
    Space: O(k)
    """
    result = []

    def backtrack(start: int, current: list[int]):
        # Base case: combination is complete
        if len(current) == k:
            result.append(current[:])
            return

        # Pruning: not enough elements left
        # Need (k - len(current)) more elements
        # Have (n - start + 1) elements left
        if n - start + 1 < k - len(current):
            return

        # Try each element from start to n
        for i in range(start, n + 1):
            current.append(i)
            backtrack(i + 1, current)
            current.pop()

    backtrack(1, [])
    return result
```

### Problem 4: Combination Sum (LC #39)

```python
def combinationSum(candidates: list[int], target: int) -> list[list[int]]:
    """
    Find all combinations that sum to target.
    Each number can be used unlimited times.

    Time: O(n^(target/min)) roughly
    Space: O(target/min) - max recursion depth
    """
    result = []

    def backtrack(start: int, current: list[int], remaining: int):
        # Base case: found valid combination
        if remaining == 0:
            result.append(current[:])
            return

        # Base case: exceeded target
        if remaining < 0:
            return

        # Try each candidate from start onwards
        for i in range(start, len(candidates)):
            # Choose
            current.append(candidates[i])

            # Explore (same index since we can reuse)
            backtrack(i, current, remaining - candidates[i])

            # Unchoose
            current.pop()

    backtrack(0, [], target)
    return result
```

---

## Visual: Permutations Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PERMUTATIONS OF [1, 2, 3]                                │
│                                                                             │
│                              []                                             │
│                    ┌─────────┼─────────┐                                    │
│                   [1]       [2]       [3]                                   │
│                  ┌─┴─┐     ┌─┴─┐     ┌─┴─┐                                  │
│               [1,2] [1,3] [2,1] [2,3] [3,1] [3,2]                           │
│                 │     │     │     │     │     │                             │
│             [1,2,3][1,3,2][2,1,3][2,3,1][3,1,2][3,2,1]                      │
│                                                                             │
│  6 permutations = 3! = 6                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Handling Duplicates

```python
def subsetsWithDup(nums: list[int]) -> list[list[int]]:
    """
    Generate subsets with duplicates in input.

    Key: Sort first, then skip duplicates at same level.
    """
    nums.sort()  # Sort to group duplicates
    result = []

    def backtrack(start: int, current: list[int]):
        result.append(current[:])

        for i in range(start, len(nums)):
            # Skip duplicates at same level
            if i > start and nums[i] == nums[i - 1]:
                continue

            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result


def permuteUnique(nums: list[int]) -> list[list[int]]:
    """
    Generate unique permutations with duplicates in input.
    """
    nums.sort()
    result = []
    used = [False] * len(nums)

    def backtrack(current: list[int]):
        if len(current) == len(nums):
            result.append(current[:])
            return

        for i in range(len(nums)):
            # Skip used elements
            if used[i]:
                continue

            # Skip duplicates: if previous same element wasn't used,
            # skip current to avoid duplicate permutations
            if i > 0 and nums[i] == nums[i - 1] and not used[i - 1]:
                continue

            used[i] = True
            current.append(nums[i])
            backtrack(current)
            current.pop()
            used[i] = False

    backtrack([])
    return result
```

---

## Common Mistakes

```python
# ❌ WRONG: Not making a copy of path
def subsets_wrong(nums):
    result = []
    def backtrack(index, current):
        result.append(current)  # Bug! Appending reference
        # ...
    # All entries in result will be the same (empty) list!

# ✅ CORRECT: Make a copy
def subsets_correct(nums):
    result = []
    def backtrack(index, current):
        result.append(current[:])  # Copy with [:]
        # or: result.append(list(current))


# ❌ WRONG: Not backtracking properly
def permute_wrong(nums):
    def backtrack(current):
        if len(current) == len(nums):
            result.append(current[:])
            return
        for num in nums:
            if num not in current:
                current.append(num)
                backtrack(current)
                # Missing: current.pop()  # Forgot to backtrack!

# ✅ CORRECT: Always undo the choice
def permute_correct(nums):
    def backtrack(current):
        if len(current) == len(nums):
            result.append(current[:])
            return
        for num in nums:
            if num not in current:
                current.append(num)
                backtrack(current)
                current.pop()  # Backtrack!
```

---

## Complexity Analysis

| Problem | Time | Space |
|---------|------|-------|
| Subsets | O(n × 2ⁿ) | O(n) |
| Permutations | O(n × n!) | O(n) |
| Combinations C(n,k) | O(k × C(n,k)) | O(k) |
| N-Queens | O(n!) | O(n) |

---

## Interview Tips

### 1. How to Explain Your Approach
```
"I'll use backtracking to explore all possibilities. At each step, I make
a choice, recurse to explore that path, then undo the choice (backtrack)
to try other options. I'll prune invalid paths early for efficiency."
```

### 2. What Interviewers Look For
- **Template mastery**: Know subsets vs permutations vs combinations
- **Pruning**: Identify when to cut off invalid branches
- **State management**: Proper backtracking (undo changes)

### 3. Common Follow-up Questions
- "Can you optimize?" → Add pruning, memoization if applicable
- "What's the time complexity?" → Usually O(2^n) or O(n!)
- "How to handle duplicates?" → Sort + skip consecutive duplicates

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
