# Backtracking - Medium Problems

## Problem 1: Word Search (LC #79) - Medium

- [LeetCode](https://leetcode.com/problems/word-search/)

### Problem Statement
Given an `m x n` grid of characters `board` and a string `word`, return `true` if `word` exists in the grid. The word can be constructed from letters of sequentially adjacent cells (horizontally or vertically). The same cell cannot be used more than once.

### Video Explanation
- [NeetCode - Word Search](https://www.youtube.com/watch?v=pfiQ_PS1g8E)

### Examples
```
Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
Explanation:
  A → B → C
          ↓
      D ← C
      ↓
      E

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE"
Output: true

Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB"
Output: false (can't reuse B)
```

### Intuition Development
```
Classic backtracking: Try all paths, undo when stuck!

Grid:
  A B C E
  S F C S
  A D E E

Word: "ABCCED"

┌─────────────────────────────────────────────────────────────────┐
│ Strategy:                                                        │
│   1. Try starting from each cell                                │
│   2. DFS exploring 4 directions                                 │
│   3. Mark visited cells (prevent reuse)                         │
│   4. BACKTRACK: unmark when returning                           │
│                                                                  │
│ Starting at (0,0) = 'A':                                        │
│   Match word[0]='A' ✓                                           │
│   Mark (0,0) visited                                            │
│   Try (0,1)='B' → matches word[1]='B' ✓                         │
│   Try (0,2)='C' → matches word[2]='C' ✓                         │
│   Try (1,2)='C' → matches word[3]='C' ✓                         │
│   Try (2,2)='E' → matches word[4]='E' ✓                         │
│   Try (2,1)='D' → matches word[5]='D' ✓                         │
│   All matched! Return true                                       │
│                                                                  │
│ Key: Temporarily mark cells as '#' to prevent reuse             │
│      Restore original character when backtracking               │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def exist(board: list[list[str]], word: str) -> bool:
    """
    Find word in grid using backtracking.

    Strategy:
    - Try starting from each cell
    - DFS with visited tracking
    - Backtrack by restoring cell

    Time: O(m * n * 4^L) where L = word length
    Space: O(L) for recursion
    """
    rows, cols = len(board), len(board[0])

    def backtrack(r: int, c: int, idx: int) -> bool:
        # Found complete word
        if idx == len(word):
            return True

        # Boundary check and character match
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] != word[idx]):
            return False

        # Mark as visited
        temp = board[r][c]
        board[r][c] = '#'

        # Explore all 4 directions
        found = (backtrack(r + 1, c, idx + 1) or
                 backtrack(r - 1, c, idx + 1) or
                 backtrack(r, c + 1, idx + 1) or
                 backtrack(r, c - 1, idx + 1))

        # Restore cell (backtrack)
        board[r][c] = temp

        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True

    return False
```

### Complexity
- **Time**: O(m × n × 4^L) - Try each cell, explore 4 directions for word length L
- **Space**: O(L) - Recursion depth equals word length

### Edge Cases
- Word longer than grid: Return false
- Single character word: Check if any cell matches
- Word starts but can't complete: Backtracking handles this
- Entire grid is one character: Handle repeated characters

---

## Problem 2: Palindrome Partitioning (LC #131) - Medium

- [LeetCode](https://leetcode.com/problems/palindrome-partitioning/)

### Problem Statement
Given a string `s`, partition `s` such that every substring of the partition is a **palindrome**. Return all possible palindrome partitionings of `s`.

### Video Explanation
- [NeetCode - Palindrome Partitioning](https://www.youtube.com/watch?v=3jvWodd7ht0)

### Examples
```
Input: s = "aab"
Output: [["a","a","b"],["aa","b"]]
Explanation:
  - "a" | "a" | "b" - all palindromes
  - "aa" | "b" - both palindromes
  - "aab" is NOT a palindrome, so not included

Input: s = "a"
Output: [["a"]]

Input: s = "aba"
Output: [["a","b","a"],["aba"]]
```

### Intuition Development
```
Try all possible first palindromes, then recurse on rest!

s = "aab"

┌─────────────────────────────────────────────────────────────────┐
│ At each position, try all possible palindrome prefixes:        │
│                                                                  │
│ Start at index 0:                                               │
│   Try "a" (palindrome ✓) → recurse on "ab"                      │
│     At index 1:                                                 │
│       Try "a" (palindrome ✓) → recurse on "b"                   │
│         At index 2:                                             │
│           Try "b" (palindrome ✓) → end                          │
│         Result: ["a", "a", "b"] ✓                               │
│       Try "ab" (not palindrome ✗) → skip                        │
│   Try "aa" (palindrome ✓) → recurse on "b"                      │
│     At index 2:                                                 │
│       Try "b" (palindrome ✓) → end                              │
│     Result: ["aa", "b"] ✓                                       │
│   Try "aab" (not palindrome ✗) → skip                           │
│                                                                  │
│ Final: [["a","a","b"], ["aa","b"]]                              │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def partition(s: str) -> list[list[str]]:
    """
    Find all palindrome partitions using backtracking.

    Strategy:
    - Try all possible first palindrome substrings
    - Recursively partition remaining string

    Time: O(n * 2^n)
    Space: O(n)
    """
    result = []

    def is_palindrome(start: int, end: int) -> bool:
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True

    def backtrack(start: int, path: list[str]):
        if start == len(s):
            result.append(path[:])
            return

        for end in range(start, len(s)):
            if is_palindrome(start, end):
                path.append(s[start:end + 1])
                backtrack(end + 1, path)
                path.pop()

    backtrack(0, [])
    return result
```

### Complexity
- **Time**: O(n × 2^n) - At most 2^n partitions, O(n) to check palindrome
- **Space**: O(n) - Recursion depth and path storage

### Edge Cases
- Single character: Always a palindrome, return [[s]]
- All same characters: Multiple valid partitions
- No palindrome partitions: Not possible (single chars are always palindromes)

---

## Problem 3: Restore IP Addresses (LC #93) - Medium

- [LeetCode](https://leetcode.com/problems/restore-ip-addresses/)

### Problem Statement
A valid IP address consists of exactly four integers separated by dots. Each integer is between 0 and 255 and cannot have leading zeros. Given a string `s` containing only digits, return all possible valid IP addresses.

### Video Explanation
- [NeetCode - Restore IP Addresses](https://www.youtube.com/watch?v=61tN4YEdiTM)

### Examples
```
Input: s = "25525511135"
Output: ["255.255.11.135","255.255.111.35"]
Explanation:
  255.255.11.135 - all parts valid
  255.255.111.35 - all parts valid
  255.255.1.1135 - 1135 > 255, invalid

Input: s = "0000"
Output: ["0.0.0.0"]

Input: s = "101023"
Output: ["1.0.10.23","1.0.102.3","10.1.0.23","10.10.2.3","101.0.2.3"]
```

### Intuition Development
```
Constraints help us prune aggressively!

IP Rules:
  - 4 parts exactly
  - Each part: 0-255
  - No leading zeros (except "0" itself)

s = "25525511135" (length 11)

┌─────────────────────────────────────────────────────────────────┐
│ Key insight: Each part has 1-3 digits                           │
│                                                                  │
│ Pruning:                                                         │
│   - Remaining chars must fit remaining parts                     │
│   - Min: 1 char × remaining_parts                               │
│   - Max: 3 chars × remaining_parts                              │
│                                                                  │
│ Try: 2 | 55 | 255 | 11135 → 11135 too long for 1 part           │
│ Try: 25 | 52 | 551 | 1135 → each valid? 551 > 255 ✗             │
│ Try: 255 | 25 | 511 | 135 → 511 > 255 ✗                         │
│ Try: 255 | 255 | 11 | 135 → all valid ✓                         │
│ Try: 255 | 255 | 111 | 35 → all valid ✓                         │
│                                                                  │
│ Result: ["255.255.11.135", "255.255.111.35"]                    │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def restoreIpAddresses(s: str) -> list[str]:
    """
    Generate valid IP addresses using backtracking.

    Constraints:
    - 4 parts, each 0-255
    - No leading zeros (except "0")

    Time: O(3^4) = O(1) since max 12 digits
    Space: O(1)
    """
    result = []

    def is_valid(segment: str) -> bool:
        # Check length and leading zeros
        if len(segment) > 3 or (len(segment) > 1 and segment[0] == '0'):
            return False
        # Check range
        return 0 <= int(segment) <= 255

    def backtrack(start: int, parts: list[str]):
        # Found 4 valid parts and used all characters
        if len(parts) == 4:
            if start == len(s):
                result.append('.'.join(parts))
            return

        # Remaining characters check
        remaining = len(s) - start
        remaining_parts = 4 - len(parts)

        # Pruning: need 1-3 chars per remaining part
        if remaining < remaining_parts or remaining > remaining_parts * 3:
            return

        # Try 1, 2, or 3 characters
        for length in range(1, 4):
            if start + length <= len(s):
                segment = s[start:start + length]
                if is_valid(segment):
                    parts.append(segment)
                    backtrack(start + length, parts)
                    parts.pop()

    backtrack(0, [])
    return result
```

### Complexity
- **Time**: O(3^4) = O(1) - At most 12 digits, 3 choices per part
- **Space**: O(1) - Fixed maximum recursion depth

### Edge Cases
- Length < 4 or > 12: Return empty list
- All zeros: "0.0.0.0" only valid for "0000"
- Leading zeros: "010" → "0.1.0" valid, "01.0" invalid

---

## Problem 4: Expression Add Operators (LC #282) - Hard

- [LeetCode](https://leetcode.com/problems/expression-add-operators/)

### Problem Statement
Given a string `num` that contains only digits and an integer `target`, return all possibilities to insert the binary operators `+`, `-`, and/or `*` between the digits so the expression evaluates to `target`.

### Video Explanation
- [NeetCode - Expression Add Operators](https://www.youtube.com/watch?v=V7M1Z-p0QeM)

### Examples
```
Input: num = "123", target = 6
Output: ["1+2+3", "1*2*3"]
Explanation: 1+2+3=6, 1*2*3=6

Input: num = "232", target = 8
Output: ["2*3+2", "2+3*2"]
Explanation: 2*3+2=8, 2+3*2=8 (multiplication has precedence)

Input: num = "3456237490", target = 9191
Output: []
```

### Intuition Development
```
Challenge: Multiplication has higher precedence!

num = "232", target = 8

Wrong approach: Simple left-to-right evaluation
  2 + 3 * 2 → (2 + 3) * 2 = 10 ✗

Correct: Handle precedence by tracking previous operand!

┌─────────────────────────────────────────────────────────────────┐
│ Track: value (current total), prev (last operand)              │
│                                                                  │
│ When we see multiplication:                                     │
│   UNDO the previous addition/subtraction                        │
│   REDO with multiplication                                      │
│                                                                  │
│ Example: 2 + 3 * 2                                              │
│   After "2+3": value=5, prev=3                                  │
│   See "*2":                                                     │
│     new_value = value - prev + prev * 2                         │
│               = 5 - 3 + 3 * 2                                   │
│               = 5 - 3 + 6 = 8 ✓                                 │
│                                                                  │
│ For subtraction: prev = -curr (negative for future undo)        │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def addOperators(num: str, target: int) -> list[str]:
    """
    Add operators to reach target using backtracking.

    Challenge: Handle multiplication precedence.
    Track previous operand to undo addition before multiplication.

    Time: O(4^n)
    Space: O(n)
    """
    result = []

    def backtrack(idx: int, path: str, value: int, prev: int):
        if idx == len(num):
            if value == target:
                result.append(path)
            return

        for i in range(idx, len(num)):
            # Skip numbers with leading zeros
            if i > idx and num[idx] == '0':
                break

            curr_str = num[idx:i + 1]
            curr = int(curr_str)

            if idx == 0:
                # First number, no operator
                backtrack(i + 1, curr_str, curr, curr)
            else:
                # Addition
                backtrack(i + 1, path + '+' + curr_str,
                         value + curr, curr)

                # Subtraction
                backtrack(i + 1, path + '-' + curr_str,
                         value - curr, -curr)

                # Multiplication: undo prev addition, do multiplication
                backtrack(i + 1, path + '*' + curr_str,
                         value - prev + prev * curr, prev * curr)

    if num:
        backtrack(0, '', 0, 0)

    return result
```

### Complexity
- **Time**: O(n × 4^n) - Try each split, 4 operators per position
- **Space**: O(n) - Recursion depth and expression string

### Edge Cases
- Leading zeros: Skip numbers like "05" (only "0" allowed)
- Empty string: Return empty list
- Target 0: "0" works, "00" gives "0+0", "0-0", "0*0"
- Overflow: May need to handle large intermediate values

---

## Problem 5: Word Search II (LC #212) - Hard

- [LeetCode](https://leetcode.com/problems/word-search-ii/)

### Problem Statement
Given an `m x n` board of characters and a list of strings `words`, return all words that can be found in the grid. Each word must be constructed from letters of sequentially adjacent cells. The same cell cannot be used more than once per word.

### Video Explanation
- [NeetCode - Word Search II](https://www.youtube.com/watch?v=asbcE9mZz_U)

### Examples
```
Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]],
       words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]

Input: board = [["a","b"],["c","d"]], words = ["abcb"]
Output: []
```

### Intuition Development
```
Naive: Run Word Search I for each word → O(W × m × n × 4^L)
Better: Build Trie, search all words in ONE traversal!

┌─────────────────────────────────────────────────────────────────┐
│ Build Trie from words:                                          │
│                                                                  │
│    words = ["oath", "eat"]                                      │
│                                                                  │
│         root                                                     │
│        /    \                                                    │
│       o      e                                                   │
│       |      |                                                   │
│       a      a                                                   │
│       |      |                                                   │
│       t      t ($="eat")                                        │
│       |                                                          │
│       h ($="oath")                                              │
│                                                                  │
│ DFS from each cell, following Trie paths:                       │
│   - If char not in Trie children → stop                         │
│   - If reach word end ($) → found word!                         │
│   - Continue exploring all 4 directions                         │
│                                                                  │
│ Optimization: Remove found words from Trie (avoid duplicates)   │
│ Optimization: Prune empty Trie branches                         │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    """
    Find all words using Trie + Backtracking.

    Optimization:
    - Build Trie from words
    - DFS from each cell
    - Remove found words to avoid duplicates

    Time: O(m * n * 4^L * W)
    Space: O(W * L) for Trie
    """
    # Build Trie
    trie = {}
    for word in words:
        node = trie
        for char in word:
            node = node.setdefault(char, {})
        node['$'] = word  # Store complete word

    rows, cols = len(board), len(board[0])
    result = []

    def backtrack(r: int, c: int, node: dict):
        char = board[r][c]

        if char not in node:
            return

        next_node = node[char]

        # Found a word
        if '$' in next_node:
            result.append(next_node['$'])
            del next_node['$']  # Remove to avoid duplicates

        # Mark visited
        board[r][c] = '#'

        # Explore neighbors
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '#':
                backtrack(nr, nc, next_node)

        # Restore
        board[r][c] = char

        # Prune empty branches
        if not next_node:
            del node[char]

    for r in range(rows):
        for c in range(cols):
            backtrack(r, c, trie)

    return result
```

### Complexity
- **Time**: O(m × n × 4^L × W) where W = total words length
- **Space**: O(W × L) for Trie storage

### Edge Cases
- Word not possible: Grid lacks required characters
- Duplicate words in list: Trie naturally handles deduplication
- Overlapping words: Each is found independently
- Single cell board: Only single character words possible

---

## Problem 6: Combination Sum IV (LC #377) - Medium

- [LeetCode](https://leetcode.com/problems/combination-sum-iv/)

### Problem Statement
Given an array of **distinct** integers `nums` and a target integer `target`, return the number of possible combinations that add up to target. Note: Different orderings count as different combinations.

### Video Explanation
- [NeetCode - Combination Sum IV](https://www.youtube.com/watch?v=dw2nMCxG0ik)

### Examples
```
Input: nums = [1,2,3], target = 4
Output: 7
Explanation: All combinations that sum to 4:
  (1, 1, 1, 1)
  (1, 1, 2)
  (1, 2, 1)
  (1, 3)
  (2, 1, 1)
  (2, 2)
  (3, 1)

Input: nums = [9], target = 3
Output: 0
```

### Intuition Development
```
Wait! Order matters → This is actually DP, not backtracking!

Why DP works better:
  - Pure backtracking: O(target^n) - too slow
  - DP: O(target × n) - much faster

┌─────────────────────────────────────────────────────────────────┐
│ DP State: dp[i] = ways to form sum i                            │
│                                                                  │
│ nums = [1, 2, 3], target = 4                                    │
│                                                                  │
│ dp[0] = 1  (empty combination)                                  │
│ dp[1] = dp[1-1] = dp[0] = 1  (just use 1)                       │
│ dp[2] = dp[2-1] + dp[2-2] = dp[1] + dp[0] = 1 + 1 = 2           │
│ dp[3] = dp[3-1] + dp[3-2] + dp[3-3] = 2 + 1 + 1 = 4             │
│ dp[4] = dp[4-1] + dp[4-2] + dp[4-3] = 4 + 2 + 1 = 7             │
│                                                                  │
│ Formula: dp[i] = Σ dp[i - num] for all num ≤ i                  │
│                                                                  │
│ Note: This counts ORDERED combinations (permutations)           │
│       For unordered, use different DP approach                  │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def combinationSum4(nums: list[int], target: int) -> int:
    """
    Count ordered combinations using DP (not backtracking).

    Note: This is actually a DP problem, not backtracking.
    Order matters = permutations, use DP.

    Time: O(target * n)
    Space: O(target)
    """
    dp = [0] * (target + 1)
    dp[0] = 1

    for i in range(1, target + 1):
        for num in nums:
            if i >= num:
                dp[i] += dp[i - num]

    return dp[target]


def combinationSum4_memo(nums: list[int], target: int) -> int:
    """
    Alternative: Memoized recursion.
    """
    memo = {}

    def count(remaining: int) -> int:
        if remaining == 0:
            return 1
        if remaining < 0:
            return 0
        if remaining in memo:
            return memo[remaining]

        total = 0
        for num in nums:
            total += count(remaining - num)

        memo[remaining] = total
        return total

    return count(target)
```

### Complexity
- **Time**: O(target × n) - Fill dp array, check each num
- **Space**: O(target) - DP array

### Edge Cases
- Target 0: Only empty combination, return 1
- No valid combination: Return 0
- Single element equals target: Return 1
- Negative numbers: Problem states distinct positive integers

---

## Problem 7: Splitting a String Into Descending Consecutive Values (LC #1849) - Medium

- [LeetCode](https://leetcode.com/problems/splitting-a-string-into-descending-consecutive-values/)

### Problem Statement
You are given a string `s` that consists of only digits. Check if we can split `s` into two or more **non-empty** substrings such that the numerical values of the substrings are in **descending** order and the difference between consecutive values is exactly 1.

### Video Explanation
- [NeetCode - Splitting String Descending](https://www.youtube.com/watch?v=wln-gJYBUuE)

### Examples
```
Input: s = "050043"
Output: true
Explanation: "05" → 5, "004" → 4, "3" → 3. Descending consecutive: 5,4,3 ✓

Input: s = "9080701"
Output: false
Explanation: No valid split exists

Input: s = "10"
Output: false
Explanation: "1" → 1, "0" → 0 is consecutive but descending, but 1-0=1, so true!
Wait, actually true: 1, 0 is valid (1 > 0, diff = 1)
```

### Intuition Development
```
Try all possible first numbers, then check if rest can form descending!

s = "050043"

┌─────────────────────────────────────────────────────────────────┐
│ Try first = "0" = 0                                             │
│   Need next = -1 (impossible for positive representation)       │
│   Skip                                                           │
│                                                                  │
│ Try first = "05" = 5                                            │
│   Need next = 4                                                  │
│   Check "0043": can we start with 4?                            │
│     Try "0" = 0 ≠ 4                                             │
│     Try "00" = 0 ≠ 4                                            │
│     Try "004" = 4 = 4 ✓                                         │
│       Need next = 3                                              │
│       Check "3": "3" = 3 = 3 ✓                                  │
│       Reached end → TRUE!                                        │
│                                                                  │
│ Key: Leading zeros allowed! "004" = 4                           │
│ Key: Can't use entire string as first number                    │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def splitString(s: str) -> bool:
    """
    Split into descending consecutive values.

    Strategy:
    - Try all possible first numbers
    - Check if rest can form descending sequence

    Time: O(n^2)
    Space: O(n)
    """
    def backtrack(idx: int, prev: int) -> bool:
        if idx == len(s):
            return True

        curr = 0
        for i in range(idx, len(s)):
            curr = curr * 10 + int(s[i])

            # Pruning: if curr >= prev, can't be descending
            if curr >= prev:
                break

            # Check if curr = prev - 1
            if curr == prev - 1:
                if backtrack(i + 1, curr):
                    return True

        return False

    # Try all possible first numbers (can't use entire string)
    first = 0
    for i in range(len(s) - 1):
        first = first * 10 + int(s[i])

        if backtrack(i + 1, first):
            return True

    return False
```

### Complexity
- **Time**: O(n²) - Try all first numbers, check remaining
- **Space**: O(n) - Recursion depth

### Edge Cases
- Single digit: Can't split into 2+ parts, return false
- All zeros: "000" → 0, 0, 0 not descending consecutive
- Leading zeros: "00" = 0, valid number representation
- Very large numbers: May overflow, use careful parsing

---

## Problem 8: Beautiful Arrangement (LC #526) - Medium

- [LeetCode](https://leetcode.com/problems/beautiful-arrangement/)

### Problem Statement
Given an integer `n`, return the number of **beautiful arrangements**. A beautiful arrangement is a permutation of integers from 1 to n such that for every position i (1-indexed): `perm[i] % i == 0` OR `i % perm[i] == 0`.

### Video Explanation
- [NeetCode - Beautiful Arrangement](https://www.youtube.com/watch?v=K0qTcUgU4VM)

### Examples
```
Input: n = 2
Output: 2
Explanation:
  [1, 2]: pos 1: 1%1=0 ✓, pos 2: 2%2=0 ✓
  [2, 1]: pos 1: 2%1=0 ✓, pos 2: 1%2≠0 but 2%1=0 ✓

Input: n = 1
Output: 1
Explanation: Only [1] is valid

Input: n = 3
Output: 3
Explanation: [1,2,3], [2,1,3], [3,2,1]
```

### Intuition Development
```
Build permutation position by position, checking divisibility!

n = 3

┌─────────────────────────────────────────────────────────────────┐
│ Position 1: Which numbers can go here?                          │
│   1: 1%1=0 ✓                                                    │
│   2: 2%1=0 ✓                                                    │
│   3: 3%1=0 ✓                                                    │
│   All work! (anything is divisible by 1)                        │
│                                                                  │
│ Position 2: Which numbers can go here?                          │
│   1: 1%2≠0, 2%1=0 ✓                                             │
│   2: 2%2=0 ✓                                                    │
│   3: 3%2≠0, 2%3≠0 ✗                                             │
│   Only 1 and 2 work!                                            │
│                                                                  │
│ Position 3: Which numbers can go here?                          │
│   1: 1%3≠0, 3%1=0 ✓                                             │
│   2: 2%3≠0, 3%2≠0 ✗                                             │
│   3: 3%3=0 ✓                                                    │
│   Only 1 and 3 work!                                            │
│                                                                  │
│ Backtracking tree prunes invalid branches early!                │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def countArrangement(n: int) -> int:
    """
    Count beautiful arrangements using backtracking.

    Time: O(k) where k = number of valid permutations
    Space: O(n)
    """
    count = [0]
    used = [False] * (n + 1)

    def backtrack(pos: int):
        if pos > n:
            count[0] += 1
            return

        for num in range(1, n + 1):
            if not used[num] and (num % pos == 0 or pos % num == 0):
                used[num] = True
                backtrack(pos + 1)
                used[num] = False

    backtrack(1)
    return count[0]
```

### Complexity
- **Time**: O(k) where k = number of valid arrangements (hard to express, less than n!)
- **Space**: O(n) - Used array and recursion stack

### Edge Cases
- n = 1: Only [1], return 1
- n = 2: [1,2] and [2,1] both valid, return 2
- Large n: Grows quickly but still manageable (n ≤ 15)

---

## Problem 9: Matchsticks to Square (LC #473) - Medium

- [LeetCode](https://leetcode.com/problems/matchsticks-to-square/)

### Problem Statement
You are given an integer array `matchsticks` where `matchsticks[i]` is the length of the `ith` matchstick. You want to use **all** matchsticks to make one square. You should not break any stick, but you can link them up. Return `true` if you can make this square and `false` otherwise.

### Video Explanation
- [NeetCode - Matchsticks to Square](https://www.youtube.com/watch?v=hUe0cUKV-YY)

### Examples
```
Input: matchsticks = [1,1,2,2,2]
Output: true
Explanation: Form square with side = 2
  Side 1: [2]
  Side 2: [2]
  Side 3: [2]
  Side 4: [1, 1]

Input: matchsticks = [3,3,3,3,4]
Output: false
Explanation: Total = 16, side = 4, but no way to form 4 sides of length 4

Input: matchsticks = [5,5,5,5,4,4,4,4,3,3,3,3]
Output: true
```

### Intuition Development
```
Partition matchsticks into 4 groups with equal sum!

Total = 8, side_length = 2

matchsticks = [1,1,2,2,2] (sorted descending: [2,2,2,1,1])

┌─────────────────────────────────────────────────────────────────┐
│ 4 buckets (sides) each need sum = 2                             │
│                                                                  │
│ Place 2 in side 0: [2, 0, 0, 0]                                 │
│ Place 2 in side 1: [2, 2, 0, 0]                                 │
│ Place 2 in side 2: [2, 2, 2, 0]                                 │
│ Place 1 in side 3: [2, 2, 2, 1]                                 │
│ Place 1 in side 3: [2, 2, 2, 2] ✓                               │
│                                                                  │
│ Optimizations:                                                   │
│   1. Sort DESCENDING - larger sticks fail faster                │
│   2. If largest stick > side, impossible                        │
│   3. If empty bucket fails, other empty buckets will too        │
│   4. Skip duplicate bucket states                               │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def makesquare(matchsticks: list[int]) -> bool:
    """
    Check if matchsticks can form square.

    Strategy:
    - Total must be divisible by 4
    - Try to fill 4 sides of equal length
    - Sort descending for better pruning

    Time: O(4^n)
    Space: O(n)
    """
    total = sum(matchsticks)

    if total % 4 != 0:
        return False

    side = total // 4
    matchsticks.sort(reverse=True)  # Pruning: try larger first

    # Early termination: largest stick > side
    if matchsticks[0] > side:
        return False

    sides = [0] * 4

    def backtrack(idx: int) -> bool:
        if idx == len(matchsticks):
            return all(s == side for s in sides)

        stick = matchsticks[idx]

        for i in range(4):
            if sides[i] + stick <= side:
                sides[i] += stick

                if backtrack(idx + 1):
                    return True

                sides[i] -= stick

                # Pruning: if this side is empty and failed,
                # other empty sides will also fail
                if sides[i] == 0:
                    break

        return False

    return backtrack(0)
```

### Complexity
- **Time**: O(4^n) - Each matchstick has 4 bucket choices
- **Space**: O(n) - Recursion depth

### Edge Cases
- Total not divisible by 4: Return false immediately
- Largest stick > side: Return false immediately
- Less than 4 sticks: Can still work if sums match
- All sticks same length: Check if count divisible by 4

---

## Problem 10: Partition to K Equal Sum Subsets (LC #698) - Medium

- [LeetCode](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/)

### Problem Statement
Given an integer array `nums` and an integer `k`, return `true` if it is possible to divide this array into `k` non-empty subsets whose sums are all equal.

### Video Explanation
- [NeetCode - Partition to K Equal Sum Subsets](https://www.youtube.com/watch?v=mBk4I0X46oI)

### Examples
```
Input: nums = [4,3,2,3,5,2,1], k = 4
Output: true
Explanation: Total = 20, target = 5
  Subset 1: [5]
  Subset 2: [4, 1]
  Subset 3: [3, 2]
  Subset 4: [3, 2]

Input: nums = [1,2,3,4], k = 3
Output: false
Explanation: Total = 10, 10/3 not integer

Input: nums = [2,2,2,2,3,4,5], k = 4
Output: false
```

### Intuition Development
```
Generalization of Matchsticks to Square (k buckets instead of 4)!

nums = [4,3,2,3,5,2,1], k = 4
Total = 20, target = 5

┌─────────────────────────────────────────────────────────────────┐
│ Sort descending: [5,4,3,3,2,2,1]                                │
│                                                                  │
│ Bucket filling approach:                                        │
│   Place each number into a bucket where it fits                 │
│   Backtrack if no bucket can accept the number                  │
│                                                                  │
│ 5 → bucket 0: [5, 0, 0, 0]  (bucket 0 full!)                   │
│ 4 → bucket 1: [5, 4, 0, 0]                                      │
│ 3 → bucket 2: [5, 4, 3, 0]                                      │
│ 3 → bucket 3: [5, 4, 3, 3]                                      │
│ 2 → bucket 2: [5, 4, 5, 3]  (bucket 2 full!)                   │
│ 2 → bucket 3: [5, 4, 5, 5]  (bucket 3 full!)                   │
│ 1 → bucket 1: [5, 5, 5, 5]  (all full!) ✓                      │
│                                                                  │
│ Optimization: Skip duplicate bucket states                      │
│   If buckets[i] == buckets[j] and i failed, skip j             │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def canPartitionKSubsets(nums: list[int], k: int) -> bool:
    """
    Partition into k equal sum subsets.

    Similar to matchsticks problem but generalized to k parts.

    Time: O(k^n)
    Space: O(n)
    """
    total = sum(nums)

    if total % k != 0:
        return False

    target = total // k
    nums.sort(reverse=True)

    if nums[0] > target:
        return False

    buckets = [0] * k

    def backtrack(idx: int) -> bool:
        if idx == len(nums):
            return all(b == target for b in buckets)

        num = nums[idx]
        seen = set()  # Avoid duplicate bucket states

        for i in range(k):
            if buckets[i] + num <= target and buckets[i] not in seen:
                seen.add(buckets[i])
                buckets[i] += num

                if backtrack(idx + 1):
                    return True

                buckets[i] -= num

        return False

    return backtrack(0)
```

### Complexity
- **Time**: O(k^n) - Each number has k bucket choices
- **Space**: O(n) - Recursion depth and bucket array

### Edge Cases
- Total not divisible by k: Return false immediately
- Any element > target: Return false immediately
- k = 1: Always true if array non-empty
- k = n: Each element must equal target

---

## Summary: Medium Backtracking

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Word Search | Grid DFS | O(mn * 4^L) |
| 2 | Palindrome Partition | Substring check | O(n * 2^n) |
| 3 | Restore IP | Constraint pruning | O(1) |
| 4 | Expression Operators | Track prev for * | O(4^n) |
| 5 | Word Search II | Trie + DFS | O(mn * 4^L) |
| 6 | Combination Sum IV | DP (not backtrack) | O(target * n) |
| 7 | Descending Split | First number choices | O(n²) |
| 8 | Beautiful Arrangement | Divisibility check | O(k) |
| 9 | Matchsticks Square | 4 buckets | O(4^n) |
| 10 | K Equal Subsets | k buckets | O(k^n) |

---

## Backtracking Optimization Tips

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BACKTRACKING OPTIMIZATIONS                               │
│                                                                             │
│  1. SORT INPUT:                                                             │
│     - Sort descending to fail faster                                        │
│     - Larger elements have fewer valid positions                            │
│                                                                             │
│  2. EARLY TERMINATION:                                                      │
│     - Check impossible cases before recursion                               │
│     - Sum checks, size checks                                               │
│                                                                             │
│  3. AVOID DUPLICATE STATES:                                                 │
│     - Skip same values at same level                                        │
│     - Use seen set for bucket states                                        │
│                                                                             │
│  4. CONSTRAINT PROPAGATION:                                                 │
│     - Reduce choices based on constraints                                   │
│     - Example: Sudoku naked singles                                         │
│                                                                             │
│  5. MEMOIZATION:                                                            │
│     - Cache subproblem results if applicable                                │
│     - Convert to DP when possible                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practice More Problems

- [ ] LC #37 - Sudoku Solver
- [ ] LC #51 - N-Queens
- [ ] LC #52 - N-Queens II
- [ ] LC #291 - Word Pattern II
- [ ] LC #1240 - Tiling a Rectangle with the Fewest Squares

