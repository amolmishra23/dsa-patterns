# Dynamic Programming - Hard Problems

## Problem 1: Edit Distance (LC #72) - Hard

- [LeetCode](https://leetcode.com/problems/edit-distance/)

### Video Explanation
- [NeetCode - Edit Distance](https://www.youtube.com/watch?v=XYi2-LPrwm4)

### Problem Statement
Find minimum operations to convert word1 to word2 (insert, delete, replace).

### Visual Intuition
```
Edit Distance: word1="horse" → word2="ros"

Pattern: 2D DP on Two Strings
Why: Each cell considers all possible operations

Step 0 (Define State):
  dp[i][j] = min operations to convert word1[0:i] to word2[0:j]

  Three operations:
  ┌─────────────────────────────────────────────────┐
  │ Insert:  dp[i][j-1] + 1   ← add char to word1   │
  │ Delete:  dp[i-1][j] + 1   ← remove from word1   │
  │ Replace: dp[i-1][j-1] + 1 ← change char         │
  │ Match:   dp[i-1][j-1]     ← chars equal, free   │
  └─────────────────────────────────────────────────┘

Step 1 (Fill Base Cases):

        ""  r  o  s
    ""   0  1  2  3  ← insert all chars
    h    1             ↑
    o    2             │ delete all chars
    r    3             │
    s    4             │
    e    5             │

Step 2 (Fill Table - Key Cells):

        ""  r  o  s
    ""   0  1  2  3
    h    1  ?

  word1[0]='h', word2[0]='r' → not equal
  dp[1][1] = 1 + min(dp[0][0], dp[0][1], dp[1][0])
           = 1 + min(0, 1, 1) = 1  (replace h→r)

Step 3 (Continue Filling):

        ""  r  o  s
    ""   0  1  2  3
    h    1  1  2  3
    o    2  2  1  2  ← 'o'='o', match! dp[1][1]=1
    r    3  2  2  2
    s    4  3  3  2  ← 's'='s', match!
    e    5  4  4  3

Step 4 (Trace Optimal Path):

  Start: dp[5][3] = 3

  "horse" → "ros" (3 operations):

  ┌─────────────────────────────────────────────────┐
  │ horse                                           │
  │   ↓ replace 'h' with 'r'                        │
  │ rorse                                           │
  │   ↓ delete second 'r'                           │
  │ rose                                            │
  │   ↓ delete 'e'                                  │
  │ ros ✓                                           │
  └─────────────────────────────────────────────────┘

Transition Visualization:

  For cell (i, j):
  ┌───────┬───────┐
  │ diag  │  up   │  diag = replace/match
  │(i-1,  │(i-1,j)│  up = delete
  │ j-1)  │       │  left = insert
  ├───────┼───────┤
  │ left  │ curr  │
  │(i,j-1)│(i, j) │
  └───────┴───────┘

Key Insight:
- If chars match: take diagonal (free move)
- If chars differ: 1 + min(all three directions)
- O(m×n) time and space
```


### Examples
```
Input: word1 = "horse", word2 = "ros"
Output: 3
horse → rorse (replace 'h' with 'r')
rorse → rose (remove 'r')
rose → ros (remove 'e')

Input: word1 = "intention", word2 = "execution"
Output: 5
```

### Intuition
```
dp[i][j] = min operations to convert word1[0:i] to word2[0:j]

If word1[i-1] == word2[j-1]:
    dp[i][j] = dp[i-1][j-1]  (no operation needed)
Else:
    dp[i][j] = 1 + min(
        dp[i-1][j],    # delete from word1
        dp[i][j-1],    # insert into word1
        dp[i-1][j-1]   # replace
    )

     ""  r  o  s
""    0  1  2  3
h     1  1  2  3
o     2  2  1  2
r     3  2  2  2
s     4  3  3  2
e     5  4  4  3
```

### Solution
```python
def minDistance(word1: str, word2: str) -> int:
    """
    Minimum edit distance (Levenshtein distance).

    Strategy:
    - dp[i][j] = min ops to convert word1[0:i] to word2[0:j]
    - Three operations: insert, delete, replace

    Time: O(m * n)
    Space: O(m * n), can be O(n) with optimization
    """
    m, n = len(word1), len(word2)

    # dp[i][j] = edit distance for word1[0:i] and word2[0:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )

    return dp[m][n]


def minDistance_optimized(word1: str, word2: str) -> int:
    """
    Space-optimized version using two rows.

    Time: O(m * n)
    Space: O(n)
    """
    m, n = len(word1), len(word2)

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i

        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])

        prev, curr = curr, prev

    return prev[n]
```

### Complexity
- **Time**: O(m × n)
- **Space**: O(m × n) or O(n) optimized

### Edge Cases
- Empty strings → return 0
- Same strings → return 0
- One empty → return length of other
- Single character difference → return 1

---

## Problem 2: Regular Expression Matching (LC #10) - Hard

- [LeetCode](https://leetcode.com/problems/regular-expression-matching/)

### Video Explanation
- [NeetCode - Regular Expression Matching](https://www.youtube.com/watch?v=HAA8mgxlov8)

### Problem Statement
Implement regex matching with '.' and '*'.

### Visual Intuition
```
Regular Expression Matching: s="aab", p="c*a*b"

Pattern: 2D DP with Special Char Handling
Why: '*' can match 0 or more of preceding element

Step 0 (Understand '*' Behavior):

  "c*" can match: "" (zero c), "c", "cc", "ccc", ...
  "a*" can match: "" (zero a), "a", "aa", "aaa", ...

  So "c*a*b" can match:
  - "b" (zero c, zero a, one b)
  - "ab" (zero c, one a, one b)
  - "aab" ✓ (zero c, two a, one b)

Step 1 (Fill Base Cases):

        ""  c  *  a  *  b
    ""   T  F  T  F  T  F
              ↑     ↑
              "c*" matches "" (zero c's)
              "c*a*" matches "" (zero c's, zero a's)

Step 2 (Handle '*' - Two Cases):

  For p[j] = '*':
  ┌─────────────────────────────────────────────────┐
  │ Case 1: Match ZERO of preceding                 │
  │   dp[i][j] = dp[i][j-2]                         │
  │   Skip both '*' and its preceding char          │
  │                                                 │
  │ Case 2: Match ONE OR MORE of preceding          │
  │   dp[i][j] = dp[i-1][j]                         │
  │   IF s[i-1] matches p[j-2] (the char before *)  │
  │   "Consume" one char from s, pattern stays      │
  └─────────────────────────────────────────────────┘

Step 3 (Trace Key Cell - dp[2][5]):

  s = "aa", p = "c*a*"

        ""  c  *  a  *
    ""   T  F  T  F  T
    a    F  F  F  T  T  ← dp[1][5]: "a" vs "c*a*"
    a    F  F  F  F  T  ← dp[2][5]: "aa" vs "c*a*"

  dp[2][5]: p[4]='*', preceding='a'
    Case 1: dp[2][3] = F (zero more a's)
    Case 2: s[1]='a' matches 'a' → dp[1][5] = T ✓
    Result: T (use one more 'a')

Step 4 (Complete Table):

        ""  c  *  a  *  b
    ""   T  F  T  F  T  F
    a    F  F  F  T  T  F
    a    F  F  F  F  T  F
    b    F  F  F  F  F  T ← Final answer!

  dp[3][6]: p[5]='b', s[2]='b'
    Chars match! dp[3][6] = dp[2][5] = T ✓

Transition Diagram:

  For '*' at position j:
  ┌───────────────────────────────────────────────┐
  │           j-2   j-1    j                      │
  │            ↓     ↓     ↓                      │
  │ Pattern:  [a]   [*]   ...                     │
  │                                               │
  │ Match 0:  dp[i][j] ← dp[i][j-2]              │
  │           (skip "a*")                         │
  │                                               │
  │ Match 1+: dp[i][j] ← dp[i-1][j]              │
  │           (if s[i-1] matches p[j-2])         │
  │           (consume one char, pattern stays)   │
  └───────────────────────────────────────────────┘

Key Insight:
- '*' always pairs with preceding char
- Check both "match zero" and "match more" cases
- '.' before '*' matches any character sequence
```

- '.' matches any single character
- '*' matches zero or more of the preceding element

### Examples
```
Input: s = "aa", p = "a"
Output: false

Input: s = "aa", p = "a*"
Output: true

Input: s = "ab", p = ".*"
Output: true
```

### Intuition
```
dp[i][j] = True if s[0:i] matches p[0:j]

Cases:
1. p[j-1] is normal char: must match s[i-1]
2. p[j-1] is '.': matches any s[i-1]
3. p[j-1] is '*':
   - Zero occurrences: dp[i][j-2]
   - One or more: dp[i-1][j] if p[j-2] matches s[i-1]
```

### Solution
```python
def isMatch(s: str, p: str) -> bool:
    """
    Regular expression matching with '.' and '*'.

    Strategy:
    - dp[i][j] = True if s[0:i] matches p[0:j]
    - Handle '*' as zero or more of preceding element

    Time: O(m * n)
    Space: O(m * n)
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]

    # Empty pattern matches empty string
    dp[0][0] = True

    # Handle patterns like a*, a*b*, a*b*c* matching empty string
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # Zero occurrences of preceding element
                dp[i][j] = dp[i][j - 2]

                # One or more occurrences
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]

            elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]

    return dp[m][n]
```

### Complexity
- **Time**: O(m × n)
- **Space**: O(m × n)

### Edge Cases
- Empty pattern → only matches empty string
- Pattern ".*" → matches any string
- Pattern "a*" → matches empty or "aaa..."
- No wildcards → exact match needed

---

## Problem 3: Burst Balloons (LC #312) - Hard

- [LeetCode](https://leetcode.com/problems/burst-balloons/)

### Video Explanation
- [NeetCode - Burst Balloons](https://www.youtube.com/watch?v=VFskby7lUbw)

### Problem Statement
Burst all balloons to maximize coins. Bursting balloon i gives nums[i-1] * nums[i] * nums[i+1] coins.

### Visual Intuition
```
Burst Balloons: nums=[3,1,5,8]

Pattern: Interval DP - Think About LAST Element
Why: Bursting order affects neighbors, think backwards

Step 0 (Key Insight):

  Wrong approach: Which balloon to burst FIRST?
  Problem: After bursting, neighbors change!

  Right approach: Which balloon to burst LAST?
  When it's last, its neighbors are the boundaries!

  Add virtual balloons: [1, 3, 1, 5, 8, 1]
                         ↑              ↑
                       left           right
                      boundary       boundary

Step 1 (Define State):

  dp[i][j] = max coins from bursting all balloons in (i, j)
             (exclusive - balloons between i and j)

  If k is the LAST balloon burst in range (i, j):

  ┌─────────────────────────────────────────────────┐
  │ dp[i][j] = dp[i][k] + dp[k][j] + nums[i]*nums[k]*nums[j] │
  │            ↑           ↑         ↑                │
  │         left of k   right of k   k is last,      │
  │         (already    (already     neighbors are   │
  │          burst)      burst)      i and j         │
  └─────────────────────────────────────────────────┘

Step 2 (Trace Example - Range (0, 5)):

  nums = [1, 3, 1, 5, 8, 1]
          0  1  2  3  4  5

  Try k=1 (balloon 3) as last:
    dp[0][1] + dp[1][5] + nums[0]*nums[1]*nums[5]
    = 0 + dp[1][5] + 1*3*1

  Try k=2 (balloon 1) as last:
    dp[0][2] + dp[2][5] + 1*1*1

  Try k=3 (balloon 5) as last:
    dp[0][3] + dp[3][5] + 1*5*1

  Try k=4 (balloon 8) as last:
    dp[0][4] + dp[4][5] + 1*8*1

Step 3 (Build Table - Small Ranges First):

  Length 2 (single balloon):
  dp[0][2] = 1*1*1 = 1   (burst balloon 1)
  dp[1][3] = 3*1*5 = 15  (burst balloon 1)
  dp[2][4] = 1*5*8 = 40  (burst balloon 5)
  dp[3][5] = 5*8*1 = 40  (burst balloon 8)

  Length 3 (two balloons):
  dp[0][3]: try k=1, k=2
    k=1: dp[0][1] + dp[1][3] + 1*3*5 = 0 + 15 + 15 = 30
    k=2: dp[0][2] + dp[2][3] + 1*1*5 = 1 + 0 + 5 = 6
    dp[0][3] = 30

  Continue building...

Step 4 (Optimal Order):

  Answer: dp[0][5] = 167

  Burst order (forward): 1 → 5 → 3 → 8

  Calculation:
  [1, 3, 1, 5, 8, 1]
       ↓ burst 1
  [1, 3, 5, 8, 1] → 3*1*5 = 15
       ↓ burst 5
  [1, 3, 8, 1] → 3*5*8 = 120
       ↓ burst 3
  [1, 8, 1] → 1*3*8 = 24
       ↓ burst 8
  [1, 1] → 1*8*1 = 8

  Total: 15 + 120 + 24 + 8 = 167 ✓

Key Insight:
- Think about LAST balloon in range, not first
- When k is last, boundaries are fixed (i and j)
- O(n³) time: O(n²) ranges × O(n) choices for k
```


### Examples
```
Input: nums = [3,1,5,8]
Output: 167
Burst order: 1, 5, 3, 8
3*1*5 + 3*5*8 + 1*3*8 + 1*8*1 = 15 + 120 + 24 + 8 = 167
```

### Intuition
```
Key insight: Think about which balloon to burst LAST in a range.

dp[i][j] = max coins for bursting all balloons in range (i, j) exclusive

If k is the last balloon burst in range (i, j):
dp[i][j] = max(dp[i][k] + dp[k][j] + nums[i]*nums[k]*nums[j])

Add virtual balloons with value 1 at boundaries.
```

### Solution
```python
def maxCoins(nums: list[int]) -> int:
    """
    Maximum coins from bursting balloons.

    Strategy:
    - Add boundary balloons with value 1
    - dp[i][j] = max coins for range (i, j) exclusive
    - Consider which balloon to burst LAST

    Time: O(n³)
    Space: O(n²)
    """
    # Add boundary balloons
    nums = [1] + nums + [1]
    n = len(nums)

    # dp[i][j] = max coins for balloons in range (i, j) exclusive
    dp = [[0] * n for _ in range(n)]

    # Iterate by range length
    for length in range(2, n):
        for i in range(n - length):
            j = i + length

            # Try each balloon k as the last one to burst
            for k in range(i + 1, j):
                coins = nums[i] * nums[k] * nums[j]
                dp[i][j] = max(dp[i][j], dp[i][k] + coins + dp[k][j])

    return dp[0][n - 1]
```

### Complexity
- **Time**: O(n³)
- **Space**: O(n²)

### Edge Cases
- Single balloon → return 1*val*1
- Two balloons → try both orders
- All same values → order doesn't matter
- Zeros in array → can strategically burst

---

## Problem 4: Longest Valid Parentheses (LC #32) - Hard

- [LeetCode](https://leetcode.com/problems/longest-valid-parentheses/)

### Video Explanation
- [NeetCode - Longest Valid Parentheses](https://www.youtube.com/watch?v=VdQuwtEd10M)

### Problem Statement
Find length of longest valid parentheses substring.

### Visual Intuition
```
Longest Valid Parentheses: s="(()" and "()(())"

Pattern: 1D DP - Track Valid Endings
Why: Only ')' can end a valid sequence

Step 0 (Define State):

  dp[i] = length of longest valid substring ENDING at i

  Key: '(' can never end valid sequence → dp[i] = 0
       ')' might end valid sequence → check previous

Step 1 (Example 1: s = "(()"):

  Index:  0   1   2
  Char:   (   (   )
  dp:     0   0   ?

  At index 2 (')'):
    Previous char s[1] = '(' → direct pair!
    dp[2] = 2 + dp[0] = 2 + 0 = 2

  Index:  0   1   2
  dp:     0   0   2

  Answer: 2

Step 2 (Example 2: s = "()(())"):

  Index:  0   1   2   3   4   5
  Char:   (   )   (   (   )   )
  dp:     0   ?   0   0   ?   ?

  At index 1 (')'):
    s[0] = '(' → pair!
    dp[1] = 2 + dp[-1] = 2 + 0 = 2

  At index 4 (')'):
    s[3] = '(' → pair!
    dp[4] = 2 + dp[2] = 2 + 0 = 2

  At index 5 (')'):
    s[4] = ')' → nested case!
    Look for matching '(' at: i - dp[i-1] - 1 = 5 - 2 - 1 = 2
    s[2] = '(' → match!
    dp[5] = dp[4] + 2 + dp[1] = 2 + 2 + 2 = 6

  Index:  0   1   2   3   4   5
  dp:     0   2   0   0   2   6

  Answer: 6

Transition Rules:
  ┌─────────────────────────────────────────────────┐
  │ Case 1: s[i-1] = '('  (pattern: ...() )         │
  │   dp[i] = 2 + dp[i-2]                           │
  │   Direct pair + previous valid                  │
  │                                                 │
  │ Case 2: s[i-1] = ')'  (pattern: ...)) )         │
  │   j = i - dp[i-1] - 1  (position before nested) │
  │   if s[j] = '(':                                │
  │     dp[i] = dp[i-1] + 2 + dp[j-1]              │
  │     inner valid + this pair + before this      │
  └─────────────────────────────────────────────────┘

Visual for Case 2:

  s = "( ( ) )"
       j     i
       0 1 2 3

  At i=3:
    dp[2] = 2 (inner valid)
    j = 3 - 2 - 1 = 0
    s[0] = '(' ✓
    dp[3] = dp[2] + 2 + dp[-1] = 2 + 2 + 0 = 4

Key Insight:
- Only process ')' characters
- Two patterns: direct pair "()" or nested "))"
- Combine: inner valid + current pair + before all
- O(n) time and space
```


### Examples
```
Input: s = "(()"
Output: 2 ("()")

Input: s = ")()())"
Output: 4 ("()()")
```

### Intuition
```
DP approach:
dp[i] = length of longest valid ending at index i

If s[i] == ')':
  If s[i-1] == '(': dp[i] = dp[i-2] + 2
  If s[i-1] == ')' and s[i-dp[i-1]-1] == '(':
     dp[i] = dp[i-1] + 2 + dp[i-dp[i-1]-2]

Stack approach:
Push indices of '(' and unmatched ')'
Valid length = current index - stack top
```

### Solution
```python
def longestValidParentheses(s: str) -> int:
    """
    Longest valid parentheses using DP.

    Strategy:
    - dp[i] = length of longest valid ending at i
    - Only ')' can end a valid sequence

    Time: O(n)
    Space: O(n)
    """
    n = len(s)
    if n == 0:
        return 0

    dp = [0] * n
    max_len = 0

    for i in range(1, n):
        if s[i] == ')':
            if s[i - 1] == '(':
                # Pattern: ...()
                dp[i] = (dp[i - 2] if i >= 2 else 0) + 2

            elif dp[i - 1] > 0:
                # Pattern: ...))
                j = i - dp[i - 1] - 1
                if j >= 0 and s[j] == '(':
                    dp[i] = dp[i - 1] + 2
                    if j >= 1:
                        dp[i] += dp[j - 1]

            max_len = max(max_len, dp[i])

    return max_len


def longestValidParentheses_stack(s: str) -> int:
    """
    Stack-based approach.

    Strategy:
    - Stack stores indices of unmatched characters
    - Valid length = current index - stack top

    Time: O(n)
    Space: O(n)
    """
    stack = [-1]  # Base for calculating length
    max_len = 0

    for i, char in enumerate(s):
        if char == '(':
            stack.append(i)
        else:
            stack.pop()

            if not stack:
                # No matching '(', push as new base
                stack.append(i)
            else:
                # Valid sequence length
                max_len = max(max_len, i - stack[-1])

    return max_len
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- Empty string → return 0
- All '(' → return 0
- All ')' → return 0
- "()" → return 2
- Nested valid → handle correctly

---

## Problem 5: Maximal Rectangle (LC #85) - Hard

- [LeetCode](https://leetcode.com/problems/maximal-rectangle/)

### Video Explanation
- [NeetCode - Maximal Rectangle](https://www.youtube.com/watch?v=g8bSdXCG-lA)

### Problem Statement
Find largest rectangle containing only 1's in binary matrix.

### Visual Intuition
```
Maximal Rectangle in Binary Matrix
matrix = ["10100",
          "10111",
          "11111",
          "10010"]

Pattern: Histogram per Row + Largest Rectangle in Histogram
Why: Reduce 2D problem to multiple 1D problems

Step 0 (Build Histograms):

  For each row, height = consecutive 1s above (including current)

  Row 0: [1,0,1,0,0]  ← heights start fresh

  Row 1: [2,0,2,1,1]  ← add 1 if current is '1'
          ↑   ↑ ↑ ↑
          1+1 1 1 1

  Row 2: [3,1,3,2,2]  ← continue building

  Row 3: [4,0,4,0,0]  ← reset to 0 if current is '0'
            ↑   ↑ ↑
           '0' '0''0'

Step 1 (Visualize Histograms):

  Row 0:        Row 1:        Row 2:        Row 3:
  █   █         █   █         █   █ █ █     █   █
              █   █ █ █     █ █ █ █ █     █   █
                            █ █ █ █ █     █   █
                                          █   █

  [1,0,1,0,0]   [2,0,2,1,1]   [3,1,3,2,2]   [4,0,4,0,0]

Step 2 (Apply Largest Rectangle in Histogram):

  For Row 2: heights = [3, 1, 3, 2, 2]

  Using monotonic stack:

  height=3:     height=1:     height=3:
    █             █             █   █
    █             █             █   █
    █           █ █           █ █ █

  Process height=1 (index 1):
    Pop height=3 (index 0): area = 3 × 1 = 3

  Process height=2 (index 3):
    Pop height=3 (index 2): area = 3 × 1 = 3

  At end, pop remaining:
    height=2: area = 2 × 2 = 4
    height=2: area = 2 × 3 = 6 ★ max!
    height=1: area = 1 × 5 = 5

Step 3 (Find Global Maximum):

  Row 0: max area = 1
  Row 1: max area = 3
  Row 2: max area = 6 ★ ANSWER
  Row 3: max area = 4

Answer: 6

Rectangle Visualization:

  matrix:           max rectangle:
  1 0 1 0 0         . . . . .
  1 0 1 1 1         . . ■ ■ ■
  1 1 1 1 1         . . ■ ■ ■
  1 0 0 1 0         . . . . .

  Area = 2 × 3 = 6

Key Insight:
- Build histogram heights row by row
- Each row: solve "Largest Rectangle in Histogram"
- Monotonic stack finds largest rectangle in O(n)
- Total: O(m × n) time
```


### Examples
```
Input: matrix = [
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
Output: 6
```

### Intuition
```
Build histogram for each row:
Row 0: [1,0,1,0,0]
Row 1: [2,0,2,1,1]
Row 2: [3,1,3,2,2]
Row 3: [4,0,0,3,0]

For each row, solve "Largest Rectangle in Histogram" problem.
```

### Solution
```python
def maximalRectangle(matrix: list[list[str]]) -> int:
    """
    Maximal rectangle in binary matrix.

    Strategy:
    - Build histogram heights for each row
    - For each row, solve largest rectangle in histogram

    Time: O(m * n)
    Space: O(n)
    """
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0

    for row in matrix:
        # Update histogram heights
        for j in range(cols):
            if row[j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0

        # Calculate max rectangle for this histogram
        max_area = max(max_area, largestRectangleArea(heights))

    return max_area


def largestRectangleArea(heights: list[int]) -> int:
    """
    Largest rectangle in histogram using monotonic stack.

    Time: O(n)
    Space: O(n)
    """
    stack = []
    max_area = 0
    heights = heights + [0]  # Sentinel

    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)

        stack.append(i)

    return max_area
```

### Complexity
- **Time**: O(m × n)
- **Space**: O(n)

### Edge Cases
- Empty matrix → return 0
- All zeros → return 0
- All ones → return m*n
- Single row/column → max consecutive 1s

---

## Problem 6: Wildcard Matching (LC #44) - Hard

- [LeetCode](https://leetcode.com/problems/wildcard-matching/)

### Video Explanation
- [NeetCode - Wildcard Matching](https://www.youtube.com/watch?v=3ZDZ-N0EPV0)

### Problem Statement
Implement wildcard pattern matching with '?' and '*'.
- '?' matches any single character
- '*' matches any sequence (including empty)

### Examples
```
Input: s = "aa", p = "a"
Output: false

Input: s = "aa", p = "*"
Output: true

Input: s = "cb", p = "?a"
Output: false
```


### Visual Intuition
```
Wildcard Matching: s = "adceb", p = "*a*b"

Pattern: 2D DP with Greedy '*' Matching
Why: '*' can match any sequence (including empty)

Step 0 (Understand Wildcards):

  '?' = matches exactly ONE character
  '*' = matches ANY sequence (including empty)

  Pattern "*a*b" can match:
  - "ab" (first * = empty, second * = empty)
  - "aab" (first * = empty, second * = "a")
  - "adceb" ✓ (first * = empty, second * = "dce")

Step 1 (Fill Base Cases):

        ""  *  a  *  b
    ""   T  T  F  F  F
             ↑
          '*' matches empty string

  Leading '*' can match empty: dp[0][j] = dp[0][j-1]

Step 2 (Handle '*' - Two Cases):

  For p[j] = '*':
  ┌─────────────────────────────────────────────────┐
  │ Case 1: Match ZERO characters                   │
  │   dp[i][j] = dp[i][j-1]                         │
  │   Skip the '*', match nothing                   │
  │                                                 │
  │ Case 2: Match ONE OR MORE characters            │
  │   dp[i][j] = dp[i-1][j]                         │
  │   "Consume" s[i-1], '*' stays to match more     │
  └─────────────────────────────────────────────────┘

Step 3 (Fill Table):

        ""  *  a  *  b
    ""   T  T  F  F  F
    a    F  ?

  dp[1][1]: p[0]='*'
    Case 1: dp[1][0] = F (zero chars)
    Case 2: dp[0][1] = T (one+ chars) ✓
    dp[1][1] = T

  dp[1][2]: p[1]='a', s[0]='a'
    Match! dp[1][2] = dp[0][1] = T

  Continue...

        ""  *  a  *  b
    ""   T  T  F  F  F
    a    F  T  T  T  F
    d    F  T  F  T  F  ← '*' absorbs 'd'
    c    F  T  F  T  F  ← '*' absorbs 'c'
    e    F  T  F  T  F  ← '*' absorbs 'e'
    b    F  T  F  T  T  ← 'b' matches 'b' ✓

Step 4 (Trace Match):

  s = "adceb", p = "*a*b"

  * → matches "" (empty)
  a → matches 'a'
  * → matches "dce"
  b → matches 'b'

  ✓ Full match!

Transition Diagram:

  For '*' at position j:
  ┌───────────────────────────────────────────────┐
  │     j-1    j                                  │
  │      ↓     ↓                                  │
  │ p:  ...   [*]                                 │
  │                                               │
  │ Match 0:  dp[i][j] ← dp[i][j-1]              │
  │           (skip '*')                          │
  │                                               │
  │ Match 1+: dp[i][j] ← dp[i-1][j]              │
  │           (consume s[i-1], '*' stays)         │
  │                                               │
  │ Result:   dp[i][j] = case1 OR case2           │
  └───────────────────────────────────────────────┘

Key Insight:
- '*' is simpler than regex '*' (no preceding char)
- '*' directly matches any sequence
- OR logic: zero chars OR one+ chars
- O(m×n) time and space
```

### Solution
```python
def isMatch(s: str, p: str) -> bool:
    """
    Wildcard pattern matching.

    Strategy:
    - dp[i][j] = True if s[0:i] matches p[0:j]
    - '*' can match empty or any sequence

    Time: O(m * n)
    Space: O(m * n)
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]

    dp[0][0] = True

    # Handle leading '*' matching empty string
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # '*' matches empty (dp[i][j-1]) or any char (dp[i-1][j])
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]

            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]

    return dp[m][n]
```

### Complexity
- **Time**: O(m × n)
- **Space**: O(m × n)

### Edge Cases
- Empty pattern → only matches empty string
- Pattern "*" → matches any string
- Pattern "?" → matches single char
- No wildcards → exact match needed

---

## Summary: Hard DP Problems

| # | Problem | Pattern | Key Insight |
|---|---------|---------|-------------|
| 1 | Edit Distance | 2D String DP | Three operations: insert, delete, replace |
| 2 | Regex Matching | 2D String DP | Handle '*' as zero or more |
| 3 | Burst Balloons | Interval DP | Think about last balloon to burst |
| 4 | Longest Valid Parens | 1D DP or Stack | Track valid endings |
| 5 | Maximal Rectangle | Histogram DP | Build histogram per row |
| 6 | Wildcard Matching | 2D String DP | '*' matches any sequence |

---

## Hard DP Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HARD DP PATTERNS                                         │
│                                                                             │
│  INTERVAL DP:                                                               │
│  • Burst Balloons: Which to process LAST in range                           │
│  • Matrix Chain Multiplication                                              │
│  • dp[i][j] = best for range [i, j]                                        │
│                                                                             │
│  STRING MATCHING DP:                                                        │
│  • Edit Distance, Regex, Wildcard                                           │
│  • dp[i][j] = match for s[0:i] and p[0:j]                                  │
│  • Handle special characters carefully                                      │
│                                                                             │
│  HISTOGRAM-BASED:                                                           │
│  • Maximal Rectangle: Build histogram per row                               │
│  • Use monotonic stack for each histogram                                   │
│                                                                             │
│  PARENTHESES DP:                                                            │
│  • Track valid endings                                                      │
│  • Consider what comes before current position                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
