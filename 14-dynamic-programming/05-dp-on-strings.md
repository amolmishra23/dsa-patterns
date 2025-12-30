# Dynamic Programming on Strings

## Common String DP Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STRING DP PATTERNS                                       │
│                                                                             │
│  1. EDIT DISTANCE: dp[i][j] = min ops for s1[:i] to s2[:j]                 │
│                                                                             │
│  2. LCS (Longest Common Subsequence):                                       │
│     dp[i][j] = length of LCS for s1[:i] and s2[:j]                         │
│                                                                             │
│  3. PALINDROME: dp[i][j] = is s[i:j+1] palindrome                          │
│                                                                             │
│  4. PATTERN MATCHING: dp[i][j] = does pattern[:j] match text[:i]           │
│                                                                             │
│  5. DISTINCT SUBSEQUENCES: dp[i][j] = ways to form t[:j] from s[:i]        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem 1: Edit Distance (LC #72) - Medium

### Problem Statement
Find minimum operations to convert word1 to word2.

### Examples
```
Input: word1 = "horse", word2 = "ros"
Output: 3 (horse → rorse → rose → ros)
```

### Solution
```python
def minDistance(word1: str, word2: str) -> int:
    """
    Find minimum edit distance (Levenshtein distance).

    State: dp[i][j] = min operations to convert word1[:i] to word2[:j]

    Transitions:
    - If word1[i-1] == word2[j-1]: dp[i][j] = dp[i-1][j-1]
    - Else: dp[i][j] = 1 + min(
        dp[i-1][j],     # Delete from word1
        dp[i][j-1],     # Insert to word1
        dp[i-1][j-1]    # Replace in word1
    )

    Time: O(m * n)
    Space: O(m * n), can optimize to O(n)
    """
    m, n = len(word1), len(word2)

    # dp[i][j] = edit distance for word1[:i] and word2[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: empty string conversions
    for i in range(m + 1):
        dp[i][0] = i  # Delete all characters
    for j in range(n + 1):
        dp[0][j] = j  # Insert all characters

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No operation needed
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Delete
                    dp[i][j - 1],      # Insert
                    dp[i - 1][j - 1]   # Replace
                )

    return dp[m][n]
```

### Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  word1 = "horse", word2 = "ros"                                             │
│                                                                             │
│      ""  r   o   s                                                          │
│  ""   0  1   2   3                                                          │
│  h    1  1   2   3                                                          │
│  o    2  2   1   2                                                          │
│  r    3  2   2   2                                                          │
│  s    4  3   3   2                                                          │
│  e    5  4   4   3  ← Answer                                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem 2: Longest Common Subsequence (LC #1143) - Medium

### Problem Statement
Find length of longest common subsequence.

### Examples
```
Input: text1 = "abcde", text2 = "ace"
Output: 3 ("ace")
```

### Solution
```python
def longestCommonSubsequence(text1: str, text2: str) -> int:
    """
    Find LCS length using 2D DP.

    State: dp[i][j] = LCS length for text1[:i] and text2[:j]

    Transitions:
    - If text1[i-1] == text2[j-1]: dp[i][j] = dp[i-1][j-1] + 1
    - Else: dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    Time: O(m * n)
    Space: O(m * n)
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def lcs_space_optimized(text1: str, text2: str) -> int:
    """
    Space-optimized version using 1D array.

    Time: O(m * n)
    Space: O(n)
    """
    m, n = len(text1), len(text2)

    # Ensure text2 is shorter for space optimization
    if m < n:
        text1, text2 = text2, text1
        m, n = n, m

    dp = [0] * (n + 1)

    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if text1[i - 1] == text2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    return dp[n]
```

---

## Problem 3: Longest Palindromic Subsequence (LC #516) - Medium

### Problem Statement
Find length of longest palindromic subsequence.

### Examples
```
Input: s = "bbbab"
Output: 4 ("bbbb")
```

### Solution
```python
def longestPalindromeSubseq(s: str) -> int:
    """
    Find longest palindromic subsequence.

    Key insight: LPS(s) = LCS(s, reverse(s))

    Alternative DP:
    State: dp[i][j] = LPS length for s[i:j+1]

    Time: O(n²)
    Space: O(n²)
    """
    n = len(s)

    # dp[i][j] = LPS length for s[i:j+1]
    dp = [[0] * n for _ in range(n)]

    # Base case: single characters
    for i in range(n):
        dp[i][i] = 1

    # Fill for increasing lengths
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

    return dp[0][n - 1]
```

---

## Problem 4: Distinct Subsequences (LC #115) - Hard

### Problem Statement
Count distinct subsequences of s that equal t.

### Examples
```
Input: s = "rabbbit", t = "rabbit"
Output: 3
```

### Solution
```python
def numDistinct(s: str, t: str) -> int:
    """
    Count distinct subsequences.

    State: dp[i][j] = ways to form t[:j] from s[:i]

    Transitions:
    - dp[i][j] = dp[i-1][j]  (skip s[i-1])
    - If s[i-1] == t[j-1]: dp[i][j] += dp[i-1][j-1]  (use s[i-1])

    Time: O(m * n)
    Space: O(n)
    """
    m, n = len(s), len(t)

    # dp[j] = ways to form t[:j]
    dp = [0] * (n + 1)
    dp[0] = 1  # Empty t can be formed 1 way

    for i in range(1, m + 1):
        # Traverse right to left to use previous row values
        for j in range(min(i, n), 0, -1):
            if s[i - 1] == t[j - 1]:
                dp[j] += dp[j - 1]

    return dp[n]
```

---

## Problem 5: Regular Expression Matching (LC #10) - Hard

### Problem Statement
Implement regex matching with '.' and '*'.

### Examples
```
Input: s = "aab", p = "c*a*b"
Output: true
```

### Solution
```python
def isMatch(s: str, p: str) -> bool:
    """
    Regular expression matching with '.' and '*'.

    State: dp[i][j] = does p[:j] match s[:i]

    Transitions:
    - If p[j-1] == '.' or p[j-1] == s[i-1]:
        dp[i][j] = dp[i-1][j-1]
    - If p[j-1] == '*':
        - Zero occurrences: dp[i][j] = dp[i][j-2]
        - One or more: if p[j-2] matches s[i-1]:
            dp[i][j] = dp[i-1][j]

    Time: O(m * n)
    Space: O(m * n)
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]

    # Base case: empty string matches empty pattern
    dp[0][0] = True

    # Base case: patterns like a*, a*b*, etc. can match empty string
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # Zero occurrences of previous char
                dp[i][j] = dp[i][j - 2]

                # One or more occurrences
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]

            elif p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]

    return dp[m][n]
```

---

## Problem 6: Wildcard Matching (LC #44) - Hard

### Problem Statement
Implement wildcard matching with '?' and '*'.

### Examples
```
Input: s = "adceb", p = "*a*b"
Output: true
```

### Solution
```python
def isMatch_wildcard(s: str, p: str) -> bool:
    """
    Wildcard matching with '?' (single char) and '*' (any sequence).

    State: dp[i][j] = does p[:j] match s[:i]

    Time: O(m * n)
    Space: O(n)
    """
    m, n = len(s), len(p)

    # dp[j] = does p[:j] match current prefix of s
    dp = [False] * (n + 1)
    dp[0] = True

    # Base case: leading *s can match empty string
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[j] = dp[j - 1]
        else:
            break

    for i in range(1, m + 1):
        new_dp = [False] * (n + 1)

        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # * matches empty or any sequence
                new_dp[j] = new_dp[j - 1] or dp[j]
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                new_dp[j] = dp[j - 1]

        dp = new_dp

    return dp[n]
```

---

## Problem 7: Shortest Common Supersequence (LC #1092) - Hard

### Problem Statement
Find shortest string containing both str1 and str2 as subsequences.

### Examples
```
Input: str1 = "abac", str2 = "cab"
Output: "cabac"
```

### Solution
```python
def shortestCommonSupersequence(str1: str, str2: str) -> str:
    """
    Find shortest common supersequence.

    Strategy:
    1. Find LCS
    2. Build SCS by including LCS once and remaining chars

    Time: O(m * n)
    Space: O(m * n)
    """
    m, n = len(str1), len(str2)

    # Build LCS DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Reconstruct SCS from DP table
    result = []
    i, j = m, n

    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            result.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            result.append(str1[i - 1])
            i -= 1
        else:
            result.append(str2[j - 1])
            j -= 1

    # Add remaining characters
    while i > 0:
        result.append(str1[i - 1])
        i -= 1
    while j > 0:
        result.append(str2[j - 1])
        j -= 1

    return ''.join(reversed(result))
```

---

## Problem 8: Interleaving String (LC #97) - Medium

### Problem Statement
Check if s3 is formed by interleaving s1 and s2.

### Examples
```
Input: s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
Output: true
```

### Solution
```python
def isInterleave(s1: str, s2: str, s3: str) -> bool:
    """
    Check if s3 is interleaving of s1 and s2.

    State: dp[i][j] = can s1[:i] and s2[:j] interleave to form s3[:i+j]

    Time: O(m * n)
    Space: O(n)
    """
    m, n = len(s1), len(s2)

    if m + n != len(s3):
        return False

    # dp[j] = can form s3[:i+j] using s1[:i] and s2[:j]
    dp = [False] * (n + 1)

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                dp[j] = True
            elif i == 0:
                dp[j] = dp[j - 1] and s2[j - 1] == s3[j - 1]
            elif j == 0:
                dp[j] = dp[j] and s1[i - 1] == s3[i - 1]
            else:
                dp[j] = (dp[j] and s1[i - 1] == s3[i + j - 1]) or \
                        (dp[j - 1] and s2[j - 1] == s3[i + j - 1])

    return dp[n]
```

---

## Summary: String DP Problems

| # | Problem | State | Time |
|---|---------|-------|------|
| 1 | Edit Distance | dp[i][j] = ops for s1[:i] → s2[:j] | O(mn) |
| 2 | LCS | dp[i][j] = LCS of s1[:i], s2[:j] | O(mn) |
| 3 | Palindrome Subseq | dp[i][j] = LPS of s[i:j+1] | O(n²) |
| 4 | Distinct Subseq | dp[i][j] = ways to form t[:j] | O(mn) |
| 5 | Regex Matching | dp[i][j] = p[:j] matches s[:i] | O(mn) |
| 6 | Wildcard | dp[i][j] = p[:j] matches s[:i] | O(mn) |
| 7 | SCS | LCS + reconstruction | O(mn) |
| 8 | Interleaving | dp[i][j] = can interleave | O(mn) |

---

## Practice More Problems

- [ ] LC #5 - Longest Palindromic Substring
- [ ] LC #647 - Palindromic Substrings
- [ ] LC #583 - Delete Operation for Two Strings
- [ ] LC #712 - Minimum ASCII Delete Sum
- [ ] LC #1312 - Minimum Insertion Steps to Make Palindrome

