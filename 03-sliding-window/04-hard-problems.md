# Sliding Window - Advanced Problems

## Advanced Sliding Window Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADVANCED SLIDING WINDOW TECHNIQUES                       │
│                                                                             │
│  1. AT MOST K Pattern:                                                      │
│     count(exactly k) = count(at most k) - count(at most k-1)               │
│                                                                             │
│  2. MINIMUM WINDOW Pattern:                                                 │
│     Expand to satisfy condition, shrink while still valid                   │
│                                                                             │
│  3. SLIDING WINDOW + HASH MAP:                                              │
│     Track character/element frequencies                                     │
│                                                                             │
│  4. SLIDING WINDOW + DEQUE:                                                 │
│     Track maximum/minimum in window                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem 1: Minimum Window Substring (LC #76) - Hard

- [LeetCode](https://leetcode.com/problems/minimum-window-substring/)

### Video Explanation
- [NeetCode - Minimum Window Substring](https://www.youtube.com/watch?v=jSto0O4AJbM)

### Problem Statement
Find minimum window in s containing all characters of t.

### Examples
```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
```


### Visual Intuition
```
Minimum Window Substring
s = "ADOBECODEBANC", t = "ABC"

Need: {A:1, B:1, C:1}
Expand right until valid, contract left to minimize:

  A D O B E C O D E B  A  N  C
  0 1 2 3 4 5 6 7 8 9 10 11 12
  L-----------R              "ADOBEC" has A,B,C ✓ len=6
    L---------R              "DOBEC" missing A ✗, expand
              L-----------R  "BANC" has A,B,C ✓ len=4

Track: have={}, need=3, formed=0
When formed == need → valid window, try shrinking

Answer: "BANC" (length 4)
```

### Solution
```python
from collections import Counter

def minWindow(s: str, t: str) -> str:
    """
    Find minimum window containing all characters of t.

    Strategy:
    - Expand window until all chars of t are included
    - Shrink from left while still valid
    - Track minimum window

    Time: O(m + n)
    Space: O(m + n)
    """
    if not t or not s:
        return ""

    # Count characters needed
    need = Counter(t)
    have = {}

    # Number of unique chars we need and currently have
    required = len(need)
    formed = 0

    left = 0
    min_len = float('inf')
    min_window = ""

    for right in range(len(s)):
        char = s[right]
        have[char] = have.get(char, 0) + 1

        # Check if this char satisfies requirement
        if char in need and have[char] == need[char]:
            formed += 1

        # Try to shrink window
        while formed == required:
            # Update minimum window
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_window = s[left:right + 1]

            # Shrink from left
            left_char = s[left]
            have[left_char] -= 1

            if left_char in need and have[left_char] < need[left_char]:
                formed -= 1

            left += 1

    return min_window
```

### Edge Cases
- t longer than s → return ""
- s equals t → return s
- No valid window → return ""
- t has duplicates → need all occurrences
- Single character match → return that character

---

## Problem 2: Sliding Window Maximum (LC #239) - Hard

- [LeetCode](https://leetcode.com/problems/sliding-window-maximum/)

### Video Explanation
- [NeetCode - Sliding Window Maximum](https://www.youtube.com/watch?v=DfljaUwZsOk)

### Problem Statement
Return maximum in each sliding window of size k.

### Examples
```
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
```


### Visual Intuition
```
Sliding Window Maximum (k=3)
nums = [1, 3, -1, -3, 5, 3, 6, 7]

Use monotonic decreasing deque (store indices):

Window [0,2]: deque=[1] → max=3
  1 < 3, pop 1, add 3
  -1 < 3, add -1 → deque=[3,-1]

Window [1,3]: deque=[3,-1,-3] → max=3
Window [2,4]: deque=[5] → max=5
  5 > all, clear deque

Window [3,5]: deque=[5,3] → max=5
Window [4,6]: deque=[6] → max=6
Window [5,7]: deque=[7] → max=7

Result: [3, 3, 5, 5, 6, 7]

Deque front = current max, remove if out of window
```

### Solution
```python
from collections import deque

def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    """
    Find maximum in each sliding window using monotonic deque.

    Strategy:
    - Deque stores indices in decreasing order of values
    - Front of deque is always the maximum
    - Remove indices outside window
    - Remove smaller elements when adding new one

    Time: O(n) - each element added and removed at most once
    Space: O(k)
    """
    result = []
    dq = deque()  # Stores indices

    for i in range(len(nums)):
        # Remove indices outside current window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements (they can't be maximum)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Add maximum to result (after first k elements)
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

### Edge Cases
- k = 1 → return first element n times
- k >= n → return entire array
- All same elements → return that element n times
- Strictly decreasing → deque always has one element

---

## Problem 3: Longest Substring with At Most K Distinct (LC #340) - Medium

- [LeetCode](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/)

### Video Explanation
- [NeetCode - Longest Substring with At Most K Distinct Characters](https://www.youtube.com/watch?v=nONCGxWoUfM)

### Problem Statement
Find longest substring with at most k distinct characters.

### Examples
```
Input: s = "eceba", k = 2
Output: 3 ("ece")
```


### Visual Intuition
```
Longest Substring with At Most K Distinct Characters
s = "eceba", k = 2

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Expand window until > k distinct, then shrink
             Track distinct count with frequency map
═══════════════════════════════════════════════════════════════

Step-by-Step:
─────────────
  Index: 0 1 2 3 4
  Chars: e c e b a

Step 0: right=0, char='e'
        ┌─┐
        │e│ c e b a
        └─┘
        freq = {e:1}, distinct=1 ≤ 2 ✓
        max_len = 1

Step 1: right=1, char='c'
        ┌───┐
        │e c│ e b a
        └───┘
        freq = {e:1, c:1}, distinct=2 ≤ 2 ✓
        max_len = 2

Step 2: right=2, char='e'
        ┌─────┐
        │e c e│ b a
        └─────┘
        freq = {e:2, c:1}, distinct=2 ≤ 2 ✓
        max_len = 3 ★

Step 3: right=3, char='b'
        ┌───────┐
        │e c e b│ a
        └───────┘
        freq = {e:2, c:1, b:1}, distinct=3 > 2 ✗

        SHRINK until distinct ≤ 2:
        left=1: remove 'e' → freq={e:1, c:1, b:1}, distinct=3 ✗
        left=2: remove 'c' → freq={e:1, b:1}, distinct=2 ✓
            ┌───┐
        e c │e b│ a
            └───┘
        max_len = 3 (unchanged)

Step 4: right=4, char='a'
            ┌─────┐
        e c │e b a│
            └─────┘
        freq = {e:1, b:1, a:1}, distinct=3 > 2 ✗

        SHRINK:
        left=3: remove 'e' → freq={b:1, a:1}, distinct=2 ✓
              ┌───┐
        e c e │b a│
              └───┘
        max_len = 3 (unchanged)

Answer: max_len = 3 (substring "ece")

WHY THIS WORKS:
════════════════
● Window always has ≤ k distinct characters
● Expand to find longer valid substrings
● Shrink when constraint violated
● Track max valid window length seen
```

### Solution
```python
def lengthOfLongestSubstringKDistinct(s: str, k: int) -> int:
    """
    Longest substring with at most k distinct characters.

    Strategy:
    - Expand window, track character counts
    - Shrink when more than k distinct

    Time: O(n)
    Space: O(k)
    """
    if k == 0:
        return 0

    char_count = {}
    left = 0
    max_length = 0

    for right in range(len(s)):
        char = s[right]
        char_count[char] = char_count.get(char, 0) + 1

        # Shrink if more than k distinct
        while len(char_count) > k:
            left_char = s[left]
            char_count[left_char] -= 1

            if char_count[left_char] == 0:
                del char_count[left_char]

            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length
```

### Edge Cases
- k = 0 → return 0
- k >= unique chars → return entire string
- All same character → return n
- Empty string → return 0

---

## Problem 4: Subarrays with K Different Integers (LC #992) - Hard

- [LeetCode](https://leetcode.com/problems/subarrays-with-k-different-integers/)

### Video Explanation
- [NeetCode - Subarrays with K Different Integers](https://www.youtube.com/watch?v=akwRFY2eyXs)

### Problem Statement
Count subarrays with exactly k different integers.

### Examples
```
Input: nums = [1,2,1,2,3], k = 2
Output: 7
```


### Visual Intuition
```
Subarrays with Exactly K Different Integers
nums = [1, 2, 1, 2, 3], k = 2

═══════════════════════════════════════════════════════════════
KEY INSIGHT: exactly(k) = atMost(k) - atMost(k-1)
═══════════════════════════════════════════════════════════════

Step 1: Count atMost(2) - subarrays with ≤2 distinct
─────────────────────────────────────────────────────
  Index:  0   1   2   3   4
  Array: [1] [2] [1] [2] [3]
          ↓
  i=0: L=0, window=[1]         distinct=1 ≤ 2 ✓
       Subarrays ending at 0: [1]                    count += 1

  i=1: L=0, window=[1,2]       distinct=2 ≤ 2 ✓
       Subarrays: [2], [1,2]                         count += 2

  i=2: L=0, window=[1,2,1]     distinct=2 ≤ 2 ✓
       Subarrays: [1], [2,1], [1,2,1]                count += 3

  i=3: L=0, window=[1,2,1,2]   distinct=2 ≤ 2 ✓
       Subarrays: [2], [1,2], [2,1,2], [1,2,1,2]     count += 4

  i=4: L=0, window=[1,2,1,2,3] distinct=3 > 2 ✗
       Shrink until ≤ 2:
       L=1: [2,1,2,3] distinct=3 ✗
       L=2: [1,2,3]   distinct=3 ✗
       L=3: [2,3]     distinct=2 ✓
       Subarrays: [3], [2,3]                         count += 2

  atMost(2) = 1 + 2 + 3 + 4 + 2 = 12

Step 2: Count atMost(1) - subarrays with ≤1 distinct
─────────────────────────────────────────────────────
  i=0: [1]         count += 1
  i=1: [2]         count += 1  (shrink past [1])
  i=2: [1]         count += 1  (shrink past [2])
  i=3: [2]         count += 1  (shrink past [1])
  i=4: [3]         count += 1  (shrink past [2])

  atMost(1) = 5

Step 3: Calculate exactly(2)
─────────────────────────────
  exactly(2) = atMost(2) - atMost(1) = 12 - 5 = 7

  The 7 subarrays: [1,2], [2,1], [1,2,1], [2,1,2],
                   [1,2,1,2], [2,3], [2,1,2,3]... wait
  Actually: [1,2], [2,1], [1,2,1], [2,1,2], [1,2,1,2], [2,3], [1,2] ✓

WHY THIS WORKS:
════════════════
● atMost(k) counts ALL subarrays with 0,1,2,...k distinct
● atMost(k-1) counts subarrays with 0,1,2,...k-1 distinct
● Subtracting removes overlap, leaving EXACTLY k distinct
● Formula: count += (right - left + 1) at each step
```

### Solution
```python
def subarraysWithKDistinct(nums: list[int], k: int) -> int:
    """
    Count subarrays with exactly k distinct integers.

    Key insight: exactly(k) = atMost(k) - atMost(k-1)

    Time: O(n)
    Space: O(k)
    """
    def at_most_k(k: int) -> int:
        """Count subarrays with at most k distinct integers."""
        count = {}
        left = 0
        result = 0

        for right in range(len(nums)):
            num = nums[right]
            count[num] = count.get(num, 0) + 1

            while len(count) > k:
                left_num = nums[left]
                count[left_num] -= 1
                if count[left_num] == 0:
                    del count[left_num]
                left += 1

            # All subarrays ending at right with at most k distinct
            result += right - left + 1

        return result

    return at_most_k(k) - at_most_k(k - 1)
```

### Edge Cases
- k = 0 → count subarrays with all same elements
- k > unique elements → return 0
- All same elements → return n*(n+1)/2 if k >= 1
- Single element → return 1 if k >= 1

---

## Problem 5: Longest Repeating Character Replacement (LC #424) - Medium

- [LeetCode](https://leetcode.com/problems/longest-repeating-character-replacement/)

### Video Explanation
- [NeetCode - Longest Repeating Character Replacement](https://www.youtube.com/watch?v=gqXU1UyA8pk)

### Problem Statement
Longest substring with same letter after at most k replacements.

### Examples
```
Input: s = "AABABBA", k = 1
Output: 4 ("AABA" → "AAAA")
```


### Visual Intuition
```
Longest Repeating Character Replacement
s = "AABABBA", k = 1 (can replace at most k characters)

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Window valid when (window_size - max_freq) ≤ k
             This means: chars_to_replace ≤ k
═══════════════════════════════════════════════════════════════

Step-by-Step Window Expansion:
──────────────────────────────
  Index: 0 1 2 3 4 5 6
  Chars: A A B A B B A
         ↓

Step 0: right=0, char='A'
        ┌─┐
        │A│ A B A B B A
        └─┘
        freq={A:1}, max_freq=1
        size=1, replace=1-1=0 ≤ 1 ✓
        max_len = 1

Step 1: right=1, char='A'
        ┌───┐
        │A A│ B A B B A
        └───┘
        freq={A:2}, max_freq=2
        size=2, replace=2-2=0 ≤ 1 ✓
        max_len = 2

Step 2: right=2, char='B'
        ┌─────┐
        │A A B│ A B B A
        └─────┘
        freq={A:2,B:1}, max_freq=2
        size=3, replace=3-2=1 ≤ 1 ✓  (replace 1 B with A → AAA)
        max_len = 3

Step 3: right=3, char='A'
        ┌───────┐
        │A A B A│ B B A
        └───────┘
        freq={A:3,B:1}, max_freq=3
        size=4, replace=4-3=1 ≤ 1 ✓  (replace 1 B → AAAA)
        max_len = 4

Step 4: right=4, char='B'
        ┌─────────┐
        │A A B A B│ B A
        └─────────┘
        freq={A:3,B:2}, max_freq=3
        size=5, replace=5-3=2 > 1 ✗  INVALID! Shrink left

        Shrink: left=1
          ┌───────┐
          A│A B A B│ B A
            └───────┘
          freq={A:2,B:2}, max_freq=2
          size=4, replace=4-2=2 > 1 ✗  Still invalid!

        Shrink: left=2
            ┌─────┐
          A A│B A B│ B A
              └─────┘
          freq={A:1,B:2}, max_freq=2
          size=3, replace=3-2=1 ≤ 1 ✓
          max_len = 4 (unchanged)

[Continue similarly for remaining characters...]

Final Answer: max_len = 4
              Window "ABBA" → replace A with B → "BBBB" ✓

WHY THIS WORKS:
════════════════
● max_freq = most frequent char in window (the one we KEEP)
● window_size - max_freq = chars we need to REPLACE
● If replacements needed ≤ k, window is valid
● We don't need to track WHICH char to keep - just keep the most frequent!
```

### Solution
```python
def characterReplacement(s: str, k: int) -> int:
    """
    Longest substring after k replacements.

    Key insight: Valid window if (window_size - max_freq) <= k

    Strategy:
    - Track frequency of each character in window
    - Window is valid if we need at most k replacements
    - Shrink when invalid

    Time: O(n)
    Space: O(26) = O(1)
    """
    count = {}
    left = 0
    max_freq = 0
    max_length = 0

    for right in range(len(s)):
        char = s[right]
        count[char] = count.get(char, 0) + 1
        max_freq = max(max_freq, count[char])

        # Window size - max frequency = chars to replace
        window_size = right - left + 1

        if window_size - max_freq > k:
            # Invalid window, shrink from left
            count[s[left]] -= 1
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length
```

### Edge Cases
- k >= n → return n (can replace all)
- All same character → return n
- k = 0 → find longest same-char substring
- Empty string → return 0

---

## Problem 6: Find All Anagrams (LC #438) - Medium

- [LeetCode](https://leetcode.com/problems/find-all-anagrams-in-a-string/)

### Video Explanation
- [NeetCode - Find All Anagrams in a String](https://www.youtube.com/watch?v=G8xtZy0fDKg)

### Problem Statement
Find all start indices of p's anagrams in s.

### Examples
```
Input: s = "cbaebabacd", p = "abc"
Output: [0, 6]
```


### Visual Intuition
```
Find All Anagrams in String
s = "cbaebabacd", p = "abc"

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Fixed window of size len(p), slide and compare
═══════════════════════════════════════════════════════════════

Target: p = "abc" → need = {a:1, b:1, c:1}
Window size = 3 (fixed)

Step-by-Step Sliding:
─────────────────────
  Index: 0 1 2 3 4 5 6 7 8 9
  Chars: c b a e b a b a c d

Window 0 [0,2]: "cba"
  ┌─────┐
  │c b a│ e b a b a c d
  └─────┘
  have = {c:1, b:1, a:1}
  have == need? ✓ YES! → result.append(0)

Window 1 [1,3]: "bae"
    ┌─────┐
  c │b a e│ b a b a c d
    └─────┘
  Remove 'c', add 'e'
  have = {b:1, a:1, e:1}
  have == need? ✗ (has 'e', missing 'c')

Window 2 [2,4]: "aeb"
      ┌─────┐
  c b │a e b│ a b a c d
      └─────┘
  have = {a:1, e:1, b:1}
  have == need? ✗

Window 3 [3,5]: "eba"
        ┌─────┐
  c b a │e b a│ b a c d
        └─────┘
  have = {e:1, b:1, a:1}
  have == need? ✗

Window 4 [4,6]: "bab"
          ┌─────┐
  c b a e │b a b│ a c d
          └─────┘
  have = {b:2, a:1}
  have == need? ✗ (b:2 ≠ b:1)

Window 5 [5,7]: "aba"
            ┌─────┐
  c b a e b │a b a│ c d
            └─────┘
  have = {a:2, b:1}
  have == need? ✗

Window 6 [6,8]: "bac"
              ┌─────┐
  c b a e b a │b a c│ d
              └─────┘
  have = {b:1, a:1, c:1}
  have == need? ✓ YES! → result.append(6)

Window 7 [7,9]: "acd"
                ┌─────┐
  c b a e b a b │a c d│
                └─────┘
  have = {a:1, c:1, d:1}
  have == need? ✗

Result: [0, 6]

WHY THIS WORKS:
════════════════
● Anagram = same chars with same frequencies (order doesn't matter)
● Fixed window ensures we check substrings of exact length
● Compare frequency maps instead of sorting (O(1) vs O(k log k))
● Optimization: Track "matches" count instead of full map comparison
```

### Solution
```python
def findAnagrams(s: str, p: str) -> list[int]:
    """
    Find all anagram start indices using sliding window.

    Strategy:
    - Fixed window size = len(p)
    - Track character counts
    - Compare window count with target count

    Time: O(n)
    Space: O(26) = O(1)
    """
    if len(p) > len(s):
        return []

    result = []
    p_count = Counter(p)
    window_count = Counter()

    for i in range(len(s)):
        # Add right character
        window_count[s[i]] += 1

        # Remove left character if window too big
        if i >= len(p):
            left_char = s[i - len(p)]
            window_count[left_char] -= 1
            if window_count[left_char] == 0:
                del window_count[left_char]

        # Check if anagram
        if window_count == p_count:
            result.append(i - len(p) + 1)

    return result
```

### Edge Cases
- p longer than s → return []
- p equals s → return [0] if anagram
- No anagrams → return []
- p has duplicates → need all occurrences

---

## Problem 7: Permutation in String (LC #567) - Medium

- [LeetCode](https://leetcode.com/problems/permutation-in-string/)

### Video Explanation
- [NeetCode - Permutation in String](https://www.youtube.com/watch?v=UbyhOgBN834)

### Problem Statement
Check if s2 contains permutation of s1.

### Examples
```
Input: s1 = "ab", s2 = "eidbaooo"
Output: true ("ba" is permutation of "ab")
```


### Visual Intuition
```
Permutation in String
s1 = "ab", s2 = "eidbaooo"

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Permutation = anagram = same frequency in any order
═══════════════════════════════════════════════════════════════

Target: s1 = "ab" → need = {a:1, b:1}
Window size = 2 (fixed = len(s1))

Step-by-Step Sliding:
─────────────────────
  Index: 0 1 2 3 4 5 6 7
  Chars: e i d b a o o o

Window 0 [0,1]: "ei"
  ┌───┐
  │e i│ d b a o o o
  └───┘
  have = {e:1, i:1}
  Match? ✗

Window 1 [1,2]: "id"
    ┌───┐
  e │i d│ b a o o o
    └───┘
  have = {i:1, d:1}
  Match? ✗

Window 2 [2,3]: "db"
      ┌───┐
  e i │d b│ a o o o
      └───┘
  have = {d:1, b:1}
  Match? ✗ (has 'd', missing 'a')

Window 3 [3,4]: "ba"
        ┌───┐
  e i d │b a│ o o o
        └───┘
  have = {b:1, a:1}
  Match? ✓ YES! → return True

  ★ FOUND: "ba" is permutation of "ab" ★

Optimized Approach - Track Matches:
───────────────────────────────────
Instead of comparing full maps, track how many chars match:

  s1_count = [0]*26, s2_count = [0]*26

  Initial (first window):
    s1: a=1, b=1
    s2: e=1, i=1
    matches = 24 (all chars except a,b,e,i match at 0)

  Slide window:
    Remove old char, add new char
    Update matches count
    If matches == 26 → found permutation!

WHY THIS WORKS:
════════════════
● Permutation means exact same character frequencies
● Fixed window ensures correct length
● Matches optimization: O(1) check instead of O(26) map compare
● Early termination: return True as soon as found
```

### Solution
```python
def checkInclusion(s1: str, s2: str) -> bool:
    """
    Check if s2 contains permutation of s1.

    Strategy:
    - Fixed window of size len(s1)
    - Track matches between window and target

    Time: O(n)
    Space: O(26) = O(1)
    """
    if len(s1) > len(s2):
        return False

    s1_count = [0] * 26
    window_count = [0] * 26

    # Initialize counts for s1 and first window
    for i in range(len(s1)):
        s1_count[ord(s1[i]) - ord('a')] += 1
        window_count[ord(s2[i]) - ord('a')] += 1

    if s1_count == window_count:
        return True

    # Slide window
    for i in range(len(s1), len(s2)):
        # Add right character
        window_count[ord(s2[i]) - ord('a')] += 1

        # Remove left character
        window_count[ord(s2[i - len(s1)]) - ord('a')] -= 1

        if s1_count == window_count:
            return True

    return False
```

### Edge Cases
- s1 longer than s2 → return False
- s1 equals s2 → return True
- s1 is single char → check if in s2
- No permutation exists → return False

---

## Problem 8: Fruit Into Baskets (LC #904) - Medium

- [LeetCode](https://leetcode.com/problems/fruit-into-baskets/)

### Video Explanation
- [NeetCode - Fruit Into Baskets](https://www.youtube.com/watch?v=yYtaV0G3mWQ)

### Problem Statement
Maximum fruits with at most 2 types.


### Visual Intuition
```
Fruit Into Baskets (at most 2 types)
fruits = [1, 2, 3, 2, 2]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Same as "longest subarray with at most K=2 distinct"
             Two baskets = two distinct fruit types allowed
═══════════════════════════════════════════════════════════════

Visualization with Baskets:
───────────────────────────
  🧺 Basket A: [type ?]
  🧺 Basket B: [type ?]

  Index: 0 1 2 3 4
  Fruit: 1 2 3 2 2
         🍎🍊🍋🍊🍊

Step-by-Step:
─────────────
Step 0: right=0, fruit=1(🍎)
        ┌─┐
        │1│ 2 3 2 2
        └─┘
        Baskets: {1:1}  types=1 ≤ 2 ✓
        len = 1

Step 1: right=1, fruit=2(🍊)
        ┌───┐
        │1 2│ 3 2 2
        └───┘
        Baskets: {1:1, 2:1}  types=2 ≤ 2 ✓
        len = 2

Step 2: right=2, fruit=3(🍋)
        ┌─────┐
        │1 2 3│ 2 2
        └─────┘
        Baskets: {1:1, 2:1, 3:1}  types=3 > 2 ✗

        OVERFLOW! Must empty one basket:
        Shrink left until types ≤ 2

        left=1: remove fruit[0]=1
          ┌───┐
        1 │2 3│ 2 2
          └───┘
        Baskets: {2:1, 3:1}  types=2 ≤ 2 ✓
        len = 2

Step 3: right=3, fruit=2(🍊)
          ┌─────┐
        1 │2 3 2│ 2
          └─────┘
        Baskets: {2:2, 3:1}  types=2 ≤ 2 ✓
        len = 3

Step 4: right=4, fruit=2(🍊)
          ┌───────┐
        1 │2 3 2 2│
          └───────┘
        Baskets: {2:3, 3:1}  types=2 ≤ 2 ✓
        len = 4 ← MAX!

Answer: 4 fruits (subarray [2,3,2,2])

WHY THIS WORKS:
════════════════
● "2 baskets" = at most 2 distinct types in window
● Expand window to collect more fruits
● Shrink when we have too many types (> 2)
● Track maximum valid window length
```

### Solution
```python
def totalFruit(fruits: list[int]) -> int:
    """
    Maximum fruits with at most 2 types (at most 2 distinct).

    This is "longest subarray with at most 2 distinct elements".

    Time: O(n)
    Space: O(1)
    """
    count = {}
    left = 0
    max_fruits = 0

    for right in range(len(fruits)):
        fruit = fruits[right]
        count[fruit] = count.get(fruit, 0) + 1

        while len(count) > 2:
            left_fruit = fruits[left]
            count[left_fruit] -= 1
            if count[left_fruit] == 0:
                del count[left_fruit]
            left += 1

        max_fruits = max(max_fruits, right - left + 1)

    return max_fruits
```

### Edge Cases
- All same fruit → return n
- Only 2 types total → return n
- Alternating types → depends on pattern
- Empty array → return 0

---

## Problem 9: Max Consecutive Ones III (LC #1004) - Medium

- [LeetCode](https://leetcode.com/problems/max-consecutive-ones-iii/)

### Video Explanation
- [NeetCode - Max Consecutive Ones III](https://www.youtube.com/watch?v=3E4JBHSLpYk)

### Problem Statement
Maximum consecutive 1s after flipping at most k 0s.


### Visual Intuition
```
Max Consecutive Ones III (can flip at most k zeros to ones)
nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Window valid when zeros_in_window ≤ k
             We're finding longest window with at most k zeros
═══════════════════════════════════════════════════════════════

  Index:  0 1 2 3 4 5 6 7 8 9 10
  Array: [1,1,1,0,0,0,1,1,1,1,0]
          █ █ █ ░ ░ ░ █ █ █ █ ░
          (█=1, ░=0)

Step-by-Step:
─────────────
Step 0-2: Expand through initial 1s
          ┌─────┐
          │1 1 1│ 0 0 0 1 1 1 1 0
          └─────┘
          zeros=0 ≤ 2 ✓, len=3

Step 3-4: Add two 0s (can flip both)
          ┌─────────┐
          │1 1 1 0 0│ 0 1 1 1 1 0
          └─────────┘
          zeros=2 ≤ 2 ✓, len=5
          Can flip: █ █ █ ░→█ ░→█ = █████

Step 5: Add third 0 → INVALID!
          ┌───────────┐
          │1 1 1 0 0 0│ 1 1 1 1 0
          └───────────┘
          zeros=3 > 2 ✗

          Shrink until valid:
          left=1: still 3 zeros
          left=2: still 3 zeros
          left=3: zeros=2 ≤ 2 ✓
                ┌───────┐
          1 1 1 │0 0 0 1│ 1 1 1 0
                └───────┘
          len=4

Step 6-9: Expand through 1s
                ┌───────────────┐
          1 1 1 │0 0 0 1 1 1 1│ 0
                └───────────────┘
          zeros=2 ≤ 2 ✓, len=8... wait let me recalculate

          Actually at step 6:
                  ┌─────────┐
          1 1 1 0│0 0 1 1 1 1│ 0
                  └─────────┘
          zeros=2, len=6 ← MAX found here!

Step 10: Add final 0 → shrink again
          Final max_len = 6

Answer: 6 (flip zeros at indices 4,5 → "0 0 1 1 1 1" becomes "1 1 1 1 1 1")

Before: 1 1 1 0 ░ ░ █ █ █ █ 0
After:  1 1 1 0 █ █ █ █ █ █ 0  (flipped 2 zeros)
                └─────────┘
                 6 consecutive

WHY THIS WORKS:
════════════════
● We're not actually flipping - just counting zeros in window
● If zeros ≤ k, we COULD flip them all → all 1s in window
● Longest such window = answer
● Same pattern as "at most k distinct" but tracking zeros specifically
```

### Solution
```python
def longestOnes(nums: list[int], k: int) -> int:
    """
    Maximum consecutive 1s after flipping at most k zeros.

    Strategy:
    - Window can have at most k zeros
    - Shrink when zeros exceed k

    Time: O(n)
    Space: O(1)
    """
    left = 0
    zeros = 0
    max_length = 0

    for right in range(len(nums)):
        if nums[right] == 0:
            zeros += 1

        while zeros > k:
            if nums[left] == 0:
                zeros -= 1
            left += 1

        max_length = max(max_length, right - left + 1)

    return max_length
```

### Edge Cases
- All ones → return n
- All zeros, k >= n → return n
- k = 0 → find longest consecutive ones
- No zeros → return n

---

## Problem 10: Minimum Size Subarray Sum (LC #209) - Medium

- [LeetCode](https://leetcode.com/problems/minimum-size-subarray-sum/)

### Video Explanation
- [NeetCode - Minimum Size Subarray Sum](https://www.youtube.com/watch?v=aYqYMIqZx5s)

### Problem Statement
Find minimum length subarray with sum >= target.


### Visual Intuition
```
Minimum Size Subarray Sum ≥ target
nums = [2,3,1,2,4,3], target = 7

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Expand until valid (sum ≥ target), shrink to minimize
             This is MINIMUM window, so shrink while STILL valid
═══════════════════════════════════════════════════════════════

  Index: 0 1 2 3 4 5
  Array: 2 3 1 2 4 3

Step-by-Step:
─────────────
Step 0: right=0, add 2
        ┌─┐
        │2│ 3 1 2 4 3     sum=2 < 7 ✗ expand
        └─┘

Step 1: right=1, add 3
        ┌───┐
        │2 3│ 1 2 4 3     sum=5 < 7 ✗ expand
        └───┘

Step 2: right=2, add 1
        ┌─────┐
        │2 3 1│ 2 4 3     sum=6 < 7 ✗ expand
        └─────┘

Step 3: right=3, add 2
        ┌───────┐
        │2 3 1 2│ 4 3     sum=8 ≥ 7 ✓ len=4, min=4
        └───────┘

        Try shrink: remove 2 (left)
          ┌─────┐
        2 │3 1 2│ 4 3     sum=6 < 7 ✗ can't shrink more
          └─────┘

Step 4: right=4, add 4
          ┌───────┐
        2 │3 1 2 4│ 3     sum=10 ≥ 7 ✓ len=4
          └───────┘

        Shrink: remove 3
            ┌─────┐
        2 3 │1 2 4│ 3     sum=7 ≥ 7 ✓ len=3, min=3
            └─────┘

        Shrink: remove 1
              ┌───┐
        2 3 1 │2 4│ 3     sum=6 < 7 ✗ stop shrinking
              └───┘

Step 5: right=5, add 3
              ┌─────┐
        2 3 1 │2 4 3│     sum=9 ≥ 7 ✓ len=3
              └─────┘

        Shrink: remove 2
                ┌───┐
        2 3 1 2 │4 3│     sum=7 ≥ 7 ✓ len=2, min=2 ← NEW MIN!
                └───┘

        Shrink: remove 4
                  ┌─┐
        2 3 1 2 4 │3│     sum=3 < 7 ✗ stop
                  └─┘

Answer: min_len = 2 (subarray [4,3])

Visualization of Answer:
────────────────────────
        2  3  1  2  4  3
                   └──┘
                   4+3=7 ≥ 7 ✓
                   Length = 2 (minimum!)

WHY THIS WORKS:
════════════════
● Expand: grow window until condition met (sum ≥ target)
● Shrink: reduce window while STILL valid to find minimum
● Different from max problems where we shrink when INVALID
● Two-pointer ensures O(n): each element added/removed at most once
```

### Solution
```python
def minSubArrayLen(target: int, nums: list[int]) -> int:
    """
    Minimum length subarray with sum >= target.

    Strategy:
    - Expand until sum >= target
    - Shrink while sum still >= target
    - Track minimum length

    Time: O(n)
    Space: O(1)
    """
    left = 0
    current_sum = 0
    min_length = float('inf')

    for right in range(len(nums)):
        current_sum += nums[right]

        while current_sum >= target:
            min_length = min(min_length, right - left + 1)
            current_sum -= nums[left]
            left += 1

    return min_length if min_length != float('inf') else 0
```

### Edge Cases
- No valid subarray → return 0
- Single element >= target → return 1
- Entire array needed → return n
- target = 0 → return 0 (empty subarray)

---

## Summary: Advanced Sliding Window

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Minimum Window | Expand/shrink with counts | O(n) |
| 2 | Sliding Maximum | Monotonic deque | O(n) |
| 3 | At Most K Distinct | Shrink when > k | O(n) |
| 4 | Exactly K Distinct | atMost(k) - atMost(k-1) | O(n) |
| 5 | Character Replacement | max_freq optimization | O(n) |
| 6 | Find Anagrams | Fixed window + count | O(n) |
| 7 | Permutation in String | Fixed window + count | O(n) |
| 8 | Fruit Baskets | At most 2 distinct | O(n) |
| 9 | Max Ones III | At most k zeros | O(n) |
| 10 | Min Subarray Sum | Shrink while valid | O(n) |

---

## Practice More Problems

- [ ] LC #30 - Substring with Concatenation of All Words
- [ ] LC #159 - Longest Substring with At Most Two Distinct Characters
- [ ] LC #395 - Longest Substring with At Least K Repeating Characters
- [ ] LC #480 - Sliding Window Median
- [ ] LC #1438 - Longest Continuous Subarray With Absolute Diff <= Limit

