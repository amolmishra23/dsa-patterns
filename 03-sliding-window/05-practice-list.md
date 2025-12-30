# Sliding Window - Complete Practice List

## Organized by Pattern and Difficulty

### Pattern 1: Fixed Window Size

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 643 | [Max Average Subarray I](https://leetcode.com/problems/maximum-average-subarray-i/) | Easy | Fixed window sum |
| 1456 | [Max Vowels in Substring](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/) | Medium | Count vowels in window |
| 1343 | [Subarrays of Size K with Avg >= Threshold](https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/) | Medium | Fixed window average |
| 239 | [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) | Hard | Monotonic deque |
| 480 | [Sliding Window Median](https://leetcode.com/problems/sliding-window-median/) | Hard | Two heaps |

### Pattern 2: Variable Window (Shrink When Invalid)

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 3 | [Longest Substring Without Repeating](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | Medium | Hash set for chars |
| 159 | [Longest Substring with At Most 2 Distinct](https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/) | Medium | Hash map count |
| 340 | [Longest Substring with At Most K Distinct](https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/) | Medium | Hash map count |
| 424 | [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/) | Medium | Max freq optimization |
| 1004 | [Max Consecutive Ones III](https://leetcode.com/problems/max-consecutive-ones-iii/) | Medium | Count zeros in window |
| 1208 | [Get Equal Substrings Within Budget](https://leetcode.com/problems/get-equal-substrings-within-budget/) | Medium | Cost within budget |

### Pattern 3: Variable Window (Find Minimum)

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 76 | [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) | Hard | Expand then shrink |
| 209 | [Minimum Size Subarray Sum](https://leetcode.com/problems/minimum-size-subarray-sum/) | Medium | Sum >= target |
| 862 | [Shortest Subarray with Sum at Least K](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/) | Hard | Monotonic deque + prefix |

### Pattern 4: String Matching/Anagram

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 438 | [Find All Anagrams](https://leetcode.com/problems/find-all-anagrams-in-a-string/) | Medium | Fixed window + count |
| 567 | [Permutation in String](https://leetcode.com/problems/permutation-in-string/) | Medium | Fixed window + count |
| 30 | [Substring with Concatenation](https://leetcode.com/problems/substring-with-concatenation-of-all-words/) | Hard | Word-level sliding |

### Pattern 5: Counting Subarrays

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 713 | [Subarray Product Less Than K](https://leetcode.com/problems/subarray-product-less-than-k/) | Medium | Count subarrays ending at right |
| 992 | [Subarrays with K Different Integers](https://leetcode.com/problems/subarrays-with-k-different-integers/) | Hard | atMost(k) - atMost(k-1) |
| 1248 | [Count Nice Subarrays](https://leetcode.com/problems/count-number-of-nice-subarrays/) | Medium | Same as k different |

---

## Essential Templates

### 1. Fixed Window
```python
def fixed_window(arr: list, k: int) -> int:
    """
    Template for fixed window size problems.

    Time: O(n)
    Space: O(1)
    """
    n = len(arr)
    if n < k:
        return -1

    # Initialize first window
    window_sum = sum(arr[:k])
    result = window_sum

    # Slide window
    for i in range(k, n):
        window_sum += arr[i] - arr[i - k]
        result = max(result, window_sum)

    return result
```

### 2. Variable Window (Maximum Length)
```python
def variable_window_max(s: str) -> int:
    """
    Find longest valid substring.
    Expand right, shrink left when invalid.

    Time: O(n)
    Space: O(k) where k = distinct elements
    """
    char_count = {}
    left = 0
    max_length = 0

    for right in range(len(s)):
        # Add right element
        char_count[s[right]] = char_count.get(s[right], 0) + 1

        # Shrink while invalid
        while is_invalid(char_count):
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1

        # Update result
        max_length = max(max_length, right - left + 1)

    return max_length
```

### 3. Variable Window (Minimum Length)
```python
def variable_window_min(s: str, target: str) -> str:
    """
    Find shortest valid substring.
    Expand until valid, then shrink while valid.

    Time: O(n)
    Space: O(k)
    """
    need = {}
    for c in target:
        need[c] = need.get(c, 0) + 1

    have = {}
    have_count = 0
    need_count = len(need)

    left = 0
    min_len = float('inf')
    result = ""

    for right in range(len(s)):
        # Add right element
        c = s[right]
        have[c] = have.get(c, 0) + 1

        if c in need and have[c] == need[c]:
            have_count += 1

        # Shrink while valid
        while have_count == need_count:
            # Update result
            if right - left + 1 < min_len:
                min_len = right - left + 1
                result = s[left:right + 1]

            # Remove left element
            c = s[left]
            have[c] -= 1
            if c in need and have[c] < need[c]:
                have_count -= 1
            left += 1

    return result
```

### 4. Anagram Detection
```python
def findAnagrams(s: str, p: str) -> list[int]:
    """
    Find all anagram starting indices.

    Time: O(n)
    Space: O(k) where k = alphabet size
    """
    if len(p) > len(s):
        return []

    p_count = {}
    s_count = {}

    for c in p:
        p_count[c] = p_count.get(c, 0) + 1

    result = []
    k = len(p)

    for i in range(len(s)):
        # Add right element
        s_count[s[i]] = s_count.get(s[i], 0) + 1

        # Remove left element if window too large
        if i >= k:
            left_char = s[i - k]
            s_count[left_char] -= 1
            if s_count[left_char] == 0:
                del s_count[left_char]

        # Check if anagram
        if s_count == p_count:
            result.append(i - k + 1)

    return result
```

### 5. Sliding Window Maximum (Monotonic Deque)
```python
from collections import deque

def maxSlidingWindow(nums: list[int], k: int) -> list[int]:
    """
    Maximum in each window using monotonic deque.

    Deque stores indices in decreasing order of values.

    Time: O(n)
    Space: O(k)
    """
    dq = deque()  # Stores indices
    result = []

    for i in range(len(nums)):
        # Remove elements outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # Remove smaller elements (they can't be max)
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        dq.append(i)

        # Add to result once window is full
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

### 6. Count Subarrays (At Most K Pattern)
```python
def subarraysWithKDistinct(nums: list[int], k: int) -> int:
    """
    Count subarrays with exactly K distinct elements.

    exactly(k) = atMost(k) - atMost(k-1)

    Time: O(n)
    Space: O(k)
    """
    def atMost(k):
        count = {}
        left = 0
        result = 0

        for right in range(len(nums)):
            count[nums[right]] = count.get(nums[right], 0) + 1

            while len(count) > k:
                count[nums[left]] -= 1
                if count[nums[left]] == 0:
                    del count[nums[left]]
                left += 1

            # Count all subarrays ending at right
            result += right - left + 1

        return result

    return atMost(k) - atMost(k - 1)
```

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SLIDING WINDOW PATTERNS                                  │
│                                                                             │
│  FIXED WINDOW (k=3):                                                        │
│  [1, 3, -1, -3, 5, 3, 6, 7]                                                │
│  [1, 3, -1]           → max = 3                                             │
│     [3, -1, -3]       → max = 3                                             │
│        [-1, -3, 5]    → max = 5                                             │
│           [-3, 5, 3]  → max = 5                                             │
│              [5, 3, 6] → max = 6                                            │
│                 [3, 6, 7] → max = 7                                         │
│                                                                             │
│  VARIABLE WINDOW (Longest without repeating):                               │
│  "abcabcbb"                                                                 │
│   L  R      → "abc" valid, length = 3                                       │
│      LR    → "bca" valid after shrinking                                    │
│       L R   → "cab" valid, length = 3                                       │
│        L R  → "abc" valid, length = 3                                       │
│                                                                             │
│  MINIMUM WINDOW SUBSTRING:                                                  │
│  s = "ADOBECODEBANC", t = "ABC"                                            │
│  Expand until valid: "ADOBEC" contains ABC                                  │
│  Shrink while valid: "DOBEC" still valid? No                                │
│  Continue: "BANC" is shortest containing ABC                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] Max Average Subarray I (LC #643)
- [ ] Longest Substring Without Repeating (LC #3)
- [ ] Minimum Size Subarray Sum (LC #209)
- [ ] Find All Anagrams (LC #438)

### Week 2: Intermediate
- [ ] Longest Repeating Character Replacement (LC #424)
- [ ] Max Consecutive Ones III (LC #1004)
- [ ] Permutation in String (LC #567)
- [ ] Subarray Product Less Than K (LC #713)

### Week 3: Advanced
- [ ] Minimum Window Substring (LC #76)
- [ ] Sliding Window Maximum (LC #239)
- [ ] Subarrays with K Different Integers (LC #992)
- [ ] Sliding Window Median (LC #480)

---

## Common Mistakes

1. **Wrong window boundaries**
   - `right - left + 1` for length (inclusive)
   - `right - left` if left is exclusive

2. **Not handling empty window**
   - Check if result was ever updated
   - Return appropriate default

3. **Forgetting to shrink**
   - Always shrink when condition violated
   - Don't just expand

4. **Off-by-one in fixed window**
   - Start adding to result at index `k-1`
   - Remove element at `i-k`, not `i-k+1`

5. **Count vs existence**
   - Use dict for counts, set for existence
   - Delete keys when count reaches 0

---

## Complexity Reference

| Pattern | Time | Space |
|---------|------|-------|
| Fixed window | O(n) | O(1) or O(k) |
| Variable (max) | O(n) | O(k) |
| Variable (min) | O(n) | O(k) |
| Anagram | O(n) | O(alphabet) |
| Sliding max (deque) | O(n) | O(k) |
| Count subarrays | O(n) | O(k) |

