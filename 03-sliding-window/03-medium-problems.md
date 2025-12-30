# Sliding Window - Medium Problems

## Problem 1: Longest Substring Without Repeating Characters (LC #3) - Medium

- [LeetCode](https://leetcode.com/problems/longest-substring-without-repeating-characters/)

### Problem Statement
Find the length of the longest substring without repeating characters.

### Video Explanation
- [NeetCode - Longest Substring Without Repeating](https://www.youtube.com/watch?v=wiGpQwVHdE0)

### Examples
```
Input: s = "abcabcbb"
Output: 3
Explanation: "abc" is the longest

Input: s = "bbbbb"
Output: 1

Input: s = "pwwkew"
Output: 3
Explanation: "wke" is the longest
```

### Intuition Development
```
Variable window: expand until duplicate found, then contract.

s = "abcabcbb"

[a]bcabcbb      seen={a}, length=1
[ab]cabcbb      seen={a,b}, length=2
[abc]abcbb      seen={a,b,c}, length=3
[abca]bcbb      'a' repeats! Contract until no duplicate
 [bca]bcbb      seen={b,c,a}, length=3
 [bcab]cbb      'b' repeats! Contract
   [cab]cbb     seen={c,a,b}, length=3
   ...

Maximum length = 3
```

### Solution
```python
def lengthOfLongestSubstring(s: str) -> int:
    """
    Find length of longest substring without repeating characters.

    Strategy (Variable Sliding Window):
    - Expand window by moving right pointer
    - Track characters in window using a set or hash map
    - When duplicate found, contract from left until no duplicate
    - Track maximum window length

    Time: O(n) - each character visited at most twice
    Space: O(min(n, m)) where m is character set size (26 for lowercase)
    """
    # Set to track characters in current window
    char_set = set()

    left = 0
    max_length = 0

    for right in range(len(s)):
        # ===== CONTRACT: Remove duplicates from left =====
        # If current character already in window, shrink until it's not
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        # ===== EXPAND: Add current character =====
        char_set.add(s[right])

        # ===== UPDATE: Track maximum length =====
        current_length = right - left + 1
        max_length = max(max_length, current_length)

    return max_length


def lengthOfLongestSubstring_optimized(s: str) -> int:
    """
    Optimized version using hash map to store last index.

    Instead of removing one character at a time, jump directly
    to the position after the duplicate.

    Time: O(n) - single pass
    Space: O(min(n, m))
    """
    # Map: character -> most recent index
    char_index = {}

    left = 0
    max_length = 0

    for right, char in enumerate(s):
        # If character seen and within current window
        if char in char_index and char_index[char] >= left:
            # Jump left pointer past the duplicate
            left = char_index[char] + 1

        # Update character's last seen index
        char_index[char] = right

        # Update maximum length
        max_length = max(max_length, right - left + 1)

    return max_length
```

### Complexity
- **Time**: O(n)
- **Space**: O(min(n, m)) where m is alphabet size

### Edge Cases
- Empty string → 0
- All unique characters → length of string
- All same character → 1
- Single character → 1
- String with spaces or special characters

---

## Problem 2: Longest Repeating Character Replacement (LC #424) - Medium

- [LeetCode](https://leetcode.com/problems/longest-repeating-character-replacement/)

### Problem Statement
Given a string and integer k, find the longest substring with all same characters after replacing at most k characters.

### Video Explanation
- [NeetCode - Longest Repeating Character Replacement](https://www.youtube.com/watch?v=gqXU1UyA8pk)

### Examples
```
Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace both 'A's with 'B's (or vice versa)

Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace middle 'B' → "AAAAABA" contains "AAAA"
```

### Intuition Development
```
Key insight: Valid window if (window_length - max_frequency) <= k
- max_frequency = count of most common character in window
- (window_length - max_frequency) = characters we need to replace
- If this <= k, we can make all characters the same

s = "AABABBA", k = 1

[A]ABABBA     count={A:1}, max_freq=1, need_replace=0 ≤ 1 ✓
[AA]BABBA     count={A:2}, max_freq=2, need_replace=0 ≤ 1 ✓
[AAB]ABBA     count={A:2,B:1}, max_freq=2, need_replace=1 ≤ 1 ✓
[AABA]BBA     count={A:3,B:1}, max_freq=3, need_replace=1 ≤ 1 ✓
[AABAB]BA     count={A:3,B:2}, max_freq=3, need_replace=2 > 1 ✗ Contract!
 [ABAB]BA     count={A:2,B:2}, max_freq=2, need_replace=2 > 1 ✗ Contract!
  [BAB]BA     count={A:1,B:2}, max_freq=2, need_replace=1 ≤ 1 ✓
  [BABB]A     count={A:1,B:3}, max_freq=3, need_replace=1 ≤ 1 ✓
  [BABBA]     count={A:2,B:3}, max_freq=3, need_replace=2 > 1 ✗
   [ABBA]     count={A:2,B:2}, max_freq=2, need_replace=2 > 1 ✗
    [BBA]     count={A:1,B:2}, max_freq=2, need_replace=1 ≤ 1 ✓

Maximum valid window length = 4
```

### Solution
```python
def characterReplacement(s: str, k: int) -> int:
    """
    Find longest substring with same characters after at most k replacements.

    Strategy (Variable Sliding Window):
    - Track frequency of each character in window
    - Track max frequency (most common character)
    - Valid window: (window_size - max_freq) <= k
    - When invalid, contract from left

    Key insight: We want to keep the most frequent character and replace others.
    Characters to replace = window_size - max_frequency

    Time: O(n) - each character visited at most twice
    Space: O(26) = O(1) - frequency array for uppercase letters
    """
    # Frequency count of characters in current window
    count = {}

    left = 0
    max_freq = 0   # Frequency of most common character in window
    max_length = 0

    for right in range(len(s)):
        # ===== EXPAND: Add character at right =====
        char = s[right]
        count[char] = count.get(char, 0) + 1

        # Update max frequency in window
        max_freq = max(max_freq, count[char])

        # ===== CHECK VALIDITY =====
        # Window is valid if we can make all chars same with <= k replacements
        # Characters to replace = window_size - max_frequency
        window_size = right - left + 1

        # ===== CONTRACT if invalid =====
        if window_size - max_freq > k:
            # Remove character at left
            count[s[left]] -= 1
            left += 1
            # Note: We don't update max_freq when shrinking
            # This is okay because we only care about finding LONGER windows

        # ===== UPDATE maximum length =====
        max_length = max(max_length, right - left + 1)

    return max_length


def characterReplacement_strict(s: str, k: int) -> int:
    """
    Stricter version that properly updates max_freq when contracting.

    Same time complexity but more intuitive.
    """
    count = {}
    left = 0
    max_length = 0

    for right in range(len(s)):
        # Add character at right
        char = s[right]
        count[char] = count.get(char, 0) + 1

        # Contract while window is invalid
        while True:
            window_size = right - left + 1
            max_freq = max(count.values()) if count else 0

            if window_size - max_freq <= k:
                break  # Window is valid

            # Remove character at left
            count[s[left]] -= 1
            if count[s[left]] == 0:
                del count[s[left]]
            left += 1

        # Update maximum length
        max_length = max(max_length, right - left + 1)

    return max_length
```

### Complexity
- **Time**: O(n) - Single pass through string
- **Space**: O(1) - At most 26 characters in count map

### Edge Cases
- All same characters: k replacements give length = n
- k = 0: Find longest run of same character
- k >= n: Entire string can be made same character
- Single character string: Return 1

---

## Problem 3: Permutation in String (LC #567) - Medium

- [LeetCode](https://leetcode.com/problems/permutation-in-string/)

### Problem Statement
Check if s2 contains a permutation of s1.

### Video Explanation
- [NeetCode - Permutation in String](https://www.youtube.com/watch?v=UbyhOgBN834)

### Examples
```
Input: s1 = "ab", s2 = "eidbaooo"
Output: true
Explanation: s2 contains "ba" which is permutation of "ab"

Input: s1 = "ab", s2 = "eidboaoo"
Output: false
```

### Intuition Development
```
Fixed window of size len(s1). Check if window has same character counts as s1.

s1 = "ab", s2 = "eidbaooo"

s1_count = {a:1, b:1}

Window [ei]: {e:1, i:1} ≠ s1_count
Window [id]: {i:1, d:1} ≠ s1_count
Window [db]: {d:1, b:1} ≠ s1_count
Window [ba]: {b:1, a:1} = s1_count ✓ Found!
```

### Solution
```python
from collections import Counter

def checkInclusion(s1: str, s2: str) -> bool:
    """
    Check if s2 contains a permutation of s1.

    Strategy (Fixed Size Sliding Window):
    - Window size = len(s1)
    - Maintain character count of current window
    - Compare with s1's character count
    - Slide window and update counts incrementally

    Time: O(n) where n = len(s2)
    Space: O(1) - at most 26 characters
    """
    if len(s1) > len(s2):
        return False

    # Character count needed (from s1)
    s1_count = Counter(s1)

    # Character count in current window
    window_count = Counter(s2[:len(s1)])

    # Check if first window matches
    if window_count == s1_count:
        return True

    # Slide the window
    for i in range(len(s1), len(s2)):
        # Add new character (entering window)
        new_char = s2[i]
        window_count[new_char] += 1

        # Remove old character (leaving window)
        old_char = s2[i - len(s1)]
        window_count[old_char] -= 1
        if window_count[old_char] == 0:
            del window_count[old_char]  # Remove zero counts for comparison

        # Check if current window matches
        if window_count == s1_count:
            return True

    return False


def checkInclusion_optimized(s1: str, s2: str) -> bool:
    """
    Optimized version tracking number of matching characters.

    Instead of comparing entire counts, track how many characters
    have the correct frequency.

    Time: O(n)
    Space: O(1)
    """
    if len(s1) > len(s2):
        return False

    # Count arrays for s1 and window
    s1_count = [0] * 26
    window_count = [0] * 26

    for char in s1:
        s1_count[ord(char) - ord('a')] += 1

    # Track number of characters with matching frequency
    matches = 0
    for i in range(26):
        if s1_count[i] == 0:
            matches += 1  # Both have 0, they match

    # Initialize first window
    for i in range(len(s1)):
        idx = ord(s2[i]) - ord('a')
        window_count[idx] += 1

        # Update matches
        if window_count[idx] == s1_count[idx]:
            matches += 1
        elif window_count[idx] == s1_count[idx] + 1:
            matches -= 1

    if matches == 26:
        return True

    # Slide window
    for i in range(len(s1), len(s2)):
        # Add new character
        add_idx = ord(s2[i]) - ord('a')
        window_count[add_idx] += 1
        if window_count[add_idx] == s1_count[add_idx]:
            matches += 1
        elif window_count[add_idx] == s1_count[add_idx] + 1:
            matches -= 1

        # Remove old character
        rem_idx = ord(s2[i - len(s1)]) - ord('a')
        window_count[rem_idx] -= 1
        if window_count[rem_idx] == s1_count[rem_idx]:
            matches += 1
        elif window_count[rem_idx] == s1_count[rem_idx] - 1:
            matches -= 1

        if matches == 26:
            return True

    return False
```

### Complexity
- **Time**: O(n) - Sliding window over s2
- **Space**: O(1) - Fixed size count arrays (26 characters)

### Edge Cases
- s1 longer than s2: Return false immediately
- s1 equals s2: Return true
- s1 is single character: Check if s2 contains that character
- s2 is empty: Return false (unless s1 is also empty)

---

## Problem 4: Fruit Into Baskets (LC #904) - Medium

- [LeetCode](https://leetcode.com/problems/fruit-into-baskets/)

### Problem Statement
You are visiting a farm that has a single row of fruit trees. Each tree produces one type of fruit. You have two baskets, and each basket can only hold one type of fruit. Starting from any tree, pick fruits from each tree moving right, stopping when you pick a fruit that can't fit. Return the maximum number of fruits you can collect.

### Video Explanation
- [NeetCode - Fruit Into Baskets](https://www.youtube.com/watch?v=yYtaV0G3mWQ)

### Examples
```
Input: fruits = [1,2,1]
Output: 3
Explanation: All fruits can be picked (types 1 and 2).

Input: fruits = [0,1,2,2]
Output: 3
Explanation: Pick from [1,2,2] (types 1 and 2).

Input: fruits = [1,2,3,2,2]
Output: 4
Explanation: Pick from [2,3,2,2] (types 2 and 3).
```

### Intuition Development
```
Reframe: Find longest subarray with AT MOST 2 distinct elements!

fruits = [1, 2, 3, 2, 2]

┌─────────────────────────────────────────────────────────────────┐
│ Variable Sliding Window with constraint: ≤ 2 fruit types       │
│                                                                  │
│ r=0: [1], basket={1:1}, 1 type ✓, max=1                        │
│ r=1: [1,2], basket={1:1,2:1}, 2 types ✓, max=2                 │
│ r=2: [1,2,3], basket={1:1,2:1,3:1}, 3 types ✗                  │
│      Contract: remove 1, basket={2:1,3:1}, l=1                 │
│      [2,3], 2 types ✓, max=2                                   │
│ r=3: [2,3,2], basket={2:2,3:1}, 2 types ✓, max=3               │
│ r=4: [2,3,2,2], basket={2:3,3:1}, 2 types ✓, max=4             │
│                                                                  │
│ Answer: 4 ✓                                                      │
│                                                                  │
│ Key: Use hash map to track fruit counts in current window       │
│ Contract when len(basket) > 2                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
from collections import defaultdict

def totalFruit(fruits: list[int]) -> int:
    """
    Find longest subarray with at most 2 distinct elements.

    Strategy (Variable Sliding Window):
    - Expand window by adding fruits
    - Track count of each fruit type
    - When more than 2 types, contract from left
    - Track maximum window size

    Time: O(n) - each element visited at most twice
    Space: O(1) - at most 3 fruit types in map at any time
    """
    # Count of each fruit type in current window
    basket = defaultdict(int)

    left = 0
    max_fruits = 0

    for right in range(len(fruits)):
        # ===== EXPAND: Add fruit at right =====
        fruit = fruits[right]
        basket[fruit] += 1

        # ===== CONTRACT: While more than 2 types =====
        while len(basket) > 2:
            # Remove fruit at left
            left_fruit = fruits[left]
            basket[left_fruit] -= 1

            # If count becomes 0, remove from basket
            if basket[left_fruit] == 0:
                del basket[left_fruit]

            left += 1

        # ===== UPDATE: Track maximum =====
        max_fruits = max(max_fruits, right - left + 1)

    return max_fruits
```

### Complexity
- **Time**: O(n) - Each element visited at most twice
- **Space**: O(1) - At most 3 fruit types in map

### Edge Cases
- All same fruit: Return entire array length
- Only 2 fruit types: Return entire array length
- Alternating types: Works with sliding window
- Single element: Return 1

---

## Problem 5: Minimum Size Subarray Sum (LC #209) - Medium

- [LeetCode](https://leetcode.com/problems/minimum-size-subarray-sum/)

### Problem Statement
Given an array of positive integers `nums` and a positive integer `target`, return the **minimal length** of a subarray whose sum is greater than or equal to `target`. If there is no such subarray, return 0.

### Video Explanation
- [NeetCode - Minimum Size Subarray Sum](https://www.youtube.com/watch?v=aYqYMIqZx5s)

### Examples
```
Input: target = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: [4,3] has sum 7 >= target, and it's the shortest.

Input: target = 4, nums = [1,4,4]
Output: 1
Explanation: [4] has sum 4 >= target.

Input: target = 11, nums = [1,1,1,1,1,1,1,1]
Output: 0
Explanation: Sum of all elements is 8 < 11, impossible.
```

### Intuition Development
```
Variable Sliding Window - but CONTRACT while valid (to minimize)!

target = 7, nums = [2, 3, 1, 2, 4, 3]

┌─────────────────────────────────────────────────────────────────┐
│ Expand until sum >= target, then contract to minimize length    │
│                                                                  │
│ r=0: sum=2 < 7, expand                                          │
│ r=1: sum=5 < 7, expand                                          │
│ r=2: sum=6 < 7, expand                                          │
│ r=3: sum=8 >= 7 ✓, len=4, try contract                          │
│      remove 2: sum=6 < 7, stop contracting                      │
│      min_len=4                                                   │
│ r=4: sum=10 >= 7 ✓, len=4                                       │
│      remove 3: sum=7 >= 7 ✓, len=3, min_len=3                   │
│      remove 1: sum=6 < 7, stop                                   │
│ r=5: sum=9 >= 7 ✓, len=3                                        │
│      remove 2: sum=7 >= 7 ✓, len=2, min_len=2 ★                 │
│      remove 4: sum=3 < 7, stop                                   │
│                                                                  │
│ Answer: 2 (subarray [4,3])                                       │
│                                                                  │
│ Key difference: Contract WHILE valid (not until invalid)        │
│ This finds minimum length satisfying condition                   │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def minSubArrayLen(target: int, nums: list[int]) -> int:
    """
    Find minimum length subarray with sum >= target.

    Strategy (Variable Sliding Window):
    - Expand window until sum >= target
    - Contract while sum still >= target, tracking minimum length
    - Continue expanding

    Time: O(n) - each element added/removed at most once
    Space: O(1)
    """
    left = 0
    window_sum = 0
    min_length = float('inf')  # Initialize to infinity

    for right in range(len(nums)):
        # ===== EXPAND: Add element at right =====
        window_sum += nums[right]

        # ===== CONTRACT: While sum >= target =====
        # Try to minimize window while maintaining valid sum
        while window_sum >= target:
            # Update minimum length
            current_length = right - left + 1
            min_length = min(min_length, current_length)

            # Remove element at left and shrink window
            window_sum -= nums[left]
            left += 1

    # Return 0 if no valid subarray found
    return min_length if min_length != float('inf') else 0
```

### Complexity
- **Time**: O(n) - Each element added/removed at most once
- **Space**: O(1) - Only use variables

### Edge Cases
- No valid subarray exists: Return 0
- Single element >= target: Return 1
- Entire array needed: Return n
- All elements equal target: Return 1

---

## Problem 6: Maximum Number of Vowels in a Substring of Given Length (LC #1456) - Medium

- [LeetCode](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/)

### Problem Statement
Given a string `s` and an integer `k`, return the maximum number of vowel letters in any substring of `s` with length `k`. Vowel letters are 'a', 'e', 'i', 'o', and 'u'.

### Video Explanation
- [NeetCode - Maximum Vowels in Substring](https://www.youtube.com/watch?v=kEfPSzgL-xo)

### Examples
```
Input: s = "abciiidef", k = 3
Output: 3
Explanation: "iii" has 3 vowels.

Input: s = "aeiou", k = 2
Output: 2
Explanation: Any substring of length 2 has 2 vowels.

Input: s = "leetcode", k = 3
Output: 2
Explanation: "lee", "eet", "ode" have 2 vowels.
```

### Intuition Development
```
Fixed Size Sliding Window - classic pattern!

s = "abciiidef", k = 3

┌─────────────────────────────────────────────────────────────────┐
│ Fixed window of size k, count vowels                            │
│                                                                  │
│ Initial window [a,b,c]: vowels = 1 (just 'a')                  │
│ max_vowels = 1                                                   │
│                                                                  │
│ Slide to [b,c,i]: remove 'a' (vowel, -1), add 'i' (vowel, +1)  │
│ vowels = 1, max = 1                                             │
│                                                                  │
│ Slide to [c,i,i]: remove 'b' (not vowel), add 'i' (vowel, +1)  │
│ vowels = 2, max = 2                                             │
│                                                                  │
│ Slide to [i,i,i]: remove 'c' (not vowel), add 'i' (vowel, +1)  │
│ vowels = 3, max = 3 ★                                           │
│                                                                  │
│ Continue sliding... max stays 3                                 │
│                                                                  │
│ Answer: 3                                                        │
│                                                                  │
│ Key: O(1) update per slide - just check entering/leaving chars  │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def maxVowels(s: str, k: int) -> int:
    """
    Find maximum vowels in any substring of length k.

    Strategy (Fixed Size Sliding Window):
    - Count vowels in first window
    - Slide: add new char (check if vowel), remove old char (check if vowel)
    - Track maximum count

    Time: O(n)
    Space: O(1)
    """
    vowels = set('aeiou')

    # Count vowels in first window
    vowel_count = sum(1 for char in s[:k] if char in vowels)
    max_vowels = vowel_count

    # Slide the window
    for i in range(k, len(s)):
        # Add new character
        if s[i] in vowels:
            vowel_count += 1

        # Remove old character
        if s[i - k] in vowels:
            vowel_count -= 1

        # Update maximum
        max_vowels = max(max_vowels, vowel_count)

    return max_vowels
```

### Complexity
- **Time**: O(n) - Single pass through string
- **Space**: O(1) - Only use variables and vowel set

### Edge Cases
- All vowels: Return k
- No vowels: Return 0
- k equals string length: Return vowel count of entire string
- k = 1: Return 1 if any vowel exists, else 0

---

## Summary: Medium Problems Checklist

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | Longest Substring No Repeat | Variable window + set | O(n) | O(m) |
| 2 | Longest Repeating Char Replace | Variable window + max freq | O(n) | O(1) |
| 3 | Permutation in String | Fixed window + count match | O(n) | O(1) |
| 4 | Fruit Into Baskets | Variable window, at most 2 | O(n) | O(1) |
| 5 | Min Size Subarray Sum | Variable window, sum >= k | O(n) | O(1) |
| 6 | Max Vowels in Substring | Fixed window count | O(n) | O(1) |

---

## Practice More Medium Problems

- [ ] LC #438 - Find All Anagrams in a String
- [ ] LC #1004 - Max Consecutive Ones III
- [ ] LC #1208 - Get Equal Substrings Within Budget
- [ ] LC #1493 - Longest Subarray of 1's After Deleting One Element
- [ ] LC #2024 - Maximize the Confusion of an Exam

