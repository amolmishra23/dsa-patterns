# Arrays & Hashing - Hard Problems

## Problem 1: First Missing Positive (LC #41) - Hard

- [LeetCode](https://leetcode.com/problems/first-missing-positive/)

### Video Explanation
- [NeetCode - First Missing Positive](https://www.youtube.com/watch?v=8g78yfzMlao)

### Problem Statement
Given an unsorted integer array `nums`, return the smallest positive integer that is not present. Must run in O(n) time and O(1) auxiliary space.

### Examples
```
Input: nums = [1,2,0]
Output: 3

Input: nums = [3,4,-1,1]
Output: 2

Input: nums = [7,8,9,11,12]
Output: 1
```

### Visual Intuition
```
First Missing Positive - Cyclic Sort Visualization

Input: nums = [3, 4, -1, 1]
Goal: Find smallest positive not in array

Step 1: Cyclic Sort (place each number k at index k-1)
┌─────────────────────────────────────────────────────┐
│ Index:    0     1     2     3                       │
│ Value:   [3]   [4]   [-1]  [1]                      │
│                                                     │
│ i=0: nums[0]=3, should be at index 2                │
│      Swap nums[0] ↔ nums[2]                         │
│         [-1]  [4]   [3]   [1]                       │
│                                                     │
│ i=0: nums[0]=-1, invalid (negative), skip           │
│                                                     │
│ i=1: nums[1]=4, should be at index 3                │
│      Swap nums[1] ↔ nums[3]                         │
│         [-1]  [1]   [3]   [4]                       │
│                                                     │
│ i=1: nums[1]=1, should be at index 0                │
│      Swap nums[1] ↔ nums[0]                         │
│         [1]   [-1]  [3]   [4]                       │
│                                                     │
│ i=1: nums[1]=-1, invalid, skip                      │
│ i=2: nums[2]=3 at index 2 ✓ (correct position)      │
│ i=3: nums[3]=4 at index 3 ✓ (correct position)      │
└─────────────────────────────────────────────────────┘

Step 2: Find first mismatch
┌─────────────────────────────────────────────────────┐
│ Index:    0     1     2     3                       │
│ Value:   [1]   [-1]  [3]   [4]                      │
│ Expect:   1     2     3     4                       │
│           ✓     ✗     ✓     ✓                       │
│                 ↑                                   │
│           nums[1] ≠ 2 → Answer = 2                  │
└─────────────────────────────────────────────────────┘
```


### Intuition Development
```
Key insight: Answer must be in range [1, n+1]
- If array has n elements, smallest missing positive is at most n+1
- We can use the array itself as a hash map!

Idea: Place each number in its "correct" position
- Number 1 should be at index 0
- Number 2 should be at index 1
- Number k should be at index k-1

After rearranging, scan for first mismatch:
- If nums[i] != i+1, answer is i+1

nums = [3, 4, -1, 1]

Swap until each valid number is in place:
[3, 4, -1, 1] → swap 3 with nums[2]: [-1, 4, 3, 1]
[-1, 4, 3, 1] → -1 invalid, move on
[-1, 4, 3, 1] → swap 4 with nums[3]: [-1, 1, 3, 4]
[-1, 1, 3, 4] → swap 1 with nums[0]: [1, -1, 3, 4]
[1, -1, 3, 4] → 1 in place, -1 invalid, 3 in place, 4 in place

Scan: nums[1] = -1 ≠ 2 → Answer is 2
```

### Solution
```python
def firstMissingPositive(nums: list[int]) -> int:
    """
    Find the smallest missing positive integer.

    Strategy (Cyclic Sort / Index as Hash):
    1. Use the array itself as a hash map
    2. Place each number k at index k-1 (if valid)
    3. Scan for first position where nums[i] != i+1

    Why O(1) space: We modify the input array in-place
    Why O(n) time: Each number is swapped at most once to its correct position

    Time: O(n) - each element moved at most once
    Space: O(1) - in-place modification
    """
    n = len(nums)

    # ========== PHASE 1: Place each number in its correct position ==========
    # Number k should be at index k-1
    # Only consider numbers in range [1, n]

    for i in range(n):
        # Keep swapping until current number is:
        # - Out of valid range [1, n], OR
        # - Already in its correct position, OR
        # - Would swap with a duplicate

        while (1 <= nums[i] <= n and           # Number is in valid range
               nums[i] != nums[nums[i] - 1]):  # Not already in correct place

            # Swap nums[i] to its correct position
            correct_index = nums[i] - 1
            nums[i], nums[correct_index] = nums[correct_index], nums[i]

    # After this phase:
    # - nums[0] should be 1 (if 1 exists)
    # - nums[1] should be 2 (if 2 exists)
    # - nums[k-1] should be k (if k exists)

    # ========== PHASE 2: Find first missing positive ==========
    # Scan for first position where nums[i] != i+1

    for i in range(n):
        if nums[i] != i + 1:
            return i + 1  # Missing positive is i+1

    # If all positions [0, n-1] have correct values [1, n]
    # Then the missing positive is n+1
    return n + 1


def firstMissingPositive_marking(nums: list[int]) -> int:
    """
    Alternative approach using index marking (negative numbers).

    Strategy:
    1. Replace non-positive and out-of-range numbers with n+1
    2. For each valid number k, mark index k-1 as negative
    3. First positive index + 1 is the answer

    Time: O(n)
    Space: O(1)
    """
    n = len(nums)

    # Step 1: Replace invalid numbers with n+1
    # After this, all numbers are positive
    for i in range(n):
        if nums[i] <= 0 or nums[i] > n:
            nums[i] = n + 1

    # Step 2: Mark indices for numbers that exist
    # If number k exists, make nums[k-1] negative
    for i in range(n):
        num = abs(nums[i])  # Use abs because might be marked negative
        if num <= n:
            # Mark index num-1 as negative (number num exists)
            nums[num - 1] = -abs(nums[num - 1])

    # Step 3: Find first positive index
    # If nums[i] is positive, number i+1 is missing
    for i in range(n):
        if nums[i] > 0:
            return i + 1

    return n + 1
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- Empty array → return 1
- [1] → return 2
- [1,2,3] → return 4 (all present, so n+1)
- All negatives → return 1
- Large gaps in numbers → return first missing

---

## Problem 2: Minimum Window Substring (LC #76) - Hard

- [LeetCode](https://leetcode.com/problems/minimum-window-substring/)

### Video Explanation
- [NeetCode - Minimum Window Substring](https://www.youtube.com/watch?v=jSto0O4AJbM)

### Problem Statement
Given strings `s` and `t`, return the minimum window substring of `s` that contains all characters of `t`. If no such window exists, return "".

### Examples
```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"

Input: s = "a", t = "a"
Output: "a"

Input: s = "a", t = "aa"
Output: "" (need two 'a's but only have one)
```

### Visual Intuition
```
Minimum Window Substring - Sliding Window
s = "ADOBECODEBANC", t = "ABC"

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Expand to satisfy, shrink to minimize
═══════════════════════════════════════════════════════════════

Need: {A:1, B:1, C:1} → required=3 unique chars

Step-by-Step Execution:
───────────────────────
  Index:  0 1 2 3 4 5 6 7 8 9 10 11 12
  Chars:  A D O B E C O D E B  A  N  C

Phase 1: EXPAND until valid
──────────────────────────
  r=0: have={A:1}                     formed=1/3 ✗
  r=1: have={A:1,D:1}                 formed=1/3 ✗
  r=2: have={A:1,D:1,O:1}             formed=1/3 ✗
  r=3: have={A:1,D:1,O:1,B:1}         formed=2/3 ✗
  r=4: have={A:1,D:1,O:1,B:1,E:1}     formed=2/3 ✗
  r=5: have={A:1,D:1,O:1,B:1,E:1,C:1} formed=3/3 ✓ VALID!

        ┌─────────────┐
        │A D O B E C│ O D E B A N C
        └─────────────┘
        Window = "ADOBEC", len=6, min_len=6

Phase 2: SHRINK while still valid
─────────────────────────────────
  l=1: remove A → have={D:1,O:1,B:1,E:1,C:1} formed=2/3 ✗
       Stop shrinking! Continue expanding...

[Continue expanding and shrinking...]

Phase 3: Find optimal window
────────────────────────────
  At r=12, l=9:
                          ┌───────┐
        A D O B E C O D E │B A N C│
                          └───────┘
        have={B:1,A:1,N:1,C:1} formed=3/3 ✓
        Window = "BANC", len=4

  Shrink: remove B → formed=2/3 ✗

  min_len = 4, result = "BANC"

Answer: "BANC"

WHY THIS WORKS:
════════════════
● Expand: Grow window until all required chars present
● Shrink: Reduce window while STILL valid to find minimum
● "formed" tracks how many unique char requirements are met
● When formed == required, window is valid
```


### Intuition Development
```
Use sliding window with character counting:
1. Expand window (move right) until all chars of t are included
2. Contract window (move left) while maintaining validity
3. Track minimum valid window

t = "ABC" → need = {A:1, B:1, C:1}
s = "ADOBECODEBANC"

Window expansion and contraction:
[A]DOBECODEBANC     have A, need B,C
[ADOBEC]ODEBANC     have A,B,C - valid! length=6
 [DOBEC]ODEBANC     lost A - invalid
 [DOBECODEBA]NC     have A,B,C - valid! length=10 (worse)
  ...
         [BANC]     have A,B,C - valid! length=4 ✓
```

### Solution
```python
from collections import Counter

def minWindow(s: str, t: str) -> str:
    """
    Find minimum window substring containing all characters of t.

    Strategy (Sliding Window with Hash Map):
    1. Count characters needed from t
    2. Expand window until we have all needed characters
    3. Contract window while still valid, tracking minimum
    4. Continue until end of string

    Key variables:
    - need: Counter of characters still needed
    - have: Count of character requirements satisfied
    - required: Total unique characters to satisfy

    Time: O(|s| + |t|) - each character visited at most twice
    Space: O(|s| + |t|) - for counters and result
    """
    if not s or not t:
        return ""

    # Count characters needed from t
    t_count = Counter(t)
    required = len(t_count)  # Number of unique chars we need

    # Sliding window state
    window_count = {}  # Characters in current window
    have = 0           # How many unique chars we have enough of

    # Result tracking
    min_length = float('inf')
    result = ""

    left = 0

    for right in range(len(s)):
        # ========== EXPAND: Add character at right ==========
        char = s[right]
        window_count[char] = window_count.get(char, 0) + 1

        # Check if this character satisfies a requirement
        if char in t_count and window_count[char] == t_count[char]:
            have += 1

        # ========== CONTRACT: Shrink window while valid ==========
        while have == required:
            # Current window is valid - check if it's minimum
            window_length = right - left + 1
            if window_length < min_length:
                min_length = window_length
                result = s[left:right + 1]

            # Try to shrink by removing leftmost character
            left_char = s[left]
            window_count[left_char] -= 1

            # Check if we lost a required character
            if left_char in t_count and window_count[left_char] < t_count[left_char]:
                have -= 1

            left += 1

    return result
```

### Complexity
- **Time**: O(|s| + |t|)
- **Space**: O(|s| + |t|)

### Edge Cases
- t longer than s → return ""
- s equals t → return s
- No valid window → return ""
- Multiple valid windows → return minimum
- t has duplicates → need all occurrences

---

## Problem 3: Substring with Concatenation of All Words (LC #30) - Hard

- [LeetCode](https://leetcode.com/problems/substring-with-concatenation-of-all-words/)

### Video Explanation
- [NeetCode - Substring with Concatenation of All Words](https://www.youtube.com/watch?v=wT7KOLDGtvs)

### Problem Statement
Given a string `s` and an array of strings `words` of the same length, find all starting indices of substring(s) in `s` that is a concatenation of each word in `words` exactly once.

### Examples
```
Input: s = "barfoothefoobarman", words = ["foo","bar"]
Output: [0,9]
Explanation:
- s[0:6] = "barfoo" = "bar" + "foo" ✓
- s[9:15] = "foobar" = "foo" + "bar" ✓

Input: s = "wordgoodgoodgoodbestword", words = ["word","good","best","word"]
Output: []
```


### Visual Intuition
```
Substring with Concatenation of All Words
s = "barfoothefoobarman", words = ["foo","bar"]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: All words same length → slide by word_len, not 1
             Check multiple starting offsets (0 to word_len-1)
═══════════════════════════════════════════════════════════════

Setup:
  word_len = 3
  num_words = 2
  total_len = 6
  need = {foo:1, bar:1}

String divided into words:
  Index: 0   3   6   9   12  15
         bar|foo|the|foo|bar|man
         ─── ─── ─── ─── ─── ───

Offset 0: Start at index 0, slide by 3
──────────────────────────────────────
  Window [0,6]: "bar" + "foo"
    ┌───────┐
    │bar foo│ the foo bar man
    └───────┘
    have = {bar:1, foo:1} = need ✓ → result.append(0)

  Shrink, then add "the":
    Window [3,9]: "foo" + "the"
        ┌───────┐
    bar │foo the│ foo bar man
        └───────┘
    "the" not in need → reset window

  Window [9,15]: "foo" + "bar"
              ┌───────┐
    ... the │foo bar│ man
            └───────┘
    have = {foo:1, bar:1} = need ✓ → result.append(9)

Offset 1: Start at index 1, slide by 3
──────────────────────────────────────
  "arf", "oot", "hef"... none match words in need
  No valid windows found

Offset 2: Start at index 2, slide by 3
──────────────────────────────────────
  "rfo", "oth", "efo"... none match words in need
  No valid windows found

Final Result: [0, 9]

WHY THIS WORKS:
════════════════
● Same word length means we can treat string as array of words
● Multiple offsets cover all possible alignments
● Sliding window within each offset avoids O(n²) checking
● Reset window when invalid word encountered
```

### Solution
```python
from collections import Counter

def findSubstring(s: str, words: list[str]) -> list[int]:
    """
    Find all starting indices where concatenation of all words appears.

    Strategy:
    Since all words have the same length, we can:
    1. Slide a window of total_length = word_length * num_words
    2. For each starting position (0 to word_length-1), use sliding window
    3. Check if window contains exact word counts

    Optimization: Instead of checking each position independently,
    use sliding window within each "phase" (starting offset 0, 1, ..., word_len-1)

    Time: O(n * word_length) where n = len(s)
    Space: O(num_words * word_length) for counters
    """
    if not s or not words:
        return []

    word_len = len(words[0])      # Length of each word
    num_words = len(words)         # Number of words
    total_len = word_len * num_words  # Total length of concatenation

    # Count of each word we need
    word_count = Counter(words)
    result = []

    # Try each starting offset (0 to word_len-1)
    # This covers all possible alignments
    for offset in range(word_len):
        # Sliding window for this offset
        left = offset
        window_count = Counter()  # Words in current window
        words_matched = 0         # Number of words matched

        # Move right pointer by word_len each step
        for right in range(offset, len(s) - word_len + 1, word_len):
            # Extract the word at current position
            word = s[right:right + word_len]

            # ===== EXPAND: Add word to window =====
            if word in word_count:
                window_count[word] += 1
                words_matched += 1

                # If we have too many of this word, shrink from left
                while window_count[word] > word_count[word]:
                    left_word = s[left:left + word_len]
                    window_count[left_word] -= 1
                    words_matched -= 1
                    left += word_len

                # Check if we have exactly all words
                if words_matched == num_words:
                    result.append(left)
                    # Shrink window to find more matches
                    left_word = s[left:left + word_len]
                    window_count[left_word] -= 1
                    words_matched -= 1
                    left += word_len
            else:
                # Word not in our list - reset window
                window_count.clear()
                words_matched = 0
                left = right + word_len

    return result
```

### Complexity
- **Time**: O(n × word_length)
- **Space**: O(num_words × word_length)

### Edge Cases
- Empty string or words → return []
- Words longer than s → return []
- Duplicate words in list → need all occurrences
- Single word → find all occurrences
- Overlapping matches → return all start indices

---

## Problem 4: Trapping Rain Water (LC #42) - Hard

- [LeetCode](https://leetcode.com/problems/trapping-rain-water/)

### Video Explanation
- [NeetCode - Trapping Rain Water](https://www.youtube.com/watch?v=ZI2z5pq0TqA)

### Problem Statement
Given n non-negative integers representing an elevation map where width of each bar is 1, compute how much water it can trap after raining.

### Examples
```
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

Visual:
       █
   █   ██ █
 █ ██ ██████
 0 1 0 2 1 0 1 3 2 1 2 1

Water fills the gaps between bars.
```

### Visual Intuition
```
Trapping Rain Water
height = [0,1,0,2,1,0,1,3,2,1,2,1]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Water at i = min(max_left, max_right) - height[i]
             Two pointers: process side with smaller max
═══════════════════════════════════════════════════════════════

Visualization with water filled:
────────────────────────────────
  Index:  0  1  2  3  4  5  6  7  8  9 10 11
  Height: 0  1  0  2  1  0  1  3  2  1  2  1

  Level 3:                   █
  Level 2:       █  ≈  ≈  ≈  ██ █
  Level 1:    █  ██ ≈  ≈  ██████ █
  Level 0: ░  █░ ██ █░ ░█ ███████ █
           ─────────────────────────
           (█=bar, ≈=water, ░=ground)

Two Pointer Approach:
─────────────────────
  left=0, right=11
  left_max=0, right_max=0

  Step 1: height[0]=0 < height[11]=1
          left_max = max(0,0) = 0
          water += 0-0 = 0
          left → 1

  Step 2: height[1]=1 < height[11]=1
          left_max = max(0,1) = 1
          water += 0 (1 >= left_max, no water)
          left → 2

  Step 3: height[2]=0 < height[11]=1
          left_max = 1
          water += 1-0 = 1 ★
          left → 3

  [Continue...]

  Step at index 5: height[5]=0
          left_max=2, right_max=2
          water += min(2,2)-0 = 2 ★★

State at each position:
───────────────────────
  i:        0  1  2  3  4  5  6  7  8  9 10 11
  height:   0  1  0  2  1  0  1  3  2  1  2  1
  left_max: 0  1  1  2  2  2  2  3  3  3  3  3
  right_max:3  3  3  3  3  3  3  3  2  2  2  1
  min:      0  1  1  2  2  2  2  3  2  2  2  1
  water:    0  0  1  0  1  2  1  0  0  1  0  0
                 ↑     ↑  ↑  ↑        ↑
                 └─────┴──┴──┴────────┴─ = 6 total

Answer: 6 units of water

WHY THIS WORKS:
════════════════
● Water level at any point = min(tallest_left, tallest_right)
● Water trapped = water_level - bar_height
● Two pointers: process smaller side because that determines water level
● O(1) space: don't need to store all max values, just track as we go
```

### Intuition Development
```
Key insight: Water at position i = min(max_left, max_right) - height[i]

For each position:
- Find max height to its left
- Find max height to its right
- Water level = min of these two maxes
- Water trapped = water level - current height (if positive)

Approaches:
1. Brute force: For each i, scan left and right → O(n²)
2. Precompute: Store max_left[] and max_right[] → O(n) space
3. Two pointers: Track max from both ends → O(1) space ✓
```

### Solution
```python
def trap(height: list[int]) -> int:
    """
    Calculate water trapped between bars.

    Strategy (Two Pointers):
    - Use two pointers from both ends
    - Track max height seen from left and right
    - Process the side with smaller max (that's the limiting factor)
    - Water at each position = max_so_far - height (if positive)

    Why this works:
    - Water level at any position is limited by the SMALLER of max_left and max_right
    - By processing from the smaller side, we know the water level

    Time: O(n) - single pass with two pointers
    Space: O(1) - only tracking two max values
    """
    if not height:
        return 0

    left = 0
    right = len(height) - 1
    left_max = 0   # Max height seen from left side
    right_max = 0  # Max height seen from right side
    water = 0

    while left < right:
        # Process the side with smaller max
        # Because water level is determined by the smaller max

        if height[left] < height[right]:
            # Left side is the limiting factor
            if height[left] >= left_max:
                # Current bar is new max - no water here
                left_max = height[left]
            else:
                # Water can be trapped here
                # Amount = left_max - current height
                water += left_max - height[left]
            left += 1
        else:
            # Right side is the limiting factor
            if height[right] >= right_max:
                # Current bar is new max - no water here
                right_max = height[right]
            else:
                # Water can be trapped here
                water += right_max - height[right]
            right -= 1

    return water


def trap_precompute(height: list[int]) -> int:
    """
    Alternative approach using precomputed arrays.

    Strategy:
    1. Precompute max height to the left of each position
    2. Precompute max height to the right of each position
    3. Water at i = min(left_max[i], right_max[i]) - height[i]

    Time: O(n)
    Space: O(n)
    """
    if not height:
        return 0

    n = len(height)

    # left_max[i] = max height from index 0 to i
    left_max = [0] * n
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i-1], height[i])

    # right_max[i] = max height from index i to n-1
    right_max = [0] * n
    right_max[n-1] = height[n-1]
    for i in range(n-2, -1, -1):
        right_max[i] = max(right_max[i+1], height[i])

    # Calculate water at each position
    water = 0
    for i in range(n):
        # Water level at i is min of max heights on both sides
        water_level = min(left_max[i], right_max[i])
        # Water trapped = water level - bar height (if positive)
        water += max(0, water_level - height[i])

    return water
```

### Complexity
- **Two Pointers**: Time O(n), Space O(1)
- **Precompute**: Time O(n), Space O(n)

### Edge Cases
- Empty array → return 0
- All same height → return 0
- Single bar → return 0
- Two bars → return 0 (no middle to trap)
- Monotonically increasing/decreasing → return 0

---

## Problem 5: N-Queens (LC #51) - Hard

- [LeetCode](https://leetcode.com/problems/n-queens/)

### Video Explanation
- [NeetCode - N-Queens](https://www.youtube.com/watch?v=Ph95IHmRp5M)

### Problem Statement
Place n queens on an n×n chessboard such that no two queens attack each other. Return all distinct solutions.

### Examples
```
Input: n = 4
Output: [[".Q..","...Q","Q...","..Q."],
         ["..Q.","Q...","...Q",".Q.."]]

Visual for one solution:
. Q . .
. . . Q
Q . . .
. . Q .
```


### Visual Intuition
```
N-Queens (place N queens, none attacking)
n = 4

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Use sets to track attacked columns & diagonals
             row-col = main diagonal, row+col = anti-diagonal
═══════════════════════════════════════════════════════════════

Diagonal Pattern Explained:
───────────────────────────
  Main diagonal (↘): cells with same (row-col) value
      0  1  2  3
    ┌──┬──┬──┬──┐
  0 │ 0│-1│-2│-3│  row-col values
    ├──┼──┼──┼──┤
  1 │ 1│ 0│-1│-2│
    ├──┼──┼──┼──┤
  2 │ 2│ 1│ 0│-1│
    ├──┼──┼──┼──┤
  3 │ 3│ 2│ 1│ 0│
    └──┴──┴──┴──┘

  Anti-diagonal (↙): cells with same (row+col) value
      0  1  2  3
    ┌──┬──┬──┬──┐
  0 │ 0│ 1│ 2│ 3│  row+col values
    ├──┼──┼──┼──┤
  1 │ 1│ 2│ 3│ 4│
    ├──┼──┼──┼──┤
  2 │ 2│ 3│ 4│ 5│
    ├──┼──┼──┼──┤
  3 │ 3│ 4│ 5│ 6│
    └──┴──┴──┴──┘

Backtracking Trace:
───────────────────
Row 0: Try each column
  col=0: Place Q
    . . . .     cols={0}, diag={0}, anti={0}
    Q . . .
    . . . .
    . . . .

  Row 1: col=0 ✗(col), col=1 ✗(anti=1+0=1,diag=1-0=1)
         Try col=2:
    . . . .     cols={0,2}, diag={0,-1}, anti={0,3}
    Q . . .
    . . Q .
    . . . .

  Row 2: All invalid! Backtrack...

  [Eventually find solution:]
    . Q . .     cols={1,3,0,2}
    . . . Q     diag={-1,-2,2,1}
    Q . . .     anti={1,4,2,5}
    . . Q .

    ★ Solution 1 found! ★

Continue backtracking to find Solution 2:
    . . Q .
    Q . . .
    . . . Q
    . Q . .

Final: 2 solutions for n=4

WHY THIS WORKS:
════════════════
● One queen per row (iterate rows, place one queen each)
● Sets give O(1) conflict checking
● Backtrack = remove queen, try next column
● Diagonal math: row±col identifies diagonal lines
```

### Solution
```python
def solveNQueens(n: int) -> list[list[str]]:
    """
    Find all valid N-Queens placements using backtracking.

    Strategy:
    - Place queens row by row (one queen per row guaranteed)
    - For each row, try each column
    - Track which columns and diagonals are under attack
    - Backtrack when no valid position exists

    Key insight for diagonals:
    - Main diagonal (↘): row - col is constant
    - Anti-diagonal (↙): row + col is constant

    Time: O(n!) - at most n choices for first row, n-1 for second, etc.
    Space: O(n) - recursion depth + tracking sets
    """
    results = []

    # Track which columns and diagonals are under attack
    cols = set()           # Columns with queens
    pos_diag = set()       # Positive diagonals (row + col)
    neg_diag = set()       # Negative diagonals (row - col)

    # Current board state (list of queen positions per row)
    board = [["."] * n for _ in range(n)]

    def backtrack(row: int):
        """
        Try to place a queen in the given row.

        Args:
            row: Current row to place queen (0 to n-1)
        """
        # ===== BASE CASE: All queens placed =====
        if row == n:
            # Convert board to required format and add to results
            solution = ["".join(row) for row in board]
            results.append(solution)
            return

        # ===== RECURSIVE CASE: Try each column =====
        for col in range(n):
            # Check if this position is under attack
            if col in cols:
                continue  # Column already has a queen
            if (row + col) in pos_diag:
                continue  # Positive diagonal under attack
            if (row - col) in neg_diag:
                continue  # Negative diagonal under attack

            # ===== PLACE QUEEN =====
            cols.add(col)
            pos_diag.add(row + col)
            neg_diag.add(row - col)
            board[row][col] = "Q"

            # ===== RECURSE to next row =====
            backtrack(row + 1)

            # ===== BACKTRACK: Remove queen =====
            cols.remove(col)
            pos_diag.remove(row + col)
            neg_diag.remove(row - col)
            board[row][col] = "."

    backtrack(0)
    return results


def totalNQueens(n: int) -> int:
    """
    Count the number of valid N-Queens solutions.

    Same approach as above, but only count solutions.

    Time: O(n!)
    Space: O(n)
    """
    count = 0
    cols = set()
    pos_diag = set()
    neg_diag = set()

    def backtrack(row: int):
        nonlocal count

        if row == n:
            count += 1
            return

        for col in range(n):
            if col in cols or (row + col) in pos_diag or (row - col) in neg_diag:
                continue

            cols.add(col)
            pos_diag.add(row + col)
            neg_diag.add(row - col)

            backtrack(row + 1)

            cols.remove(col)
            pos_diag.remove(row + col)
            neg_diag.remove(row - col)

    backtrack(0)
    return count
```

### Complexity
- **Time**: O(n!)
- **Space**: O(n)

### Edge Cases
- n = 1 → return [["Q"]]
- n = 2 or n = 3 → return [] (no solution)
- n = 4 → return 2 solutions
- Large n → exponential but pruning helps
- Symmetry → solutions are mirror images

---

## Summary: Hard Problems Checklist

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | First Missing Positive | Cyclic sort / Index marking | O(n) | O(1) |
| 2 | Minimum Window Substring | Sliding window + counting | O(s+t) | O(s+t) |
| 3 | Substring Concatenation | Sliding window by word | O(n×w) | O(words) |
| 4 | Trapping Rain Water | Two pointers | O(n) | O(1) |
| 5 | N-Queens | Backtracking + sets | O(n!) | O(n) |

---

## Practice More Hard Problems

- [ ] LC #239 - Sliding Window Maximum
- [ ] LC #295 - Find Median from Data Stream
- [ ] LC #37 - Sudoku Solver
- [ ] LC #84 - Largest Rectangle in Histogram
- [ ] LC #85 - Maximal Rectangle

