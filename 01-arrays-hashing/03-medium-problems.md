# Arrays & Hashing - Medium Problems

## Problem 1: Group Anagrams (LC #49) - Medium

- [LeetCode](https://leetcode.com/problems/group-anagrams/)

### Problem Statement
Given an array of strings `strs`, group the anagrams together. An anagram is a word formed by rearranging letters of another word.

### Video Explanation
- [NeetCode - Group Anagrams](https://www.youtube.com/watch?v=vzdNOK2oB2E)
- [Take U Forward - Group Anagrams](https://www.youtube.com/watch?v=0I6IL3TnIZs)

### Examples
```
Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Input: strs = [""]
Output: [[""]]

Input: strs = ["a"]
Output: [["a"]]
```

### Intuition Development
```
Key insight: Anagrams have the SAME sorted characters
"eat" → sorted → "aet"
"tea" → sorted → "aet"
"ate" → sorted → "aet"

So we can use sorted string as a KEY to group anagrams!

Alternative: Use character frequency tuple as key (faster for long strings)
"eat" → (1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0)
         a                 e                             t
```

### Solution
```python
from collections import defaultdict

def groupAnagrams(strs: list[str]) -> list[list[str]]:
    """
    Group strings that are anagrams of each other.

    Strategy:
    - Use sorted string as dictionary key
    - All anagrams will have the same sorted form
    - Group them in a defaultdict(list)

    Time: O(n * k log k) where n = number of strings, k = max string length
    Space: O(n * k) to store all strings in groups
    """
    # Dictionary to group anagrams: sorted_string -> list of original strings
    anagram_groups = defaultdict(list)

    for word in strs:
        # Create a key by sorting the characters
        # "eat" -> "aet", "tea" -> "aet" (same key!)
        sorted_key = ''.join(sorted(word))

        # Add original word to its anagram group
        anagram_groups[sorted_key].append(word)

    # Return all groups as a list of lists
    return list(anagram_groups.values())


def groupAnagrams_optimized(strs: list[str]) -> list[list[str]]:
    """
    Optimized version using character count tuple as key.

    Advantage: O(k) to create key instead of O(k log k) for sorting

    Time: O(n * k) where n = number of strings, k = max string length
    Space: O(n * k)
    """
    anagram_groups = defaultdict(list)

    for word in strs:
        # Create frequency count for each letter (a-z)
        # This creates a tuple like (1, 0, 0, ..., 1, ..., 1)
        # representing count of each letter
        count = [0] * 26  # 26 lowercase letters

        for char in word:
            # ord('a') = 97, so ord(char) - ord('a') gives index 0-25
            count[ord(char) - ord('a')] += 1

        # Convert to tuple (lists can't be dict keys, tuples can)
        key = tuple(count)
        anagram_groups[key].append(word)

    return list(anagram_groups.values())
```

### Complexity
- **Time**: O(n × k log k) for sorting approach, O(n × k) for count approach
- **Space**: O(n × k) to store all strings

### Edge Cases
- Empty string in input → forms its own group ""
- Single character strings
- All strings are anagrams of each other → single group
- All strings are unique (no anagrams) → each in its own group

---

## Problem 2: Top K Frequent Elements (LC #347) - Medium

- [LeetCode](https://leetcode.com/problems/top-k-frequent-elements/)

### Problem Statement
Given an integer array `nums` and an integer `k`, return the `k` most frequent elements.

### Video Explanation
- [NeetCode - Top K Frequent Elements](https://www.youtube.com/watch?v=YPTqKIgVk-k)

### Examples
```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Input: nums = [1], k = 1
Output: [1]
```

### Intuition Development
```
Step 1: Count frequency of each element
Step 2: Get top k by frequency

Approaches:
1. Sort by frequency → O(n log n)
2. Heap of size k → O(n log k)
3. Bucket sort → O(n) ← Best!

Bucket Sort idea:
- Create buckets where index = frequency
- bucket[3] = elements that appear 3 times
- Walk buckets from high to low, collect k elements
```

### Solution
```python
from collections import Counter
import heapq

def topKFrequent_heap(nums: list[int], k: int) -> list[int]:
    """
    Find k most frequent elements using a min-heap.

    Strategy:
    1. Count frequency of each number
    2. Maintain a min-heap of size k
    3. Heap stores (frequency, number) pairs
    4. After processing, heap contains k most frequent

    Time: O(n log k) - n insertions, each O(log k)
    Space: O(n) for frequency map + O(k) for heap
    """
    # Step 1: Count frequency of each number
    frequency = Counter(nums)  # {1: 3, 2: 2, 3: 1}

    # Step 2: Use min-heap to keep track of top k elements
    # We use min-heap so we can efficiently remove smallest
    # when heap size exceeds k
    min_heap = []

    for num, freq in frequency.items():
        # Push (frequency, number) - heap sorts by first element
        heapq.heappush(min_heap, (freq, num))

        # If heap exceeds size k, remove the least frequent
        if len(min_heap) > k:
            heapq.heappop(min_heap)  # Removes smallest frequency

    # Step 3: Extract numbers from heap (frequency is index 0, num is index 1)
    return [num for freq, num in min_heap]


def topKFrequent_bucket(nums: list[int], k: int) -> list[int]:
    """
    Find k most frequent elements using bucket sort.

    Strategy:
    1. Count frequency of each number
    2. Create buckets where index = frequency count
    3. Walk buckets from highest frequency down
    4. Collect k elements

    Time: O(n) - counting is O(n), bucket creation is O(n)
    Space: O(n) for frequency map and buckets

    This is the OPTIMAL solution!
    """
    # Step 1: Count frequency of each number
    frequency = Counter(nums)  # {1: 3, 2: 2, 3: 1}

    # Step 2: Create buckets - index represents frequency
    # bucket[i] = list of numbers that appear exactly i times
    # Max possible frequency is len(nums)
    buckets = [[] for _ in range(len(nums) + 1)]

    for num, freq in frequency.items():
        buckets[freq].append(num)

    # Example after this step:
    # buckets[1] = [3]     (3 appears 1 time)
    # buckets[2] = [2]     (2 appears 2 times)
    # buckets[3] = [1]     (1 appears 3 times)

    # Step 3: Collect k most frequent by walking buckets from high to low
    result = []
    for freq in range(len(buckets) - 1, 0, -1):  # Start from highest frequency
        for num in buckets[freq]:
            result.append(num)
            if len(result) == k:
                return result

    return result
```

### Complexity
- **Heap**: Time O(n log k), Space O(n)
- **Bucket**: Time O(n), Space O(n) ← Optimal

### Edge Cases
- k equals number of unique elements → return all
- All same elements → return that element
- Single element → return it
- k = 1 → return most frequent element

---

## Problem 3: Product of Array Except Self (LC #238) - Medium

- [LeetCode](https://leetcode.com/problems/product-of-array-except-self/)

### Problem Statement
Given an integer array `nums`, return an array `answer` where `answer[i]` equals the product of all elements except `nums[i]`. **Cannot use division.**

### Video Explanation
- [NeetCode - Product of Array Except Self](https://www.youtube.com/watch?v=bNvIQI2wAjk)

### Examples
```
Input: nums = [1,2,3,4]
Output: [24,12,8,6]
Explanation:
- answer[0] = 2*3*4 = 24
- answer[1] = 1*3*4 = 12
- answer[2] = 1*2*4 = 8
- answer[3] = 1*2*3 = 6

Input: nums = [-1,1,0,-3,3]
Output: [0,0,9,0,0]
```

### Intuition Development
```
Key insight: answer[i] = (product of all left of i) × (product of all right of i)

nums =    [1,  2,  3,  4]
           ↓   ↓   ↓   ↓
left  =   [1,  1,  2,  6]   ← product of elements to the LEFT
right =   [24, 12, 4,  1]   ← product of elements to the RIGHT
           ↓   ↓   ↓   ↓
answer =  [24, 12, 8,  6]   ← left[i] × right[i]

We can do this in O(1) extra space by:
1. First pass: build left products into result
2. Second pass: multiply by right products on the fly
```

### Solution
```python
def productExceptSelf(nums: list[int]) -> list[int]:
    """
    Calculate product of all elements except self without division.

    Strategy (Two-pass approach):
    - answer[i] = (product of all left elements) × (product of all right elements)
    - First pass: Calculate left products
    - Second pass: Multiply by right products

    Time: O(n) - two passes through array
    Space: O(1) - output array doesn't count as extra space

    Example walkthrough for nums = [1, 2, 3, 4]:

    After first pass (left products):
    result = [1, 1, 2, 6]
    - result[0] = 1 (nothing to the left)
    - result[1] = 1 (only 1 to the left)
    - result[2] = 1*2 = 2
    - result[3] = 1*2*3 = 6

    After second pass (multiply by right products):
    result = [24, 12, 8, 6]
    - result[3] = 6 * 1 = 6 (nothing to the right)
    - result[2] = 2 * 4 = 8
    - result[1] = 1 * 12 = 12
    - result[0] = 1 * 24 = 24
    """
    n = len(nums)
    result = [1] * n  # Initialize with 1s

    # ========== FIRST PASS: Calculate left products ==========
    # result[i] will contain product of all elements to the LEFT of i
    left_product = 1
    for i in range(n):
        result[i] = left_product      # Store product of elements before i
        left_product *= nums[i]        # Update for next iteration

    # After this: result = [1, 1, 2, 6] for nums = [1, 2, 3, 4]

    # ========== SECOND PASS: Multiply by right products ==========
    # Traverse from right, multiply each result[i] by product of elements to RIGHT
    right_product = 1
    for i in range(n - 1, -1, -1):    # Go from right to left
        result[i] *= right_product     # Multiply by product of elements after i
        right_product *= nums[i]       # Update for next iteration

    # After this: result = [24, 12, 8, 6]

    return result
```

### Complexity
- **Time**: O(n) - Two passes through array
- **Space**: O(1) - Only output array (doesn't count as extra space per problem)

### Edge Cases
- Array with zeros → product might be zero
- Single element → return [1] (empty product)
- Two elements [a, b] → return [b, a]
- Negative numbers → sign changes
- Very large products → potential overflow (use modulo if needed)

---

## Problem 4: Valid Sudoku (LC #36) - Medium

- [LeetCode](https://leetcode.com/problems/valid-sudoku/)

### Problem Statement
Determine if a 9×9 Sudoku board is valid. Only filled cells need to be validated according to:
1. Each row must contain digits 1-9 without repetition
2. Each column must contain digits 1-9 without repetition
3. Each of the 9 3×3 sub-boxes must contain digits 1-9 without repetition

### Video Explanation
- [NeetCode - Valid Sudoku](https://www.youtube.com/watch?v=TjFXEUCMqI8)

### Examples
```
Input: Valid board with some empty cells (".")
Output: true

Input: Board with duplicate in row/column/box
Output: false
```

### Intuition Development
```
Need to check three things for each number:
1. Not already in its row
2. Not already in its column
3. Not already in its 3×3 box

Use sets to track what we've seen:
- rows[r] = set of numbers seen in row r
- cols[c] = set of numbers seen in column c
- boxes[b] = set of numbers seen in box b

Box index formula: box_index = (row // 3) * 3 + (col // 3)

  0  1  2 | 3  4  5 | 6  7  8
  --------+---------+--------
0 [box 0] | [box 1] | [box 2]
1 [box 0] | [box 1] | [box 2]
2 [box 0] | [box 1] | [box 2]
  --------+---------+--------
3 [box 3] | [box 4] | [box 5]
...
```

### Solution
```python
def isValidSudoku(board: list[list[str]]) -> bool:
    """
    Validate a 9x9 Sudoku board.

    Strategy:
    - Use sets to track numbers seen in each row, column, and 3x3 box
    - For each cell, check if number already exists in its row/col/box
    - If duplicate found, return False

    Time: O(81) = O(1) - fixed 9x9 board
    Space: O(81) = O(1) - at most 81 numbers tracked
    """
    # Initialize sets for each row, column, and 3x3 box
    rows = [set() for _ in range(9)]    # rows[i] = numbers seen in row i
    cols = [set() for _ in range(9)]    # cols[j] = numbers seen in column j
    boxes = [set() for _ in range(9)]   # boxes[k] = numbers seen in box k

    # Iterate through every cell in the board
    for row in range(9):
        for col in range(9):
            num = board[row][col]

            # Skip empty cells (marked with ".")
            if num == ".":
                continue

            # Calculate which 3x3 box this cell belongs to
            # Box layout:
            # 0 1 2
            # 3 4 5
            # 6 7 8
            box_index = (row // 3) * 3 + (col // 3)

            # Check if this number already exists in row, column, or box
            if num in rows[row]:
                return False  # Duplicate in row
            if num in cols[col]:
                return False  # Duplicate in column
            if num in boxes[box_index]:
                return False  # Duplicate in 3x3 box

            # Mark this number as seen in its row, column, and box
            rows[row].add(num)
            cols[col].add(num)
            boxes[box_index].add(num)

    # No duplicates found - board is valid
    return True
```

### Complexity
- **Time**: O(1) - Board is always 9×9
- **Space**: O(1) - Fixed number of sets

### Edge Cases
- Empty board (all ".") → valid
- Single filled cell → valid if 1-9
- Fully filled valid board
- Duplicate in first/last row/column/box

---

## Problem 5: Longest Consecutive Sequence (LC #128) - Medium

- [LeetCode](https://leetcode.com/problems/longest-consecutive-sequence/)

### Problem Statement
Given an unsorted array of integers `nums`, return the length of the longest consecutive elements sequence. Must run in O(n) time.

### Video Explanation
- [NeetCode - Longest Consecutive Sequence](https://www.youtube.com/watch?v=P6RZZMu_maU)

### Examples
```
Input: nums = [100, 4, 200, 1, 3, 2]
Output: 4
Explanation: Longest consecutive sequence is [1, 2, 3, 4]

Input: nums = [0,3,7,2,5,8,4,6,0,1]
Output: 9
Explanation: [0,1,2,3,4,5,6,7,8]
```

### Intuition Development
```
Naive: Sort and scan → O(n log n) ❌

Better: Use a set for O(1) lookups

Key insight: Only start counting from the BEGINNING of a sequence
- A number is a sequence START if (num - 1) is NOT in the set
- From each start, count how long the sequence goes

nums = [100, 4, 200, 1, 3, 2]
set = {100, 4, 200, 1, 3, 2}

100: 99 not in set → START! 100, 101? No. Length = 1
4:   3 in set → NOT a start, skip
200: 199 not in set → START! 200, 201? No. Length = 1
1:   0 not in set → START! 1→2→3→4→5? Length = 4 ✓
3:   2 in set → NOT a start, skip
2:   1 in set → NOT a start, skip
```

### Solution
```python
def longestConsecutive(nums: list[int]) -> int:
    """
    Find the length of the longest consecutive sequence.

    Strategy:
    1. Put all numbers in a set for O(1) lookup
    2. For each number, check if it's the START of a sequence
       (i.e., num-1 is NOT in the set)
    3. If it's a start, count how long the sequence goes
    4. Track the maximum length found

    Why this is O(n):
    - Each number is visited at most twice:
      once in the outer loop, once when extending a sequence

    Time: O(n) - each number processed at most twice
    Space: O(n) - set stores all numbers
    """
    if not nums:
        return 0

    # Convert to set for O(1) lookup
    num_set = set(nums)
    longest = 0

    for num in num_set:
        # ===== KEY OPTIMIZATION =====
        # Only start counting if this is the BEGINNING of a sequence
        # A number is a sequence start if (num - 1) is NOT in the set
        if num - 1 not in num_set:
            # This is the start of a sequence!
            current_num = num
            current_length = 1

            # Count consecutive numbers
            while current_num + 1 in num_set:
                current_num += 1
                current_length += 1

            # Update longest if this sequence is longer
            longest = max(longest, current_length)

    return longest
```

### Complexity
- **Time**: O(n) - Each number visited at most twice
- **Space**: O(n) - Set stores all numbers

### Edge Cases
- Empty array → 0
- Single element → 1
- All same elements → 1
- Negative numbers (can form sequences like -3,-2,-1,0,1)
- Duplicates in array → use set to handle

---

## Problem 6: Encode and Decode Strings (LC #271) - Medium

- [LeetCode](https://leetcode.com/problems/encode-and-decode-strings/)

### Problem Statement
Design an algorithm to encode a list of strings to a single string, and decode it back to the original list.

### Video Explanation
- [NeetCode - Encode and Decode Strings](https://www.youtube.com/watch?v=B1k_sxOSgv8)

### Examples
```
Input: ["Hello","World"]
Encoded: "5#Hello5#World"
Decoded: ["Hello","World"]

Input: ["we","say",":","yes"]
Encoded: "2#we3#say1#:3#yes"
Decoded: ["we","say",":","yes"]
```

### Intuition Development
```
Challenge: Strings can contain ANY characters, including delimiters!

Solution: Use length-prefixed encoding
Format: length + delimiter + string

"Hello" → "5#Hello"
"World" → "5#World"

Combined: "5#Hello5#World"

To decode:
1. Read digits until '#'
2. Parse length
3. Read exactly that many characters
4. Repeat
```

### Solution
```python
class Codec:
    """
    Encode and decode strings using length-prefixed format.

    Format: Each string is encoded as: <length>#<string>

    Example:
    ["Hello", "World"] → "5#Hello5#World"

    Why this works:
    - Length tells us exactly how many characters to read
    - '#' separates length from content
    - Works even if strings contain '#' or digits
    """

    def encode(self, strs: list[str]) -> str:
        """
        Encode a list of strings into a single string.

        Format: For each string s, append: len(s) + '#' + s

        Time: O(n) where n = total characters in all strings
        Space: O(n) for the encoded string
        """
        encoded = []

        for s in strs:
            # Append length, delimiter, and the string itself
            # Example: "Hello" → "5#Hello"
            encoded.append(f"{len(s)}#{s}")

        return ''.join(encoded)

    def decode(self, s: str) -> list[str]:
        """
        Decode a single string back into a list of strings.

        Strategy:
        1. Find the '#' delimiter
        2. Parse the length before '#'
        3. Extract exactly that many characters after '#'
        4. Move pointer and repeat

        Time: O(n) where n = length of encoded string
        Space: O(n) for the decoded list
        """
        decoded = []
        i = 0  # Current position in string

        while i < len(s):
            # Step 1: Find the '#' delimiter
            j = i
            while s[j] != '#':
                j += 1

            # Step 2: Parse the length (characters from i to j-1)
            length = int(s[i:j])

            # Step 3: Extract the string (starts at j+1, has 'length' characters)
            # j is at '#', so string starts at j+1
            string_start = j + 1
            string_end = string_start + length
            decoded.append(s[string_start:string_end])

            # Step 4: Move pointer to start of next encoded string
            i = string_end

        return decoded
```

### Complexity
- **Encode**: Time O(n), Space O(n)
- **Decode**: Time O(n), Space O(n)

### Edge Cases
- Empty list → encode to "", decode back to []
- Empty string in list [""] → encode as "0#"
- Strings containing "#" → handled by length prefix
- Strings containing digits → handled by length prefix
- Very long strings → length prefix handles any size

---

## Summary: Medium Problems Checklist

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | Group Anagrams | Sorted key / Count tuple | O(nk log k) | O(nk) |
| 2 | Top K Frequent | Bucket sort / Heap | O(n) | O(n) |
| 3 | Product Except Self | Prefix/Suffix products | O(n) | O(1) |
| 4 | Valid Sudoku | Sets for row/col/box | O(1) | O(1) |
| 5 | Longest Consecutive | Set + sequence start | O(n) | O(n) |
| 6 | Encode/Decode Strings | Length-prefix encoding | O(n) | O(n) |

---

## Practice More Medium Problems

- [ ] LC #454 - 4Sum II
- [ ] LC #560 - Subarray Sum Equals K
- [ ] LC #380 - Insert Delete GetRandom O(1)
- [ ] LC #49 - Group Anagrams (if not done)
- [ ] LC #438 - Find All Anagrams in a String

