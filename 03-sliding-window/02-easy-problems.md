# Sliding Window - Easy Problems

## Problem 1: Maximum Average Subarray I (LC #643) - Easy

- [LeetCode](https://leetcode.com/problems/maximum-average-subarray-i/)

### Problem Statement
Find contiguous subarray of length `k` with maximum average value.

### Video Explanation
- [NeetCode - Maximum Average Subarray](https://www.youtube.com/watch?v=WpYHNHofwjc)

### Examples
```
Input: nums = [1,12,-5,-6,50,3], k = 4
Output: 12.75
Explanation: Maximum average is (12-5-6+50)/4 = 51/4 = 12.75
```

### Intuition Development
```
Fixed window of size k - classic sliding window!

nums = [1, 12, -5, -6, 50, 3], k = 4

Window 1: [1, 12, -5, -6] sum = 2, avg = 0.5
Window 2: [12, -5, -6, 50] sum = 51, avg = 12.75 ← max
Window 3: [-5, -6, 50, 3] sum = 42, avg = 10.5

Slide: new_sum = old_sum + new_element - old_element
```

### Solution
```python
def findMaxAverage(nums: list[int], k: int) -> float:
    """
    Find maximum average of any contiguous subarray of length k.

    Strategy (Fixed Size Sliding Window):
    1. Calculate sum of first k elements
    2. Slide window: add new element, remove old element
    3. Track maximum sum (average = sum / k)

    Time: O(n) - single pass through array
    Space: O(1) - only tracking sum and max
    """
    # ===== STEP 1: Initialize first window =====
    # Sum of first k elements
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # ===== STEP 2: Slide the window =====
    for i in range(k, len(nums)):
        # Add new element entering the window
        window_sum += nums[i]

        # Remove old element leaving the window
        # Element at index (i - k) is leaving
        window_sum -= nums[i - k]

        # Update maximum sum
        max_sum = max(max_sum, window_sum)

    # ===== STEP 3: Return maximum average =====
    # Divide by k to get average
    return max_sum / k
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- k equals array length: Return average of entire array
- All same values: Any window gives same average
- Negative numbers: Works correctly with sum formula
- Single window: k = n, just return sum/k

---

## Problem 2: Best Time to Buy and Sell Stock (LC #121) - Easy

- [LeetCode](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

### Problem Statement
Find maximum profit from buying and selling stock once. You must buy before you sell.

### Video Explanation
- [NeetCode - Best Time to Buy/Sell Stock](https://www.youtube.com/watch?v=1pkOgXD63yU)

### Examples
```
Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy at 1, sell at 6 → profit = 5

Input: prices = [7,6,4,3,1]
Output: 0
Explanation: Prices only decrease, no profit possible
```

### Intuition Development
```
Track minimum price seen so far, calculate profit at each day.

prices = [7, 1, 5, 3, 6, 4]

Day 0: price=7, min_price=7, profit=0
Day 1: price=1, min_price=1, profit=0
Day 2: price=5, min_price=1, profit=5-1=4
Day 3: price=3, min_price=1, profit=3-1=2
Day 4: price=6, min_price=1, profit=6-1=5 ← max
Day 5: price=4, min_price=1, profit=4-1=3

Maximum profit = 5
```

### Solution
```python
def maxProfit(prices: list[int]) -> int:
    """
    Find maximum profit from one buy-sell transaction.

    Strategy:
    - Track minimum price seen so far (best day to buy)
    - At each day, calculate profit if we sell today
    - Track maximum profit

    This is a "sliding window" in the sense that we're looking at
    the best window [buy_day, sell_day] where buy_day < sell_day.

    Time: O(n) - single pass
    Space: O(1) - only tracking min and max
    """
    if not prices:
        return 0

    min_price = float('inf')  # Minimum price seen so far (best buy price)
    max_profit = 0            # Maximum profit achievable

    for price in prices:
        # Update minimum price (best day to have bought)
        min_price = min(min_price, price)

        # Calculate profit if we sell today
        current_profit = price - min_price

        # Update maximum profit
        max_profit = max(max_profit, current_profit)

    return max_profit


def maxProfit_verbose(prices: list[int]) -> int:
    """
    Same solution with more detailed tracking for learning.
    """
    if not prices:
        return 0

    min_price = prices[0]  # Best price to buy at
    max_profit = 0         # Best profit so far

    for i in range(1, len(prices)):
        current_price = prices[i]

        # Option 1: Sell today (bought at min_price)
        profit_if_sell_today = current_price - min_price

        # Option 2: Don't sell, update min_price if today is cheaper
        if current_price < min_price:
            min_price = current_price

        # Track best profit
        max_profit = max(max_profit, profit_if_sell_today)

    return max_profit
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Edge Cases
- Decreasing prices: `[7,6,5,4]` → return `0` (no profit possible)
- Single day: `[5]` → return `0`
- Two days: `[1,5]` → return `4`
- Best buy at last day: No sale possible, return `0`

---

## Problem 3: Contains Duplicate II (LC #219) - Easy

- [LeetCode](https://leetcode.com/problems/contains-duplicate-ii/)

### Problem Statement
Check if array contains two equal elements within distance `k` of each other.

### Video Explanation
- [NeetCode - Contains Duplicate II](https://www.youtube.com/watch?v=ypn0aZ0nrL4)

### Examples
```
Input: nums = [1,2,3,1], k = 3
Output: true
Explanation: nums[0] = nums[3] = 1, and 3 - 0 = 3 ≤ k

Input: nums = [1,2,3,1,2,3], k = 2
Output: false
```

### Intuition Development
```
Maintain a sliding window of size k using a set.
If we see a duplicate within the window → return True.

nums = [1, 2, 3, 1, 2, 3], k = 2

i=0: window={1}
i=1: window={1,2}
i=2: window={1,2,3}  (window full, size = k+1)
     remove nums[0]=1, window={2,3}
i=3: check if 1 in {2,3}? No. window={2,3,1}
     remove nums[1]=2, window={3,1}
i=4: check if 2 in {3,1}? No. window={3,1,2}
     remove nums[2]=3, window={1,2}
i=5: check if 3 in {1,2}? No. window={1,2,3}

No duplicates found within distance k.
```

### Solution
```python
def containsNearbyDuplicate(nums: list[int], k: int) -> bool:
    """
    Check if array has duplicate elements within distance k.

    Strategy (Sliding Window with Set):
    - Maintain a set of elements in current window of size k
    - For each new element:
      1. Check if it's already in the set (duplicate within k!)
      2. Add it to the set
      3. If window exceeds size k, remove oldest element

    Time: O(n) - single pass, O(1) set operations
    Space: O(min(n, k)) - set stores at most k elements
    """
    # Set to track elements in current window
    window = set()

    for i, num in enumerate(nums):
        # Check if current number is already in window
        if num in window:
            return True  # Found duplicate within distance k!

        # Add current number to window
        window.add(num)

        # If window exceeds size k, remove the oldest element
        # Window should contain indices [i-k, i], so size is k+1
        if len(window) > k:
            # Remove element that's now too far (index i-k)
            window.remove(nums[i - k])

    return False  # No nearby duplicates found


def containsNearbyDuplicate_hashmap(nums: list[int], k: int) -> bool:
    """
    Alternative approach using hash map to store last index of each element.

    Time: O(n)
    Space: O(n) - stores all unique elements
    """
    # Map: element -> most recent index
    last_index = {}

    for i, num in enumerate(nums):
        if num in last_index:
            # Check if previous occurrence is within distance k
            if i - last_index[num] <= k:
                return True

        # Update last seen index
        last_index[num] = i

    return False
```

### Complexity
- **Set approach**: Time O(n), Space O(min(n, k))
- **Hash map approach**: Time O(n), Space O(n)

### Edge Cases
- k = 0: No window, return `False`
- k >= n: Entire array is one window
- Adjacent duplicates: `[1,1], k=1` → return `True`
- No duplicates: Return `False`

---

## Problem 4: Number of Sub-arrays of Size K with Average >= Threshold (LC #1343) - Easy

- [LeetCode](https://leetcode.com/problems/number-of-sub-arrays-of-size-k-with-average-threshold/)

### Problem Statement
Count subarrays of size `k` with average greater than or equal to `threshold`.

### Video Explanation
- [LeetCode - Subarray Average](https://leetcode.com/problems/number-of-sub-arrays-of-size-k-and-average-greater-than-or-equal-to-threshold/)

### Examples
```
Input: arr = [2,2,2,2,5,5,5,8], k = 3, threshold = 4
Output: 3
Explanation:
- [2,5,5] avg = 4 ✓
- [5,5,5] avg = 5 ✓
- [5,5,8] avg = 6 ✓
```

### Solution
```python
def numOfSubarrays(arr: list[int], k: int, threshold: int) -> int:
    """
    Count subarrays of size k with average >= threshold.

    Strategy (Fixed Size Sliding Window):
    - Instead of comparing average, compare sum
    - avg >= threshold equivalent to sum >= threshold * k
    - Slide window and count valid windows

    Time: O(n) - single pass
    Space: O(1) - only tracking sum and count
    """
    # Calculate target sum (avoid division)
    target_sum = threshold * k

    # Initialize first window
    window_sum = sum(arr[:k])
    count = 1 if window_sum >= target_sum else 0

    # Slide the window
    for i in range(k, len(arr)):
        # Update window sum: add new, remove old
        window_sum += arr[i] - arr[i - k]

        # Check if current window meets threshold
        if window_sum >= target_sum:
            count += 1

    return count
```

### Complexity
- **Time**: O(n)
- **Space**: O(1)

### Intuition Development
```
Convert average comparison to sum comparison!

avg >= threshold is equivalent to:
sum / k >= threshold
sum >= threshold * k

This avoids floating point issues and is faster.

arr = [2, 2, 2, 2, 5, 5, 5, 8], k = 3, threshold = 4
target_sum = 4 * 3 = 12

Window [2,2,2]: sum = 6 < 12 ✗
Window [2,2,5]: sum = 9 < 12 ✗
Window [2,5,5]: sum = 12 >= 12 ✓
Window [5,5,5]: sum = 15 >= 12 ✓
Window [5,5,8]: sum = 18 >= 12 ✓

Count = 3
```

### Edge Cases
- All elements meet threshold: Count all windows
- No window meets threshold: Return 0
- k = 1: Each element is a window, compare directly
- Threshold = 0: All windows with non-negative sum count

---

## Problem 5: Longest Nice Substring (LC #1763) - Easy

- [LeetCode](https://leetcode.com/problems/longest-nice-substring/)

### Problem Statement
A string is "nice" if for every letter, both uppercase and lowercase appear. Find the longest nice substring.

### Video Explanation
- [LeetCode - Longest Nice Substring](https://leetcode.com/problems/longest-nice-substring/)

### Examples
```
Input: s = "YazaAay"
Output: "aAa"

Input: s = "Bb"
Output: "Bb"

Input: s = "c"
Output: ""
```

### Solution
```python
def longestNiceSubstring(s: str) -> str:
    """
    Find longest substring where every letter has both cases present.

    Strategy (Divide and Conquer):
    - If string is nice, return it
    - Find a character that doesn't have both cases
    - Split string at that character and recurse on both halves
    - Return the longer nice substring from the halves

    Why divide and conquer?
    - A character without its pair CAN'T be in a nice substring
    - So the nice substring must be entirely to its left or right

    Time: O(n²) worst case, but often faster
    Space: O(n) for recursion
    """
    if len(s) < 2:
        return ""

    # Check which characters have both cases
    char_set = set(s)

    for i, char in enumerate(s):
        # If this character doesn't have its pair, split here
        if char.lower() not in char_set or char.upper() not in char_set:
            # Recurse on both halves (excluding the bad character)
            left = longestNiceSubstring(s[:i])
            right = longestNiceSubstring(s[i+1:])

            # Return the longer one (prefer left if equal)
            return left if len(left) >= len(right) else right

    # All characters have both cases - entire string is nice
    return s


def longestNiceSubstring_bruteforce(s: str) -> str:
    """
    Brute force approach: Check all substrings.

    Time: O(n³) - O(n²) substrings, O(n) to check each
    Space: O(n) for set
    """
    def is_nice(sub: str) -> bool:
        """Check if substring is nice."""
        char_set = set(sub)
        for char in char_set:
            if char.lower() not in char_set or char.upper() not in char_set:
                return False
        return True

    longest = ""
    n = len(s)

    for i in range(n):
        for j in range(i + 2, n + 1):  # Need at least 2 characters
            substring = s[i:j]
            if is_nice(substring) and len(substring) > len(longest):
                longest = substring

    return longest
```

### Complexity
- **Divide and Conquer**: Time O(n²), Space O(n)
- **Brute Force**: Time O(n³), Space O(n)

### Intuition Development
```
A character without its pair CANNOT be in any nice substring!

s = "YazaAay"

Check each character:
- 'Y' has no 'y' → Y can't be in nice substring
- 'a' has 'A' → OK
- 'z' has no 'Z' → z can't be in nice substring
- 'A' has 'a' → OK
- 'y' has no 'Y' → y can't be in nice substring

Split at 'Y': "" | "azaAay"
Split at 'z': "a" | "aAay" → but 'y' has no 'Y'
Continue splitting until we find "aAa" which is nice!
```

### Edge Cases
- Single character: `"a"` → `""` (can't be nice alone)
- Two same case: `"aa"` → `""` (no uppercase)
- Perfect pair: `"aA"` → `"aA"`
- Multiple nice substrings: Return longest (leftmost if tie)

---

## Problem 6: Defuse the Bomb (LC #1652) - Easy

- [LeetCode](https://leetcode.com/problems/defuse-the-bomb/)

### Problem Statement
Circular array code. Replace each element with sum of next/previous k elements based on k's sign.

### Video Explanation
- [LeetCode - Defuse the Bomb](https://leetcode.com/problems/defuse-the-bomb/)

### Examples
```
Input: code = [5,7,1,4], k = 3
Output: [12,10,16,13]
Explanation:
- code[0] = 7+1+4 = 12 (next 3)
- code[1] = 1+4+5 = 10 (next 3, wrapping)
- etc.

Input: code = [2,4,9,3], k = -2
Output: [12,5,6,13]
Explanation: Sum of previous 2 elements
```

### Solution
```python
def decrypt(code: list[int], k: int) -> list[int]:
    """
    Decrypt circular array by replacing each element with sum of k neighbors.

    Strategy:
    - k > 0: sum of next k elements
    - k < 0: sum of previous |k| elements
    - k = 0: all zeros

    Use sliding window on circular array (modulo for wrapping).

    Time: O(n) - single pass with sliding window
    Space: O(n) - result array
    """
    n = len(code)
    result = [0] * n

    if k == 0:
        return result

    # Determine window start and direction
    if k > 0:
        # Sum next k elements: window starts at index 1
        start = 1
        end = k
    else:
        # Sum previous |k| elements: window starts at index -|k|
        start = n + k  # Equivalent to -|k| in circular array
        end = n - 1
        k = -k  # Make k positive for window size

    # Calculate initial window sum
    window_sum = sum(code[start:(start + k) % n] if start + k <= n
                     else code[start:] + code[:(start + k) % n])

    # Actually, let's use a cleaner approach with modulo
    window_sum = 0
    for i in range(start, start + k):
        window_sum += code[i % n]

    # Slide the window for each position
    for i in range(n):
        result[i] = window_sum

        # Slide window: remove element leaving, add element entering
        window_sum -= code[(start + i) % n]
        window_sum += code[(start + i + k) % n]

    return result


def decrypt_simple(code: list[int], k: int) -> list[int]:
    """
    Simpler but slightly less efficient approach.

    Time: O(n * k) - for each position, sum k elements
    Space: O(n) - result array
    """
    n = len(code)
    result = [0] * n

    if k == 0:
        return result

    for i in range(n):
        if k > 0:
            # Sum next k elements
            for j in range(1, k + 1):
                result[i] += code[(i + j) % n]
        else:
            # Sum previous |k| elements
            for j in range(1, -k + 1):
                result[i] += code[(i - j) % n]

    return result
```

### Complexity
- **Sliding Window**: Time O(n), Space O(n)
- **Simple**: Time O(n × k), Space O(n)

### Intuition Development
```
Circular array means we use modulo for wrapping!

code = [5, 7, 1, 4], k = 3 (next 3 elements)

For code[0]: sum of code[1], code[2], code[3] = 7+1+4 = 12
For code[1]: sum of code[2], code[3], code[0] = 1+4+5 = 10 (wraps!)
For code[2]: sum of code[3], code[0], code[1] = 4+5+7 = 16
For code[3]: sum of code[0], code[1], code[2] = 5+7+1 = 13

Result: [12, 10, 16, 13]

Use modulo: (i + j) % n for circular indexing
```

### Edge Cases
- k = 0: Return all zeros
- k positive: Sum next k elements (circular)
- k negative: Sum previous |k| elements (circular)
- k >= n: Will wrap around multiple times (but problem constrains k < n)

---

## Summary: Easy Problems Checklist

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | Max Average Subarray | Fixed window sum | O(n) | O(1) |
| 2 | Best Time Buy/Sell Stock | Track min price | O(n) | O(1) |
| 3 | Contains Duplicate II | Window set of size k | O(n) | O(k) |
| 4 | Subarrays with Avg >= K | Fixed window count | O(n) | O(1) |
| 5 | Longest Nice Substring | Divide and conquer | O(n²) | O(n) |
| 6 | Defuse the Bomb | Circular sliding window | O(n) | O(n) |

---

## Practice More Easy Problems

- [ ] LC #594 - Longest Harmonious Subsequence
- [ ] LC #1984 - Minimum Difference Between Highest and Lowest of K Scores
- [ ] LC #2269 - Find the K-Beauty of a Number
- [ ] LC #1876 - Substrings of Size Three with Distinct Characters

