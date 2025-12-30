# Trie - Advanced Problems & Applications

## Advanced Trie Variants

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRIE VARIANTS                                            │
│                                                                             │
│  1. STANDARD TRIE: Basic prefix tree                                        │
│     - insert, search, startsWith                                            │
│                                                                             │
│  2. COMPRESSED TRIE (Radix Tree): Merge single-child chains                 │
│     - Saves space for sparse tries                                          │
│                                                                             │
│  3. SUFFIX TRIE: Store all suffixes                                         │
│     - Pattern matching, substring search                                    │
│                                                                             │
│  4. BITWISE TRIE (XOR Trie): Binary representation                          │
│     - Maximum XOR problems                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem 1: Maximum XOR of Two Numbers (LC #421) - Medium

- [LeetCode](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/)

### Problem Statement
Find maximum XOR of any two numbers in array.

### Examples
```
Input: nums = [3,10,5,25,2,8]
Output: 28

3  = 00011
25 = 11001
3 XOR 25 = 11010 = 26

5  = 00101
25 = 11001
5 XOR 25 = 11100 = 28 ← Maximum!
```

### Intuition Development
```
BITWISE TRIE APPROACH:
Build trie of binary representations (32 bits, MSB first).

For each number, traverse trie taking OPPOSITE bits when possible.
This maximizes XOR bit by bit from MSB.

nums = [3, 10, 5, 25]

Build trie:
         root
        /    \
       0      1
      / \      \
     0   1      1
    /     \      \
   0       0      0
   ...

For num = 5 (00101):
  Try opposite of 0 → 1 exists! XOR bit = 1
  Try opposite of 0 → 1 exists! XOR bit = 1
  Try opposite of 1 → 0 exists! XOR bit = 1
  ...

Result: 11100 = 28
```

### Video Explanation
- [NeetCode - Maximum XOR of Two Numbers](https://www.youtube.com/watch?v=EIhAwfHubE8)

### Solution
```python
class TrieNode:
    def __init__(self):
        self.children = {}  # 0 or 1


def findMaximumXOR(nums: list[int]) -> int:
    """
    Find maximum XOR using bitwise trie.

    Strategy:
    - Build trie of binary representations (32 bits)
    - For each number, traverse trie taking opposite bits
    - This maximizes XOR bit by bit
    """
    # Build trie
    root = TrieNode()

    for num in nums:
        node = root
        for i in range(31, -1, -1):  # MSB to LSB
            bit = (num >> i) & 1
            if bit not in node.children:
                node.children[bit] = TrieNode()
            node = node.children[bit]

    # Find max XOR for each number
    max_xor = 0

    for num in nums:
        node = root
        current_xor = 0

        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            opposite = 1 - bit

            # Try to take opposite bit (maximizes XOR)
            if opposite in node.children:
                current_xor |= (1 << i)
                node = node.children[opposite]
            else:
                node = node.children[bit]

        max_xor = max(max_xor, current_xor)

    return max_xor
```

### Complexity
- **Time**: O(32n) = O(n)
- **Space**: O(32n) = O(n)

### Edge Cases
- Single element → XOR with itself = 0
- All same elements → 0
- Powers of 2 → specific XOR patterns
- Negative numbers (need unsigned handling)

---

## Problem 2: Palindrome Pairs (LC #336) - Hard

- [LeetCode](https://leetcode.com/problems/palindrome-pairs/)

### Problem Statement
Find all pairs where concatenation is palindrome.

### Examples
```
Input: words = ["abcd","dcba","lls","s","sssll"]
Output: [[0,1],[1,0],[3,2],[2,4]]

"abcd" + "dcba" = "abcddcba" ✓
"dcba" + "abcd" = "dcbaabcd" ✓
"s" + "lls" = "slls" ✓
"lls" + "sssll" = "llssssll" ✓
```

### Intuition Development
```
TRIE OF REVERSED WORDS:
Build trie of reversed words.
For each word, search for palindrome pairs.

Cases:
1. word + reversed(word) → exact match
2. word is longer: remaining suffix must be palindrome
3. word is shorter: remaining prefix in trie must be palindrome

Example: words = ["abcd", "dcba"]

Trie of reversed: "dcba" → d-c-b-a#, "abcd" → a-b-c-d#

Search "abcd":
  Match a-b-c-d → find #end at "dcba" index
  "abcd" + "dcba" = palindrome ✓
```

### Video Explanation
- [NeetCode - Palindrome Pairs](https://www.youtube.com/watch?v=iTwnWsK9xEQ)

### Solution
```python
def palindromePairs(words: list[str]) -> list[list[int]]:
    """
    Find palindrome pairs using trie of reversed words.

    Cases:
    1. word + reversed(word) - exact match
    2. word is longer, remaining suffix is palindrome
    3. word is shorter, remaining prefix in trie is palindrome
    """
    def is_palindrome(s, start, end):
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True

    # Build trie of reversed words
    root = {}

    for i, word in enumerate(words):
        node = root
        reversed_word = word[::-1]

        for j, char in enumerate(reversed_word):
            # Check if remaining suffix is palindrome
            if is_palindrome(reversed_word, j, len(reversed_word) - 1):
                node.setdefault('#palindrome', []).append(i)

            node = node.setdefault(char, {})

        node['#end'] = i

    result = []

    for i, word in enumerate(words):
        node = root

        for j, char in enumerate(word):
            # Case: word is longer, check if remaining is palindrome
            if '#end' in node and node['#end'] != i:
                if is_palindrome(word, j, len(word) - 1):
                    result.append([i, node['#end']])

            if char not in node:
                break
            node = node[char]
        else:
            # Reached end of word
            # Case: exact match (word + reversed word)
            if '#end' in node and node['#end'] != i:
                result.append([i, node['#end']])

            # Case: word is shorter, remaining in trie is palindrome
            if '#palindrome' in node:
                for j in node['#palindrome']:
                    if j != i:
                        result.append([i, j])

    return result
```

### Complexity
- **Time**: O(n × k²) where k = avg word length
- **Space**: O(n × k)

### Edge Cases
- Empty string in words → pairs with all palindromes
- Single character words
- All same words → no valid pairs
- Word is its own reverse

---

## Problem 3: Word Search II (LC #212) - Hard

- [LeetCode](https://leetcode.com/problems/word-search-ii/)

### Problem Statement
Find all dictionary words in a 2D board.

### Examples
```
Input: board = [["o","a","a","n"],
                ["e","t","a","e"],
                ["i","h","k","r"],
                ["i","f","l","v"]],
       words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
```

### Intuition Development
```
TRIE + DFS:
Build trie from dictionary.
DFS from each cell, following trie paths.

Board:
  o a a n
  e t a e
  i h k r
  i f l v

Trie for ["oath", "pea", "eat", "rain"]:
        root
       /  |  \  \
      o   p   e  r
      |   |   |  |
      a   e   a  a
      |   |   |  |
      t   a   t  i
      |       |  |
      h       #  n
      |          |
      #          #

DFS from 'o' at (0,0):
  o → a (0,1) → t (1,1) → h (2,1) → # found "oath"!

DFS from 'e' at (1,0):
  e → a (0,1 or 1,2) → t (1,1) → # found "eat"!
```

### Video Explanation
- [NeetCode - Word Search II](https://www.youtube.com/watch?v=asbcE9mZz_U)

### Solution
```python
def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    """
    Find all words from dictionary in board using Trie + DFS.

    Optimization:
    - Remove found words from trie to avoid duplicates
    - Prune empty branches
    """
    # Build trie
    root = {}
    for word in words:
        node = root
        for char in word:
            node = node.setdefault(char, {})
        node['#'] = word

    rows, cols = len(board), len(board[0])
    result = []

    def dfs(r, c, node):
        char = board[r][c]

        if char not in node:
            return

        next_node = node[char]

        # Found a word
        if '#' in next_node:
            result.append(next_node['#'])
            del next_node['#']  # Remove to avoid duplicates

        # Mark visited
        board[r][c] = '*'

        # Explore neighbors
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and board[nr][nc] != '*':
                dfs(nr, nc, next_node)

        # Restore
        board[r][c] = char

        # Prune empty branches
        if not next_node:
            del node[char]

    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)

    return result
```

### Complexity
- **Time**: O(m × n × 4^L × W) where L = max word length, W = words count
- **Space**: O(W × L)

### Edge Cases
- Empty board or empty words list
- Single cell board
- No words found
- Overlapping words sharing same path
- Word longer than total cells

---

## Problem 4: Search Suggestions System (LC #1268) - Medium

- [LeetCode](https://leetcode.com/problems/search-suggestions-system/)

### Problem Statement
Return list of suggestions as user types each character.

### Examples
```
Input: products = ["mobile","mouse","moneypot","monitor","mousepad"],
       searchWord = "mouse"
Output: [["mobile","moneypot","monitor"],
         ["mobile","moneypot","monitor"],
         ["mouse","mousepad"],
         ["mouse","mousepad"],
         ["mouse","mousepad"]]
```

### Intuition Development
```
SORTED PRODUCTS + TRIE:
Sort products first, then store at most 3 per prefix.

Sorted: ["mobile", "moneypot", "monitor", "mouse", "mousepad"]

Build trie with top 3 at each node:
  m: ["mobile", "moneypot", "monitor"]
  mo: ["mobile", "moneypot", "monitor"]
  mou: ["mouse", "mousepad"]
  mous: ["mouse", "mousepad"]
  mouse: ["mouse", "mousepad"]

Query "mouse":
  m → ["mobile", "moneypot", "monitor"]
  mo → ["mobile", "moneypot", "monitor"]
  mou → ["mouse", "mousepad"]
  mous → ["mouse", "mousepad"]
  mouse → ["mouse", "mousepad"]
```

### Video Explanation
- [NeetCode - Search Suggestions System](https://www.youtube.com/watch?v=D4T2N0yAr20)

### Solution
```python
def suggestedProducts(products: list[str], searchWord: str) -> list[list[str]]:
    """
    Return top 3 suggestions for each prefix.

    Strategy:
    - Sort products lexicographically
    - Build trie with sorted product lists at each node
    """
    # Sort products
    products.sort()

    # Build trie
    root = {}

    for product in products:
        node = root
        for char in product:
            node = node.setdefault(char, {'#': []})
            # Keep only top 3
            if len(node['#']) < 3:
                node['#'].append(product)

    result = []
    node = root

    for i, char in enumerate(searchWord):
        if node and char in node:
            node = node[char]
            result.append(node['#'])
        else:
            # No more matches
            node = None
            result.append([])

    return result


def suggestedProducts_binary_search(products: list[str], searchWord: str) -> list[list[str]]:
    """
    Alternative: Binary search approach.
    """
    import bisect

    products.sort()
    result = []
    prefix = ""

    for char in searchWord:
        prefix += char

        # Find leftmost product with this prefix
        idx = bisect.bisect_left(products, prefix)

        # Get up to 3 products starting with prefix
        suggestions = []
        for i in range(idx, min(idx + 3, len(products))):
            if products[i].startswith(prefix):
                suggestions.append(products[i])
            else:
                break

        result.append(suggestions)

    return result
```

### Complexity
- **Time**: O(n × L + m × 3) where n = products, L = avg length, m = searchWord length
- **Space**: O(n × L)

### Edge Cases
- No matching products → empty lists
- All products match → return top 3 each time
- Single character search word
- Products with common prefixes

---

## Problem 5: Concatenated Words (LC #472) - Hard

- [LeetCode](https://leetcode.com/problems/concatenated-words/)

### Problem Statement
Find all words that are concatenation of other words in list.

### Examples
```
Input: words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]
Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]

"catsdogcats" = "cat" + "s" + "dog" + "cats" ✗ (s not in list)
Wait, "cats" is in list!
"catsdogcats" = "cats" + "dog" + "cats" ✓
```

### Intuition Development
```
TRIE + DFS:
Build trie from all words.
For each word, check if it can be split into 2+ trie words.

Word "catsdogcats":
  Start at root
  c-a-t-s → end found! Try remaining "dogcats"
    d-o-g → end found! Try remaining "cats"
      c-a-t-s → end found! Remaining empty!
      Count = 3 ≥ 2 → valid!

DFS explores all possible splits.
```

### Video Explanation
- [NeetCode - Concatenated Words](https://www.youtube.com/watch?v=iHp7fjw1R28)

### Solution
```python
def findAllConcatenatedWordsInADict(words: list[str]) -> list[str]:
    """
    Find words that are concatenation of other words.

    Strategy:
    - Build trie from all words
    - For each word, check if it can be split into multiple trie words
    """
    # Build trie
    root = {}

    for word in words:
        if not word:
            continue
        node = root
        for char in word:
            node = node.setdefault(char, {})
        node['#'] = True

    def can_form(word, start, count):
        """Check if word[start:] can be formed by trie words."""
        if start == len(word):
            return count >= 2  # Need at least 2 words

        node = root
        for i in range(start, len(word)):
            char = word[i]
            if char not in node:
                return False

            node = node[char]

            # If this is end of a word, try to form rest
            if '#' in node:
                if can_form(word, i + 1, count + 1):
                    return True

        return False

    result = []
    for word in words:
        if word and can_form(word, 0, 0):
            result.append(word)

    return result
```

### Complexity
- **Time**: O(n × L²) where L = max word length
- **Space**: O(n × L)

### Edge Cases
- Empty string in words (skip)
- Single word list → empty result
- All single characters → empty result
- Very long concatenated words

---

## Problem 6: Prefix and Suffix Search (LC #745) - Hard

- [LeetCode](https://leetcode.com/problems/prefix-and-suffix-search/)

### Problem Statement
Design data structure for prefix-suffix search.

### Examples
```
Input: words = ["apple"]
WordFilter wf = new WordFilter(words);
wf.f("a", "e"); // returns 0 (apple matches)
wf.f("b", ""); // returns -1 (no match)
```

### Intuition Development
```
COMBINED PREFIX#SUFFIX TRIE:
For word "apple", store all suffix#word combinations:
  "#apple"     (no suffix constraint)
  "e#apple"    (suffix "e")
  "le#apple"   (suffix "le")
  "ple#apple"  (suffix "ple")
  "pple#apple" (suffix "pple")
  "apple#apple" (suffix "apple")

Query f("ap", "le"):
  Search for "le#ap" in trie
  Matches "le#apple" at position where "ap" prefix exists
  Return stored weight (index)
```

### Video Explanation
- [NeetCode - Prefix and Suffix Search](https://www.youtube.com/watch?v=OOmRXcDRMYg)

### Solution
```python
class WordFilter:
    """
    Search by prefix and suffix.

    Strategy:
    - Store all prefix#suffix combinations in trie
    - For word "apple", store: "#apple", "e#apple", "le#apple", etc.
    """

    def __init__(self, words: list[str]):
        self.trie = {}

        for weight, word in enumerate(words):
            # Generate all suffix#prefix combinations
            for i in range(len(word) + 1):
                # suffix is word[i:], combined as suffix#word
                key = word[i:] + '#' + word

                node = self.trie
                for char in key:
                    node = node.setdefault(char, {})
                    node['weight'] = weight  # Store latest weight

    def f(self, pref: str, suff: str) -> int:
        """Find word with given prefix and suffix."""
        key = suff + '#' + pref

        node = self.trie
        for char in key:
            if char not in node:
                return -1
            node = node[char]

        return node.get('weight', -1)
```

### Complexity
- **Time**: O(n × L²) build, O(p + s) query
- **Space**: O(n × L²)

### Edge Cases
- Empty prefix or suffix → still valid query
- Multiple words with same prefix/suffix → return highest index
- Single character words
- Very long words

---

## Summary: Advanced Trie Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Maximum XOR | Bitwise trie, opposite bits | O(32n) |
| 2 | Palindrome Pairs | Reversed trie + palindrome check | O(n × k²) |
| 3 | Word Search II | Trie + DFS + pruning | O(mn × 4^L) |
| 4 | Search Suggestions | Sorted trie or binary search | O(n log n) |
| 5 | Concatenated Words | Trie + DFS word splitting | O(n × L²) |
| 6 | Prefix-Suffix Search | suffix#prefix combinations | O(n × L²) |

---

## When to Use Trie

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRIE DECISION GUIDE                                      │
│                                                                             │
│  USE TRIE when:                                                             │
│  • Prefix-based operations (autocomplete, prefix search)                    │
│  • Word dictionary with many lookups                                        │
│  • Pattern matching with wildcards                                          │
│  • XOR operations on binary numbers                                         │
│                                                                             │
│  USE HASH SET when:                                                         │
│  • Only exact word matching needed                                          │
│  • No prefix operations                                                     │
│  • Memory is constrained                                                    │
│                                                                             │
│  TRIE OPTIMIZATIONS:                                                        │
│  • Compressed trie for sparse data                                          │
│  • Array children for fixed alphabet (faster)                               │
│  • Prune empty branches during search                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practice More Problems

- [ ] LC #1032 - Stream of Characters
- [ ] LC #1707 - Maximum XOR With an Element From Array
- [ ] LC #1803 - Count Pairs With XOR in a Range
- [ ] LC #2416 - Sum of Prefix Scores of Strings
