# Trie - Hard Problems

## Problem 1: Word Search II (LC #212) - Hard

- [LeetCode](https://leetcode.com/problems/word-search-ii/)

### Video Explanation
- [NeetCode - Word Search II](https://www.youtube.com/watch?v=asbcE9mZz_U)

### Problem Statement
Find all words from dictionary that exist in the board.


### Visual Intuition
```
Word Search II (Trie + DFS)
board = [["o","a","a","n"],     words = ["oath","pea"]
         ["e","t","a","e"],
         ["i","h","k","r"],
         ["i","f","l","v"]]

Build Trie:
      root
      /  \
     o    p
     |    |
     a    e
     |    |
     t    a*
     |
     h*

DFS from each cell following Trie paths:
  Start (0,0)='o' → Trie has 'o'
  Move (0,1)='a' → Trie: o→a exists
  Move (1,1)='t' → Trie: o→a→t exists
  Move (2,1)='h' → Trie: o→a→t→h* (word end!)

Found: "oath"
```

### Solution
```python
def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    """
    Trie + DFS backtracking.

    Time: O(m * n * 4^L * W) where L = max word length
    Space: O(total chars in words)
    """
    # Build Trie
    trie = {}
    for word in words:
        node = trie
        for c in word:
            node = node.setdefault(c, {})
        node["$"] = word  # Store complete word

    m, n = len(board), len(board[0])
    result = []

    def dfs(i, j, node):
        char = board[i][j]
        if char not in node:
            return

        next_node = node[char]

        # Found a word
        if "$" in next_node:
            result.append(next_node["$"])
            del next_node["$"]  # Avoid duplicates

        # Mark visited
        board[i][j] = "#"

        # Explore 4 directions
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and board[ni][nj] != "#":
                dfs(ni, nj, next_node)

        # Restore cell
        board[i][j] = char

    for i in range(m):
        for j in range(n):
            dfs(i, j, trie)

    return result
```

### Edge Cases
- No words found → return []
- All words found → return all
- Single cell board → check single char words
- Overlapping words → Trie handles

---

## Problem 2: Palindrome Pairs (LC #336) - Hard

- [LeetCode](https://leetcode.com/problems/palindrome-pairs/)

### Video Explanation
- [NeetCode - Palindrome Pairs](https://www.youtube.com/watch?v=j4EH0I0SWQY)

### Problem Statement
Find all pairs of indices forming palindromes when concatenated.


### Visual Intuition
```
Palindrome Pairs
words = ["bat","tab","cat"]

═══════════════════════════════════════════════════════════════
KEY INSIGHT: word1 + word2 is palindrome if:
             - word2 == reverse(word1), OR
             - word1 has palindrome prefix, rest matches reverse(word2)
             - word1 has palindrome suffix, rest matches reverse(word2)
═══════════════════════════════════════════════════════════════

Word Map: {bat:0, tab:1, cat:2}

Check each word:
──────────────────
Word "bat" (index 0):
  Check all split points:

  j=0: prefix="" (palindrome), suffix="bat"
       reverse(suffix) = "tab" → exists at index 1!
       ["tab", "bat"] → "tabbat" ✓ → pair [1, 0]

  j=1: prefix="b" (palindrome), suffix="at"
       reverse(suffix) = "ta" → not in words

  j=2: prefix="ba" (not palindrome), skip

  j=3: prefix="bat" (not palindrome), skip
       suffix="" (palindrome), prefix="bat"
       reverse(prefix) = "tab" → exists at index 1!
       ["bat", "tab"] → "battab" ✓ → pair [0, 1]

Word "tab" (index 1):
  j=0: prefix="", suffix="tab"
       reverse("tab") = "bat" → exists at index 0!
       ["bat", "tab"] → "battab" (already found)

  j=3: suffix="", prefix="tab"
       reverse("tab") = "bat" → exists!
       ["tab", "bat"] → "tabbat" (already found)

Word "cat" (index 2):
  reverse("cat") = "tac" → not in words
  No palindrome pairs

Result: [[0,1], [1,0]]

Visual of Palindrome Formation:
───────────────────────────────
  "bat" + "tab" = "battab"
   ↓↓↓    ↓↓↓
   b a t  t a b
   └─┴─┴──┴─┴─┘ reads same forwards & backwards ✓

  "tab" + "bat" = "tabbat"
   ↓↓↓    ↓↓↓
   t a b  b a t
   └─┴─┴──┴─┴─┘ reads same forwards & backwards ✓

WHY THIS WORKS:
════════════════
● If word2 = reverse(word1), then word1+word2 is palindrome
● If word1 = palindrome_part + rest, and reverse(rest) exists,
  then reverse(rest) + word1 is palindrome
● Hash map gives O(1) lookup for reverse words
```

### Solution
```python
def palindromePairs(words: list[str]) -> list[list[int]]:
    """
    Use Trie with reversed words.

    Time: O(n * k²) where k = avg word length
    Space: O(n * k)
    """
    def is_palindrome(s, start, end):
        while start < end:
            if s[start] != s[end]:
                return False
            start += 1
            end -= 1
        return True

    # Map word to index
    word_map = {word: i for i, word in enumerate(words)}
    result = []

    for i, word in enumerate(words):
        n = len(word)

        for j in range(n + 1):
            # Case 1: prefix is palindrome, check if suffix reversed exists
            if is_palindrome(word, 0, j - 1):
                suffix_rev = word[j:][::-1]
                if suffix_rev in word_map and word_map[suffix_rev] != i:
                    result.append([word_map[suffix_rev], i])

            # Case 2: suffix is palindrome, check if prefix reversed exists
            if j != n and is_palindrome(word, j, n - 1):
                prefix_rev = word[:j][::-1]
                if prefix_rev in word_map and word_map[prefix_rev] != i:
                    result.append([i, word_map[prefix_rev]])

    return result
```

### Edge Cases
- Empty string in words → pairs with palindromes
- Single char words → check reverse exists
- All same word → no pairs with self
- Word is palindrome → check empty string

---

## Problem 3: Stream of Characters (LC #1032) - Hard

- [LeetCode](https://leetcode.com/problems/stream-of-characters/)

### Video Explanation
- [NeetCode - Stream of Characters](https://www.youtube.com/watch?v=pG3ALxrLfAc)

### Problem Statement
Query if suffix of stream matches any word in dictionary.


### Visual Intuition
```
Stream of Characters (Suffix Trie)
words = ["cd","f","kl"]
stream: 'a','b','c','d'

Build Trie of REVERSED words:
  "cd" → "dc", "f" → "f", "kl" → "lk"

      root
     / | \
    d  f* l
    |     |
    c*    k*

Query: check if any word ends at current position
  Stream buffer: "abcd"

  Search reversed: "dcba"
    d → c* → FOUND "cd"!

  Return True (word "cd" ends here)

Keep buffer of max word length, search backwards
```

### Solution
```python
from collections import deque

class StreamChecker:
    """
    Build Trie with reversed words, check stream suffix.

    Time: O(L) per query where L = max word length
    Space: O(total chars)
    """

    def __init__(self, words: list[str]):
        self.trie = {}
        self.stream = deque()
        self.max_len = 0

        # Build Trie with reversed words
        for word in words:
            self.max_len = max(self.max_len, len(word))
            node = self.trie
            for c in reversed(word):
                node = node.setdefault(c, {})
            node["$"] = True

    def query(self, letter: str) -> bool:
        self.stream.appendleft(letter)

        # Limit stream size
        if len(self.stream) > self.max_len:
            self.stream.pop()

        # Check if any word matches suffix
        node = self.trie
        for c in self.stream:
            if c not in node:
                return False
            node = node[c]
            if "$" in node:
                return True

        return False
```

### Edge Cases
- First query → check single char words
- Stream longer than max word → trim old chars
- No matching suffix → return False
- Word at stream start → may not be suffix

---

## Problem 4: Design Add and Search Words (LC #211) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/design-add-and-search-words-data-structure/)

### Video Explanation
- [NeetCode - Design Add and Search Words](https://www.youtube.com/watch?v=BTf05gs_8iU)

### Problem Statement
Design data structure supporting addWord and search with '.' wildcard.

### Visual Intuition
```
Design Add and Search Words Data Structure
addWord("bad"), addWord("dad"), addWord("mad")

═══════════════════════════════════════════════════════════════
KEY INSIGHT: '.' wildcard = try ALL children at that level
             Use DFS to explore all possible paths
═══════════════════════════════════════════════════════════════

Trie Structure After Adding Words:
──────────────────────────────────
            root
           / | \
          b  d  m
          |  |  |
          a  a  a
          |  |  |
          d* d* d*

    * = word end marker

Search Examples:
────────────────
search("pad"):
  root → 'p' ? ✗ (no 'p' child)
  Return: False

search("bad"):
  root → 'b' ✓ → 'a' ✓ → 'd' ✓ → end marker? ✓
  Return: True

search(".ad"):
  root → '.' means try ALL children: {b, d, m}

  Path 1: 'b' → 'a' → 'd' → end marker? ✓ FOUND!
          Return True immediately

  (Paths 2 & 3 not explored - early termination)

search("b.."):
  root → 'b' ✓
       → '.' try all children of 'b': {a}
          → 'a' → '.' try all children of 'a': {d}
             → 'd' → end marker? ✓ FOUND!
  Return: True

search("..."):
  root → '.' try {b, d, m}
       → 'b' → '.' try {a}
            → 'a' → '.' try {d}
                 → 'd' → end marker? ✓ FOUND!
  Return: True

  (Worst case: explores ALL paths = 26^L)

DFS Branching Visualization:
────────────────────────────
  search(".ad")
       │
       ├─→ try 'b' → a → d → ✓ (return True)
       │
       ├─→ try 'd' → (not explored)
       │
       └─→ try 'm' → (not explored)

WHY THIS WORKS:
════════════════
● Trie provides O(L) lookup for exact matches
● '.' wildcard requires exploring all branches at that level
● DFS with early termination when match found
● Worst case O(26^L) when all wildcards, but rare in practice
```


### Intuition
```
addWord("bad")
addWord("dad")
addWord("mad")

search("pad") → false
search("bad") → true
search(".ad") → true (matches bad, dad, mad)
search("b..") → true (matches bad)

'.' requires exploring all children at that level.
```

### Solution
```python
class WordDictionary:
    """
    Trie with wildcard search using DFS.

    Time: O(L) for add, O(26^L) worst for search with wildcards
    Space: O(total chars)
    """

    def __init__(self):
        self.trie = {}

    def addWord(self, word: str) -> None:
        node = self.trie
        for c in word:
            node = node.setdefault(c, {})
        node["$"] = True

    def search(self, word: str) -> bool:
        def dfs(idx, node):
            if idx == len(word):
                return "$" in node

            char = word[idx]

            if char == ".":
                # Try all possible characters
                for key in node:
                    if key != "$" and dfs(idx + 1, node[key]):
                        return True
                return False
            else:
                if char not in node:
                    return False
                return dfs(idx + 1, node[char])

        return dfs(0, self.trie)
```

### Complexity
- **Add**: O(L) time
- **Search**: O(L) best, O(26^L) worst with all wildcards
- **Space**: O(total characters)

### Edge Cases
- All wildcards → explore all paths
- No wildcards → exact match
- Empty word → check root
- Word not added → return False

---

## Problem 5: Concatenated Words (LC #472) - Hard

- [LeetCode](https://leetcode.com/problems/concatenated-words/)

### Video Explanation
- [NeetCode - Concatenated Words](https://www.youtube.com/watch?v=iHp7fjw1R28)

### Problem Statement
Find all words that can be formed by concatenating other words in the list.

### Visual Intuition
```
Concatenated Words
words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]

Build trie of all words, then for each word:
Check if it can be formed by 2+ other words (DP + Trie)

"catsdogcats":
  cat|s... → "cats" in trie ✓
  cats|dog... → "dog" in trie ✓
  cats|dog|cats → "cats" in trie ✓
  3 words → concatenated!

"cat": only 1 word → not concatenated

Answer: ["catsdogcats","dogcatsdog","ratcatdogcat"]
```


### Intuition
```
words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]

"catsdogcats" = "cats" + "dog" + "cats" ✓
"dogcatsdog" = "dog" + "cats" + "dog" ✓
"ratcatdogcat" = "rat" + "cat" + "dog" + "cat" ✓

Build Trie, then DFS to check if word can be split.
```

### Solution
```python
def findAllConcatenatedWordsInADict(words: list[str]) -> list[str]:
    """
    Trie + DFS/DP to check concatenation.

    Strategy:
    - Build Trie with all words
    - For each word, check if it can be split into 2+ words
    - Use DFS with memoization

    Time: O(n * L²) where L = max word length
    Space: O(total chars)
    """
    # Build Trie
    trie = {}
    for word in words:
        if word:  # Skip empty strings
            node = trie
            for c in word:
                node = node.setdefault(c, {})
            node["$"] = True

    def can_form(word, start, count, memo):
        """Check if word[start:] can be formed by concatenating words."""
        if start == len(word):
            return count >= 2  # Need at least 2 words

        if start in memo:
            return memo[start]

        node = trie
        for i in range(start, len(word)):
            char = word[i]
            if char not in node:
                break

            node = node[char]

            if "$" in node:
                # Found a word ending here, try to form rest
                if can_form(word, i + 1, count + 1, memo):
                    memo[start] = True
                    return True

        memo[start] = False
        return False

    result = []
    for word in words:
        if word and can_form(word, 0, 0, {}):
            result.append(word)

    return result
```

### DP Alternative
```python
def findAllConcatenatedWordsInADict(words: list[str]) -> list[str]:
    """
    DP approach using word set.
    """
    word_set = set(words)
    result = []

    def can_form(word):
        if not word:
            return False

        n = len(word)
        dp = [False] * (n + 1)
        dp[0] = True

        for i in range(1, n + 1):
            for j in range(i):
                # Don't use the whole word as one piece
                if j == 0 and i == n:
                    continue

                if dp[j] and word[j:i] in word_set:
                    dp[i] = True
                    break

        return dp[n]

    for word in words:
        if can_form(word):
            result.append(word)

    return result
```

### Complexity
- **Time**: O(n * L²)
- **Space**: O(n * L)

### Edge Cases
- No concatenated words → return []
- Empty string in list → skip it
- Word = word + word → valid
- Single word can't concatenate → not included

---

## Problem 6: Search Suggestions System (LC #1268) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/search-suggestions-system/)

### Video Explanation
- [NeetCode - Search Suggestions System](https://www.youtube.com/watch?v=D4T2N0yAr20)

### Problem Statement
Return top 3 lexicographically smallest products matching each prefix.

### Visual Intuition
```
Search Suggestions System
products = ["mobile","mouse","moneypot","monitor","mousepad"]
searchWord = "mouse"

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Store top 3 sorted products at each Trie node
             As user types, traverse Trie and return stored list
═══════════════════════════════════════════════════════════════

Trie Structure with Top-3 Lists:
────────────────────────────────
  root
    │
    m → words: [mobile, moneypot, monitor]
    │
    o → words: [mobile, moneypot, monitor]
    │
    ├── b → words: [mobile]
    │   └── i → l → e* (mobile)
    │
    ├── n → words: [moneypot, monitor]
    │   ├── e → y → p → o → t* (moneypot)
    │   └── i → t → o → r* (monitor)
    │
    └── u → words: [mouse, mousepad]
        └── s → words: [mouse, mousepad]
            └── e* → words: [mouse, mousepad]
                └── p → a → d* (mousepad)

Step-by-Step as User Types:
───────────────────────────
Type 'm':
  Traverse: root → m
  Return: ["mobile", "moneypot", "monitor"]

Type 'mo':
  Traverse: root → m → o
  Return: ["mobile", "moneypot", "monitor"]

Type 'mou':
  Traverse: root → m → o → u
  Return: ["mouse", "mousepad"]

Type 'mous':
  Traverse: root → m → o → u → s
  Return: ["mouse", "mousepad"]

Type 'mouse':
  Traverse: root → m → o → u → s → e
  Return: ["mouse", "mousepad"]

Building the Trie:
──────────────────
  For each product, insert into Trie
  At each node, maintain sorted list of ≤3 products passing through

  insert("mobile"):
    m → o → b → i → l → e*
    Each node adds "mobile" to its list (if < 3 or lex smaller)

Final Result:
  [["mobile","moneypot","monitor"],
   ["mobile","moneypot","monitor"],
   ["mouse","mousepad"],
   ["mouse","mousepad"],
   ["mouse","mousepad"]]

WHY THIS WORKS:
════════════════
● Trie naturally groups words by prefix
● Pre-storing top 3 at each node → O(1) lookup per character
● Sorted insertion ensures lexicographic order
● Alternative: binary search on sorted array (also efficient)
```


### Intuition
```
products = ["mobile","mouse","moneypot","monitor","mousepad"]
searchWord = "mouse"

Typing "m" → ["mobile","moneypot","monitor"]
Typing "mo" → ["mobile","moneypot","monitor"]
Typing "mou" → ["mouse","mousepad"]
Typing "mous" → ["mouse","mousepad"]
Typing "mouse" → ["mouse","mousepad"]
```

### Solution
```python
def suggestedProducts(products: list[str], searchWord: str) -> list[list[str]]:
    """
    Trie with sorted children for lexicographic order.

    Time: O(n*L + m*3) where n=products, L=max length, m=searchWord length
    Space: O(n*L)
    """
    # Build Trie
    trie = {}

    for product in products:
        node = trie
        for c in product:
            node = node.setdefault(c, {"words": []})
            # Keep top 3 words at each node
            node["words"].append(product)
            node["words"].sort()
            if len(node["words"]) > 3:
                node["words"].pop()

    result = []
    node = trie

    for i, c in enumerate(searchWord):
        if node and c in node:
            node = node[c]
            result.append(node["words"])
        else:
            # No more matches
            node = None
            result.append([])

    return result
```

### Binary Search Alternative
```python
def suggestedProducts(products: list[str], searchWord: str) -> list[list[str]]:
    """
    Sort + binary search approach.

    Time: O(n log n + m * log n)
    Space: O(1) excluding output
    """
    import bisect

    products.sort()
    result = []
    prefix = ""

    for c in searchWord:
        prefix += c

        # Find insertion point
        idx = bisect.bisect_left(products, prefix)

        # Get up to 3 matches
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
- **Time**: O(n*L) for Trie, O(n log n) for binary search
- **Space**: O(n*L) for Trie, O(1) for binary search

### Edge Cases
- No matching products → empty lists
- Less than 3 matches → return available
- Exact match only → single element lists
- All products match → return top 3

---

## Summary

| # | Problem | Key Technique |
|---|---------|---------------|
| 1 | Word Search II | Trie + DFS backtracking |
| 2 | Palindrome Pairs | Hash map + palindrome check |
| 3 | Stream of Characters | Reversed Trie + suffix match |
| 4 | Add/Search Words | Trie + wildcard DFS |
| 5 | Concatenated Words | Trie + DP/DFS |
| 6 | Search Suggestions | Trie with sorted words |
