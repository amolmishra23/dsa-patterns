# Trie - Easy & Medium Problems

This section covers foundational Trie problems. Note: Trie problems are inherently more complex, so there are fewer "Easy" problems. The Implement Trie problem (LC #208) is the essential starting point.

---

## Problem 1: Longest Word in Dictionary (LC #720) - Easy

- [LeetCode](https://leetcode.com/problems/longest-word-in-dictionary/)

### Problem Statement
Find the longest word in the dictionary that can be built one character at a time by other words in the dictionary.

### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Examples
```
Input: words = ["w","wo","wor","worl","world"]
Output: "world"
Explanation: "world" can be built: w → wo → wor → worl → world

Input: words = ["a","banana","app","appl","ap","apply","apple"]
Output: "apple"
Explanation: Both "apply" and "apple" can be built, but "apple" is lexicographically smaller.
```

### Intuition
```
Key insight: A word is "buildable" only if ALL its prefixes exist in the dictionary.

For "world" to be valid, we need: "w", "wo", "wor", "worl", "world"

Approach:
1. Sort words by length (shorter first), then lexicographically
2. A word is buildable if word[:-1] (its prefix) was already built
3. Track the longest buildable word

Why sorting works: Shorter words are processed first, so when we check
a longer word, all potential prefixes have already been seen.
```

### Solution
```python
def longestWord(words: list[str]) -> str:
    """
    Find longest word buildable one character at a time.

    Time: O(n * L * log n) for sorting, where L = avg word length
    Space: O(n * L) for the set
    """
    # Sort: shorter words first, then lexicographically
    words.sort(key=lambda x: (len(x), x))

    # Set of buildable words (empty string is base case)
    buildable = set([''])
    result = ''

    for word in words:
        # Check if prefix exists (word is buildable)
        prefix = word[:-1]

        if prefix in buildable:
            buildable.add(word)

            # Update result if this is longer
            if len(word) > len(result):
                result = word

    return result
```

### Complexity
- **Time**: O(n * L * log n) - sorting dominates
- **Space**: O(n * L) - storing all buildable words

### Edge Cases
- Empty words list → return ""
- Single character words only → return lexicographically smallest
- No buildable word → return ""
- Tie in length → return lexicographically smallest

---

## Problem 2: Index Pairs of a String (LC #1065) - Easy

- [LeetCode](https://leetcode.com/problems/index-pairs-of-a-string/)

### Problem Statement
Given a text string and a list of words, find all index pairs [i, j] where text[i:j+1] is a word in the list.

### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Examples
```
Input: text = "thestoryofleetcodeandme", words = ["story","fleet","leetcode"]
Output: [[3,7],[9,13],[10,17]]
Explanation:
- "story" at indices [3,7]
- "fleet" at indices [9,13]
- "leetcode" at indices [10,17]
```

### Intuition
```
Key insight: Use Trie to efficiently check all substrings starting at each position.

Without Trie: For each starting position, check if each word matches → O(n * m * L)
With Trie: For each starting position, traverse Trie until no match → O(n * L)

Process:
1. Build Trie from all words
2. For each starting index i in text:
   - Traverse Trie character by character
   - When hitting a word end, record [i, current_j]
   - Stop when character not in Trie
```

### Solution
```python
def indexPairs(text: str, words: list[str]) -> list[list[int]]:
    """
    Find all index pairs where substring is a word.

    Time: O(n * L + m * L) where n = text length, m = num words, L = avg word length
    Space: O(m * L) for Trie
    """
    # Build Trie from words
    root = {}
    for word in words:
        node = root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['$'] = True  # Mark word end

    result = []

    # Check each starting position
    for i in range(len(text)):
        node = root

        for j in range(i, len(text)):
            char = text[j]

            if char not in node:
                break

            node = node[char]

            # Found a word ending here
            if '$' in node:
                result.append([i, j])

    return result
```

### Complexity
- **Time**: O(n * L + m * L) - traverse text with Trie lookups
- **Space**: O(m * L) - Trie storage

### Edge Cases
- Empty text → return []
- Empty words list → return []
- No matches → return []
- Overlapping words → return all matches
- Same word appears multiple times in text → return all occurrences

---

## Problem 3: Implement Trie (LC #208) - Medium

- [LeetCode](https://leetcode.com/problems/implement-trie-prefix-tree/)

### Problem Statement
Implement a trie with `insert`, `search`, and `startsWith` methods.

### Video Explanation
- [NeetCode - Implement Trie](https://www.youtube.com/watch?v=oobqoCJlHA0)

### Intuition
```
A Trie is a tree where each path from root represents a word/prefix!

Visual: Insert "apple", "app", "bat"

                root
               /    \
              a      b
              |      |
              p      a
              |      |
              p*     t*
              |
              l
              |
              e*

        * = end of word marker

        search("app") → traverse a→p→p, check is_end=True ✓
        search("ap") → traverse a→p, is_end=False ✗
        startsWith("ap") → traverse a→p, found! ✓

Operations:
- insert: Create nodes for each char, mark last as word end
- search: Traverse all chars, check is_end at last node
- startsWith: Just traverse all chars (don't check is_end)
```

### Solution
```python
class TrieNode:
    """
    Node in a Trie data structure.

    Each node contains:
    - children: dictionary mapping character to child node
    - is_end: boolean indicating if this node marks end of a word
    """
    def __init__(self):
        self.children = {}   # char -> TrieNode
        self.is_end = False  # True if word ends here


class Trie:
    """
    Trie (Prefix Tree) implementation.

    Provides efficient prefix-based operations:
    - insert(word): Add word to trie - O(L)
    - search(word): Check if exact word exists - O(L)
    - startsWith(prefix): Check if any word has prefix - O(L)

    where L = length of word/prefix
    """

    def __init__(self):
        """Initialize trie with empty root node."""
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Insert word into trie.

        Traverse/create nodes for each character, mark last as word end.

        Time: O(L) where L = len(word)
        Space: O(L) for new nodes
        """
        node = self.root

        for char in word:
            # Create child node if it doesn't exist
            if char not in node.children:
                node.children[char] = TrieNode()

            # Move to child node
            node = node.children[char]

        # Mark end of word
        node.is_end = True

    def search(self, word: str) -> bool:
        """
        Check if word exists in trie.

        Must find all characters AND end at a word-end node.

        Time: O(L)
        """
        node = self._traverse(word)
        # Word exists only if we found all chars AND it's marked as end
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        """
        Check if any word in trie starts with prefix.

        Just need to find all characters, don't need word-end.

        Time: O(L)
        """
        return self._traverse(prefix) is not None

    def _traverse(self, s: str) -> TrieNode:
        """
        Helper: Traverse trie following characters in s.

        Returns: Node at end of path, or None if path doesn't exist
        """
        node = self.root

        for char in s:
            if char not in node.children:
                return None
            node = node.children[char]

        return node
```

### Edge Cases
- Empty string → insert creates word at root
- Single character → simple path of length 1
- Prefix of existing word → startsWith true, search false
- Word is prefix of another → both searchable
- Same word inserted twice → no change (idempotent)

---

## Problem 2: Design Add and Search Words (LC #211) - Medium

- [LeetCode](https://leetcode.com/problems/design-add-and-search-words-data-structure/)

### Problem Statement
Design a data structure that supports adding words and searching with `.` wildcard.

### Examples
```
addWord("bad")
addWord("dad")
addWord("mad")
search("pad") → false
search("bad") → true
search(".ad") → true
search("b..") → true
```

### Video Explanation
- [NeetCode - Design Add and Search Words](https://www.youtube.com/watch?v=BTf05gs_8iU)

### Intuition
```
Trie + DFS for wildcard handling!

'.' can match ANY character, so we try all children.

Visual: Words = ["bad", "dad", "mad"]

                root
               /  |  \
              b   d   m
              |   |   |
              a   a   a
              |   |   |
              d*  d*  d*

search(".ad"):
- At root, '.' matches any → try b, d, m
- At b: 'a' matches 'a' → continue
- At a: 'd' matches 'd', is_end=True → found!

search("b.."):
- At root, 'b' matches 'b' → continue
- At b: '.' matches any → try 'a'
- At a: '.' matches any → try 'd', is_end=True → found!
```

### Solution
```python
class WordDictionary:
    """
    Dictionary supporting wildcard search.

    '.' matches any single character.
    Uses Trie with DFS for wildcard handling.

    Time:
    - addWord: O(L)
    - search: O(L) without wildcards, O(26^w * L) with w wildcards
    Space: O(total characters)
    """

    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        """Add word to dictionary."""
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end = True

    def search(self, word: str) -> bool:
        """
        Search for word with '.' wildcard support.

        Uses DFS to handle wildcards - try all children.
        """
        def dfs(node: TrieNode, index: int) -> bool:
            """
            DFS search starting from node at index in word.

            Args:
                node: Current trie node
                index: Current position in search word

            Returns: True if match found
            """
            # Base case: reached end of word
            if index == len(word):
                return node.is_end

            char = word[index]

            if char == '.':
                # Wildcard: try all possible children
                for child in node.children.values():
                    if dfs(child, index + 1):
                        return True
                return False
            else:
                # Regular character: follow specific path
                if char not in node.children:
                    return False
                return dfs(node.children[char], index + 1)

        return dfs(self.root, 0)
```

### Edge Cases
- All wildcards "..." → matches any word of that length
- No wildcards → same as regular trie search
- Wildcard at start → try all first characters
- Empty pattern → return True (matches empty word if exists)
- Pattern longer than any word → return False

---

## Problem 3: Word Search II (LC #212) - Hard

- [LeetCode](https://leetcode.com/problems/word-search-ii/)

### Problem Statement
Find all words from dictionary that exist in a 2D board.

### Examples
```
Input: board = [["o","a","a","n"],
                ["e","t","a","e"],
                ["i","h","k","r"],
                ["i","f","l","v"]]
       words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
```

### Video Explanation
- [NeetCode - Word Search II](https://www.youtube.com/watch?v=asbcE9mZz_U)

### Intuition
```
Trie + DFS on board = efficient word search!

Why Trie? Without it, we'd search each word separately O(words * m*n*4^L).
With Trie, we search all words simultaneously!

Visual:
        board:         Trie for ["oath","eat"]:
        o a a n              root
        e t a e             /    \
        i h k r            o      e
        i f l v            |      |
                           a      a
                           |      |
                           t      t*
                           |
                           h*

DFS from 'o' at (0,0):
- o → matches trie root→o
- o→a → matches o→a
- o→a→t → matches a→t
- o→a→t→h → matches t→h, is_end=True → found "oath"!

Key optimization: Prune search when prefix not in trie!
```

### Solution
```python
def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    """
    Find all dictionary words in board using Trie + DFS.

    Strategy:
    1. Build trie from all words
    2. DFS from each cell, following trie paths
    3. Prune search when prefix not in trie

    Time: O(m * n * 4^L) where L = max word length
    Space: O(total chars in words)
    """
    # Build trie from words
    root = TrieNode()
    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.word = word  # Store word at end node for easy retrieval

    rows, cols = len(board), len(board[0])
    result = []

    def dfs(r: int, c: int, node: TrieNode):
        """
        DFS from cell (r, c) following trie node.

        Explores all 4 directions, backtracks when done.
        """
        # Check bounds
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return

        char = board[r][c]

        # Check if character exists in trie
        if char not in node.children:
            return

        # Move to next trie node
        next_node = node.children[char]

        # Check if we found a word
        if next_node.is_end:
            result.append(next_node.word)
            next_node.is_end = False  # Avoid duplicates

        # Mark cell as visited
        board[r][c] = '#'

        # Explore all 4 directions
        dfs(r + 1, c, next_node)  # Down
        dfs(r - 1, c, next_node)  # Up
        dfs(r, c + 1, next_node)  # Right
        dfs(r, c - 1, next_node)  # Left

        # Restore cell (backtrack)
        board[r][c] = char

    # Start DFS from each cell
    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)

    return result
```

### Edge Cases
- Empty board → return []
- Empty words list → return []
- Single cell board → check if any word is single char
- Word longer than board allows → can't find it
- Same word found multiple times → return once

---

## Problem 4: Replace Words (LC #648) - Medium

- [LeetCode](https://leetcode.com/problems/replace-words/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Replace words in sentence with their shortest root from dictionary.

### Examples
```
Input: dictionary = ["cat","bat","rat"]
       sentence = "the cattle was rattled by the battery"
Output: "the cat was rat by the bat"
```


### Intuition
```
Key insight: Use trie to find shortest matching prefix efficiently.

Build trie from dictionary roots, then for each word in sentence:
- Traverse trie character by character
- Stop at first node marked as word end (shortest root)
- If found, replace word with that prefix

Why trie works: Finding shortest prefix among many candidates
is O(L) per word instead of O(d * L) with brute force.
```

### Solution
```python
def replaceWords(dictionary: list[str], sentence: str) -> str:
    """
    Replace words with shortest root from dictionary.

    Strategy:
    1. Build trie from dictionary roots
    2. For each word, find shortest prefix in trie
    3. Replace word with root if found

    Time: O(d * L + s * L) where d = dict size, s = sentence words, L = avg length
    Space: O(d * L)
    """
    # Build trie from dictionary
    root = TrieNode()

    for word in dictionary:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def find_root(word: str) -> str:
        """
        Find shortest root for word in trie.

        Returns root if found, otherwise original word.
        """
        node = root
        prefix = []

        for char in word:
            # No path in trie
            if char not in node.children:
                break

            node = node.children[char]
            prefix.append(char)

            # Found a root (shortest prefix that's a word)
            if node.is_end:
                return ''.join(prefix)

        # No root found, return original word
        return word

    # Process each word in sentence
    words = sentence.split()
    replaced = [find_root(word) for word in words]

    return ' '.join(replaced)
```

### Edge Cases
- Empty dictionary → return sentence unchanged
- No word has root → return sentence unchanged
- Multiple roots for same word → use shortest
- Root equals word → replace with itself
- Empty sentence → return empty string

---

## Problem 5: Longest Word in Dictionary (LC #720) - Medium

- [LeetCode](https://leetcode.com/problems/longest-word-in-dictionary/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Find longest word that can be built one character at a time.

### Examples
```
Input: words = ["w","wo","wor","worl","world"]
Output: "world"

Input: words = ["a","banana","app","appl","ap","apply","apple"]
Output: "apple"
```


### Intuition
```
Key insight: A word is "buildable" only if all its prefixes exist.

For "world" to be valid, we need: "w", "wo", "wor", "worl", "world"

Two approaches:
1. Sort by length, track buildable words in set. Word is buildable
   if word[:-1] is in set.
2. Build trie, BFS/DFS only through nodes marked as word endings.

Sorting approach: Process shorter words first, so prefixes are
always seen before the words that need them.
```

### Solution
```python
def longestWord(words: list[str]) -> str:
    """
    Find longest word buildable one character at a time.

    Strategy:
    - Sort words by length, then lexicographically
    - A word is buildable if its prefix (word[:-1]) was already built
    - Track longest buildable word

    Time: O(n * L * log n) for sorting
    Space: O(n * L)
    """
    # Sort: shorter words first, then lexicographically
    words.sort(key=lambda x: (len(x), x))

    # Set of buildable words (empty string is base case)
    buildable = set([''])
    result = ''

    for word in words:
        # Check if prefix exists (word is buildable)
        prefix = word[:-1]

        if prefix in buildable:
            buildable.add(word)

            # Update result if this is longer
            if len(word) > len(result):
                result = word

    return result


def longestWord_trie(words: list[str]) -> str:
    """
    Alternative: Use trie with BFS.

    Build trie, then BFS to find longest word where
    every prefix is also a word.
    """
    # Build trie
    root = TrieNode()
    root.is_end = True  # Empty string is valid

    for word in words:
        node = root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.word = word

    # BFS to find longest word
    from collections import deque

    queue = deque([root])
    result = ''

    while queue:
        node = queue.popleft()

        # Only explore children that are word endings
        for char in sorted(node.children.keys()):  # Sorted for lexicographic order
            child = node.children[char]
            if child.is_end:
                queue.append(child)
                if len(child.word) > len(result):
                    result = child.word

    return result
```

### Edge Cases
- Empty words list → return ""
- Single character words → return lexicographically smallest
- No buildable word → return ""
- All words buildable → return longest (or lex smallest if tie)
- Words with common prefixes → check all paths

---

## Problem 6: Map Sum Pairs (LC #677) - Medium

- [LeetCode](https://leetcode.com/problems/map-sum-pairs/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Implement a map where you can insert key-value pairs and get sum of values with given prefix.

### Examples
```
insert("apple", 3)
sum("ap") → 3
insert("app", 2)
sum("ap") → 5
insert("apple", 2)  # Update
sum("ap") → 4
```


### Intuition
```
Key insight: Store running prefix sums at each trie node.

Each node tracks sum of ALL values passing through it.
- insert("apple", 3): nodes a→p→p→l→e each get +3
- sum("ap"): return sum stored at node 'p' after 'a'

For updates: calculate delta = new_val - old_val, then
add delta along the path. This handles overwrites correctly.

Why this works: Prefix sum at node = sum of all words
sharing that prefix, exactly what we need.
```

### Solution
```python
class MapSum:
    """
    Map with prefix sum functionality.

    Strategy:
    - Trie where each node stores sum of all words passing through it
    - When updating, subtract old value and add new value along the path

    Time: O(L) for insert and sum
    Space: O(total characters)
    """

    def __init__(self):
        self.root = {}
        self.map = {}  # key -> value (for updates)

    def insert(self, key: str, val: int) -> None:
        """
        Insert key-value pair. Update if key exists.
        """
        # Calculate delta (new value - old value)
        delta = val - self.map.get(key, 0)
        self.map[key] = val

        # Update sums along the path
        node = self.root
        for char in key:
            if char not in node:
                node[char] = {'_sum': 0}
            node = node[char]
            node['_sum'] += delta

    def sum(self, prefix: str) -> int:
        """
        Return sum of all values with given prefix.
        """
        node = self.root

        for char in prefix:
            if char not in node:
                return 0
            node = node[char]

        return node['_sum']
```

### Edge Cases
- Empty prefix → return sum of all values
- Prefix not in trie → return 0
- Update existing key → subtract old, add new
- Single key → sum equals its value
- All keys share prefix → sum includes all

---

## Problem 7: Design Search Autocomplete System (LC #642) - Hard

- [LeetCode](https://leetcode.com/problems/design-search-autocomplete-system/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Design autocomplete system that returns top 3 suggestions based on history.


### Intuition
```
Key insight: Trie + frequency tracking for efficient prefix matching.

Design decisions:
1. Trie stores all sentences for O(L) prefix lookup
2. HashMap tracks sentence → frequency for ranking
3. Track current node as user types (avoid re-traversing)

On each character input:
- Navigate to next trie node
- Collect all sentences in subtree
- Return top 3 by frequency (then lexicographic)

On '#': Save current input as new sentence, reset state.
```

### Solution
```python
import heapq
from collections import defaultdict

class AutocompleteSystem:
    """
    Autocomplete system with history-based suggestions.

    Strategy:
    - Trie stores sentences with their frequencies
    - Track current input prefix
    - Return top 3 by frequency (then lexicographic)

    Time: O(L) per character input
    Space: O(total characters in all sentences)
    """

    def __init__(self, sentences: list[str], times: list[int]):
        """
        Initialize with historical sentences and their frequencies.
        """
        self.root = {}
        self.freq = defaultdict(int)  # sentence -> frequency
        self.current_node = self.root
        self.current_input = []

        # Add historical data
        for sentence, count in zip(sentences, times):
            self.freq[sentence] = count
            self._add_to_trie(sentence)

    def _add_to_trie(self, sentence: str):
        """Add sentence to trie."""
        node = self.root
        for char in sentence:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = sentence  # Mark end with sentence

    def input(self, c: str) -> list[str]:
        """
        Process input character and return suggestions.

        '#' marks end of input - save sentence and reset.
        """
        if c == '#':
            # End of input - save sentence
            sentence = ''.join(self.current_input)
            self.freq[sentence] += 1
            self._add_to_trie(sentence)

            # Reset state
            self.current_node = self.root
            self.current_input = []
            return []

        # Add character to current input
        self.current_input.append(c)

        # Navigate trie
        if self.current_node is None or c not in self.current_node:
            self.current_node = None
            return []

        self.current_node = self.current_node[c]

        # Find all sentences with current prefix
        sentences = []
        self._collect_sentences(self.current_node, sentences)

        # Sort by frequency (desc), then lexicographically
        sentences.sort(key=lambda x: (-self.freq[x], x))

        return sentences[:3]

    def _collect_sentences(self, node: dict, sentences: list):
        """Collect all sentences in subtree."""
        if node is None:
            return

        if '#' in node:
            sentences.append(node['#'])

        for char, child in node.items():
            if char != '#':
                self._collect_sentences(child, sentences)
```

### Edge Cases
- Empty input history → start fresh
- First character typed → show top 3 from all
- No matches for prefix → return []
- '#' as first input → save empty sentence
- Same sentence entered multiple times → increase frequency

---

## Summary: Trie Problems

| # | Problem | Difficulty | Key Technique | Time |
|---|---------|------------|---------------|------|
| 1 | Longest Word in Dictionary | Easy | Sort + buildable set | O(n log n) |
| 2 | Index Pairs of a String | Easy | Substring search | O(n * L) |
| 3 | Implement Trie | Medium | Basic trie ops | O(L) |
| 4 | Add/Search Words | Medium | DFS for wildcards | O(26^w * L) |
| 5 | Replace Words | Medium | Prefix lookup | O(n * L) |
| 6 | Map Sum Pairs | Medium | Trie with sums | O(L) |
| 7 | Word Search II | Hard | Trie + board DFS | O(mn * 4^L) |
| 8 | Autocomplete | Hard | Trie + frequency | O(L) |

---

## Practice More Problems

- [ ] LC #1268 - Search Suggestions System (Medium)
- [ ] LC #745 - Prefix and Suffix Search (Hard)
- [ ] LC #472 - Concatenated Words (Hard)
- [ ] LC #336 - Palindrome Pairs (Hard)

