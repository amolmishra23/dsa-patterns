# Trie (Prefix Tree) - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE TRIE                                         │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Prefix matching"                                                        │
│  ✓ "Autocomplete"                                                           │
│  ✓ "Word search"                                                            │
│  ✓ "Dictionary implementation"                                              │
│  ✓ "Longest common prefix"                                                  │
│  ✓ "Word break"                                                             │
│                                                                             │
│  Key insight: Trie provides O(L) operations where L = word length           │
│               Much faster than O(n) hash table for prefix operations        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Tree data structure
- [ ] Hash map basics
- [ ] String operations

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRIE MEMORY MAP                                          │
│                                                                             │
│                    ┌─────────────┐                                          │
│         ┌─────────│    TRIE     │─────────┐                                 │
│         │         └─────────────┘         │                                 │
│         ▼                                 ▼                                 │
│  ┌─────────────┐                   ┌─────────────┐                          │
│  │   SEARCH    │                   │   PREFIX    │                          │
│  │  PROBLEMS   │                   │  PROBLEMS   │                          │
│  └──────┬──────┘                   └──────┬──────┘                          │
│         │                                 │                                 │
│    ┌────┴────┐                      ┌─────┴─────┐                           │
│    ▼         ▼                      ▼           ▼                           │
│ ┌──────┐ ┌──────┐               ┌──────┐   ┌──────┐                        │
│ │Word  │ │Add & │               │Auto- │   │Longest│                        │
│ │Search│ │Search│               │complete│  │Prefix │                        │
│ └──────┘ └──────┘               └──────┘   └──────┘                        │
│                                                                             │
│  Related Patterns:                                                          │
│  • Hash Map - Alternative for exact match (not prefix)                      │
│  • DFS - Trie traversal uses DFS                                            │
│  • Backtracking - Word search combines trie + backtracking                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRIE vs HASH SET DECISION TREE                           │
│                                                                             │
│  Need prefix operations?                                                    │
│       │                                                                     │
│       ├── YES → Use Trie                                                    │
│       │         startsWith, autocomplete, prefix count                      │
│       │                                                                     │
│       └── NO → Only exact match needed?                                     │
│                    │                                                        │
│                    ├── YES → Hash Set is simpler                            │
│                    │                                                        │
│                    └── Need ordered iteration?                              │
│                                 │                                           │
│                                 ├── YES → Trie gives lexicographic order    │
│                                 │                                           │
│                                 └── NO → Hash Set                           │
│                                                                             │
│  COMPLEXITY COMPARISON:                                                     │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ Operation      │ Trie    │ Hash Set                                │     │
│  ├────────────────┼─────────┼─────────────────────────────────────────┤     │
│  │ Insert         │ O(L)    │ O(L)                                    │     │
│  │ Search         │ O(L)    │ O(L)                                    │     │
│  │ StartsWith     │ O(P)    │ O(n*L) - must check all words!          │     │
│  │ Space          │ O(N*L)  │ O(N*L)                                  │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concept

A Trie is a tree-like data structure for storing strings where each node represents a character.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRIE VISUALIZATION                                       │
│                                                                             │
│  Words: ["apple", "app", "application", "bat", "ball"]                      │
│                                                                             │
│                        root                                                 │
│                       /    \                                                │
│                      a      b                                               │
│                      |      |                                               │
│                      p      a                                               │
│                      |     / \                                              │
│                      p    t*  l                                             │
│                     /|        |                                             │
│                   l* i        l*                                            │
│                   |  |                                                      │
│                   e* c                                                      │
│                      |                                                      │
│                      a                                                      │
│                      |                                                      │
│                      t                                                      │
│                      |                                                      │
│                      i                                                      │
│                      |                                                      │
│                      o                                                      │
│                      |                                                      │
│                      n*                                                     │
│                                                                             │
│  * = end of word                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Trie Implementation

```python
class TrieNode:
    """
    Node in a Trie.

    Each node has:
    - children: dictionary mapping character to child node
    - is_end: whether this node marks end of a word
    """
    def __init__(self):
        self.children = {}  # char -> TrieNode
        self.is_end = False  # True if word ends here


class Trie:
    """
    Trie (Prefix Tree) implementation.

    Operations:
    - insert(word): Add word to trie - O(L)
    - search(word): Check if word exists - O(L)
    - startsWith(prefix): Check if any word has prefix - O(L)

    where L = length of word/prefix

    Space: O(total characters across all words)
    """

    def __init__(self):
        """Initialize with empty root node."""
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Insert word into trie.

        Strategy:
        - Start from root
        - For each character, create child node if doesn't exist
        - Mark last node as end of word

        Time: O(L) where L = len(word)
        Space: O(L) for new nodes
        """
        node = self.root

        for char in word:
            # Create child node if doesn't exist
            if char not in node.children:
                node.children[char] = TrieNode()

            # Move to child node
            node = node.children[char]

        # Mark end of word
        node.is_end = True

    def search(self, word: str) -> bool:
        """
        Check if word exists in trie.

        Strategy:
        - Traverse trie following characters
        - Return True only if we reach a node marked as end

        Time: O(L)
        """
        node = self._find_node(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        """
        Check if any word in trie starts with prefix.

        Strategy:
        - Traverse trie following prefix characters
        - Return True if we can traverse entire prefix

        Time: O(L)
        """
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> TrieNode:
        """
        Helper: Find node corresponding to prefix.

        Returns: Node at end of prefix, or None if prefix not found
        """
        node = self.root

        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]

        return node
```

---

## Trie with Array Children

More memory efficient for lowercase letters only:

```python
class TrieNodeArray:
    """
    Trie node using array for children (for lowercase letters).

    More memory efficient when most nodes have many children.
    """
    def __init__(self):
        self.children = [None] * 26  # a-z
        self.is_end = False

    def get_child(self, char: str) -> 'TrieNodeArray':
        return self.children[ord(char) - ord('a')]

    def set_child(self, char: str, node: 'TrieNodeArray') -> None:
        self.children[ord(char) - ord('a')] = node


class TrieArray:
    """Trie using array-based nodes."""

    def __init__(self):
        self.root = TrieNodeArray()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            idx = ord(char) - ord('a')
            if node.children[idx] is None:
                node.children[idx] = TrieNodeArray()
            node = node.children[idx]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            idx = ord(char) - ord('a')
            if node.children[idx] is None:
                return False
            node = node.children[idx]
        return node.is_end

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            idx = ord(char) - ord('a')
            if node.children[idx] is None:
                return False
            node = node.children[idx]
        return True
```

---

## Common Trie Problems

### Problem 1: Implement Trie (LC #208)

```python
# See implementation above - that's the solution!
```

### Problem 2: Word Search II (LC #212)

```python
def findWords(board: list[list[str]], words: list[str]) -> list[str]:
    """
    Find all words from dictionary that exist in board.

    Strategy:
    - Build trie from words
    - DFS from each cell, following trie
    - Prune search when prefix not in trie

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
        node.word = word  # Store word at end node

    rows, cols = len(board), len(board[0])
    result = []

    def dfs(r: int, c: int, node: TrieNode):
        """DFS from cell (r,c) following trie node."""
        # Check bounds and if character exists in trie
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] not in node.children):
            return

        char = board[r][c]
        next_node = node.children[char]

        # Found a word
        if next_node.is_end:
            result.append(next_node.word)
            next_node.is_end = False  # Avoid duplicates

        # Mark as visited
        board[r][c] = '#'

        # Explore 4 directions
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            dfs(r + dr, c + dc, next_node)

        # Restore cell
        board[r][c] = char

    # Start DFS from each cell
    for r in range(rows):
        for c in range(cols):
            dfs(r, c, root)

    return result
```

### Problem 3: Design Add and Search Words (LC #211)

```python
class WordDictionary:
    """
    Add words and search with '.' wildcard.

    '.' matches any single character.
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
        Search for word. '.' matches any character.

        Use DFS to handle wildcards.
        """
        def dfs(node: TrieNode, i: int) -> bool:
            """Search from node starting at index i in word."""
            if i == len(word):
                return node.is_end

            char = word[i]

            if char == '.':
                # Wildcard: try all children
                for child in node.children.values():
                    if dfs(child, i + 1):
                        return True
                return False
            else:
                # Regular character
                if char not in node.children:
                    return False
                return dfs(node.children[char], i + 1)

        return dfs(self.root, 0)
```

### Problem 4: Longest Word in Dictionary (LC #720)

```python
def longestWord(words: list[str]) -> str:
    """
    Find longest word that can be built one character at a time.

    Strategy:
    - Sort words by length, then lexicographically
    - Build trie, only add word if prefix (word[:-1]) exists
    - Track longest valid word
    """
    words.sort(key=lambda x: (len(x), x))

    valid = set([''])  # Empty string is valid base
    result = ''

    for word in words:
        # Check if prefix exists (can be built)
        if word[:-1] in valid:
            valid.add(word)

            # Update result if longer
            if len(word) > len(result):
                result = word

    return result
```

### Problem 5: Replace Words (LC #648)

```python
def replaceWords(dictionary: list[str], sentence: str) -> str:
    """
    Replace words with their shortest root from dictionary.

    Strategy:
    - Build trie from dictionary
    - For each word, find shortest prefix in trie
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
        """Find shortest root for word, or return word itself."""
        node = root
        prefix = []

        for char in word:
            if char not in node.children:
                break

            node = node.children[char]
            prefix.append(char)

            if node.is_end:
                # Found a root
                return ''.join(prefix)

        # No root found
        return word

    words = sentence.split()
    return ' '.join(find_root(word) for word in words)
```

---

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Insert | O(L) | O(L) |
| Search | O(L) | O(1) |
| StartsWith | O(L) | O(1) |
| Delete | O(L) | O(1) |

Where L = length of word/prefix

Total space: O(N × L) where N = number of words

---

## Trie vs Hash Set

| Operation | Trie | Hash Set |
|-----------|------|----------|
| Insert | O(L) | O(L) |
| Search exact | O(L) | O(L) |
| Prefix search | O(L) | O(N × L) |
| Autocomplete | O(L + results) | O(N × L) |
| Space | O(N × L) | O(N × L) |

**Use Trie when**: Prefix operations are common
**Use Hash Set when**: Only exact match needed

---

## Common Mistakes

```python
# ❌ WRONG: Not marking end of word
def insert_wrong(self, word):
    node = self.root
    for char in word:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    # Forgot to set node.is_end = True!

# ✅ CORRECT: Always mark end
def insert_correct(self, word):
    node = self.root
    for char in word:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    node.is_end = True  # Mark end of word


# ❌ WRONG: Confusing search and startsWith
def search_wrong(self, word):
    node = self._find_node(word)
    return node is not None  # This is startsWith, not search!

# ✅ CORRECT: Check is_end for exact match
def search_correct(self, word):
    node = self._find_node(word)
    return node is not None and node.is_end
```

---

## Interview Tips

### 1. How to Explain Your Approach
```
"I'll use a Trie because we need efficient prefix operations. Each node
represents a character, and paths from root represent words. Insert and
search are O(L) where L is word length."
```

### 2. What Interviewers Look For
- **Implementation**: Can you implement Trie from scratch?
- **is_end flag**: Remember to mark word endings
- **Space trade-off**: Trie uses more space than hash set

### 3. Common Follow-up Questions
- "Can you use hash set instead?" → Only for exact match, not prefix
- "How to delete a word?" → Mark is_end = False, optionally prune
- "What about wildcard search?" → DFS with backtracking

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
