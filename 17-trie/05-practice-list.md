# Trie - Complete Practice List

## Organized by Difficulty

### Easy Problems

| # | Problem | Key Technique |
|---|---------|---------------|
| 720 | [Longest Word in Dictionary](https://leetcode.com/problems/longest-word-in-dictionary/) | Build step by step |
| 1065 | [Index Pairs of a String](https://leetcode.com/problems/index-pairs-of-a-string/) | Search all substrings |

### Medium Problems

| # | Problem | Key Technique |
|---|---------|---------------|
| 208 | [Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/) | Basic implementation |
| 211 | [Design Add and Search Words](https://leetcode.com/problems/design-add-and-search-words-data-structure/) | Wildcard with DFS |
| 648 | [Replace Words](https://leetcode.com/problems/replace-words/) | Find shortest prefix |
| 677 | [Map Sum Pairs](https://leetcode.com/problems/map-sum-pairs/) | Sum with prefix |
| 1268 | [Search Suggestions System](https://leetcode.com/problems/search-suggestions-system/) | Autocomplete |
| 1233 | [Remove Sub-Folders](https://leetcode.com/problems/remove-sub-folders-from-the-filesystem/) | Path trie |
| 421 | [Maximum XOR of Two Numbers](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/) | Bitwise trie |

### Hard Problems

| # | Problem | Key Technique |
|---|---------|---------------|
| 212 | [Word Search II](https://leetcode.com/problems/word-search-ii/) | Trie + DFS on grid |
| 336 | [Palindrome Pairs](https://leetcode.com/problems/palindrome-pairs/) | Reversed trie |
| 472 | [Concatenated Words](https://leetcode.com/problems/concatenated-words/) | Trie + DFS |
| 588 | [Design In-Memory File System](https://leetcode.com/problems/design-in-memory-file-system/) | Path trie |
| 642 | [Design Search Autocomplete](https://leetcode.com/problems/design-search-autocomplete-system/) | Trie + ranking |
| 745 | [Prefix and Suffix Search](https://leetcode.com/problems/prefix-and-suffix-search/) | suffix#prefix trick |
| 1032 | [Stream of Characters](https://leetcode.com/problems/stream-of-characters/) | Reversed trie |

---

## Essential Templates

### 1. Basic Trie Implementation
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """Insert word into trie. O(L)"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word: str) -> bool:
        """Check if word exists. O(L)"""
        node = self._find_node(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        """Check if any word starts with prefix. O(L)"""
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> TrieNode:
        """Navigate to node for prefix."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

### 2. Trie with Wildcard Search
```python
class WordDictionary:
    """Supports '.' as wildcard."""

    def __init__(self):
        self.root = {}

    def addWord(self, word: str) -> None:
        node = self.root
        for char in word:
            node = node.setdefault(char, {})
        node['$'] = True

    def search(self, word: str) -> bool:
        def dfs(node, i):
            if i == len(word):
                return '$' in node

            char = word[i]

            if char == '.':
                # Try all children
                for child in node:
                    if child != '$' and dfs(node[child], i + 1):
                        return True
                return False
            else:
                if char not in node:
                    return False
                return dfs(node[char], i + 1)

        return dfs(self.root, 0)
```

### 3. Bitwise Trie (for XOR problems)
```python
class BitwiseTrie:
    """Trie for binary representations."""

    def __init__(self):
        self.root = {}

    def insert(self, num: int) -> None:
        node = self.root
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]

    def find_max_xor(self, num: int) -> int:
        """Find number in trie that maximizes XOR with num."""
        node = self.root
        xor_result = 0

        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            opposite = 1 - bit

            if opposite in node:
                xor_result |= (1 << i)
                node = node[opposite]
            else:
                node = node[bit]

        return xor_result
```

### 4. Autocomplete Trie
```python
class AutocompleteTrie:
    """Trie with top-k suggestions."""

    def __init__(self):
        self.root = {}

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            node = node.setdefault(char, {'suggestions': []})
            # Keep top 3 suggestions at each node
            if len(node['suggestions']) < 3:
                node['suggestions'].append(word)
            node = node.setdefault(char, {})

    def search(self, prefix: str) -> list:
        node = self.root
        for char in prefix:
            if char not in node:
                return []
            node = node[char]
        return node.get('suggestions', [])
```

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRIE STRUCTURE                                           │
│                                                                             │
│  Words: ["apple", "app", "apt", "bat"]                                      │
│                                                                             │
│                    root                                                     │
│                   /    \                                                    │
│                  a      b                                                   │
│                 /        \                                                  │
│                p          a                                                 │
│               / \          \                                                │
│              p   t*         t*                                              │
│             /                                                               │
│            l                                                                │
│           /                                                                 │
│          e*                                                                 │
│                                                                             │
│  * = is_end (complete word)                                                 │
│                                                                             │
│  Space: O(ALPHABET_SIZE * KEY_LENGTH * NUM_KEYS) worst case                 │
│  Time: O(KEY_LENGTH) for insert/search/startsWith                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## When to Use Trie

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRIE VS OTHER DATA STRUCTURES                            │
│                                                                             │
│  USE TRIE when:                                                             │
│  • Prefix-based operations (autocomplete, prefix search)                    │
│  • Many strings share common prefixes                                       │
│  • Need to find all words with given prefix                                 │
│  • Pattern matching with wildcards                                          │
│  • XOR optimization problems                                                │
│                                                                             │
│  USE HASH SET when:                                                         │
│  • Only exact word matching needed                                          │
│  • No prefix operations required                                            │
│  • Memory is constrained                                                    │
│                                                                             │
│  USE HASH MAP when:                                                         │
│  • Need to store values associated with keys                                │
│  • No prefix operations required                                            │
│                                                                             │
│  TRIE ADVANTAGES:                                                           │
│  • O(L) operations regardless of dictionary size                            │
│  • Space-efficient for shared prefixes                                      │
│  • Natural for prefix operations                                            │
│                                                                             │
│  TRIE DISADVANTAGES:                                                        │
│  • Can use more memory than hash for sparse data                            │
│  • Slower than hash for exact matches                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] Implement Trie (LC #208)
- [ ] Longest Word in Dictionary (LC #720)
- [ ] Replace Words (LC #648)

### Week 2: Intermediate
- [ ] Design Add and Search Words (LC #211)
- [ ] Map Sum Pairs (LC #677)
- [ ] Search Suggestions System (LC #1268)

### Week 3: Advanced
- [ ] Word Search II (LC #212)
- [ ] Maximum XOR (LC #421)
- [ ] Palindrome Pairs (LC #336)
- [ ] Design Search Autocomplete (LC #642)

---

## Common Mistakes

1. **Not handling edge cases**
   - Empty string
   - Single character
   - Prefix is entire word

2. **Memory leaks in deletion**
   - Need to clean up empty nodes
   - Track parent pointers or use recursion

3. **Wrong termination check**
   - `is_end` vs reaching end of path
   - Prefix search vs exact match

4. **Inefficient implementation**
   - Using array[26] when hash map suffices
   - Not pruning empty branches

---

## Complexity Reference

| Operation | Time | Space |
|-----------|------|-------|
| Insert | O(L) | O(L) |
| Search | O(L) | O(1) |
| StartsWith | O(L) | O(1) |
| Delete | O(L) | O(1) |
| Autocomplete | O(L + k) | O(k) |

Where L = length of word/prefix, k = number of results

