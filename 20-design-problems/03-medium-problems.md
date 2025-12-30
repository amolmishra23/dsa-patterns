# Design Problems - Advanced

## Advanced Design Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DESIGN PROBLEM STRATEGIES                                │
│                                                                             │
│  1. COMBINE DATA STRUCTURES:                                                │
│     • HashMap + Doubly Linked List = LRU Cache                              │
│     • HashMap + Array = O(1) RandomizedSet                                  │
│     • HashMap + Two Heaps = Median Finder                                   │
│                                                                             │
│  2. LAZY OPERATIONS:                                                        │
│     • Delay expensive operations until needed                               │
│     • Batch updates                                                         │
│                                                                             │
│  3. AMORTIZED ANALYSIS:                                                     │
│     • Some operations expensive, but average is cheap                       │
│     • Example: Queue using two stacks                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem 1: LFU Cache (LC #460) - Hard

- [LeetCode](https://leetcode.com/problems/lfu-cache/)

### Problem Statement
Design Least Frequently Used cache with O(1) operations.

### Examples
```
LFUCache cache = new LFUCache(2);
cache.put(1, 1);   // freq[1] = 1
cache.put(2, 2);   // freq[2] = 1
cache.get(1);      // returns 1, freq[1] = 2
cache.put(3, 3);   // evicts key 2 (least frequent)
cache.get(2);      // returns -1 (not found)
cache.get(3);      // returns 3, freq[3] = 2
```

### Intuition Development
```
DATA STRUCTURES NEEDED:
1. key → value mapping
2. key → frequency mapping
3. frequency → keys at that frequency (ordered by recency)
4. minimum frequency tracker

VISUALIZATION:
  cache: {1:1, 3:3}
  freq:  {1:2, 3:1}

  freq_to_keys:
    1: [3]        (keys with frequency 1)
    2: [1]        (keys with frequency 2)

  min_freq: 1

PUT(key, value):
  If exists: update value, increment frequency
  If new:
    If at capacity: evict LRU at min_freq
    Add with frequency 1
    min_freq = 1

GET(key):
  If not exists: return -1
  Increment frequency
  Return value
```

### Video Explanation
- [NeetCode - LFU Cache](https://www.youtube.com/watch?v=0PSB9y8ehbk)

### Solution
```python
from collections import defaultdict, OrderedDict

class LFUCache:
    """
    LFU Cache with O(1) get and put.

    Data structures:
    - key_to_val: key -> value
    - key_to_freq: key -> frequency
    - freq_to_keys: frequency -> OrderedDict of keys (for LRU within freq)
    - min_freq: minimum frequency for O(1) eviction
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key_to_val = {}
        self.key_to_freq = {}
        self.freq_to_keys = defaultdict(OrderedDict)
        self.min_freq = 0

    def get(self, key: int) -> int:
        if key not in self.key_to_val:
            return -1

        self._update_freq(key)
        return self.key_to_val[key]

    def put(self, key: int, value: int) -> None:
        if self.capacity <= 0:
            return

        if key in self.key_to_val:
            self.key_to_val[key] = value
            self._update_freq(key)
            return

        # Evict if at capacity
        if len(self.key_to_val) >= self.capacity:
            # Get LRU key at min frequency
            evict_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
            del self.key_to_val[evict_key]
            del self.key_to_freq[evict_key]

        # Add new key with frequency 1
        self.key_to_val[key] = value
        self.key_to_freq[key] = 1
        self.freq_to_keys[1][key] = None
        self.min_freq = 1

    def _update_freq(self, key: int):
        """Increment frequency of key."""
        freq = self.key_to_freq[key]
        self.key_to_freq[key] = freq + 1

        # Remove from old frequency list
        del self.freq_to_keys[freq][key]

        # Update min_freq if needed
        if not self.freq_to_keys[freq] and self.min_freq == freq:
            self.min_freq = freq + 1

        # Add to new frequency list
        self.freq_to_keys[freq + 1][key] = None
```

### Complexity
- **Time**: O(1) for get and put
- **Space**: O(capacity)

### Edge Cases
- Capacity = 0 → all operations no-op
- Multiple accesses → frequency increases
- Ties in frequency → evict LRU among those
- Update existing key → doesn't count as new

---

## Problem 2: All O(1) Data Structure (LC #432) - Hard

- [LeetCode](https://leetcode.com/problems/all-oone-data-structure/)

### Problem Statement
Design data structure with O(1) inc, dec, getMaxKey, getMinKey.

### Examples
```
AllOne allOne = new AllOne();
allOne.inc("hello");  // count["hello"] = 1
allOne.inc("hello");  // count["hello"] = 2
allOne.getMaxKey();   // returns "hello"
allOne.getMinKey();   // returns "hello"
allOne.inc("leet");   // count["leet"] = 1
allOne.getMinKey();   // returns "leet"
```

### Intuition Development
```
DOUBLY LINKED LIST OF BUCKETS:
Each bucket has: count, set of keys with that count

head ↔ [count=1, {leet}] ↔ [count=2, {hello}] ↔ tail

Operations:
  inc("leet"):
    Move "leet" from bucket 1 to bucket 2
    If bucket 1 empty, remove it

  getMinKey(): return any key from first bucket after head
  getMaxKey(): return any key from last bucket before tail

KEY INSIGHT:
Counts are always consecutive or have gaps.
When incrementing, either move to existing next bucket or create new one.
```

### Video Explanation
- [NeetCode - All O(1) Data Structure](https://www.youtube.com/watch?v=YFRqvjjSwcc)

### Solution
```python
class AllOne:
    """
    All O(1) operations for increment, decrement, max, min.

    Strategy:
    - Doubly linked list of buckets (each bucket has count and keys)
    - HashMap: key -> bucket node
    - Buckets ordered by count
    """

    class Bucket:
        def __init__(self, count=0):
            self.count = count
            self.keys = set()
            self.prev = None
            self.next = None

    def __init__(self):
        # Dummy head and tail
        self.head = self.Bucket()
        self.tail = self.Bucket()
        self.head.next = self.tail
        self.tail.prev = self.head

        self.key_to_bucket = {}  # key -> bucket

    def _insert_after(self, node, new_node):
        """Insert new_node after node."""
        new_node.prev = node
        new_node.next = node.next
        node.next.prev = new_node
        node.next = new_node

    def _remove(self, node):
        """Remove node from list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def inc(self, key: str) -> None:
        """Increment count of key."""
        if key in self.key_to_bucket:
            bucket = self.key_to_bucket[key]
            bucket.keys.remove(key)

            # Move to next bucket (count + 1)
            if bucket.next.count == bucket.count + 1:
                next_bucket = bucket.next
            else:
                next_bucket = self.Bucket(bucket.count + 1)
                self._insert_after(bucket, next_bucket)

            next_bucket.keys.add(key)
            self.key_to_bucket[key] = next_bucket

            # Remove empty bucket
            if not bucket.keys:
                self._remove(bucket)
        else:
            # New key with count 1
            if self.head.next.count == 1:
                bucket = self.head.next
            else:
                bucket = self.Bucket(1)
                self._insert_after(self.head, bucket)

            bucket.keys.add(key)
            self.key_to_bucket[key] = bucket

    def dec(self, key: str) -> None:
        """Decrement count of key."""
        if key not in self.key_to_bucket:
            return

        bucket = self.key_to_bucket[key]
        bucket.keys.remove(key)

        if bucket.count == 1:
            # Remove key entirely
            del self.key_to_bucket[key]
        else:
            # Move to previous bucket (count - 1)
            if bucket.prev.count == bucket.count - 1:
                prev_bucket = bucket.prev
            else:
                prev_bucket = self.Bucket(bucket.count - 1)
                self._insert_after(bucket.prev, prev_bucket)

            prev_bucket.keys.add(key)
            self.key_to_bucket[key] = prev_bucket

        # Remove empty bucket
        if not bucket.keys:
            self._remove(bucket)

    def getMaxKey(self) -> str:
        """Get key with max count."""
        if self.tail.prev == self.head:
            return ""
        return next(iter(self.tail.prev.keys))

    def getMinKey(self) -> str:
        """Get key with min count."""
        if self.head.next == self.tail:
            return ""
        return next(iter(self.head.next.keys))
```

### Complexity
- **Time**: O(1) for all operations
- **Space**: O(n)

### Edge Cases
- Empty data structure → getMax/getMin return ""
- Single key → it's both max and min
- Decrement to 0 → remove key
- Multiple keys with same count

---

## Problem 3: Design Search Autocomplete System (LC #642) - Hard

- [LeetCode](https://leetcode.com/problems/design-search-autocomplete-system/)

### Problem Statement
Design autocomplete with history-based suggestions.

### Examples
```
AutocompleteSystem(["i love you", "island", "ironman"], [5,3,2])
input('i') → ["i love you", "island", "ironman"]
input(' ') → ["i love you"]
input('a') → []
input('#') → [] (saves "i a")
```

### Intuition Development
```
TRIE + FREQUENCY:
Store sentences in trie, track frequencies.

Trie structure:
  root → i → ' ' → l → o → v → e → ' ' → y → o → u → # (freq=5)
       ↓
       s → l → a → n → d → # (freq=3)
       ↓
       r → o → n → m → a → n → # (freq=2)

Query process:
  Track current node as user types
  Collect all sentences from current subtree
  Sort by (-frequency, lexicographic)
  Return top 3
```

### Video Explanation
- [NeetCode - Design Search Autocomplete System](https://www.youtube.com/watch?v=xgVwRgdKXt4)

### Solution
```python
import heapq
from collections import defaultdict

class AutocompleteSystem:
    """
    Autocomplete system with history-based ranking.

    Strategy:
    - Trie stores sentences and their frequencies
    - Track current input prefix
    - Return top 3 by frequency (then lexicographic)
    """

    def __init__(self, sentences: list[str], times: list[int]):
        self.trie = {}
        self.freq = defaultdict(int)
        self.current_node = self.trie
        self.current_input = []

        # Build initial data
        for sentence, count in zip(sentences, times):
            self.freq[sentence] = count
            self._add_to_trie(sentence)

    def _add_to_trie(self, sentence: str):
        node = self.trie
        for char in sentence:
            node = node.setdefault(char, {})
        node['#'] = sentence

    def _collect_sentences(self, node, sentences):
        """Collect all sentences in subtree."""
        if '#' in node:
            sentences.append(node['#'])

        for char, child in node.items():
            if char != '#':
                self._collect_sentences(child, sentences)

    def input(self, c: str) -> list[str]:
        if c == '#':
            # End of input - save sentence
            sentence = ''.join(self.current_input)
            self.freq[sentence] += 1
            self._add_to_trie(sentence)

            # Reset
            self.current_node = self.trie
            self.current_input = []
            return []

        self.current_input.append(c)

        # Navigate trie
        if self.current_node is None or c not in self.current_node:
            self.current_node = None
            return []

        self.current_node = self.current_node[c]

        # Collect matching sentences
        sentences = []
        self._collect_sentences(self.current_node, sentences)

        # Sort by frequency (desc), then lexicographically
        sentences.sort(key=lambda x: (-self.freq[x], x))

        return sentences[:3]
```

### Complexity
- **Time**: O(n × L) build, O(p + matches) per input
- **Space**: O(n × L)

### Edge Cases
- Empty initial sentences
- Query with no matches
- Very long sentences
- '#' in sentence content (shouldn't happen per problem)

---

## Problem 4: Design In-Memory File System (LC #588) - Hard

- [LeetCode](https://leetcode.com/problems/design-in-memory-file-system/)

### Problem Statement
Design a file system with ls, mkdir, addContentToFile, readContentFromFile.

### Examples
```
FileSystem fs = new FileSystem();
fs.ls("/");                       // []
fs.mkdir("/a/b/c");
fs.addContentToFile("/a/b/c/d", "hello");
fs.ls("/");                       // ["a"]
fs.readContentFromFile("/a/b/c/d"); // "hello"
```

### Intuition Development
```
TRIE-LIKE STRUCTURE:
Each node can be directory (has children) or file (has content).

Structure:
  root
   └── a (dir)
        └── b (dir)
             └── c (dir)
                  └── d (file, content="hello")

Node structure:
  {
    is_file: bool,
    content: str,
    children: {name: node}
  }
```

### Video Explanation
- [NeetCode - Design In-Memory File System](https://www.youtube.com/watch?v=ZPPBkk0U6UM)

### Solution
```python
class FileSystem:
    """
    In-memory file system using trie-like structure.

    Each node can be a directory (children dict) or file (content).
    """

    def __init__(self):
        self.root = {'is_file': False, 'content': '', 'children': {}}

    def _get_node(self, path: str):
        """Navigate to node at path, creating directories as needed."""
        node = self.root

        if path == '/':
            return node

        parts = path.split('/')[1:]  # Skip empty string before first /

        for part in parts:
            if part not in node['children']:
                node['children'][part] = {
                    'is_file': False,
                    'content': '',
                    'children': {}
                }
            node = node['children'][part]

        return node

    def ls(self, path: str) -> list[str]:
        """List directory contents or file name."""
        node = self._get_node(path)

        if node['is_file']:
            # Return just the file name
            return [path.split('/')[-1]]

        # Return sorted directory contents
        return sorted(node['children'].keys())

    def mkdir(self, path: str) -> None:
        """Create directory path."""
        self._get_node(path)

    def addContentToFile(self, filePath: str, content: str) -> None:
        """Add content to file (create if doesn't exist)."""
        node = self._get_node(filePath)
        node['is_file'] = True
        node['content'] += content

    def readContentFromFile(self, filePath: str) -> str:
        """Read file content."""
        node = self._get_node(filePath)
        return node['content']
```

### Complexity
- **Time**: O(path length) for all operations
- **Space**: O(total content)

### Edge Cases
- Root directory operations
- File and directory with same name (not allowed typically)
- Empty file
- Deep nested paths

---

## Problem 5: Design Skiplist (LC #1206) - Hard

- [LeetCode](https://leetcode.com/problems/design-skiplist/)

### Problem Statement
Design a Skiplist with search, add, erase in O(log n) average.

### Examples
```
Skiplist sl = new Skiplist();
sl.add(1);
sl.add(2);
sl.add(3);
sl.search(0); // false
sl.add(4);
sl.search(1); // true
sl.erase(0);  // false
sl.erase(1);  // true
sl.search(1); // false
```

### Intuition Development
```
MULTIPLE LEVELS OF SORTED LISTS:
Higher levels skip more elements, like express lanes.

Level 3: head ────────────────────────→ 9 → null
Level 2: head ──────→ 3 ──────────────→ 9 → null
Level 1: head ──→ 1 → 3 ──────→ 6 ──→ 9 → null
Level 0: head → 1 → 2 → 3 → 4 → 6 → 7 → 9 → null

SEARCH for 7:
  Level 3: 9 > 7, go down
  Level 2: 3 < 7, go right; 9 > 7, go down
  Level 1: 3 < 7, go right; 6 < 7, go right; 9 > 7, go down
  Level 0: 7 found!

PROBABILISTIC BALANCING:
When inserting, flip coin to decide level.
P(level k) = (1/2)^k
```

### Video Explanation
- [NeetCode - Design Skiplist](https://www.youtube.com/watch?v=kBwUoWpeH_Q)

### Solution
```python
import random

class Skiplist:
    """
    Skiplist with probabilistic balancing.

    Structure:
    - Multiple levels of sorted linked lists
    - Higher levels skip more elements
    - Probability p = 0.5 for level promotion
    """

    class Node:
        def __init__(self, val=-1, levels=1):
            self.val = val
            self.next = [None] * levels

    def __init__(self):
        self.max_level = 16
        self.head = self.Node(-1, self.max_level)
        self.level = 1

    def _random_level(self):
        """Generate random level with geometric distribution."""
        level = 1
        while random.random() < 0.5 and level < self.max_level:
            level += 1
        return level

    def search(self, target: int) -> bool:
        """Search for target value."""
        node = self.head

        for i in range(self.level - 1, -1, -1):
            while node.next[i] and node.next[i].val < target:
                node = node.next[i]

        node = node.next[0]
        return node is not None and node.val == target

    def add(self, num: int) -> None:
        """Add value to skiplist."""
        update = [None] * self.max_level
        node = self.head

        # Find insert position at each level
        for i in range(self.level - 1, -1, -1):
            while node.next[i] and node.next[i].val < num:
                node = node.next[i]
            update[i] = node

        # Random level for new node
        new_level = self._random_level()

        if new_level > self.level:
            for i in range(self.level, new_level):
                update[i] = self.head
            self.level = new_level

        # Insert new node
        new_node = self.Node(num, new_level)
        for i in range(new_level):
            new_node.next[i] = update[i].next[i]
            update[i].next[i] = new_node

    def erase(self, num: int) -> bool:
        """Remove one occurrence of value."""
        update = [None] * self.max_level
        node = self.head

        for i in range(self.level - 1, -1, -1):
            while node.next[i] and node.next[i].val < num:
                node = node.next[i]
            update[i] = node

        node = node.next[0]

        if node is None or node.val != num:
            return False

        # Remove node from each level
        for i in range(self.level):
            if update[i].next[i] != node:
                break
            update[i].next[i] = node.next[i]

        # Reduce level if needed
        while self.level > 1 and self.head.next[self.level - 1] is None:
            self.level -= 1

        return True
```

### Complexity
- **Time**: O(log n) average for all operations
- **Space**: O(n) average

### Edge Cases
- Empty skiplist
- Duplicate values (add multiple, erase one)
- Search/erase non-existent value
- Many operations causing level changes

---

## Problem 6: Design Excel Sum Formula (LC #631) - Hard

- [LeetCode](https://leetcode.com/problems/design-excel-sum-formula/)

### Problem Statement
Design Excel with sum formulas that update automatically.

### Examples
```
Excel excel = new Excel(3, 'C');
excel.set(1, 'A', 2);
excel.sum(3, 'C', ["A1", "A1:B2"]);  // C3 = sum of A1 and A1:B2
excel.set(2, 'A', 2);
excel.get(3, 'C');  // returns updated sum
```

### Intuition Development
```
LAZY EVALUATION:
Store formulas, evaluate on get.

Grid state:
  A   B   C
1 2   0   0
2 2   0   0
3 0   0   formula:["A1","A1:B2"]

get(3, 'C'):
  Evaluate formula:
  A1 = 2
  A1:B2 = A1 + A2 + B1 + B2 = 2 + 2 + 0 + 0 = 4
  Total = 2 + 4 = 6

If we set(2, 'A', 5):
  Next get(3, 'C') would return 2 + (2+5+0+0) = 9
```

### Video Explanation
- [NeetCode - Design Excel Sum Formula](https://www.youtube.com/watch?v=8GKGH1sVm_M)

### Solution
```python
class Excel:
    """
    Excel with sum formulas.

    Strategy:
    - Store cell values and formulas separately
    - Recalculate on get (lazy evaluation)
    """

    def __init__(self, height: int, width: str):
        self.height = height
        self.width = ord(width) - ord('A') + 1
        self.values = [[0] * self.width for _ in range(height)]
        self.formulas = [[None] * self.width for _ in range(height)]

    def _parse_cell(self, cell: str) -> tuple:
        """Parse cell reference like 'A1' to (row, col)."""
        col = ord(cell[0]) - ord('A')
        row = int(cell[1:]) - 1
        return row, col

    def set(self, row: int, column: str, val: int) -> None:
        """Set cell to value (clears formula)."""
        r = row - 1
        c = ord(column) - ord('A')
        self.values[r][c] = val
        self.formulas[r][c] = None

    def get(self, row: int, column: str) -> int:
        """Get cell value (evaluates formula if present)."""
        r = row - 1
        c = ord(column) - ord('A')

        if self.formulas[r][c] is None:
            return self.values[r][c]

        # Evaluate formula
        total = 0
        for cell_range in self.formulas[r][c]:
            if ':' in cell_range:
                # Range like "A1:B2"
                start, end = cell_range.split(':')
                r1, c1 = self._parse_cell(start)
                r2, c2 = self._parse_cell(end)

                for i in range(r1, r2 + 1):
                    for j in range(c1, c2 + 1):
                        total += self.get(i + 1, chr(ord('A') + j))
            else:
                # Single cell like "A1"
                ri, ci = self._parse_cell(cell_range)
                total += self.get(ri + 1, chr(ord('A') + ci))

        return total

    def sum(self, row: int, column: str, numbers: list[str]) -> int:
        """Set cell to sum formula."""
        r = row - 1
        c = ord(column) - ord('A')
        self.formulas[r][c] = numbers
        return self.get(row, column)
```

### Complexity
- **Time**: O(cells in formula) per get
- **Space**: O(cells)

### Edge Cases
- Circular references (may need detection)
- Empty range
- Setting cell clears formula
- Nested formulas (formula references formula cell)

---

## Summary: Advanced Design Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | LFU Cache | HashMap + freq buckets | O(1) |
| 2 | All O(1) | Doubly linked buckets | O(1) |
| 3 | Autocomplete | Trie + frequency | O(matches) |
| 4 | File System | Trie-like structure | O(path) |
| 5 | Skiplist | Probabilistic levels | O(log n) |
| 6 | Excel | Lazy formula evaluation | O(cells) |

---

## Design Problem Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DESIGN PATTERNS                                          │
│                                                                             │
│  O(1) ACCESS + ORDER:                                                       │
│     HashMap + Doubly Linked List                                            │
│     Examples: LRU Cache, LFU Cache, All O(1)                               │
│                                                                             │
│  O(1) RANDOM ACCESS + INSERT/DELETE:                                       │
│     HashMap + Array with swap-to-end deletion                              │
│     Example: RandomizedSet                                                  │
│                                                                             │
│  PREFIX OPERATIONS:                                                         │
│     Trie structure                                                          │
│     Examples: Autocomplete, File System                                     │
│                                                                             │
│  LOGARITHMIC OPERATIONS:                                                    │
│     Balanced trees, Heaps, Skiplists                                       │
│     Examples: Median Finder, Skiplist                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practice More Problems

- [ ] LC #716 - Max Stack
- [ ] LC #895 - Maximum Frequency Stack
- [ ] LC #1146 - Snapshot Array
- [ ] LC #1172 - Dinner Plate Stacks
