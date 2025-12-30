# Design Problems - Complete Practice List

## Organized by Category

### Category 1: Cache Design

| # | Problem | Difficulty | Key Data Structures |
|---|---------|------------|---------------------|
| 146 | [LRU Cache](https://leetcode.com/problems/lru-cache/) | Medium | HashMap + Doubly Linked List |
| 460 | [LFU Cache](https://leetcode.com/problems/lfu-cache/) | Hard | HashMap + Frequency Buckets |
| 432 | [All O(1) Data Structure](https://leetcode.com/problems/all-oone-data-structure/) | Hard | HashMap + Doubly Linked Buckets |

### Category 2: Stack/Queue Design

| # | Problem | Difficulty | Key Data Structures |
|---|---------|------------|---------------------|
| 155 | [Min Stack](https://leetcode.com/problems/min-stack/) | Medium | Stack + Min tracking |
| 232 | [Queue using Stacks](https://leetcode.com/problems/implement-queue-using-stacks/) | Easy | Two stacks |
| 225 | [Stack using Queues](https://leetcode.com/problems/implement-stack-using-queues/) | Easy | One/two queues |
| 716 | [Max Stack](https://leetcode.com/problems/max-stack/) | Hard | DLL + TreeMap |
| 895 | [Maximum Frequency Stack](https://leetcode.com/problems/maximum-frequency-stack/) | Hard | HashMap + Stack per freq |

### Category 3: Iterator Design

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 173 | [BST Iterator](https://leetcode.com/problems/binary-search-tree-iterator/) | Medium | Controlled inorder |
| 284 | [Peeking Iterator](https://leetcode.com/problems/peeking-iterator/) | Medium | Cache next element |
| 341 | [Flatten Nested List Iterator](https://leetcode.com/problems/flatten-nested-list-iterator/) | Medium | Stack of iterators |
| 251 | [Flatten 2D Vector](https://leetcode.com/problems/flatten-2d-vector/) | Medium | Two pointers |
| 281 | [Zigzag Iterator](https://leetcode.com/problems/zigzag-iterator/) | Medium | Queue of iterators |

### Category 4: Data Stream

| # | Problem | Difficulty | Key Data Structures |
|---|---------|------------|---------------------|
| 295 | [Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/) | Hard | Two heaps |
| 346 | [Moving Average](https://leetcode.com/problems/moving-average-from-data-stream/) | Easy | Queue |
| 352 | [Data Stream as Disjoint Intervals](https://leetcode.com/problems/data-stream-as-disjoint-intervals/) | Hard | TreeMap |
| 703 | [Kth Largest in Stream](https://leetcode.com/problems/kth-largest-element-in-a-stream/) | Easy | Min heap |
| 1352 | [Product of Last K Numbers](https://leetcode.com/problems/product-of-the-last-k-numbers/) | Medium | Prefix product |

### Category 5: Set/Map Design

| # | Problem | Difficulty | Key Data Structures |
|---|---------|------------|---------------------|
| 380 | [Insert Delete GetRandom O(1)](https://leetcode.com/problems/insert-delete-getrandom-o1/) | Medium | HashMap + Array |
| 381 | [Insert Delete GetRandom Duplicates](https://leetcode.com/problems/insert-delete-getrandom-o1-duplicates-allowed/) | Hard | HashMap + Set + Array |
| 705 | [Design HashSet](https://leetcode.com/problems/design-hashset/) | Easy | Bucket array |
| 706 | [Design HashMap](https://leetcode.com/problems/design-hashmap/) | Easy | Bucket array |
| 1146 | [Snapshot Array](https://leetcode.com/problems/snapshot-array/) | Medium | Binary search on history |

### Category 6: Tree/Graph Design

| # | Problem | Difficulty | Key Data Structures |
|---|---------|------------|---------------------|
| 208 | [Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree/) | Medium | Trie node |
| 211 | [Add and Search Word](https://leetcode.com/problems/design-add-and-search-words-data-structure/) | Medium | Trie + DFS |
| 588 | [Design In-Memory File System](https://leetcode.com/problems/design-in-memory-file-system/) | Hard | Trie for paths |
| 642 | [Design Search Autocomplete](https://leetcode.com/problems/design-search-autocomplete-system/) | Hard | Trie + ranking |

### Category 7: Application Design

| # | Problem | Difficulty | Key Data Structures |
|---|---------|------------|---------------------|
| 355 | [Design Twitter](https://leetcode.com/problems/design-twitter/) | Medium | HashMap + Heap |
| 362 | [Design Hit Counter](https://leetcode.com/problems/design-hit-counter/) | Medium | Queue or circular array |
| 379 | [Design Phone Directory](https://leetcode.com/problems/design-phone-directory/) | Medium | Set or queue |
| 1166 | [Design File System](https://leetcode.com/problems/design-file-system/) | Medium | HashMap of paths |

---

## Essential Design Patterns

### 1. LRU Cache
```python
class LRUCache:
    """HashMap + Doubly Linked List."""

    class Node:
        def __init__(self, key=0, val=0):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add(self, node):
        """Add node right after head."""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove(self, node):
        """Remove node from list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self._remove(node)
        self._add(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])

        node = self.Node(key, value)
        self.cache[key] = node
        self._add(node)

        if len(self.cache) > self.capacity:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
```

### 2. Min Stack
```python
class MinStack:
    """Stack with O(1) min retrieval."""

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)

        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        val = self.stack.pop()

        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```

### 3. Insert Delete GetRandom O(1)
```python
import random

class RandomizedSet:
    """HashMap + Array for O(1) operations."""

    def __init__(self):
        self.val_to_idx = {}
        self.values = []

    def insert(self, val: int) -> bool:
        if val in self.val_to_idx:
            return False

        self.val_to_idx[val] = len(self.values)
        self.values.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.val_to_idx:
            return False

        idx = self.val_to_idx[val]
        last = self.values[-1]

        # Swap with last element
        self.values[idx] = last
        self.val_to_idx[last] = idx

        # Remove last
        self.values.pop()
        del self.val_to_idx[val]

        return True

    def getRandom(self) -> int:
        return random.choice(self.values)
```

### 4. BST Iterator
```python
class BSTIterator:
    """Controlled inorder traversal."""

    def __init__(self, root):
        self.stack = []
        self._push_left(root)

    def _push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left

    def next(self) -> int:
        node = self.stack.pop()
        self._push_left(node.right)
        return node.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0
```

---

## Design Interview Tips

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DESIGN PROBLEM APPROACH                                  │
│                                                                             │
│  1. CLARIFY REQUIREMENTS                                                    │
│     • What operations are needed?                                           │
│     • What are the time complexity requirements?                            │
│     • What are the space constraints?                                       │
│     • How will the data structure be used?                                  │
│                                                                             │
│  2. CONSIDER DATA STRUCTURES                                                │
│     • HashMap: O(1) lookup by key                                           │
│     • Array: O(1) random access, O(n) search                               │
│     • Linked List: O(1) insert/delete at known position                    │
│     • Heap: O(log n) insert, O(1) min/max                                  │
│     • Balanced BST: O(log n) all operations, ordered                       │
│                                                                             │
│  3. COMBINE DATA STRUCTURES                                                 │
│     • HashMap + DLL = LRU Cache                                            │
│     • HashMap + Array = O(1) RandomizedSet                                 │
│     • HashMap + Heap = Priority queue with update                          │
│     • Two Heaps = Median finder                                            │
│                                                                             │
│  4. HANDLE EDGE CASES                                                       │
│     • Empty data structure                                                  │
│     • Single element                                                        │
│     • Duplicate values                                                      │
│     • Capacity limits                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] Min Stack
- [ ] Queue using Stacks
- [ ] Design HashSet
- [ ] Design HashMap
- [ ] Moving Average

### Week 2: Intermediate
- [ ] LRU Cache
- [ ] Insert Delete GetRandom O(1)
- [ ] BST Iterator
- [ ] Kth Largest in Stream
- [ ] Design Hit Counter

### Week 3: Advanced
- [ ] LFU Cache
- [ ] Find Median from Data Stream
- [ ] Maximum Frequency Stack
- [ ] Design Twitter
- [ ] All O(1) Data Structure

---

## Complexity Requirements

| Problem | Get | Put/Insert | Delete | Extra |
|---------|-----|------------|--------|-------|
| LRU Cache | O(1) | O(1) | O(1) | - |
| LFU Cache | O(1) | O(1) | O(1) | - |
| Min Stack | O(1) | O(1) | O(1) | getMin O(1) |
| RandomizedSet | - | O(1) | O(1) | getRandom O(1) |
| Median Finder | - | O(log n) | - | findMedian O(1) |

---

## Common Mistakes

1. **Not handling edge cases**
   - Empty structure operations
   - Capacity overflow
   - Duplicate handling

2. **Wrong complexity**
   - Using wrong data structure
   - Missing optimization opportunity

3. **Memory leaks**
   - Not cleaning up deleted nodes
   - Circular references

4. **Thread safety** (if required)
   - Race conditions
   - Deadlocks

