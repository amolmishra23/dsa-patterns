# Design Problems - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE DESIGN PATTERNS                              │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Design a data structure"                                                │
│  ✓ "Implement a class"                                                      │
│  ✓ "LRU Cache" / "LFU Cache"                                                │
│  ✓ "Rate limiter"                                                           │
│  ✓ "Iterator" / "Flatten"                                                   │
│  ✓ "Serialize/Deserialize"                                                  │
│  ✓ "O(1) operations"                                                        │
│  ✓ "Random access"                                                          │
│                                                                             │
│  Key insight: Combine multiple data structures for optimal operations       │
│               Hash map + Linked list = O(1) lookup + O(1) order change      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Hash maps/dictionaries (O(1) lookup)
- [ ] Linked lists (doubly linked for O(1) removal)
- [ ] Heaps/Priority queues
- [ ] Basic OOP concepts (classes, methods)

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DESIGN PROBLEMS MEMORY MAP                               │
│                                                                             │
│                    ┌─────────────────┐                                      │
│         ┌─────────│ DESIGN PROBLEMS │─────────┐                             │
│         │         └─────────────────┘         │                             │
│         ▼                                     ▼                             │
│  ┌─────────────┐                       ┌─────────────┐                      │
│  │   CACHING   │                       │   STORAGE   │                      │
│  └──────┬──────┘                       └──────┬──────┘                      │
│         │                                     │                             │
│    ┌────┴────┐                         ┌──────┴──────┐                      │
│    ▼         ▼                         ▼             ▼                      │
│ ┌─────┐  ┌─────┐                   ┌────────┐  ┌──────────┐                │
│ │ LRU │  │ LFU │                   │HashMap │  │RandomSet │                │
│ │Cache│  │Cache│                   │  Trie  │  │  Stack   │                │
│ └─────┘  └─────┘                   └────────┘  └──────────┘                │
│                                                                             │
│  Data Structure Combinations:                                               │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Problem          │ Structures Used                                 │    │
│  ├──────────────────┼────────────────────────────────────────────────┤    │
│  │ LRU Cache        │ HashMap + Doubly Linked List                   │    │
│  │ LFU Cache        │ HashMap + HashMap + OrderedDict                │    │
│  │ RandomizedSet    │ Array + HashMap                                │    │
│  │ Min Stack        │ Stack + Min tracking                           │    │
│  │ Twitter Feed     │ HashMap + Heap                                 │    │
│  │ Median Finder    │ Two Heaps (min + max)                          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DESIGN PROBLEM DECISION TREE                             │
│                                                                             │
│  Need O(1) lookup AND maintain order?                                       │
│       │                                                                     │
│       ├── YES → HashMap + Linked List (LRU/LFU Cache)                       │
│       │                                                                     │
│       └── NO → Need O(1) random access?                                     │
│                    │                                                        │
│                    ├── YES → Array + HashMap (RandomizedSet)                │
│                    │                                                        │
│                    └── NO → Need O(1) min/max?                              │
│                                 │                                           │
│                                 ├── YES → Track min/max with each element   │
│                                 │         or use Two Heaps                  │
│                                 │                                           │
│                                 └── NO → Need priority ordering?            │
│                                              │                              │
│                                              ├── YES → Heap + HashMap       │
│                                              │                              │
│                                              └── NO → Standard data struct  │
│                                                                             │
│  QUICK REFERENCE - Data Structure Selection:                                │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ Requirement              │ Data Structure                          │     │
│  ├──────────────────────────┼─────────────────────────────────────────┤     │
│  │ O(1) lookup              │ HashMap                                 │     │
│  │ O(1) order maintenance   │ Doubly Linked List                      │     │
│  │ O(1) random element      │ Array + HashMap                         │     │
│  │ O(1) min/max             │ Track with each element                 │     │
│  │ O(log n) priority        │ Heap                                    │     │
│  │ O(1) prefix lookup       │ Trie                                    │     │
│  │ Streaming median         │ Two Heaps                               │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Design Problems

### Problem 1: LRU Cache (LC #146)

```python
from collections import OrderedDict

class LRUCache:
    """
    Least Recently Used Cache.

    Operations: get(key), put(key, value) - both O(1)

    Strategy: Use OrderedDict (hash map + doubly linked list)
    - Hash map: O(1) lookup
    - Doubly linked list: O(1) removal and insertion

    Time: O(1) for get and put
    Space: O(capacity)
    """

    def __init__(self, capacity: int):
        """
        Initialize LRU cache with given capacity.
        """
        self.capacity = capacity
        self.cache = OrderedDict()  # Maintains insertion order

    def get(self, key: int) -> int:
        """
        Get value for key. Return -1 if not found.
        Move to end (most recently used) if found.
        """
        if key not in self.cache:
            return -1

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        """
        Put key-value pair. Evict LRU if at capacity.
        """
        if key in self.cache:
            # Update existing key, move to end
            self.cache.move_to_end(key)

        self.cache[key] = value

        # Evict LRU (first item) if over capacity
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class LRUCacheManual:
    """
    LRU Cache with manual doubly linked list implementation.
    """

    class Node:
        def __init__(self, key=0, val=0):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Dummy head and tail for easier operations
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        """Remove node from linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_end(self, node):
        """Add node right before tail (most recent)."""
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]
        self._remove(node)
        self._add_to_end(node)
        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self._remove(self.cache[key])

        node = self.Node(key, value)
        self.cache[key] = node
        self._add_to_end(node)

        if len(self.cache) > self.capacity:
            # Remove LRU (node after head)
            lru = self.head.next
            self._remove(lru)
            del self.cache[lru.key]
```

### Problem 2: LFU Cache (LC #460)

```python
from collections import defaultdict, OrderedDict

class LFUCache:
    """
    Least Frequently Used Cache.

    Evict least frequently used. If tie, evict least recently used.

    Strategy:
    - key_to_val: key -> value
    - key_to_freq: key -> frequency
    - freq_to_keys: frequency -> OrderedDict of keys (for LRU within freq)
    - min_freq: track minimum frequency for O(1) eviction

    Time: O(1) for get and put
    Space: O(capacity)
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

        # Add new key
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

### Problem 3: Min Stack (LC #155)

```python
class MinStack:
    """
    Stack with O(1) getMin operation.

    Strategy: Store (value, min_so_far) pairs.

    Time: O(1) for all operations
    Space: O(n)
    """

    def __init__(self):
        self.stack = []  # [(value, min_so_far), ...]

    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
        else:
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

### Problem 4: Design Twitter (LC #355)

```python
import heapq
from collections import defaultdict

class Twitter:
    """
    Design Twitter with follow, unfollow, postTweet, getNewsFeed.

    Time:
    - postTweet: O(1)
    - getNewsFeed: O(n log k) where n = tweets, k = followees
    - follow/unfollow: O(1)
    """

    def __init__(self):
        self.time = 0
        self.tweets = defaultdict(list)  # userId -> [(time, tweetId), ...]
        self.following = defaultdict(set)  # userId -> set of followeeIds

    def postTweet(self, userId: int, tweetId: int) -> None:
        """Post a tweet."""
        self.tweets[userId].append((self.time, tweetId))
        self.time += 1

    def getNewsFeed(self, userId: int) -> list[int]:
        """
        Get 10 most recent tweets from user and followees.

        Strategy: Merge k sorted lists using heap.
        """
        # Get all users to fetch tweets from
        users = self.following[userId] | {userId}

        # Min heap: (-time, tweetId, userId, index)
        # Negative time for max behavior
        heap = []

        for user in users:
            if self.tweets[user]:
                idx = len(self.tweets[user]) - 1
                time, tweet_id = self.tweets[user][idx]
                heapq.heappush(heap, (-time, tweet_id, user, idx))

        result = []
        while heap and len(result) < 10:
            _, tweet_id, user, idx = heapq.heappop(heap)
            result.append(tweet_id)

            # Add next tweet from same user
            if idx > 0:
                time, tweet_id = self.tweets[user][idx - 1]
                heapq.heappush(heap, (-time, tweet_id, user, idx - 1))

        return result

    def follow(self, followerId: int, followeeId: int) -> None:
        """User follows another user."""
        if followerId != followeeId:
            self.following[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        """User unfollows another user."""
        self.following[followerId].discard(followeeId)
```

### Problem 5: Insert Delete GetRandom O(1) (LC #380)

```python
import random

class RandomizedSet:
    """
    Set with O(1) insert, remove, and getRandom.

    Strategy:
    - List for O(1) random access
    - Hash map for O(1) lookup (value -> index)
    - Swap with last element before removal

    Time: O(1) for all operations
    Space: O(n)
    """

    def __init__(self):
        self.list = []
        self.index_map = {}  # value -> index in list

    def insert(self, val: int) -> bool:
        """Insert value. Return True if not present."""
        if val in self.index_map:
            return False

        self.index_map[val] = len(self.list)
        self.list.append(val)
        return True

    def remove(self, val: int) -> bool:
        """Remove value. Return True if present."""
        if val not in self.index_map:
            return False

        # Swap with last element
        idx = self.index_map[val]
        last_val = self.list[-1]

        self.list[idx] = last_val
        self.index_map[last_val] = idx

        # Remove last element
        self.list.pop()
        del self.index_map[val]

        return True

    def getRandom(self) -> int:
        """Get random element."""
        return random.choice(self.list)
```

### Problem 6: Serialize and Deserialize Binary Tree (LC #297)

```python
class Codec:
    """
    Serialize and deserialize binary tree.

    Strategy: Preorder traversal with null markers.

    Time: O(n)
    Space: O(n)
    """

    def serialize(self, root) -> str:
        """
        Encode tree to string.

        Use preorder traversal, mark null nodes with 'N'.
        """
        result = []

        def preorder(node):
            if not node:
                result.append('N')
                return

            result.append(str(node.val))
            preorder(node.left)
            preorder(node.right)

        preorder(root)
        return ','.join(result)

    def deserialize(self, data: str):
        """
        Decode string to tree.
        """
        values = iter(data.split(','))

        def build():
            val = next(values)

            if val == 'N':
                return None

            node = TreeNode(int(val))
            node.left = build()
            node.right = build()
            return node

        return build()
```

### Problem 7: Design HashMap (LC #706)

```python
class MyHashMap:
    """
    Design HashMap with put, get, remove.

    Strategy: Array of buckets with chaining (linked list).

    Time: O(n/k) average where k = number of buckets
    Space: O(k + n)
    """

    def __init__(self):
        self.size = 1000  # Number of buckets
        self.buckets = [[] for _ in range(self.size)]

    def _hash(self, key: int) -> int:
        """Hash function."""
        return key % self.size

    def put(self, key: int, value: int) -> None:
        """Add or update key-value pair."""
        bucket_idx = self._hash(key)
        bucket = self.buckets[bucket_idx]

        # Check if key exists
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        # Add new key
        bucket.append((key, value))

    def get(self, key: int) -> int:
        """Get value for key. Return -1 if not found."""
        bucket_idx = self._hash(key)
        bucket = self.buckets[bucket_idx]

        for k, v in bucket:
            if k == key:
                return v

        return -1

    def remove(self, key: int) -> None:
        """Remove key."""
        bucket_idx = self._hash(key)
        bucket = self.buckets[bucket_idx]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                return
```

---

## Design Tips

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DESIGN PROBLEM APPROACH                                  │
│                                                                             │
│  1. CLARIFY REQUIREMENTS                                                    │
│     - What operations are needed?                                           │
│     - What are the time/space constraints?                                  │
│     - What are the edge cases?                                              │
│                                                                             │
│  2. CHOOSE DATA STRUCTURES                                                  │
│     - Hash map for O(1) lookup                                              │
│     - Linked list for O(1) insertion/deletion                               │
│     - Heap for O(log n) priority operations                                 │
│     - Array for O(1) random access                                          │
│                                                                             │
│  3. COMBINE STRUCTURES                                                      │
│     - LRU Cache: Hash map + Doubly linked list                              │
│     - RandomizedSet: Array + Hash map                                       │
│     - Twitter: Hash map + Heap                                              │
│                                                                             │
│  4. HANDLE EDGE CASES                                                       │
│     - Empty state                                                           │
│     - Capacity limits                                                       │
│     - Duplicate operations                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

---

## Complexity Analysis

| Design Problem | Operation | Time | Space | Key Insight |
|----------------|-----------|------|-------|-------------|
| LRU Cache | get/put | O(1) | O(n) | HashMap + DLL |
| LFU Cache | get/put | O(1) | O(n) | 3 HashMaps |
| Min Stack | push/pop/getMin | O(1) | O(n) | Track min with each element |
| RandomizedSet | insert/remove/random | O(1) | O(n) | Swap with last before remove |
| Twitter Feed | getNewsFeed | O(k log n) | O(n) | Merge k sorted lists |
| Median Finder | addNum/findMedian | O(log n) / O(1) | O(n) | Two heaps |
| Trie | insert/search | O(m) | O(m×n) | m = word length |
| HashMap | get/put | O(1) avg | O(n) | Chaining or open addressing |

---

## Common Mistakes

```python
# ❌ WRONG: LRU Cache - Not updating position on get()
class LRUCacheWrong:
    def get(self, key):
        if key in self.cache:
            return self.cache[key]  # Forgot to move to end!
        return -1

# ✅ CORRECT: Move to end on access
class LRUCacheCorrect:
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
            return self.cache[key]
        return -1


# ❌ WRONG: RandomizedSet - Wrong removal (leaves gap)
class RandomizedSetWrong:
    def remove(self, val):
        if val in self.index_map:
            idx = self.index_map[val]
            del self.list[idx]  # O(n) and leaves index gaps!
            del self.index_map[val]

# ✅ CORRECT: Swap with last element before removal
class RandomizedSetCorrect:
    def remove(self, val):
        if val in self.index_map:
            idx = self.index_map[val]
            last_val = self.list[-1]

            # Swap with last
            self.list[idx] = last_val
            self.index_map[last_val] = idx

            # Remove last (O(1))
            self.list.pop()
            del self.index_map[val]


# ❌ WRONG: Min Stack - Recalculating min on every call
class MinStackWrong:
    def getMin(self):
        return min(self.stack)  # O(n) every time!

# ✅ CORRECT: Track min with each element
class MinStackCorrect:
    def push(self, val):
        current_min = min(val, self.stack[-1][1]) if self.stack else val
        self.stack.append((val, current_min))

    def getMin(self):
        return self.stack[-1][1]  # O(1)


# ❌ WRONG: HashMap - Not handling collisions
class MyHashMapWrong:
    def __init__(self):
        self.buckets = [None] * 1000

    def put(self, key, value):
        self.buckets[key % 1000] = value  # Overwrites on collision!

# ✅ CORRECT: Use chaining for collisions
class MyHashMapCorrect:
    def __init__(self):
        self.buckets = [[] for _ in range(1000)]

    def put(self, key, value):
        bucket = self.buckets[key % 1000]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))


# ❌ WRONG: LFU Cache - Not handling min_freq update
class LFUCacheWrong:
    def _update_freq(self, key):
        freq = self.key_to_freq[key]
        del self.freq_to_keys[freq][key]
        self.freq_to_keys[freq + 1][key] = None
        self.key_to_freq[key] = freq + 1
        # Forgot to update min_freq!

# ✅ CORRECT: Update min_freq when frequency bucket becomes empty
class LFUCacheCorrect:
    def _update_freq(self, key):
        freq = self.key_to_freq[key]
        del self.freq_to_keys[freq][key]

        # Update min_freq if this was the only key at min frequency
        if not self.freq_to_keys[freq] and self.min_freq == freq:
            self.min_freq = freq + 1

        self.freq_to_keys[freq + 1][key] = None
        self.key_to_freq[key] = freq + 1
```

---

## Interview Tips

### 1. How to Approach Design Problems
```
Step 1: CLARIFY requirements
   - What operations are needed?
   - What are the time complexity requirements?
   - What are the constraints (capacity, range)?

Step 2: IDENTIFY data structure needs
   - O(1) lookup → HashMap
   - O(1) order → Linked List
   - O(1) random → Array
   - O(log n) priority → Heap

Step 3: COMBINE structures
   - Think about what each structure provides
   - How can they complement each other?

Step 4: HANDLE edge cases
   - Empty state
   - Capacity limits
   - Duplicate operations
```

### 2. What Interviewers Look For
- **Clear API design**: Well-named methods, clear parameters
- **Time/Space analysis**: Know the complexity of each operation
- **Trade-offs**: Why this combination of data structures?
- **Edge case handling**: Empty, full, duplicate, concurrent

### 3. Common Follow-up Questions
- "What if we need thread safety?" → Add locks, use concurrent structures
- "Can you optimize space?" → Trade time for space or vice versa
- "What if capacity is very large?" → Consider distributed design
- "How would you test this?" → Unit tests, edge cases, stress tests

### 4. Key Patterns to Memorize

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DESIGN PATTERN CHEAT SHEET                                                 │
│                                                                             │
│  LRU Cache:                                                                 │
│  ┌──────────┐     ┌─────────────────────────────────────┐                  │
│  │ HashMap  │────▶│  Doubly Linked List                 │                  │
│  │ key→node │     │  head ←→ node ←→ node ←→ tail       │                  │
│  └──────────┘     └─────────────────────────────────────┘                  │
│  • get(): HashMap lookup + move to tail                                     │
│  • put(): HashMap insert + add to tail + evict head if full                │
│                                                                             │
│  RandomizedSet:                                                             │
│  ┌──────────┐     ┌─────────────────────────────────────┐                  │
│  │ HashMap  │────▶│  Array [val1, val2, val3, ...]      │                  │
│  │ val→idx  │     └─────────────────────────────────────┘                  │
│  • insert(): Append to array + add to map                                   │
│  • remove(): Swap with last + pop + update map                              │
│  • random(): random.choice(array)                                           │
│                                                                             │
│  Median Finder:                                                             │
│  ┌──────────────┐     ┌──────────────┐                                     │
│  │  Max Heap    │     │  Min Heap    │                                     │
│  │ (small half) │     │ (large half) │                                     │
│  └──────────────┘     └──────────────┘                                     │
│  • addNum(): Add to appropriate heap + rebalance                            │
│  • findMedian(): Top of heaps (avg if even count)                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Problem | Data Structures | Key Operations |
|---------|-----------------|----------------|
| LRU Cache | OrderedDict or HashMap + DLL | O(1) get/put |
| LFU Cache | Multiple HashMaps | O(1) get/put |
| Min Stack | Stack with min tracking | O(1) getMin |
| Twitter | HashMap + Heap | O(k log n) feed |
| RandomizedSet | Array + HashMap | O(1) all ops |
| Serialize Tree | String + Preorder | O(n) |
| HashMap | Array of buckets | O(1) average |

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges

## Practice More Problems

- [ ] LC #208 - Implement Trie
- [ ] LC #211 - Design Add and Search Words
- [ ] LC #295 - Find Median from Data Stream
- [ ] LC #341 - Flatten Nested List Iterator
- [ ] LC #384 - Shuffle an Array
