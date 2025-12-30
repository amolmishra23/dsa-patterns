# Design Problems - Practice Problems

## Problem 1: LRU Cache (LC #146) - Medium

- [LeetCode](https://leetcode.com/problems/lru-cache/)

### Problem Statement
Design a data structure that follows LRU (Least Recently Used) eviction policy.

### Examples
```
LRUCache cache = new LRUCache(2);
cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
```

### Video Explanation
- [NeetCode - LRU Cache](https://www.youtube.com/watch?v=7ABFKPK2hD4)

### Intuition
```
Hash Map + Doubly Linked List = O(1) everything!

Hash Map: key → node (O(1) lookup)
Doubly Linked List: maintains order (O(1) add/remove)

Visual: capacity = 2

        put(1,1): [1]         map: {1:node1}
        put(2,2): [1,2]       map: {1:node1, 2:node2}
        get(1):   [2,1]       move 1 to end (most recent)
        put(3,3): [1,3]       evict 2 (LRU), add 3
        get(2):   -1          not found (evicted)

        Doubly Linked List:
        head ←→ node2 ←→ node1 ←→ tail
        (LRU)            (MRU)

        On access: move node to tail
        On eviction: remove from head
```

### Solution
```python
from collections import OrderedDict

class LRUCache:
    """
    LRU Cache using OrderedDict.

    OrderedDict maintains insertion order and provides:
    - O(1) access by key
    - O(1) move_to_end() to mark as recently used
    - O(1) popitem(last=False) to remove LRU

    Time: O(1) for get and put
    Space: O(capacity)
    """

    def __init__(self, capacity: int):
        """Initialize cache with given capacity."""
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        """
        Get value for key.

        If found, move to end (mark as recently used).
        If not found, return -1.
        """
        if key not in self.cache:
            return -1

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        """
        Put key-value pair.

        If key exists, update and move to end.
        If at capacity, evict LRU (first item) before adding.
        """
        if key in self.cache:
            # Update existing - move to end
            self.cache.move_to_end(key)

        self.cache[key] = value

        # Evict LRU if over capacity
        if len(self.cache) > self.capacity:
            # popitem(last=False) removes first (oldest) item
            self.cache.popitem(last=False)


class LRUCacheManual:
    """
    LRU Cache with manual doubly linked list.

    Demonstrates the underlying data structure:
    - Hash map: key -> node for O(1) lookup
    - Doubly linked list: for O(1) add/remove
    """

    class Node:
        """Doubly linked list node."""
        def __init__(self, key=0, val=0):
            self.key = key
            self.val = val
            self.prev = None
            self.next = None

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key -> Node

        # Dummy head and tail for easier boundary handling
        self.head = self.Node()  # Dummy head (LRU side)
        self.tail = self.Node()  # Dummy tail (MRU side)
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node: 'Node') -> None:
        """Remove node from linked list."""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_to_end(self, node: 'Node') -> None:
        """Add node right before tail (most recently used)."""
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        node = self.cache[key]

        # Move to end (most recently used)
        self._remove(node)
        self._add_to_end(node)

        return node.val

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # Update existing
            node = self.cache[key]
            node.val = value
            self._remove(node)
            self._add_to_end(node)
        else:
            # Add new
            node = self.Node(key, value)
            self.cache[key] = node
            self._add_to_end(node)

            # Evict if over capacity
            if len(self.cache) > self.capacity:
                # Remove LRU (node after head)
                lru = self.head.next
                self._remove(lru)
                del self.cache[lru.key]
```

### Edge Cases
- Capacity 1 → every put evicts previous
- Get non-existent key → return -1
- Put same key twice → update value, move to recent
- Get moves to recent → affects eviction order
- Empty cache get → return -1

---

## Problem 2: Min Stack (LC #155) - Medium

- [LeetCode](https://leetcode.com/problems/min-stack/)

### Problem Statement
Design a stack that supports push, pop, top, and getMin in O(1).

### Video Explanation
- [NeetCode - Min Stack](https://www.youtube.com/watch?v=qkLl7nAwDPo)

### Intuition
```
Store (value, min_at_this_point) pairs!

Each entry remembers the minimum when it was pushed.
When we pop, we restore the previous minimum automatically.

Visual: push(3), push(5), push(2), push(1)

        Stack: [(3,3), (5,3), (2,2), (1,1)]
                       ↑      ↑      ↑
                      min    min    min

        getMin() → 1 (top's min)
        pop() → removes (1,1)
        Stack: [(3,3), (5,3), (2,2)]
        getMin() → 2 (top's min)

Alternative: Use two stacks
- Main stack: values
- Min stack: minimums (push only when new min)
```

### Solution
```python
class MinStack:
    """
    Stack with O(1) minimum retrieval.

    Strategy: Store (value, current_min) pairs.
    Each entry remembers the minimum at that point in time.

    Time: O(1) for all operations
    Space: O(n)
    """

    def __init__(self):
        """Initialize empty stack."""
        self.stack = []  # [(value, min_at_this_point), ...]

    def push(self, val: int) -> None:
        """Push value onto stack."""
        if not self.stack:
            # First element - it's the minimum
            self.stack.append((val, val))
        else:
            # Compare with current minimum
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))

    def pop(self) -> None:
        """Remove top element."""
        self.stack.pop()

    def top(self) -> int:
        """Return top element."""
        return self.stack[-1][0]

    def getMin(self) -> int:
        """Return minimum element in stack."""
        return self.stack[-1][1]


class MinStackTwoStacks:
    """
    Alternative: Use separate min stack.

    More space efficient when many duplicates.
    """

    def __init__(self):
        self.stack = []
        self.min_stack = []  # Only stores when min changes

    def push(self, val: int) -> None:
        self.stack.append(val)

        # Push to min_stack if empty or new minimum
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        val = self.stack.pop()

        # Pop from min_stack if it was the minimum
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```

### Edge Cases
- Single element → it's the min
- Push smaller → new min
- Pop the min → restore previous min
- All same values → min stays same
- Empty stack operations → undefined (assume valid input)

---

## Problem 3: Insert Delete GetRandom O(1) (LC #380) - Medium

- [LeetCode](https://leetcode.com/problems/insert-delete-getrandom-o1/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Design a data structure with O(1) insert, remove, and getRandom.


### Intuition
```
Key insight: Combine array + hashmap for O(1) on all operations.

- Array: O(1) random access (pick random index)
- HashMap: value → index for O(1) lookup/delete

Deletion trick: Swap element with last, then pop.
This avoids O(n) shift when removing from middle.

Example: Remove 'B' from [A, B, C]
1. Swap B with C: [A, C, B]
2. Pop last: [A, C]
3. Update index_map[C] = 1
```

### Solution
```python
import random

class RandomizedSet:
    """
    Set with O(1) insert, remove, and getRandom.

    Strategy:
    - List: for O(1) random access
    - Hash map: value -> index for O(1) lookup
    - Swap with last before removal to maintain O(1)

    Time: O(1) for all operations
    Space: O(n)
    """

    def __init__(self):
        self.list = []
        self.index_map = {}  # value -> index in list

    def insert(self, val: int) -> bool:
        """
        Insert value if not present.

        Returns True if inserted, False if already exists.
        """
        if val in self.index_map:
            return False

        # Add to end of list
        self.index_map[val] = len(self.list)
        self.list.append(val)
        return True

    def remove(self, val: int) -> bool:
        """
        Remove value if present.

        Strategy: Swap with last element, then pop.
        This maintains O(1) removal.

        Returns True if removed, False if not found.
        """
        if val not in self.index_map:
            return False

        # Get index of value to remove
        idx = self.index_map[val]
        last_val = self.list[-1]

        # Swap with last element
        self.list[idx] = last_val
        self.index_map[last_val] = idx

        # Remove last element
        self.list.pop()
        del self.index_map[val]

        return True

    def getRandom(self) -> int:
        """Return random element."""
        return random.choice(self.list)
```

### Edge Cases
- Insert duplicate → return False
- Remove non-existent → return False
- GetRandom single element → return that element
- Remove last element → list becomes empty
- Insert after remove → index map updated

---

## Problem 4: Design Twitter (LC #355) - Medium

- [LeetCode](https://leetcode.com/problems/design-twitter/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Design a simplified Twitter with postTweet, getNewsFeed, follow, unfollow.


### Intuition
```
Key insight: Use heap to merge k sorted lists (each user's tweets).

Data structures:
- tweets: userId → list of (timestamp, tweetId)
- following: userId → set of followeeIds
- Global timestamp for ordering

getNewsFeed: Merge tweets from user + followees.
Use min-heap of size 10 or merge k sorted lists.

Each user follows themselves implicitly for their own feed.
```

### Solution
```python
import heapq
from collections import defaultdict

class Twitter:
    """
    Simplified Twitter design.

    Data structures:
    - tweets: userId -> [(timestamp, tweetId), ...]
    - following: userId -> set of followeeIds
    - timestamp: global counter for ordering

    Time:
    - postTweet: O(1)
    - getNewsFeed: O(k log k) where k = total tweets from followees
    - follow/unfollow: O(1)
    """

    def __init__(self):
        self.timestamp = 0
        self.tweets = defaultdict(list)    # userId -> [(time, tweetId), ...]
        self.following = defaultdict(set)  # userId -> set of followeeIds

    def postTweet(self, userId: int, tweetId: int) -> None:
        """Post a new tweet."""
        self.tweets[userId].append((self.timestamp, tweetId))
        self.timestamp += 1

    def getNewsFeed(self, userId: int) -> list[int]:
        """
        Get 10 most recent tweets from user and followees.

        Strategy: Merge k sorted lists using min-heap.
        """
        # Get all users to fetch tweets from (self + followees)
        users = self.following[userId] | {userId}

        # Min-heap: (-timestamp, tweetId, userId, tweetIndex)
        # Use negative timestamp for max-heap behavior
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
        if followerId != followeeId:  # Can't follow yourself
            self.following[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        """User unfollows another user."""
        self.following[followerId].discard(followeeId)
```

### Edge Cases
- No tweets → return []
- No followees → only own tweets
- Follow self → ignored
- Unfollow non-followed → no error
- More than 10 tweets → return only 10 most recent

---

## Problem 5: Serialize and Deserialize Binary Tree (LC #297) - Hard

- [LeetCode](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Design algorithm to serialize and deserialize a binary tree.


### Intuition
```
Key insight: Preorder traversal with null markers uniquely defines a tree.

Serialize: DFS preorder, use 'N' for null nodes.
Tree [1,2,3] → "1,2,N,N,3,N,N"

Deserialize: Consume values in same preorder.
- If 'N': return null
- Else: create node, recursively build left then right

Why preorder works: Root comes first, then we know exactly
where left subtree ends (when we hit its nulls).

Alternative: BFS level-order (more intuitive for visualization).
```

### Solution
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Codec:
    """
    Serialize/deserialize binary tree using preorder traversal.

    Strategy:
    - Serialize: Preorder traversal, use 'N' for null nodes
    - Deserialize: Build tree from preorder sequence

    Time: O(n)
    Space: O(n)
    """

    def serialize(self, root: TreeNode) -> str:
        """
        Encode tree to string.

        Use preorder traversal with null markers.
        Format: "val,val,N,N,val,N,N"
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

    def deserialize(self, data: str) -> TreeNode:
        """
        Decode string to tree.

        Use iterator to consume values in preorder.
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


class CodecBFS:
    """
    Alternative: Level-order (BFS) serialization.

    More intuitive for visualization.
    """

    def serialize(self, root: TreeNode) -> str:
        if not root:
            return ''

        from collections import deque

        result = []
        queue = deque([root])

        while queue:
            node = queue.popleft()

            if node:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append('N')

        return ','.join(result)

    def deserialize(self, data: str) -> TreeNode:
        if not data:
            return None

        from collections import deque

        values = data.split(',')
        root = TreeNode(int(values[0]))
        queue = deque([root])
        i = 1

        while queue and i < len(values):
            node = queue.popleft()

            # Left child
            if values[i] != 'N':
                node.left = TreeNode(int(values[i]))
                queue.append(node.left)
            i += 1

            # Right child
            if i < len(values) and values[i] != 'N':
                node.right = TreeNode(int(values[i]))
                queue.append(node.right)
            i += 1

        return root
```

### Edge Cases
- Empty tree → serialize to empty/null marker
- Single node → "val,N,N"
- Skewed tree → many null markers
- Complete tree → minimal null markers
- Negative values → handle in parsing

---

## Problem 6: Design HashMap (LC #706) - Easy

- [LeetCode](https://leetcode.com/problems/design-hashmap/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Design a HashMap without using built-in hash table libraries.


### Intuition
```
Key insight: Hash function + collision handling (chaining or open addressing).

Components:
1. Hash function: key % bucket_size (simple modulo)
2. Bucket array: fixed size array of linked lists
3. Chaining: each bucket holds list of (key, value) pairs

Operations:
- put: hash → bucket → append or update
- get: hash → bucket → search list
- remove: hash → bucket → remove from list

Choose bucket_size as prime for better distribution.
```

### Solution
```python
class MyHashMap:
    """
    HashMap using array of buckets with chaining.

    Strategy:
    - Array of buckets (lists)
    - Hash function: key % num_buckets
    - Chaining: each bucket is a list of (key, value) pairs

    Time: O(n/k) average, O(n) worst case
    Space: O(k + n) where k = buckets, n = elements
    """

    def __init__(self):
        """Initialize with prime number of buckets for better distribution."""
        self.size = 1000
        self.buckets = [[] for _ in range(self.size)]

    def _hash(self, key: int) -> int:
        """Hash function: simple modulo."""
        return key % self.size

    def put(self, key: int, value: int) -> None:
        """Add or update key-value pair."""
        bucket_idx = self._hash(key)
        bucket = self.buckets[bucket_idx]

        # Check if key exists - update if so
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        # Key doesn't exist - add new pair
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
        """Remove key if present."""
        bucket_idx = self._hash(key)
        bucket = self.buckets[bucket_idx]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                return
```

### Edge Cases
- Get non-existent key → return -1
- Put same key twice → update value
- Remove non-existent → no error
- Hash collision → chaining handles
- Empty map get → return -1

---

## Problem 7: Find Median from Data Stream (LC #295) - Hard

- [LeetCode](https://leetcode.com/problems/find-median-from-data-stream/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Design a data structure that supports adding numbers and finding median.


### Intuition
```
Key insight: Two heaps - max-heap for smaller half, min-heap for larger.

Maintain invariant: max_heap.size >= min_heap.size (differ by at most 1)

           max_heap    |    min_heap
        (smaller half) | (larger half)
           [1,2,3]     |    [4,5,6]
              ↑               ↑
           max=3           min=4

Median:
- Odd count: top of max_heap (the extra element)
- Even count: average of both tops

addNum: Add to max_heap, rebalance through min_heap.
```

### Solution
```python
import heapq

class MedianFinder:
    """
    Find median from data stream using two heaps.

    Strategy:
    - max_heap: stores smaller half (use negation for max behavior)
    - min_heap: stores larger half
    - Keep heaps balanced (size difference <= 1)

    Median:
    - If equal size: average of tops
    - If unequal: top of larger heap

    Time: O(log n) addNum, O(1) findMedian
    Space: O(n)
    """

    def __init__(self):
        # max_heap for smaller half (store negatives)
        self.max_heap = []  # Stores negatives for max behavior

        # min_heap for larger half
        self.min_heap = []

    def addNum(self, num: int) -> None:
        """
        Add number to data structure.

        Strategy:
        1. Add to max_heap (smaller half)
        2. Balance: move max of smaller half to larger half
        3. Rebalance if needed: move min of larger to smaller
        """
        # Add to max_heap (negate for max behavior)
        heapq.heappush(self.max_heap, -num)

        # Balance: move largest of smaller half to larger half
        heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))

        # Rebalance if min_heap is larger
        if len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def findMedian(self) -> float:
        """
        Return median of all numbers.

        If equal size: average of two middle elements
        If unequal: middle element (from larger heap)
        """
        if len(self.max_heap) > len(self.min_heap):
            # Odd total: max_heap has extra element
            return -self.max_heap[0]
        else:
            # Even total: average of tops
            return (-self.max_heap[0] + self.min_heap[0]) / 2
```

### Edge Cases
- Single element → return that element
- Two elements → return average
- All same values → return that value
- Sorted input → heaps rebalance
- Large stream → O(log n) per add

---

## Problem 8: Implement Trie (LC #208) - Medium

- [LeetCode](https://leetcode.com/problems/implement-trie-prefix-tree/)


### Video Explanation
- [NeetCode](https://www.youtube.com/c/NeetCode) - Search for problem name

### Problem Statement
Implement a trie with insert, search, and startsWith.


### Intuition
```
Key insight: Tree where each node represents a character, path = prefix.

Structure:
- Each node has children map (char → node) and is_end flag
- Root is empty, words branch from root

        root
       / | \
      a  b  c
     /       \
    p         a
   / \         \
  p   e        t  (is_end=true)
 (is_end)

insert: Create nodes along path, mark last as word end.
search: Follow path, check is_end at final node.
startsWith: Follow path, return true if path exists.
```

### Solution
```python
class TrieNode:
    """Node in Trie."""
    def __init__(self):
        self.children = {}   # char -> TrieNode
        self.is_end = False  # True if word ends here


class Trie:
    """
    Trie (Prefix Tree) implementation.

    Time: O(L) for all operations where L = word length
    Space: O(total characters)
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """Insert word into trie."""
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        node.is_end = True

    def search(self, word: str) -> bool:
        """Check if word exists in trie."""
        node = self._find_node(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        """Check if any word starts with prefix."""
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str) -> TrieNode:
        """Find node at end of prefix path."""
        node = self.root

        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]

        return node
```

### Edge Cases
- Empty string → insert at root
- Search empty → check root.is_end
- Prefix of word → startsWith true, search false
- Word is prefix → both return true
- Same word twice → idempotent insert

---

## Summary: Design Problems

| # | Problem | Key Data Structures | Time |
|---|---------|---------------------|------|
| 1 | LRU Cache | OrderedDict / HashMap + DLL | O(1) |
| 2 | Min Stack | Stack with min tracking | O(1) |
| 3 | RandomizedSet | Array + HashMap | O(1) |
| 4 | Twitter | HashMap + Heap | O(k log k) |
| 5 | Serialize Tree | String + Traversal | O(n) |
| 6 | HashMap | Array of buckets | O(1) avg |
| 7 | Median Finder | Two heaps | O(log n) |
| 8 | Trie | Tree of nodes | O(L) |

---

## Practice More Problems

- [ ] LC #146 - LRU Cache
- [ ] LC #460 - LFU Cache
- [ ] LC #341 - Flatten Nested List Iterator
- [ ] LC #384 - Shuffle an Array
- [ ] LC #705 - Design HashSet

