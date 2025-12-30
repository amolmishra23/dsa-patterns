# Design Problems - Hard Problems

## Problem 1: LFU Cache (LC #460) - Hard

- [LeetCode](https://leetcode.com/problems/lfu-cache/)

### Video Explanation
- [NeetCode - LFU Cache](https://www.youtube.com/watch?v=0PSB9y8ehbk)

### Problem Statement
Design Least Frequently Used cache.


### Visual Intuition
```
LFU Cache (Least Frequently Used)
capacity = 2

Pattern: Frequency Buckets + LRU within Each Bucket
Why: Evict least frequent, break ties by LRU

Step 0 (Data Structures):

  ┌─────────────────────────────────────────────────┐
  │ key_to_val:  {key → value}                      │
  │ key_to_freq: {key → frequency}                  │
  │ freq_to_keys: {freq → OrderedDict of keys}      │
  │ min_freq: track minimum frequency               │
  └─────────────────────────────────────────────────┘

  OrderedDict maintains insertion order for LRU

Step 1 (put(1,1)):

  key_to_val = {1: 1}
  key_to_freq = {1: 1}
  freq_to_keys = {1: {1}}
  min_freq = 1

  Freq buckets:
    freq=1: [1]
             ↑ LRU

Step 2 (put(2,2)):

  key_to_val = {1: 1, 2: 2}
  key_to_freq = {1: 1, 2: 1}
  freq_to_keys = {1: {1, 2}}
  min_freq = 1

  Freq buckets:
    freq=1: [1, 2]
             ↑    ↑
            LRU  MRU

Step 3 (get(1)):

  Access key 1 → increase frequency

  key_to_freq = {1: 2, 2: 1}
  freq_to_keys = {1: {2}, 2: {1}}
  min_freq = 1 (still has key 2)

  Freq buckets:
    freq=1: [2]
    freq=2: [1]

Step 4 (put(3,3) - Eviction!):

  Cache full (capacity=2), need to evict

  Find min_freq = 1
  Evict LRU from freq=1 bucket → key 2

  Before:
    freq=1: [2]
    freq=2: [1]

  After eviction:
    freq=1: [3]  ← new key
    freq=2: [1]

  key_to_val = {1: 1, 3: 3}
  min_freq = 1

Eviction Logic:
  ┌─────────────────────────────────────────────────┐
  │ 1. Find min_freq bucket                         │
  │ 2. Pop leftmost (LRU) from that bucket          │
  │ 3. Delete from key_to_val and key_to_freq       │
  │ 4. If bucket empty, min_freq may increase       │
  └─────────────────────────────────────────────────┘

Key Insight:
- Frequency buckets group keys by access count
- OrderedDict within bucket maintains LRU order
- min_freq tracks where to evict from
- All operations O(1) with hash maps
```

### Solution
```python
from collections import defaultdict, OrderedDict

class LFUCache:
    """
    LFU Cache using frequency buckets.

    Time: O(1) for get/put
    Space: O(capacity)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_freq = 0
        self.key_to_val = {}
        self.key_to_freq = {}
        self.freq_to_keys = defaultdict(OrderedDict)

    def _update_freq(self, key):
        """Move key to next frequency bucket."""
        freq = self.key_to_freq[key]

        # Remove from current bucket
        del self.freq_to_keys[freq][key]

        # Update min_freq if bucket is empty
        if not self.freq_to_keys[freq] and self.min_freq == freq:
            self.min_freq += 1

        # Add to next bucket
        self.key_to_freq[key] = freq + 1
        self.freq_to_keys[freq + 1][key] = None

    def get(self, key: int) -> int:
        if key not in self.key_to_val:
            return -1

        self._update_freq(key)
        return self.key_to_val[key]

    def put(self, key: int, value: int) -> None:
        if self.capacity == 0:
            return

        if key in self.key_to_val:
            self.key_to_val[key] = value
            self._update_freq(key)
            return

        # Evict if at capacity
        if len(self.key_to_val) >= self.capacity:
            # Remove LFU (leftmost in min_freq bucket)
            evict_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
            del self.key_to_val[evict_key]
            del self.key_to_freq[evict_key]

        # Add new key
        self.key_to_val[key] = value
        self.key_to_freq[key] = 1
        self.freq_to_keys[1][key] = None
        self.min_freq = 1
```

### Edge Cases
- Capacity 0 → no operations work
- Get non-existent → return -1
- Tie in frequency → evict LRU among them
- Update existing key → increase frequency

---

## Problem 2: All O(1) Data Structure (LC #432) - Hard

- [LeetCode](https://leetcode.com/problems/all-oone-data-structure/)

### Video Explanation
- [NeetCode - All O(1) Data Structure](https://www.youtube.com/watch?v=YRaT9qEBpNQ)

### Problem Statement
Design data structure with O(1) inc, dec, getMaxKey, getMinKey.


### Visual Intuition
```
All O(1) Data Structure
Operations: inc("a"), inc("a"), inc("b"), dec("a")

Pattern: Doubly Linked List of Count Buckets
Why: O(1) access to min/max count buckets

Step 0 (Data Structure):

  Doubly linked list of buckets (sorted by count):

  HEAD ←→ [bucket] ←→ [bucket] ←→ ... ←→ TAIL
           count=1     count=2

  Each bucket: {count, set of keys}
  key_to_bucket: {key → bucket reference}

Step 1 (inc("a")):

  "a" not exists → create bucket count=1

  HEAD ←→ [count=1: {a}] ←→ TAIL

  key_to_bucket = {"a": bucket1}

Step 2 (inc("a")):

  "a" exists in count=1 → move to count=2

  HEAD ←→ [count=1: {}] ←→ [count=2: {a}] ←→ TAIL
                              ↑
  Remove empty bucket:

  HEAD ←→ [count=2: {a}] ←→ TAIL

Step 3 (inc("b")):

  "b" not exists → create bucket count=1

  HEAD ←→ [count=1: {b}] ←→ [count=2: {a}] ←→ TAIL
           ↑                  ↑
          min                max

Step 4 (dec("a")):

  "a" in count=2 → move to count=1

  HEAD ←→ [count=1: {a,b}] ←→ TAIL

  Bucket count=2 empty → removed

Final State:
  ┌─────────────────────────────────────────────────┐
  │ HEAD ←→ [count=1: {a,b}] ←→ TAIL                │
  │                                                 │
  │ getMinKey() → "a" or "b" (from head.next)       │
  │ getMaxKey() → "a" or "b" (from tail.prev)       │
  │                                                 │
  │ Both return from same bucket (count=1)          │
  └─────────────────────────────────────────────────┘

Bucket Movement:

  inc(key):
    ┌─────────────────────────────────────────────┐
    │ Remove key from current bucket              │
    │ Add to next bucket (count+1)                │
    │ Create bucket if doesn't exist              │
    │ Remove empty bucket                         │
    └─────────────────────────────────────────────┘

  dec(key):
    ┌─────────────────────────────────────────────┐
    │ Remove key from current bucket              │
    │ If count=1: delete key entirely             │
    │ Else: add to prev bucket (count-1)          │
    │ Remove empty bucket                         │
    └─────────────────────────────────────────────┘

Key Insight:
- Buckets sorted by count in linked list
- Head.next = min count bucket
- Tail.prev = max count bucket
- Moving between adjacent buckets is O(1)
```

### Solution
```python
class Node:
    """Doubly linked list node for frequency bucket."""
    def __init__(self, freq=0):
        self.freq = freq
        self.keys = set()
        self.prev = None
        self.next = None

class AllOne:
    """
    Doubly linked list of frequency buckets.

    Time: O(1) for all operations
    Space: O(n)
    """

    def __init__(self):
        self.head = Node()  # Dummy head (min)
        self.tail = Node()  # Dummy tail (max)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.key_to_node = {}

    def _insert_after(self, node, new_node):
        new_node.prev = node
        new_node.next = node.next
        node.next.prev = new_node
        node.next = new_node

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def inc(self, key: str) -> None:
        if key in self.key_to_node:
            node = self.key_to_node[key]
            new_freq = node.freq + 1

            # Get or create next bucket
            if node.next.freq != new_freq:
                new_node = Node(new_freq)
                self._insert_after(node, new_node)
            else:
                new_node = node.next

            new_node.keys.add(key)
            node.keys.remove(key)
            self.key_to_node[key] = new_node

            if not node.keys:
                self._remove(node)
        else:
            # New key with freq 1
            if self.head.next.freq != 1:
                new_node = Node(1)
                self._insert_after(self.head, new_node)
            else:
                new_node = self.head.next

            new_node.keys.add(key)
            self.key_to_node[key] = new_node

    def dec(self, key: str) -> None:
        node = self.key_to_node[key]
        new_freq = node.freq - 1

        node.keys.remove(key)

        if new_freq == 0:
            del self.key_to_node[key]
        else:
            if node.prev.freq != new_freq:
                new_node = Node(new_freq)
                self._insert_after(node.prev, new_node)
            else:
                new_node = node.prev

            new_node.keys.add(key)
            self.key_to_node[key] = new_node

        if not node.keys:
            self._remove(node)

    def getMaxKey(self) -> str:
        if self.tail.prev == self.head:
            return ""
        return next(iter(self.tail.prev.keys))

    def getMinKey(self) -> str:
        if self.head.next == self.tail:
            return ""
        return next(iter(self.head.next.keys))
```

### Edge Cases
- Empty structure → getMax/getMin return ""
- Dec to 0 → remove key completely
- Multiple keys same freq → any valid for getMax/Min
- Inc new key → starts at freq 1

---

## Problem 3: Design Search Autocomplete System (LC #642) - Hard

- [LeetCode](https://leetcode.com/problems/design-search-autocomplete-system/)

### Video Explanation
- [NeetCode - Design Search Autocomplete System](https://www.youtube.com/watch?v=xVZL-A0XVUE)

### Problem Statement
Design autocomplete with sentence history and frequency.


### Visual Intuition
```
Design Search Autocomplete System
sentences = ["i love you","island","iroman","i love leetcode"]
times = [5, 3, 2, 2]

Pattern: Trie with Frequency Tracking at Each Node
Why: Each node stores all sentences passing through it

Step 0 (Build Trie):

  Each node stores: {char → child, "$" → {sentence → freq}}

              root
               |
               i ($: {"i love you":5, "island":3,
               |       "iroman":2, "i love leetcode":2})
              / \
             ' ' s
             |   |
             l   l ($: {"island":3})
             |   |
             o   a
             |   |
             v   n
             |   |
             e   d ($: {"island":3})
             |
             ' ' ($: {"i love you":5, "i love leetcode":2})

Step 1 (input('i')):

  Navigate to 'i' node
  Get all sentences: {"i love you":5, "island":3,
                      "iroman":2, "i love leetcode":2}

  Sort by: (-freq, sentence) for top 3

  Top 3: ["i love you", "island", "iroman"]
          freq=5       freq=3     freq=2

Step 2 (input(' ')):

  Current prefix: "i "
  Navigate to ' ' node (under 'i')

  Sentences: {"i love you":5, "i love leetcode":2}

  Top 3: ["i love you", "i love leetcode"]

Step 3 (input('l')):

  Current prefix: "i l"
  Navigate to 'l' node

  Sentences: {"i love you":5, "i love leetcode":2}

  Top 3: ["i love you", "i love leetcode"]

Step 4 (input('#')):

  End of input → save sentence "i l" with freq=1
  Reset prefix to ""

  (In practice, user would type full sentence before #)

Trie Node Structure:
  ┌─────────────────────────────────────────────────┐
  │ node = {                                        │
  │   'a': child_node,  # child for char 'a'        │
  │   'b': child_node,  # child for char 'b'        │
  │   ...                                           │
  │   '$': {            # sentences through here    │
  │     "sentence1": freq1,                         │
  │     "sentence2": freq2,                         │
  │   }                                             │
  │ }                                               │
  └─────────────────────────────────────────────────┘

Key Insight:
- Store sentences at EVERY node along path
- O(1) lookup for current prefix's sentences
- Heap to get top 3 by (-freq, lexicographic)
- '#' saves new sentence, updates all nodes on path
```

### Solution
```python
from collections import defaultdict
import heapq

class AutocompleteSystem:
    """
    Trie-based autocomplete with frequency tracking.

    Time: O(n) for input, O(m log 3) for suggestions
    Space: O(total chars)
    """

    def __init__(self, sentences: list[str], times: list[int]):
        self.trie = {}
        self.curr_input = ""
        self.curr_node = self.trie

        for sentence, count in zip(sentences, times):
            self._add(sentence, count)

    def _add(self, sentence: str, count: int):
        node = self.trie
        for c in sentence:
            if c not in node:
                node[c] = {"$": defaultdict(int)}
            node = node[c]
            node["$"][sentence] += count

    def input(self, c: str) -> list[str]:
        if c == "#":
            # Save current input
            self._add(self.curr_input, 1)
            self.curr_input = ""
            self.curr_node = self.trie
            return []

        self.curr_input += c

        if self.curr_node is None or c not in self.curr_node:
            self.curr_node = None
            return []

        self.curr_node = self.curr_node[c]

        # Get top 3 by frequency (desc), then lexicographically
        sentences = self.curr_node["$"]

        # Use heap: (-freq, sentence) for top 3
        heap = [(-freq, s) for s, freq in sentences.items()]
        heapq.heapify(heap)

        result = []
        for _ in range(min(3, len(heap))):
            result.append(heapq.heappop(heap)[1])

        return result
```

### Edge Cases
- No matching prefix → return []
- # character → save and reset
- Same sentence multiple times → accumulate frequency
- Top 3 with ties → sort lexicographically

---

## Problem 4: Design Twitter (LC #355) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/design-twitter/)

### Video Explanation
- [NeetCode - Design Twitter](https://www.youtube.com/watch?v=pNichitDD2E)

### Problem Statement
Design a simplified Twitter with post, follow, unfollow, and news feed.

### Visual Intuition
```
Design Twitter
Operations: postTweet, getNewsFeed, follow, unfollow

Pattern: Merge K Sorted Lists for News Feed
Why: Each user's tweets are sorted by time

Step 0 (Data Structures):

  ┌─────────────────────────────────────────────────┐
  │ tweets: {userId → [(time, tweetId), ...]}       │
  │         Most recent at end of list              │
  │                                                 │
  │ following: {userId → set of followeeIds}        │
  │            User always sees own tweets          │
  └─────────────────────────────────────────────────┘

Step 1 (Example State):

  User 1 follows: [2, 3]

  tweets:
    User 1: [(t2, 100), (t5, 101)]  ← time order
    User 2: [(t1, 200), (t4, 201)]
    User 3: [(t3, 301)]

  Timeline (sorted by time):
    t1: User 2 posts 200
    t2: User 1 posts 100
    t3: User 3 posts 301
    t4: User 2 posts 201
    t5: User 1 posts 101

Step 2 (getNewsFeed(1)):

  Sources: User 1's tweets + followees (2, 3)

  User 1: [(t2, 100), (t5, 101)]
                            ↑ start from most recent
  User 2: [(t1, 200), (t4, 201)]
                            ↑
  User 3: [(t3, 301)]
                  ↑

  Max-heap: [(-t5, 101, u1, idx1),
             (-t4, 201, u2, idx1),
             (-t3, 301, u3, idx0)]

Step 3 (Pop from Heap):

  Pop (-t5, 101, u1, idx1) → result = [101]
    Push User 1's next: (-t2, 100, u1, idx0)

  Pop (-t4, 201, u2, idx1) → result = [101, 201]
    Push User 2's next: (-t1, 200, u2, idx0)

  Pop (-t3, 301, u3, idx0) → result = [101, 201, 301]
    User 3 has no more tweets

  Pop (-t2, 100, u1, idx0) → result = [101, 201, 301, 100]
    User 1 has no more tweets

  Pop (-t1, 200, u2, idx0) → result = [101, 201, 301, 100, 200]
    User 2 has no more tweets

Final News Feed (top 10): [101, 201, 301, 100, 200]

Merge K Lists Visualization:

  User 1: 101 ← 100
            ↘
  User 2: 201 ← 200  →  [101, 201, 301, 100, 200]
            ↗           (merged by time)
  User 3: 301

Key Insight:
- Heap always has most recent unprocessed tweet from each user
- Pop gives next most recent overall
- Push next tweet from same user
- O(k log k) where k = number of users
```


### Intuition
```
Operations:
- postTweet(userId, tweetId): Post a tweet
- getNewsFeed(userId): Get 10 most recent tweets from user and followees
- follow(followerId, followeeId): Follow a user
- unfollow(followerId, followeeId): Unfollow a user

Use heap to merge tweets from multiple users efficiently.
```

### Solution
```python
import heapq
from collections import defaultdict

class Twitter:
    """
    Twitter with heap-based news feed.

    Time: O(1) post/follow/unfollow, O(n log k) getNewsFeed
    Space: O(users + tweets)
    """

    def __init__(self):
        self.time = 0
        self.tweets = defaultdict(list)    # userId -> [(time, tweetId)]
        self.following = defaultdict(set)  # userId -> set of followeeIds

    def postTweet(self, userId: int, tweetId: int) -> None:
        self.tweets[userId].append((self.time, tweetId))
        self.time += 1

    def getNewsFeed(self, userId: int) -> list[int]:
        """Get 10 most recent tweets from user and followees."""
        # Include self in feed
        users = self.following[userId] | {userId}

        # Max heap: (-time, tweetId, userId, index)
        heap = []

        for uid in users:
            if self.tweets[uid]:
                idx = len(self.tweets[uid]) - 1
                time, tweetId = self.tweets[uid][idx]
                heapq.heappush(heap, (-time, tweetId, uid, idx))

        result = []

        while heap and len(result) < 10:
            _, tweetId, uid, idx = heapq.heappop(heap)
            result.append(tweetId)

            # Add next tweet from same user
            if idx > 0:
                time, nextTweetId = self.tweets[uid][idx - 1]
                heapq.heappush(heap, (-time, nextTweetId, uid, idx - 1))

        return result

    def follow(self, followerId: int, followeeId: int) -> None:
        if followerId != followeeId:
            self.following[followerId].add(followeeId)

    def unfollow(self, followerId: int, followeeId: int) -> None:
        self.following[followerId].discard(followeeId)
```

### Complexity
- **Post/Follow/Unfollow**: O(1)
- **GetNewsFeed**: O(n log k) where n = followees, k = 10
- **Space**: O(users + tweets)

### Edge Cases
- User follows themselves → ignore
- Unfollow non-followee → no-op
- No tweets → return []
- User with no followees → own tweets only

---

## Problem 5: Design In-Memory File System (LC #588) - Hard

- [LeetCode](https://leetcode.com/problems/design-in-memory-file-system/)

### Video Explanation
- [NeetCode - Design In-Memory File System](https://www.youtube.com/watch?v=2LYrZS4sKBg)

### Problem Statement
Design in-memory file system with ls, mkdir, addContentToFile, readContentFromFile.

### Visual Intuition
```
Design In-Memory File System
mkdir("/a/b/c"), addContentToFile("/a/b/c/d", "hello")

Pattern: Trie with Directories and Files
Why: Path navigation like trie traversal

Step 0 (Node Structure):

  ┌─────────────────────────────────────────────────┐
  │ node = {                                        │
  │   "dirs": {name → child_node},  # subdirs       │
  │   "files": {name → content}     # files         │
  │ }                                               │
  └─────────────────────────────────────────────────┘

Step 1 (mkdir("/a/b/c")):

  Split path: ["a", "b", "c"]
  Create each directory if not exists:

        root (/)
          │
          a (dir)
          │
          b (dir)
          │
          c (dir)

Step 2 (addContentToFile("/a/b/c/d", "hello")):

  Navigate to /a/b/c
  Add file "d" with content "hello"

        root (/)
          │
          a (dir)
          │
          b (dir)
          │
          c (dir)
          │
          d (file) ─── content: "hello"

Step 3 (ls("/a/b")):

  Navigate to /a/b
  List contents: dirs + files (sorted)

        b (dir)
          │
          c (dir)  ← only child

  Result: ["c"]

Step 4 (ls("/a/b/c/d")):

  Path points to FILE, not directory
  Return just the filename

  Result: ["d"]

Step 5 (readContentFromFile("/a/b/c/d")):

  Navigate to parent /a/b/c
  Read file "d"

  Result: "hello"

Full Tree Visualization:

  /
  ├── a/
  │   └── b/
  │       └── c/
  │           └── d (file: "hello")
  └── x/
      └── y.txt (file: "world")

ls("/") → ["a", "x"]
ls("/a/b/c") → ["d"]
ls("/x/y.txt") → ["y.txt"]

Key Insight:
- Directories have "dirs" and "files" dicts
- Files have content string
- ls on file returns [filename]
- addContent appends to existing content
- O(path_length) for all operations
```


### Intuition
```
File system as Trie:
/
├── a/
│   └── b/
│       └── c (file: "hello")
└── d/
    └── e/
        └── f (file: "world")

Each node: {name: {children} or content}
```

### Solution
```python
class FileSystem:
    """
    Trie-based file system.

    Time: O(path_length) for all operations
    Space: O(total content)
    """

    def __init__(self):
        self.root = {"dirs": {}, "files": {}}

    def _traverse(self, path: str):
        """Navigate to directory, return node."""
        node = self.root
        if path == "/":
            return node

        parts = path.split("/")[1:]  # Skip empty first element

        for part in parts:
            if part in node["dirs"]:
                node = node["dirs"][part]
            else:
                return None

        return node

    def ls(self, path: str) -> list[str]:
        """List directory contents or file name."""
        parts = path.split("/")[1:] if path != "/" else []
        node = self.root

        # Navigate path
        for i, part in enumerate(parts):
            if part in node["files"]:
                # It's a file - return just the filename
                return [part]
            node = node["dirs"][part]

        # It's a directory - list contents
        result = list(node["dirs"].keys()) + list(node["files"].keys())
        return sorted(result)

    def mkdir(self, path: str) -> None:
        """Create directory path."""
        node = self.root
        parts = path.split("/")[1:]

        for part in parts:
            if part not in node["dirs"]:
                node["dirs"][part] = {"dirs": {}, "files": {}}
            node = node["dirs"][part]

    def addContentToFile(self, filePath: str, content: str) -> None:
        """Add content to file, create if doesn't exist."""
        parts = filePath.split("/")[1:]
        filename = parts[-1]
        dir_path = "/" + "/".join(parts[:-1]) if len(parts) > 1 else "/"

        # Create directory if needed
        self.mkdir(dir_path)

        # Navigate to parent directory
        node = self.root
        for part in parts[:-1]:
            node = node["dirs"][part]

        # Add/append content
        if filename not in node["files"]:
            node["files"][filename] = ""
        node["files"][filename] += content

    def readContentFromFile(self, filePath: str) -> str:
        """Read file content."""
        parts = filePath.split("/")[1:]
        filename = parts[-1]

        node = self.root
        for part in parts[:-1]:
            node = node["dirs"][part]

        return node["files"][filename]
```

### Complexity
- **All operations**: O(path_length + content_length)
- **Space**: O(total paths + total content)

### Edge Cases
- ls on root "/" → list all top-level
- ls on file → return just filename
- Add content to existing file → append
- mkdir existing path → no-op

---

## Problem 6: Design Skiplist (LC #1206) - Hard

- [LeetCode](https://leetcode.com/problems/design-skiplist/)

### Video Explanation
- [NeetCode - Design Skiplist](https://www.youtube.com/watch?v=UGaOXaXAM5M)

### Problem Statement
Design a skiplist with search, add, and erase operations.

### Visual Intuition
```
Design Skiplist - Probabilistic Balanced Structure

Pattern: Multi-Level Linked Lists with Express Lanes
Why: O(log n) average by skipping elements

Step 0 (Structure):

  Level 3: HEAD ──────────────────────────────→ 9 → TAIL
  Level 2: HEAD ────────→ 4 ──────────────────→ 9 → TAIL
  Level 1: HEAD ──→ 3 ──→ 4 ────→ 6 ──────────→ 9 → TAIL
  Level 0: HEAD ──→ 3 ──→ 4 ──→ 5 ──→ 6 ──→ 7 ──→ 9 → TAIL
                   ↑     ↑     ↑     ↑     ↑     ↑
                   all elements at level 0

Step 1 (Search for 6):

  Start at HEAD, highest level (3)

  L3: HEAD → 9
      6 < 9 → drop down to L2

  L2: HEAD → 4 → 9
      6 > 4 → move right to 4
      6 < 9 → drop down to L1

  L1: 4 → 6 → 9
      6 = 6 → FOUND! ✓

  Path: HEAD → 4 → 6
        (skipped 3, 5)

Step 2 (Insert 5):

  1. Search for position (like search)
  2. Generate random level (coin flips)
     P(level ≥ k) = (1/2)^k
     Say we get level = 1

  3. Update pointers at levels 0 and 1

  Before:
  L1: ... → 4 ────→ 6 → ...
  L0: ... → 4 ────→ 6 → ...

  After:
  L1: ... → 4 ──→ 5 ──→ 6 → ...
  L0: ... → 4 ──→ 5 ──→ 6 → ...

Step 3 (Delete 4):

  1. Search for 4 at each level
  2. Update all pointers that point to 4

  Before:
  L2: HEAD ────→ 4 ────→ 9
  L1: HEAD → 3 → 4 → 5 → 6
  L0: HEAD → 3 → 4 → 5 → 6

  After:
  L2: HEAD ────────────→ 9
  L1: HEAD → 3 ────→ 5 → 6
  L0: HEAD → 3 ────→ 5 → 6

Random Level Generation:
  ┌─────────────────────────────────────────────────┐
  │ level = 0                                       │
  │ while random() < 0.5 and level < MAX_LEVEL:     │
  │     level += 1                                  │
  │ return level                                    │
  │                                                 │
  │ Expected: 50% at L0, 25% at L1, 12.5% at L2... │
  └─────────────────────────────────────────────────┘

Why O(log n):
  - Each level has ~half the nodes of level below
  - Search: O(log n) levels × O(1) moves per level
  - Similar to binary search tree, but probabilistic

Key Insight:
- Higher levels = express lanes (skip more nodes)
- Random levels = probabilistic balance
- No rotations needed (unlike BST)
- Average O(log n), worst O(n) if unlucky
```


### Intuition
```
Skiplist levels (probabilistic):
Level 3: 1 -----------------> 9
Level 2: 1 -----> 4 -------> 9
Level 1: 1 -> 3 -> 4 -> 7 -> 9
Level 0: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9

Search: Start from top, move right until next > target, then down.
Average O(log n) operations.
```

### Solution
```python
import random

class SkiplistNode:
    def __init__(self, val=-1, level=0):
        self.val = val
        self.next = [None] * (level + 1)

class Skiplist:
    """
    Probabilistic skiplist.

    Time: O(log n) average for all operations
    Space: O(n)
    """

    def __init__(self):
        self.max_level = 16
        self.head = SkiplistNode(-1, self.max_level)
        self.level = 0

    def _random_level(self):
        """Generate random level with probability 1/2."""
        lvl = 0
        while random.random() < 0.5 and lvl < self.max_level:
            lvl += 1
        return lvl

    def search(self, target: int) -> bool:
        curr = self.head

        for i in range(self.level, -1, -1):
            while curr.next[i] and curr.next[i].val < target:
                curr = curr.next[i]

        curr = curr.next[0]
        return curr is not None and curr.val == target

    def add(self, num: int) -> None:
        update = [None] * (self.max_level + 1)
        curr = self.head

        # Find position at each level
        for i in range(self.level, -1, -1):
            while curr.next[i] and curr.next[i].val < num:
                curr = curr.next[i]
            update[i] = curr

        # Random level for new node
        lvl = self._random_level()

        if lvl > self.level:
            for i in range(self.level + 1, lvl + 1):
                update[i] = self.head
            self.level = lvl

        # Create and insert node
        new_node = SkiplistNode(num, lvl)

        for i in range(lvl + 1):
            new_node.next[i] = update[i].next[i]
            update[i].next[i] = new_node

    def erase(self, num: int) -> bool:
        update = [None] * (self.max_level + 1)
        curr = self.head

        for i in range(self.level, -1, -1):
            while curr.next[i] and curr.next[i].val < num:
                curr = curr.next[i]
            update[i] = curr

        curr = curr.next[0]

        if curr is None or curr.val != num:
            return False

        # Remove node from each level
        for i in range(self.level + 1):
            if update[i].next[i] != curr:
                break
            update[i].next[i] = curr.next[i]

        # Update level if needed
        while self.level > 0 and self.head.next[self.level] is None:
            self.level -= 1

        return True
```

### Complexity
- **All operations**: O(log n) average, O(n) worst
- **Space**: O(n)

### Edge Cases
- Search empty list → false
- Erase non-existent → false
- Add duplicates → allowed (multiple nodes)
- Worst case → all same level (degenerates to linked list)

---

## Summary

| # | Problem | Key Technique |
|---|---------|---------------|
| 1 | LFU Cache | Frequency buckets + OrderedDict |
| 2 | All O(1) | Doubly linked list + hash map |
| 3 | Autocomplete | Trie + frequency tracking |
| 4 | Design Twitter | Heap merge for news feed |
| 5 | File System | Trie-based directory structure |
| 6 | Skiplist | Probabilistic multi-level linked list |
