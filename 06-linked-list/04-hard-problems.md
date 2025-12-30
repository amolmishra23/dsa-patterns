# Linked List - Hard Problems

## Problem 1: Merge k Sorted Lists (LC #23) - Hard

- [LeetCode](https://leetcode.com/problems/merge-k-sorted-lists/)

### Video Explanation
- [NeetCode - Merge k Sorted Lists](https://www.youtube.com/watch?v=q5a5OiGbT6Q)

### Problem Statement
Merge k sorted linked lists into one sorted list.

### Examples
```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
```


### Visual Intuition
```
Merge K Sorted Lists using Min-Heap
lists = [[1,4,5], [1,3,4], [2,6]]

Min-heap stores (value, list_index, node):
Initial: [(1,0,node), (1,1,node), (2,2,node)]

Pop (1,0): result=[1], push (4,0,next)
  heap: [(1,1), (2,2), (4,0)]

Pop (1,1): result=[1,1], push (3,1,next)
  heap: [(2,2), (3,1), (4,0)]

Pop (2,2): result=[1,1,2], push (6,2,next)
  heap: [(3,1), (4,0), (6,2)]

Continue until heap empty...
Result: 1→1→2→3→4→4→5→6

Time: O(N log K) where N=total nodes, K=lists
```

### Solution
```python
import heapq
from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def mergeKLists(lists: list[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Merge k sorted linked lists using min-heap.

    Strategy:
    - Use min-heap to always get smallest element across all lists
    - Push first element of each list to heap
    - Pop smallest, add to result, push next element from that list

    Time: O(N log k) where N = total nodes, k = number of lists
    Space: O(k) for heap
    """
    # Min-heap: (value, list_index, node)
    # list_index used as tiebreaker (nodes aren't comparable)
    heap = []

    # Initialize heap with first node of each list
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))

    # Dummy head for result list
    dummy = ListNode(0)
    current = dummy

    while heap:
        val, idx, node = heapq.heappop(heap)

        # Add to result
        current.next = node
        current = current.next

        # Push next node from same list
        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))

    return dummy.next


def mergeKLists_divide_conquer(lists: list[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Alternative: Divide and conquer approach.

    Recursively merge pairs of lists.

    Time: O(N log k)
    Space: O(log k) for recursion
    """
    if not lists:
        return None

    def merge_two(l1, l2):
        """Merge two sorted lists."""
        dummy = ListNode(0)
        current = dummy

        while l1 and l2:
            if l1.val <= l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next

        current.next = l1 if l1 else l2
        return dummy.next

    # Merge pairs until one list remains
    while len(lists) > 1:
        merged = []
        for i in range(0, len(lists), 2):
            l1 = lists[i]
            l2 = lists[i + 1] if i + 1 < len(lists) else None
            merged.append(merge_two(l1, l2))
        lists = merged

    return lists[0]
```

### Edge Cases
- Empty list array → return None
- Single list → return that list
- Lists with different lengths → heap handles naturally
- All empty lists → return None

---

## Problem 2: Reverse Nodes in k-Group (LC #25) - Hard

- [LeetCode](https://leetcode.com/problems/reverse-nodes-in-k-group/)

### Video Explanation
- [NeetCode - Reverse Nodes in k-Group](https://www.youtube.com/watch?v=1UOPsfP85V4)

### Problem Statement
Reverse nodes in groups of k. Remaining nodes stay as-is.

### Examples
```
Input: head = [1,2,3,4,5], k = 2
Output: [2,1,4,3,5]

Input: head = [1,2,3,4,5], k = 3
Output: [3,2,1,4,5]
```


### Visual Intuition
```
Reverse Nodes in k-Group (k=3)
1 → 2 → 3 → 4 → 5

Step 1: Check if k nodes exist
  [1 → 2 → 3] → 4 → 5  (3 nodes ✓)

Step 2: Reverse first k nodes
  Before: prev=null, curr=1
  1 ← 2 ← 3    4 → 5
      ↑ new head

Step 3: Connect and recurse
  3 → 2 → 1 → [reverse(4→5)]
            ↓
  3 → 2 → 1 → 4 → 5  (only 2 nodes, keep as-is)

Result: 3 → 2 → 1 → 4 → 5
```

### Solution
```python
def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Reverse nodes in k-groups.

    Strategy:
    1. Check if k nodes available
    2. Reverse k nodes
    3. Connect reversed group to rest
    4. Recursively process remaining nodes

    Time: O(n)
    Space: O(n/k) for recursion
    """
    # Check if k nodes available
    count = 0
    node = head
    while node and count < k:
        node = node.next
        count += 1

    if count < k:
        # Not enough nodes - return as-is
        return head

    # Reverse k nodes
    prev = None
    current = head
    for _ in range(k):
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    # prev is now head of reversed group
    # current is head of remaining list
    # head is now tail of reversed group

    # Recursively process remaining and connect
    head.next = reverseKGroup(current, k)

    return prev


def reverseKGroup_iterative(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Iterative version.

    Time: O(n)
    Space: O(1)
    """
    def get_kth(node, k):
        """Get k-th node from current, or None if not enough."""
        while node and k > 1:
            node = node.next
            k -= 1
        return node

    def reverse(start, end):
        """Reverse nodes from start to end (exclusive)."""
        prev = end
        current = start
        while current != end:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        return prev

    dummy = ListNode(0)
    dummy.next = head
    group_prev = dummy

    while True:
        # Find k-th node
        kth = get_kth(group_prev.next, k)
        if not kth:
            break

        # Save next group's start
        group_next = kth.next

        # Reverse current group
        new_head = reverse(group_prev.next, group_next)

        # Connect to previous part
        old_head = group_prev.next
        group_prev.next = new_head

        # Move group_prev to end of reversed group
        group_prev = old_head

    return dummy.next
```

### Edge Cases
- k = 1 → no change needed
- k > n → return as-is
- k = n → reverse entire list
- Single node → return as-is

---

## Problem 3: Copy List with Random Pointer (LC #138) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/copy-list-with-random-pointer/)

### Video Explanation
- [NeetCode - Copy List with Random Pointer](https://www.youtube.com/watch?v=5Y2EiZST97Y)

### Problem Statement
Deep copy list where each node has a random pointer.


### Visual Intuition
```
Copy List with Random Pointer

Original:  1 → 2 → 3
           ↓   ↓   ↓
          [3] [1] [2]  (random pointers)

Step 1: Interleave copies
  1 → 1' → 2 → 2' → 3 → 3'

Step 2: Set random pointers
  1.random = 3  → 1'.random = 3.next = 3'
  2.random = 1  → 2'.random = 1.next = 1'
  3.random = 2  → 3'.random = 2.next = 2'

Step 3: Separate lists
  Original: 1 → 2 → 3
  Copy:     1' → 2' → 3'

O(1) space by interleaving!
```

### Solution
```python
class Node:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random


def copyRandomList(head: Optional[Node]) -> Optional[Node]:
    """
    Deep copy list with random pointers using O(1) space.

    Strategy (Interleaving):
    1. Insert copy after each original: A -> A' -> B -> B' -> ...
    2. Set random pointers for copies
    3. Separate into two lists

    Time: O(n)
    Space: O(1)
    """
    if not head:
        return None

    # Step 1: Create copy nodes and interleave
    current = head
    while current:
        copy = Node(current.val)
        copy.next = current.next
        current.next = copy
        current = copy.next

    # Step 2: Set random pointers for copies
    current = head
    while current:
        if current.random:
            # Copy's random = original's random's copy (next node)
            current.next.random = current.random.next
        current = current.next.next

    # Step 3: Separate lists
    dummy = Node(0)
    copy_current = dummy
    current = head

    while current:
        # Extract copy
        copy_current.next = current.next
        copy_current = copy_current.next

        # Restore original's next
        current.next = current.next.next
        current = current.next

    return dummy.next


def copyRandomList_hashmap(head: Optional[Node]) -> Optional[Node]:
    """
    Alternative: Hash map approach.

    Time: O(n)
    Space: O(n)
    """
    if not head:
        return None

    # Map: original node -> copy node
    node_map = {}

    # First pass: create all copy nodes
    current = head
    while current:
        node_map[current] = Node(current.val)
        current = current.next

    # Second pass: set next and random pointers
    current = head
    while current:
        copy = node_map[current]
        copy.next = node_map.get(current.next)
        copy.random = node_map.get(current.random)
        current = current.next

    return node_map[head]
```

### Edge Cases
- Empty list → return None
- No random pointers → simple copy
- Random points to self → handle correctly
- Cycle via random → not possible (random is pointer)

---

## Problem 4: LRU Cache (LC #146) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/lru-cache/)

### Video Explanation
- [NeetCode - LRU Cache](https://www.youtube.com/watch?v=7ABFKPK2hD4)

### Problem Statement
Implement LRU cache with O(1) get and put.


### Visual Intuition
```
LRU Cache - HashMap + Doubly Linked List
capacity = 2

Operations:
put(1,1): cache={1:1}, list: [1]
put(2,2): cache={1:1,2:2}, list: [2,1]
get(1):   return 1, move to front: [1,2]
put(3,3): evict LRU(2), cache={1:1,3:3}, list: [3,1]
get(2):   return -1 (evicted)

Structure:
  HEAD ←→ [MRU] ←→ ... ←→ [LRU] ←→ TAIL
           ↑ most recent    ↑ evict this

HashMap: key → node (O(1) lookup)
DLL: O(1) add/remove with node reference
```

### Solution
```python
class LRUCache:
    """
    LRU Cache using doubly linked list + hash map.

    - Hash map: key -> node for O(1) lookup
    - Doubly linked list: for O(1) add/remove
    - Most recently used at tail, LRU at head

    Time: O(1) for get and put
    Space: O(capacity)
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

        # Dummy head and tail
        self.head = self.Node()  # LRU side
        self.tail = self.Node()  # MRU side
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
        """
        Get value for key, mark as recently used.
        """
        if key not in self.cache:
            return -1

        node = self.cache[key]

        # Move to end (most recently used)
        self._remove(node)
        self._add_to_end(node)

        return node.val

    def put(self, key: int, value: int) -> None:
        """
        Put key-value pair, evict LRU if at capacity.
        """
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
- Capacity = 1 → always evict on new put
- Get non-existent key → return -1
- Update existing key → don't evict
- Multiple gets → updates recency

---

## Problem 5: Flatten a Multilevel Doubly Linked List (LC #430) - Medium/Hard

- [LeetCode](https://leetcode.com/problems/flatten-a-multilevel-doubly-linked-list/)

### Video Explanation
- [NeetCode - Flatten a Multilevel Doubly Linked List](https://www.youtube.com/watch?v=RIyPgR7AF7M)

### Problem Statement
Flatten a multilevel doubly linked list with child pointers.


### Visual Intuition
```
Flatten Multilevel Doubly Linked List

Input:
  1 ←→ 2 ←→ 3 ←→ 4 ←→ 5
            ↓
            7 ←→ 8 ←→ 9
                 ↓
                 11

DFS approach - when child exists:
1. Save next (4)
2. Connect current → child: 3 → 7
3. Recursively flatten child level
4. Find tail of flattened child (9)
5. Connect tail → saved next: 9 → 4

Result:
  1 ←→ 2 ←→ 3 ←→ 7 ←→ 8 ←→ 11 ←→ 9 ←→ 4 ←→ 5
```

### Solution
```python
class Node:
    def __init__(self, val=0, prev=None, next=None, child=None):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child


def flatten(head: Optional[Node]) -> Optional[Node]:
    """
    Flatten multilevel doubly linked list.

    Strategy:
    - When encountering child, insert child list between current and next
    - Recursively flatten child list
    - Clear child pointer

    Time: O(n)
    Space: O(depth) for recursion
    """
    if not head:
        return None

    current = head

    while current:
        if current.child:
            # Save next node
            next_node = current.next

            # Flatten child list
            child_head = flatten(current.child)

            # Connect current to child
            current.next = child_head
            child_head.prev = current
            current.child = None

            # Find tail of child list
            child_tail = child_head
            while child_tail.next:
                child_tail = child_tail.next

            # Connect child tail to next node
            if next_node:
                child_tail.next = next_node
                next_node.prev = child_tail

        current = current.next

    return head


def flatten_iterative(head: Optional[Node]) -> Optional[Node]:
    """
    Iterative version using stack.

    Time: O(n)
    Space: O(n) for stack
    """
    if not head:
        return None

    stack = []
    current = head

    while current:
        if current.child:
            # Save next for later
            if current.next:
                stack.append(current.next)

            # Process child
            current.next = current.child
            current.child.prev = current
            current.child = None

        if not current.next and stack:
            # Get next from stack
            next_node = stack.pop()
            current.next = next_node
            next_node.prev = current

        current = current.next

    return head
```

### Edge Cases
- No children → return as-is
- Single node with child → flatten child
- Deep nesting → recursion handles it
- Empty list → return None

---

## Summary: Hard Linked List Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Merge k Sorted | Min-heap or divide & conquer | O(N log k) |
| 2 | Reverse k-Group | Count + reverse + connect | O(n) |
| 3 | Copy Random List | Interleaving or hash map | O(n) |
| 4 | LRU Cache | HashMap + Doubly Linked List | O(1) |
| 5 | Flatten Multilevel | Recursion or stack | O(n) |

---

## Key Patterns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HARD LINKED LIST PATTERNS                                │
│                                                                             │
│  1. MERGE K LISTS: Use min-heap for O(log k) selection                      │
│                                                                             │
│  2. REVERSE IN GROUPS: Count first, reverse, connect, recurse               │
│                                                                             │
│  3. RANDOM POINTERS: Interleave copies or use hash map                      │
│                                                                             │
│  4. O(1) OPERATIONS: Combine hash map + doubly linked list                  │
│                                                                             │
│  5. MULTILEVEL: Use stack or recursion for depth                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Practice More Problems

- [ ] LC #460 - LFU Cache
- [ ] LC #432 - All O'one Data Structure
- [ ] LC #1206 - Design Skiplist

