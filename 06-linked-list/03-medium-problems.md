# Linked List - Medium Problems

## Problem 1: Add Two Numbers (LC #2) - Medium

- [LeetCode](https://leetcode.com/problems/add-two-numbers/)

### Problem Statement
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in **reverse order**, and each node contains a single digit. Add the two numbers and return the sum as a linked list. You may assume the two numbers do not contain any leading zero, except the number 0 itself.

### Video Explanation
- [NeetCode - Add Two Numbers](https://www.youtube.com/watch?v=wgFPrzTjm7s)

### Examples
```
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807

Input: l1 = [0], l2 = [0]
Output: [0]

Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]
Explanation: 9999999 + 9999 = 10009998
```

### Intuition Development
```
Simulate grade-school addition digit by digit!

l1: 2 → 4 → 3  (represents 342)
l2: 5 → 6 → 4  (represents 465)
           +
──────────────

Step-by-step visualization:
┌─────────────────────────────────────────────────────────────────┐
│  Position   l1    l2   carry   sum    digit   new_carry        │
│  ────────  ────  ────  ──────  ────   ─────   ─────────        │
│     0       2     5      0      7       7        0              │
│     1       4     6      0     10       0        1              │
│     2       3     4      1      8       8        0              │
│                                                                  │
│  Result: 7 → 0 → 8  (represents 807) ✓                          │
└─────────────────────────────────────────────────────────────────┘

Different length handling:
l1: 9 → 9 → 9      (999)
l2: 9 → 9          (99)
                   +
────────────────────
    8 → 9 → 0 → 1  (1098)

Use 0 for missing digits, continue while carry exists!
```

### Solution
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    """
    Add two numbers represented as reversed linked lists.

    Strategy:
    - Traverse both lists simultaneously
    - Add corresponding digits plus carry
    - Handle different lengths and final carry

    Time: O(max(m, n)) - traverse longer list
    Space: O(max(m, n)) - result list
    """
    # Dummy node to simplify head handling
    dummy = ListNode(0)
    current = dummy
    carry = 0

    # Process while either list has nodes OR there's a carry
    while l1 or l2 or carry:
        # Get values (0 if list ended)
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0

        # Calculate sum and new carry
        total = val1 + val2 + carry
        carry = total // 10
        digit = total % 10

        # Create new node with digit
        current.next = ListNode(digit)
        current = current.next

        # Move to next nodes (if they exist)
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None

    return dummy.next
```

### Complexity
- **Time**: O(max(m, n)) - Traverse the longer list once
- **Space**: O(max(m, n)) - Result list has at most max(m,n) + 1 nodes

### Edge Cases
- Different lengths: Use 0 for missing digits
- Final carry: `[9] + [1] = [0,1]` - Don't forget the final carry!
- Both zeros: `[0] + [0] = [0]`
- Very large numbers: Works since we never convert to actual integers

---

## Problem 2: Remove Nth Node From End (LC #19) - Medium

- [LeetCode](https://leetcode.com/problems/remove-nth-node-from-end-of-list/)

### Problem Statement
Given the head of a linked list, remove the `n`th node from the **end** of the list and return its head. Follow up: Could you do this in one pass?

### Video Explanation
- [NeetCode - Remove Nth Node From End](https://www.youtube.com/watch?v=XVuQxVej6y8)

### Examples
```
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]
Explanation: Remove node 4 (2nd from end)

Input: head = [1], n = 1
Output: []
Explanation: Remove the only node

Input: head = [1,2], n = 1
Output: [1]
```

### Intuition Development
```
Two pointers with n-node gap - one pass solution!

List: 1 → 2 → 3 → 4 → 5, n = 2

Step 1: Create dummy node (handles removing head)
        dummy → 1 → 2 → 3 → 4 → 5

Step 2: Move fast n+1 steps ahead
┌─────────────────────────────────────────────────────────────────┐
│ Start:  slow=dummy, fast=dummy                                  │
│                                                                  │
│ After n+1 = 3 steps:                                            │
│         dummy → 1 → 2 → 3 → 4 → 5 → null                        │
│           ↑              ↑                                       │
│          slow           fast                                     │
│                                                                  │
│ Gap = 3 nodes (when fast reaches null, slow is before target)   │
└─────────────────────────────────────────────────────────────────┘

Step 3: Move both until fast reaches null
        dummy → 1 → 2 → 3 → 4 → 5 → null
                          ↑              ↑
                        slow           fast

Step 4: Remove slow.next (node 4)
        slow.next = slow.next.next
        Result: 1 → 2 → 3 → 5
```

### Solution
```python
def removeNthFromEnd(head: ListNode, n: int) -> ListNode:
    """
    Remove nth node from end using two pointers.

    Strategy:
    - Use two pointers with n nodes gap
    - When fast reaches end, slow is at node before target
    - Remove the next node

    Time: O(L) where L = list length
    Space: O(1)
    """
    # Dummy node handles edge case of removing head
    dummy = ListNode(0)
    dummy.next = head

    slow = dummy
    fast = dummy

    # Move fast pointer n+1 steps ahead
    # This creates gap of n nodes between slow and fast
    for _ in range(n + 1):
        fast = fast.next

    # Move both until fast reaches end
    while fast:
        slow = slow.next
        fast = fast.next

    # slow is now at node BEFORE the one to remove
    # Skip the target node
    slow.next = slow.next.next

    return dummy.next
```

### Complexity
- **Time**: O(L) - Single pass through list of length L
- **Space**: O(1) - Only use two pointers

### Edge Cases
- Single node: `n = 1` removes the only node → return None
- Remove head: `n = length` → dummy node handles this
- `n > length`: Not possible per constraints (1 ≤ n ≤ length)

---

## Problem 3: Swap Nodes in Pairs (LC #24) - Medium

- [LeetCode](https://leetcode.com/problems/swap-nodes-in-pairs/)

### Problem Statement
Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem **without modifying the values** in the list's nodes (i.e., only nodes themselves may be changed).

### Video Explanation
- [NeetCode - Swap Nodes in Pairs](https://www.youtube.com/watch?v=o811TZLAWOo)

### Examples
```
Input: head = [1,2,3,4]
Output: [2,1,4,3]

Input: head = []
Output: []

Input: head = [1]
Output: [1]
```

### Intuition Development
```
Rewire pointers for each pair of nodes!

Before: prev → A → B → C → D
After:  prev → B → A → C → D

Pointer operations for swapping A and B:
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: prev.next = B      (prev points to second node)        │
│         prev → B     A → B → C                                   │
│                                                                  │
│ Step 2: A.next = B.next    (A points past B)                    │
│         A → C                                                    │
│                                                                  │
│ Step 3: B.next = A         (B points to A)                      │
│         prev → B → A → C → D                                     │
│                                                                  │
│ Step 4: Move prev to A (now second in pair)                     │
│         Continue with next pair (C, D)                          │
└─────────────────────────────────────────────────────────────────┘

Full example: 1 → 2 → 3 → 4
┌─────────────────────────────────────────────────────────────────┐
│ Initial: dummy → 1 → 2 → 3 → 4                                  │
│          prev=dummy, first=1, second=2                          │
│                                                                  │
│ Swap 1:  dummy → 2 → 1 → 3 → 4                                  │
│          prev=1, first=3, second=4                              │
│                                                                  │
│ Swap 2:  dummy → 2 → 1 → 4 → 3                                  │
│          prev=3, first=null (stop)                              │
│                                                                  │
│ Result:  2 → 1 → 4 → 3 ✓                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def swapPairs(head: ListNode) -> ListNode:
    """
    Swap adjacent nodes in pairs.

    Strategy:
    - Use dummy node for easier head handling
    - For each pair, rewire the pointers

    Before: prev -> A -> B -> C
    After:  prev -> B -> A -> C

    Time: O(n)
    Space: O(1)
    """
    dummy = ListNode(0)
    dummy.next = head
    prev = dummy

    while prev.next and prev.next.next:
        # Identify the pair
        first = prev.next      # A
        second = prev.next.next  # B

        # Swap the pair
        # 1. prev points to second (B)
        prev.next = second

        # 2. first (A) points to what's after second
        first.next = second.next

        # 3. second (B) points to first (A)
        second.next = first

        # Move prev to first (now second in the pair)
        prev = first

    return dummy.next


def swapPairs_recursive(head: ListNode) -> ListNode:
    """
    Recursive solution - swap first pair, recurse on rest.

    Time: O(n)
    Space: O(n) - recursion stack
    """
    # Base case: 0 or 1 node
    if not head or not head.next:
        return head

    # Identify pair
    first = head
    second = head.next

    # Swap: second becomes new head
    # first points to recursively swapped rest
    first.next = swapPairs_recursive(second.next)
    second.next = first

    return second
```

### Complexity
- **Time**: O(n) - Visit each node once
- **Space**: O(1) iterative, O(n) recursive (call stack)

### Edge Cases
- Empty list: Return empty
- Single node: Return as-is (nothing to swap)
- Odd length: Last node stays in place

---

## Problem 4: Reorder List (LC #143) - Medium

- [LeetCode](https://leetcode.com/problems/reorder-list/)

### Problem Statement
You are given the head of a singly linked list: `L0 → L1 → ... → Ln-1 → Ln`. Reorder the list to be: `L0 → Ln → L1 → Ln-1 → L2 → Ln-2 → ...`. You may not modify the values in the list's nodes. Only nodes themselves may be changed.

### Video Explanation
- [NeetCode - Reorder List](https://www.youtube.com/watch?v=S5bfdUTrKLM)

### Examples
```
Input: head = [1,2,3,4]
Output: [1,4,2,3]

Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]

Input: head = [1,2]
Output: [1,2]
```

### Intuition Development
```
Three-step algorithm: Split → Reverse → Merge

Example: 1 → 2 → 3 → 4 → 5

┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Find Middle (Fast/Slow Pointers)                       │
│                                                                  │
│ 1 → 2 → 3 → 4 → 5                                               │
│         ↑       ↑                                                │
│       slow    fast                                               │
│                                                                  │
│ Middle = 3, Second half starts at 4                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Reverse Second Half                                     │
│                                                                  │
│ Before: 4 → 5                                                    │
│ After:  5 → 4                                                    │
│                                                                  │
│ Now we have:                                                     │
│   First half:  1 → 2 → 3                                        │
│   Second half: 5 → 4                                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Merge Alternately                                       │
│                                                                  │
│ Take from first:  1 →                                            │
│ Take from second:     5 →                                        │
│ Take from first:          2 →                                    │
│ Take from second:             4 →                                │
│ Take from first:                  3                              │
│                                                                  │
│ Result: 1 → 5 → 2 → 4 → 3 ✓                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def reorderList(head: ListNode) -> None:
    """
    Reorder list by interleaving front and back.

    Strategy:
    1. Find middle of list
    2. Reverse second half
    3. Merge two halves alternately

    Time: O(n)
    Space: O(1)
    """
    if not head or not head.next:
        return

    # Step 1: Find middle using slow/fast pointers
    slow = fast = head
    while fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next

    # slow is at middle (or left-middle for even length)
    # Second half starts at slow.next

    # Step 2: Reverse second half
    second = slow.next
    slow.next = None  # Cut the list

    prev = None
    while second:
        next_node = second.next
        second.next = prev
        prev = second
        second = next_node

    # prev is now head of reversed second half

    # Step 3: Merge two halves alternately
    first = head
    second = prev

    while second:
        # Save next pointers
        first_next = first.next
        second_next = second.next

        # Interleave
        first.next = second
        second.next = first_next

        # Move to next pair
        first = first_next
        second = second_next
```

### Complexity
- **Time**: O(n) - Each operation is O(n)
- **Space**: O(1) - In-place modifications

### Edge Cases
- Empty or single node: Return as-is
- Two nodes: `[1,2]` → `[1,2]` (already correct)
- Even vs odd length: Both work with this algorithm

---

## Problem 5: Linked List Cycle II (LC #142) - Medium

- [LeetCode](https://leetcode.com/problems/linked-list-cycle-ii/)

### Problem Statement
Given the head of a linked list, return the node where the cycle begins. If there is no cycle, return `null`. There is a cycle if some node can be reached again by following the `next` pointer.

### Video Explanation
- [NeetCode - Linked List Cycle II](https://www.youtube.com/watch?v=wjYnzkAhcNk)

### Examples
```
Input: head = [3,2,0,-4], pos = 1
Output: Node with value 2
Explanation: The tail connects to the 1st node (0-indexed)

Input: head = [1,2], pos = 0
Output: Node with value 1
Explanation: The tail connects to the 0th node

Input: head = [1], pos = -1
Output: null
Explanation: No cycle
```

### Intuition Development
```
Floyd's Cycle Detection - Two Phases with Mathematical Proof!

┌─────────────────────────────────────────────────────────────────┐
│ VISUALIZATION:                                                   │
│                                                                  │
│    head ──→ ○ ──→ ○ ──→ ○ ──→ ○ (cycle start)                  │
│                             ↙   ↖                                │
│                           ○       ○                              │
│                           ↓       ↑                              │
│                           ○ ──→ ○ (meeting point)               │
│                                                                  │
│    |←───── a ─────→|←─ b ─→|                                    │
│                             |←──── c ────→|                      │
│                                                                  │
│    a = distance from head to cycle start                        │
│    b = distance from cycle start to meeting point               │
│    c = cycle length                                              │
└─────────────────────────────────────────────────────────────────┘

Mathematical Proof:
┌─────────────────────────────────────────────────────────────────┐
│ When slow and fast meet:                                         │
│   slow traveled: a + b                                           │
│   fast traveled: a + b + k×c (k complete cycles, k ≥ 1)         │
│                                                                  │
│ Since fast = 2 × slow:                                           │
│   a + b + k×c = 2(a + b)                                        │
│   k×c = a + b                                                    │
│   a = k×c - b = (k-1)×c + (c - b)                               │
│                                                                  │
│ This means:                                                      │
│   Distance from head to cycle start (a)                          │
│   = Distance from meeting point to cycle start (c - b)          │
│   + some complete cycles                                         │
│                                                                  │
│ So starting two pointers at head and meeting point,             │
│ moving at SAME speed, they'll meet at cycle start!              │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def detectCycle(head: ListNode) -> ListNode:
    """
    Find the start of cycle using Floyd's algorithm.

    Mathematical proof:
    - Let distance from head to cycle start = a
    - Let cycle length = c
    - Let distance from cycle start to meeting point = b

    When slow and fast meet:
    - slow traveled: a + b
    - fast traveled: a + b + k*c (for some k >= 1)
    - Since fast = 2 * slow: a + b + k*c = 2(a + b)
    - Therefore: k*c = a + b
    - So: a = k*c - b = (k-1)*c + (c - b)

    This means: distance from head to cycle start (a)
              = distance from meeting point to cycle start (c - b)
              + some complete cycles

    So if we start two pointers from head and meeting point,
    moving at same speed, they'll meet at cycle start!

    Time: O(n)
    Space: O(1)
    """
    if not head or not head.next:
        return None

    # Phase 1: Detect cycle using fast/slow
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            # Cycle detected! Move to phase 2
            break
    else:
        # No cycle (fast reached end)
        return None

    # Phase 2: Find cycle start
    # Reset slow to head, keep fast at meeting point
    slow = head

    while slow != fast:
        slow = slow.next
        fast = fast.next

    # They meet at cycle start
    return slow
```

### Complexity
- **Time**: O(n) - Each phase is O(n)
- **Space**: O(1) - Only use two pointers

### Edge Cases
- No cycle: Fast reaches null → return null
- Single node with no cycle: Return null
- Cycle at head: Works correctly with the math

---

## Problem 6: Copy List with Random Pointer (LC #138) - Medium

- [LeetCode](https://leetcode.com/problems/copy-list-with-random-pointer/)

### Problem Statement
A linked list of length `n` is given such that each node contains an additional random pointer, which could point to any node in the list, or `null`. Construct a **deep copy** of the list. The deep copy should consist of exactly `n` brand new nodes.

### Video Explanation
- [NeetCode - Copy List with Random Pointer](https://www.youtube.com/watch?v=5Y2EiZST97Y)

### Examples
```
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]
Explanation: Each pair [val, random_index]

Input: head = [[1,1],[2,1]]
Output: [[1,1],[2,1]]

Input: head = [[3,null],[3,0],[3,null]]
Output: [[3,null],[3,0],[3,null]]
```

### Intuition Development
```
Challenge: Random pointers can point to nodes we haven't created yet!

Two approaches:

APPROACH 1: Hash Map (O(n) space)
┌─────────────────────────────────────────────────────────────────┐
│ Pass 1: Create all new nodes, map old → new                     │
│                                                                  │
│   Original: A → B → C                                           │
│   Map: {A: A', B: B', C: C'}                                    │
│                                                                  │
│ Pass 2: Set next and random pointers using map                  │
│   A'.next = map[A.next] = B'                                    │
│   A'.random = map[A.random]                                     │
└─────────────────────────────────────────────────────────────────┘

APPROACH 2: Interleaving (O(1) space)
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Insert copy after each original                         │
│   A → A' → B → B' → C → C'                                      │
│                                                                  │
│ Step 2: Set random pointers for copies                          │
│   If A.random = C, then A'.random = A.random.next = C'          │
│                                                                  │
│ Step 3: Separate the two lists                                  │
│   Original: A → B → C                                           │
│   Copy:     A' → B' → C'                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
class Node:
    def __init__(self, val=0, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random


def copyRandomList(head: Node) -> Node:
    """
    Deep copy list with random pointers.

    Strategy 1: Hash map approach
    - First pass: Create all new nodes, map old -> new
    - Second pass: Set next and random pointers

    Time: O(n)
    Space: O(n)
    """
    if not head:
        return None

    # Map: original node -> copied node
    node_map = {}

    # First pass: Create all nodes
    current = head
    while current:
        node_map[current] = Node(current.val)
        current = current.next

    # Second pass: Set pointers
    current = head
    while current:
        copy = node_map[current]

        # Set next pointer
        copy.next = node_map.get(current.next)

        # Set random pointer
        copy.random = node_map.get(current.random)

        current = current.next

    return node_map[head]


def copyRandomList_O1_space(head: Node) -> Node:
    """
    O(1) space solution by interleaving nodes.

    Strategy:
    1. Insert copy after each original: A -> A' -> B -> B' -> ...
    2. Set random pointers for copies
    3. Separate into two lists

    Time: O(n)
    Space: O(1)
    """
    if not head:
        return None

    # Step 1: Insert copies after originals
    current = head
    while current:
        copy = Node(current.val)
        copy.next = current.next
        current.next = copy
        current = copy.next

    # Step 2: Set random pointers
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
```

### Complexity
- **Time**: O(n) - Two or three passes through the list
- **Space**: O(n) for hash map approach, O(1) for interleaving approach

### Edge Cases
- Empty list: Return null
- Random points to null: Handle with `map.get()` returning None
- Random points to self: Works correctly

---

## Problem 7: Sort List (LC #148) - Medium

- [LeetCode](https://leetcode.com/problems/sort-list/)

### Problem Statement
Given the head of a linked list, return the list after sorting it in **ascending order**. Can you sort the linked list in O(n log n) time and O(1) memory (i.e., constant space)?

### Video Explanation
- [NeetCode - Sort List](https://www.youtube.com/watch?v=TGveA1oFhrc)

### Examples
```
Input: head = [4,2,1,3]
Output: [1,2,3,4]

Input: head = [-1,5,3,4,0]
Output: [-1,0,3,4,5]

Input: head = []
Output: []
```

### Intuition Development
```
Merge Sort is perfect for linked lists!

Why Merge Sort?
- O(n log n) time ✓
- O(log n) space for recursion (better than O(n) for arrays)
- No random access needed (unlike Quick Sort)

Algorithm:
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Find Middle (Fast/Slow Pointers)                       │
│ Step 2: Split into two halves                                   │
│ Step 3: Recursively sort each half                              │
│ Step 4: Merge sorted halves                                     │
└─────────────────────────────────────────────────────────────────┘

Example: [4, 2, 1, 3]

┌─────────────────────────────────────────────────────────────────┐
│                    [4, 2, 1, 3]                                 │
│                    /          \                                  │
│               [4, 2]          [1, 3]                            │
│               /    \          /    \                             │
│            [4]     [2]      [1]    [3]                          │
│               \    /          \    /                             │
│               [2, 4]          [1, 3]                            │
│                    \          /                                  │
│                    [1, 2, 3, 4]                                 │
└─────────────────────────────────────────────────────────────────┘

Key insight for finding middle:
  Start fast at head.next to get LEFT-middle for even length
  This ensures proper splitting (e.g., [1,2] splits to [1] and [2])
```

### Solution
```python
def sortList(head: ListNode) -> ListNode:
    """
    Sort linked list using merge sort.

    Strategy:
    1. Find middle, split into two halves
    2. Recursively sort each half
    3. Merge sorted halves

    Time: O(n log n)
    Space: O(log n) - recursion stack
    """
    # Base case: empty or single node
    if not head or not head.next:
        return head

    # Find middle (left-middle for even length)
    slow = head
    fast = head.next  # Start fast one ahead to get left-middle

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # Split the list
    mid = slow.next
    slow.next = None

    # Recursively sort both halves
    left = sortList(head)
    right = sortList(mid)

    # Merge sorted halves
    return merge(left, right)


def merge(l1: ListNode, l2: ListNode) -> ListNode:
    """
    Merge two sorted linked lists.

    Time: O(n + m)
    Space: O(1)
    """
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

    # Attach remaining nodes
    current.next = l1 if l1 else l2

    return dummy.next
```

### Complexity
- **Time**: O(n log n) - Standard merge sort complexity
- **Space**: O(log n) - Recursion stack depth

### Edge Cases
- Empty list: Return null
- Single node: Already sorted
- All same values: Works correctly
- Already sorted: Still O(n log n)

---

## Problem 8: Rotate List (LC #61) - Medium

- [LeetCode](https://leetcode.com/problems/rotate-list/)

### Problem Statement
Given the head of a linked list, rotate the list to the right by `k` places. Note: `k` can be larger than the length of the list.

### Video Explanation
- [NeetCode - Rotate List](https://www.youtube.com/watch?v=UcGtPs2LE_c)

### Examples
```
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]
Explanation: Rotate right 2 places

Input: head = [0,1,2], k = 4
Output: [2,0,1]
Explanation: k=4 is same as k=1 for length 3

Input: head = [1], k = 1
Output: [1]
```

### Intuition Development
```
Key Insight: Make it circular, then break at the right spot!

Example: [1,2,3,4,5], k = 2

┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Find length and tail (length = 5)                       │
│         1 → 2 → 3 → 4 → 5                                       │
│                         ↑                                        │
│                       tail                                       │
│                                                                  │
│ Step 2: Optimize k (k = k % length = 2 % 5 = 2)                 │
│                                                                  │
│ Step 3: Make circular (tail.next = head)                        │
│         1 → 2 → 3 → 4 → 5                                       │
│         ↑               |                                        │
│         └───────────────┘                                        │
│                                                                  │
│ Step 4: Find new tail (length - k - 1 = 5 - 2 - 1 = 2 steps)   │
│         New tail = node at index 2 = 3                          │
│         New head = node at index 3 = 4                          │
│                                                                  │
│ Step 5: Break circle (new_tail.next = null)                     │
│         4 → 5 → 1 → 2 → 3                                       │
└─────────────────────────────────────────────────────────────────┘

Why (length - k) positions from start?
  Rotating right by k = new head is at position (length - k)
  New tail is at position (length - k - 1)
```

### Solution
```python
def rotateRight(head: ListNode, k: int) -> ListNode:
    """
    Rotate list right by k positions.

    Strategy:
    1. Find length and connect tail to head (make circular)
    2. Find new tail position (length - k % length - 1)
    3. Break the circle at new tail

    Time: O(n)
    Space: O(1)
    """
    if not head or not head.next or k == 0:
        return head

    # Find length and tail
    length = 1
    tail = head
    while tail.next:
        length += 1
        tail = tail.next

    # Optimize k (handle k > length)
    k = k % length
    if k == 0:
        return head  # No rotation needed

    # Make circular
    tail.next = head

    # Find new tail (length - k - 1 steps from head)
    steps = length - k - 1
    new_tail = head
    for _ in range(steps):
        new_tail = new_tail.next

    # New head is next of new tail
    new_head = new_tail.next

    # Break the circle
    new_tail.next = None

    return new_head
```

### Complexity
- **Time**: O(n) - Two passes (find length + find new tail)
- **Space**: O(1) - Only use a few pointers

### Edge Cases
- `k = 0`: No rotation needed
- `k = length`: Same as no rotation
- `k > length`: Use k % length
- Empty list: Return null
- Single node: Return as-is

---

## Summary: Medium Linked List Problems

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | Add Two Numbers | Digit by digit with carry | O(n) | O(n) |
| 2 | Remove Nth From End | Two pointers with gap | O(n) | O(1) |
| 3 | Swap Pairs | Pointer rewiring | O(n) | O(1) |
| 4 | Reorder List | Find mid + reverse + merge | O(n) | O(1) |
| 5 | Cycle Start | Floyd's algorithm | O(n) | O(1) |
| 6 | Copy Random List | Hash map or interleaving | O(n) | O(n)/O(1) |
| 7 | Sort List | Merge sort | O(n log n) | O(log n) |
| 8 | Rotate List | Make circular, break | O(n) | O(1) |

---

## Practice More Problems

- [ ] LC #82 - Remove Duplicates from Sorted List II
- [ ] LC #86 - Partition List
- [ ] LC #92 - Reverse Linked List II
- [ ] LC #147 - Insertion Sort List
- [ ] LC #328 - Odd Even Linked List

