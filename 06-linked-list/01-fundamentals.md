# Linked List - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE LINKED LIST PATTERNS                         │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Linked list" (obviously!)                                               │
│  ✓ "Reverse" a list                                                         │
│  ✓ "Find middle"                                                            │
│  ✓ "Detect cycle"                                                           │
│  ✓ "Merge sorted lists"                                                     │
│  ✓ "Remove nth from end"                                                    │
│  ✓ "Intersection of lists"                                                  │
│                                                                             │
│  Key patterns:                                                              │
│  1. Fast & Slow Pointers (Tortoise & Hare)                                  │
│  2. In-place Reversal                                                       │
│  3. Dummy Node technique                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Pointers/references concept
- [ ] Basic class/object structure
- [ ] Iteration with while loops

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LINKED LIST MEMORY MAP                                   │
│                                                                             │
│                    ┌─────────────┐                                          │
│         ┌─────────│ LINKED LIST │─────────┐                                 │
│         │         └─────────────┘         │                                 │
│         ▼                                 ▼                                 │
│  ┌─────────────┐                   ┌─────────────┐                          │
│  │ FAST/SLOW   │                   │  REVERSAL   │                          │
│  │  POINTERS   │                   │  PATTERNS   │                          │
│  └──────┬──────┘                   └──────┬──────┘                          │
│         │                                 │                                 │
│    ┌────┴────┐                      ┌─────┴─────┐                           │
│    ▼         ▼                      ▼           ▼                           │
│ ┌──────┐ ┌──────┐               ┌──────┐   ┌──────┐                        │
│ │Cycle │ │Find  │               │Full  │   │Partial│                        │
│ │Detect│ │Middle│               │Reverse│  │Reverse│                        │
│ └──────┘ └──────┘               └──────┘   └──────┘                        │
│                                                                             │
│  Related Patterns:                                                          │
│  • Two Pointers - Fast/slow is specialized two pointers                     │
│  • Recursion - Many linked list problems have recursive solutions           │
│  • Stack - For reversing or backtracking                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LINKED LIST DECISION TREE                                │
│                                                                             │
│  Need to find middle or detect cycle?                                       │
│       │                                                                     │
│       ├── YES → Fast/Slow Pointers                                          │
│       │         Fast moves 2x, slow moves 1x                                │
│       │                                                                     │
│       └── NO → Need to reverse (part of) list?                              │
│                    │                                                        │
│                    ├── YES → In-place reversal                              │
│                    │         prev, curr, next pattern                       │
│                    │                                                        │
│                    └── NO → Need to handle edge cases easily?               │
│                                 │                                           │
│                                 ├── YES → Use dummy node                    │
│                                 │         Simplifies head manipulation      │
│                                 │                                           │
│                                 └── NO → Standard traversal                 │
│                                                                             │
│  TECHNIQUE SELECTION:                                                       │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ Problem                    │ Technique                             │     │
│  ├────────────────────────────┼───────────────────────────────────────┤     │
│  │ Find middle                │ Fast/Slow pointers                    │     │
│  │ Detect cycle               │ Fast/Slow (Floyd's)                   │     │
│  │ Find cycle start           │ Fast/Slow + reset                     │     │
│  │ Reverse list               │ prev/curr/next iteration              │     │
│  │ Remove nth from end        │ Two pointers with gap                 │     │
│  │ Merge sorted lists         │ Dummy node + comparison               │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concept: ListNode Structure

```python
class ListNode:
    """Standard singly linked list node."""
    def __init__(self, val=0, next=None):
        self.val = val    # Value stored in node
        self.next = next  # Pointer to next node (None if last)


# Creating a linked list: 1 -> 2 -> 3 -> None
head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)

# Traversal
def print_list(head: ListNode) -> None:
    """Print all values in linked list."""
    current = head
    while current:
        print(current.val, end=" -> ")
        current = current.next
    print("None")
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LINKED LIST VISUALIZATION                                │
│                                                                             │
│  Singly Linked List:                                                        │
│                                                                             │
│  head                                                                       │
│   │                                                                         │
│   ▼                                                                         │
│  ┌───┬───┐    ┌───┬───┐    ┌───┬───┐    ┌───┬───┐                          │
│  │ 1 │ ●─┼───►│ 2 │ ●─┼───►│ 3 │ ●─┼───►│ 4 │ ╱ │                          │
│  └───┴───┘    └───┴───┘    └───┴───┘    └───┴───┘                          │
│   val next     val next     val next     val next=None                      │
│                                                                             │
│  Doubly Linked List:                                                        │
│                                                                             │
│  ┌───┬───┬───┐    ┌───┬───┬───┐    ┌───┬───┬───┐                           │
│  │ ╱ │ 1 │ ●─┼◄──►│ ● │ 2 │ ●─┼◄──►│ ● │ 3 │ ╱ │                           │
│  └───┴───┴───┘    └───┴───┴───┘    └───┴───┴───┘                           │
│  prev val next    prev val next    prev val next                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Essential Techniques

### Technique 1: Dummy Node

Use a dummy node to simplify edge cases (empty list, single node, operations at head).

```python
def remove_elements(head: ListNode, val: int) -> ListNode:
    """
    Remove all nodes with given value.

    Dummy node eliminates special case for removing head.

    Time: O(n)
    Space: O(1)
    """
    # Create dummy node pointing to head
    dummy = ListNode(0)
    dummy.next = head

    # Traverse with previous pointer
    prev = dummy
    current = head

    while current:
        if current.val == val:
            # Skip current node (remove it)
            prev.next = current.next
        else:
            # Keep current node, move prev forward
            prev = current
        current = current.next

    # Return new head (dummy.next handles case where head was removed)
    return dummy.next
```

### Technique 2: Fast & Slow Pointers (Floyd's Algorithm)

Two pointers moving at different speeds to detect cycles or find middle.

```python
def find_middle(head: ListNode) -> ListNode:
    """
    Find middle node of linked list.

    Strategy:
    - Slow pointer moves 1 step
    - Fast pointer moves 2 steps
    - When fast reaches end, slow is at middle

    Time: O(n)
    Space: O(1)
    """
    slow = fast = head

    # For odd length: fast stops at last node
    # For even length: fast stops at None
    while fast and fast.next:
        slow = slow.next        # Move 1 step
        fast = fast.next.next   # Move 2 steps

    return slow  # Middle node


def has_cycle(head: ListNode) -> bool:
    """
    Detect if linked list has a cycle.

    Strategy:
    - If cycle exists, fast will eventually catch slow
    - If no cycle, fast reaches end

    Time: O(n)
    Space: O(1)
    """
    if not head or not head.next:
        return False

    slow = head
    fast = head.next  # Start fast one step ahead

    while slow != fast:
        if not fast or not fast.next:
            return False  # Reached end, no cycle
        slow = slow.next
        fast = fast.next.next

    return True  # Pointers met, cycle exists


def detect_cycle_start(head: ListNode) -> ListNode:
    """
    Find the node where cycle begins.

    Strategy (Floyd's Cycle Detection):
    1. Find meeting point using fast/slow
    2. Reset one pointer to head
    3. Move both at same speed - they meet at cycle start

    Math proof:
    - Let distance to cycle start = a
    - Let cycle length = c
    - At meeting: slow traveled a + b, fast traveled a + b + c
    - Since fast = 2 × slow: a + b + c = 2(a + b)
    - Therefore: c = a + b, so a = c - b
    - Starting from meeting point, c - b steps reaches cycle start
    - Starting from head, a steps also reaches cycle start

    Time: O(n)
    Space: O(1)
    """
    if not head or not head.next:
        return None

    # Phase 1: Find meeting point
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle

    # Phase 2: Find cycle start
    # Reset slow to head, keep fast at meeting point
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next  # Both move at same speed now

    return slow  # Cycle start node
```

### Technique 3: In-Place Reversal

Reverse links without extra space.

```python
def reverse_list(head: ListNode) -> ListNode:
    """
    Reverse entire linked list in-place.

    Strategy:
    - Track three pointers: prev, current, next
    - Reverse each link one at a time

    Before: 1 -> 2 -> 3 -> None
    After:  None <- 1 <- 2 <- 3

    Time: O(n)
    Space: O(1)
    """
    prev = None
    current = head

    while current:
        # Save next node before breaking link
        next_node = current.next

        # Reverse the link
        current.next = prev

        # Move pointers forward
        prev = current
        current = next_node

    # prev is now the new head
    return prev


def reverse_between(head: ListNode, left: int, right: int) -> ListNode:
    """
    Reverse nodes from position left to right (1-indexed).

    Example: 1->2->3->4->5, left=2, right=4
    Result:  1->4->3->2->5

    Time: O(n)
    Space: O(1)
    """
    if not head or left == right:
        return head

    # Use dummy to handle edge case where left = 1
    dummy = ListNode(0)
    dummy.next = head

    # Move to node before reversal starts
    prev = dummy
    for _ in range(left - 1):
        prev = prev.next

    # Start reversal
    current = prev.next

    # Reverse (right - left) links
    for _ in range(right - left):
        # Node to move
        next_node = current.next

        # Remove next_node from its position
        current.next = next_node.next

        # Insert next_node at the beginning of reversed portion
        next_node.next = prev.next
        prev.next = next_node

    return dummy.next
```

---

## Visual: In-Place Reversal

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REVERSAL STEP BY STEP                                    │
│                                                                             │
│  Initial: 1 -> 2 -> 3 -> None                                               │
│           ↑                                                                 │
│         current                                                             │
│         prev = None                                                         │
│                                                                             │
│  Step 1: None <- 1    2 -> 3 -> None                                        │
│                  ↑    ↑                                                     │
│                prev current                                                 │
│                                                                             │
│  Step 2: None <- 1 <- 2    3 -> None                                        │
│                       ↑    ↑                                                │
│                     prev current                                            │
│                                                                             │
│  Step 3: None <- 1 <- 2 <- 3                                                │
│                            ↑    current = None                              │
│                          prev   (loop ends)                                 │
│                                                                             │
│  Return prev (new head = 3)                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Operations

```python
def get_length(head: ListNode) -> int:
    """Get length of linked list."""
    length = 0
    while head:
        length += 1
        head = head.next
    return length


def get_nth_node(head: ListNode, n: int) -> ListNode:
    """Get nth node (0-indexed)."""
    for _ in range(n):
        if not head:
            return None
        head = head.next
    return head


def merge_two_sorted(l1: ListNode, l2: ListNode) -> ListNode:
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

---

## Common Mistakes

```python
# ❌ WRONG: Losing reference to next node
def reverse_wrong(head):
    current = head
    while current:
        current.next = prev  # Lost reference to original next!
        prev = current
        current = current.next  # current.next is now prev!

# ✅ CORRECT: Save next before modifying
def reverse_correct(head):
    prev = None
    current = head
    while current:
        next_node = current.next  # Save first!
        current.next = prev
        prev = current
        current = next_node

# ❌ WRONG: Not handling empty list
def find_middle_wrong(head):
    slow = fast = head
    while fast.next:  # Crashes if head is None!
        slow = slow.next
        fast = fast.next.next

# ✅ CORRECT: Check for None
def find_middle_correct(head):
    if not head:
        return None
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

---

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Traversal | O(n) | O(1) |
| Find middle | O(n) | O(1) |
| Detect cycle | O(n) | O(1) |
| Reverse | O(n) | O(1) |
| Merge sorted | O(n+m) | O(1) |

---

## Interview Tips

### 1. How to Explain Your Approach
```
"I'll use the fast/slow pointer technique. Fast moves two steps while
slow moves one. When fast reaches the end, slow is at the middle. For
cycle detection, if they meet, there's a cycle."
```

### 2. What Interviewers Look For
- **Pointer manipulation**: Clean, bug-free pointer updates
- **Edge cases**: Empty list, single node, cycle at head
- **Space efficiency**: O(1) space solutions preferred

### 3. Common Follow-up Questions
- "Can you do it recursively?" → Yes, but O(n) stack space
- "What if it's doubly linked?" → Easier reversal, different traversal
- "How to find cycle start?" → After detection, reset one pointer to head

---

## Related Patterns

- **Two Pointers**: Fast/slow pointer is a specialized two-pointer technique
- **Recursion**: Many linked list problems have elegant recursive solutions
- **Stacks**: Can simulate recursion for linked list operations

### When to Combine

- **Linked List + Merge Sort**: Sort a linked list in O(n log n)
- **Linked List + Hash Set**: Detect cycle or find intersection in O(n) space
- **Linked List + Recursion**: Reverse, palindrome check with implicit stack

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
