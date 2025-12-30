# Linked List - Easy Problems

## Problem 1: Reverse Linked List (LC #206) - Easy

- [LeetCode](https://leetcode.com/problems/reverse-linked-list/)

### Problem Statement
Given the head of a singly linked list, reverse the list, and return the reversed list.

### Examples
```
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Input: head = [1,2]
Output: [2,1]

Input: head = []
Output: []
```

### Video Explanation
- [NeetCode - Reverse Linked List](https://www.youtube.com/watch?v=G0_I-ZF0S38)
- [Take U Forward - Reverse LL](https://www.youtube.com/watch?v=iRtLEoL-r-g)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  REVERSING A LINKED LIST                                                    │
│                                                                             │
│  Original: 1 → 2 → 3 → 4 → 5 → NULL                                        │
│  Goal:     NULL ← 1 ← 2 ← 3 ← 4 ← 5                                        │
│                                                                             │
│  Step-by-step visualization:                                                │
│                                                                             │
│  Step 0:  NULL   1 → 2 → 3 → 4 → 5                                         │
│            ↑     ↑                                                          │
│          prev  curr                                                         │
│                                                                             │
│  Step 1:  NULL ← 1   2 → 3 → 4 → 5                                         │
│                  ↑   ↑                                                      │
│                prev curr                                                    │
│                                                                             │
│  Step 2:  NULL ← 1 ← 2   3 → 4 → 5                                         │
│                      ↑   ↑                                                  │
│                    prev curr                                                │
│                                                                             │
│  ... continue until curr is NULL                                            │
│                                                                             │
│  Final:   NULL ← 1 ← 2 ← 3 ← 4 ← 5                                         │
│                                  ↑                                          │
│                                prev (new head!)                             │
│                                                                             │
│  Key insight: Save next, reverse pointer, advance all pointers              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverseList(head: ListNode) -> ListNode:
    """
    Reverse linked list iteratively.

    Strategy:
    - Use three pointers: prev, current, next
    - For each node, reverse its pointer to point to prev
    - Move all pointers one step forward

    Time: O(n) - visit each node once
    Space: O(1) - only using pointers
    """
    prev = None      # Will become new tail
    current = head   # Start at head

    while current:
        # Step 1: Save reference to next node (before we break the link)
        next_node = current.next

        # Step 2: Reverse the pointer
        current.next = prev

        # Step 3: Move prev and current forward
        prev = current
        current = next_node

    # prev is now pointing to the last node (new head)
    return prev


def reverseList_recursive(head: ListNode) -> ListNode:
    """
    Reverse linked list recursively.

    Base case: Empty list or single node
    Recursive case: Reverse rest, then fix current node's links

    Time: O(n)
    Space: O(n) - recursion stack
    """
    # Base case: empty list or single node
    if not head or not head.next:
        return head

    # Recursively reverse the rest of the list
    new_head = reverseList_recursive(head.next)

    # head.next is now the tail of reversed portion
    # Make it point back to head
    head.next.next = head
    head.next = None

    return new_head
```

### Complexity
- **Iterative**: Time O(n), Space O(1)
- **Recursive**: Time O(n), Space O(n) for call stack

### Edge Cases
- Empty list: Return `None`
- Single node: Return the node itself
- Two nodes: Simple swap of pointers
- Very long list: Works in O(n) time, O(1) space

### Common Mistakes
- Forgetting to save `next_node` before breaking the link
- Not updating all three pointers correctly
- Returning `current` instead of `prev` (current is NULL at end)

### Related Problems
- LC #92 Reverse Linked List II
- LC #25 Reverse Nodes in k-Group
- LC #234 Palindrome Linked List

---

## Problem 2: Merge Two Sorted Lists (LC #21) - Easy

- [LeetCode](https://leetcode.com/problems/merge-two-sorted-lists/)

### Problem Statement
Merge two sorted linked lists into one sorted list by splicing together the nodes of the first two lists.

### Examples
```
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]

Input: list1 = [], list2 = []
Output: []

Input: list1 = [], list2 = [0]
Output: [0]
```

### Video Explanation
- [NeetCode - Merge Two Sorted Lists](https://www.youtube.com/watch?v=XIdigk956u0)
- [Take U Forward - Merge Two Lists](https://www.youtube.com/watch?v=Xb4slcp1U38)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MERGING TWO SORTED LISTS                                                   │
│                                                                             │
│  list1: 1 → 2 → 4                                                          │
│  list2: 1 → 3 → 4                                                          │
│                                                                             │
│  Use a DUMMY node to simplify head handling:                                │
│                                                                             │
│  dummy → ?                                                                  │
│    ↑                                                                        │
│  current                                                                    │
│                                                                             │
│  Compare heads, pick smaller:                                               │
│  Step 1: 1 vs 1 → pick list1's 1                                           │
│          dummy → 1                                                          │
│                                                                             │
│  Step 2: 2 vs 1 → pick list2's 1                                           │
│          dummy → 1 → 1                                                      │
│                                                                             │
│  Step 3: 2 vs 3 → pick list1's 2                                           │
│          dummy → 1 → 1 → 2                                                  │
│                                                                             │
│  Step 4: 4 vs 3 → pick list2's 3                                           │
│          dummy → 1 → 1 → 2 → 3                                              │
│                                                                             │
│  Step 5: 4 vs 4 → pick either, say list1's 4                               │
│          dummy → 1 → 1 → 2 → 3 → 4                                          │
│                                                                             │
│  Step 6: list1 empty, attach remaining list2                                │
│          dummy → 1 → 1 → 2 → 3 → 4 → 4                                      │
│                                                                             │
│  Return dummy.next                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def mergeTwoLists(list1: ListNode, list2: ListNode) -> ListNode:
    """
    Merge two sorted linked lists.

    Strategy:
    - Use dummy node to simplify edge cases
    - Compare heads of both lists
    - Attach smaller one to result, advance that pointer
    - Attach remaining nodes at the end

    Time: O(n + m) where n, m are lengths of lists
    Space: O(1) - only rearranging pointers
    """
    # Dummy node simplifies handling of head
    dummy = ListNode(0)
    current = dummy  # Pointer to build result list

    # Compare and merge while both lists have nodes
    while list1 and list2:
        if list1.val <= list2.val:
            current.next = list1
            list1 = list1.next
        else:
            current.next = list2
            list2 = list2.next
        current = current.next

    # Attach remaining nodes (only one list can have remaining)
    current.next = list1 if list1 else list2

    return dummy.next
```

### Complexity
- **Time**: O(n + m) - visit each node once
- **Space**: O(1) - only rearranging pointers

### Edge Cases
- Both lists empty: Return `None`
- One list empty: Return the other list
- Lists of different lengths: Works correctly
- All elements from one list smaller: That list becomes prefix

### Common Mistakes
- Not using dummy node (makes head handling complex)
- Forgetting to attach remaining nodes at the end
- Moving `current` before assigning `current.next`

### Related Problems
- LC #23 Merge k Sorted Lists
- LC #88 Merge Sorted Array
- LC #148 Sort List

---

## Problem 3: Linked List Cycle (LC #141) - Easy

- [LeetCode](https://leetcode.com/problems/linked-list-cycle/)

### Problem Statement
Given the head of a linked list, determine if the linked list has a cycle in it.

### Examples
```
Input: head = [3,2,0,-4], pos = 1 (cycle at index 1)
Output: true

Input: head = [1,2], pos = 0 (cycle at index 0)
Output: true

Input: head = [1], pos = -1 (no cycle)
Output: false
```

### Video Explanation
- [NeetCode - Linked List Cycle](https://www.youtube.com/watch?v=gBTe7lFR3vc)
- [Take U Forward - Detect Cycle](https://www.youtube.com/watch?v=wiOo4DC5GGA)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FLOYD'S TORTOISE AND HARE ALGORITHM                                        │
│                                                                             │
│  Cycle example:                                                             │
│  3 → 2 → 0 → -4                                                            │
│      ↑        ↓                                                             │
│      └────────┘                                                             │
│                                                                             │
│  Use TWO pointers:                                                          │
│  - Slow (tortoise): moves 1 step at a time                                 │
│  - Fast (hare): moves 2 steps at a time                                    │
│                                                                             │
│  If there's a cycle, fast will eventually "lap" slow                        │
│                                                                             │
│  Step 0: S=3, F=3                                                          │
│  Step 1: S=2, F=0  (slow +1, fast +2)                                      │
│  Step 2: S=0, F=2  (fast went around)                                      │
│  Step 3: S=-4, F=-4 → MEET! Cycle detected!                                │
│                                                                             │
│  Why it works:                                                              │
│  In cycle, fast gains 1 step per iteration                                 │
│  Eventually fast catches up to slow                                         │
│                                                                             │
│  If NO cycle:                                                               │
│  Fast reaches NULL before meeting slow                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def hasCycle(head: ListNode) -> bool:
    """
    Detect cycle using Floyd's Tortoise and Hare algorithm.

    Strategy:
    - Slow pointer moves 1 step at a time
    - Fast pointer moves 2 steps at a time
    - If cycle exists, they will eventually meet
    - If no cycle, fast reaches None

    Time: O(n) - at most 2n steps
    Space: O(1) - only two pointers
    """
    if not head or not head.next:
        return False

    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next        # Move 1 step
        fast = fast.next.next   # Move 2 steps

        if slow == fast:
            return True  # They met, cycle exists

    return False  # Fast reached end, no cycle
```

### Complexity
- **Time**: O(n) - at most 2n iterations
- **Space**: O(1) - only two pointers

### Edge Cases
- Empty list: Return `False`
- Single node without cycle: Return `False`
- Single node with self-cycle: Return `True`
- Cycle at the end: Detected correctly

### Common Mistakes
- Not checking `fast.next` before accessing `fast.next.next`
- Starting slow and fast at different positions unnecessarily
- Using hash set (works but uses O(n) space)

### Related Problems
- LC #142 Linked List Cycle II (find cycle start)
- LC #287 Find the Duplicate Number
- LC #202 Happy Number

---

## Problem 4: Middle of the Linked List (LC #876) - Easy

- [LeetCode](https://leetcode.com/problems/middle-of-the-linked-list/)

### Problem Statement
Given the head of a singly linked list, return the middle node. If there are two middle nodes, return the second middle node.

### Examples
```
Input: head = [1,2,3,4,5]
Output: [3,4,5] (middle is 3)

Input: head = [1,2,3,4,5,6]
Output: [4,5,6] (two middles: 3 and 4, return 4)
```

### Video Explanation
- [NeetCode - Middle of Linked List](https://www.youtube.com/watch?v=A2_ldqM4QcY)
- [Take U Forward - Find Middle](https://www.youtube.com/watch?v=7LjQ57RqgEc)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FINDING THE MIDDLE WITH FAST/SLOW POINTERS                                 │
│                                                                             │
│  Odd length (5 nodes):                                                      │
│  1 → 2 → 3 → 4 → 5 → NULL                                                  │
│  S   F                        Start                                         │
│      S       F                Step 1                                        │
│          S           F        Step 2 (fast.next is NULL, stop)             │
│          ↑                                                                  │
│        MIDDLE                                                               │
│                                                                             │
│  Even length (6 nodes):                                                     │
│  1 → 2 → 3 → 4 → 5 → 6 → NULL                                              │
│  S   F                            Start                                     │
│      S       F                    Step 1                                    │
│          S           F            Step 2                                    │
│              S               F    Step 3 (fast is NULL, stop)              │
│              ↑                                                              │
│           MIDDLE (second one)                                               │
│                                                                             │
│  Key: When fast reaches end, slow is at middle                             │
│  Fast travels 2x speed, so when fast finishes, slow is halfway             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def middleNode(head: ListNode) -> ListNode:
    """
    Find middle node using fast and slow pointers.

    Strategy:
    - Slow moves 1 step, fast moves 2 steps
    - When fast reaches end, slow is at middle

    Time: O(n) - traverse half the list
    Space: O(1) - only two pointers
    """
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow
```

### Complexity
- **Time**: O(n) - traverse list once
- **Space**: O(1) - only two pointers

### Edge Cases
- Single node: Return that node
- Two nodes: Return second node (per problem spec)
- Odd length: Return exact middle
- Even length: Return second of two middles

### Common Mistakes
- Using `while fast.next and fast.next.next` (wrong for even length)
- Not handling single node case
- Returning wrong middle for even length lists

### Related Problems
- LC #234 Palindrome Linked List
- LC #143 Reorder List
- LC #148 Sort List

---

## Problem 5: Remove Duplicates from Sorted List (LC #83) - Easy

- [LeetCode](https://leetcode.com/problems/remove-duplicates-from-sorted-list/)

### Problem Statement
Given the head of a sorted linked list, delete all duplicates such that each element appears only once.

### Examples
```
Input: head = [1,1,2]
Output: [1,2]

Input: head = [1,1,2,3,3]
Output: [1,2,3]
```

### Video Explanation
- [NeetCode - Remove Duplicates](https://www.youtube.com/watch?v=p10f-VpO4nE)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  REMOVING DUPLICATES FROM SORTED LIST                                       │
│                                                                             │
│  Since list is SORTED, duplicates are ADJACENT!                            │
│                                                                             │
│  Input: 1 → 1 → 2 → 3 → 3                                                  │
│         ↑                                                                   │
│       curr                                                                  │
│                                                                             │
│  Step 1: curr.val (1) == curr.next.val (1)                                 │
│          Skip duplicate: curr.next = curr.next.next                        │
│          1 → 2 → 3 → 3                                                     │
│          ↑                                                                  │
│        curr (don't move, might be more dups)                               │
│                                                                             │
│  Step 2: curr.val (1) != curr.next.val (2)                                 │
│          Move forward: curr = curr.next                                    │
│          1 → 2 → 3 → 3                                                     │
│              ↑                                                              │
│            curr                                                             │
│                                                                             │
│  Step 3: curr.val (2) != curr.next.val (3)                                 │
│          Move forward                                                       │
│          1 → 2 → 3 → 3                                                     │
│                  ↑                                                          │
│                curr                                                         │
│                                                                             │
│  Step 4: curr.val (3) == curr.next.val (3)                                 │
│          Skip: curr.next = curr.next.next (NULL)                           │
│          1 → 2 → 3 → NULL                                                  │
│                                                                             │
│  Output: 1 → 2 → 3                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def deleteDuplicates(head: ListNode) -> ListNode:
    """
    Remove duplicates from sorted linked list.

    Strategy:
    - Since list is sorted, duplicates are adjacent
    - For each node, skip all following nodes with same value

    Time: O(n) - visit each node once
    Space: O(1) - in-place modification
    """
    current = head

    while current and current.next:
        if current.val == current.next.val:
            # Skip the duplicate node
            current.next = current.next.next
            # Don't move current - there might be more duplicates
        else:
            # Move to next unique value
            current = current.next

    return head
```

### Complexity
- **Time**: O(n) - single pass
- **Space**: O(1) - in-place

### Edge Cases
- Empty list: Return `None`
- Single node: Return the node (no duplicates possible)
- All duplicates: `[1,1,1]` → `[1]`
- No duplicates: Return list unchanged

### Common Mistakes
- Moving `current` forward when finding a duplicate (miss consecutive dups)
- Not handling empty list or single node
- Using extra space unnecessarily

### Related Problems
- LC #82 Remove Duplicates from Sorted List II
- LC #26 Remove Duplicates from Sorted Array
- LC #203 Remove Linked List Elements

---

## Problem 6: Intersection of Two Linked Lists (LC #160) - Easy

- [LeetCode](https://leetcode.com/problems/intersection-of-two-linked-lists/)

### Problem Statement
Given the heads of two singly linked lists, return the node at which the two lists intersect. If they don't intersect, return null.

### Examples
```
Input:
listA = [4,1,8,4,5]
listB = [5,6,1,8,4,5]
Output: Node with value 8 (intersection point)

Visual:
    4 → 1 ↘
              8 → 4 → 5
5 → 6 → 1 ↗
```

### Video Explanation
- [NeetCode - Intersection of Two Linked Lists](https://www.youtube.com/watch?v=D0X0BONOQhI)
- [Take U Forward - Intersection Point](https://www.youtube.com/watch?v=u4FWXfgS8jw)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FINDING INTERSECTION WITH TWO POINTERS                                     │
│                                                                             │
│  listA: 4 → 1 → 8 → 4 → 5       (length a = 5)                             │
│  listB: 5 → 6 → 1 → 8 → 4 → 5   (length b = 6)                             │
│                   ↑                                                         │
│              intersection                                                   │
│                                                                             │
│  Clever trick: Make both pointers travel SAME total distance               │
│                                                                             │
│  Pointer A: travels listA, then listB                                      │
│  Pointer B: travels listB, then listA                                      │
│                                                                             │
│  A's path: 4→1→8→4→5→5→6→1→[8]                                             │
│  B's path: 5→6→1→8→4→5→4→1→[8]                                             │
│                            ↑                                                │
│                          MEET!                                              │
│                                                                             │
│  Why it works:                                                              │
│  A travels: a + (b - shared) = a + b - c                                   │
│  B travels: b + (a - shared) = a + b - c                                   │
│  Same distance! They meet at intersection.                                  │
│                                                                             │
│  If no intersection: both reach NULL at same time                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def getIntersectionNode(headA: ListNode, headB: ListNode) -> ListNode:
    """
    Find intersection node of two linked lists.

    Strategy (Two Pointers):
    - Pointer A traverses: listA -> listB
    - Pointer B traverses: listB -> listA
    - They will meet at intersection (or both reach None)

    Time: O(n + m)
    Space: O(1)
    """
    if not headA or not headB:
        return None

    pointerA = headA
    pointerB = headB

    # Traverse until they meet (at intersection or both None)
    while pointerA != pointerB:
        # When reaching end, switch to other list's head
        pointerA = pointerA.next if pointerA else headB
        pointerB = pointerB.next if pointerB else headA

    return pointerA  # Either intersection node or None
```

### Complexity
- **Time**: O(n + m) - each pointer traverses both lists
- **Space**: O(1) - only two pointers

### Edge Cases
- One list empty: Return `None`
- No intersection: Both pointers reach `None` together
- Same starting node: Return that node immediately
- Different lengths: Algorithm handles by switching lists

### Common Mistakes
- Using `pointerA.next` instead of checking `pointerA` first (null pointer)
- Not handling case where lists don't intersect
- Using hash set (works but O(n) space)

### Related Problems
- LC #141 Linked List Cycle
- LC #142 Linked List Cycle II
- LC #2 Add Two Numbers

---

## Problem 7: Palindrome Linked List (LC #234) - Easy

- [LeetCode](https://leetcode.com/problems/palindrome-linked-list/)

### Problem Statement
Given the head of a singly linked list, return true if it is a palindrome, false otherwise.

### Examples
```
Input: head = [1,2,2,1]
Output: true

Input: head = [1,2]
Output: false
```

### Video Explanation
- [NeetCode - Palindrome Linked List](https://www.youtube.com/watch?v=yOzXms1J6Nk)
- [Take U Forward - Palindrome LL](https://www.youtube.com/watch?v=-DtNInqFUXs)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CHECKING PALINDROME IN O(1) SPACE                                          │
│                                                                             │
│  Input: 1 → 2 → 2 → 1                                                      │
│                                                                             │
│  Step 1: Find middle using fast/slow                                        │
│  1 → 2 → 2 → 1                                                             │
│          ↑                                                                  │
│        slow (middle)                                                        │
│                                                                             │
│  Step 2: Reverse second half                                                │
│  First half:  1 → 2                                                        │
│  Second half: 1 → 2 (reversed from 2 → 1)                                  │
│                                                                             │
│  Step 3: Compare both halves                                                │
│  1 → 2                                                                      │
│  1 → 2                                                                      │
│  ↑   ↑                                                                      │
│  1=1 2=2 ✓ PALINDROME!                                                     │
│                                                                             │
│  For odd length: [1,2,3,2,1]                                               │
│  First half:  1 → 2                                                        │
│  Middle:      3 (ignored)                                                   │
│  Second half: 1 → 2 (reversed)                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def isPalindrome(head: ListNode) -> bool:
    """
    Check if linked list is palindrome.

    Strategy:
    1. Find middle of list
    2. Reverse second half
    3. Compare first half with reversed second half

    Time: O(n)
    Space: O(1)
    """
    if not head or not head.next:
        return True

    # Step 1: Find middle using fast/slow pointers
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # Step 2: Reverse second half
    prev = None
    current = slow
    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    # Step 3: Compare both halves
    first_half = head
    second_half = prev  # Head of reversed second half

    while second_half:
        if first_half.val != second_half.val:
            return False
        first_half = first_half.next
        second_half = second_half.next

    return True


def isPalindrome_simple(head: ListNode) -> bool:
    """
    Simple approach: Convert to array and check.

    Time: O(n)
    Space: O(n)
    """
    vals = []
    while head:
        vals.append(head.val)
        head = head.next

    return vals == vals[::-1]
```

### Complexity
- **In-place**: Time O(n), Space O(1)
- **Array**: Time O(n), Space O(n)

### Edge Cases
- Empty list: Return `True` (vacuously true)
- Single node: Return `True`
- Two same nodes: `[1,1]` → `True`
- Two different nodes: `[1,2]` → `False`

### Common Mistakes
- Not finding the correct middle for even/odd length lists
- Forgetting to reverse the second half
- Comparing wrong halves or wrong direction

### Related Problems
- LC #206 Reverse Linked List
- LC #876 Middle of the Linked List
- LC #9 Palindrome Number

---

## Summary: Easy Problems Checklist

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | Reverse Linked List | Three pointers | O(n) | O(1) |
| 2 | Merge Two Sorted | Dummy node + compare | O(n+m) | O(1) |
| 3 | Linked List Cycle | Fast & slow pointers | O(n) | O(1) |
| 4 | Middle of List | Fast & slow pointers | O(n) | O(1) |
| 5 | Remove Duplicates | Skip duplicates | O(n) | O(1) |
| 6 | Intersection | Two pointer switch | O(n+m) | O(1) |
| 7 | Palindrome List | Reverse half + compare | O(n) | O(1) |

---

## Practice More Easy Problems

- [ ] LC #203 - Remove Linked List Elements
- [ ] LC #237 - Delete Node in a Linked List
- [ ] LC #1290 - Convert Binary Number in a Linked List to Integer
- [ ] LC #707 - Design Linked List
