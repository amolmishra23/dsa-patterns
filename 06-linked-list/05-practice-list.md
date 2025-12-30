# Linked List - Complete Practice List

## Organized by Pattern and Difficulty

### Pattern 1: Basic Operations

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 206 | [Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) | Easy | Iterative/Recursive |
| 21 | [Merge Two Sorted Lists](https://leetcode.com/problems/merge-two-sorted-lists/) | Easy | Dummy head |
| 83 | [Remove Duplicates](https://leetcode.com/problems/remove-duplicates-from-sorted-list/) | Easy | Two pointers |
| 203 | [Remove Elements](https://leetcode.com/problems/remove-linked-list-elements/) | Easy | Dummy head |
| 237 | [Delete Node](https://leetcode.com/problems/delete-node-in-a-linked-list/) | Medium | Copy next |
| 876 | [Middle of List](https://leetcode.com/problems/middle-of-the-linked-list/) | Easy | Fast/slow |

### Pattern 2: Fast & Slow Pointers

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 141 | [Linked List Cycle](https://leetcode.com/problems/linked-list-cycle/) | Easy | Floyd's detection |
| 142 | [Linked List Cycle II](https://leetcode.com/problems/linked-list-cycle-ii/) | Medium | Find cycle start |
| 234 | [Palindrome Linked List](https://leetcode.com/problems/palindrome-linked-list/) | Easy | Find mid + reverse |
| 287 | [Find Duplicate Number](https://leetcode.com/problems/find-the-duplicate-number/) | Medium | Array as linked list |
| 457 | [Circular Array Loop](https://leetcode.com/problems/circular-array-loop/) | Medium | Cycle detection |

### Pattern 3: In-Place Reversal

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 92 | [Reverse Linked List II](https://leetcode.com/problems/reverse-linked-list-ii/) | Medium | Reverse sublist |
| 24 | [Swap Nodes in Pairs](https://leetcode.com/problems/swap-nodes-in-pairs/) | Medium | Pairwise swap |
| 25 | [Reverse Nodes in k-Group](https://leetcode.com/problems/reverse-nodes-in-k-group/) | Hard | k-group reversal |
| 61 | [Rotate List](https://leetcode.com/problems/rotate-list/) | Medium | Find new head |

### Pattern 4: Two Pointers / Distance

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 19 | [Remove Nth From End](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) | Medium | n-gap pointers |
| 160 | [Intersection](https://leetcode.com/problems/intersection-of-two-linked-lists/) | Easy | Length difference |
| 328 | [Odd Even Linked List](https://leetcode.com/problems/odd-even-linked-list/) | Medium | Separate chains |

### Pattern 5: Merge & Split

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 23 | [Merge k Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists/) | Hard | Heap / Divide & Conquer |
| 148 | [Sort List](https://leetcode.com/problems/sort-list/) | Medium | Merge sort |
| 86 | [Partition List](https://leetcode.com/problems/partition-list/) | Medium | Two chains |
| 143 | [Reorder List](https://leetcode.com/problems/reorder-list/) | Medium | Split + reverse + merge |

### Pattern 6: Advanced

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 2 | [Add Two Numbers](https://leetcode.com/problems/add-two-numbers/) | Medium | Digit by digit |
| 445 | [Add Two Numbers II](https://leetcode.com/problems/add-two-numbers-ii/) | Medium | Stack or reverse |
| 138 | [Copy List with Random](https://leetcode.com/problems/copy-list-with-random-pointer/) | Medium | Hash map or interleave |
| 146 | [LRU Cache](https://leetcode.com/problems/lru-cache/) | Medium | HashMap + DLL |
| 460 | [LFU Cache](https://leetcode.com/problems/lfu-cache/) | Hard | HashMap + freq buckets |

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LINKED LIST PATTERNS                                   │
│                                                                             │
│  REVERSE LINKED LIST:                                                       │
│                                                                             │
│  Initial:  1 → 2 → 3 → 4 → NULL                                             │
│                                                                             │
│  Step 1:   NULL ← 1   2 → 3 → 4 → NULL                                      │
│            prev  curr next                                                  │
│                                                                             │
│  Step 2:   NULL ← 1 ← 2   3 → 4 → NULL                                      │
│                  prev curr next                                             │
│                                                                             │
│  Step 3:   NULL ← 1 ← 2 ← 3   4 → NULL                                      │
│                       prev curr next                                        │
│                                                                             │
│  Final:    NULL ← 1 ← 2 ← 3 ← 4                                             │
│                            prev (new head)                                  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FAST & SLOW POINTERS (Cycle Detection):                                    │
│                                                                             │
│  1 → 2 → 3 → 4 → 5                                                          │
│            ↑     ↓                                                          │
│            └─────┘  (cycle)                                                 │
│                                                                             │
│  Step 1: slow=1, fast=1                                                     │
│  Step 2: slow=2, fast=3                                                     │
│  Step 3: slow=3, fast=5                                                     │
│  Step 4: slow=4, fast=4  ← MEET! Cycle detected                             │
│                                                                             │
│  No cycle: fast reaches NULL before meeting slow                            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FIND MIDDLE (Fast moves 2x):                                               │
│                                                                             │
│  1 → 2 → 3 → 4 → 5 → NULL                                                   │
│  s   f                        Start: slow=1, fast=1                         │
│      s       f                Step 1: slow=2, fast=3                        │
│          s           f        Step 2: slow=3, fast=5                        │
│          ↑                    Step 3: fast.next=NULL, STOP                  │
│        middle                 Return slow (node 3)                          │
│                                                                             │
│  Even length: 1 → 2 → 3 → 4 → NULL                                          │
│                   s       f   Returns first middle (node 2)                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MERGE TWO SORTED LISTS (Dummy Head):                                       │
│                                                                             │
│  L1: 1 → 3 → 5                                                              │
│  L2: 2 → 4 → 6                                                              │
│                                                                             │
│  dummy → ?                                                                  │
│     ↑                                                                       │
│   curr                                                                      │
│                                                                             │
│  Compare 1 vs 2: pick 1    dummy → 1                                        │
│  Compare 3 vs 2: pick 2    dummy → 1 → 2                                    │
│  Compare 3 vs 4: pick 3    dummy → 1 → 2 → 3                                │
│  Compare 5 vs 4: pick 4    dummy → 1 → 2 → 3 → 4                            │
│  Compare 5 vs 6: pick 5    dummy → 1 → 2 → 3 → 4 → 5                        │
│  L1 empty: attach L2       dummy → 1 → 2 → 3 → 4 → 5 → 6                    │
│                                                                             │
│  Return dummy.next                                                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  REMOVE NTH FROM END (N-Gap Technique):                                     │
│                                                                             │
│  Remove 2nd from end:  1 → 2 → 3 → 4 → 5 → NULL                             │
│                                    ↑                                        │
│                               remove this                                   │
│                                                                             │
│  dummy → 1 → 2 → 3 → 4 → 5 → NULL                                           │
│    ↑                                                                        │
│  first, second                                                              │
│                                                                             │
│  Move first n+1 (3) steps:                                                  │
│  dummy → 1 → 2 → 3 → 4 → 5 → NULL                                           │
│    ↑              ↑                                                         │
│  second        first                                                        │
│                                                                             │
│  Move both until first=NULL:                                                │
│  dummy → 1 → 2 → 3 → 4 → 5 → NULL                                           │
│                   ↑              ↑                                          │
│                second         first                                         │
│                                                                             │
│  Remove: second.next = second.next.next                                     │
│  Result: 1 → 2 → 3 → 5 → NULL                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Essential Templates

### 1. Reverse Linked List
```python
def reverse(head):
    prev = None
    current = head

    while current:
        next_node = current.next
        current.next = prev
        prev = current
        current = next_node

    return prev
```

### 2. Find Middle
```python
def find_middle(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow
```

### 3. Detect Cycle
```python
def has_cycle(head):
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True

    return False
```

### 4. Merge Two Lists
```python
def merge(l1, l2):
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

    current.next = l1 or l2
    return dummy.next
```

### 5. Remove Nth from End
```python
def remove_nth(head, n):
    dummy = ListNode(0, head)
    first = second = dummy

    # Move first n+1 steps ahead
    for _ in range(n + 1):
        first = first.next

    # Move both until first reaches end
    while first:
        first = first.next
        second = second.next

    # Remove nth node
    second.next = second.next.next

    return dummy.next
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] Reverse Linked List
- [ ] Merge Two Sorted Lists
- [ ] Linked List Cycle
- [ ] Middle of List
- [ ] Remove Duplicates

### Week 2: Two Pointers
- [ ] Linked List Cycle II
- [ ] Remove Nth From End
- [ ] Intersection of Two Lists
- [ ] Palindrome Linked List

### Week 3: In-Place Operations
- [ ] Reverse Linked List II
- [ ] Swap Nodes in Pairs
- [ ] Odd Even Linked List
- [ ] Reorder List

### Week 4: Advanced
- [ ] Merge k Sorted Lists
- [ ] Sort List
- [ ] Add Two Numbers
- [ ] Copy List with Random
- [ ] LRU Cache

---

## Common Mistakes to Avoid

1. **Forgetting to handle empty list**
   ```python
   if not head:
       return None
   ```

2. **Losing reference to head**
   ```python
   # Use dummy node
   dummy = ListNode(0, head)
   # Work with dummy.next
   return dummy.next
   ```

3. **Not updating pointers correctly**
   ```python
   # Save next before changing
   next_node = current.next
   current.next = prev
   ```

4. **Off-by-one in n-steps ahead**
   ```python
   # For removing nth from end, move n+1 steps
   for _ in range(n + 1):
       first = first.next
   ```

5. **Infinite loop in cycle detection**
   ```python
   # Always check fast AND fast.next
   while fast and fast.next:
   ```

---

## Complexity Reference

| Operation | Time | Space |
|-----------|------|-------|
| Access nth | O(n) | O(1) |
| Search | O(n) | O(1) |
| Insert at head | O(1) | O(1) |
| Insert at tail | O(n) | O(1) |
| Delete | O(n) | O(1) |
| Reverse | O(n) | O(1) |
| Find middle | O(n) | O(1) |
| Detect cycle | O(n) | O(1) |

---

## Tips for Interviews

1. **Always clarify**: Singly or doubly linked? Sorted? Cycles possible?

2. **Draw it out**: Visualize pointer movements before coding

3. **Use dummy node**: Simplifies edge cases (empty list, single node)

4. **Consider edge cases**:
   - Empty list
   - Single node
   - Two nodes
   - Cycle at head/tail

5. **Practice both iterative and recursive** solutions

