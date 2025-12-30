# Trees BFS - Practice Problems

## Problem 1: Binary Tree Level Order Traversal (LC #102) - Medium

- [LeetCode](https://leetcode.com/problems/binary-tree-level-order-traversal/)

### Problem Statement
Return level order traversal of binary tree.

### Video Explanation
- [NeetCode - Level Order Traversal](https://www.youtube.com/watch?v=6ZnyEApgFYg)
- [Take U Forward - BFS Traversal](https://www.youtube.com/watch?v=EoAsWbO7sqg)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BFS LEVEL BY LEVEL                                                         │
│                                                                             │
│       3         Level 0: [3]                                               │
│      / \                                                                    │
│     9  20       Level 1: [9, 20]                                           │
│       /  \                                                                  │
│      15   7     Level 2: [15, 7]                                           │
│                                                                             │
│  Use a QUEUE to process nodes level by level:                              │
│                                                                             │
│  Queue: [3]                                                                 │
│  Process 3, add children → Queue: [9, 20], Output: [[3]]                   │
│                                                                             │
│  Queue: [9, 20]  (level_size = 2)                                          │
│  Process 9, 20, add children → Queue: [15, 7], Output: [[3], [9, 20]]     │
│                                                                             │
│  Queue: [15, 7]  (level_size = 2)                                          │
│  Process 15, 7 → Queue: [], Output: [[3], [9, 20], [15, 7]]               │
│                                                                             │
│  Key: Track level_size to know when one level ends and next begins         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Examples
```
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]
```

### Solution
```python
from collections import deque
from typing import Optional, List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Level order traversal using BFS.

    Strategy:
    - Use queue to process nodes level by level
    - Track level size to group nodes

    Time: O(n) - visit each node once
    Space: O(n) - queue can hold up to n/2 nodes (last level)
    """
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        # Process all nodes at current level
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            # Add children for next level
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

### Edge Cases
- Empty tree → return []
- Single node → return [[val]]
- Skewed tree (all left/right) → one node per level
- Complete binary tree → levels grow exponentially

---

## Problem 2: Binary Tree Zigzag Level Order (LC #103) - Medium

- [LeetCode](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/)

### Problem Statement
Return zigzag level order (alternating left-right, right-left).

### Video Explanation
- [NeetCode - Zigzag Level Order](https://www.youtube.com/watch?v=igbboQbiwqw)


### Intuition
```
Zigzag: alternate direction each level

Level 0: left→right  [3]
Level 1: right→left  [20, 9]
Level 2: left→right  [15, 7]

Use deque for efficient append to either end.
```

### Examples
```
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[20,9],[15,7]]
```

### Solution
```python
def zigzagLevelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Zigzag level order traversal.

    Strategy:
    - Standard BFS with level tracking
    - Reverse alternate levels

    Time: O(n)
    Space: O(n)
    """
    if not root:
        return []

    result = []
    queue = deque([root])
    left_to_right = True

    while queue:
        level_size = len(queue)
        level = deque()

        for _ in range(level_size):
            node = queue.popleft()

            # Add to level based on direction
            if left_to_right:
                level.append(node.val)
            else:
                level.appendleft(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(list(level))
        left_to_right = not left_to_right

    return result
```

### Edge Cases
- Empty tree → return []
- Single node → return [[val]]
- Two levels → second level reversed
- Odd vs even levels → different directions

---

## Problem 3: Binary Tree Right Side View (LC #199) - Medium

- [LeetCode](https://leetcode.com/problems/binary-tree-right-side-view/)

### Problem Statement
Return values visible from right side.

### Video Explanation
- [NeetCode - Right Side View](https://www.youtube.com/watch?v=d4zLyf32e3I)


### Intuition
```
Right side view = last node at each level

      1       ← see 1
     / \
    2   3     ← see 3 (rightmost)
     \   \
      5   4   ← see 4 (rightmost)

BFS: take last node of each level
DFS: visit right first, track depth
```

### Examples
```
Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]

      1          <-- 1
     / \
    2   3        <-- 3
     \   \
      5   4      <-- 4
```

### Solution
```python
def rightSideView(root: Optional[TreeNode]) -> List[int]:
    """
    Right side view using BFS.

    Strategy:
    - BFS level by level
    - Take last node of each level

    Time: O(n)
    Space: O(n)
    """
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)

        for i in range(level_size):
            node = queue.popleft()

            # Last node in level is visible from right
            if i == level_size - 1:
                result.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return result


def rightSideView_dfs(root: Optional[TreeNode]) -> List[int]:
    """
    Alternative: DFS with depth tracking.

    Visit right child first, add first node at each depth.
    """
    result = []

    def dfs(node, depth):
        if not node:
            return

        # First node at this depth (rightmost due to traversal order)
        if depth == len(result):
            result.append(node.val)

        # Visit right first
        dfs(node.right, depth + 1)
        dfs(node.left, depth + 1)

    dfs(root, 0)
    return result
```

### Edge Cases
- Empty tree → return []
- Single node → return [val]
- All left children → see leftmost path
- All right children → see rightmost path

---

## Problem 4: Average of Levels (LC #637) - Easy

- [LeetCode](https://leetcode.com/problems/average-of-levels-in-binary-tree/)

### Problem Statement
Return average value of nodes at each level.

### Video Explanation
- [NeetCode - Average of Levels](https://www.youtube.com/watch?v=Y0_H3LHQG0w)


### Intuition
```
Average = sum / count for each level

Level 0: sum=3, count=1 → avg=3.0
Level 1: sum=29, count=2 → avg=14.5
Level 2: sum=22, count=2 → avg=11.0
```

### Solution
```python
def averageOfLevels(root: Optional[TreeNode]) -> List[float]:
    """
    Calculate average at each level.

    Time: O(n)
    Space: O(n)
    """
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level_sum = 0

        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level_sum / level_size)

    return result
```

### Edge Cases
- Empty tree → return []
- Single node → return [val as float]
- Large values → watch for overflow
- Negative values → handled correctly

---

## Problem 5: Minimum Depth of Binary Tree (LC #111) - Easy

- [LeetCode](https://leetcode.com/problems/minimum-depth-of-binary-tree/)

### Problem Statement
Find minimum depth (root to nearest leaf).

### Video Explanation
- [NeetCode - Minimum Depth](https://www.youtube.com/watch?v=tZS4VHtbYoo)


### Intuition
```
BFS finds shortest path first!

      3
     / \
    9  20    ← 9 is leaf, depth=2
      /  \
     15   7

BFS stops at first leaf found.
```

### Solution
```python
def minDepth(root: Optional[TreeNode]) -> int:
    """
    Find minimum depth using BFS.

    BFS is optimal here - finds shortest path first.

    Time: O(n) worst case, but often much less
    Space: O(n)
    """
    if not root:
        return 0

    queue = deque([(root, 1)])  # (node, depth)

    while queue:
        node, depth = queue.popleft()

        # Found a leaf - this is minimum depth
        if not node.left and not node.right:
            return depth

        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))

    return 0
```

### Edge Cases
- Empty tree → return 0
- Single node → return 1
- Skewed tree → depth equals node count
- Leaf at level 1 → return 1

---

## Problem 6: Populating Next Right Pointers (LC #116) - Medium

- [LeetCode](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/)

### Problem Statement
Connect each node to its next right node (perfect binary tree).

### Video Explanation
- [NeetCode - Populating Next Right Pointers](https://www.youtube.com/watch?v=U4hFQCa1Cq0)


### Intuition
```
Connect siblings at same level:

    1 → NULL
   / \
  2 → 3 → NULL
 / \ / \
4→5→6→7 → NULL

O(1) space: use established next pointers
to traverse current level.
```

### Solution
```python
class Node:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


def connect(root: Optional[Node]) -> Optional[Node]:
    """
    Connect nodes at same level using BFS.

    Time: O(n)
    Space: O(n)
    """
    if not root:
        return None

    queue = deque([root])

    while queue:
        level_size = len(queue)
        prev = None

        for _ in range(level_size):
            node = queue.popleft()

            if prev:
                prev.next = node
            prev = node

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return root


def connect_O1_space(root: Optional[Node]) -> Optional[Node]:
    """
    O(1) space solution using established next pointers.

    Use current level's next pointers to traverse,
    while setting up next level's next pointers.
    """
    if not root:
        return None

    leftmost = root

    while leftmost.left:
        # Traverse current level using next pointers
        current = leftmost

        while current:
            # Connect left child to right child
            current.left.next = current.right

            # Connect right child to next node's left child
            if current.next:
                current.right.next = current.next.left

            current = current.next

        # Move to next level
        leftmost = leftmost.left

    return root
```

### Edge Cases
- Empty tree → return None
- Single node → next is None
- Last node in level → next is None
- Perfect tree guaranteed

---

## Problem 7: Populating Next Right Pointers II (LC #117) - Medium

- [LeetCode](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)

### Problem Statement
Same as above but for any binary tree (not necessarily perfect).

### Video Explanation
- [NeetCode - Populating Next Right Pointers II](https://www.youtube.com/watch?v=yl-fdkyQD8A)


### Intuition
```
Non-perfect tree - some nodes missing:

    1 → NULL
   / \
  2 → 3 → NULL
 /     \
4   →   7 → NULL

Use dummy node to track next level start.
```

### Solution
```python
def connect_any_tree(root: Optional[Node]) -> Optional[Node]:
    """
    Connect nodes in any binary tree.

    Strategy:
    - Use dummy node to track start of next level
    - Use tail pointer to build next level connections

    Time: O(n)
    Space: O(1)
    """
    if not root:
        return None

    current = root

    while current:
        # Dummy node for next level
        dummy = Node(0)
        tail = dummy

        # Traverse current level
        while current:
            if current.left:
                tail.next = current.left
                tail = tail.next
            if current.right:
                tail.next = current.right
                tail = tail.next

            current = current.next

        # Move to next level
        current = dummy.next

    return root
```

### Edge Cases
- Empty tree → return None
- Single node → next is None
- Sparse tree → skip missing nodes
- All nodes on one side → chain correctly

---

## Problem 8: Cousins in Binary Tree (LC #993) - Easy

- [LeetCode](https://leetcode.com/problems/cousins-in-binary-tree/)

### Problem Statement
Check if two nodes are cousins (same depth, different parents).

### Video Explanation
- [NeetCode - Cousins in Binary Tree](https://www.youtube.com/watch?v=xA3RmHskpM0)


### Intuition
```
Cousins: same depth, different parents

      1
     / \
    2   3
   /     \
  4       5

4 and 5 are cousins (depth=2, parents=2,3)
4 and 3 are NOT cousins (different depths)
```

### Solution
```python
def isCousins(root: Optional[TreeNode], x: int, y: int) -> bool:
    """
    Check if x and y are cousins using BFS.

    Cousins: same depth, different parents.

    Time: O(n)
    Space: O(n)
    """
    if not root:
        return False

    queue = deque([(root, None)])  # (node, parent)

    while queue:
        level_size = len(queue)
        x_parent = y_parent = None

        for _ in range(level_size):
            node, parent = queue.popleft()

            if node.val == x:
                x_parent = parent
            if node.val == y:
                y_parent = parent

            if node.left:
                queue.append((node.left, node))
            if node.right:
                queue.append((node.right, node))

        # Both found at this level
        if x_parent and y_parent:
            return x_parent != y_parent

        # One found but not other - not same depth
        if x_parent or y_parent:
            return False

    return False
```

### Edge Cases
- x or y is root → cannot be cousins
- x and y are siblings → not cousins (same parent)
- x or y not in tree → return False
- Same node for x and y → return False

---

## Problem 9: Binary Tree Level Order Traversal II (LC #107) - Medium

- [LeetCode](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/)

### Problem Statement
Return bottom-up level order traversal.

### Video Explanation
- [NeetCode - Level Order Traversal II](https://www.youtube.com/watch?v=6ZnyEApgFYg)


### Intuition
```
Bottom-up = reverse of top-down

Top-down: [[3], [9,20], [15,7]]
Bottom-up: [[15,7], [9,20], [3]]

Just reverse the result array.
```

### Solution
```python
def levelOrderBottom(root: Optional[TreeNode]) -> List[List[int]]:
    """
    Bottom-up level order traversal.

    Strategy: Standard BFS, then reverse result.

    Time: O(n)
    Space: O(n)
    """
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result[::-1]
```

### Edge Cases
- Empty tree → return []
- Single node → return [[val]]
- Multiple levels → reverse order
- Same as level order but reversed

---

## Problem 10: N-ary Tree Level Order Traversal (LC #429) - Medium

- [LeetCode](https://leetcode.com/problems/n-ary-tree-level-order-traversal/)

### Problem Statement
Level order traversal of N-ary tree.

### Video Explanation
- [NeetCode - N-ary Tree Level Order](https://www.youtube.com/watch?v=MY2_HOFA2Bc)


### Intuition
```
N-ary tree: each node has list of children

      1
   /  |  \
  3   2   4
 / \
5   6

Same BFS, but iterate over children list
instead of just left/right.
```

### Solution
```python
class NaryNode:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children else []


def levelOrder_nary(root: Optional[NaryNode]) -> List[List[int]]:
    """
    Level order traversal of N-ary tree.

    Same as binary tree, but iterate over all children.

    Time: O(n)
    Space: O(n)
    """
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            # Add all children
            for child in node.children:
                queue.append(child)

        result.append(level)

    return result
```

### Edge Cases
- Empty tree → return []
- Single node → return [[val]]
- Node with no children → leaf node
- Variable children count → handle all

---

## Summary: Tree BFS Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Level Order | Standard BFS | O(n) |
| 2 | Zigzag | BFS + alternate direction | O(n) |
| 3 | Right Side View | Last node per level | O(n) |
| 4 | Average of Levels | Sum per level | O(n) |
| 5 | Minimum Depth | BFS finds shortest | O(n) |
| 6 | Next Pointers (Perfect) | BFS or O(1) space | O(n) |
| 7 | Next Pointers (Any) | Dummy node technique | O(n) |
| 8 | Cousins | Track parent + depth | O(n) |
| 9 | Bottom-up Order | BFS + reverse | O(n) |
| 10 | N-ary Level Order | BFS with children list | O(n) |

---

## BFS Template

```python
def bfs_template(root):
    """Standard BFS template for trees."""
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()

            # Process node
            level.append(node.val)

            # Add children
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

---

## Practice More Problems

- [ ] LC #515 - Find Largest Value in Each Tree Row
- [ ] LC #623 - Add One Row to Tree
- [ ] LC #662 - Maximum Width of Binary Tree
- [ ] LC #958 - Check Completeness of a Binary Tree

