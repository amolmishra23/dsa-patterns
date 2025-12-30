# Trees BFS - Advanced Problems

## Problem 1: Serialize and Deserialize Binary Tree (LC #297) - Hard

- [LeetCode](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

### Problem Statement
Design algorithm to serialize and deserialize a binary tree using BFS.

### Examples
```
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]

     1
    / \
   2   3
      / \
     4   5
```

### Intuition Development
```
SERIALIZATION (BFS):
                1
               / \
              2   3
                 / \
                4   5

Level-order traversal:
  Queue: [1]
  Process 1 → result: ["1"], Queue: [2, 3]
  Process 2 → result: ["1","2"], Queue: [3, null, null]
  Process 3 → result: ["1","2","3"], Queue: [null, null, 4, 5]
  Process nulls and children...

Final: "1,2,3,N,N,4,5"

DESERIALIZATION:
  String: "1,2,3,N,N,4,5"

  Create root 1, Queue: [1]
  For node 1: left=2, right=3, Queue: [2, 3]
  For node 2: left=N, right=N, Queue: [3]
  For node 3: left=4, right=5, Queue: [4, 5]
  ...
```

### Video Explanation
- [NeetCode - Serialize and Deserialize Binary Tree](https://www.youtube.com/watch?v=u4JAi2JJhI8)

### Solution
```python
from collections import deque
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Codec:
    """
    Serialize/deserialize using level-order traversal.
    """

    def serialize(self, root: Optional[TreeNode]) -> str:
        """Encode tree to string."""
        if not root:
            return ''

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

        # Remove trailing nulls
        while result and result[-1] == 'N':
            result.pop()

        return ','.join(result)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Decode string to tree."""
        if not data:
            return None

        values = data.split(',')
        root = TreeNode(int(values[0]))
        queue = deque([root])
        i = 1

        while queue and i < len(values):
            node = queue.popleft()

            # Left child
            if i < len(values) and values[i] != 'N':
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

### Complexity
- **Time**: O(n) for both operations
- **Space**: O(n) for storing result/queue

### Edge Cases
- Empty tree → return empty string
- Single node tree → "1"
- Skewed tree (all left or all right children)
- Negative values in nodes

---

## Problem 2: Vertical Order Traversal (LC #987) - Hard

- [LeetCode](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)

### Problem Statement
Return vertical order traversal with proper sorting.

### Examples
```
Input:
        3
       / \
      9  20
         / \
        15  7

Output: [[9],[3,15],[20],[7]]
```

### Intuition Development
```
ASSIGN COORDINATES:
              (0,0) 3
             /       \
        (-1,1) 9    (1,1) 20
                    /       \
               (0,2) 15   (2,2) 7

Column -1: [9]
Column 0:  [3, 15] → sorted by row, then value
Column 1:  [20]
Column 2:  [7]

BFS with (node, row, col):
  Start: [(3, 0, 0)]
  Process 3: col[0] = [(0, 3)]
             Add (9, 1, -1) and (20, 1, 1)
  Process 9: col[-1] = [(1, 9)]
  Process 20: col[1] = [(1, 20)]
              Add (15, 2, 0) and (7, 2, 2)
  ...
```

### Video Explanation
- [NeetCode - Vertical Order Traversal](https://www.youtube.com/watch?v=q_a6lpbKJdw)

### Solution
```python
def verticalTraversal(root: Optional[TreeNode]) -> list[list[int]]:
    """
    Vertical order with BFS and sorting.

    Rules:
    - Sort by column (left to right)
    - Within column, sort by row (top to bottom)
    - Same row and column, sort by value
    """
    from collections import defaultdict

    # Store (row, val) for each column
    columns = defaultdict(list)
    queue = deque([(root, 0, 0)])  # (node, row, col)

    while queue:
        node, row, col = queue.popleft()

        if node:
            columns[col].append((row, node.val))
            queue.append((node.left, row + 1, col - 1))
            queue.append((node.right, row + 1, col + 1))

    # Sort columns and extract values
    result = []
    for col in sorted(columns.keys()):
        # Sort by row, then by value
        result.append([val for row, val in sorted(columns[col])])

    return result
```

### Complexity
- **Time**: O(n log n) due to sorting
- **Space**: O(n) for storing nodes

### Edge Cases
- Single node → [[node.val]]
- All nodes in same column (straight line tree)
- Nodes with same position and value

---

## Problem 3: Populating Next Right Pointers II (LC #117) - Medium

- [LeetCode](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/)

### Problem Statement
Populate next pointer for each node (not perfect binary tree).

### Examples
```
Input:          1               Output:       1 → NULL
               / \                           / \
              2   3                         2 → 3 → NULL
             / \   \                       / \   \
            4   5   7                     4→ 5 → 7 → NULL
```

### Intuition Development
```
LEVEL-BY-LEVEL PROCESSING:
Level 0: [1] → 1.next = NULL

Level 1: [2, 3] → 2.next = 3, 3.next = NULL

Level 2: [4, 5, 7] → 4.next = 5, 5.next = 7, 7.next = NULL

O(1) SPACE APPROACH:
Use established next pointers as a "virtual queue":

  Current Level: 1
  Next Level Build:
    1.left = 2, 1.right = 3
    prev = None → leftmost = 2
    2.next = 3 (via prev tracking)

  Move to next level using leftmost pointer
```

### Video Explanation
- [NeetCode - Populating Next Right Pointers II](https://www.youtube.com/watch?v=U4hFQCa1Cq0)

### Solution
```python
class Node:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


def connect(root: Node) -> Node:
    """
    Connect next pointers level by level.
    O(1) space using next pointers as queue.
    """
    if not root:
        return None

    # Process level by level using next pointers
    leftmost = root

    while leftmost:
        # Traverse current level, connect next level
        curr = leftmost
        prev = None
        leftmost = None

        while curr:
            # Process left child
            if curr.left:
                if prev:
                    prev.next = curr.left
                else:
                    leftmost = curr.left
                prev = curr.left

            # Process right child
            if curr.right:
                if prev:
                    prev.next = curr.right
                else:
                    leftmost = curr.right
                prev = curr.right

            curr = curr.next

    return root


def connect_bfs(root: Node) -> Node:
    """
    Alternative: Standard BFS with O(n) space.
    """
    if not root:
        return None

    queue = deque([root])

    while queue:
        size = len(queue)
        prev = None

        for _ in range(size):
            node = queue.popleft()

            if prev:
                prev.next = node
            prev = node

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return root
```

### Complexity
- **Time**: O(n)
- **Space**: O(1) for optimized, O(n) for BFS approach

### Edge Cases
- Empty tree → return None
- Single node → node.next = None
- Perfect binary tree (all levels full)
- Skewed tree (only left or right children)

---

## Problem 4: Complete Binary Tree Inserter (LC #919) - Medium

- [LeetCode](https://leetcode.com/problems/complete-binary-tree-inserter/)

### Problem Statement
Design data structure for complete binary tree with O(1) insert.

### Examples
```
Initial:        1
               / \
              2   3
             /
            4

Insert(5):     1
              / \
             2   3
            / \
           4   5
```

### Intuition Development
```
COMPLETE BINARY TREE PROPERTY:
All levels filled except possibly last, which fills left-to-right.

TRACK INCOMPLETE NODES:
Use deque to track nodes that can accept children.

Initial tree:
        1
       / \
      2   3
     /
    4

Incomplete nodes: [2, 3, 4] → 2 has right slot, 3 has both, 4 is leaf

Insert(5):
  Parent = deque[0] = 2
  2 has no right → 2.right = 5
  2 is now complete → pop from deque
  Add 5 to deque

Deque after: [3, 4, 5]
```

### Video Explanation
- [NeetCode - Complete Binary Tree Inserter](https://www.youtube.com/watch?v=Vj7-q0P9jE4)

### Solution
```python
class CBTInserter:
    """
    Complete binary tree inserter.

    Strategy:
    - Keep deque of nodes that can accept children
    - Front of deque is next parent for insertion
    """

    def __init__(self, root: TreeNode):
        self.root = root
        self.deque = deque()

        # BFS to find incomplete nodes
        queue = deque([root])
        while queue:
            node = queue.popleft()

            if not node.left or not node.right:
                self.deque.append(node)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    def insert(self, val: int) -> int:
        """Insert value and return parent's value."""
        node = TreeNode(val)
        parent = self.deque[0]

        if not parent.left:
            parent.left = node
        else:
            parent.right = node
            self.deque.popleft()

        self.deque.append(node)
        return parent.val

    def get_root(self) -> TreeNode:
        return self.root
```

### Complexity
- **Time**: O(n) init, O(1) insert
- **Space**: O(n) for deque

### Edge Cases
- Single node tree
- Tree with only left child at last level
- Large number of consecutive inserts

---

## Problem 5: Maximum Width of Binary Tree (LC #662) - Medium

- [LeetCode](https://leetcode.com/problems/maximum-width-of-binary-tree/)

### Problem Statement
Find maximum width at any level (including nulls).

### Examples
```
Input:
        1
       / \
      3   2
     / \   \
    5   3   9

Width calculation:
Level 0: 1 node → width = 1
Level 1: 2 nodes → width = 2
Level 2: positions 0,1,3 → width = 3-0+1 = 4

Output: 4 (level 2 from 5 to 9, including null)
```

### Intuition Development
```
POSITION INDEXING:
Assign positions like heap:
  root = 0
  left child = 2*pos
  right child = 2*pos + 1

        1 (pos=0)
       / \
      3   2 (pos=0,1)
     / \   \
    5   3   9 (pos=0,1,3)

Level 2: rightmost(3) - leftmost(0) + 1 = 4

NORMALIZATION to prevent overflow:
At each level, subtract first position from all positions.
```

### Video Explanation
- [NeetCode - Maximum Width of Binary Tree](https://www.youtube.com/watch?v=FPzLE2L7uHs)

### Solution
```python
def widthOfBinaryTree(root: Optional[TreeNode]) -> int:
    """
    Maximum width using position indexing.

    Strategy:
    - Assign positions: root=0, left=2*pos, right=2*pos+1
    - Width = rightmost - leftmost + 1
    """
    if not root:
        return 0

    max_width = 0
    queue = deque([(root, 0)])  # (node, position)

    while queue:
        level_size = len(queue)
        _, first_pos = queue[0]

        for i in range(level_size):
            node, pos = queue.popleft()

            # Normalize position to prevent overflow
            normalized_pos = pos - first_pos

            if i == level_size - 1:
                max_width = max(max_width, normalized_pos + 1)

            if node.left:
                queue.append((node.left, 2 * normalized_pos))
            if node.right:
                queue.append((node.right, 2 * normalized_pos + 1))

    return max_width
```

### Complexity
- **Time**: O(n)
- **Space**: O(n) for queue

### Edge Cases
- Single node → width = 1
- Perfect binary tree
- Skewed tree (width = 1 at each level)
- Very deep tree (position overflow without normalization)

---

## Problem 6: Cousins in Binary Tree (LC #993) - Easy

- [LeetCode](https://leetcode.com/problems/cousins-in-binary-tree/)

### Video Explanation
- [NeetCode - Cousins in Binary Tree](https://www.youtube.com/watch?v=wz9lIjqFZfQ)

### Problem Statement
Check if two nodes are cousins (same depth, different parents).

### Examples
```
Input: root = [1,2,3,4], x = 4, y = 3
        1
       / \
      2   3
     /
    4

Output: false (4 is at depth 2, 3 is at depth 1)
```

### Intuition Development
```
COUSIN CONDITIONS:
1. Same depth (level)
2. Different parents

BFS TRACKING:
  Level 0: [(1, null)]  → not x or y
  Level 1: [(2, 1), (3, 1)]
           y=3 found with parent 1
  Level 2: [(4, 2)]
           x=4 found with parent 2

x and y at different levels → NOT cousins

Example with cousins:
        1
       / \
      2   3
     /     \
    4       5

x=4, y=5:
  Level 2: 4 parent=2, 5 parent=3
  Same level, different parents → COUSINS
```

### Solution
```python
def isCousins(root: Optional[TreeNode], x: int, y: int) -> bool:
    """
    Check if x and y are cousins using BFS.

    Cousins: same depth, different parents
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

        # Only one found - not cousins
        if x_parent or y_parent:
            return False

    return False
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- x and y are siblings (same parent) → false
- x or y is the root → no cousins possible
- x and y at different levels → false
- x and y don't exist in tree

---

## Problem 7: Even Odd Tree (LC #1609) - Medium

- [LeetCode](https://leetcode.com/problems/even-odd-tree/)

### Video Explanation
- [NeetCode - Even Odd Tree](https://www.youtube.com/watch?v=_E7KF3nBfqg)

### Problem Statement
Check if tree satisfies even-odd level conditions.

### Examples
```
Even-Odd Rules:
- Even levels (0, 2, 4...): strictly increasing ODD values
- Odd levels (1, 3, 5...): strictly decreasing EVEN values

Input:
        1         Level 0: odd, increasing ✓
       / \
      10  4       Level 1: even, 10 > 4 ✓
     /  \  \
    3    7  6     Level 2: odd, 3 < 7... wait, 7 > 6? ✗

Output: false
```

### Intuition Development
```
VALIDATION AT EACH LEVEL:

Level 0 (even): values must be ODD and STRICTLY INCREASING
  prev = -∞, val must be > prev and val % 2 == 1

Level 1 (odd): values must be EVEN and STRICTLY DECREASING
  prev = +∞, val must be < prev and val % 2 == 0

Process level by level, validate each node.
```

### Solution
```python
def isEvenOddTree(root: Optional[TreeNode]) -> bool:
    """
    Check even-odd tree conditions.

    Even levels: strictly increasing odd values
    Odd levels: strictly decreasing even values
    """
    queue = deque([root])
    level = 0

    while queue:
        level_size = len(queue)
        prev = float('-inf') if level % 2 == 0 else float('inf')

        for _ in range(level_size):
            node = queue.popleft()

            if level % 2 == 0:
                # Even level: odd values, strictly increasing
                if node.val % 2 == 0 or node.val <= prev:
                    return False
            else:
                # Odd level: even values, strictly decreasing
                if node.val % 2 == 1 or node.val >= prev:
                    return False

            prev = node.val

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        level += 1

    return True
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- Single node at level 0 must be odd
- All nodes with same value (fails strictly increasing/decreasing)
- Very large values (still need correct parity)

---

## Problem 8: Deepest Leaves Sum (LC #1302) - Medium

- [LeetCode](https://leetcode.com/problems/deepest-leaves-sum/)

### Video Explanation
- [NeetCode - Deepest Leaves Sum](https://www.youtube.com/watch?v=u-HgyUq3Eyc)

### Problem Statement
Return sum of values of deepest leaves.

### Examples
```
Input:
        1
       / \
      2   3
     / \   \
    4   5   6
   /         \
  7           8

Deepest level: [7, 8]
Output: 7 + 8 = 15
```

### Intuition Development
```
BFS APPROACH:
Process level by level, keep only the last level's sum.

Level 0: sum = 1
Level 1: sum = 2 + 3 = 5
Level 2: sum = 4 + 5 + 6 = 15
Level 3: sum = 7 + 8 = 15

Return last computed sum (level 3) = 15
```

### Solution
```python
def deepestLeavesSum(root: Optional[TreeNode]) -> int:
    """
    Sum of deepest level using BFS.
    """
    queue = deque([root])

    while queue:
        level_sum = 0
        level_size = len(queue)

        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    # Last level_sum is the deepest level
    return level_sum
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- Single node → return its value
- Perfect binary tree → sum all leaves
- Skewed tree → return single deepest leaf

---

## Problem 9: Add One Row to Tree (LC #623) - Medium

- [LeetCode](https://leetcode.com/problems/add-one-row-to-tree/)

### Video Explanation
- [NeetCode - Add One Row to Tree](https://www.youtube.com/watch?v=juj9zTpNtMw)

### Problem Statement
Add row of nodes at given depth.

### Examples
```
Input: depth = 2, val = 1
        4                   4
       / \                 / \
      2   6      →        1   1
     / \ / \             /     \
    3  1 5              2       6
                       / \     / \
                      3   1   5   (null)
```

### Intuition Development
```
INSERT AT DEPTH d:
1. Find nodes at depth d-1
2. Insert new nodes between parent and original children

Before (depth-1 nodes): [4]
Insert val=1 at depth 2:
  4.left → new_node(1) → original 4.left (2)
  4.right → new_node(1) → original 4.right (6)

Special case: depth = 1
  Create new root with val
  Original root becomes left child
```

### Solution
```python
def addOneRow(root: Optional[TreeNode], val: int, depth: int) -> Optional[TreeNode]:
    """
    Add row at specified depth.
    """
    # Special case: insert at root
    if depth == 1:
        new_root = TreeNode(val)
        new_root.left = root
        return new_root

    queue = deque([root])
    current_depth = 1

    while queue:
        if current_depth == depth - 1:
            # Insert new row
            for node in queue:
                old_left = node.left
                old_right = node.right

                node.left = TreeNode(val)
                node.right = TreeNode(val)
                node.left.left = old_left
                node.right.right = old_right

            return root

        level_size = len(queue)
        for _ in range(level_size):
            node = queue.popleft()

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        current_depth += 1

    return root
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- depth = 1 → new root
- depth = 2 → insert just below root
- depth > tree height → no change (depends on interpretation)

---

## Problem 10: Find Bottom Left Tree Value (LC #513) - Medium

- [LeetCode](https://leetcode.com/problems/find-bottom-left-tree-value/)

### Video Explanation
- [NeetCode - Find Bottom Left Tree Value](https://www.youtube.com/watch?v=u_by_cTsNJA)

### Problem Statement
Find leftmost value in last row.

### Examples
```
Input:
        1
       / \
      2   3
     /   / \
    4   5   6
       /
      7

Last row: [7]
Output: 7 (leftmost of last row)
```

### Intuition Development
```
TRICK: Process RIGHT to LEFT
BFS from right to left, last node processed is bottom-left!

Standard BFS (left to right): 1, 2, 3, 4, 5, 6, 7
Right-to-left BFS: 1, 3, 2, 6, 5, 4, 7

Last processed = 7 = bottom left!

Alternative: Track first node of each level
```

### Solution
```python
def findBottomLeftValue(root: Optional[TreeNode]) -> int:
    """
    Find bottom-left value using BFS.

    Trick: Process right to left, last node is answer.
    """
    queue = deque([root])
    node = root

    while queue:
        node = queue.popleft()

        # Add right first, then left
        # Last processed node will be bottom-left
        if node.right:
            queue.append(node.right)
        if node.left:
            queue.append(node.left)

    return node.val


def findBottomLeftValue_standard(root: Optional[TreeNode]) -> int:
    """
    Standard approach: track first node of each level.
    """
    queue = deque([root])
    result = root.val

    while queue:
        level_size = len(queue)

        for i in range(level_size):
            node = queue.popleft()

            if i == 0:
                result = node.val

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

    return result
```

### Complexity
- **Time**: O(n)
- **Space**: O(n)

### Edge Cases
- Single node → return its value
- Only left children (skewed left)
- Only right children (skewed right) → still returns bottom-left

---

## Summary: Tree BFS Advanced Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Serialize Tree | Level-order encoding | O(n) |
| 2 | Vertical Order | Column + row tracking | O(n log n) |
| 3 | Next Pointers II | O(1) space with next | O(n) |
| 4 | CBT Inserter | Deque of incomplete nodes | O(1) insert |
| 5 | Max Width | Position indexing | O(n) |
| 6 | Cousins | Track parent at level | O(n) |
| 7 | Even Odd Tree | Level-wise validation | O(n) |
| 8 | Deepest Sum | Last level sum | O(n) |
| 9 | Add Row | Insert at depth-1 | O(n) |
| 10 | Bottom Left | Right-to-left BFS | O(n) |

---

## BFS Template for Trees

```python
from collections import deque

def bfs_template(root):
    """Generic tree BFS template."""
    if not root:
        return result_for_empty

    queue = deque([root])
    result = []

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

    return result
```

---

## Practice More Problems

- [ ] LC #314 - Binary Tree Vertical Order Traversal
- [ ] LC #515 - Find Largest Value in Each Tree Row
- [ ] LC #958 - Check Completeness of a Binary Tree
- [ ] LC #1161 - Maximum Level Sum of a Binary Tree
- [ ] LC #1602 - Find Nearest Right Node in Binary Tree
