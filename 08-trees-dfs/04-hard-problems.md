# Trees DFS - Hard Problems

## Problem 1: Binary Tree Maximum Path Sum (LC #124) - Hard

- [LeetCode](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

### Video Explanation
- [NeetCode - Binary Tree Maximum Path Sum](https://www.youtube.com/watch?v=Hr5cWUld4vU)

### Problem Statement
Find maximum path sum where path can start and end at any node.

### Examples
```
Input: root = [-10,9,20,null,null,15,7]
Output: 42 (15 → 20 → 7)

        -10
        /  \
       9   20
          /  \
         15   7
```


### Visual Intuition
```
Binary Tree Maximum Path Sum
        -10
        /  \
       9    20
           /  \
          15   7

Pattern: Post-order DFS with Two Values per Node
Why: At each node, track both "through path" and "return path"

Step 0 (Understand Two Computations):
  ┌─────────────────────────────────────────────┐
  │ At each node:                               │
  │                                             │
  │ 1. Through path (can split):                │
  │        ●                                    │
  │       /│\    left + node + right            │
  │      ← │ →   (update global max)            │
  │                                             │
  │ 2. Return path (no split):                  │
  │        ●                                    │
  │       /│    node + max(left, right)         │
  │      ← ↑    (return to parent)              │
  └─────────────────────────────────────────────┘

Step 1 (Process Leaves - Post-order):
  Node 9:  left=0, right=0
           through = 0 + 9 + 0 = 9
           return = 9 + max(0,0) = 9
           global_max = 9

  Node 15: through = 15, return = 15
           global_max = 15

  Node 7:  through = 7, return = 7
           global_max = 15

Step 2 (Process Node 20):
          20
         /  \
        15   7

  left_max = max(0, 15) = 15  (0 if negative)
  right_max = max(0, 7) = 7

  through = 15 + 20 + 7 = 42 ★ NEW MAX!
  return = 20 + max(15, 7) = 35

  global_max = 42

Step 3 (Process Root -10):
        -10
        /  \
       9    20→35

  left_max = max(0, 9) = 9
  right_max = max(0, 35) = 35

  through = 9 + (-10) + 35 = 34 < 42
  return = -10 + max(9, 35) = 25

  global_max = 42 (unchanged)

Answer: 42

Path Visualization:
        -10
        /  \
       9    20
           /  \
         [15]→[20]→[7]  ← max path = 42
           ●────●────●

Key Insight:
- Use max(0, child_return) to ignore negative paths
- "Through" path updates global max (can't go higher)
- "Return" path goes to parent (can only pick one direction)
- Post-order ensures children processed before parent
```

### Solution
```python
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def maxPathSum(root: Optional[TreeNode]) -> int:
    """
    Find maximum path sum in binary tree.

    Key insight: At each node, we have two choices:
    1. Use node as part of path through parent (return single direction)
    2. Use node as highest point in path (update global max)

    Strategy:
    - For each node, calculate max path going down left/right
    - Update global max with path through current node
    - Return max single-direction path for parent to use

    Time: O(n)
    Space: O(h) for recursion stack
    """
    max_sum = [float('-inf')]

    def dfs(node: TreeNode) -> int:
        if not node:
            return 0

        # Get max path sum from children (0 if negative - don't include)
        left_max = max(0, dfs(node.left))
        right_max = max(0, dfs(node.right))

        # Path through current node (potential global max)
        path_through = left_max + node.val + right_max
        max_sum[0] = max(max_sum[0], path_through)

        # Return max single-direction path for parent
        return node.val + max(left_max, right_max)

    dfs(root)
    return max_sum[0]
```

### Edge Cases
- Single node → return that node's value
- All negative values → return max single node
- Linear tree (skewed) → path is the tree itself
- Empty tree → return 0 or handle specially

---

## Problem 2: Serialize and Deserialize Binary Tree (LC #297) - Hard

- [LeetCode](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

### Video Explanation
- [NeetCode - Serialize and Deserialize Binary Tree](https://www.youtube.com/watch?v=u4JAi2JJhI8)

### Problem Statement
Design algorithm to serialize and deserialize a binary tree.


### Visual Intuition
```
Serialize and Deserialize Binary Tree
        1
       / \
      2   3
         / \
        4   5

Pattern: Preorder Traversal with Null Markers
Why: Preorder uniquely identifies tree structure with nulls

Step 0 (Serialize - Preorder DFS):

  Visit order: 1 → 2 → N → N → 3 → 4 → N → N → 5 → N → N

        1 ─────────────────────────────────────→ "1"
       / \
      2   3 ─────────────────────────────────→ "1,2"
     /\  / \
    N  N 4   5 ──────────────────────────────→ "1,2,N,N,3"
        /\ /\
       N N N N ──────────────────────────────→ "1,2,N,N,3,4,N,N,5,N,N"

  Traversal trace:
  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
  │ 1  │ 2  │ N  │ N  │ 3  │ 4  │ N  │ N  │ 5  │ N  │ N  │
  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
   root left L.L L.R right R.L ...

Step 1 (Deserialize - Rebuild):
  Input: "1,2,N,N,3,4,N,N,5,N,N"
  Iterator: [1, 2, N, N, 3, 4, N, N, 5, N, N]
                ↑
              current

  Read "1" → create node(1), recurse left
        1
       /
      ?

  Read "2" → create node(2), recurse left
        1
       /
      2
     /
    ?

  Read "N" → return None (left of 2)
  Read "N" → return None (right of 2)
        1
       /
      2
     / \
    N   N

  Read "3" → create node(3) (right of 1)
        1
       / \
      2   3
         /
        ?

  Continue until complete...

Final Result:
        1
       / \
      2   3
         / \
        4   5

Key Insight:
- Preorder: root → left → right
- Null markers indicate where subtrees end
- Deserialize uses same order (recursive call stack)
- Iterator consumes values in exact preorder sequence

Why This Works:
  Preorder + nulls = unique tree representation
  No ambiguity about tree structure
  O(n) time for both operations
```

### Solution
```python
class Codec:
    """
    Serialize/deserialize binary tree using preorder traversal.

    Time: O(n) for both operations
    Space: O(n)
    """

    def serialize(self, root: Optional[TreeNode]) -> str:
        """Encode tree to string."""
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

    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Decode string to tree."""
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
    """Alternative: Level-order serialization."""

    def serialize(self, root: Optional[TreeNode]) -> str:
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

    def deserialize(self, data: str) -> Optional[TreeNode]:
        if not data:
            return None

        from collections import deque
        values = data.split(',')
        root = TreeNode(int(values[0]))
        queue = deque([root])
        i = 1

        while queue and i < len(values):
            node = queue.popleft()

            if values[i] != 'N':
                node.left = TreeNode(int(values[i]))
                queue.append(node.left)
            i += 1

            if i < len(values) and values[i] != 'N':
                node.right = TreeNode(int(values[i]))
                queue.append(node.right)
            i += 1

        return root
```

### Edge Cases
- Empty tree → return empty string
- Single node → "val" or "val,N,N"
- Negative values → handle negative signs
- Very deep tree → may need iterative approach

---

## Problem 3: Binary Tree Cameras (LC #968) - Hard

- [LeetCode](https://leetcode.com/problems/binary-tree-cameras/)

### Video Explanation
- [NeetCode - Binary Tree Cameras](https://www.youtube.com/watch?v=2Gh5WPjAgJk)

### Problem Statement
Minimum cameras to monitor all nodes. Camera monitors parent, itself, and children.

### Examples
```
Input: root = [0,0,null,0,0]
Output: 1

      0
     /
    0
   / \
  0   0

One camera at middle node monitors all.
```


### Visual Intuition
```
Binary Tree Cameras (minimum cameras to monitor all)

Pattern: Greedy Post-order with 3 States
Why: Place cameras at parents of leaves (covers most nodes)

Step 0 (Define States):
  ┌─────────────────────────────────────────────┐
  │ State 0: NOT_COVERED (needs camera nearby)  │
  │ State 1: HAS_CAMERA (monitors self+adj)     │
  │ State 2: COVERED (by child's camera)        │
  └─────────────────────────────────────────────┘

Step 1 (Example Tree):
        0
       /
      0
     / \
    0   0
   /
  0  ← leaf

Step 2 (Post-order Processing - Bottom Up):

  Leaf node (bottom):
        0
       /
      0
     / \
    0   0
   /
  ○  ← state = 0 (NOT_COVERED, needs parent)

  Parent of leaf:
        0
       /
      0
     / \
   [●]  0  ← child is 0, so I need camera!
   /        state = 1 (HAS_CAMERA)
  ○

  Sibling (covered by parent's camera):
        0
       /
      0
     / \
   [●]  ◐  ← state = 2 (COVERED by sibling's camera)
   /
  ◐  ← also COVERED

  Grandparent:
        0
       /
      ◐   ← child has camera, so I'm COVERED
     / \      state = 2
   [●]  ◐
   /
  ◐

  Root:
        ○   ← state = 0 (NOT_COVERED!)
       /       Need to add camera here
      ◐
     / \
   [●]  ◐
   /
  ◐

Step 3 (Final - Add Camera at Root):
       [●]  ← camera added (root was NOT_COVERED)
       /
      ◐
     / \
   [●]  ◐
   /
  ◐

Total cameras: 2

Decision Tree at Each Node:
  ┌─────────────────────────────────────────────┐
  │ if any child == NOT_COVERED:                │
  │     return HAS_CAMERA (place camera here)   │
  │                                             │
  │ elif any child == HAS_CAMERA:               │
  │     return COVERED (monitored by child)     │
  │                                             │
  │ else (all children COVERED):                │
  │     return NOT_COVERED (let parent handle)  │
  └─────────────────────────────────────────────┘

Key Insight:
- Null nodes return COVERED (don't need monitoring)
- Leaves return NOT_COVERED (force parent to have camera)
- Greedy: cameras at parents of leaves is optimal
- Post-order ensures children processed first
```

### Solution
```python
def minCameraCover(root: Optional[TreeNode]) -> int:
    """
    Minimum cameras using greedy DFS.

    States for each node:
    0 = NOT_MONITORED (needs camera from parent)
    1 = HAS_CAMERA
    2 = MONITORED (by child camera)

    Strategy:
    - Post-order traversal (process children first)
    - If any child not monitored, place camera at current
    - Greedy: place cameras as low as possible

    Time: O(n)
    Space: O(h)
    """
    NOT_MONITORED = 0
    HAS_CAMERA = 1
    MONITORED = 2

    cameras = [0]

    def dfs(node: TreeNode) -> int:
        if not node:
            return MONITORED  # Null nodes are "monitored"

        left = dfs(node.left)
        right = dfs(node.right)

        # If any child is not monitored, we need camera here
        if left == NOT_MONITORED or right == NOT_MONITORED:
            cameras[0] += 1
            return HAS_CAMERA

        # If any child has camera, we are monitored
        if left == HAS_CAMERA or right == HAS_CAMERA:
            return MONITORED

        # Both children are monitored (no camera), we need monitoring
        return NOT_MONITORED

    # Check if root needs monitoring
    if dfs(root) == NOT_MONITORED:
        cameras[0] += 1

    return cameras[0]
```

### Edge Cases
- Single node → 1 camera
- Linear tree → cameras at every other node
- Perfect binary tree → optimal placement at parents of leaves
- Two nodes → 1 camera at either

---

## Problem 4: Vertical Order Traversal (LC #987) - Hard

- [LeetCode](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)

### Video Explanation
- [NeetCode - Vertical Order Traversal](https://www.youtube.com/watch?v=q_a6lpbKJdw)

### Problem Statement
Return vertical order traversal with sorting rules.


### Visual Intuition
```
Vertical Order Traversal
        1
       / \
      2   3
     / \ / \
    4  5 6  7

Pattern: DFS with (col, row, val) Tracking + Sort
Why: Need to handle same position nodes by value

Step 0 (Assign Coordinates):

  Column assignment: root=0, left=col-1, right=col+1
  Row assignment: root=0, child=row+1

           col: -2  -1   0   1   2
  row 0:              1
  row 1:          2       3
  row 2:      4      5 6      7

  Coordinate grid:
  ┌────┬────┬────┬────┬────┐
  │-2  │-1  │ 0  │ 1  │ 2  │ col
  ├────┼────┼────┼────┼────┤
  │    │    │ 1  │    │    │ row 0
  │    │ 2  │    │ 3  │    │ row 1
  │ 4  │ 5  │ 6  │    │ 7  │ row 2
  └────┴────┴────┴────┴────┘

Step 1 (DFS - Collect All Nodes):

  Visit order with coordinates:
  node 1: (col=0, row=0, val=1)
  node 2: (col=-1, row=1, val=2)
  node 4: (col=-2, row=2, val=4)
  node 5: (col=0, row=2, val=5)  ← same col as 6!
  node 3: (col=1, row=1, val=3)
  node 6: (col=0, row=2, val=6)  ← same col as 5!
  node 7: (col=2, row=2, val=7)

Step 2 (Sort by (col, row, val)):

  Before sort:
  [(0,0,1), (-1,1,2), (-2,2,4), (0,2,5), (1,1,3), (0,2,6), (2,2,7)]

  After sort:
  [(-2,2,4), (-1,1,2), (0,0,1), (0,2,5), (0,2,6), (1,1,3), (2,2,7)]
   ↑         ↑         ↑        ↑        ↑        ↑        ↑
   col=-2    col=-1    col=0    col=0    col=0    col=1    col=2
                       row=0    row=2    row=2
                                val=5    val=6 (5<6)

Step 3 (Group by Column):

  col -2: [4]
  col -1: [2]
  col  0: [1, 5, 6]  ← note: 5 before 6 (same row, sort by val)
  col  1: [3]
  col  2: [7]

Answer: [[4], [2], [1,5,6], [3], [7]]

Special Case - Same Position:
  If nodes 5 and 6 had same (col, row):

       0
      / \
     5   6   ← both at (col=0, row=1)

  Sort by value: 5 < 6, so [5, 6]

Key Insight:
- Collect (col, row, val) tuples during DFS
- Sort handles all ordering requirements
- Python tuple sort: compares col, then row, then val
- O(n log n) for sorting
```

### Solution
```python
from collections import defaultdict

def verticalTraversal(root: Optional[TreeNode]) -> list[list[int]]:
    """
    Vertical order traversal with proper sorting.

    Rules:
    - Sort by column (left to right)
    - Within column, sort by row (top to bottom)
    - Same row and column, sort by value

    Time: O(n log n)
    Space: O(n)
    """
    # Store (row, col, val) for each node
    nodes = []

    def dfs(node: TreeNode, row: int, col: int):
        if not node:
            return

        nodes.append((col, row, node.val))
        dfs(node.left, row + 1, col - 1)
        dfs(node.right, row + 1, col + 1)

    dfs(root, 0, 0)

    # Sort by col, then row, then value
    nodes.sort()

    # Group by column
    result = []
    prev_col = float('-inf')

    for col, row, val in nodes:
        if col != prev_col:
            result.append([])
            prev_col = col
        result[-1].append(val)

    return result
```

### Edge Cases
- Single node → [[val]]
- Same column, same row → sort by value
- All left children → single column
- Wide tree → many columns

---

## Problem 5: Recover Binary Search Tree (LC #99) - Hard

- [LeetCode](https://leetcode.com/problems/recover-binary-search-tree/)

### Video Explanation
- [NeetCode - Recover Binary Search Tree](https://www.youtube.com/watch?v=ZWGW7FminDM)

### Problem Statement
Two nodes were swapped by mistake. Recover the BST.


### Visual Intuition
```
Recover Binary Search Tree (two nodes swapped)

Pattern: Inorder Traversal to Find Violations
Why: BST inorder = sorted, swapped nodes break this

Step 0 (Understand the Problem):

  Correct BST:        Broken BST (3 and 2 swapped):
       2                    3  ← wrong
      / \                  / \
     1   3                1   4
        /                    /
       4                    2  ← wrong

  Inorder (correct): [1, 2, 3, 4] ✓ sorted
  Inorder (broken):  [1, 3, 4, 2] ✗ not sorted
                         ↑     ↑
                      should be 2 and 3

Step 1 (Find Violations in Inorder):

  Trace: [1, 3, 4, 2]
          ↓
  prev=null, curr=1: skip (no prev)
  prev=1, curr=3:    1 < 3 ✓ OK
  prev=3, curr=4:    3 < 4 ✓ OK
  prev=4, curr=2:    4 > 2 ✗ VIOLATION!
                     ↑   ↑
                   first second

  Wait! Let me reconsider the example...

Step 2 (Two Cases of Swapping):

  Case A: Adjacent nodes swapped
  [1, 3, 2, 4]  ← 2 and 3 swapped
       ↑  ↑
      ONE violation: prev=3 > curr=2
      first = prev = 3
      second = curr = 2

  Case B: Non-adjacent nodes swapped
  [1, 4, 3, 2]  ← 2 and 4 swapped (non-adjacent in inorder)
       ↑     ↑
      TWO violations:
      Violation 1: prev=4 > curr=3 → first = 4
      Violation 2: prev=3 > curr=2 → second = 2

Step 3 (Algorithm):

  ┌─────────────────────────────────────────────────┐
  │ During inorder traversal:                       │
  │                                                 │
  │ if prev.val > curr.val:  ← violation found      │
  │     if first is None:                           │
  │         first = prev     ← first violation      │
  │     second = curr        ← always update second │
  │                                                 │
  │ prev = curr              ← move forward         │
  └─────────────────────────────────────────────────┘

Step 4 (Trace Example [1, 4, 3, 2]):

  prev=1, curr=4: 1 < 4 ✓
  prev=4, curr=3: 4 > 3 ✗ first=4, second=3
  prev=3, curr=2: 3 > 2 ✗ second=2 (update!)

  Swap first(4) and second(2):
  Before: [1, 4, 3, 2]
  After:  [1, 2, 3, 4] ✓

Key Insight:
- First violation: first = prev (the larger one)
- Always update second = curr (the smaller one)
- Works for both adjacent and non-adjacent swaps
- O(n) time, O(h) space (or O(1) with Morris)
```

### Solution
```python
def recoverTree(root: Optional[TreeNode]) -> None:
    """
    Recover BST where two nodes were swapped.

    Strategy:
    - Inorder traversal of BST should be sorted
    - Find two nodes that break the sorted order
    - Swap their values

    Time: O(n)
    Space: O(h) for recursion, O(1) for Morris traversal
    """
    first = second = prev = None

    def inorder(node):
        nonlocal first, second, prev

        if not node:
            return

        inorder(node.left)

        # Check if current breaks sorted order
        if prev and prev.val > node.val:
            if not first:
                first = prev  # First violation
            second = node     # Second (or update second)

        prev = node

        inorder(node.right)

    inorder(root)

    # Swap values
    first.val, second.val = second.val, first.val


def recoverTree_morris(root: Optional[TreeNode]) -> None:
    """
    O(1) space using Morris traversal.
    """
    first = second = prev = None
    current = root

    while current:
        if not current.left:
            # Process current
            if prev and prev.val > current.val:
                if not first:
                    first = prev
                second = current
            prev = current
            current = current.right
        else:
            # Find inorder predecessor
            predecessor = current.left
            while predecessor.right and predecessor.right != current:
                predecessor = predecessor.right

            if not predecessor.right:
                # Make thread
                predecessor.right = current
                current = current.left
            else:
                # Remove thread, process current
                predecessor.right = None
                if prev and prev.val > current.val:
                    if not first:
                        first = prev
                    second = current
                prev = current
                current = current.right

    first.val, second.val = second.val, first.val
```

### Edge Cases
- Adjacent swapped nodes → one violation
- Non-adjacent swapped → two violations
- Root swapped → still detectable via inorder
- Two nodes only → swap if out of order

---

## Problem 6: Count Complete Tree Nodes (LC #222) - Medium

- [LeetCode](https://leetcode.com/problems/count-complete-tree-nodes/)

### Video Explanation
- [NeetCode - Count Complete Tree Nodes](https://www.youtube.com/watch?v=u-yWemKGWO0)

### Problem Statement
Count nodes in complete binary tree in O(log²n) time.


### Visual Intuition
```
Count Complete Tree Nodes in O(log²n)
        1
       / \
      2   3
     / \  /
    4  5 6

Pattern: Height Comparison to Skip Subtrees
Why: In complete tree, at least one subtree is always perfect

Step 0 (Complete Tree Property):

  Complete tree: all levels full except last
  Last level: filled left to right

        1          ← level 0 (full)
       / \
      2   3        ← level 1 (full)
     / \  /
    4  5 6         ← level 2 (partial, left-filled)

Step 1 (Key Insight):

  Left height = go all the way left
  Right height = go all the way right

        1
       /↓\
      2   3
     /↓\  /↓
    4  5 6
    ↓     ↓

  left_height = 3 (1→2→4)
  right_height = 2 (1→3→?)  ← stops at 3, no right child of 3

  Heights different → right subtree is perfect!

Step 2 (Decision Logic):

  ┌─────────────────────────────────────────────────┐
  │ if left_height == right_height:                 │
  │     Tree is PERFECT → return 2^h - 1            │
  │                                                 │
  │ else:                                           │
  │     Recurse: 1 + count(left) + count(right)     │
  │                                                 │
  │ Key: At least one subtree is always perfect!    │
  └─────────────────────────────────────────────────┘

Step 3 (Trace Example):

  Node 1: left_h=3, right_h=2 (different)
          → recurse on both subtrees

        [1]
       /   \
      2     3
     / \   /
    4   5 6

  Node 2: left_h=2, right_h=2 (equal!)
          → perfect subtree: 2² - 1 = 3 nodes

       [2]
       / \
      4   5   ← perfect tree of height 2

  Node 3: left_h=2, right_h=1 (different)
          → recurse

       [3]
       /
      6

  Node 6: left_h=1, right_h=1 (equal!)
          → perfect: 2¹ - 1 = 1 node

Step 4 (Calculate Total):

  count(1) = 1 + count(2) + count(3)
           = 1 + 3 + count(3)
           = 1 + 3 + (1 + count(6) + 0)
           = 1 + 3 + (1 + 1 + 0)
           = 1 + 3 + 2
           = 6 ✓

Complexity Analysis:
  Each level: O(log n) to compute heights
  Number of levels: O(log n)
  Total: O(log²n)

Key Insight:
- Don't count nodes one by one (O(n))
- Use complete tree property to skip perfect subtrees
- Perfect tree of height h has 2^h - 1 nodes
```

### Solution
```python
def countNodes(root: Optional[TreeNode]) -> int:
    """
    Count nodes in complete binary tree.

    Key insight: In complete tree, at least one subtree is perfect.

    Strategy:
    - Calculate left and right heights
    - If equal, tree is perfect: 2^h - 1 nodes
    - Otherwise, recurse on both subtrees

    Time: O(log²n) - log n levels, log n for height calculation
    Space: O(log n)
    """
    if not root:
        return 0

    def get_height(node, go_left=True):
        height = 0
        while node:
            height += 1
            node = node.left if go_left else node.right
        return height

    left_height = get_height(root, True)
    right_height = get_height(root, False)

    if left_height == right_height:
        # Perfect tree
        return (1 << left_height) - 1
    else:
        # Recurse
        return 1 + countNodes(root.left) + countNodes(root.right)
```

### Edge Cases
- Empty tree → 0
- Perfect tree → 2^h - 1
- Single node → 1
- Last level has one node → handle correctly

---

## Problem 7: House Robber III (LC #337) - Medium

- [LeetCode](https://leetcode.com/problems/house-robber-iii/)

### Video Explanation
- [NeetCode - House Robber III](https://www.youtube.com/watch?v=nHR8ytpzz7c)

### Problem Statement
Maximum money without robbing adjacent houses (tree structure).


### Visual Intuition
```
House Robber III (can't rob adjacent nodes)
        3
       / \
      2   3
       \   \
        3   1

Pattern: Post-order DFS returning (rob, skip) tuple
Why: At each node, two choices affect children differently

Step 0 (Understand Choices):

  ┌─────────────────────────────────────────────────┐
  │ If we ROB current node:                         │
  │   → Cannot rob children (adjacent)              │
  │   → Can only use skip values from children      │
  │                                                 │
  │ If we SKIP current node:                        │
  │   → Children can be robbed OR skipped           │
  │   → Take max of each child's options            │
  └─────────────────────────────────────────────────┘

Step 1 (Process Leaves - Post-order):

  Leaf 3 (left→right child of 2):
    rob = 3, skip = 0
    return (3, 0)

  Leaf 1 (right child of right 3):
    rob = 1, skip = 0
    return (1, 0)

Step 2 (Process Node 3 - right child of root):

        3
         \
          1 → (1, 0)

  left child = None → (0, 0)
  right child = (1, 0)

  rob = 3 + skip_left + skip_right = 3 + 0 + 0 = 3
  skip = max(left) + max(right) = max(0,0) + max(1,0) = 0 + 1 = 1

  return (3, 1)

Step 3 (Process Node 2):

      2
       \
        3 → (3, 0)

  left child = None → (0, 0)
  right child = (3, 0)

  rob = 2 + 0 + 0 = 2
  skip = max(0,0) + max(3,0) = 0 + 3 = 3

  return (2, 3)

Step 4 (Process Root 3):

        3
       / \
    (2,3) (3,1)

  left = (2, 3)   ← rob=2, skip=3
  right = (3, 1)  ← rob=3, skip=1

  rob = 3 + skip_left + skip_right
      = 3 + 3 + 1 = 7 ★

  skip = max(left) + max(right)
       = max(2,3) + max(3,1)
       = 3 + 3 = 6

  return (7, 6)

Answer: max(7, 6) = 7

Optimal Robbery Path:
        [3] ← ROB (value=3)
       /   \
      2     3
       \     \
       [3]   [1] ← skip=1
        ↑
       ROB (value=3)

  Total: 3 + 3 + 1 = 7

Key Insight:
- Return tuple (rob, skip) from each subtree
- Rob current: add node.val + children's skip values
- Skip current: take max from each child (they decide)
- Post-order ensures children computed before parent
```

### Solution
```python
def rob(root: Optional[TreeNode]) -> int:
    """
    Maximum robbery on tree structure.

    For each node, two choices:
    1. Rob this node + grandchildren
    2. Don't rob this node, rob children

    Return (rob_current, skip_current) for each subtree.

    Time: O(n)
    Space: O(h)
    """
    def dfs(node: TreeNode) -> tuple[int, int]:
        if not node:
            return (0, 0)

        left = dfs(node.left)
        right = dfs(node.right)

        # Rob current: can't rob children
        rob_current = node.val + left[1] + right[1]

        # Skip current: take max from each child
        skip_current = max(left) + max(right)

        return (rob_current, skip_current)

    return max(dfs(root))
```

### Edge Cases
- All zeros → return 0
- Single node → return its value
- Linear tree → simple path optimization
- All same values → skip vs rob decision

---

## Problem 8: Distribute Coins in Binary Tree (LC #979) - Medium

- [LeetCode](https://leetcode.com/problems/distribute-coins-in-binary-tree/)

### Video Explanation
- [NeetCode - Distribute Coins in Binary Tree](https://www.youtube.com/watch?v=RkVF2PjXgHg)

### Problem Statement
Minimum moves to balance coins (each node has 1 coin).


### Visual Intuition
```
Distribute Coins in Binary Tree
        3
       / \
      0   0

Pattern: Post-order DFS tracking Excess/Deficit
Why: Coins flow through edges, count edge crossings

Step 0 (Problem Setup):

  Each node needs exactly 1 coin
  Move = passing 1 coin across 1 edge

  Initial state:     Goal state:
        3                 1
       / \               / \
      0   0             1   1

  Need to move 2 coins from root to children

Step 1 (Define Excess):

  excess = (coins at node - 1) + left_excess + right_excess

  Positive excess: node has extra coins to give
  Negative excess: node needs coins from parent

  Moves at node = |left_excess| + |right_excess|
  (coins crossing left edge + coins crossing right edge)

Step 2 (Process Left Child - Post-order):

        3
       /
      0 ← process first

  coins = 0, need = 1
  excess = 0 - 1 = -1 (deficit)

  This means: 1 coin must flow DOWN from parent

        3
       /↓ ← 1 move
      0

Step 3 (Process Right Child):

        3
         \
          0 ← process second

  excess = 0 - 1 = -1 (deficit)

        3
         ↓\ ← 1 move
          0

Step 4 (Process Root):

        3
       / \
     -1  -1  ← children's excess

  left_excess = -1
  right_excess = -1

  moves = |−1| + |−1| = 2 ★

  root_excess = 3 - 1 + (-1) + (-1) = 0 ✓
  (balanced: root gives away 2, keeps 1)

Answer: 2 moves

More Complex Example:
        0
       / \
      3   0

  Step 1: Left child (3 coins)
    excess = 3 - 1 = +2 (has 2 extra)

  Step 2: Right child (0 coins)
    excess = 0 - 1 = -1 (needs 1)

  Step 3: Root (0 coins)
    left_excess = +2
    right_excess = -1

    moves = |+2| + |−1| = 3

    root_excess = 0 - 1 + 2 + (-1) = 0 ✓

  Coin flow:
        0
       ↑↗ ↘
      3     0

  2 coins up from left, 1 coin down to right = 3 moves

Key Insight:
- Excess = coins - 1 + children's excess
- Moves = |left_excess| + |right_excess|
- Sign of excess: + means give, - means receive
- Absolute value: coins must cross edge either way
```

### Solution
```python
def distributeCoins(root: Optional[TreeNode]) -> int:
    """
    Minimum moves to distribute coins.

    Strategy:
    - Post-order: process children first
    - Each node returns excess/deficit coins
    - Moves = |excess| for each edge

    Time: O(n)
    Space: O(h)
    """
    moves = [0]

    def dfs(node: TreeNode) -> int:
        if not node:
            return 0

        # Get excess from children
        left_excess = dfs(node.left)
        right_excess = dfs(node.right)

        # Moves = coins passing through edges
        moves[0] += abs(left_excess) + abs(right_excess)

        # Return excess: coins - 1 (keep one) + children's excess
        return node.val - 1 + left_excess + right_excess

    dfs(root)
    return moves[0]
```

### Edge Cases
- All nodes have 1 coin → 0 moves
- Root has all coins → distribute down
- Leaves have all coins → distribute up
- Single node with n coins → 0 moves

---

## Problem 9: Smallest String Starting From Leaf (LC #988) - Medium

- [LeetCode](https://leetcode.com/problems/smallest-string-starting-from-leaf/)

### Video Explanation
- [NeetCode - Smallest String Starting From Leaf](https://www.youtube.com/watch?v=cEkPqOCEB1U)

### Problem Statement
Find lexicographically smallest string from leaf to root.


### Visual Intuition
```
Smallest String Starting From Leaf
        z(25)
       /     \
     b(1)    a(0)
     /   \
   a(0)  c(2)

Pattern: DFS with Path Tracking, Compare at Leaves
Why: Need all leaf-to-root paths, compare lexicographically

Step 0 (Understand Direction):

  String goes LEAF → ROOT (bottom to top)
  But we traverse ROOT → LEAF (top to bottom)

  So we build path during DFS, then REVERSE at leaf

        z
       / \
      b   a
     / \
    a   c
    ↑
   START here, go UP to root

Step 1 (DFS Traversal - Track Path):

  Visit z: path = ['z']
  Visit b: path = ['z', 'b']
  Visit a: path = ['z', 'b', 'a'] ← LEAF!

  Reverse: "abz" ← candidate

  Backtrack: path = ['z', 'b']
  Visit c: path = ['z', 'b', 'c'] ← LEAF!

  Reverse: "cbz" ← compare with "abz"
  "abz" < "cbz" (a < c) → keep "abz"

  Backtrack: path = ['z']
  Visit a: path = ['z', 'a'] ← LEAF!

  Reverse: "az" ← compare with "abz"

Step 2 (Lexicographic Comparison):

  "abz" vs "az":

  a b z
  a z
  ↑ ↑
  = compare next char

  'b' < 'z' → "abz" is smaller ✓

  Keep "abz" as answer

Step 3 (All Paths Visualization):

        z
       / \
      b   a
     / \
    a   c

  Path 1: a→b→z = "abz" ★ smallest
  Path 2: c→b→z = "cbz"
  Path 3: a→z   = "az"

  Sorted: "abz" < "az" < "cbz"

Answer: "abz"

Implementation Detail:
  ┌─────────────────────────────────────────────────┐
  │ def dfs(node, path):                            │
  │     path.append(chr(ord('a') + node.val))       │
  │                                                 │
  │     if is_leaf(node):                           │
  │         candidate = ''.join(reversed(path))    │
  │         update_min(candidate)                   │
  │                                                 │
  │     dfs(node.left, path)                        │
  │     dfs(node.right, path)                       │
  │                                                 │
  │     path.pop()  # backtrack                     │
  └─────────────────────────────────────────────────┘

Key Insight:
- Build path root→leaf, reverse at comparison
- Use backtracking to reuse path list
- Compare strings at every leaf
- O(n × h) time: n nodes, h-length string comparison
```

### Solution
```python
def smallestFromLeaf(root: Optional[TreeNode]) -> str:
    """
    Find smallest string from leaf to root.

    Strategy:
    - DFS to all leaves
    - Build path string (reversed)
    - Compare at leaves

    Time: O(n * h) for string comparisons
    Space: O(h)
    """
    smallest = [None]

    def dfs(node: TreeNode, path: list[str]):
        if not node:
            return

        path.append(chr(ord('a') + node.val))

        if not node.left and not node.right:
            # Leaf node - compare path
            current = ''.join(reversed(path))
            if smallest[0] is None or current < smallest[0]:
                smallest[0] = current

        dfs(node.left, path)
        dfs(node.right, path)

        path.pop()

    dfs(root, [])
    return smallest[0]
```

### Edge Cases
- Single node → return that character
- All same characters → shortest path
- Multiple leaves with same string → return any
- Deep tree → long string comparison

---

## Problem 10: All Nodes Distance K (LC #863) - Medium

- [LeetCode](https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/)

### Video Explanation
- [NeetCode - All Nodes Distance K in Binary Tree](https://www.youtube.com/watch?v=nPtARJ2cYrg)

### Problem Statement
Find all nodes at distance K from target node.


### Visual Intuition
```
All Nodes Distance K from Target
        3
       / \
      5   1
     / \ / \
    6  2 0  8
      / \
     7   4

Target = 5, K = 2

Pattern: Convert Tree to Graph + BFS
Why: Need to traverse UP to parent (not just down)

Step 0 (Build Parent Pointers):

  Tree only has child pointers, need parent too

        3 ←─────────┐
       / \          │ parent
      5   1         │
     / \            │
    6   2 ──────────┘
       / \
      7   4

  parent[5] = 3
  parent[6] = 5
  parent[2] = 5
  parent[7] = 2
  parent[4] = 2
  ...

Step 1 (BFS from Target - Level 0):

  queue = [(5, dist=0)]
  visited = {5}

        3
       / \
     [5]  1    ← distance 0
     / \
    6   2
       / \
      7   4

Step 2 (BFS - Level 1):

  Neighbors of 5: parent(3), left(6), right(2)

        3 ←────────┐
       / \         │ dist=1
     [5]  1        │
     / \           │
    6   2 ←────────┘
       / \         dist=1
      7   4

  queue = [(3, 1), (6, 1), (2, 1)]
  visited = {5, 3, 6, 2}

Step 3 (BFS - Level 2):

  From 3: neighbors = [parent(None), left(5✗visited), right(1)]
          add: (1, 2)

  From 6: neighbors = [parent(5✗), left(None), right(None)]
          add: nothing

  From 2: neighbors = [parent(5✗), left(7), right(4)]
          add: (7, 2), (4, 2)

        3
       / \
      5  [1] ←─────── dist=2 ★
     / \ / \
    6   2 0  8
       / \
     [7] [4] ←─────── dist=2 ★

  queue = [(1, 2), (7, 2), (4, 2)]

Step 4 (Collect Distance K=2):

  All nodes with distance 2: [1, 7, 4]

Answer: [1, 7, 4]

BFS Visualization:
  ┌───────────────────────────────────────────────┐
  │         3                                     │
  │        /│\                                    │
  │       / │ \                                   │
  │      5──┼──1  ← Level 2 (from 3)              │
  │     /│\ │                                     │
  │    6 │ 2                                      │
  │      │/│\                                     │
  │      ↑ 7 4  ← Level 2 (from 2)                │
  │   target                                      │
  │                                               │
  │   Levels: 0→5, 1→{3,6,2}, 2→{1,7,4}          │
  └───────────────────────────────────────────────┘

Key Insight:
- Tree becomes undirected graph with parent pointers
- BFS finds all nodes at exact distance K
- visited set prevents revisiting (going back)
- O(n) time to build parents + O(n) BFS
```

### Solution
```python
def distanceK(root: TreeNode, target: TreeNode, k: int) -> list[int]:
    """
    Find all nodes at distance K from target.

    Strategy:
    - Build parent pointers (or adjacency list)
    - BFS from target node

    Time: O(n)
    Space: O(n)
    """
    from collections import deque

    # Build parent map
    parent = {}

    def build_parent(node, par=None):
        if not node:
            return
        parent[node] = par
        build_parent(node.left, node)
        build_parent(node.right, node)

    build_parent(root)

    # BFS from target
    queue = deque([(target, 0)])
    visited = {target}
    result = []

    while queue:
        node, dist = queue.popleft()

        if dist == k:
            result.append(node.val)
            continue

        # Explore neighbors (parent, left, right)
        for neighbor in [parent.get(node), node.left, node.right]:
            if neighbor and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return result
```

### Edge Cases
- k = 0 → return target node only
- k > tree height → may return empty
- Target is root → no parent direction
- Target is leaf → only parent direction

---

## Summary: Tree DFS Hard Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Max Path Sum | Post-order, track global max | O(n) |
| 2 | Serialize Tree | Preorder/BFS encoding | O(n) |
| 3 | Binary Tree Cameras | Greedy post-order | O(n) |
| 4 | Vertical Order | Sort by (col, row, val) | O(n log n) |
| 5 | Recover BST | Inorder finds violations | O(n) |
| 6 | Count Complete Nodes | Height comparison | O(log²n) |
| 7 | House Robber III | Return (rob, skip) tuple | O(n) |
| 8 | Distribute Coins | Post-order excess tracking | O(n) |
| 9 | Smallest From Leaf | DFS with path tracking | O(n*h) |
| 10 | Distance K | Parent map + BFS | O(n) |

---

## Practice More Problems

- [ ] LC #145 - Binary Tree Postorder Traversal (Iterative)
- [ ] LC #156 - Binary Tree Upside Down
- [ ] LC #250 - Count Univalue Subtrees
- [ ] LC #366 - Find Leaves of Binary Tree
- [ ] LC #1028 - Recover a Tree From Preorder Traversal

