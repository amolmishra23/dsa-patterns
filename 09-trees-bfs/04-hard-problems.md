# Trees BFS - Hard Problems

## Problem 1: Serialize and Deserialize Binary Tree (LC #297) - Hard

- [LeetCode](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

### Video Explanation
- [NeetCode - Serialize and Deserialize Binary Tree](https://www.youtube.com/watch?v=u4JAi2JJhI8)

### Problem Statement
Design algorithm to serialize and deserialize a binary tree.


### Visual Intuition
```
Serialize and Deserialize Binary Tree (BFS)
        1
       / \
      2   3
         / \
        4   5

BFS serialization (level order with nulls):
Queue: [1] → output "1"
Queue: [2,3] → output "1,2,3"
Queue: [null,null,4,5] → output "1,2,3,null,null,4,5"
Queue: [null,null,null,null] → output "...,null,null,null,null"

Deserialization:
Read "1" → root = Node(1)
Read "2","3" → root.left=2, root.right=3
Read "null","null" → 2.left=null, 2.right=null
Read "4","5" → 3.left=4, 3.right=5
```

### Solution
```python
from collections import deque

class Codec:
    """
    Serialize/deserialize using level-order BFS.

    Time: O(n)
    Space: O(n)
    """

    def serialize(self, root) -> str:
        if not root:
            return ""

        result = []
        queue = deque([root])

        while queue:
            node = queue.popleft()
            if node:
                result.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append("N")

        return ",".join(result)

    def deserialize(self, data: str):
        if not data:
            return None

        values = data.split(",")
        root = TreeNode(int(values[0]))
        queue = deque([root])
        i = 1

        while queue and i < len(values):
            node = queue.popleft()

            if values[i] != "N":
                node.left = TreeNode(int(values[i]))
                queue.append(node.left)
            i += 1

            if i < len(values) and values[i] != "N":
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

## Problem 2: Vertical Order Traversal (LC #987) - Hard

- [LeetCode](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)

### Video Explanation
- [NeetCode - Vertical Order Traversal](https://www.youtube.com/watch?v=q_a6lpbKJdw)

### Problem Statement
Return vertical order traversal with proper sorting rules.


### Visual Intuition
```
Vertical Order Traversal (BFS with coordinates)
        1
       / \
      2   3
     / \   \
    4   5   6

Assign (col, row) to each node:
  1: (0, 0)
  2: (-1, 1), 3: (1, 1)
  4: (-2, 2), 5: (0, 2), 6: (2, 2)

Group by column, sort by (row, value):
  col -2: [(2,4)] → [4]
  col -1: [(1,2)] → [2]
  col  0: [(0,1), (2,5)] → [1,5]
  col  1: [(1,3)] → [3]
  col  2: [(2,6)] → [6]

Result: [[4], [2], [1,5], [3], [6]]
```

### Solution
```python
from collections import defaultdict, deque

def verticalTraversal(root) -> list[list[int]]:
    """
    Vertical traversal with sorting by (col, row, value).

    Time: O(n log n)
    Space: O(n)
    """
    if not root:
        return []

    # Store (row, val) for each column
    columns = defaultdict(list)
    queue = deque([(root, 0, 0)])  # (node, row, col)

    while queue:
        node, row, col = queue.popleft()
        columns[col].append((row, node.val))

        if node.left:
            queue.append((node.left, row + 1, col - 1))
        if node.right:
            queue.append((node.right, row + 1, col + 1))

    result = []
    for col in sorted(columns.keys()):
        # Sort by row, then by value
        result.append([val for row, val in sorted(columns[col])])

    return result
```

### Edge Cases
- Single node → [[val]]
- Same column, same row → sort by value
- All left children → single column
- Wide tree → many columns

---

## Problem 3: Bus Routes (LC #815) - Hard

- [LeetCode](https://leetcode.com/problems/bus-routes/)

### Video Explanation
- [NeetCode - Bus Routes](https://www.youtube.com/watch?v=vEcm5farBls)

### Problem Statement
Find minimum buses to travel from source to target.


### Visual Intuition
```
Bus Routes
routes = [[1,2,7],[3,6,7]], source = 1, target = 6

Build graph: stop → buses that visit it
  1: [bus0]
  2: [bus0]
  7: [bus0, bus1]
  3: [bus1]
  6: [bus1]

BFS on buses (not stops):
  Start: stops reachable = {1}, buses taken = 0

  Level 0: At stop 1, can take bus0
  Level 1: Bus0 visits {1,2,7}, check if 6 in set? No
           From stop 7, can take bus1
  Level 2: Bus1 visits {3,6,7}, check if 6 in set? Yes!

Answer: 2 buses
```

### Solution
```python
from collections import defaultdict, deque

def numBusesToDestination(routes: list[list[int]], source: int, target: int) -> int:
    """
    BFS on bus routes (not stops).

    Time: O(n * m) where n = routes, m = stops per route
    Space: O(n * m)
    """
    if source == target:
        return 0

    # Map stop to routes passing through it
    stop_to_routes = defaultdict(set)
    for i, route in enumerate(routes):
        for stop in route:
            stop_to_routes[stop].add(i)

    # BFS on routes
    visited_routes = set()
    visited_stops = {source}
    queue = deque([(source, 0)])

    while queue:
        stop, buses = queue.popleft()

        for route_idx in stop_to_routes[stop]:
            if route_idx in visited_routes:
                continue
            visited_routes.add(route_idx)

            for next_stop in routes[route_idx]:
                if next_stop == target:
                    return buses + 1

                if next_stop not in visited_stops:
                    visited_stops.add(next_stop)
                    queue.append((next_stop, buses + 1))

    return -1
```

### Edge Cases
- source == target → return 0
- No path exists → return -1
- Single route → check if both stops on it
- Overlapping routes → BFS handles

---

## Problem 4: Binary Tree Cameras (LC #968) - Hard

- [LeetCode](https://leetcode.com/problems/binary-tree-cameras/)

### Video Explanation
- [NeetCode - Binary Tree Cameras](https://www.youtube.com/watch?v=uoFrIIrp5_g)

### Problem Statement
Install minimum cameras on tree nodes to monitor all nodes. A camera monitors its parent, itself, and children.

### Visual Intuition
```
Binary Tree Cameras - Greedy DFS
       0 (no camera, covered by child)
      /
    [C] (camera here covers parent + children)
    / \
   0   0 (covered by parent camera)
  /
[C] (camera needed - leaf's parent)
 |
 L (leaf - no camera, needs parent)

States: 0=not covered, 1=has camera, 2=covered
Post-order: process children first, then decide parent
If any child not covered → parent needs camera
```


### Intuition
```
        0 (no camera, monitored by child)
       /
      1 (CAMERA) ← monitors parent, self, children
     / \
    2   3 (leaf nodes, monitored by parent)

Greedy: Place cameras at parents of leaves, not at leaves.
Post-order traversal: process children first, then decide for parent.

States:
- 0: Not monitored (needs camera from parent)
- 1: Has camera
- 2: Monitored (by child camera)
```

### Solution
```python
def minCameraCover(root) -> int:
    """
    Greedy post-order traversal.

    Strategy:
    - Process leaves first (post-order)
    - Don't put camera on leaf, put on its parent
    - States: 0=needs coverage, 1=has camera, 2=covered

    Time: O(n)
    Space: O(h) for recursion stack
    """
    cameras = 0

    def dfs(node):
        nonlocal cameras

        if not node:
            return 2  # Null nodes are "covered"

        left = dfs(node.left)
        right = dfs(node.right)

        # If any child needs coverage, place camera here
        if left == 0 or right == 0:
            cameras += 1
            return 1  # Has camera

        # If any child has camera, this node is covered
        if left == 1 or right == 1:
            return 2  # Covered

        # Both children are covered but no camera nearby
        return 0  # Needs coverage from parent

    # Handle root needing coverage
    if dfs(root) == 0:
        cameras += 1

    return cameras
```

### Complexity
- **Time**: O(n)
- **Space**: O(h) height of tree

### Edge Cases
- Single node → 1 camera
- Linear tree → cameras at every other node
- Perfect binary tree → optimal at parents of leaves
- Two nodes → 1 camera at either

---

## Problem 5: Smallest String Starting From Leaf (LC #988) - Hard

- [LeetCode](https://leetcode.com/problems/smallest-string-starting-from-leaf/)

### Video Explanation
- [NeetCode - Smallest String Starting From Leaf](https://www.youtube.com/watch?v=cPYqfgJfRHM)

### Problem Statement
Find lexicographically smallest string from leaf to root.

### Visual Intuition
```
Smallest String Starting From Leaf
        a
       / \
      b   c
     / \
    d   e

Paths (leaf to root):
  d→b→a = "dba"
  e→b→a = "eba"
  c→a   = "ca"

Lexicographically smallest: "ca"
BFS/DFS tracking path string, compare at leaves
```


### Intuition
```
        a(0)
       /    \
      b(1)   c(2)
     / \      \
    d   e      a

Paths (leaf to root):
- d→b→a = "dba"
- e→b→a = "eba"
- a→c→a = "aca"

Answer: "aca" (lexicographically smallest)
```

### Solution
```python
from collections import deque

def smallestFromLeaf(root) -> str:
    """
    BFS with path tracking, compare at leaves.

    Time: O(n * h) where h = height for string operations
    Space: O(n)
    """
    if not root:
        return ""

    result = None
    queue = deque([(root, chr(ord('a') + root.val))])

    while queue:
        node, path = queue.popleft()

        # Leaf node - compare path
        if not node.left and not node.right:
            # Reverse path (leaf to root)
            candidate = path[::-1]
            if result is None or candidate < result:
                result = candidate

        # Add children with updated path
        if node.left:
            char = chr(ord('a') + node.left.val)
            queue.append((node.left, path + char))

        if node.right:
            char = chr(ord('a') + node.right.val)
            queue.append((node.right, path + char))

    return result
```

### DFS Alternative
```python
def smallestFromLeaf(root) -> str:
    """
    DFS approach - more memory efficient.
    """
    result = [None]

    def dfs(node, path):
        if not node:
            return

        # Prepend current char (building from leaf to root)
        path = chr(ord('a') + node.val) + path

        if not node.left and not node.right:
            if result[0] is None or path < result[0]:
                result[0] = path
            return

        dfs(node.left, path)
        dfs(node.right, path)

    dfs(root, "")
    return result[0]
```

### Complexity
- **Time**: O(n * h)
- **Space**: O(n) for BFS, O(h) for DFS

### Edge Cases
- Single node → return that character
- All same characters → shortest path
- Multiple leaves with same string → return any
- Deep tree → long string comparison

---

## Problem 6: Complete Binary Tree Inserter (LC #919) - Hard

- [LeetCode](https://leetcode.com/problems/complete-binary-tree-inserter/)

### Video Explanation
- [NeetCode - Complete Binary Tree Inserter](https://www.youtube.com/watch?v=hwR5Sp7JJRg)

### Problem Statement
Design a data structure to insert nodes into a complete binary tree efficiently.

### Visual Intuition
```
Complete Binary Tree Inserter
Complete tree fills level by level, left to right

═══════════════════════════════════════════════════════════════
KEY INSIGHT: Use queue to track nodes that can accept children
             Front of queue = next insertion point
═══════════════════════════════════════════════════════════════

Initial Tree:
─────────────
         1
        / \
       2   3
      / \
     4   5

  Queue of incomplete nodes: [3]
  (Node 2 is full, Node 3 has no children yet)

Insert 6:
─────────
  Step 1: Get front of queue → Node 3
  Step 2: Node 3 has no left child → insert as left
  Step 3: Node 3 still has empty right → stays in queue

         1
        / \
       2   3
      / \ /
     4  5 6 ← NEW

  Queue: [3] (right still empty)

Insert 7:
─────────
  Step 1: Get front of queue → Node 3
  Step 2: Node 3 has left child → insert as right
  Step 3: Node 3 is now full → remove from queue
  Step 4: Add new node 6 to queue (can accept children)

         1
        / \
       2   3
      / \ / \
     4  5 6  7 ← NEW

  Queue: [4, 5, 6, 7] (all can accept children)

Insert 8:
─────────
  Step 1: Get front of queue → Node 4
  Step 2: Insert as left child

           1
          / \
         2   3
        / \ / \
       4  5 6  7
      /
     8 ← NEW

  Queue: [4, 5, 6, 7]

WHY THIS WORKS:
════════════════
● Complete tree property: fill left to right, level by level
● Queue maintains BFS order of incomplete nodes
● O(1) insert: just check front of queue
● Parent returned immediately for verification
```


### Intuition
```
Complete Binary Tree - all levels full except last, filled left to right.

        1
       / \
      2   3
     / \
    4   5

Next insert goes to node 3's left child.

BFS maintains queue of nodes that can accept children.
```

### Solution
```python
from collections import deque

class CBTInserter:
    """
    Complete Binary Tree Inserter using BFS.

    Strategy:
    - Maintain queue of nodes that can accept children
    - A node leaves queue when both children are filled

    Time: O(n) init, O(1) insert
    Space: O(n)
    """

    def __init__(self, root):
        self.root = root
        self.queue = deque()

        # BFS to find nodes that can accept children
        q = deque([root])
        while q:
            node = q.popleft()

            # Node can accept children if not full
            if not node.left or not node.right:
                self.queue.append(node)

            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)

    def insert(self, val: int) -> int:
        """Insert node and return parent's value."""
        parent = self.queue[0]
        new_node = TreeNode(val)

        if not parent.left:
            parent.left = new_node
        else:
            parent.right = new_node
            self.queue.popleft()  # Parent is now full

        self.queue.append(new_node)
        return parent.val

    def get_root(self):
        return self.root
```

### Complexity
- **Init**: O(n) time, O(n) space
- **Insert**: O(1) time
- **Get Root**: O(1) time

### Edge Cases
- Single node tree → insert as left child
- Full last level → start new level
- Empty tree → not valid per problem
- Multiple inserts → queue handles ordering

---

## Summary

| # | Problem | Key Technique |
|---|---------|---------------|
| 1 | Serialize Tree | Level-order encoding |
| 2 | Vertical Order | Sort by (col, row, val) |
| 3 | Bus Routes | BFS on routes, not stops |
| 4 | Binary Tree Cameras | Greedy post-order, 3 states |
| 5 | Smallest String From Leaf | BFS/DFS with path tracking |
| 6 | CBT Inserter | Queue of incomplete nodes |
