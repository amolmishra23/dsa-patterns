# Dynamic Programming on Trees

## Tree DP Fundamentals

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TREE DP PATTERNS                                         │
│                                                                             │
│  1. SUBTREE DP:                                                             │
│     - Process children first (post-order)                                   │
│     - Combine children results at parent                                    │
│     - dp[node] = f(dp[children])                                           │
│                                                                             │
│  2. REROOTING DP:                                                           │
│     - First pass: compute dp for subtrees                                   │
│     - Second pass: compute dp when each node is root                        │
│                                                                             │
│  3. PATH DP:                                                                │
│     - Track paths through nodes                                             │
│     - Often return (include_node, exclude_node)                             │
│                                                                             │
│  4. TREE DIAMETER:                                                          │
│     - Longest path = max(left_depth + right_depth) at any node             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Problem 1: House Robber III (LC #337) - Medium

### Problem Statement
Maximum money without robbing adjacent houses (tree structure).

### Solution
```python
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def rob(root: Optional[TreeNode]) -> int:
    """
    Maximum robbery on tree.

    State: For each node, return (rob_this, skip_this)

    Transitions:
    - rob_this = node.val + skip_left + skip_right
    - skip_this = max(left) + max(right)

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

---

## Problem 2: Binary Tree Maximum Path Sum (LC #124) - Hard

### Problem Statement
Find maximum path sum where path can start/end at any node.

### Solution
```python
def maxPathSum(root: Optional[TreeNode]) -> int:
    """
    Maximum path sum in tree.

    At each node:
    - Calculate max path through this node (update global)
    - Return max single-direction path for parent

    Time: O(n)
    Space: O(h)
    """
    max_sum = [float('-inf')]

    def dfs(node: TreeNode) -> int:
        if not node:
            return 0

        # Max path from children (0 if negative)
        left = max(0, dfs(node.left))
        right = max(0, dfs(node.right))

        # Path through current node
        max_sum[0] = max(max_sum[0], left + node.val + right)

        # Return max single direction for parent
        return node.val + max(left, right)

    dfs(root)
    return max_sum[0]
```

---

## Problem 3: Diameter of Binary Tree (LC #543) - Easy

### Problem Statement
Find longest path between any two nodes.

### Solution
```python
def diameterOfBinaryTree(root: Optional[TreeNode]) -> int:
    """
    Find tree diameter (longest path).

    Diameter through node = left_depth + right_depth

    Time: O(n)
    Space: O(h)
    """
    diameter = [0]

    def depth(node: TreeNode) -> int:
        if not node:
            return 0

        left = depth(node.left)
        right = depth(node.right)

        # Update diameter
        diameter[0] = max(diameter[0], left + right)

        # Return depth
        return 1 + max(left, right)

    depth(root)
    return diameter[0]
```

---

## Problem 4: Longest Univalue Path (LC #687) - Medium

### Problem Statement
Find longest path where all nodes have same value.

### Solution
```python
def longestUnivaluePath(root: Optional[TreeNode]) -> int:
    """
    Longest path with same values.

    Similar to diameter but only count matching values.

    Time: O(n)
    Space: O(h)
    """
    longest = [0]

    def dfs(node: TreeNode) -> int:
        if not node:
            return 0

        left = dfs(node.left)
        right = dfs(node.right)

        # Extend path if values match
        left_path = left + 1 if node.left and node.left.val == node.val else 0
        right_path = right + 1 if node.right and node.right.val == node.val else 0

        # Update longest
        longest[0] = max(longest[0], left_path + right_path)

        return max(left_path, right_path)

    dfs(root)
    return longest[0]
```

---

## Problem 5: Binary Tree Cameras (LC #968) - Hard

### Problem Statement
Minimum cameras to monitor all nodes.

### Solution
```python
def minCameraCover(root: Optional[TreeNode]) -> int:
    """
    Minimum cameras using greedy tree DP.

    States:
    0 = NOT_MONITORED
    1 = HAS_CAMERA
    2 = MONITORED (by child)

    Time: O(n)
    Space: O(h)
    """
    cameras = [0]

    def dfs(node: TreeNode) -> int:
        if not node:
            return 2  # Null nodes are "monitored"

        left = dfs(node.left)
        right = dfs(node.right)

        # If any child not monitored, need camera here
        if left == 0 or right == 0:
            cameras[0] += 1
            return 1

        # If any child has camera, we're monitored
        if left == 1 or right == 1:
            return 2

        # Both children monitored, we need monitoring
        return 0

    if dfs(root) == 0:
        cameras[0] += 1

    return cameras[0]
```

---

## Problem 6: Sum of Distances in Tree (LC #834) - Hard

### Problem Statement
For each node, find sum of distances to all other nodes.

### Solution
```python
def sumOfDistancesInTree(n: int, edges: list[list[int]]) -> list[int]:
    """
    Sum of distances using rerooting DP.

    Two passes:
    1. Post-order: compute subtree sizes and distances from root
    2. Pre-order: compute distances when each node is root

    Key formula: When moving root from parent to child:
    ans[child] = ans[parent] - count[child] + (n - count[child])

    Time: O(n)
    Space: O(n)
    """
    from collections import defaultdict

    # Build adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    count = [1] * n  # Subtree sizes
    ans = [0] * n    # Answer for each node

    # First pass: compute subtree sizes and distances from node 0
    def post_order(node: int, parent: int):
        for child in graph[node]:
            if child != parent:
                post_order(child, node)
                count[node] += count[child]
                ans[node] += ans[child] + count[child]

    # Second pass: compute answer for all nodes
    def pre_order(node: int, parent: int):
        for child in graph[node]:
            if child != parent:
                # Moving root from node to child:
                # - Nodes in child's subtree get closer by 1 (count[child] nodes)
                # - Other nodes get farther by 1 (n - count[child] nodes)
                ans[child] = ans[node] - count[child] + (n - count[child])
                pre_order(child, node)

    post_order(0, -1)
    pre_order(0, -1)

    return ans
```

---

## Problem 7: Distribute Coins in Binary Tree (LC #979) - Medium

### Problem Statement
Minimum moves to balance coins (each node has 1 coin).

### Solution
```python
def distributeCoins(root: Optional[TreeNode]) -> int:
    """
    Minimum moves to distribute coins.

    Each node returns excess/deficit.
    Moves = sum of |excess| across all edges.

    Time: O(n)
    Space: O(h)
    """
    moves = [0]

    def dfs(node: TreeNode) -> int:
        if not node:
            return 0

        left_excess = dfs(node.left)
        right_excess = dfs(node.right)

        # Moves through this node's edges
        moves[0] += abs(left_excess) + abs(right_excess)

        # Return excess: coins - 1 (keep one) + children excess
        return node.val - 1 + left_excess + right_excess

    dfs(root)
    return moves[0]
```

---

## Problem 8: Delete Nodes And Return Forest (LC #1110) - Medium

### Problem Statement
Delete nodes and return remaining trees.

### Solution
```python
def delNodes(root: Optional[TreeNode], to_delete: list[int]) -> list[TreeNode]:
    """
    Delete nodes and return forest.

    Strategy:
    - Post-order to handle children first
    - If node deleted, children become new roots
    - Return None if deleted, node otherwise

    Time: O(n)
    Space: O(n)
    """
    to_delete_set = set(to_delete)
    result = []

    def dfs(node: TreeNode, is_root: bool) -> TreeNode:
        if not node:
            return None

        is_deleted = node.val in to_delete_set

        # If current is root and not deleted, add to result
        if is_root and not is_deleted:
            result.append(node)

        # Process children (they become roots if current is deleted)
        node.left = dfs(node.left, is_deleted)
        node.right = dfs(node.right, is_deleted)

        return None if is_deleted else node

    dfs(root, True)
    return result
```

---

## Problem 9: Maximum Product of Splitted Binary Tree (LC #1339) - Medium

### Problem Statement
Maximum product of two subtree sums after removing one edge.

### Solution
```python
def maxProduct(root: Optional[TreeNode]) -> int:
    """
    Maximum product after splitting tree.

    Strategy:
    1. Calculate total sum
    2. For each subtree, product = subtree_sum * (total - subtree_sum)

    Time: O(n)
    Space: O(h)
    """
    MOD = 10**9 + 7
    subtree_sums = []

    def get_sum(node: TreeNode) -> int:
        if not node:
            return 0

        total = node.val + get_sum(node.left) + get_sum(node.right)
        subtree_sums.append(total)
        return total

    total = get_sum(root)

    max_product = 0
    for s in subtree_sums:
        max_product = max(max_product, s * (total - s))

    return max_product % MOD
```

---

## Problem 10: Linked List in Binary Tree (LC #1367) - Medium

### Problem Statement
Check if linked list exists as downward path in tree.

### Solution
```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def isSubPath(head: ListNode, root: TreeNode) -> bool:
    """
    Check if linked list is a path in tree.

    Strategy:
    - Try starting from each tree node
    - DFS to match list going downward

    Time: O(n * m) where n = tree nodes, m = list length
    Space: O(h + m)
    """
    def match(list_node: ListNode, tree_node: TreeNode) -> bool:
        # List exhausted - found match
        if not list_node:
            return True

        # Tree exhausted or mismatch
        if not tree_node or list_node.val != tree_node.val:
            return False

        # Continue matching in either direction
        return match(list_node.next, tree_node.left) or \
               match(list_node.next, tree_node.right)

    def dfs(node: TreeNode) -> bool:
        if not node:
            return False

        # Try starting match from this node
        if match(head, node):
            return True

        # Try other starting points
        return dfs(node.left) or dfs(node.right)

    return dfs(root)
```

---

## Summary: Tree DP Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | House Robber III | Return (rob, skip) tuple | O(n) |
| 2 | Max Path Sum | Track global max | O(n) |
| 3 | Diameter | left_depth + right_depth | O(n) |
| 4 | Longest Univalue | Extend only if match | O(n) |
| 5 | Binary Tree Cameras | 3-state greedy | O(n) |
| 6 | Sum of Distances | Rerooting DP | O(n) |
| 7 | Distribute Coins | Track excess | O(n) |
| 8 | Delete Nodes | Post-order deletion | O(n) |
| 9 | Max Product Split | Subtree sums | O(n) |
| 10 | List in Tree | DFS + match | O(nm) |

---

## Tree DP Template

```python
def tree_dp(root):
    """Generic tree DP template."""
    result = [initial_value]

    def dfs(node):
        if not node:
            return base_case

        # Process children first (post-order)
        left = dfs(node.left)
        right = dfs(node.right)

        # Update global result if needed
        result[0] = update(result[0], left, right, node)

        # Return value for parent
        return combine(left, right, node)

    dfs(root)
    return result[0]
```

---

## Practice More Problems

- [ ] LC #250 - Count Univalue Subtrees
- [ ] LC #298 - Binary Tree Longest Consecutive Sequence
- [ ] LC #333 - Largest BST Subtree
- [ ] LC #549 - Binary Tree Longest Consecutive Sequence II
- [ ] LC #1372 - Longest ZigZag Path in a Binary Tree

