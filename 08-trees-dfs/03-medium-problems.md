# Trees DFS - Medium Problems

## Problem 1: Validate Binary Search Tree (LC #98) - Medium

- [LeetCode](https://leetcode.com/problems/validate-binary-search-tree/)

### Problem Statement
Given the root of a binary tree, determine if it is a valid **Binary Search Tree (BST)**. A valid BST is defined as:
- The left subtree of a node contains only nodes with keys **less than** the node's key.
- The right subtree of a node contains only nodes with keys **greater than** the node's key.
- Both left and right subtrees must also be valid BSTs.

### Video Explanation
- [NeetCode - Validate Binary Search Tree](https://www.youtube.com/watch?v=s6ATEkipzow)

### Examples
```
Input: root = [2,1,3]
Output: true
Explanation:
      2
     / \
    1   3   ← 1 < 2 < 3 ✓

Input: root = [5,1,4,null,null,3,6]
Output: false
Explanation:
        5
       / \
      1   4
         / \
        3   6   ← 4 is in right subtree but 4 < 5!

Input: root = [5,4,6,null,null,3,7]
Output: false
Explanation: 3 is in right subtree of 5 but 3 < 5!
```

### Intuition Development
```
WRONG approach: Just check node.left < node < node.right
  This fails because we need to check ALL ancestors!

        5
       / \
      1   6
         / \
        3   7   ← 3 < 6 ✓ BUT 3 < 5 and it's in RIGHT subtree of 5!

CORRECT approach: Track valid RANGE for each node!

┌─────────────────────────────────────────────────────────────────┐
│ As we traverse, maintain [min, max] bounds:                     │
│                                                                  │
│       5 [-∞, +∞]                                                │
│      / \                                                         │
│     1   6                                                        │
│ [-∞,5] [5,+∞]                                                   │
│        / \                                                       │
│       3   7                                                      │
│    [5,6] [6,+∞]                                                  │
│      ↑                                                           │
│      3 is NOT in range [5,6]! INVALID                           │
│                                                                  │
│ Rules:                                                           │
│   - Go LEFT: update max to current node's value                 │
│   - Go RIGHT: update min to current node's value                │
└─────────────────────────────────────────────────────────────────┘

Alternative: INORDER traversal must be strictly increasing!
  Inorder of BST: [1, 3, 5, 6, 7]
  If we see decreasing → INVALID
```

### Solution
```python
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def isValidBST(root: Optional[TreeNode]) -> bool:
    """
    Validate BST using range checking.

    Strategy:
    - Each node must be within valid range
    - Left subtree: upper bound = parent value
    - Right subtree: lower bound = parent value

    Time: O(n)
    Space: O(h) for recursion stack
    """
    def validate(node: TreeNode, min_val: float, max_val: float) -> bool:
        if not node:
            return True

        # Check current node is within valid range
        if node.val <= min_val or node.val >= max_val:
            return False

        # Recursively validate subtrees with updated ranges
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))

    return validate(root, float('-inf'), float('inf'))


def isValidBST_inorder(root: Optional[TreeNode]) -> bool:
    """
    Alternative: Inorder traversal should be strictly increasing.

    Time: O(n)
    Space: O(h)
    """
    prev = [float('-inf')]

    def inorder(node):
        if not node:
            return True

        # Check left subtree
        if not inorder(node.left):
            return False

        # Check current node
        if node.val <= prev[0]:
            return False
        prev[0] = node.val

        # Check right subtree
        return inorder(node.right)

    return inorder(root)
```

### Complexity
- **Time**: O(n) - Visit each node once
- **Space**: O(h) - Recursion stack, where h is tree height

### Edge Cases
- Single node: Always valid
- All left children: Valid if decreasing values
- Equal values: Invalid (BST requires strictly less/greater)
- Negative values: Work correctly with range checking

---

## Problem 2: Lowest Common Ancestor (LC #236) - Medium

- [LeetCode](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

### Problem Statement
Given a binary tree, find the **lowest common ancestor (LCA)** of two given nodes `p` and `q`. The LCA is the lowest node that has both `p` and `q` as descendants (a node can be a descendant of itself).

### Video Explanation
- [NeetCode - Lowest Common Ancestor](https://www.youtube.com/watch?v=gs2LMfuOR9k)

### Examples
```
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation:
         3 ← LCA
        / \
       5   1
      / \ / \
     6  2 0  8
       / \
      7   4

Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: 5 is ancestor of 4, and 5 is ancestor of itself

Input: root = [1,2], p = 1, q = 2
Output: 1
```

### Intuition Development
```
Key insight: Recursively search for p and q in left and right subtrees!

Case analysis at each node:
┌─────────────────────────────────────────────────────────────────┐
│ Case 1: Current node is p or q                                  │
│   → Return current node (might be LCA or one of the targets)    │
│                                                                  │
│ Case 2: Both subtrees return non-null                           │
│   → p is in one subtree, q is in other → current is LCA!       │
│                                                                  │
│ Case 3: Only one subtree returns non-null                       │
│   → Both p and q are in that subtree → return that result       │
│                                                                  │
│ Case 4: Both subtrees return null                               │
│   → Neither p nor q in this subtree → return null               │
└─────────────────────────────────────────────────────────────────┘

Example: p=5, q=4
         3
        / \
       5   1      At node 3: left returns 5, right returns null
      / \         → return 5 (LCA is 5)
     6   2
        / \
       7   4      At node 5: node==p, return 5
                  At node 2: left returns null, right returns 4
```

### Solution
```python
def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    Find LCA using recursive DFS.

    Strategy:
    - If current node is p or q, return it
    - Recursively search left and right subtrees
    - If both return non-null, current is LCA
    - Otherwise, return the non-null result

    Time: O(n)
    Space: O(h)
    """
    # Base case
    if not root or root == p or root == q:
        return root

    # Search in subtrees
    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    # If both subtrees return non-null, root is LCA
    if left and right:
        return root

    # Return non-null result
    return left if left else right
```

### Complexity
- **Time**: O(n) - Visit each node once
- **Space**: O(h) - Recursion stack depth

### Edge Cases
- p is ancestor of q: Return p
- q is ancestor of p: Return q
- p and q are the same node: Return that node
- p and q are siblings: Return parent

---

## Problem 3: Binary Tree from Preorder and Inorder (LC #105) - Medium

- [LeetCode](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

### Problem Statement
Given two integer arrays `preorder` and `inorder` where `preorder` is the preorder traversal of a binary tree and `inorder` is the inorder traversal of the same tree, construct and return the binary tree. All values are unique.

### Video Explanation
- [NeetCode - Construct Binary Tree from Preorder and Inorder](https://www.youtube.com/watch?v=ihj4IQGZ2zc)

### Examples
```
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
Explanation:
       3
      / \
     9  20
       /  \
      15   7

Input: preorder = [-1], inorder = [-1]
Output: [-1]
```

### Intuition Development
```
Key insight: Preorder's FIRST element is always the ROOT!
             Inorder splits left and right subtrees!

preorder = [3, 9, 20, 15, 7]
             ↑
           ROOT

inorder  = [9, 3, 15, 20, 7]
            ↑  ↑  ↑────↑
          LEFT ROOT RIGHT

┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Root = preorder[0] = 3                                  │
│         Find 3 in inorder → index 1                             │
│         Left subtree inorder: [9]                               │
│         Right subtree inorder: [15, 20, 7]                      │
│                                                                  │
│ Step 2: Next preorder element = 9 (left subtree root)           │
│         Left subtree has no children (single element)           │
│                                                                  │
│ Step 3: Next preorder element = 20 (right subtree root)         │
│         Find 20 in remaining inorder [15, 20, 7]                │
│         Left: [15], Right: [7]                                   │
│                                                                  │
│ Result:     3                                                    │
│            / \                                                   │
│           9   20                                                 │
│              / \                                                 │
│             15  7                                                │
└─────────────────────────────────────────────────────────────────┘

Use hash map for O(1) lookup of root position in inorder!
```

### Solution
```python
def buildTree(preorder: list[int], inorder: list[int]) -> Optional[TreeNode]:
    """
    Build tree from preorder and inorder traversals.

    Strategy:
    - First element of preorder is root
    - Find root in inorder to split left/right subtrees
    - Use hashmap for O(1) index lookup

    Time: O(n)
    Space: O(n)
    """
    # Map value to index in inorder
    inorder_map = {val: i for i, val in enumerate(inorder)}
    preorder_idx = [0]  # Use list to maintain state across calls

    def build(left: int, right: int) -> Optional[TreeNode]:
        if left > right:
            return None

        # Get root value from preorder
        root_val = preorder[preorder_idx[0]]
        preorder_idx[0] += 1

        # Create root node
        root = TreeNode(root_val)

        # Find root position in inorder
        mid = inorder_map[root_val]

        # Build subtrees
        root.left = build(left, mid - 1)
        root.right = build(mid + 1, right)

        return root

    return build(0, len(inorder) - 1)
```

### Complexity
- **Time**: O(n) - Process each node once, O(1) lookup with hash map
- **Space**: O(n) - Hash map and recursion stack

### Edge Cases
- Single node: Return that node
- Left-skewed tree: All left children
- Right-skewed tree: All right children
- Duplicate values: Not allowed per problem constraints

---

## Problem 4: Binary Tree from Inorder and Postorder (LC #106) - Medium

- [LeetCode](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

### Problem Statement
Given two integer arrays `inorder` and `postorder` where `inorder` is the inorder traversal and `postorder` is the postorder traversal of the same tree, construct and return the binary tree.

### Video Explanation
- [NeetCode - Construct Binary Tree from Inorder and Postorder](https://www.youtube.com/watch?v=vm63HuIU7kw)

### Examples
```
Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
Output: [3,9,20,null,null,15,7]

Input: inorder = [-1], postorder = [-1]
Output: [-1]
```

### Intuition Development
```
Key insight: Postorder's LAST element is always the ROOT!
             Process postorder RIGHT to LEFT!

postorder = [9, 15, 7, 20, 3]
                          ↑
                        ROOT

inorder   = [9, 3, 15, 20, 7]
             ↑  ↑  ↑─────↑
           LEFT ROOT RIGHT

┌─────────────────────────────────────────────────────────────────┐
│ Key difference from preorder:                                    │
│   - Postorder: LEFT → RIGHT → ROOT                              │
│   - We read from END, so: ROOT → RIGHT → LEFT                   │
│   - Build RIGHT subtree BEFORE left!                            │
│                                                                  │
│ Step 1: Root = postorder[-1] = 3                                │
│         Find 3 in inorder → index 1                             │
│                                                                  │
│ Step 2: Process RIGHT first! Next = 20                          │
│         Find 20 in inorder [15, 20, 7]                          │
│         Build right subtree of 20                               │
│                                                                  │
│ Step 3: Then process LEFT (9)                                   │
│                                                                  │
│ This reverse order is crucial!                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def buildTree_postorder(inorder: list[int], postorder: list[int]) -> Optional[TreeNode]:
    """
    Build tree from inorder and postorder traversals.

    Strategy:
    - Last element of postorder is root
    - Process postorder from right to left
    - Build right subtree before left (reverse of preorder)

    Time: O(n)
    Space: O(n)
    """
    inorder_map = {val: i for i, val in enumerate(inorder)}
    postorder_idx = [len(postorder) - 1]

    def build(left: int, right: int) -> Optional[TreeNode]:
        if left > right:
            return None

        root_val = postorder[postorder_idx[0]]
        postorder_idx[0] -= 1

        root = TreeNode(root_val)
        mid = inorder_map[root_val]

        # Build right subtree first (postorder is reversed)
        root.right = build(mid + 1, right)
        root.left = build(left, mid - 1)

        return root

    return build(0, len(inorder) - 1)
```

### Complexity
- **Time**: O(n) - Process each node once
- **Space**: O(n) - Hash map and recursion stack

### Edge Cases
- Single node: Return that node
- Left-skewed tree: All nodes in inorder before postorder root
- Right-skewed tree: All nodes in inorder after postorder root

---

## Problem 5: Flatten Binary Tree to Linked List (LC #114) - Medium

- [LeetCode](https://leetcode.com/problems/flatten-binary-tree-to-linked-list/)

### Problem Statement
Given the root of a binary tree, flatten the tree into a "linked list". The "linked list" should use the same TreeNode class where the `right` child pointer points to the next node and the `left` child is always `null`. The list should be in preorder traversal order.

### Video Explanation
- [NeetCode - Flatten Binary Tree to Linked List](https://www.youtube.com/watch?v=rKnD7rLT0lI)

### Examples
```
Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]

        1                1
       / \                \
      2   5      →         2
     / \   \                \
    3   4   6                3
                              \
                               4
                                \
                                 5
                                  \
                                   6
```

### Intuition Development
```
Approach 1: Reverse Preorder (right → left → root)
┌─────────────────────────────────────────────────────────────────┐
│ Normal preorder: ROOT → LEFT → RIGHT                            │
│ Reverse preorder: RIGHT → LEFT → ROOT                           │
│                                                                  │
│ Process nodes in reverse order, linking each to previous:        │
│                                                                  │
│   Process order: 6 → 5 → 4 → 3 → 2 → 1                          │
│                                                                  │
│   6.right = null (prev=null)                                    │
│   5.right = 6    (prev=6)                                       │
│   4.right = 5    (prev=5)                                       │
│   3.right = 4    (prev=4)                                       │
│   2.right = 3    (prev=3)                                       │
│   1.right = 2    (prev=2)                                       │
│                                                                  │
│   Result: 1 → 2 → 3 → 4 → 5 → 6                                 │
└─────────────────────────────────────────────────────────────────┘

Approach 2: Morris-like O(1) space
┌─────────────────────────────────────────────────────────────────┐
│ For each node with left child:                                  │
│   1. Find rightmost node in left subtree                        │
│   2. Connect rightmost.right to current.right                   │
│   3. Move left subtree to right                                 │
│   4. Set left to null                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def flatten(root: Optional[TreeNode]) -> None:
    """
    Flatten tree to linked list in-place.

    Strategy:
    - Process right-to-left reverse preorder
    - Each node's right points to previously processed node

    Time: O(n)
    Space: O(h)
    """
    prev = [None]

    def flatten_dfs(node):
        if not node:
            return

        # Process right, then left, then current (reverse preorder)
        flatten_dfs(node.right)
        flatten_dfs(node.left)

        # Link current to previous
        node.right = prev[0]
        node.left = None
        prev[0] = node

    flatten_dfs(root)


def flatten_iterative(root: Optional[TreeNode]) -> None:
    """
    Alternative: Morris-like traversal.

    Time: O(n)
    Space: O(1)
    """
    current = root

    while current:
        if current.left:
            # Find rightmost node in left subtree
            rightmost = current.left
            while rightmost.right:
                rightmost = rightmost.right

            # Connect rightmost to current's right
            rightmost.right = current.right

            # Move left subtree to right
            current.right = current.left
            current.left = None

        current = current.right
```

### Complexity
- **Time**: O(n) - Visit each node once
- **Space**: O(h) for recursive, O(1) for Morris-like

### Edge Cases
- Empty tree: Nothing to do
- Single node: Already flattened
- Right-skewed tree: Already flattened
- Left-skewed tree: All nodes move to right chain

---

## Problem 6: Path Sum II (LC #113) - Medium

- [LeetCode](https://leetcode.com/problems/path-sum-ii/)

### Problem Statement
Given the root of a binary tree and an integer `targetSum`, return all **root-to-leaf** paths where the sum of the node values in the path equals `targetSum`. Each path should be returned as a list of node values.

### Video Explanation
- [NeetCode - Path Sum II](https://www.youtube.com/watch?v=7_-s0fqGlPk)

### Examples
```
Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
Output: [[5,4,11,2],[5,8,4,5]]
Explanation:
         5
        / \
       4   8
      /   / \
     11  13  4
    / \     / \
   7   2   5   1

   Path 5→4→11→2 = 22 ✓
   Path 5→8→4→5 = 22 ✓

Input: root = [1,2,3], targetSum = 5
Output: []

Input: root = [1,2], targetSum = 1
Output: []
Explanation: 1 is not a leaf node
```

### Intuition Development
```
DFS with path tracking and backtracking!

┌─────────────────────────────────────────────────────────────────┐
│ Strategy:                                                        │
│   1. Traverse using DFS, tracking current path                  │
│   2. At each node, add to path and subtract from remaining      │
│   3. At leaf: if remaining == 0, save path copy                 │
│   4. BACKTRACK: remove node from path after exploring           │
│                                                                  │
│ Example: target = 22                                             │
│                                                                  │
│   Visit 5: path=[5], remaining=17                               │
│   Visit 4: path=[5,4], remaining=13                             │
│   Visit 11: path=[5,4,11], remaining=2                          │
│   Visit 7: path=[5,4,11,7], remaining=-5 (not 0, skip)          │
│   Backtrack: path=[5,4,11]                                      │
│   Visit 2: path=[5,4,11,2], remaining=0 ★ FOUND!                │
│   Save [5,4,11,2]                                               │
│   Backtrack: path=[5,4,11]                                      │
│   Backtrack: path=[5,4]                                         │
│   ... continue ...                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def pathSum(root: Optional[TreeNode], targetSum: int) -> list[list[int]]:
    """
    Find all root-to-leaf paths with target sum.

    Strategy:
    - DFS with path tracking
    - When reaching leaf, check if sum matches

    Time: O(n²) - n nodes, O(n) to copy path
    Space: O(n)
    """
    result = []

    def dfs(node: TreeNode, remaining: int, path: list[int]):
        if not node:
            return

        path.append(node.val)
        remaining -= node.val

        # Check if leaf with target sum
        if not node.left and not node.right and remaining == 0:
            result.append(path[:])  # Copy path

        # Continue DFS
        dfs(node.left, remaining, path)
        dfs(node.right, remaining, path)

        # Backtrack
        path.pop()

    dfs(root, targetSum, [])
    return result
```

### Complexity
- **Time**: O(n²) - Visit n nodes, copy path of length O(n) each time
- **Space**: O(n) - Path list and recursion stack

### Edge Cases
- No valid path: Return empty list
- Single node equals target: Return [[node]]
- Negative values: Can reduce sum, still find paths
- Multiple paths with same sum: Return all

---

## Problem 7: Binary Tree Maximum Path Sum (LC #124) - Hard

- [LeetCode](https://leetcode.com/problems/binary-tree-maximum-path-sum/)

### Problem Statement
A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes has an edge. A node can only appear in the path once. The path does not need to pass through the root. Return the maximum sum of any non-empty path.

### Video Explanation
- [NeetCode - Binary Tree Maximum Path Sum](https://www.youtube.com/watch?v=Hr5cWUld4vU)

### Examples
```
Input: root = [1,2,3]
Output: 6
Explanation:
     1
    / \
   2   3
Path 2 → 1 → 3 has sum 6

Input: root = [-10,9,20,null,null,15,7]
Output: 42
Explanation:
       -10
       /  \
      9   20
         /  \
        15   7
Path 15 → 20 → 7 has sum 42

Input: root = [-3]
Output: -3
```

### Intuition Development
```
Key insight: Path can "turn" at most once (like an inverted V)!

At each node, we consider:
┌─────────────────────────────────────────────────────────────────┐
│ 1. Path THROUGH this node (can include both subtrees)          │
│    left_max + node.val + right_max                              │
│    This is a candidate for global maximum                       │
│                                                                  │
│ 2. Path extending TO parent (can only include one subtree)      │
│    node.val + max(left_max, right_max)                          │
│    This is what we return to parent                             │
│                                                                  │
│ Example:                                                         │
│        -10                                                       │
│       /   \                                                      │
│      9    20                                                     │
│          /  \                                                    │
│         15   7                                                   │
│                                                                  │
│ At node 20:                                                      │
│   left_max = 15, right_max = 7                                  │
│   Path through 20: 15 + 20 + 7 = 42 ★ (update global max)       │
│   Return to -10: 20 + max(15, 7) = 35                           │
│                                                                  │
│ At node -10:                                                     │
│   left_max = 9, right_max = 35                                  │
│   Path through -10: 9 + (-10) + 35 = 34 (< 42)                  │
│                                                                  │
│ Key: Use 0 if subtree sum is negative (don't include it)        │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def maxPathSum(root: Optional[TreeNode]) -> int:
    """
    Find maximum path sum.

    Strategy:
    - For each node, calculate max path through it
    - Path can go left-node-right
    - Return max single-direction path for parent

    Time: O(n)
    Space: O(h)
    """
    max_sum = [float('-inf')]

    def dfs(node: TreeNode) -> int:
        if not node:
            return 0

        # Get max path sum from children (0 if negative)
        left_max = max(0, dfs(node.left))
        right_max = max(0, dfs(node.right))

        # Update global max (path through current node)
        max_sum[0] = max(max_sum[0], left_max + node.val + right_max)

        # Return max single-direction path
        return node.val + max(left_max, right_max)

    dfs(root)
    return max_sum[0]
```

### Complexity
- **Time**: O(n) - Visit each node once
- **Space**: O(h) - Recursion stack depth

### Edge Cases
- All negative values: Must still pick a path (can't be empty)
- Single node: Return that node's value
- Linear tree: Max is either whole tree or largest contiguous segment

---

## Problem 8: Kth Smallest Element in BST (LC #230) - Medium

- [LeetCode](https://leetcode.com/problems/kth-smallest-element-in-a-bst/)

### Problem Statement
Given the root of a binary search tree, and an integer `k`, return the `kth` smallest value (1-indexed) of all the values of the nodes in the tree.

### Video Explanation
- [NeetCode - Kth Smallest Element in BST](https://www.youtube.com/watch?v=5LUXSvjmGCw)

### Examples
```
Input: root = [3,1,4,null,2], k = 1
Output: 1
Explanation:
       3
      / \
     1   4
      \
       2
Inorder: [1, 2, 3, 4], 1st smallest = 1

Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3
Explanation: Inorder: [1, 2, 3, 4, 5, 6], 3rd smallest = 3
```

### Intuition Development
```
Key insight: INORDER traversal of BST gives SORTED order!

BST Property: left < root < right
Inorder: LEFT → ROOT → RIGHT
Result: Elements in ascending order!

        5
       / \
      3   6
     / \
    2   4
   /
  1

Inorder traversal: 1 → 2 → 3 → 4 → 5 → 6

┌─────────────────────────────────────────────────────────────────┐
│ Strategy: Count nodes during inorder traversal                  │
│                                                                  │
│ For k = 3:                                                       │
│   Visit 1: count = 1                                            │
│   Visit 2: count = 2                                            │
│   Visit 3: count = 3 = k → FOUND! Return 3                      │
│                                                                  │
│ Early termination: Stop as soon as we find kth element          │
│ Average time: O(h + k), not O(n)                                │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def kthSmallest(root: Optional[TreeNode], k: int) -> int:
    """
    Find kth smallest using inorder traversal.

    Strategy:
    - Inorder traversal of BST gives sorted order
    - Count nodes until we reach kth

    Time: O(k) average, O(n) worst
    Space: O(h)
    """
    count = [0]
    result = [0]

    def inorder(node):
        if not node or count[0] >= k:
            return

        inorder(node.left)

        count[0] += 1
        if count[0] == k:
            result[0] = node.val
            return

        inorder(node.right)

    inorder(root)
    return result[0]


def kthSmallest_iterative(root: Optional[TreeNode], k: int) -> int:
    """
    Alternative: Iterative inorder.
    """
    stack = []
    current = root

    while stack or current:
        while current:
            stack.append(current)
            current = current.left

        current = stack.pop()
        k -= 1

        if k == 0:
            return current.val

        current = current.right

    return -1
```

### Complexity
- **Time**: O(h + k) average, O(n) worst case (skewed tree)
- **Space**: O(h) - Recursion stack or explicit stack

### Edge Cases
- k = 1: Return minimum (leftmost node)
- k = n: Return maximum (rightmost node)
- k > n: Not possible per constraints

---

## Problem 9: Serialize and Deserialize Binary Tree (LC #297) - Hard

- [LeetCode](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/)

### Problem Statement
Design an algorithm to serialize and deserialize a binary tree. Serialization converts a tree to a string. Deserialization reconstructs the tree from the string. Your serialized format can be anything as long as you can reconstruct the tree.

### Video Explanation
- [NeetCode - Serialize and Deserialize Binary Tree](https://www.youtube.com/watch?v=u4JAi2JJhI8)

### Examples
```
Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]

        1                 Serialize: "1,2,N,N,3,4,N,N,5,N,N"
       / \                           (preorder with null markers)
      2   3
         / \
        4   5

Input: root = []
Output: []
```

### Intuition Development
```
Why not just store values? Need to know tree STRUCTURE too!

PREORDER with NULL markers preserves structure completely!

┌─────────────────────────────────────────────────────────────────┐
│ Serialize (Preorder: ROOT → LEFT → RIGHT):                     │
│                                                                  │
│        1                                                         │
│       / \                                                        │
│      2   3                                                       │
│         / \                                                      │
│        4   5                                                     │
│                                                                  │
│   Visit 1 → "1"                                                  │
│   Visit 2 → "1,2"                                               │
│   Visit 2.left (null) → "1,2,N"                                 │
│   Visit 2.right (null) → "1,2,N,N"                              │
│   Visit 3 → "1,2,N,N,3"                                         │
│   Visit 4 → "1,2,N,N,3,4"                                       │
│   Visit 4.left (null) → "1,2,N,N,3,4,N"                         │
│   Visit 4.right (null) → "1,2,N,N,3,4,N,N"                      │
│   Visit 5 → "1,2,N,N,3,4,N,N,5"                                 │
│   Visit 5.left (null) → "1,2,N,N,3,4,N,N,5,N"                   │
│   Visit 5.right (null) → "1,2,N,N,3,4,N,N,5,N,N"                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Deserialize (read preorder, build recursively):                 │
│                                                                  │
│   Use iterator over values                                      │
│   For each value:                                               │
│     - If "N", return null                                       │
│     - Else create node, build left, build right                 │
│                                                                  │
│   The preorder with null markers uniquely defines the tree!     │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
class Codec:
    """
    Serialize/deserialize using preorder traversal.

    Time: O(n)
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
```

### Complexity
- **Time**: O(n) - Visit each node for both serialize and deserialize
- **Space**: O(n) - String storage and recursion stack

### Edge Cases
- Empty tree: Serialize as "N" or empty string
- Single node: "val,N,N"
- Negative values: Handle negative signs in parsing

---

## Problem 10: Count Good Nodes (LC #1448) - Medium

- [LeetCode](https://leetcode.com/problems/count-good-nodes-in-binary-tree/)

### Problem Statement
Given a binary tree root, a node X is called **good** if in the path from root to X there are no nodes with a value greater than X. Return the number of good nodes in the tree.

### Video Explanation
- [NeetCode - Count Good Nodes in Binary Tree](https://www.youtube.com/watch?v=7cp5imvDzl4)

### Examples
```
Input: root = [3,1,4,3,null,1,5]
Output: 4
Explanation:
         3 ← good (root is always good)
        / \
       1   4 ← good (3 < 4)
      /   / \
     3   1   5 ← good (3,4 < 5)
     ↑
     good (path max is 3, node is 3)

Node 1 (left of 3): Not good (3 > 1)
Node 1 (left of 4): Not good (4 > 1)

Input: root = [3,3,null,4,2]
Output: 3
Explanation: 3, 3, and 4 are good nodes

Input: root = [1]
Output: 1
```

### Intuition Development
```
Track maximum value seen on path from root!

A node is "good" if node.val >= max_so_far

┌─────────────────────────────────────────────────────────────────┐
│         3 (max=3)                                               │
│        / \                                                       │
│       1   4 (max=3 → 4 >= 3 ✓, update max=4)                   │
│      /   / \                                                     │
│     3   1   5                                                    │
│                                                                  │
│ Path to left 1: max=3, 1 < 3 ✗ (not good)                       │
│ Path to left 3: max=3, 3 >= 3 ✓ (good!)                         │
│ Path to right 4: max=3, 4 >= 3 ✓ (good!), update max=4          │
│ Path to 4's left 1: max=4, 1 < 4 ✗ (not good)                   │
│ Path to 4's right 5: max=4, 5 >= 4 ✓ (good!)                    │
│                                                                  │
│ Total good nodes: 3, 3, 4, 5 = 4                                │
└─────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def goodNodes(root: TreeNode) -> int:
    """
    Count good nodes using DFS with max tracking.

    A node is good if no node in path from root is greater.

    Time: O(n)
    Space: O(h)
    """
    def dfs(node: TreeNode, max_so_far: int) -> int:
        if not node:
            return 0

        # Check if current node is good
        count = 1 if node.val >= max_so_far else 0

        # Update max for children
        new_max = max(max_so_far, node.val)

        # Count good nodes in subtrees
        count += dfs(node.left, new_max)
        count += dfs(node.right, new_max)

        return count

    return dfs(root, root.val)
```

### Complexity
- **Time**: O(n) - Visit each node once
- **Space**: O(h) - Recursion stack depth

### Edge Cases
- Single node: Always good (1)
- Strictly decreasing: Only root is good
- Strictly increasing: All nodes are good
- All same values: All nodes are good

---

## Summary: Tree DFS Medium Problems

| # | Problem | Key Technique | Time |
|---|---------|---------------|------|
| 1 | Validate BST | Range checking | O(n) |
| 2 | LCA | Recursive search | O(n) |
| 3 | Build from Pre+In | Divide and conquer | O(n) |
| 4 | Build from In+Post | Reverse processing | O(n) |
| 5 | Flatten to List | Reverse preorder | O(n) |
| 6 | Path Sum II | DFS + backtracking | O(n²) |
| 7 | Max Path Sum | Post-order processing | O(n) |
| 8 | Kth Smallest BST | Inorder traversal | O(k) |
| 9 | Serialize Tree | Preorder encoding | O(n) |
| 10 | Good Nodes | DFS + max tracking | O(n) |

---

## Practice More Problems

- [ ] LC #199 - Binary Tree Right Side View
- [ ] LC #337 - House Robber III
- [ ] LC #437 - Path Sum III
- [ ] LC #543 - Diameter of Binary Tree
- [ ] LC #662 - Maximum Width of Binary Tree

