# Trees DFS - Easy Problems

## Problem 1: Maximum Depth of Binary Tree (LC #104) - Easy

- [LeetCode](https://leetcode.com/problems/maximum-depth-of-binary-tree/)

### Problem Statement
Given the root of a binary tree, return its maximum depth. A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

### Examples
```
Input: root = [3,9,20,null,null,15,7]
Output: 3

    3         ← Level 1
   / \
  9  20       ← Level 2
    /  \
   15   7     ← Level 3
```

### Video Explanation
- [NeetCode - Maximum Depth](https://www.youtube.com/watch?v=hTM3phVI6YQ)
- [Take U Forward - Max Depth](https://www.youtube.com/watch?v=eD3tmO66aBA)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  BOTTOM-UP RECURSION                                                        │
│                                                                             │
│  Key insight: Depth of a tree = 1 + max(left depth, right depth)           │
│                                                                             │
│       3                                                                     │
│      / \                                                                    │
│     9  20                                                                   │
│       /  \                                                                  │
│      15   7                                                                 │
│                                                                             │
│  Work from BOTTOM UP:                                                       │
│                                                                             │
│  depth(15) = 1 + max(0, 0) = 1    (leaf node)                              │
│  depth(7)  = 1 + max(0, 0) = 1    (leaf node)                              │
│  depth(9)  = 1 + max(0, 0) = 1    (leaf node)                              │
│  depth(20) = 1 + max(1, 1) = 2    (has children 15, 7)                     │
│  depth(3)  = 1 + max(1, 2) = 3    (has children 9, 20)                     │
│                                                                             │
│  This is the classic DFS pattern:                                           │
│  1. Base case: null → return 0                                             │
│  2. Recurse left, recurse right                                            │
│  3. Combine: 1 + max(left, right)                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def maxDepth(root: TreeNode) -> int:
    """
    Find maximum depth of binary tree.

    Strategy (Bottom-Up):
    - Depth of empty tree = 0
    - Depth of tree = 1 + max(depth of left, depth of right)

    Time: O(n) - visit each node once
    Space: O(h) - recursion stack, h = height
    """
    # Base case: empty tree has depth 0
    if not root:
        return 0

    # Recursively get depth of left and right subtrees
    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)

    # Current depth = 1 (for current node) + max of children
    return 1 + max(left_depth, right_depth)
```

### Complexity
- **Time**: O(n) - visit each node once
- **Space**: O(h) - recursion stack depth, where h = height

### Edge Cases
- Empty tree: Return `0`
- Single node: Return `1`
- Skewed tree (all left): Depth = n
- Balanced tree: Depth = log(n)

### Common Mistakes
- Forgetting base case (null node)
- Returning 0 for leaf nodes instead of 1
- Confusing depth (nodes) with height (edges)

### Related Problems
- LC #111 Minimum Depth of Binary Tree
- LC #110 Balanced Binary Tree
- LC #543 Diameter of Binary Tree

---

## Problem 2: Same Tree (LC #100) - Easy

- [LeetCode](https://leetcode.com/problems/same-tree/)

### Problem Statement
Given the roots of two binary trees `p` and `q`, check if they are the same (structurally identical with same node values).

### Examples
```
Input: p = [1,2,3], q = [1,2,3]
Output: true

Input: p = [1,2], q = [1,null,2]
Output: false (different structure)

Input: p = [1,2,1], q = [1,1,2]
Output: false (different values)
```

### Video Explanation
- [NeetCode - Same Tree](https://www.youtube.com/watch?v=vRbbcKXCxOw)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PARALLEL TRAVERSAL                                                         │
│                                                                             │
│  Compare two trees node by node:                                           │
│                                                                             │
│  Tree p:    1       Tree q:    1                                           │
│            / \                / \                                           │
│           2   3              2   3                                          │
│                                                                             │
│  Check at each step:                                                        │
│  1. Both null? → Same (base case)                                          │
│  2. One null, one not? → Different                                         │
│  3. Values different? → Different                                          │
│  4. Values same? → Check children recursively                              │
│                                                                             │
│  Step-by-step:                                                              │
│  compare(1, 1): values match ✓                                             │
│    compare(2, 2): values match ✓                                           │
│      compare(null, null): both null ✓                                      │
│      compare(null, null): both null ✓                                      │
│    compare(3, 3): values match ✓                                           │
│      compare(null, null): both null ✓                                      │
│      compare(null, null): both null ✓                                      │
│  Result: TRUE                                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def isSameTree(p: TreeNode, q: TreeNode) -> bool:
    """
    Check if two binary trees are identical.

    Strategy:
    - Both None → same (base case)
    - One None, one not → different
    - Both exist → check values AND recursively check children

    Time: O(min(n, m)) - compare until difference found
    Space: O(min(h1, h2)) - recursion depth
    """
    # Base case: both empty
    if not p and not q:
        return True

    # One empty, one not
    if not p or not q:
        return False

    # Both exist: check value and recursively check children
    if p.val != q.val:
        return False

    # Check both left subtrees AND both right subtrees match
    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
```

### Complexity
- **Time**: O(min(n, m)) - stops at first difference
- **Space**: O(min(h1, h2)) - recursion depth

### Edge Cases
- Both empty: Return `True`
- One empty, one not: Return `False`
- Same structure, different values: Return `False`
- Different structure, same values: Return `False`

### Common Mistakes
- Not handling the case where one is null and other isn't
- Checking only values, forgetting structure
- Using `or` instead of `and` for child comparisons

### Related Problems
- LC #101 Symmetric Tree
- LC #572 Subtree of Another Tree
- LC #617 Merge Two Binary Trees

---

## Problem 3: Invert Binary Tree (LC #226) - Easy

- [LeetCode](https://leetcode.com/problems/invert-binary-tree/)

### Problem Statement
Given the root of a binary tree, invert the tree (mirror it), and return its root.

### Examples
```
Input:     4          Output:     4
          / \                    / \
         2   7                  7   2
        / \ / \                / \ / \
       1  3 6  9              9  6 3  1
```

### Video Explanation
- [NeetCode - Invert Binary Tree](https://www.youtube.com/watch?v=OnSn2XEQ4MY)
- [Take U Forward - Invert Tree](https://www.youtube.com/watch?v=_i0jqdVkObU)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SWAP AT EVERY LEVEL                                                        │
│                                                                             │
│  Original:        4                                                         │
│                  / \                                                        │
│                 2   7                                                       │
│                / \ / \                                                      │
│               1  3 6  9                                                     │
│                                                                             │
│  Step 1: Swap children of root (4)                                         │
│                  4                                                          │
│                 / \                                                         │
│                7   2    ← swapped!                                         │
│               / \ / \                                                       │
│              6  9 1  3                                                      │
│                                                                             │
│  Step 2: Recursively invert subtree at 7                                   │
│                  4                                                          │
│                 / \                                                         │
│                7   2                                                        │
│               / \ / \                                                       │
│              9  6 1  3  ← 9,6 swapped                                      │
│                                                                             │
│  Step 3: Recursively invert subtree at 2                                   │
│                  4                                                          │
│                 / \                                                         │
│                7   2                                                        │
│               / \ / \                                                       │
│              9  6 3  1  ← 3,1 swapped                                      │
│                                                                             │
│  Pattern: swap(left, right) then recurse on both                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def invertTree(root: TreeNode) -> TreeNode:
    """
    Invert (mirror) a binary tree.

    Strategy:
    - Swap left and right children
    - Recursively invert left and right subtrees

    Time: O(n) - visit each node once
    Space: O(h) - recursion stack
    """
    # Base case: empty tree
    if not root:
        return None

    # Swap left and right children
    root.left, root.right = root.right, root.left

    # Recursively invert subtrees
    invertTree(root.left)
    invertTree(root.right)

    return root
```

### Complexity
- **Time**: O(n) - visit each node once
- **Space**: O(h) - recursion stack

### Edge Cases
- Empty tree: Return `None`
- Single node: Return same node
- Already inverted: Inverting twice gives original
- Skewed tree: Swaps to other side

### Common Mistakes
- Forgetting to return root
- Only swapping at one level
- Overcomplicating with temp variables (Python swap is elegant)

### Related Problems
- LC #101 Symmetric Tree
- LC #100 Same Tree
- LC #951 Flip Equivalent Binary Trees

---

## Problem 4: Symmetric Tree (LC #101) - Easy

- [LeetCode](https://leetcode.com/problems/symmetric-tree/)

### Problem Statement
Given the root of a binary tree, check whether it is a mirror of itself (symmetric around its center).

### Examples
```
Input:       1           Output: true
            / \
           2   2
          / \ / \
         3  4 4  3

Input:       1           Output: false
            / \
           2   2
            \   \
            3    3
```

### Video Explanation
- [NeetCode - Symmetric Tree](https://www.youtube.com/watch?v=Mao9uzxwvmc)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  MIRROR COMPARISON                                                          │
│                                                                             │
│  A tree is symmetric if left subtree mirrors right subtree                 │
│                                                                             │
│       1                                                                     │
│      / \                                                                    │
│     2   2    ← These should be mirrors                                     │
│    / \ / \                                                                  │
│   3  4 4  3                                                                 │
│                                                                             │
│  Two trees are mirrors if:                                                 │
│  1. Both null → mirrors                                                    │
│  2. One null → not mirrors                                                 │
│  3. Values equal AND                                                       │
│     - left1.left mirrors right2.right                                      │
│     - left1.right mirrors right2.left                                      │
│                                                                             │
│  Visual of mirror comparison:                                               │
│                                                                             │
│     Left subtree:  2       Right subtree:  2                               │
│                   / \                     / \                               │
│                  3   4                   4   3                              │
│                                                                             │
│  Compare: left.left (3) with right.right (3) ✓                             │
│  Compare: left.right (4) with right.left (4) ✓                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def isSymmetric(root: TreeNode) -> bool:
    """
    Check if tree is symmetric (mirror of itself).

    Strategy:
    - A tree is symmetric if left subtree is mirror of right subtree
    - Two trees are mirrors if:
      - Both empty, OR
      - Roots equal AND left1 mirrors right2 AND right1 mirrors left2

    Time: O(n)
    Space: O(h)
    """
    def is_mirror(t1: TreeNode, t2: TreeNode) -> bool:
        """Check if two trees are mirrors of each other."""
        # Both empty
        if not t1 and not t2:
            return True

        # One empty
        if not t1 or not t2:
            return False

        # Check values and mirrored children
        return (t1.val == t2.val and
                is_mirror(t1.left, t2.right) and
                is_mirror(t1.right, t2.left))

    if not root:
        return True

    return is_mirror(root.left, root.right)
```

### Complexity
- **Time**: O(n) - visit each node once
- **Space**: O(h) - recursion depth

### Edge Cases
- Empty tree: Return `True`
- Single node: Return `True`
- Two levels symmetric: `[1,2,2]` → `True`
- Values same but structure different: Return `False`

### Common Mistakes
- Comparing left.left with right.left (should be right.right)
- Forgetting to check values, only checking structure
- Not handling empty root case

### Related Problems
- LC #100 Same Tree
- LC #226 Invert Binary Tree
- LC #572 Subtree of Another Tree

---

## Problem 5: Path Sum (LC #112) - Easy

- [LeetCode](https://leetcode.com/problems/path-sum/)

### Problem Statement
Given the root of a binary tree and an integer `targetSum`, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals `targetSum`.

### Examples
```
Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
Output: true (path: 5 → 4 → 11 → 2 = 22)

        5
       / \
      4   8
     /   / \
    11  13  4
   /  \      \
  7    2      1
```

### Video Explanation
- [NeetCode - Path Sum](https://www.youtube.com/watch?v=LSKQyOz_P8I)
- [Take U Forward - Path Sum](https://www.youtube.com/watch?v=Hg82DzMemMI)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SUBTRACT AND CHECK AT LEAF                                                 │
│                                                                             │
│  Target = 22                                                                │
│                                                                             │
│        5          remaining = 22 - 5 = 17                                  │
│       / \                                                                   │
│      4   8        remaining = 17 - 4 = 13                                  │
│     /                                                                       │
│    11             remaining = 13 - 11 = 2                                  │
│   /  \                                                                      │
│  7    2           At leaf 2: remaining = 2 - 2 = 0 ✓ FOUND!               │
│                                                                             │
│  Algorithm:                                                                 │
│  1. Subtract current node's value from target                              │
│  2. If at leaf: check if remaining == 0                                    │
│  3. If not leaf: recurse on children                                       │
│                                                                             │
│  Key insight: Only check sum at LEAF nodes                                 │
│  A path must go from root to leaf, not stop in middle                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def hasPathSum(root: TreeNode, targetSum: int) -> bool:
    """
    Check if tree has root-to-leaf path with given sum.

    Strategy:
    - Subtract current value from target
    - At leaf, check if remaining target is 0
    - Recursively check left and right subtrees

    Time: O(n) - might visit all nodes
    Space: O(h) - recursion stack
    """
    # Base case: empty tree
    if not root:
        return False

    # Subtract current node's value from target
    remaining = targetSum - root.val

    # If leaf node, check if we've reached target
    if not root.left and not root.right:
        return remaining == 0

    # Recursively check left and right subtrees
    return (hasPathSum(root.left, remaining) or
            hasPathSum(root.right, remaining))
```

### Complexity
- **Time**: O(n) - may visit all nodes
- **Space**: O(h) - recursion stack

### Edge Cases
- Empty tree: Return `False`
- Single node equals target: Return `True`
- Single node not equals target: Return `False`
- Negative values: Works correctly with subtraction

### Common Mistakes
- Checking sum at non-leaf nodes
- Forgetting that path must end at leaf
- Not handling empty tree case

### Related Problems
- LC #113 Path Sum II (find all paths)
- LC #437 Path Sum III (any path)
- LC #124 Binary Tree Maximum Path Sum

---

## Problem 6: Subtree of Another Tree (LC #572) - Easy

- [LeetCode](https://leetcode.com/problems/subtree-of-another-tree/)

### Problem Statement
Given the roots of two binary trees `root` and `subRoot`, return true if there is a subtree of `root` with the same structure and node values as `subRoot`.

### Examples
```
Input:
root =     3        subRoot =  4
          / \                 / \
         4   5               1   2
        / \
       1   2
Output: true
```

### Video Explanation
- [NeetCode - Subtree of Another Tree](https://www.youtube.com/watch?v=E36O5SWp-LE)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CHECK SAME TREE AT EVERY NODE                                              │
│                                                                             │
│  root:       3           subRoot:  4                                       │
│             / \                   / \                                       │
│            4   5                 1   2                                      │
│           / \                                                               │
│          1   2                                                              │
│                                                                             │
│  At each node in root, check: is this subtree == subRoot?                  │
│                                                                             │
│  Check at node 3:                                                           │
│    3 vs 4 → values differ → not same tree                                  │
│                                                                             │
│  Check at node 4:                                                           │
│    4 vs 4 → values match                                                   │
│    1 vs 1 → values match                                                   │
│    2 vs 2 → values match                                                   │
│    All null children match → SAME TREE! ✓                                  │
│                                                                             │
│  Algorithm:                                                                 │
│  1. At each node, try isSameTree(current, subRoot)                         │
│  2. If match found, return true                                            │
│  3. Otherwise, recurse on left and right children                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def isSubtree(root: TreeNode, subRoot: TreeNode) -> bool:
    """
    Check if subRoot is a subtree of root.

    Strategy:
    - For each node in root, check if subtree starting there equals subRoot
    - Use isSameTree helper

    Time: O(n * m) where n = nodes in root, m = nodes in subRoot
    Space: O(h) - recursion depth
    """
    def is_same_tree(p: TreeNode, q: TreeNode) -> bool:
        """Check if two trees are identical."""
        if not p and not q:
            return True
        if not p or not q:
            return False
        return (p.val == q.val and
                is_same_tree(p.left, q.left) and
                is_same_tree(p.right, q.right))

    # Base case
    if not root:
        return False

    # Check if current subtree matches
    if is_same_tree(root, subRoot):
        return True

    # Recursively check left and right subtrees
    return isSubtree(root.left, subRoot) or isSubtree(root.right, subRoot)
```

### Complexity
- **Time**: O(n × m) - check at each of n nodes, each check is O(m)
- **Space**: O(h) - recursion depth

### Edge Cases
- subRoot is empty: Depends on definition (usually `True`)
- root is empty: Return `False`
- subRoot equals root: Return `True`
- subRoot is single node: Check if any node matches

### Common Mistakes
- Only checking at root, not at every node
- Confusing subtree (must include all descendants) with substructure
- Not handling null subRoot case

### Related Problems
- LC #100 Same Tree
- LC #652 Find Duplicate Subtrees
- LC #508 Most Frequent Subtree Sum

---

## Problem 7: Diameter of Binary Tree (LC #543) - Easy

- [LeetCode](https://leetcode.com/problems/diameter-of-binary-tree/)

### Problem Statement
Given the root of a binary tree, return the length of the diameter. The diameter is the length of the longest path between any two nodes (measured in edges).

### Examples
```
Input:       1
            / \
           2   3
          / \
         4   5
Output: 3 (path: 4 → 2 → 1 → 3 or 5 → 2 → 1 → 3)
```

### Video Explanation
- [NeetCode - Diameter of Binary Tree](https://www.youtube.com/watch?v=bkxqA8Rfv04)
- [Take U Forward - Diameter](https://www.youtube.com/watch?v=Rezetez59Nk)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DIAMETER = LEFT HEIGHT + RIGHT HEIGHT                                      │
│                                                                             │
│       1                                                                     │
│      / \                                                                    │
│     2   3                                                                   │
│    / \                                                                      │
│   4   5                                                                     │
│                                                                             │
│  The longest path might not go through root!                               │
│                                                                             │
│  At each node, calculate:                                                   │
│  - Left height (longest path going down left)                              │
│  - Right height (longest path going down right)                            │
│  - Diameter through this node = left_height + right_height                 │
│                                                                             │
│  At node 2:                                                                 │
│    left_height = 1 (path to 4)                                             │
│    right_height = 1 (path to 5)                                            │
│    diameter_here = 1 + 1 = 2                                               │
│                                                                             │
│  At node 1:                                                                 │
│    left_height = 2 (path 1→2→4 or 1→2→5)                                   │
│    right_height = 1 (path 1→3)                                             │
│    diameter_here = 2 + 1 = 3 ← MAXIMUM!                                    │
│                                                                             │
│  Track maximum diameter seen during traversal                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def diameterOfBinaryTree(root: TreeNode) -> int:
    """
    Find diameter of binary tree.

    Diameter = longest path between any two nodes
    Path length = number of edges

    Strategy:
    - For each node, diameter through it = left_height + right_height
    - Track maximum diameter seen
    - Return height from each node for parent's calculation

    Time: O(n)
    Space: O(h)
    """
    diameter = 0

    def height(node: TreeNode) -> int:
        """Return height of subtree, update diameter."""
        nonlocal diameter

        if not node:
            return 0

        # Get height of left and right subtrees
        left_height = height(node.left)
        right_height = height(node.right)

        # Diameter through this node = left + right heights
        diameter = max(diameter, left_height + right_height)

        # Return height of this subtree (for parent's calculation)
        return 1 + max(left_height, right_height)

    height(root)
    return diameter
```

### Complexity
- **Time**: O(n) - visit each node once
- **Space**: O(h) - recursion stack

### Edge Cases
- Empty tree: Return `0`
- Single node: Return `0` (no edges)
- Linear tree: Diameter = n - 1
- Diameter not through root: Algorithm handles correctly

### Common Mistakes
- Forgetting that diameter might not pass through root
- Confusing nodes with edges (diameter counts edges)
- Not tracking maximum across all nodes

### Related Problems
- LC #104 Maximum Depth of Binary Tree
- LC #124 Binary Tree Maximum Path Sum
- LC #687 Longest Univalue Path

---

## Problem 8: Balanced Binary Tree (LC #110) - Easy

- [LeetCode](https://leetcode.com/problems/balanced-binary-tree/)

### Problem Statement
Given a binary tree, determine if it is height-balanced. A height-balanced tree is one where the heights of the two subtrees of every node never differ by more than one.

### Examples
```
Input:       3        Output: true
            / \
           9  20
             /  \
            15   7

Input:       1        Output: false
            / \
           2   2
          / \
         3   3
        / \
       4   4
```

### Video Explanation
- [NeetCode - Balanced Binary Tree](https://www.youtube.com/watch?v=QfJsau0ItOY)
- [Take U Forward - Balanced Tree](https://www.youtube.com/watch?v=Yt50Jfbd8Po)

### Intuition
```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CHECK BALANCE WHILE COMPUTING HEIGHT                                       │
│                                                                             │
│  Naive approach: For each node, compute left and right heights             │
│  Problem: O(n²) because we recompute heights                               │
│                                                                             │
│  Better approach: Compute height once, check balance during computation    │
│  Use -1 as sentinel for "unbalanced"                                       │
│                                                                             │
│  Example (unbalanced):                                                      │
│       1                                                                     │
│      / \                                                                    │
│     2   2                                                                   │
│    / \                                                                      │
│   3   3                                                                     │
│  / \                                                                        │
│ 4   4                                                                       │
│                                                                             │
│  At node 2 (left):                                                         │
│    left_height = 2 (subtree with 3,4,4)                                    │
│    right_height = 0 (null)                                                 │
│    |2 - 0| = 2 > 1 → UNBALANCED! Return -1                                │
│                                                                             │
│  At node 1:                                                                 │
│    left_height = -1 (sentinel) → immediately return -1                     │
│                                                                             │
│  -1 propagates up, final check: height(root) != -1                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Solution
```python
def isBalanced(root: TreeNode) -> bool:
    """
    Check if tree is height-balanced.

    Strategy:
    - Calculate height while checking balance
    - Return -1 if unbalanced (sentinel value)
    - A tree is balanced if both subtrees are balanced AND height diff <= 1

    Time: O(n) - visit each node once
    Space: O(h) - recursion stack
    """
    def check_height(node: TreeNode) -> int:
        """
        Return height if balanced, -1 if unbalanced.
        """
        if not node:
            return 0

        # Check left subtree
        left_height = check_height(node.left)
        if left_height == -1:
            return -1  # Left subtree unbalanced

        # Check right subtree
        right_height = check_height(node.right)
        if right_height == -1:
            return -1  # Right subtree unbalanced

        # Check if current node is balanced
        if abs(left_height - right_height) > 1:
            return -1  # Current node unbalanced

        # Return height
        return 1 + max(left_height, right_height)

    return check_height(root) != -1
```

### Complexity
- **Time**: O(n) - single pass
- **Space**: O(h) - recursion stack

### Edge Cases
- Empty tree: Return `True`
- Single node: Return `True`
- Skewed tree: Return `False` (if height > 1)
- Off by one at leaf: Still balanced

### Common Mistakes
- Computing height separately for each node (O(n²))
- Only checking at root, not at every node
- Using wrong comparison (> 1 vs >= 1)

### Related Problems
- LC #104 Maximum Depth of Binary Tree
- LC #543 Diameter of Binary Tree
- LC #1382 Balance a Binary Search Tree

---

## Summary: Easy Problems Checklist

| # | Problem | Key Technique | Time | Space |
|---|---------|---------------|------|-------|
| 1 | Maximum Depth | Bottom-up recursion | O(n) | O(h) |
| 2 | Same Tree | Parallel recursion | O(n) | O(h) |
| 3 | Invert Tree | Swap + recurse | O(n) | O(h) |
| 4 | Symmetric Tree | Mirror comparison | O(n) | O(h) |
| 5 | Path Sum | Subtract + recurse | O(n) | O(h) |
| 6 | Subtree Check | isSameTree at each node | O(nm) | O(h) |
| 7 | Diameter | Height + track max | O(n) | O(h) |
| 8 | Balanced Tree | Height with sentinel | O(n) | O(h) |

---

## Practice More Easy Problems

- [ ] LC #94 - Binary Tree Inorder Traversal
- [ ] LC #144 - Binary Tree Preorder Traversal
- [ ] LC #145 - Binary Tree Postorder Traversal
- [ ] LC #617 - Merge Two Binary Trees
- [ ] LC #700 - Search in a BST
