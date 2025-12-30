# Trees DFS - Fundamentals

## Pattern Recognition

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE TREE DFS                                     │
│                                                                             │
│  Keywords that signal this pattern:                                         │
│  ✓ "Binary tree"                                                            │
│  ✓ "Tree traversal" (inorder, preorder, postorder)                          │
│  ✓ "Path from root to leaf"                                                 │
│  ✓ "Maximum/Minimum depth"                                                  │
│  ✓ "Validate BST"                                                           │
│  ✓ "Lowest Common Ancestor"                                                 │
│  ✓ "Serialize/Deserialize tree"                                             │
│                                                                             │
│  Key insight: Tree problems are naturally recursive!                        │
│  Most tree problems = solve for left + solve for right + combine            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

Before learning this pattern, ensure you understand:
- [ ] Recursion basics
- [ ] Tree terminology (root, leaf, height, depth)
- [ ] Binary tree structure

---

## Memory Map (Pattern Connections)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TREES DFS MEMORY MAP                                     │
│                                                                             │
│                    ┌───────────┐                                            │
│         ┌─────────│ TREE DFS  │─────────┐                                   │
│         │         └───────────┘         │                                   │
│         ▼                               ▼                                   │
│  ┌─────────────┐                 ┌─────────────┐                            │
│  │ TRAVERSALS  │                 │  PROBLEMS   │                            │
│  └──────┬──────┘                 └──────┬──────┘                            │
│         │                               │                                   │
│    ┌────┴────┬────────┐          ┌──────┴──────┐                            │
│    ▼         ▼        ▼          ▼             ▼                            │
│ ┌──────┐ ┌──────┐ ┌──────┐  ┌────────┐  ┌──────────┐                       │
│ │Pre-  │ │In-   │ │Post- │  │Path    │  │Structure │                       │
│ │order │ │order │ │order │  │Problems│  │Problems  │                       │
│ └──────┘ └──────┘ └──────┘  └────────┘  └──────────┘                       │
│                                                                             │
│  Related Patterns:                                                          │
│  • BFS - For level-order, shortest path in tree                             │
│  • Recursion - DFS is recursive by nature                                   │
│  • Stack - Iterative DFS uses explicit stack                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TREE DFS DECISION TREE                                   │
│                                                                             │
│  Need to visit all nodes?                                                   │
│       │                                                                     │
│       ├── YES → Which traversal order?                                      │
│       │         Preorder: process node BEFORE children (copy tree)          │
│       │         Inorder: process node BETWEEN children (BST sorted)         │
│       │         Postorder: process node AFTER children (delete tree)        │
│       │                                                                     │
│       └── NO → What are you computing?                                      │
│                    │                                                        │
│                    ├── Height/Depth → Bottom-up (postorder-like)            │
│                    │                                                        │
│                    ├── Path sum → Top-down with accumulator                 │
│                    │                                                        │
│                    └── BST property → Inorder gives sorted order            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Concept: Tree Structure

```python
class TreeNode:
    """Standard binary tree node."""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left    # Left child
        self.right = right  # Right child
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BINARY TREE VISUALIZATION                                │
│                                                                             │
│                         1  ← root                                           │
│                        / \                                                  │
│                       2   3                                                 │
│                      / \   \                                                │
│                     4   5   6                                               │
│                    /                                                        │
│                   7  ← leaf (no children)                                   │
│                                                                             │
│  Terminology:                                                               │
│  - Root: Node with no parent (node 1)                                       │
│  - Leaf: Node with no children (nodes 5, 6, 7)                              │
│  - Height: Longest path from root to leaf (4 in this tree)                  │
│  - Depth: Distance from root (root has depth 0)                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Three DFS Traversals

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DFS TRAVERSAL ORDERS                                     │
│                                                                             │
│  Tree:        1                                                             │
│              / \                                                            │
│             2   3                                                           │
│            / \                                                              │
│           4   5                                                             │
│                                                                             │
│  PREORDER (Root, Left, Right):  1, 2, 4, 5, 3                              │
│  - Visit root FIRST, then children                                          │
│  - Use: Copy tree, serialize tree                                           │
│                                                                             │
│  INORDER (Left, Root, Right):   4, 2, 5, 1, 3                              │
│  - Visit left, then root, then right                                        │
│  - Use: BST gives sorted order!                                             │
│                                                                             │
│  POSTORDER (Left, Right, Root): 4, 5, 2, 3, 1                              │
│  - Visit children FIRST, then root                                          │
│  - Use: Delete tree, calculate size/height                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Traversal Implementations

```python
# ==================== RECURSIVE (Most Natural) ====================

def preorder(root: TreeNode) -> list[int]:
    """
    Preorder traversal: Root -> Left -> Right

    Time: O(n) - visit each node once
    Space: O(h) - recursion stack, h = height
    """
    if not root:
        return []

    # Visit root first, then left subtree, then right subtree
    return [root.val] + preorder(root.left) + preorder(root.right)


def inorder(root: TreeNode) -> list[int]:
    """
    Inorder traversal: Left -> Root -> Right

    For BST, this gives sorted order!
    """
    if not root:
        return []

    return inorder(root.left) + [root.val] + inorder(root.right)


def postorder(root: TreeNode) -> list[int]:
    """
    Postorder traversal: Left -> Right -> Root

    Useful when you need to process children before parent.
    """
    if not root:
        return []

    return postorder(root.left) + postorder(root.right) + [root.val]


# ==================== ITERATIVE (Using Stack) ====================

def preorder_iterative(root: TreeNode) -> list[int]:
    """
    Iterative preorder using explicit stack.

    Time: O(n)
    Space: O(h)
    """
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        # Push right first so left is processed first (LIFO)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result


def inorder_iterative(root: TreeNode) -> list[int]:
    """
    Iterative inorder using stack.

    Go left as far as possible, then process, then go right.
    """
    result = []
    stack = []
    current = root

    while current or stack:
        # Go left as far as possible
        while current:
            stack.append(current)
            current = current.left

        # Process current node
        current = stack.pop()
        result.append(current.val)

        # Move to right subtree
        current = current.right

    return result


def postorder_iterative(root: TreeNode) -> list[int]:
    """
    Iterative postorder using two stacks or modified preorder.

    Trick: Reverse of (Root, Right, Left) = (Left, Right, Root)
    """
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        # Push left first, then right (opposite of preorder)
        if node.left:
            stack.append(node.left)
        if node.right:
            stack.append(node.right)

    # Reverse to get postorder
    return result[::-1]
```

---

## Common DFS Patterns

### Pattern 1: Top-Down (Pass info from parent to children)

```python
def max_depth_top_down(root: TreeNode) -> int:
    """
    Calculate max depth using top-down approach.

    Pass current depth DOWN to children.
    """
    max_depth = 0

    def dfs(node: TreeNode, depth: int):
        nonlocal max_depth

        if not node:
            return

        # Update max depth at each node
        max_depth = max(max_depth, depth)

        # Pass depth + 1 to children
        dfs(node.left, depth + 1)
        dfs(node.right, depth + 1)

    dfs(root, 1)
    return max_depth
```

### Pattern 2: Bottom-Up (Collect info from children)

```python
def max_depth_bottom_up(root: TreeNode) -> int:
    """
    Calculate max depth using bottom-up approach.

    Collect results from children, combine at parent.

    This is the more common and elegant approach for tree problems.
    """
    if not root:
        return 0

    # Get depth of left and right subtrees
    left_depth = max_depth_bottom_up(root.left)
    right_depth = max_depth_bottom_up(root.right)

    # Current depth = 1 + max of children
    return 1 + max(left_depth, right_depth)
```

### Pattern 3: Path Problems

```python
def has_path_sum(root: TreeNode, target_sum: int) -> bool:
    """
    Check if tree has root-to-leaf path with given sum.

    Strategy: Subtract current value, check if leaf has remaining = 0
    """
    if not root:
        return False

    # Subtract current node's value
    remaining = target_sum - root.val

    # If leaf node, check if remaining is 0
    if not root.left and not root.right:
        return remaining == 0

    # Check left or right subtree
    return (has_path_sum(root.left, remaining) or
            has_path_sum(root.right, remaining))


def all_paths(root: TreeNode, target_sum: int) -> list[list[int]]:
    """
    Find ALL root-to-leaf paths with given sum.

    Strategy: Backtracking - add node to path, recurse, remove node
    """
    result = []

    def dfs(node: TreeNode, remaining: int, path: list[int]):
        if not node:
            return

        # Add current node to path
        path.append(node.val)

        # Check if leaf with correct sum
        if not node.left and not node.right and remaining == node.val:
            result.append(path[:])  # Add copy of path

        # Recurse on children
        dfs(node.left, remaining - node.val, path)
        dfs(node.right, remaining - node.val, path)

        # Backtrack: remove current node from path
        path.pop()

    dfs(root, target_sum, [])
    return result
```

---

## Binary Search Tree (BST) Properties

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BST PROPERTY                                             │
│                                                                             │
│  For every node:                                                            │
│  - All values in LEFT subtree < node's value                                │
│  - All values in RIGHT subtree > node's value                               │
│                                                                             │
│  Example BST:                                                               │
│                         8                                                   │
│                        / \                                                  │
│                       3   10                                                │
│                      / \    \                                               │
│                     1   6   14                                              │
│                        / \  /                                               │
│                       4  7 13                                               │
│                                                                             │
│  Inorder traversal gives: 1, 3, 4, 6, 7, 8, 10, 13, 14 (SORTED!)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
def search_bst(root: TreeNode, val: int) -> TreeNode:
    """
    Search for value in BST.

    Time: O(h) where h = height
    Space: O(h) recursive, O(1) iterative
    """
    if not root or root.val == val:
        return root

    if val < root.val:
        return search_bst(root.left, val)
    else:
        return search_bst(root.right, val)


def is_valid_bst(root: TreeNode) -> bool:
    """
    Validate if tree is a valid BST.

    Strategy: Pass valid range to each node.
    """
    def validate(node: TreeNode, min_val: float, max_val: float) -> bool:
        if not node:
            return True

        # Check current node is within valid range
        if node.val <= min_val or node.val >= max_val:
            return False

        # Left subtree must be less than current
        # Right subtree must be greater than current
        return (validate(node.left, min_val, node.val) and
                validate(node.right, node.val, max_val))

    return validate(root, float('-inf'), float('inf'))
```

---

## Common Mistakes

```python
# ❌ WRONG: Not handling None case
def max_depth_wrong(root):
    return 1 + max(max_depth_wrong(root.left), max_depth_wrong(root.right))
    # Crashes when root is None!

# ✅ CORRECT: Always check for None
def max_depth_correct(root):
    if not root:
        return 0
    return 1 + max(max_depth_correct(root.left), max_depth_correct(root.right))


# ❌ WRONG: BST validation only checking immediate children
def is_valid_bst_wrong(root):
    if not root:
        return True
    if root.left and root.left.val >= root.val:
        return False
    if root.right and root.right.val <= root.val:
        return False
    return is_valid_bst_wrong(root.left) and is_valid_bst_wrong(root.right)
    # This misses cases where a node violates BST property with ancestor!

# ✅ CORRECT: Pass valid range
def is_valid_bst_correct(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return (is_valid_bst_correct(root.left, min_val, root.val) and
            is_valid_bst_correct(root.right, root.val, max_val))
```

---

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| Traversal | O(n) | O(h) |
| Search (BST) | O(h) | O(h) or O(1) |
| Insert (BST) | O(h) | O(h) or O(1) |
| Delete (BST) | O(h) | O(h) or O(1) |

Where h = height:
- Balanced tree: h = O(log n)
- Skewed tree: h = O(n)

---

## Interview Tips

### 1. How to Explain Your Approach
```
"I'll solve this recursively. The base case is when node is None.
For each node, I recursively solve for left and right subtrees,
then combine the results. This naturally follows the tree structure."
```

### 2. What Interviewers Look For
- **Recursive thinking**: Break problem into subproblems
- **Base case handling**: Empty tree, single node
- **Return value design**: What does each recursive call return?

### 3. Common Follow-up Questions
- "Can you do it iteratively?" → Use explicit stack
- "What's the space complexity?" → O(h) for recursion stack
- "What if tree is very deep?" → Consider iterative to avoid stack overflow

---

## Next: Practice Problems

Continue to:
- [02-easy-problems.md](./02-easy-problems.md) - Build foundation
- [03-medium-problems.md](./03-medium-problems.md) - Core techniques
- [04-hard-problems.md](./04-hard-problems.md) - Advanced challenges
