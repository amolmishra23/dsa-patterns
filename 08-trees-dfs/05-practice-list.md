# Trees DFS - Complete Practice List

## Organized by Pattern and Difficulty

### Pattern 1: Tree Traversal

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 94 | [Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/) | Easy | Iterative with stack |
| 144 | [Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/) | Easy | Root-left-right |
| 145 | [Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/) | Easy | Left-right-root |
| 589 | [N-ary Tree Preorder](https://leetcode.com/problems/n-ary-tree-preorder-traversal/) | Easy | Generalized preorder |
| 590 | [N-ary Tree Postorder](https://leetcode.com/problems/n-ary-tree-postorder-traversal/) | Easy | Generalized postorder |

### Pattern 2: Tree Properties

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 104 | [Maximum Depth](https://leetcode.com/problems/maximum-depth-of-binary-tree/) | Easy | Return 1 + max(left, right) |
| 111 | [Minimum Depth](https://leetcode.com/problems/minimum-depth-of-binary-tree/) | Easy | Handle leaf nodes |
| 110 | [Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/) | Easy | Check height diff |
| 100 | [Same Tree](https://leetcode.com/problems/same-tree/) | Easy | Compare recursively |
| 101 | [Symmetric Tree](https://leetcode.com/problems/symmetric-tree/) | Easy | Mirror comparison |
| 226 | [Invert Binary Tree](https://leetcode.com/problems/invert-binary-tree/) | Easy | Swap children |
| 543 | [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/) | Easy | Track max path |

### Pattern 3: Path Problems

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 112 | [Path Sum](https://leetcode.com/problems/path-sum/) | Easy | Check at leaf |
| 113 | [Path Sum II](https://leetcode.com/problems/path-sum-ii/) | Medium | Backtrack paths |
| 437 | [Path Sum III](https://leetcode.com/problems/path-sum-iii/) | Medium | Prefix sum + DFS |
| 124 | [Binary Tree Maximum Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/) | Hard | Track global max |
| 129 | [Sum Root to Leaf Numbers](https://leetcode.com/problems/sum-root-to-leaf-numbers/) | Medium | Accumulate value |
| 257 | [Binary Tree Paths](https://leetcode.com/problems/binary-tree-paths/) | Easy | Build path strings |

### Pattern 4: BST Operations

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 98 | [Validate BST](https://leetcode.com/problems/validate-binary-search-tree/) | Medium | Pass min/max bounds |
| 700 | [Search in BST](https://leetcode.com/problems/search-in-a-binary-search-tree/) | Easy | Binary search property |
| 701 | [Insert into BST](https://leetcode.com/problems/insert-into-a-binary-search-tree/) | Medium | Find correct position |
| 450 | [Delete Node in BST](https://leetcode.com/problems/delete-node-in-a-bst/) | Medium | Handle 3 cases |
| 230 | [Kth Smallest in BST](https://leetcode.com/problems/kth-smallest-element-in-a-bst/) | Medium | Inorder traversal |
| 235 | [LCA of BST](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) | Medium | Use BST property |
| 236 | [LCA of Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) | Medium | Return when found |

### Pattern 5: Tree Construction

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 105 | [Build from Preorder & Inorder](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) | Medium | Preorder gives root |
| 106 | [Build from Inorder & Postorder](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/) | Medium | Postorder gives root |
| 108 | [Sorted Array to BST](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/) | Easy | Mid as root |
| 1008 | [BST from Preorder](https://leetcode.com/problems/construct-binary-search-tree-from-preorder-traversal/) | Medium | Use bounds |

### Pattern 6: Advanced DFS

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 297 | [Serialize and Deserialize](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) | Hard | Preorder encoding |
| 337 | [House Robber III](https://leetcode.com/problems/house-robber-iii/) | Medium | Return (rob, skip) |
| 968 | [Binary Tree Cameras](https://leetcode.com/problems/binary-tree-cameras/) | Hard | 3-state greedy |
| 979 | [Distribute Coins](https://leetcode.com/problems/distribute-coins-in-binary-tree/) | Medium | Track excess |
| 1372 | [Longest ZigZag Path](https://leetcode.com/problems/longest-zigzag-path-in-a-binary-tree/) | Medium | Track direction |

---

## Visual Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TREE DFS PATTERNS                                   │
│                                                                             │
│  TRAVERSAL ORDERS:                                                          │
│                                                                             │
│           1                                                                 │
│          / \                                                                │
│         2   3                                                               │
│        / \                                                                  │
│       4   5                                                                 │
│                                                                             │
│  Preorder  (Root-L-R): 1 → 2 → 4 → 5 → 3    Process BEFORE recursing        │
│  Inorder   (L-Root-R): 4 → 2 → 5 → 1 → 3    Process BETWEEN children        │
│  Postorder (L-R-Root): 4 → 5 → 2 → 3 → 1    Process AFTER recursing         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  RECURSION CALL STACK (Max Depth):                                          │
│                                                                             │
│           1           Call Stack:                                           │
│          / \          ┌─────────┐                                           │
│         2   3         │ dfs(1)  │ ← returns 1 + max(2, 1) = 3               │
│        /              ├─────────┤                                           │
│       4               │ dfs(2)  │ ← returns 1 + max(1, 0) = 2               │
│                       ├─────────┤                                           │
│                       │ dfs(4)  │ ← returns 1 + max(0, 0) = 1               │
│                       ├─────────┤                                           │
│                       │ dfs(None)│ ← returns 0 (base case)                  │
│                       └─────────┘                                           │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PATH SUM (Target = 22):                                                    │
│                                                                             │
│           5              Path: 5 → 4 → 11 → 2 = 22 ✓                        │
│          / \                                                                │
│         4   8            dfs(5, 22)                                         │
│        /   / \             └─ dfs(4, 17)                                    │
│       11  13  4                └─ dfs(11, 13)                               │
│      /  \      \                   └─ dfs(2, 2) ← Leaf! 2 == 2 ✓            │
│     7    2      1                                                           │
│                                                                             │
│  At each node: remaining = target - node.val                                │
│  At leaf: check if remaining == node.val                                    │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VALIDATE BST (Pass Bounds):                                                │
│                                                                             │
│           5                                                                 │
│          / \                                                                │
│         3   7                                                               │
│        / \ / \                                                              │
│       1  4 6  8                                                             │
│                                                                             │
│  validate(5, -∞, +∞)     ← 5 in range ✓                                     │
│    ├─ validate(3, -∞, 5) ← 3 in range ✓                                     │
│    │    ├─ validate(1, -∞, 3) ← 1 in range ✓                                │
│    │    └─ validate(4, 3, 5)  ← 4 in range ✓                                │
│    └─ validate(7, 5, +∞) ← 7 in range ✓                                     │
│         ├─ validate(6, 5, 7)  ← 6 in range ✓                                │
│         └─ validate(8, 7, +∞) ← 8 in range ✓                                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LOWEST COMMON ANCESTOR (LCA):                                              │
│                                                                             │
│           3          Find LCA of 5 and 1                                    │
│          / \                                                                │
│         5   1        dfs(3):                                                │
│        / \   \         left = dfs(5) → returns 5 (found p)                  │
│       6   2   8        right = dfs(1) → returns 1 (found q)                 │
│                        Both non-null → return 3 (LCA!)                      │
│                                                                             │
│  Rule: If both left and right return non-null, current is LCA               │
│        Otherwise, return whichever is non-null                              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DIAMETER (Track Global Max):                                               │
│                                                                             │
│           1          max_diameter = 0                                       │
│          / \                                                                │
│         2   3        At node 2: left_h=1, right_h=1                         │
│        / \           diameter through 2 = 1+1 = 2                           │
│       4   5          max_diameter = max(0, 2) = 2                           │
│                                                                             │
│  At each node: diameter = left_height + right_height                        │
│  Return to parent: 1 + max(left_height, right_height)                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Essential Templates

### 1. Basic DFS Template
```python
def dfs(node):
    if not node:
        return base_case

    # Process current node (preorder)

    left = dfs(node.left)
    right = dfs(node.right)

    # Process current node (postorder)

    return result
```

### 2. Path Sum Template
```python
def hasPathSum(root, target):
    if not root:
        return False

    # Check at leaf
    if not root.left and not root.right:
        return root.val == target

    # Recurse with remaining sum
    remaining = target - root.val
    return hasPathSum(root.left, remaining) or \
           hasPathSum(root.right, remaining)
```

### 3. Validate BST Template
```python
def isValidBST(root):
    def validate(node, min_val, max_val):
        if not node:
            return True

        if node.val <= min_val or node.val >= max_val:
            return False

        return validate(node.left, min_val, node.val) and \
               validate(node.right, node.val, max_val)

    return validate(root, float('-inf'), float('inf'))
```

### 4. LCA Template
```python
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root

    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    if left and right:
        return root

    return left if left else right
```

### 5. Tree DP Template
```python
def tree_dp(root):
    result = [initial_value]

    def dfs(node):
        if not node:
            return base_case

        left = dfs(node.left)
        right = dfs(node.right)

        # Update global result
        result[0] = update(result[0], left, right, node)

        # Return value for parent
        return combine(left, right, node)

    dfs(root)
    return result[0]
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] Maximum Depth (LC #104)
- [ ] Same Tree (LC #100)
- [ ] Invert Binary Tree (LC #226)
- [ ] Path Sum (LC #112)
- [ ] Binary Tree Paths (LC #257)

### Week 2: BST Operations
- [ ] Validate BST (LC #98)
- [ ] Search in BST (LC #700)
- [ ] Kth Smallest in BST (LC #230)
- [ ] LCA of BST (LC #235)
- [ ] LCA of Binary Tree (LC #236)

### Week 3: Path Problems
- [ ] Path Sum II (LC #113)
- [ ] Path Sum III (LC #437)
- [ ] Diameter of Binary Tree (LC #543)
- [ ] Binary Tree Maximum Path Sum (LC #124)

### Week 4: Advanced
- [ ] Construct from Preorder & Inorder (LC #105)
- [ ] Serialize and Deserialize (LC #297)
- [ ] House Robber III (LC #337)
- [ ] Binary Tree Cameras (LC #968)

---

## Common Mistakes

1. **Not handling null nodes**
   ```python
   # Always check for None first
   if not node:
       return base_case
   ```

2. **Wrong leaf check**
   ```python
   # Leaf = no children (not just one None)
   if not node.left and not node.right:
       # This is a leaf
   ```

3. **Modifying global state incorrectly**
   ```python
   # Use list for mutable global in nested function
   result = [0]  # Not: result = 0
   ```

4. **BST bounds confusion**
   ```python
   # Use strict inequality for BST
   if node.val <= min_val or node.val >= max_val:
   ```

---

## Complexity Reference

| Pattern | Time | Space |
|---------|------|-------|
| Basic traversal | O(n) | O(h) |
| Path finding | O(n) | O(h) |
| BST operations | O(h) | O(h) |
| Tree construction | O(n) | O(n) |
| Tree DP | O(n) | O(h) |

Where h = height (log n for balanced, n for skewed)
