# Trees BFS - Complete Practice List

## Organized by Pattern and Difficulty

### Pattern 1: Level Order Traversal

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 102 | [Binary Tree Level Order](https://leetcode.com/problems/binary-tree-level-order-traversal/) | Medium | Basic BFS |
| 107 | [Level Order Bottom](https://leetcode.com/problems/binary-tree-level-order-traversal-ii/) | Medium | Reverse result |
| 103 | [Binary Tree Zigzag](https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/) | Medium | Alternate direction |
| 199 | [Binary Tree Right Side View](https://leetcode.com/problems/binary-tree-right-side-view/) | Medium | Last node each level |
| 637 | [Average of Levels](https://leetcode.com/problems/average-of-levels-in-binary-tree/) | Easy | Sum / count per level |
| 515 | [Find Largest Value Each Row](https://leetcode.com/problems/find-largest-value-in-each-tree-row/) | Medium | Max per level |

### Pattern 2: Tree Properties via BFS

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 111 | [Minimum Depth](https://leetcode.com/problems/minimum-depth-of-binary-tree/) | Easy | First leaf found |
| 662 | [Maximum Width](https://leetcode.com/problems/maximum-width-of-binary-tree/) | Medium | Position indexing |
| 958 | [Check Completeness](https://leetcode.com/problems/check-completeness-of-a-binary-tree/) | Medium | No gaps allowed |
| 993 | [Cousins in Binary Tree](https://leetcode.com/problems/cousins-in-binary-tree/) | Easy | Same depth, diff parent |
| 513 | [Find Bottom Left Value](https://leetcode.com/problems/find-bottom-left-tree-value/) | Medium | First node last level |

### Pattern 3: Connect Nodes

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 116 | [Populating Next Right Pointers](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/) | Medium | Perfect binary tree |
| 117 | [Populating Next Right Pointers II](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/) | Medium | Any binary tree |

### Pattern 4: Serialization

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 297 | [Serialize and Deserialize](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) | Hard | Level order encoding |
| 449 | [Serialize and Deserialize BST](https://leetcode.com/problems/serialize-and-deserialize-bst/) | Medium | Use BST property |

### Pattern 5: Advanced BFS

| # | Problem | Difficulty | Key Technique |
|---|---------|------------|---------------|
| 314 | [Binary Tree Vertical Order](https://leetcode.com/problems/binary-tree-vertical-order-traversal/) | Medium | Track column |
| 987 | [Vertical Order Traversal](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/) | Hard | Sort by (col, row, val) |
| 623 | [Add One Row to Tree](https://leetcode.com/problems/add-one-row-to-tree/) | Medium | Insert at depth |
| 1161 | [Maximum Level Sum](https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/) | Medium | Track level sums |
| 1302 | [Deepest Leaves Sum](https://leetcode.com/problems/deepest-leaves-sum/) | Medium | Sum last level |

---

## Essential Templates

### 1. Basic Level Order
```python
from collections import deque

def levelOrder(root):
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
    
    return result
```

### 2. Right Side View
```python
def rightSideView(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        
        for i in range(level_size):
            node = queue.popleft()
            
            # Last node in level
            if i == level_size - 1:
                result.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
    
    return result
```

### 3. Zigzag Traversal
```python
def zigzagLevelOrder(root):
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

### 4. Maximum Width
```python
def widthOfBinaryTree(root):
    if not root:
        return 0
    
    max_width = 0
    queue = deque([(root, 0)])  # (node, position)
    
    while queue:
        level_size = len(queue)
        _, first_pos = queue[0]
        
        for i in range(level_size):
            node, pos = queue.popleft()
            
            # Normalize position
            normalized = pos - first_pos
            
            if i == level_size - 1:
                max_width = max(max_width, normalized + 1)
            
            if node.left:
                queue.append((node.left, 2 * normalized))
            if node.right:
                queue.append((node.right, 2 * normalized + 1))
    
    return max_width
```

### 5. Connect Next Pointers
```python
def connect(root):
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
```

---

## Study Plan

### Week 1: Fundamentals
- [ ] Binary Tree Level Order (LC #102)
- [ ] Level Order Bottom (LC #107)
- [ ] Average of Levels (LC #637)
- [ ] Minimum Depth (LC #111)

### Week 2: Variations
- [ ] Binary Tree Zigzag (LC #103)
- [ ] Right Side View (LC #199)
- [ ] Find Largest Value (LC #515)
- [ ] Cousins in Binary Tree (LC #993)

### Week 3: Advanced
- [ ] Maximum Width (LC #662)
- [ ] Populating Next Right Pointers (LC #116, #117)
- [ ] Vertical Order Traversal (LC #987)
- [ ] Serialize and Deserialize (LC #297)

---

## BFS vs DFS for Trees

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHEN TO USE BFS vs DFS                                   │
│                                                                             │
│  USE BFS when:                                                              │
│  • Level-by-level processing needed                                         │
│  • Finding minimum depth (first leaf)                                       │
│  • Connecting nodes at same level                                           │
│  • Width-related problems                                                   │
│                                                                             │
│  USE DFS when:                                                              │
│  • Path from root to leaf                                                   │
│  • Tree properties (height, diameter)                                       │
│  • BST operations                                                           │
│  • Subtree problems                                                         │
│                                                                             │
│  BOTH WORK:                                                                 │
│  • Maximum depth                                                            │
│  • Serialization                                                            │
│  • Tree traversal                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Common Mistakes

1. **Forgetting to check empty tree**
   ```python
   if not root:
       return []  # or appropriate default
   ```

2. **Not tracking level size**
   ```python
   # Must capture size BEFORE processing
   level_size = len(queue)  # Capture here
   for _ in range(level_size):
       # Process level
   ```

3. **Wrong position calculation for width**
   ```python
   # Normalize positions to avoid overflow
   normalized = pos - first_pos
   ```

4. **Modifying queue while iterating**
   ```python
   # Use level_size to know when level ends
   # Don't use while queue in inner loop
   ```

---

## Complexity Reference

| Pattern | Time | Space |
|---------|------|-------|
| Level order traversal | O(n) | O(w) |
| Right side view | O(n) | O(w) |
| Maximum width | O(n) | O(w) |
| Connect next pointers | O(n) | O(w) or O(1) |
| Vertical order | O(n log n) | O(n) |

Where w = maximum width of tree (up to n/2 for complete tree)
